# SPDX-FileCopyrightText: 2026 Nextcloud GmbH and Nextcloud contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for folder-level batch redaction and its guards."""

import datetime
import logging
import posixpath
import re
import threading
from pathlib import Path

import pytest
from nc_py_api import NextcloudException
from nc_py_api.files import ActionFileInfo, FsNode

import batch
import main
from batch import (
    DEFAULT_MAX_FILE_BYTES,
    MAX_FILE_BYTES_ENV,
    collect_text_files,
    is_text_mime,
    max_file_bytes,
    redact_folder,
)

NO_LIMIT = 10**9


def fake_redact(text: str) -> str:
    return f"<redacted {len(text)} chars>"


def _parent_path(user_path: str) -> str:
    parent = posixpath.dirname(user_path.rstrip("/"))
    return f"{parent}/" if parent else ""


class FakeFiles:
    """In-memory ``nc.files`` double keyed by user path."""

    #: Nodes default to ADDED_AT while uploads are stamped UPLOADED_AT, so a redacted copy is
    #: newer than its original like on a real storage.
    ADDED_AT = datetime.datetime(2026, 1, 1, 12, 0, 0)
    UPLOADED_AT = datetime.datetime(2026, 1, 1, 13, 0, 0)

    def __init__(self):
        self.nodes: dict[str, FsNode] = {}
        self.contents: dict[str, bytes] = {}
        self.uploads: dict[str, bytes] = {}
        self.downloads: list[str] = []
        self._next_id = 1000

    def add_dir(self, user_path: str, permissions: str = "RGDNVCK") -> FsNode:
        dir_path = user_path.rstrip("/") + "/"
        node = FsNode(f"files/admin/{dir_path}", fileid=self._next_id, mimetype="httpd/unix-directory", permissions=permissions)
        self._next_id += 1
        self.nodes[dir_path] = node
        return node

    def add_file(self, user_path: str, content: bytes, mimetype: str, last_modified=None) -> FsNode:
        node = FsNode(
            f"files/admin/{user_path}",
            fileid=self._next_id,
            mimetype=mimetype,
            size=len(content),
            content_length=len(content),
            last_modified=last_modified or self.ADDED_AT,
        )
        self._next_id += 1
        self.nodes[user_path] = node
        self.contents[user_path] = content
        return node

    def listdir(self, path: FsNode | str) -> list[FsNode]:
        # As in nc_py_api, a folder is given as a node or as a user path; ``""`` is the root.
        parent = (path.user_path if isinstance(path, FsNode) else path).strip("/")
        folder = f"{parent}/" if parent else ""
        return [node for user_path, node in self.nodes.items() if _parent_path(user_path) == folder]

    def by_id(self, file_id) -> FsNode | None:
        for node in self.nodes.values():
            if str(node.info.fileid) == str(file_id):
                return node
        return None

    def by_path(self, user_path: str) -> FsNode:
        if user_path not in self.nodes:
            raise NextcloudException(404, "Not Found")
        return self.nodes[user_path]

    def download(self, node: FsNode) -> bytes:
        self.downloads.append(node.user_path)
        return self.contents[node.user_path]

    def upload(self, path: str, content) -> FsNode:
        data = content.encode() if isinstance(content, str) else content
        self.uploads[path] = data
        # Each upload is a minute later than the previous one, as on a real storage.
        uploaded_at = self.UPLOADED_AT + datetime.timedelta(minutes=len(self.uploads) - 1)
        return self.add_file(path, data, "text/plain", last_modified=uploaded_at)


class FakeNotifications:
    def __init__(self):
        self.created: list[tuple[str, str]] = []

    def create(self, subject: str, message: str = "", **_kwargs) -> str:
        self.created.append((subject, message))
        return str(len(self.created))


class FakeDropdownMenu:
    def __init__(self):
        self.registered: dict[str, dict] = {}

    def register_ex(self, name: str, display_name: str, callback_url: str, **kwargs) -> None:
        self.registered[name] = {"display_name": display_name, "callback_url": callback_url, **kwargs}

    def unregister(self, name: str, not_fail: bool = True) -> None:
        self.registered.pop(name, None)


class FakeTaskProcessing:
    def register(self, provider, task_type) -> None:
        pass

    def unregister(self, provider_id: str) -> None:
        pass


class FakeSettingsUI:
    def __init__(self):
        self.forms: list[object] = []

    def register_form(self, form) -> None:
        self.forms.append(form)


class FakeUI:
    def __init__(self):
        self.files_dropdown_menu = FakeDropdownMenu()
        self.settings = FakeSettingsUI()


class FakeProviders:
    def __init__(self):
        self.task_processing = FakeTaskProcessing()


class FakeNc:
    """Minimal ``NextcloudApp`` double covering the APIs used by the Files actions."""

    def __init__(self):
        self.files = FakeFiles()
        self.notifications = FakeNotifications()
        self.ui = FakeUI()
        self.providers = FakeProviders()
        self.logs: list[tuple[int, str]] = []
        self.user_id = ""

    def set_user(self, user_id: str) -> None:
        self.user_id = user_id

    def log(self, level, message: str) -> None:
        self.logs.append((level, message))


def build_folder_tree(files: list[tuple]) -> tuple[FakeNc, FsNode]:
    """Nodes from ``(path, content, mimetype[, last_modified])`` under ``docs``, which has the subfolder ``sub``."""
    nc = FakeNc()
    docs = nc.files.add_dir("docs")
    nc.files.add_dir("docs/sub")
    for user_path, content, mimetype, *last_modified in files:
        nc.files.add_file(user_path, content, mimetype, *last_modified)
    return nc, docs


def action_info(node: FsNode, file_type: str) -> ActionFileInfo:
    return ActionFileInfo(
        fileId=node.info.fileid,
        name=node.name,
        directory=posixpath.dirname(node.user_path.rstrip("/")),
        etag="etag",
        mime=node.info.mimetype,
        fileType=file_type,
        size=node.info.size,
        favorite="false",
        permissions=31,
        mtime=0,
        userId="admin",
    )


def test_folder_action_registered_for_directories_only():
    nc = FakeNc()
    assert main.enabled_handler(True, nc) == ""
    registered = nc.ui.files_dropdown_menu.registered
    assert main.REDACT_FOLDER_ACTION_NAME in registered
    folder_action = registered[main.REDACT_FOLDER_ACTION_NAME]
    assert folder_action["display_name"] == "Redact all text files"
    assert folder_action["mime"] == main.REDACT_FOLDER_ACTION_MIME
    # AppAPI enables an action when the node MIME contains the configured string.
    assert main.REDACT_FOLDER_ACTION_MIME in "httpd/unix-directory"
    assert main.REDACT_FOLDER_ACTION_MIME not in "text/plain"
    assert main.REDACT_FOLDER_ACTION_MIME not in "application/pdf"
    # The single-file action stays on text files only.
    assert main.REDACT_FILE_ACTION_MIME in "text/plain"
    assert main.REDACT_FILE_ACTION_MIME not in "httpd/unix-directory"
    main.enabled_handler(False, nc)
    assert main.REDACT_FOLDER_ACTION_NAME not in nc.ui.files_dropdown_menu.registered
    assert main.REDACT_FILE_ACTION_NAME not in nc.ui.files_dropdown_menu.registered


def test_folder_batch_redacts_each_text_file_with_copy_naming():
    nc, docs = build_folder_tree(
        [
            ("docs/alpha.txt", b"hello", "text/plain"),
            ("docs/notes.md", b"world", "text/markdown"),
            ("docs/sub/deep.txt", b"down", "text/plain"),
            ("docs/data.json", b"[]", "application/json"),
        ]
    )
    progress: list[tuple[int, int, str]] = []
    report = redact_folder(
        nc,
        docs,
        redact=fake_redact,
        on_progress=lambda i, total, name: progress.append((i, total, name)),
        max_bytes=NO_LIMIT,
    )
    assert [result.node.name for result in report.redacted] == [
        "alpha.txt",
        "data.json",
        "notes.md",
        "deep.txt",
    ]
    assert nc.files.uploads["docs/alpha (redacted).txt"] == fake_redact("hello").encode()
    assert "docs/notes (redacted).md" in nc.files.uploads
    assert "docs/data (redacted).json" in nc.files.uploads
    # Copies land next to their original, also inside subfolders.
    assert "docs/sub/deep (redacted).txt" in nc.files.uploads
    assert len(nc.files.uploads) == 4
    # Progress is reported per file as "N of M" with the filename.
    assert progress == [
        (1, 4, "alpha.txt"),
        (2, 4, "data.json"),
        (3, 4, "notes.md"),
        (4, 4, "deep.txt"),
    ]


def test_folder_batch_numbers_copies_when_one_exists():
    nc, docs = build_folder_tree(
        [
            ("docs/notes.txt", b"new", "text/plain"),
            ("docs/notes (redacted).txt", b"old copy", "text/plain"),
        ]
    )
    report = redact_folder(nc, docs, redact=fake_redact, max_bytes=NO_LIMIT)
    assert "docs/notes (redacted 2).txt" in nc.files.uploads
    assert list(nc.files.uploads) == ["docs/notes (redacted 2).txt"]
    # The old copy is older than its original and is still left alone.
    assert [result.node.name for result in report.skipped] == ["notes (redacted).txt"]


def test_partial_failure_names_failed_file_and_keeps_other_copies():
    nc, docs = build_folder_tree(
        [
            ("docs/01-good.txt", b"one", "text/plain"),
            ("docs/02-bad.txt", b"\xff\xfe\x00 not utf-8", "text/plain"),
            ("docs/03-good.txt", b"three", "text/plain"),
        ]
    )
    report = redact_folder(nc, docs, redact=fake_redact, max_bytes=NO_LIMIT)
    assert [result.node.name for result in report.redacted] == ["01-good.txt", "03-good.txt"]
    assert [result.node.name for result in report.failed] == ["02-bad.txt"]
    assert "UTF-8" in report.failed[0].detail
    # Files before and after the failure still get their redacted copies.
    assert "docs/01-good (redacted).txt" in nc.files.uploads
    assert "docs/03-good (redacted).txt" in nc.files.uploads
    assert len(nc.files.uploads) == 2


@pytest.mark.parametrize("status_code", [403, 423, 500])
def test_a_file_whose_freshness_check_fails_is_reported_and_the_run_goes_on(status_code, monkeypatch):
    """The check whether a file is up to date is part of the per-file error handling, like download and upload."""
    nc, docs = build_folder_tree(
        [
            ("docs/01-good.txt", b"one", "text/plain"),
            ("docs/02-bad.txt", b"two", "text/plain"),
            ("docs/03-good.txt", b"three", "text/plain"),
        ]
    )
    check = batch.redacted_copy_is_fresh

    def failing_check(nc_, node, copies=None):
        if node.name == "02-bad.txt":
            raise NextcloudException(status_code, "Server error")
        return check(nc_, node, copies)

    monkeypatch.setattr(batch, "redacted_copy_is_fresh", failing_check)
    report = redact_folder(nc, docs, redact=fake_redact, max_bytes=NO_LIMIT)
    assert [result.node.name for result in report.redacted] == ["01-good.txt", "03-good.txt"]
    assert [(result.node.name, result.detail) for result in report.failed] == [
        ("02-bad.txt", f"[{status_code}] Server error")
    ]


def test_batch_summary_notification_counts_and_no_file_names():
    """The notification counts; it does not list files, as a long list would grow with every skipped file."""
    nc, docs = build_folder_tree(
        [
            ("docs/01-good.txt", b"one", "text/plain"),
            ("docs/02-bad.txt", b"\xff not utf-8", "text/plain"),
            ("docs/03-huge.txt", b"x" * 100, "text/plain"),
        ]
    )
    report = redact_folder(nc, docs, redact=fake_redact, max_bytes=50)
    main.notify_batch_summary(nc, report)
    assert len(nc.notifications.created) == 1
    subject, message = nc.notifications.created[0]
    assert subject == "Redacted text files in docs"
    assert message == "2 files processed, 1 redacted, 1 failed, 1 skipped."


def test_folder_batch_logs_each_oversize_file(caplog):
    """The notification no longer names the skipped files, so the log names each one with its reason."""
    nc, docs = build_folder_tree(
        [
            ("docs/small.txt", b"ok", "text/plain"),
            ("docs/huge.txt", b"x" * 100, "text/plain"),
        ]
    )
    with caplog.at_level(logging.WARNING, logger="batch"):
        redact_folder(nc, docs, redact=fake_redact, max_bytes=50)
    assert "Skipped docs/huge.txt: larger than 50 bytes" in caplog.text
    assert "docs/small.txt" not in caplog.text


def test_size_guard_skips_oversize_file_with_warning():
    nc, docs = build_folder_tree(
        [
            ("docs/small.txt", b"ok", "text/plain"),
            ("docs/huge.txt", b"x" * 100, "text/plain"),
        ]
    )
    report = redact_folder(nc, docs, redact=fake_redact, max_bytes=50)
    assert [result.node.name for result in report.skipped] == ["huge.txt"]
    assert "larger than" in report.skipped[0].detail
    assert [result.node.name for result in report.redacted] == ["small.txt"]
    # The oversize file is never downloaded.
    assert nc.files.downloads == ["docs/small.txt"]


def test_max_file_bytes_env_var(monkeypatch):
    monkeypatch.delenv("TEXT_ANONYMIZER_MAX_FILE_BYTES", raising=False)
    assert max_file_bytes() == DEFAULT_MAX_FILE_BYTES == 10 * 1024 * 1024
    monkeypatch.setenv("TEXT_ANONYMIZER_MAX_FILE_BYTES", "2048")
    assert max_file_bytes() == 2048
    monkeypatch.setenv("TEXT_ANONYMIZER_MAX_FILE_BYTES", "not-a-number")
    assert max_file_bytes() == DEFAULT_MAX_FILE_BYTES
    monkeypatch.setenv("TEXT_ANONYMIZER_MAX_FILE_BYTES", "-5")
    assert max_file_bytes() == DEFAULT_MAX_FILE_BYTES


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("", DEFAULT_MAX_FILE_BYTES),
        (" ", DEFAULT_MAX_FILE_BYTES),
        ("0", DEFAULT_MAX_FILE_BYTES),
        (" 2048 ", 2048),
        ("+2048", 2048),
        ("2048.5", DEFAULT_MAX_FILE_BYTES),
        ("0x1000", DEFAULT_MAX_FILE_BYTES),
    ],
)
def test_max_file_bytes_reads_the_variable_as_an_integer(monkeypatch, raw, expected):
    """Surrounding spaces and a sign are read as int() reads them; a fraction or a hexadecimal value is not a size."""
    monkeypatch.setenv("TEXT_ANONYMIZER_MAX_FILE_BYTES", raw)
    assert max_file_bytes() == expected


def test_setting_description_names_the_default_of_the_code():
    """The description the admin reads names the default the code falls back to, in bytes and in MiB."""
    info_xml = (Path(__file__).resolve().parents[1] / "appinfo" / "info.xml").read_text(encoding="utf-8")
    description = re.search(rf"<name>{MAX_FILE_BYTES_ENV}</name>.*?<description>(.*?)</description>", info_xml, re.DOTALL)
    assert description is not None, "the setting is not declared in appinfo/info.xml"
    assert f"Default: {DEFAULT_MAX_FILE_BYTES} ({DEFAULT_MAX_FILE_BYTES // (1024 * 1024)} MiB)" in description.group(1)


def test_folder_batch_ignores_binary_files():
    nc, docs = build_folder_tree(
        [
            ("docs/readme.txt", b"text", "text/plain"),
            ("docs/photo.png", b"\x89PNG", "image/png"),
            ("docs/report.pdf", b"%PDF", "application/pdf"),
            ("docs/backup.bin", b"\x00\x01", "application/octet-stream"),
            ("docs/sub/image.jpeg", b"\xff\xd8", "image/jpeg"),
        ]
    )
    files, oversize, _copies = collect_text_files(nc, docs, NO_LIMIT)
    assert [node.user_path for node in files] == ["docs/readme.txt"]
    assert oversize == []
    report = redact_folder(nc, docs, redact=fake_redact, max_bytes=NO_LIMIT)
    assert report.processed_count == 1
    assert nc.files.downloads == ["docs/readme.txt"]
    assert list(nc.files.uploads) == ["docs/readme (redacted).txt"]


def test_collect_text_files_indexes_the_redacted_copies_by_original():
    nc, docs = build_folder_tree(
        [
            ("docs/a.txt", b"one", "text/plain"),
            ("docs/a (redacted).txt", b"copy", "text/plain"),
            ("docs/a (redacted 2).txt", b"copy", "text/plain"),
            ("docs/b (redacted).md", b"copy of another file", "text/markdown"),
            ("docs/sub/a (redacted).txt", b"copy in another folder", "text/plain"),
            ("docs/report (redacted final).txt", b"not a redacted copy", "text/plain"),
        ]
    )
    _files, _oversize, copies = collect_text_files(nc, docs, NO_LIMIT)
    assert {original: [copy.user_path for copy in found] for original, found in sorted(copies.items())} == {
        "docs/a.txt": ["docs/a (redacted).txt", "docs/a (redacted 2).txt"],
        "docs/b.md": ["docs/b (redacted).md"],
        "docs/sub/a.txt": ["docs/sub/a (redacted).txt"],
    }


@pytest.mark.parametrize(
    "mimetype",
    ["text/plain", "text/csv", "text/markdown", "TEXT/PLAIN", "application/json", "application/xml",
     "application/yaml", "application/toml", "application/rss+xml", "application/geo+json"],
)
def test_is_text_mime_accepts_text_like(mimetype):
    assert is_text_mime(mimetype)


@pytest.mark.parametrize(
    "mimetype",
    ["", "image/png", "image/jpeg", "application/pdf", "application/octet-stream",
     "application/vnd.oasis.opendocument.text", "httpd/unix-directory", "image/svg+xml"],
)
def test_is_text_mime_rejects_binary(mimetype):
    assert not is_text_mime(mimetype)


def test_folder_batch_leaves_svg_images_alone():
    nc, docs = build_folder_tree(
        [
            ("docs/readme.txt", b"text", "text/plain"),
            ("docs/logo.svg", b'<svg xmlns="http://www.w3.org/2000/svg"/>', "image/svg+xml"),
        ]
    )
    report = redact_folder(nc, docs, redact=fake_redact, max_bytes=NO_LIMIT)
    assert [result.node.name for result in report.redacted] == ["readme.txt"]
    assert report.skipped == [] and report.failed == []
    assert nc.files.downloads == ["docs/readme.txt"]
    assert list(nc.files.uploads) == ["docs/readme (redacted).txt"]


def test_folder_action_dispatches_directories_to_batch(monkeypatch):
    nc = FakeNc()
    docs = nc.files.add_dir("docs")
    nc.files.add_file("docs/a.txt", b"hi", "text/plain")
    monkeypatch.setattr(main, "NextcloudApp", lambda: nc)
    monkeypatch.setattr(batch, "redact_text", fake_redact)
    main.redact_files("admin", [action_info(docs, "dir")])
    assert "docs/a (redacted).txt" in nc.files.uploads
    # Only the single summary notification is sent for a folder.
    assert [subject for subject, _ in nc.notifications.created] == ["Redacted text files in docs"]


def test_folder_action_reports_folder_without_text_files(monkeypatch):
    nc = FakeNc()
    docs = nc.files.add_dir("docs")
    nc.files.add_file("docs/photo.png", b"\x89PNG", "image/png")
    monkeypatch.setattr(main, "NextcloudApp", lambda: nc)
    main.redact_files("admin", [action_info(docs, "dir")])
    assert nc.notifications.created == [("No text files in docs", "No redactable text files were found in docs.")]


def test_folder_action_missing_folder_notifies(monkeypatch):
    nc = FakeNc()
    ghost = FsNode("files/admin/ghost/", fileid=4242, mimetype="httpd/unix-directory")
    monkeypatch.setattr(main, "NextcloudApp", lambda: nc)
    main.redact_files("admin", [action_info(ghost, "dir")])
    assert nc.notifications.created == [
        ("Could not redact text files in ghost", "The folder could not be found.")
    ]

def test_single_file_action_still_notifies_per_file(monkeypatch):
    nc = FakeNc()
    node = nc.files.add_file("notes.txt", b"call John at 555-0100", "text/plain")
    monkeypatch.setattr(main, "redact_text", fake_redact)
    main.redact_file(nc, action_info(node, "file"))
    assert nc.files.uploads["notes (redacted).txt"] == fake_redact("call John at 555-0100").encode()
    assert nc.notifications.created == [("Redacted copy of notes.txt saved as notes (redacted).txt", "")]


def test_single_file_action_skips_oversize_file(monkeypatch):
    nc = FakeNc()
    node = nc.files.add_file("big.txt", b"x" * 100, "text/plain")
    monkeypatch.setenv("TEXT_ANONYMIZER_MAX_FILE_BYTES", "50")
    main.redact_file(nc, action_info(node, "file"))
    assert nc.files.uploads == {}
    assert nc.files.downloads == []
    assert nc.notifications.created == [
        ("Could not redact big.txt", "The file is larger than 50 bytes and was skipped.")
    ]


def test_folder_action_does_not_start_where_no_copy_can_be_written(monkeypatch, caplog):
    """A folder in which no file can be created is refused before the first download: one notification, one log line."""
    nc = FakeNc()
    docs = nc.files.add_dir("docs", permissions="SGDN")
    nc.files.add_file("docs/a.txt", b"one", "text/plain")
    nc.files.add_file("docs/b.txt", b"two", "text/plain")
    monkeypatch.setattr(main, "NextcloudApp", lambda: nc)
    monkeypatch.setattr(batch, "redact_text", fake_redact)
    with caplog.at_level(logging.WARNING, logger="main"):
        main.redact_files("admin", [action_info(docs, "folder")])
    assert nc.files.downloads == []
    assert nc.files.uploads == {}
    assert nc.notifications.created == [
        ("Could not redact text files in docs", "No file can be created in this folder, so no redacted copy can be written.")
    ]
    assert "Not redacting docs/: no file can be created in this folder" in caplog.text


def test_folder_action_runs_where_a_copy_can_be_written(monkeypatch):
    nc = FakeNc()
    docs = nc.files.add_dir("docs", permissions="RGDNVCK")
    nc.files.add_file("docs/a.txt", b"one", "text/plain")
    monkeypatch.setattr(main, "NextcloudApp", lambda: nc)
    monkeypatch.setattr(batch, "redact_text", fake_redact)
    main.redact_files("admin", [action_info(docs, "folder")])
    assert list(nc.files.uploads) == ["docs/a (redacted).txt"]
    assert [subject for subject, _message in nc.notifications.created] == ["Redacted text files in docs"]


def test_files_action_accepts_the_folder_file_type(monkeypatch):
    """AppAPI's current files plugin reports directories as "folder"; it must reach the batch."""
    nc = FakeNc()
    docs = nc.files.add_dir("docs")
    nc.files.add_file("docs/a.txt", b"hi", "text/plain")
    monkeypatch.setattr(main, "NextcloudApp", lambda: nc)
    monkeypatch.setattr(batch, "redact_text", fake_redact)
    main.redact_files("admin", [action_info(docs, "folder")])
    assert "docs/a (redacted).txt" in nc.files.uploads


def test_double_dispatch_redacts_each_file_once(monkeypatch):
    """Two concurrent dispatches of one folder action produce a single copy per file."""
    nc = FakeNc()
    docs = nc.files.add_dir("docs")
    nc.files.add_file("docs/a.txt", b"one", "text/plain")
    nc.files.add_file("docs/b.txt", b"two", "text/plain")
    monkeypatch.setattr(main, "NextcloudApp", lambda: nc)
    monkeypatch.setattr(batch, "redact_text", fake_redact)
    dispatches = [
        threading.Thread(target=main.redact_files, args=("admin", [action_info(docs, "dir")]))
        for _ in range(2)
    ]
    for thread in dispatches:
        thread.start()
    for thread in dispatches:
        thread.join()
    assert sorted(nc.files.uploads) == ["docs/a (redacted).txt", "docs/b (redacted).txt"]


def test_edited_original_gets_a_new_copy_and_the_old_one_stays(monkeypatch):
    """A file edited after its first redaction gets a second copy; the first one is not touched."""
    nc = FakeNc()
    docs = nc.files.add_dir("docs")
    nc.files.add_file("docs/a.txt", b"one", "text/plain")
    monkeypatch.setattr(main, "NextcloudApp", lambda: nc)
    monkeypatch.setattr(batch, "redact_text", fake_redact)
    main.redact_files("admin", [action_info(docs, "dir")])
    edited_at = FakeFiles.UPLOADED_AT + datetime.timedelta(minutes=1)
    nc.files.add_file("docs/a.txt", b"one edited", "text/plain", last_modified=edited_at)
    main.redact_files("admin", [action_info(docs, "dir")])
    assert sorted(nc.files.uploads) == ["docs/a (redacted 2).txt", "docs/a (redacted).txt"]


def _copies_tree():
    """A file edited after its first redaction, a second redaction newer than the edit, and the stale first copy."""
    stale_at = FakeFiles.UPLOADED_AT
    edited_at = stale_at + datetime.timedelta(minutes=1)
    current_at = stale_at + datetime.timedelta(minutes=2)
    nc, docs = build_folder_tree(
        [
            ("docs/a.txt", b"edited", "text/plain", edited_at),
            ("docs/a (redacted).txt", b"stale copy", "text/plain", stale_at),
            ("docs/a (redacted 2).txt", b"current copy", "text/plain", current_at),
        ]
    )
    return nc, docs


def test_original_is_fresh_when_any_of_its_copies_is_newer():
    """The edited original is compared against every redacted copy, not only against the first one."""
    nc, docs = _copies_tree()
    report = redact_folder(nc, docs, redact=fake_redact, max_bytes=NO_LIMIT)
    assert [result.node.name for result in report.skipped] == ["a (redacted 2).txt", "a (redacted).txt", "a.txt"]
    # Nothing is read: the stale first copy is left alone like the current one.
    assert nc.files.downloads == []
    assert nc.files.uploads == {}


def _never_called(text: str) -> str:
    pytest.fail(f"redact was called with {len(text)} characters")


def test_redacted_copies_are_skipped_whatever_their_age():
    """A redacted copy is never redacted again; the stale one and the current one are told apart in the report."""
    nc, docs = _copies_tree()
    report = redact_folder(nc, docs, redact=_never_called, max_bytes=NO_LIMIT)
    skipped = {result.node.name: result.detail for result in report.skipped}
    assert skipped == {
        "a (redacted 2).txt": "a redacted copy is never redacted again",
        "a (redacted).txt": "a redacted copy is never redacted again",
        "a.txt": "a redacted copy is already up to date",
    }
    assert nc.files.downloads == []
    assert nc.files.uploads == {}


def test_a_file_named_like_a_copy_is_skipped_even_without_an_original():
    """A file called "report (redacted).txt" from the start is taken for what its name says."""
    nc, docs = build_folder_tree([("docs/report (redacted).txt", b"named like a copy", "text/plain")])
    for _run in (1, 2):
        report = redact_folder(nc, docs, redact=_never_called, max_bytes=NO_LIMIT)
        assert [(result.node.name, result.status) for result in report.results] == [("report (redacted).txt", "skipped")]
    assert nc.files.downloads == []
    assert nc.files.uploads == {}


def test_repeated_runs_after_an_edit_add_no_copy():
    """Edit, run, run again: the third and the fourth run add no copy and read no file."""
    nc = FakeNc()
    docs = nc.files.add_dir("docs")
    nc.files.add_file("docs/a.txt", b"one", "text/plain")
    for run in (1, 2):
        if run == 2:
            edited_at = FakeFiles.UPLOADED_AT + datetime.timedelta(seconds=30)
            nc.files.add_file("docs/a.txt", b"one edited", "text/plain", last_modified=edited_at)
        redact_folder(nc, docs, redact=fake_redact, max_bytes=NO_LIMIT)
    after_the_edit = sorted(nc.files.uploads)
    assert after_the_edit == ["docs/a (redacted 2).txt", "docs/a (redacted).txt"]
    for _run in (3, 4):
        nc.files.downloads.clear()
        report = redact_folder(nc, docs, redact=fake_redact, max_bytes=NO_LIMIT)
        assert nc.files.downloads == []
        assert sorted(nc.files.uploads) == after_the_edit
        assert [result.node.name for result in report.skipped] == ["a (redacted 2).txt", "a (redacted).txt", "a.txt"]


def test_single_file_action_on_a_copy_is_a_no_op_whatever_its_age(monkeypatch):
    nc, _docs = _copies_tree()
    monkeypatch.setattr(main, "redact_text", fake_redact)
    for name in ("docs/a (redacted).txt", "docs/a (redacted 2).txt"):
        main.redact_file(nc, action_info(nc.files.nodes[name], "file"))
    assert nc.files.uploads == {}
    assert nc.notifications.created == [
        ("A redacted copy of a (redacted).txt is already up to date", ""),
        ("A redacted copy of a (redacted 2).txt is already up to date", ""),
    ]


def test_single_file_action_is_a_no_op_when_a_later_copy_is_current(monkeypatch):
    nc, _docs = _copies_tree()
    monkeypatch.setattr(main, "redact_text", fake_redact)
    main.redact_file(nc, action_info(nc.files.nodes["docs/a.txt"], "file"))
    assert nc.files.uploads == {}
    assert nc.notifications.created == [("A redacted copy of a.txt is already up to date", "")]


def test_single_file_action_is_a_no_op_when_the_copy_is_fresh(monkeypatch):
    nc = FakeNc()
    node = nc.files.add_file("notes.txt", b"call John at 555-0100", "text/plain")
    monkeypatch.setattr(main, "redact_text", fake_redact)
    main.redact_file(nc, action_info(node, "file"))
    assert nc.files.downloads == ["notes.txt"]
    main.redact_file(nc, action_info(node, "file"))
    # The repeat neither downloads nor re-uploads; it only reports the up-to-date copy.
    assert nc.files.downloads == ["notes.txt"]
    assert list(nc.files.uploads) == ["notes (redacted).txt"]
    assert nc.notifications.created[-1] == ("A redacted copy of notes.txt is already up to date", "")


def test_configure_logging_lets_batch_progress_reach_the_root_handlers(caplog):
    """Batch progress logs at INFO and must reach the handlers attached to the root logger,
    which the ExApp logging setup lowers from the WARNING default."""
    root = logging.getLogger()
    handlers_before = list(root.handlers)
    level_before = root.level
    try:
        main.configure_logging()
        assert root.level <= logging.INFO
        assert len(root.handlers) > len(handlers_before)
        root.handlers[:] = handlers_before
        root.setLevel(logging.INFO)
        with caplog.at_level(logging.INFO):
            logging.getLogger("batch").info("Redacting alpha.txt (1 of 4)")
        assert "Redacting alpha.txt" in caplog.text
    finally:
        root.handlers[:] = handlers_before
        root.setLevel(level_before)
