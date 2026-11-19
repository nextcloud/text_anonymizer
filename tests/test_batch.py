# SPDX-FileCopyrightText: 2026 Nextcloud GmbH and Nextcloud contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for folder-level batch redaction and its guards."""

import datetime
import logging
import posixpath
import threading

import pytest
from nc_py_api import NextcloudException
from nc_py_api.files import ActionFileInfo, FsNode

import batch
import main
from batch import (
    DEFAULT_MAX_FILE_BYTES,
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

    def add_dir(self, user_path: str) -> FsNode:
        dir_path = user_path.rstrip("/") + "/"
        node = FsNode(f"files/admin/{dir_path}", fileid=self._next_id, mimetype="httpd/unix-directory")
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

    def listdir(self, path: FsNode) -> list[FsNode]:
        return [node for user_path, node in self.nodes.items() if _parent_path(user_path) == path.user_path]

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
        return self.add_file(path, data, "text/plain", last_modified=self.UPLOADED_AT)


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


def build_folder_tree(files: list[tuple[str, bytes, str]]) -> tuple[FakeNc, FsNode]:
    nc = FakeNc()
    docs = nc.files.add_dir("docs")
    nc.files.add_dir("docs/sub")
    for user_path, content, mimetype in files:
        nc.files.add_file(user_path, content, mimetype)
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
    redact_folder(nc, docs, redact=fake_redact, max_bytes=NO_LIMIT)
    assert "docs/notes (redacted 2).txt" in nc.files.uploads
    assert "docs/notes (redacted 3).txt" in nc.files.uploads


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


def test_batch_summary_notification_counts_and_reasons():
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
    assert "2 files processed, 1 redacted, 1 failed." in message
    assert "02-bad.txt (not a UTF-8 encoded text file)" in message
    assert "03-huge.txt (larger than" in message


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
    files, oversize = collect_text_files(nc, docs, NO_LIMIT)
    assert [node.user_path for node in files] == ["docs/readme.txt"]
    assert oversize == []
    report = redact_folder(nc, docs, redact=fake_redact, max_bytes=NO_LIMIT)
    assert report.processed_count == 1
    assert nc.files.downloads == ["docs/readme.txt"]
    assert list(nc.files.uploads) == ["docs/readme (redacted).txt"]


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
     "application/vnd.oasis.opendocument.text", "httpd/unix-directory"],
)
def test_is_text_mime_rejects_binary(mimetype):
    assert not is_text_mime(mimetype)


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


def test_files_action_accepts_the_folder_file_type(monkeypatch):
    """AppAPI's current files plugin reports directories as "folder"; it must reach the batch."""
    nc = FakeNc()
    docs = nc.files.add_dir("docs")
    nc.files.add_file("docs/a.txt", b"hi", "text/plain")
    monkeypatch.setattr(main, "NextcloudApp", lambda: nc)
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


def test_edited_original_gets_a_refreshed_copy(monkeypatch):
    """A copy older than its original is not fresh: the edited original is redacted again."""
    nc = FakeNc()
    docs = nc.files.add_dir("docs")
    nc.files.add_file("docs/a.txt", b"one", "text/plain")
    monkeypatch.setattr(main, "NextcloudApp", lambda: nc)
    monkeypatch.setattr(batch, "redact_text", fake_redact)
    main.redact_files("admin", [action_info(docs, "dir")])
    edited_at = FakeFiles.UPLOADED_AT + datetime.timedelta(minutes=1)
    nc.files.add_file("docs/a.txt", b"one edited", "text/plain", last_modified=edited_at)
    main.redact_files("admin", [action_info(docs, "dir")])
    # The edited original gets a new copy; the now-stale old copy is redacted like any text file.
    assert sorted(nc.files.uploads) == [
        "docs/a (redacted 2).txt",
        "docs/a (redacted 3).txt",
        "docs/a (redacted).txt",
    ]


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
