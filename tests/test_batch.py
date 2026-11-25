# SPDX-FileCopyrightText: 2026 Nextcloud GmbH and Nextcloud contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for folder-level batch redaction and its guards."""

import asyncio
import contextlib
import dataclasses
import datetime
import functools
import logging
import logging.handlers
import posixpath
import re
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from nc_py_api import NextcloudException
from nc_py_api.ex_app import LogLvl
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
        #: Every upload in order, also the second one to the same path, which ``uploads`` would hide.
        self.upload_log: list[str] = []
        #: The folders listed so far, in order: a listing is a request, and a folder run should make one per folder.
        self.listdir_log: list[str] = []
        self.downloads: list[str] = []
        #: Folders whose listing fails, as it does for an external storage that is unavailable.
        self.unlistable: set[str] = set()
        self._next_id = 1000
        self._next_etag = 0

    def add_dir(self, user_path: str, permissions: str = "RGDNVCK") -> FsNode:
        dir_path = user_path.rstrip("/") + "/"
        node = FsNode(f"files/admin/{dir_path}", fileid=self._next_id, mimetype="httpd/unix-directory", permissions=permissions)
        self._next_id += 1
        self.nodes[dir_path] = node
        return node

    def add_file(self, user_path: str, content: bytes, mimetype: str, last_modified=None, etag=None) -> FsNode:
        """Add a file; every node gets its own etag, as a storage gives a new one to every write."""
        self._next_etag += 1
        node = FsNode(
            f"files/admin/{user_path}",
            fileid=self._next_id,
            mimetype=mimetype,
            size=len(content),
            content_length=len(content),
            last_modified=last_modified or self.ADDED_AT,
            etag=etag or f'"e{self._next_etag}"',
        )
        self._next_id += 1
        self.nodes[user_path] = node
        self.contents[user_path] = content
        return node

    def listdir(self, path: FsNode | str) -> list[FsNode]:
        # As in nc_py_api, a folder is given as a node or as a user path; ``""`` is the root.
        parent = (path.user_path if isinstance(path, FsNode) else path).strip("/")
        if parent in self.unlistable:
            raise NextcloudException(503, "Storage is temporarily unavailable")
        self.listdir_log.append(parent)
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
        self.upload_log.append(path)
        # Each upload is a minute later than the previous one, as on a real storage.
        uploaded_at = self.UPLOADED_AT + datetime.timedelta(minutes=len(self.uploads) - 1)
        return self.add_file(path, data, "text/plain", last_modified=uploaded_at)


@dataclasses.dataclass
class FakeRecord:
    """What ``get_values`` returns for a found key, with the two attributes the app reads."""
    key: str
    value: str


class FakeAppConfig:
    """In-memory ``nc.appconfig_ex`` double: the key/value store of the ExApp in Nextcloud's database."""

    def __init__(self):
        self.values: dict[str, str] = {}
        #: Every call, as ``(method, what)``; a test counts the requests with it.
        self.calls: list[tuple[str, object]] = []
        #: Keys whose reading fails, as it does when Nextcloud cannot be reached.
        self.unreadable: set[str] = set()

    def get_value(self, key: str, default=None):
        self.calls.append(("get_values", [key]))
        if key in self.unreadable:
            raise NextcloudException(503, "Server error")
        return self.values.get(key, default)

    def get_values(self, keys: list[str]):
        self.calls.append(("get_values", list(keys)))
        if any(key in self.unreadable for key in keys):
            raise NextcloudException(503, "Server error")
        return [FakeRecord(key, self.values[key]) for key in keys if key in self.values]

    def set_value(self, key: str, value: str, sensitive=None) -> None:
        self.calls.append(("set_value", key))
        self.values[key] = value


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
        self.appconfig_ex = FakeAppConfig()
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


def app_api_offers_action(node_mime: str, configured_mime: str) -> bool:
    """The rule of AppAPI's ``mimeMatches`` (src/filesplugin.js): the node MIME contains one configured value."""
    return any(part and part in node_mime for part in (value.strip() for value in configured_mime.split(",")))


def test_folder_action_registered_for_directories_only():
    nc = FakeNc()
    assert main.enabled_handler(True, nc) == ""
    registered = nc.ui.files_dropdown_menu.registered
    assert main.REDACT_FOLDER_ACTION_NAME in registered
    folder_action = registered[main.REDACT_FOLDER_ACTION_NAME]
    assert folder_action["display_name"] == "Redact all text files"
    assert folder_action["mime"] == main.REDACT_FOLDER_ACTION_MIME
    # The Files app gives every folder the MIME type "httpd/unix-directory".
    assert app_api_offers_action("httpd/unix-directory", main.REDACT_FOLDER_ACTION_MIME)
    # A file type that merely contains "dir" must not get the folder action.
    for file_mime in ("application/x-director", "inode/directory", "text/directory", "text/plain", "application/pdf"):
        assert not app_api_offers_action(file_mime, main.REDACT_FOLDER_ACTION_MIME), file_mime
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
    # Nothing is remembered about the old copy, and it is still left alone.
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


def test_files_whose_records_cannot_be_read_are_reported_and_the_run_goes_on():
    """The records are read where the freshness of each file is decided; their failure must not end the run."""
    nc, docs = build_folder_tree(
        [
            ("docs/01-good.txt", b"one", "text/plain"),
            ("docs/02-copy.txt", b"two", "text/plain"),
            ("docs/02-copy (redacted).txt", b"copy", "text/plain"),
        ]
    )
    with_copy = nc.files.nodes["docs/02-copy.txt"]
    remember(nc, with_copy)
    nc.appconfig_ex.unreadable.add(batch.RECORD_KEY_PREFIX + str(with_copy.info.fileid))
    report = redact_folder(nc, docs, redact=fake_redact, max_bytes=NO_LIMIT)
    assert [result.node.name for result in report.redacted] == ["01-good.txt"]
    # The file with a copy cannot be judged; it is neither redacted again nor taken for done.
    assert [(result.node.name, result.detail) for result in report.failed] == [("02-copy.txt", "[503] Server error")]
    assert list(nc.files.uploads) == ["docs/01-good (redacted).txt"]


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
    files, oversize, _copies, _unreadable = collect_text_files(nc, docs, NO_LIMIT)
    assert [node.user_path for node in files] == ["docs/readme.txt"]
    assert oversize == []
    report = redact_folder(nc, docs, redact=fake_redact, max_bytes=NO_LIMIT)
    assert report.processed_count == 1
    assert nc.files.downloads == ["docs/readme.txt"]
    assert list(nc.files.uploads) == ["docs/readme (redacted).txt"]


def unreadable_tree(nc: FakeNc) -> tuple[FsNode, FsNode, FsNode]:
    """``docs`` with a readable file, the subfolder ``docs/broken`` that cannot be listed, and ``docs/fine``."""
    docs = nc.files.add_dir("docs")
    nc.files.add_file("docs/a.txt", b"one", "text/plain")
    broken = nc.files.add_dir("docs/broken")
    nc.files.add_file("docs/broken/c.txt", b"three", "text/plain")
    nc.files.unlistable.add("docs/broken")
    fine = nc.files.add_dir("docs/fine")
    nc.files.add_file("docs/fine/b.txt", b"two", "text/plain")
    return docs, broken, fine


def test_folder_batch_goes_on_past_a_subfolder_that_cannot_be_listed(caplog):
    """An unavailable external storage or share in the tree must not cost the user the files around it."""
    nc = FakeNc()
    docs, broken, _fine = unreadable_tree(nc)
    with caplog.at_level(logging.WARNING, logger="batch"):
        report = redact_folder(nc, docs, redact=fake_redact, max_bytes=NO_LIMIT)
    assert [result.node.user_path for result in report.redacted] == ["docs/a.txt", "docs/fine/b.txt"]
    assert nc.files.downloads == ["docs/a.txt", "docs/fine/b.txt"]
    assert report.failed == [] and report.skipped == []
    assert report.unreadable_folders == [broken]
    # The notification counts the folder; the server log names it with the reason.
    assert "Skipped docs/broken/: the folder could not be listed ([503] Storage is temporarily unavailable)" in caplog.text


@pytest.mark.parametrize("status_code", [403, 404, 503])
def test_folder_batch_leaves_out_a_subfolder_whatever_the_error(status_code):
    nc = FakeNc()
    docs, _broken, _fine = unreadable_tree(nc)
    listdir = nc.files.listdir

    def failing_listdir(path):
        if (path.user_path if isinstance(path, FsNode) else path).strip("/") == "docs/broken":
            raise NextcloudException(status_code, "Server error")
        return listdir(path)

    nc.files.listdir = failing_listdir
    report = redact_folder(nc, docs, redact=fake_redact, max_bytes=NO_LIMIT)
    assert [result.node.user_path for result in report.redacted] == ["docs/a.txt", "docs/fine/b.txt"]
    assert len(report.unreadable_folders) == 1


def test_folder_batch_still_fails_when_the_folder_itself_cannot_be_listed():
    """The folder the action was started on is not a subfolder to skip: nothing can be said about it."""
    nc = FakeNc()
    docs, _broken, _fine = unreadable_tree(nc)
    nc.files.unlistable.add("docs")
    with pytest.raises(NextcloudException):
        redact_folder(nc, docs, redact=fake_redact, max_bytes=NO_LIMIT)
    assert nc.files.uploads == {}


@pytest.mark.parametrize(
    ("unreadable", "sentence"),
    [(1, "1 folder could not be read."), (2, "2 folders could not be read.")],
)
def test_batch_summary_counts_the_folders_that_could_not_be_read(unreadable, sentence):
    """The user must learn that part of the tree was left out, or they take the folder for done."""
    nc, docs = build_folder_tree([("docs/a.txt", b"one", "text/plain")])
    report = redact_folder(nc, docs, redact=fake_redact, max_bytes=NO_LIMIT)
    for number in range(unreadable):
        report.unreadable_folders.append(nc.files.add_dir(f"docs/gone{number}"))
    main.notify_batch_summary(nc, report)
    assert nc.notifications.created == [("Redacted text files in docs", f"1 files processed, 1 redacted, 0 failed, 0 skipped. {sentence}")]


def test_folder_action_reports_a_folder_that_left_out_a_subfolder(monkeypatch):
    nc = FakeNc()
    docs, _broken, _fine = unreadable_tree(nc)
    monkeypatch.setattr(main, "NextcloudApp", lambda: nc)
    monkeypatch.setattr(batch, "redact_text", fake_redact)
    main.redact_files("admin", [action_info(docs, "dir")])
    assert sorted(nc.files.uploads) == ["docs/a (redacted).txt", "docs/fine/b (redacted).txt"]
    assert nc.notifications.created == [
        ("Redacted text files in docs", "2 files processed, 2 redacted, 0 failed, 0 skipped. 1 folder could not be read.")
    ]


def test_folder_action_with_only_an_unreadable_subfolder_does_not_say_no_text_files(monkeypatch):
    nc = FakeNc()
    docs = nc.files.add_dir("docs")
    nc.files.add_dir("docs/broken")
    nc.files.unlistable.add("docs/broken")
    monkeypatch.setattr(main, "NextcloudApp", lambda: nc)
    monkeypatch.setattr(batch, "redact_text", fake_redact)
    main.redact_files("admin", [action_info(docs, "dir")])
    assert nc.notifications.created == [
        ("Redacted text files in docs", "0 files processed, 0 redacted, 0 failed, 0 skipped. 1 folder could not be read.")
    ]


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
    _files, _oversize, copies, _unreadable = collect_text_files(nc, docs, NO_LIMIT)
    assert {original: [copy.user_path for copy in found] for original, found in sorted(copies.items())} == {
        "docs/a.txt": ["docs/a (redacted).txt", "docs/a (redacted 2).txt"],
        "docs/b.md": ["docs/b (redacted).md"],
        "docs/sub/a.txt": ["docs/sub/a (redacted).txt"],
    }


@pytest.mark.parametrize(
    ("user_path", "expected"),
    [
        ("docs/notes.txt", "docs/notes (redacted).txt"),
        ("docs/notes", "docs/notes (redacted)"),
        ("docs/archive.tar.gz", "docs/archive.tar (redacted).gz"),
        ("docs/notes (redacted).txt", "docs/notes (redacted).txt"),
        ("docs/notes (redacted 2).txt", "docs/notes (redacted).txt"),
    ],
)
def test_canonical_redacted_copy_path_names_the_first_candidate(user_path, expected):
    assert batch.canonical_redacted_copy_path(user_path) == expected


@pytest.mark.parametrize(
    ("existing", "expected"),
    [
        ([], "docs/notes (redacted).txt"),
        (["docs/notes (redacted).txt"], "docs/notes (redacted 2).txt"),
        (["docs/notes (redacted).txt", "docs/notes (redacted 2).txt"], "docs/notes (redacted 3).txt"),
        (["docs/notes (redacted 2).txt"], "docs/notes (redacted).txt"),
        (["docs/notes (redacted).txt", "docs/notes (redacted 3).txt"], "docs/notes (redacted 2).txt"),
    ],
)
def test_redacted_copy_path_takes_the_first_free_number(existing, expected):
    """The copies of an original that was redacted again are numbered from the first free candidate."""
    nc = FakeNc()
    for path in existing:
        nc.files.add_file(path, b"copy", "text/plain")
    assert batch.redacted_copy_path(nc, "docs/notes.txt") == expected


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


def run_in_threads(*targets) -> None:
    """Run each target in a thread of its own and wait for all of them; a failure inside a thread fails here."""
    failures: list[BaseException] = []

    def guarded(target):
        try:
            target()
        except BaseException as failure:  # noqa  pylint: disable=broad-exception-caught
            failures.append(failure)

    threads = [threading.Thread(target=guarded, args=(target,), name=f"thread {number}") for number, target in enumerate(targets)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
        assert not thread.is_alive(), "a thread did not end: the actions wait for each other"
    assert not failures, failures


def test_double_dispatch_redacts_each_file_once_and_sends_one_summary(monkeypatch):
    """A double click dispatches the same folder action twice: one run redacts, the other is dropped."""
    nc = FakeNc()
    docs = nc.files.add_dir("docs")
    nc.files.add_file("docs/a.txt", b"one", "text/plain")
    nc.files.add_file("docs/b.txt", b"two", "text/plain")
    monkeypatch.setattr(main, "NextcloudApp", lambda: nc)
    started = threading.Event()
    release = threading.Event()

    def slow_redact(text):
        started.set()
        assert release.wait(timeout=10)
        return fake_redact(text)

    monkeypatch.setattr(batch, "redact_text", slow_redact)
    first = threading.Thread(target=main.redact_files, args=("admin", [action_info(docs, "dir")]), name="first")
    first.start()
    assert started.wait(timeout=10)
    # The second dispatch returns at once, while the first run still works on its first file.
    main.redact_files("admin", [action_info(docs, "dir")])
    release.set()
    first.join(timeout=10)
    assert not first.is_alive()
    assert sorted(nc.files.uploads) == ["docs/a (redacted).txt", "docs/b (redacted).txt"]
    assert [subject for subject, _message in nc.notifications.created] == ["Redacted text files in docs"]


def test_folder_run_and_file_action_on_one_file_write_one_copy(monkeypatch):
    """Both reach a file before either has uploaded: one writes the copy, the other finds it."""
    nc = FakeNc()
    docs = nc.files.add_dir("docs")
    node = nc.files.add_file("docs/a.txt", b"one", "text/plain")
    monkeypatch.setattr(main, "NextcloudApp", lambda: nc)

    def slow_redact(text):
        time.sleep(0.3)
        return fake_redact(text)

    monkeypatch.setattr(batch, "redact_text", slow_redact)
    monkeypatch.setattr(main, "redact_text", slow_redact)
    run_in_threads(
        lambda: main.redact_files("admin", [action_info(docs, "dir")]),
        lambda: main.redact_files("admin", [action_info(node, "file")]),
    )
    assert nc.files.upload_log == ["docs/a (redacted).txt"]
    subjects = [subject for subject, _message in nc.notifications.created]
    assert sum(subject.startswith("Redacted text files in docs") for subject in subjects) == 1
    # Whoever wrote the copy, the other action found it.
    assert "A redacted copy of a.txt is already up to date" in subjects or "Redacted copy of a.txt saved as a (redacted).txt" in subjects
    assert batch._FILE_LOCKS == {} and main._ACTIONS_RUNNING == set()


def test_a_folder_run_lists_each_folder_once():
    """The listing of a folder is one request whatever its size; it must not be repeated for every file."""
    nc = FakeNc()
    docs = nc.files.add_dir("docs")
    nc.files.add_dir("docs/sub")
    for name in ("docs/a.txt", "docs/b.md", "docs/sub/c.txt", "docs/sub/d.json"):
        nc.files.add_file(name, b"one", "text/plain")
    redact_folder(nc, docs, redact=fake_redact, max_bytes=NO_LIMIT)
    assert len(nc.files.uploads) == 4
    assert nc.files.listdir_log == ["docs", "docs/sub"]


def test_folder_run_sees_the_copy_a_file_action_wrote_meanwhile(monkeypatch):
    """The folder run listed its folder before the file action wrote a copy of one of its files."""
    nc = FakeNc()
    docs = nc.files.add_dir("docs")
    nc.files.add_file("docs/01-blocking.txt", b"one", "text/plain")
    later = nc.files.add_file("docs/02-later.txt", b"two", "text/plain")
    monkeypatch.setattr(main, "NextcloudApp", lambda: nc)
    blocked = threading.Event()
    release = threading.Event()

    def redact(text):
        if text == "one":
            blocked.set()
            assert release.wait(timeout=10)
        return fake_redact(text)

    monkeypatch.setattr(batch, "redact_text", redact)
    monkeypatch.setattr(main, "redact_text", redact)
    folder_run = threading.Thread(target=main.redact_files, args=("admin", [action_info(docs, "dir")]))
    folder_run.start()
    assert blocked.wait(timeout=10)
    main.redact_files("admin", [action_info(later, "file")])
    release.set()
    folder_run.join(timeout=10)
    assert not folder_run.is_alive()
    assert sorted(nc.files.upload_log) == ["docs/01-blocking (redacted).txt", "docs/02-later (redacted).txt"]
    assert [subject for subject, _message in nc.notifications.created] == [
        "Redacted copy of 02-later.txt saved as 02-later (redacted).txt",
        "Redacted text files in docs",
    ]
    assert batch._FILE_LOCKS == {} and main._ACTIONS_RUNNING == set()


def test_file_lock_holds_one_lock_per_file_and_drops_it_afterwards():
    """Waiting threads share the lock of the holder; nothing is kept once the last one has left."""
    entered = []
    inside = 0

    def use(number):
        nonlocal inside
        with batch._file_lock(42):
            assert inside == 0
            inside = 1
            entered.append(number)
            time.sleep(0.005)
            inside = 0

    run_in_threads(*[functools.partial(use, number) for number in range(8)])
    with batch._file_lock(42):
        assert list(batch._FILE_LOCKS) == [42]
    assert sorted(entered) == list(range(8))
    assert batch._FILE_LOCKS == {}


def test_file_lock_keeps_its_entry_while_a_thread_waits_for_it():
    """A thread that waits for the lock counts as a user of it: no second lock can be made for the file meanwhile."""
    holding = threading.Event()
    release = threading.Event()
    passed = []

    def holder():
        with batch._file_lock(7):
            holding.set()
            release.wait(timeout=10)

    def waiter():
        with batch._file_lock(7):
            passed.append("waiter")

    holder_thread = threading.Thread(target=holder)
    holder_thread.start()
    assert holding.wait(timeout=10)
    waiter_thread = threading.Thread(target=waiter)
    waiter_thread.start()
    deadline = time.monotonic() + 10
    while batch._FILE_LOCKS.get(7, (None, 0))[1] < 2 and time.monotonic() < deadline:
        time.sleep(0.005)
    assert batch._FILE_LOCKS[7][1] == 2  # the holder and the one who waits
    release.set()
    for thread in (holder_thread, waiter_thread):
        thread.join(timeout=10)
        assert not thread.is_alive()
    assert passed == ["waiter"]
    assert batch._FILE_LOCKS == {}


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


def remember(nc: FakeNc, node: FsNode) -> None:
    """Remember for ``node`` the etag it has now, as the redaction of a copy of it does."""
    nc.appconfig_ex.values[batch.RECORD_KEY_PREFIX + str(node.info.fileid)] = node.etag_unquoted


def _copies_tree():
    """A file with two redacted copies: the first of an older content, the second of the content it has now."""
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
    remember(nc, nc.files.nodes["docs/a.txt"])
    return nc, docs


def _never_called(text: str) -> str:
    pytest.fail(f"redact was called with {len(text)} characters")


def test_original_is_fresh_while_one_of_its_copies_is_of_its_current_content():
    """What counts is the remembered etag, not the times: the copies may be older than the original."""
    nc, docs = _copies_tree()
    report = redact_folder(nc, docs, redact=_never_called, max_bytes=NO_LIMIT)
    assert [result.node.name for result in report.skipped] == ["a (redacted 2).txt", "a (redacted).txt", "a.txt"]
    assert nc.files.downloads == []
    assert nc.files.uploads == {}


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


def test_single_file_action_is_a_no_op_while_a_copy_is_of_the_current_content(monkeypatch):
    nc, _docs = _copies_tree()
    monkeypatch.setattr(main, "redact_text", fake_redact)
    main.redact_file(nc, action_info(nc.files.nodes["docs/a.txt"], "file"))
    assert nc.files.uploads == {}
    assert nc.notifications.created == [("A redacted copy of a.txt is already up to date", "")]


def _redacted_once() -> tuple[FakeNc, FsNode, FsNode]:
    """A folder with one file and the redacted copy a first run wrote for it."""
    nc = FakeNc()
    docs = nc.files.add_dir("docs")
    original = nc.files.add_file("docs/a.txt", b"call John at 555-0100", "text/plain")
    redact_folder(nc, docs, redact=fake_redact, max_bytes=NO_LIMIT)
    assert list(nc.files.uploads) == ["docs/a (redacted).txt"]
    nc.files.downloads.clear()
    return nc, docs, original


@pytest.mark.parametrize(
    ("change", "time_of_the_change"),
    [
        ("a restored older version", datetime.timedelta(days=-30)),
        ("a synced file that kept an old local timestamp", datetime.timedelta(days=-300)),
    ],
)
def test_a_content_change_that_makes_the_file_no_newer_gets_a_new_copy(change, time_of_the_change):
    """The copy is newer than the original both times, so the modification time alone would say "up to date"."""
    nc, docs, original = _redacted_once()
    old_time = FakeFiles.ADDED_AT + time_of_the_change
    nc.files.add_file("docs/a.txt", b"call Jane at 555-0200", "text/plain", last_modified=old_time)
    assert nc.files.nodes["docs/a (redacted).txt"].info.last_modified > old_time
    report = redact_folder(nc, docs, redact=fake_redact, max_bytes=NO_LIMIT)
    assert change and [result.node.name for result in report.redacted] == ["a.txt"]
    assert sorted(nc.files.uploads) == ["docs/a (redacted 2).txt", "docs/a (redacted).txt"]


def test_a_deleted_copy_leaves_the_original_not_fresh():
    """The remembered etag alone does not make a file up to date: a copy has to exist."""
    nc, docs, _original = _redacted_once()
    del nc.files.nodes["docs/a (redacted).txt"]
    report = redact_folder(nc, docs, redact=fake_redact, max_bytes=NO_LIMIT)
    assert [result.detail for result in report.redacted] == ["docs/a (redacted).txt"]


def test_the_copy_remembers_the_etag_its_original_had_in_the_listing():
    """The record is the etag the decision was based on, so a change after the listing shows up in the next run."""
    nc, docs, original = _redacted_once()
    [recorded] = nc.appconfig_ex.values.values()
    assert list(nc.appconfig_ex.values) == [batch.RECORD_KEY_PREFIX + str(original.info.fileid)]
    assert recorded == original.etag_unquoted


def test_a_repeat_reads_the_records_of_a_folder_in_one_request_and_no_file():
    """An unchanged folder costs one request for all its records and no download."""
    nc = FakeNc()
    docs = nc.files.add_dir("docs")
    originals = [nc.files.add_file(f"docs/{number}.txt", b"one", "text/plain") for number in range(5)]
    keys = [batch.RECORD_KEY_PREFIX + str(node.info.fileid) for node in originals]
    redact_folder(nc, docs, redact=fake_redact, max_bytes=NO_LIMIT)
    # The first run finds no copy, so it reads no record; it writes one per file.
    assert nc.appconfig_ex.calls == [("set_value", key) for key in keys]
    nc.appconfig_ex.calls.clear()
    nc.files.downloads.clear()
    report = redact_folder(nc, docs, redact=_never_called, max_bytes=NO_LIMIT)
    assert nc.files.downloads == []
    assert nc.appconfig_ex.calls == [("get_values", keys)]
    details = [result.detail for result in report.skipped]
    assert details.count("a redacted copy is already up to date") == 5
    assert details.count("a redacted copy is never redacted again") == 5


def test_single_file_action_redacts_again_after_a_restored_version(monkeypatch):
    nc, _docs, _original = _redacted_once()
    nc.files.add_file("docs/a.txt", b"call Jane at 555-0200", "text/plain", last_modified=FakeFiles.ADDED_AT - datetime.timedelta(days=1))
    monkeypatch.setattr(main, "redact_text", fake_redact)
    main.redact_file(nc, action_info(nc.files.nodes["docs/a.txt"], "file"))
    assert sorted(nc.files.uploads) == ["docs/a (redacted 2).txt", "docs/a (redacted).txt"]


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


@contextlib.contextmanager
def logging_to(posted: list, stall: float = 0.0, gate: threading.Event | None = None):
    """Run ``configure_logging`` with what it would post to Nextcloud collected in ``posted`` instead.

    Each post is ``(thread id, level, message)`` and takes ``stall`` seconds. With a ``gate`` that is not set,
    every post waits for it, as it does for a Nextcloud that hangs. The loggers are restored and the sender
    stopped at the end.
    """

    class Nextcloud:
        def log(self, level, message, fast_send=False):
            posted.append((threading.get_ident(), level, message))
            time.sleep(stall)
            if gate is not None:
                gate.wait(timeout=30)

    loggers = (main.LOGGER, batch.LOGGER)
    saved = [(logger, logger.handlers[:], logger.level) for logger in loggers]
    root = logging.getLogger()
    root_handlers, root_level = root.handlers[:], root.level
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("nc_py_api.ex_app.logger.NextcloudApp", lambda: Nextcloud())
        listener = main.configure_logging()
        try:
            yield
        finally:
            main.flush_logs(listener)
            for logger, handlers, level in saved:
                logger.handlers[:] = handlers
                logger.setLevel(level)
            root.handlers[:] = root_handlers
            root.setLevel(root_level)


def test_only_the_app_loggers_reach_nextcloud():
    """A library logs one line per request or per redaction; each line sent would be one more request."""
    posted = []
    with logging_to(posted):
        logging.getLogger("httpx").info('HTTP Request: GET https://nextcloud/ocs/v2.php/tasks_provider/next "HTTP/1.1 204"')
        logging.getLogger("presidio_analyzer.analyzer_engine").warning("Recognizer not added to registry")
        logging.getLogger("niquests").error("Connection failed")
        batch.LOGGER.info("Redacting alpha.txt (1 of 4)")
        main.LOGGER.warning("Not redacting docs/: no file can be created in this folder")
    assert [(level, message) for _thread, level, message in posted] == [
        (LogLvl.INFO, "Redacting alpha.txt (1 of 4)"),
        (LogLvl.WARNING, "Not redacting docs/: no file can be created in this folder"),
    ]
    # The root logger is left alone: it neither carries a handler to Nextcloud nor passes INFO.
    root = logging.getLogger()
    assert not any(isinstance(handler, logging.handlers.QueueHandler) for handler in root.handlers)
    assert root.level == logging.WARNING


def test_a_library_logger_lowered_to_info_does_not_reach_nextcloud():
    posted = []
    with logging_to(posted):
        logging.getLogger("httpx").setLevel(logging.INFO)
        logging.getLogger("httpx").info("HTTP Request: GET https://nextcloud/")
    logging.getLogger("httpx").setLevel(logging.NOTSET)
    assert posted == []


def test_the_app_logs_at_info_and_caplog_still_sees_it(caplog):
    """The batch progress lines are INFO and must pass, and tests keep reading them through the root handlers."""
    with logging_to([]):
        with caplog.at_level(logging.INFO):
            batch.LOGGER.info("Redacting alpha.txt (1 of 4)")
    assert "Redacting alpha.txt (1 of 4)" in caplog.text


def test_no_line_is_posted_from_the_event_loops_thread():
    """The poller logs its errors on the event loop; the loop must not wait for the request that follows."""
    posted = []
    loop_thread = []

    async def scenario():
        loop_thread.append(threading.get_ident())
        loop = asyncio.get_running_loop()
        gaps = []

        async def ticker():
            last = loop.time()
            for _ in range(30):
                await asyncio.sleep(0.01)
                now = loop.time()
                gaps.append(now - last)
                last = now

        ticks = asyncio.create_task(ticker())
        await asyncio.sleep(0.05)
        main.LOGGER.error("Failed to fetch or process the next task")
        await ticks
        return max(gaps)

    # Each post takes 0.3 seconds, as it does for a Nextcloud that is slow to answer.
    with logging_to(posted, stall=0.3):
        worst_gap = asyncio.run(scenario())
    assert posted and posted[0][0] != loop_thread[0]
    assert worst_gap < 0.25


def test_the_traceback_of_a_failure_reaches_nextcloud():
    """The lines sent while the app runs name the file that failed and carry the traceback of the failure."""
    posted = []
    with logging_to(posted):
        try:
            raise ValueError("boom")
        except ValueError:
            batch.LOGGER.exception("Failed to redact docs/a.txt")
    [(level, message)] = [(level, message) for _thread, level, message in posted]
    assert level == LogLvl.ERROR
    assert message.startswith("Failed to redact docs/a.txt\nTraceback (most recent call last):")
    assert message.rstrip().endswith("ValueError: boom")


def test_a_full_queue_drops_new_lines_without_an_error(capsys):
    """The queue fills only while Nextcloud does not answer; a line more must not raise or block."""
    posted = []
    gate = threading.Event()
    with logging_to(posted, gate=gate):
        for number in range(main.LOG_QUEUE_SIZE + 10):
            batch.LOGGER.info("Redacting f%d.txt", number)
        gate.set()
    assert main.LOG_QUEUE_SIZE <= len(posted) <= main.LOG_QUEUE_SIZE + 1
    assert "Logging error" not in capsys.readouterr().err


def test_the_shutdown_does_not_wait_for_a_hanging_request():
    """A post that never gets an answer must not hold up the shutdown of the app."""
    posted = []
    gate = threading.Event()
    clock = time.monotonic()
    with logging_to(posted, gate=gate):
        batch.LOGGER.info("Redacting a.txt (1 of 1)")
        # The sending thread has taken the line and now waits for its answer.
        deadline = time.monotonic() + 5
        while not posted and time.monotonic() < deadline:
            time.sleep(0.01)
        assert posted
    elapsed = time.monotonic() - clock
    gate.set()
    assert main.LOG_FLUSH_SECONDS <= elapsed < main.LOG_FLUSH_SECONDS + 2


def test_lifespan_flushes_the_logs_after_the_worker_ended(monkeypatch):
    """Lines of the worker's last moments are sent too: the sender stops only after the worker."""
    events = []
    listener = object()
    monkeypatch.setattr(main, "configure_logging", lambda: events.append("configured") or listener)
    monkeypatch.setattr(main, "flush_logs", lambda sender: events.append("flushed" if sender is listener else "other"))
    monkeypatch.setattr(main, "set_handlers", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(main, "NextcloudApp", lambda: SimpleNamespace(enabled_state=True))
    monkeypatch.setattr(main, "SHUTDOWN", threading.Event())

    async def worker():
        events.append("worker started")
        while not main.SHUTDOWN.is_set():
            await asyncio.sleep(0.01)
        events.append("worker ended")

    monkeypatch.setattr(main, "process_tasks", worker)

    async def run():
        async with main.lifespan(main.APP):
            events.append("app up")

    asyncio.run(run())
    # The worker task starts when the event loop next runs, i.e. after the app is up.
    assert events == ["configured", "app up", "worker started", "worker ended", "flushed"]
