# SPDX-FileCopyrightText: 2026 Nextcloud GmbH and Nextcloud contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Folder-level batch redaction: recursive text file collection, size and MIME guards, per-file results."""

import json
import logging
import os
import posixpath
import re
import threading
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field

from anonymizer import redact_text
from nc_py_api import NextcloudApp, NextcloudException
from nc_py_api.files import FsNode

LOGGER = logging.getLogger(__name__)

# Redacting a file of this size takes minutes and would monopolize the worker, so larger files are skipped.
DEFAULT_MAX_FILE_BYTES = 10 * 1024 * 1024  # 10 MiB
MAX_FILE_BYTES_ENV = "TEXT_ANONYMIZER_MAX_FILE_BYTES"

REDACTED_SUFFIX_RE = re.compile(r" \(redacted(?: \d+)?\)$")

# Every original remembers the etag it had when its redacted copies were made and which files those copies are, in the
# key/value store of the ExApp in Nextcloud (``nc.appconfig_ex``): one record per original, under
# ``RECORD_KEY_PREFIX`` and its file id.
RECORD_KEY_PREFIX = "redacted_copy_etag/"
RECORDS_PER_REQUEST = 200

# Structured-text types are accepted beyond text/* so exports such as JSON and YAML are covered too.
TEXT_MIMES = frozenset(
    {
        "application/ecmascript",
        "application/javascript",
        "application/json",
        "application/toml",
        "application/x-sh",
        "application/x-shellscript",
        "application/x-yaml",
        "application/xml",
        "application/yaml",
    }
)
TEXT_MIME_SUFFIXES = ("+json", "+xml", "+yaml", "+yml")
# An SVG ends in "+xml" but is a drawing: redaction would rewrite its namespace URL and path data.
NON_TEXT_MIMES = frozenset({"image/svg+xml"})

RedactFn = Callable[[str], str]
ProgressCallback = Callable[[int, int, str], None]

STATUS_REDACTED = "redacted"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"


def max_file_bytes() -> int:
    raw = os.environ.get(MAX_FILE_BYTES_ENV, "")
    if not raw:
        return DEFAULT_MAX_FILE_BYTES
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if value <= 0:
        LOGGER.warning("Invalid %s=%r, using the default of %d bytes", MAX_FILE_BYTES_ENV, raw, DEFAULT_MAX_FILE_BYTES)
        return DEFAULT_MAX_FILE_BYTES
    return value


def format_bytes(size: int) -> str:
    if size >= 1024 * 1024:
        return f"{size / (1024 * 1024):.1f} MB"
    if size >= 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size} bytes"


def oversize_detail(size: int, max_bytes: int | None = None) -> str | None:
    """Return the reason ``size`` exceeds the limit, or ``None`` when the file may be redacted."""
    limit = max_file_bytes() if max_bytes is None else max_bytes
    if size > limit:
        return f"larger than {format_bytes(limit)}"
    return None


def is_text_mime(mimetype: str) -> bool:
    """Whether a DAV-reported MIME type holds redactable text (text/* plus structured-text types)."""
    mime = (mimetype or "").lower()
    if mime in NON_TEXT_MIMES:
        return False
    return mime.startswith("text/") or mime in TEXT_MIMES or mime.endswith(TEXT_MIME_SUFFIXES)


def path_exists(nc: NextcloudApp, user_path: str) -> bool:
    try:
        nc.files.by_path(user_path)
    except NextcloudException as e:
        # WebDAV lookups raise a plain NextcloudException for a missing path, not NextcloudExceptionNotFound.
        if e.status_code == 404:
            return False
        raise
    return True


def _redacted_copy_path(user_path: str, suffix: str) -> str:
    """The path of a redacted copy of ``user_path`` whose name carries ``suffix``, e.g. `` (redacted 2)``."""
    directory, name = posixpath.split(user_path)
    stem, ext = posixpath.splitext(name)
    stem = REDACTED_SUFFIX_RE.sub("", stem)
    return posixpath.join(directory, f"{stem}{suffix}{ext}")


def canonical_redacted_copy_path(user_path: str) -> str:
    """The first candidate path for the redacted copy of ``user_path``, e.g. ``notes (redacted).txt``."""
    return _redacted_copy_path(user_path, " (redacted)")


def _free_copy_path(user_path: str, taken: Callable[[str], bool]) -> str:
    """The first candidate path for the redacted copy of ``user_path`` that ``taken`` does not report as taken."""
    candidate = canonical_redacted_copy_path(user_path)
    counter = 2
    while taken(candidate):
        candidate = _redacted_copy_path(user_path, f" (redacted {counter})")
        counter += 1
    return candidate


def redacted_copy_path(nc: NextcloudApp, user_path: str) -> str:
    """Return a free path next to ``user_path`` for the redacted copy, e.g. ``notes (redacted).txt``.

    A taken candidate is numbered: an original that is redacted again while ``notes (redacted).txt``
    exists gets ``notes (redacted 2).txt``.
    """
    return _free_copy_path(user_path, lambda candidate: path_exists(nc, candidate))


def _original_path(user_path: str) -> str:
    """The path of the file a redacted copy was made from, e.g. ``notes.txt`` for ``notes (redacted).txt``."""
    directory, name = posixpath.split(user_path)
    stem, ext = posixpath.splitext(name)
    return posixpath.join(directory, f"{REDACTED_SUFFIX_RE.sub('', stem)}{ext}")


def is_redacted_copy(user_path: str) -> bool:
    """Whether the file at ``user_path`` is named like a redacted copy, e.g. ``notes (redacted 2).txt``."""
    return REDACTED_SUFFIX_RE.search(posixpath.splitext(posixpath.basename(user_path))[0]) is not None


def redacted_copies_by_original(nodes: Iterable[FsNode]) -> dict[str, list[FsNode]]:
    """The redacted copies among ``nodes``, keyed by the path of the original each was made from."""
    copies: dict[str, list[FsNode]] = {}
    for node in nodes:
        if not node.is_dir and is_redacted_copy(node.user_path):
            copies.setdefault(_original_path(node.user_path), []).append(node)
    return copies


@dataclass(frozen=True)
class CopyRecord:
    """What is remembered about the redacted copies of one original."""

    etag: str
    """The etag the original had when its copies were made."""
    copy_ids: tuple[str, ...]
    """The full file ids (``FsNode.file_id``) of the copies made at that etag, wherever they are."""


def record_key(node: FsNode) -> str:
    return RECORD_KEY_PREFIX + str(node.info.fileid)


def copy_records(nc: NextcloudApp, nodes: Iterable[FsNode]) -> dict[int, CopyRecord]:
    """The records of ``nodes``, by file id; a file without a record, or with one that cannot be read, is left out."""
    keys = [record_key(node) for node in nodes]
    records: dict[int, CopyRecord] = {}
    for start in range(0, len(keys), RECORDS_PER_REQUEST):
        for record in nc.appconfig_ex.get_values(keys[start : start + RECORDS_PER_REQUEST]):
            with suppress(ValueError, KeyError, TypeError):
                data = json.loads(record.value)
                records[int(record.key[len(RECORD_KEY_PREFIX) :])] = CopyRecord(
                    data["etag"], tuple(str(copy_id) for copy_id in data["copies"])
                )
    return records


def remember_copy(nc: NextcloudApp, node: FsNode, copy: FsNode, record: CopyRecord | None) -> None:
    """Remember ``copy`` as made from the content ``node`` has now, next to the copies made at that etag already."""
    made = record.copy_ids if record is not None and record.etag == node.etag_unquoted else ()
    value = json.dumps({"etag": node.etag_unquoted, "copies": [*made, copy.file_id]})
    nc.appconfig_ex.set_value(record_key(node), value)


def redacted_copy_is_fresh(node: FsNode, copies: dict[str, list[FsNode]], record: CopyRecord | None) -> bool:
    """Whether ``node`` is already redacted and up to date, so an action on it is a no-op.

    A redacted copy is never redacted again, whatever its age. A regular file is fresh when one of the redacted copies
    of it in ``copies`` (the index of the listing of its folder, see ``redacted_copies_by_original``) is among those
    that ``record`` names, i.e. a copy the app made, and ``record`` carries the etag the file has now, i.e. a copy
    was made from the content the file still has. The modification time cannot tell that: a restored version or a
    file that a sync client uploaded with its old local timestamp changes the content and leaves the file no newer
    than its copy. ``record`` is what ``copy_records`` returned for the file; ``None`` when nothing is remembered.
    """
    if is_redacted_copy(node.user_path):
        return True
    if record is None or not node.etag_unquoted or record.etag != node.etag_unquoted:
        return False
    return any(copy.file_id in record.copy_ids for copy in copies.get(node.user_path, []))


@dataclass
class FileResult:
    node: FsNode
    status: str
    detail: str = ""
    """Copy path for redacted files, skip reason or error message otherwise."""


@dataclass
class BatchReport:
    folder_name: str
    results: list[FileResult] = field(default_factory=list)
    """Subfolders that could not be listed; their files are not part of the results."""
    unreadable_folders: list[FsNode] = field(default_factory=list)

    def _with_status(self, status: str) -> list[FileResult]:
        return [result for result in self.results if result.status == status]

    @property
    def redacted(self) -> list[FileResult]:
        return self._with_status(STATUS_REDACTED)

    @property
    def failed(self) -> list[FileResult]:
        return self._with_status(STATUS_FAILED)

    @property
    def skipped(self) -> list[FileResult]:
        return self._with_status(STATUS_SKIPPED)

    @property
    def processed_count(self) -> int:
        return len(self.redacted) + len(self.failed)

    @property
    def found_any(self) -> bool:
        return bool(self.results or self.unreadable_folders)


def collect_text_files(
    nc: NextcloudApp, folder: FsNode, max_bytes: int
) -> tuple[list[FsNode], list[FileResult], dict[str, list[FsNode]], list[FsNode]]:
    """Recursively collect text files under ``folder`` as ``(processable, oversize-skipped, redacted-copies, unreadable)``.

    Non-text files are ignored outright; text files over ``max_bytes`` are reported so each can be warned about.
    ``redacted-copies`` holds every redacted copy found, keyed by the path of its original. ``unreadable`` holds the
    subfolders that could not be listed; their files are left out. An error on ``folder`` itself propagates.
    """
    files: list[FsNode] = []
    oversize: list[FileResult] = []
    copies: dict[str, list[FsNode]] = {}
    unreadable: list[FsNode] = []
    folders = [folder]
    while folders:
        current = folders.pop()
        try:
            nodes = nc.files.listdir(current)
        except NextcloudException as e:
            if current is folder:
                raise
            LOGGER.warning("Skipped %s: the folder could not be listed (%s)", current.user_path, e)
            unreadable.append(current)
            continue
        for original, found in redacted_copies_by_original(nodes).items():
            copies.setdefault(original, []).extend(found)
        for node in nodes:
            if node.is_dir:
                folders.append(node)
            elif is_text_mime(node.info.mimetype):
                reason = oversize_detail(node.info.size, max_bytes)
                if reason is None:
                    files.append(node)
                else:
                    oversize.append(FileResult(node, STATUS_SKIPPED, reason))
                    LOGGER.warning("Skipped %s: %s", node.user_path, reason)
    files.sort(key=lambda node: node.user_path)
    oversize.sort(key=lambda result: result.node.user_path)
    return files, oversize, copies, unreadable


# An original is redacted while the lock of its file id is held, so that two actions on the same file, such as
# a folder run and a file action on a file of that folder, never write the same copy. The locks live in this
# process only: they cover the actions that run in the threads of one ExApp process.
_FILE_LOCKS: dict[int, tuple[threading.Lock, int]] = {}
_FILE_LOCKS_GUARD = threading.Lock()


@contextmanager
def _file_lock(file_id: int) -> Iterator[None]:
    """Hold the lock of the file while the block runs; the lock goes once its last holder and waiter are gone."""
    with _FILE_LOCKS_GUARD:
        lock, users = _FILE_LOCKS.get(file_id, (threading.Lock(), 0))
        _FILE_LOCKS[file_id] = (lock, users + 1)
    try:
        with lock:
            yield
    finally:
        with _FILE_LOCKS_GUARD:
            lock, users = _FILE_LOCKS[file_id]
            if users == 1:
                del _FILE_LOCKS[file_id]
            else:
                _FILE_LOCKS[file_id] = (lock, users - 1)


def redact_file_copy(
    nc: NextcloudApp,
    node: FsNode,
    redact: RedactFn,
    copies: dict[str, list[FsNode]] | None = None,
    record: CopyRecord | None = None,
) -> str | None:
    """Write a redacted copy of ``node`` next to the original and return the copy's path.

    Returns ``None`` when a copy is up to date by then. The check runs under the lock of the original, because
    another action may have written the copy since ``node`` was found: against a new listing and a new record,
    or, when ``copies`` and ``record`` are the index of the listing that found ``node`` and what ``copy_records``
    returned then, only if the first free path is no longer the one that index announces, i.e. if a copy appeared
    since. ``record`` is also the one the new copy is added to.
    """
    with _file_lock(node.info.fileid):
        if copies is None:
            copies = redacted_copies_by_original(nc.files.listdir(posixpath.dirname(node.user_path)))
            record = copy_records(nc, [node]).get(node.info.fileid)
            if redacted_copy_is_fresh(node, copies, record):
                return None
            copy_path = redacted_copy_path(nc, node.user_path)
        else:
            copy_path = redacted_copy_path(nc, node.user_path)
            taken = {copy.user_path for copy in copies.get(node.user_path, [])}
            if copy_path != _free_copy_path(node.user_path, taken.__contains__):
                # A copy appeared since the listing: decide again, on the folder and the record as they are now.
                copies = redacted_copies_by_original(nc.files.listdir(posixpath.dirname(node.user_path)))
                record = copy_records(nc, [node]).get(node.info.fileid)
                if redacted_copy_is_fresh(node, copies, record):
                    return None
        text = nc.files.download(node).decode("utf-8")
        copy = nc.files.upload(copy_path, redact(text))
        remember_copy(nc, node, copy, record)
        return copy_path


def redact_folder(
    nc: NextcloudApp,
    folder: FsNode,
    redact: RedactFn | None = None,
    on_progress: ProgressCallback | None = None,
    max_bytes: int | None = None,
) -> BatchReport:
    """Redact every text file under ``folder`` recursively, writing each copy next to its original.

    Files are processed one at a time; a failing file is recorded and the run continues, so files redacted
    before the failure keep their copies. A subfolder that cannot be listed is recorded too and its files are
    left out. A redacted copy is skipped whatever its age, and a file whose redacted copy is newer than itself is
    skipped too, so a repeated action on an unchanged folder is a no-op. ``on_progress`` is called with
    ``(index, total, filename)`` before each file is processed.
    """
    if redact is None:
        redact = redact_text
    limit = max_file_bytes() if max_bytes is None else max_bytes
    files, oversize, copies, unreadable = collect_text_files(nc, folder, limit)
    report = BatchReport(folder_name=folder.name, results=oversize, unreadable_folders=unreadable)
    # Every original needs its record, also one with no copy in this folder: another user may hold a current copy.
    try:
        records = copy_records(nc, [node for node in files if not is_redacted_copy(node.user_path)])
    except NextcloudException as e:
        # Without the records a file with a copy can be neither redacted again nor taken for done.
        records = {}
        for node in files:
            if node.user_path in copies and not is_redacted_copy(node.user_path):
                report.results.append(FileResult(node, STATUS_FAILED, str(e) or e.__class__.__name__))
        files = [node for node in files if node.user_path not in copies]
    for index, node in enumerate(files, 1):
        if on_progress is not None:
            on_progress(index, len(files), node.name)
        try:
            if is_redacted_copy(node.user_path):
                report.results.append(FileResult(node, STATUS_SKIPPED, "a redacted copy is never redacted again"))
                continue
            if redacted_copy_is_fresh(node, copies, records.get(node.info.fileid)):
                report.results.append(FileResult(node, STATUS_SKIPPED, "a redacted copy is already up to date"))
                continue
            detail = redact_file_copy(nc, node, redact, copies, records.get(node.info.fileid))
        except UnicodeDecodeError:
            report.results.append(FileResult(node, STATUS_FAILED, "not a UTF-8 encoded text file"))
        except Exception as e:
            LOGGER.exception("Failed to redact %s", node.user_path)
            report.results.append(FileResult(node, STATUS_FAILED, str(e) or e.__class__.__name__))
        else:
            if detail is None:
                report.results.append(FileResult(node, STATUS_SKIPPED, "a redacted copy is already up to date"))
            else:
                report.results.append(FileResult(node, STATUS_REDACTED, detail))
    return report
