# SPDX-FileCopyrightText: 2026 Nextcloud GmbH and Nextcloud contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Folder-level batch redaction: recursive text file collection, size and MIME guards, per-file results."""

import logging
import os
import posixpath
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from anonymizer import redact_text
from nc_py_api import NextcloudApp, NextcloudException
from nc_py_api.files import FsNode

LOGGER = logging.getLogger(__name__)

# Redacting a file of this size takes minutes and would monopolize the worker, so larger files are skipped.
DEFAULT_MAX_FILE_BYTES = 10 * 1024 * 1024  # 10 MiB
MAX_FILE_BYTES_ENV = "TEXT_ANONYMIZER_MAX_FILE_BYTES"

REDACTED_SUFFIX_RE = re.compile(r" \(redacted(?: \d+)?\)$")

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


def canonical_redacted_copy_path(user_path: str) -> str:
    """The first candidate path for the redacted copy of ``user_path``, e.g. ``notes (redacted).txt``."""
    directory, name = posixpath.split(user_path)
    stem, ext = posixpath.splitext(name)
    stem = REDACTED_SUFFIX_RE.sub("", stem)
    return posixpath.join(directory, f"{stem} (redacted){ext}")


def redacted_copy_path(nc: NextcloudApp, user_path: str) -> str:
    """Return a free path next to ``user_path`` for the redacted copy, e.g. ``notes (redacted).txt``.

    Redacting an existing copy such as ``notes (redacted).txt`` yields ``notes (redacted 2).txt``.
    """
    directory, name = posixpath.split(user_path)
    stem, ext = posixpath.splitext(name)
    stem = REDACTED_SUFFIX_RE.sub("", stem)
    candidate = canonical_redacted_copy_path(user_path)
    counter = 2
    while path_exists(nc, candidate):
        candidate = posixpath.join(directory, f"{stem} (redacted {counter}){ext}")
        counter += 1
    return candidate


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


def redacted_copy_is_fresh(
    nc: NextcloudApp, node: FsNode, copies: dict[str, list[FsNode]] | None = None
) -> bool:
    """Whether ``node`` is already redacted and up to date, so an action on it is a no-op.

    A regular file is fresh when one of its redacted copies is newer than the file itself.
    A file that is itself a redacted copy is fresh when it is newer than the original it was
    made from; editing the original makes the copy stale and the next run redacts both again.

    ``copies`` is the index of a folder listing (see ``redacted_copies_by_original``); without it,
    the folder of ``node`` is listed.
    """
    if is_redacted_copy(node.user_path):
        try:
            original = nc.files.by_path(_original_path(node.user_path))
        except NextcloudException as e:
            if e.status_code == 404:
                return False
            raise
        return node.info.last_modified > original.info.last_modified
    if copies is None:
        copies = redacted_copies_by_original(nc.files.listdir(posixpath.dirname(node.user_path)))
    return any(copy.info.last_modified > node.info.last_modified for copy in copies.get(node.user_path, []))


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
        return bool(self.results)


def collect_text_files(
    nc: NextcloudApp, folder: FsNode, max_bytes: int
) -> tuple[list[FsNode], list[FileResult], dict[str, list[FsNode]]]:
    """Recursively collect text files under ``folder`` as ``(processable, oversize-skipped, redacted-copies)``.

    Non-text files are ignored outright; text files over ``max_bytes`` are reported so each can be warned about.
    ``redacted-copies`` holds every redacted copy found, keyed by the path of its original.
    """
    files: list[FsNode] = []
    oversize: list[FileResult] = []
    copies: dict[str, list[FsNode]] = {}
    folders = [folder]
    while folders:
        current = folders.pop()
        nodes = nc.files.listdir(current)
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
    return files, oversize, copies


def redact_file_copy(nc: NextcloudApp, node: FsNode, redact: RedactFn) -> str:
    """Write a redacted copy of ``node`` next to the original and return the copy's path."""
    text = nc.files.download(node).decode("utf-8")
    copy_path = redacted_copy_path(nc, node.user_path)
    nc.files.upload(copy_path, redact(text))
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
    before the failure keep their copies. A file whose redacted copy already exists and is newer than the
    original is skipped, so a repeated action on an unchanged folder is a no-op. ``on_progress`` is called
    with ``(index, total, filename)`` before each file is processed.
    """
    if redact is None:
        redact = redact_text
    limit = max_file_bytes() if max_bytes is None else max_bytes
    report = BatchReport(folder_name=folder.name)
    files, oversize, copies = collect_text_files(nc, folder, limit)
    report.results.extend(oversize)
    total = len(files)
    for index, node in enumerate(files, 1):
        if on_progress is not None:
            on_progress(index, total, node.name)
        if redacted_copy_is_fresh(nc, node, copies):
            report.results.append(FileResult(node, STATUS_SKIPPED, "a redacted copy is already up to date"))
            continue
        try:
            detail = redact_file_copy(nc, node, redact)
        except UnicodeDecodeError:
            report.results.append(FileResult(node, STATUS_FAILED, "not a UTF-8 encoded text file"))
        except Exception as e:
            LOGGER.exception("Failed to redact %s", node.user_path)
            report.results.append(FileResult(node, STATUS_FAILED, str(e) or e.__class__.__name__))
        else:
            report.results.append(FileResult(node, STATUS_REDACTED, detail))
    return report
