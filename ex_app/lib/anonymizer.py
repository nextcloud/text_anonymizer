# SPDX-FileCopyrightText: 2026 Nextcloud GmbH and Nextcloud contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""PII detection and redaction using Presidio."""

import threading
from collections.abc import Callable, Iterator

from presidio_analyzer import AnalyzerEngine, RecognizerResult
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig, OperatorResult

ProgressCallback = Callable[[float], None]

_LOCK = threading.Lock()
_ENGINES: tuple[AnalyzerEngine, AnonymizerEngine] | None = None
_KEEP_OPERATORS = {"DEFAULT": OperatorConfig("keep")}
# Entity types that turn up in ordinary sentences without identifying anyone, so contains_pii does not count them.
NON_IDENTIFYING_TYPES = frozenset({"DATE_TIME", "URL"})
# Analysis time grows faster than text length, and spaCy rejects texts over 1,000,000 characters.
CHUNK_SIZE = 10_000


def _get_engines() -> tuple[AnalyzerEngine, AnonymizerEngine]:
    """Create the Presidio engines on first use, as loading the spaCy model takes several seconds."""
    global _ENGINES
    with _LOCK:
        if _ENGINES is None:
            _ENGINES = (AnalyzerEngine(), AnonymizerEngine())
        return _ENGINES


def _chunk_bounds(text: str, max_size: int) -> list[tuple[int, int]]:
    """Split ``text`` into ``(start, end)`` ranges of at most ``max_size`` characters.

    Ranges end after the last line break in the window, or after the last space or tab if there is none, so that
    entities are rarely cut in two.
    """
    bounds = []
    start = 0
    while len(text) - start > max_size:
        window = text[start : start + max_size]
        cut = window.rfind("\n") + 1 or max(window.rfind(" "), window.rfind("\t")) + 1 or max_size
        bounds.append((start, start + cut))
        start += cut
    bounds.append((start, len(text)))
    return bounds


def _analyze_chunks(
    analyzer: AnalyzerEngine, text: str, language: str, on_progress: ProgressCallback | None
) -> Iterator[tuple[int, int, list[RecognizerResult]]]:
    """Yield ``(start, end, results)`` for each chunk ``text[start:end]``, the results positioned in the chunk.

    Calls ``on_progress`` with the percentage of the text analyzed after each chunk.
    """
    bounds = _chunk_bounds(text, CHUNK_SIZE)
    for i, (start, end) in enumerate(bounds, 1):
        results = analyzer.analyze(text=text[start:end], language=language)
        if on_progress:
            on_progress(i / len(bounds) * 100)
        yield start, end, results


def _analyze(
    analyzer: AnalyzerEngine, text: str, language: str, on_progress: ProgressCallback | None
) -> list[RecognizerResult]:
    """Analyze ``text`` chunk by chunk, returning the results with their positions in the whole text."""
    results = []
    for start, _, chunk_results in _analyze_chunks(analyzer, text, language, on_progress):
        for result in chunk_results:
            result.start += start
            result.end += start
        results.extend(chunk_results)
    return results


def _resolve_overlaps(
    anonymizer: AnonymizerEngine, text: str, results: list[RecognizerResult]
) -> list[OperatorResult]:
    """Return the entities left after resolving the overlaps in ``results`` as :func:`redact_text` does."""
    return anonymizer.anonymize(text=text, analyzer_results=results, operators=_KEEP_OPERATORS).items


def detect_pii(text: str, language: str = "en", on_progress: ProgressCallback | None = None) -> str:
    """List each distinct PII entity in ``text`` as a ``TYPE: value`` line, in order of first appearance.

    Overlapping entities are resolved the same way as in :func:`redact_text`, so the list matches what would be
    redacted. Returns an empty string if no PII was found.
    """
    analyzer, anonymizer = _get_engines()
    items = _resolve_overlaps(anonymizer, text, _analyze(analyzer, text, language, on_progress))
    entities = dict.fromkeys((item.entity_type, item.text) for item in sorted(items, key=lambda item: item.start))
    return "\n".join(f"{entity_type}: {value}" for entity_type, value in entities)


def contains_pii(text: str, language: str = "en", on_progress: ProgressCallback | None = None) -> bool:
    """Return whether ``text`` holds an entity that can identify a person.

    Overlapping entities are resolved as in :func:`detect_pii` before ``NON_IDENTIFYING_TYPES`` are left out, so
    the answer agrees with what the other tasks report. The chunks after the first one holding such an entity are
    not analyzed.
    """
    analyzer, anonymizer = _get_engines()
    for start, end, chunk_results in _analyze_chunks(analyzer, text, language, on_progress):
        items = _resolve_overlaps(anonymizer, text[start:end], chunk_results)
        if any(item.entity_type not in NON_IDENTIFYING_TYPES for item in items):
            return True
    return False


def redact_text(text: str, language: str = "en", on_progress: ProgressCallback | None = None) -> str:
    """Replace every PII entity in ``text`` with a placeholder naming its type, e.g. ``<PERSON>``."""
    analyzer, anonymizer = _get_engines()
    results = _analyze(analyzer, text, language, on_progress)
    return anonymizer.anonymize(text=text, analyzer_results=results).text
