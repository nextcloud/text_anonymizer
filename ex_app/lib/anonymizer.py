# SPDX-FileCopyrightText: 2026 Nextcloud GmbH and Nextcloud contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""PII detection and redaction using Presidio."""

import threading
from collections.abc import Callable

from presidio_analyzer import AnalyzerEngine, RecognizerResult
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

ProgressCallback = Callable[[float], None]

_LOCK = threading.Lock()
_ENGINES: tuple[AnalyzerEngine, AnonymizerEngine] | None = None
_KEEP_OPERATORS = {"DEFAULT": OperatorConfig("keep")}
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


def _analyze(
    analyzer: AnalyzerEngine, text: str, language: str, on_progress: ProgressCallback | None
) -> list[RecognizerResult]:
    """Analyze ``text`` chunk by chunk, calling ``on_progress`` with a percentage after each chunk."""
    bounds = _chunk_bounds(text, CHUNK_SIZE)
    results = []
    for i, (start, end) in enumerate(bounds, 1):
        for result in analyzer.analyze(text=text[start:end], language=language):
            result.start += start
            result.end += start
            results.append(result)
        if on_progress:
            on_progress(i / len(bounds) * 100)
    return results


def detect_pii(text: str, language: str = "en", on_progress: ProgressCallback | None = None) -> str:
    """List each distinct PII entity in ``text`` as a ``TYPE: value`` line, in order of first appearance.

    Overlapping entities are resolved the same way as in :func:`redact_text`, so the list matches what would be
    redacted. Returns an empty string if no PII was found.
    """
    analyzer, anonymizer = _get_engines()
    results = _analyze(analyzer, text, language, on_progress)
    items = anonymizer.anonymize(text=text, analyzer_results=results, operators=_KEEP_OPERATORS).items
    entities = dict.fromkeys((item.entity_type, item.text) for item in sorted(items, key=lambda item: item.start))
    return "\n".join(f"{entity_type}: {value}" for entity_type, value in entities)


def contains_pii(text: str, language: str = "en", on_progress: ProgressCallback | None = None) -> bool:
    """Return whether any PII entity was found in ``text``.

    Overlap resolution can only drop entities, never add them, so any raw analyzer result means
    PII is present and the (slower) resolution pass is skipped.
    """
    analyzer, _ = _get_engines()
    return bool(_analyze(analyzer, text, language, on_progress))


def pseudonymize_text(
    text: str, language: str = "en", on_progress: ProgressCallback | None = None
) -> tuple[str, dict[str, str]]:
    """Replace every PII entity with a numbered placeholder per unique entity, e.g. ``<PERSON_1>``.

    Each distinct pair of entity type and value is numbered per type in order of first appearance, so the same
    value always resolves to the same placeholder. Returns the pseudonymized text and a mapping of each
    placeholder, e.g. ``PERSON_1``, to its original value, to be kept by the data controller for
    re-identification.
    """
    analyzer, anonymizer = _get_engines()
    results = _analyze(analyzer, text, language, on_progress)
    # The keep pass resolves overlapping entities the same way as the replacement pass below, and its items
    # carry the original values, so placeholders can be assigned before any text is replaced.
    items = anonymizer.anonymize(text=text, analyzer_results=results, operators=_KEEP_OPERATORS).items
    placeholders: dict[tuple[str, str], str] = {}
    counters: dict[str, int] = {}
    for item in sorted(items, key=lambda item: item.start):
        key = (item.entity_type, item.text)
        if key not in placeholders:
            counters[item.entity_type] = counters.get(item.entity_type, 0) + 1
            placeholders[key] = f"{item.entity_type}_{counters[item.entity_type]}"

    def assign(entity_type: str) -> Callable[[str], str]:
        def replace(value: str) -> str:
            return f"<{placeholders[(entity_type, value)]}>"

        return replace

    operators = {
        entity_type: OperatorConfig("custom", {"lambda": assign(entity_type)})
        for entity_type in {result.entity_type for result in results}
    }
    pseudonymized = anonymizer.anonymize(text=text, analyzer_results=results, operators=operators).text
    return pseudonymized, {placeholder: value for (_, value), placeholder in placeholders.items()}


def redact_text(text: str, language: str = "en", on_progress: ProgressCallback | None = None) -> str:
    """Replace every PII entity in ``text`` with a placeholder naming its type, e.g. ``<PERSON>``."""
    analyzer, anonymizer = _get_engines()
    results = _analyze(analyzer, text, language, on_progress)
    return anonymizer.anonymize(text=text, analyzer_results=results).text
