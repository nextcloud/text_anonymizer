# SPDX-FileCopyrightText: 2026 Nextcloud GmbH and Nextcloud contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Detection settings: built-in defaults, environment overrides and admin appconfig overrides.

Values are resolved lowest to highest: built-in default (no threshold, nothing ignored,
nothing allow-listed), ``TEXT_ANONYMIZER_*`` environment variable, admin value saved
through the declarative settings form (``oc_appconfig_ex``). The declarative form stores
by field id, so the form field ids and the appconfig keys are the environment variable
names (see :data:`ADMIN_CONFIG_KEYS`).

:func:`filter_results` applies a resolved :class:`Settings` to analyzer results, so the
detection pipeline (anonymizer module), the evaluation harness (evals/) and any task wiring
share one predicate for the three knobs.
"""

from __future__ import annotations

import logging
import os
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import regex

if TYPE_CHECKING:
    from presidio_analyzer import RecognizerResult

LOGGER = logging.getLogger("text_anonymizer.config")

#: Wall-clock seconds one allow-list pattern may spend on one analysis. A sound pattern needs
#: microseconds there; one that needs more, such as a quantified group that backtracks on long
#: text, is ignored for that analysis, so its entities are treated as not allow-listed.
ALLOW_PATTERN_TIME_BUDGET = 0.25

#: Allow-list patterns that ran out of time, so that the warning is logged once per process.
_TIMED_OUT_PATTERNS: set[str] = set()

#: The confidence threshold suggested in the settings form. No threshold is in force until
#: the administrator saves one or the environment sets ``TEXT_ANONYMIZER_SCORE_THRESHOLD``;
#: this value is what the form pre-fills, not an implicit filter.
DEFAULT_SCORE_THRESHOLD = 0.5

ENV_SCORE_THRESHOLD = "TEXT_ANONYMIZER_SCORE_THRESHOLD"
ENV_IGNORED_ENTITY_TYPES = "TEXT_ANONYMIZER_IGNORED_ENTITY_TYPES"
ENV_ALLOW_PATTERNS = "TEXT_ANONYMIZER_ALLOW_PATTERNS"

#: The oc_appconfig_ex keys the declarative settings form writes (one per knob).
ADMIN_CONFIG_KEYS = (ENV_SCORE_THRESHOLD, ENV_IGNORED_ENTITY_TYPES, ENV_ALLOW_PATTERNS)


def _parse_score_threshold(raw: str) -> float | None:
    """Parse a threshold string, returning None for anything outside ``[0.0, 1.0]`` or unparseable."""
    try:
        value = float(raw)
    except ValueError:
        return None
    if not 0.0 <= value <= 1.0:
        return None
    return value


def parse_ignored_entity_types(raw: str) -> frozenset[str]:
    """Split a comma-separated entity-type list, dropping empty items.

    Presidio entity types are uppercase by convention (``PERSON``, ``URL``, ...), so items
    are normalized; whitespace around separators is ignored.
    """
    return frozenset(item.strip().upper() for item in raw.split(",") if item.strip())


#: A counted repeat: ``{m}``, ``{m,}``, ``{,n}`` or ``{m,n}``.
_COUNTED_REPEAT = re.compile(r"\{\d*(,\d*)?\}")


def _reads_as_fuzzy_match(source: str) -> bool:
    """Whether ``source`` has a ``{`` outside a character class that opens no counted repeat.

    Python's ``re`` reads such a brace as a literal, but the ``regex`` engine, which runs the
    allow-list for its time bound, reads what follows it as a fuzzy-match constraint, so the
    pattern would then keep texts it does not name.
    """
    pos = 0
    while pos < len(source):
        char = source[pos]
        if char == "\\":
            pos += 2
        elif char == "[":
            end = pos + 1 + (source[pos + 1 : pos + 2] == "^")
            end += source[end : end + 1] == "]"  # a leading ] is a member of the class, not its end
            while end < len(source) and source[end] != "]":
                end += 2 if source[end] == "\\" else 1
            pos = end + 1
        elif char == "{" and not _COUNTED_REPEAT.match(source, pos):
            return True
        else:
            pos += 1
    return False


def _compile_allow_pattern(source: str) -> regex.Pattern | None:
    """``source`` compiled for the engine that bounds the match time, or None when it is ignored.

    Python's ``re`` decides what is a valid pattern, so the syntax is the documented one and
    the engine's own extensions cannot give a pattern another meaning.
    """
    try:
        re.compile(source)
        if _reads_as_fuzzy_match(source):
            LOGGER.warning(
                "Ignoring allow-list pattern %r: a '{' that opens no counted repeat is a fuzzy match for the regex engine, write it as '\\{'",
                source,
            )
            return None
        compiled = regex.compile(source)
        if compiled.fullmatch("") is not None:
            LOGGER.warning("Ignoring allow-list pattern %r: it can match nothing, so it would stand for any text", source)
            return None
        return compiled
    except (re.error, regex.error) as e:
        LOGGER.warning("Ignoring invalid allow-list pattern %r: %s", source, e)
        return None


def parse_allow_patterns(raw: str) -> tuple[tuple[str, regex.Pattern], ...]:
    """Split a newline-separated regular-expression list into ``(source, compiled)`` pairs.

    Patterns are validated here and dropped with a warning instead of failing every
    later analysis: an invalid one, and one that can match nothing, since it would then
    stand for any text; the remaining patterns stay in force. The patterns are compiled
    for the ``regex`` engine because it bounds the time of a match (see
    :data:`ALLOW_PATTERN_TIME_BUDGET`).
    """
    patterns = []
    for line in raw.splitlines():
        source = line.strip()
        if source and (compiled := _compile_allow_pattern(source)) is not None:
            patterns.append((source, compiled))
    return tuple(patterns)


@dataclass(frozen=True)
class Settings:
    """Resolved detection knobs; see the module docstring for the precedence order.

    ``score_threshold`` is ``None`` until the threshold itself is set through the
    environment or by the administrator: other knobs being active must not silently
    start dropping low-score detections.
    """

    score_threshold: float | None = None
    ignored_entity_types: frozenset[str] = frozenset()
    allow_patterns: tuple[str, ...] = ()

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        """Build settings from ``env`` (defaults to the process environment).

        Raises ``ValueError`` on an invalid threshold, unlike the fail-soft admin layering: a
        container misconfiguration should surface at startup, not silently weaken detection.
        """
        env = os.environ if env is None else env
        raw = env.get(ENV_SCORE_THRESHOLD)
        threshold = None
        if raw:
            threshold = _parse_score_threshold(raw)
            if threshold is None:
                raise ValueError(f"Invalid value for {ENV_SCORE_THRESHOLD}: {raw!r} (expected 0.0 to 1.0)")
        return cls(
            score_threshold=threshold,
            ignored_entity_types=parse_ignored_entity_types(env.get(ENV_IGNORED_ENTITY_TYPES, "")),
            allow_patterns=tuple(source for source, _ in parse_allow_patterns(env.get(ENV_ALLOW_PATTERNS, ""))),
        )


def apply_admin_config(base: Settings, values: Mapping[str, str]) -> Settings:
    """Layer admin-saved appconfig values over ``base``.

    A value set on the server wins; an unset or empty key keeps the lower level. Invalid
    values fail soft to the lower level with a warning naming the key, so an admin typo can
    never break task processing.
    """
    overrides: dict[str, object] = {}
    raw = values.get(ENV_SCORE_THRESHOLD)
    if raw:
        threshold = _parse_score_threshold(raw)
        if threshold is None:
            LOGGER.warning("Ignoring invalid admin value for %s: %r (expected 0.0 to 1.0)", ENV_SCORE_THRESHOLD, raw)
        else:
            overrides["score_threshold"] = threshold
    raw = values.get(ENV_IGNORED_ENTITY_TYPES)
    if raw is not None and raw.strip():
        overrides["ignored_entity_types"] = parse_ignored_entity_types(raw)
    raw = values.get(ENV_ALLOW_PATTERNS)
    if raw is not None and raw.strip():
        overrides["allow_patterns"] = tuple(source for source, _ in parse_allow_patterns(raw))
    return replace(base, **overrides) if overrides else base


#: A possessive at the end of a detection, "Anna Meyer's", which a recognizer counts as part of the name.
_POSSESSIVE = re.compile(r"['\u2019]s$", re.IGNORECASE)


def _name_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    """The part of a detection that is the name itself: without the spaces around it and without a
    possessive at its end."""
    start, end = max(0, start), min(len(text), end)
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if end - start > 2 and _POSSESSIVE.search(text, start, end):
        end -= 2
    return start, end


def _widened_span(text: str, start: int, end: int) -> str:
    """The span extended to the surrounding word boundaries (apostrophes and hyphens count
    as word-internal), so a detection cutting a word in half still hits allow-list patterns
    written for the full word. Spans outside the text are returned unsliced."""
    if not 0 <= start < end <= len(text):
        return text[start:end]
    while start > 0 and (text[start - 1].isalnum() or text[start - 1] in "'-"):
        start -= 1
    while end < len(text) and (text[end].isalnum() or text[end] in "'-"):
        end += 1
    return text[start:end]


def _warn_timed_out(source: str) -> None:
    """Log, once per process, that ``source`` needs more time than the budget allows."""
    if source not in _TIMED_OUT_PATTERNS:
        _TIMED_OUT_PATTERNS.add(source)
        LOGGER.warning(
            "Ignoring allow-list pattern %r where it needs more than %.2f s; a quantified group in it may backtrack",
            source,
            ALLOW_PATTERN_TIME_BUDGET,
        )


def _allow_listed(pattern: regex.Pattern, results: list[RecognizerResult], text: str) -> set[int]:
    """The indexes of the ``results`` that ``pattern`` keeps as-is.

    Raises ``TimeoutError`` once the pattern has spent :data:`ALLOW_PATTERN_TIME_BUDGET` on
    this call, so the caller drops every verdict of the pattern at once instead of keeping the
    entities it had matched before running out of time.
    """
    deadline = time.perf_counter() + ALLOW_PATTERN_TIME_BUDGET

    def seconds_left() -> float:
        left = deadline - time.perf_counter()
        if left <= 0:
            raise TimeoutError
        return left

    regions = []
    for match in pattern.finditer(text, timeout=seconds_left()):
        if match.end() > match.start():
            regions.append(match.span())
        seconds_left()
    listed = set()
    for index, result in enumerate(results):
        start, end = _name_bounds(text, result.start, result.end)
        if end <= start:
            continue
        span = text[start:end]
        widened = _widened_span(text, start, end)
        if pattern.fullmatch(span, timeout=seconds_left()) is not None:
            listed.add(index)
        elif pattern.fullmatch(widened, timeout=seconds_left()) is not None:
            listed.add(index)
        elif any(region_start <= start and end <= region_end for region_start, region_end in regions):
            listed.add(index)
    return listed


def filter_results(results: list[RecognizerResult], text: str, settings: Settings) -> list[RecognizerResult]:
    """Drop results the settings suppress, keeping order.

    A result is dropped when its entity type is ignored, or when its matched text is
    allow-listed (e.g. a public company name that must never be redacted); a score
    threshold drops low-scoring results only when the threshold itself is set (see
    :class:`Settings`). An allow-list pattern suppresses a result when the pattern
    matches the whole span text, the whole span text widened to word boundaries, or
    when the span lies inside a match of the pattern in the analyzed text; the last
    two tolerate recognizer boundary drift around an allow-listed phrase, such as a
    detector bounding only "Nextcloud" inside "Nextcloud GmbH", and a possessive or a space
    a detector adds to a name is not part of it. A pattern that matches only part of a
    longer name does not keep that name.

    Each pattern gets :data:`ALLOW_PATTERN_TIME_BUDGET` per call; one that runs out of it
    is ignored for the whole call, so no administrator's pattern can hold the task worker.
    """
    kept = []
    for result in results:
        if settings.score_threshold is not None and result.score < settings.score_threshold:
            continue
        if result.entity_type in settings.ignored_entity_types:
            continue
        kept.append(result)
    if not settings.allow_patterns or not kept:
        return kept
    suppressed = set()
    for source, pattern in parse_allow_patterns("\n".join(settings.allow_patterns)):
        try:
            suppressed |= _allow_listed(pattern, kept, text)
        except TimeoutError:
            _warn_timed_out(source)
    return [result for index, result in enumerate(kept) if index not in suppressed]
