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
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from presidio_analyzer import RecognizerResult

LOGGER = logging.getLogger("text_anonymizer.config")

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


def parse_allow_patterns(raw: str) -> tuple[tuple[str, re.Pattern[str]], ...]:
    """Split a newline-separated regular-expression list into ``(source, compiled)`` pairs.

    Patterns are validated here: an invalid pattern is dropped with a warning instead of
    failing every later analysis, and the remaining patterns stay in force.
    """
    patterns = []
    for line in raw.splitlines():
        source = line.strip()
        if not source:
            continue
        try:
            patterns.append((source, re.compile(source)))
        except re.error as e:
            LOGGER.warning("Ignoring invalid allow-list pattern %r: %s", source, e)
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


def filter_results(results: list[RecognizerResult], text: str, settings: Settings) -> list[RecognizerResult]:
    """Drop results the settings suppress, keeping order.

    A result is dropped when its entity type is ignored, or when its matched text is
    allow-listed (e.g. a public company name that must never be redacted); a score
    threshold drops low-scoring results only when the threshold itself is set (see
    :class:`Settings`). An allow-list pattern suppresses a result when it matches the
    span text, the span text widened to word boundaries, or a region of the analyzed
    text that contains the span; the last two tolerate recognizer boundary drift around
    an allow-listed phrase, such as a detector bounding only "Nextcloud" inside
    "Nextcloud GmbH".
    """
    allow = parse_allow_patterns("\n".join(settings.allow_patterns))
    allowed_regions = [match.span() for _, pattern in allow for match in pattern.finditer(text)]
    kept = []
    for result in results:
        if settings.score_threshold is not None and result.score < settings.score_threshold:
            continue
        if result.entity_type in settings.ignored_entity_types:
            continue
        span = text[result.start : result.end]
        widened = _widened_span(text, result.start, result.end)
        if any(pattern.search(span) is not None or pattern.search(widened) is not None for _, pattern in allow):
            continue
        if any(start <= result.start and result.end <= end for start, end in allowed_regions):
            continue
        kept.append(result)
    return kept
