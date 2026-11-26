# SPDX-FileCopyrightText: 2026 Nextcloud GmbH and Nextcloud contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""What an allow-list pattern keeps: a detection whose whole text it matches, or one inside its match in the text."""

import logging
import random
import time

import pytest
import regex
from presidio_analyzer import RecognizerResult
from presidio_anonymizer import AnonymizerEngine

import anonymizer
import config
from config import ENV_ALLOW_PATTERNS, Settings, filter_results, parse_allow_patterns

LOGGER_NAME = "text_anonymizer.config"


class FixedAnalyzer:
    """Stand-in for Presidio's analyzer: one detection of 0.85 per ``(entity type, text)`` rule."""

    def __init__(self, *rules):
        self.rules = rules

    def analyze(self, text, language):
        return [
            RecognizerResult(entity_type, text.index(value), text.index(value) + len(value), 0.85)
            for entity_type, value in self.rules
        ]


@pytest.fixture
def engines(monkeypatch):
    """Make redact_text run the real AnonymizerEngine on the detections of a stand-in analyzer."""

    def install(*rules):
        monkeypatch.setattr(anonymizer, "_get_engines", lambda: (FixedAnalyzer(*rules), AnonymizerEngine()))

    return install


LINE = "Anna Meyer grew up in Berlin.\n"


def _result(entity_type, text, value, start=None):
    start = text.index(value) if start is None else start
    return RecognizerResult(entity_type, start, start + len(value), 0.85)


def _long_text(lines):
    """``lines`` lines of ``LINE``, with a detection of "Anna Meyer" and one of "Berlin" per line."""
    text = LINE * lines
    results = []
    for index in range(lines):
        offset = index * len(LINE)
        for entity_type, value in (("PERSON", "Anna Meyer"), ("LOCATION", "Berlin")):
            start = offset + LINE.index(value)
            results.append(_result(entity_type, text, value, start))
    return text, results


def _kept_by_scanning(pattern, results, text):
    """The rule of the allow-list followed to the letter: kept when the pattern matches the whole text of the
    detection, or that text widened to word boundaries, or when the detection lies inside a match of the
    pattern in the text, looked for among every match."""
    regions = [match.span() for match in pattern.finditer(text) if match.end() > match.start()]
    kept = set()
    for index, result in enumerate(results):
        start, end = config._name_bounds(text, result.start, result.end)
        if end <= start:
            continue
        if pattern.fullmatch(text[start:end]) is not None or pattern.fullmatch(config._widened_span(text, start, end)):
            kept.add(index)
        elif any(region_start <= start and end <= region_end for region_start, region_end in regions):
            kept.add(index)
    return kept


@pytest.mark.parametrize(
    ("text", "detection", "pattern", "redacted"),
    [
        pytest.param("Paris Hilton attended the premiere.", "Paris Hilton", "Paris", "<PERSON> attended the premiere.", id="reviewer"),
        pytest.param("Nextcloud GmbH employs Anna Meyer.", "Nextcloud GmbH", "Nextcloud", "<PERSON> employs Anna Meyer.", id="company"),
        pytest.param("Berlin was founded in 1237.", "Berlin", "erli", "<PERSON> was founded in 1237.", id="inside-a-word"),
    ],
)
def test_a_pattern_that_is_only_part_of_a_name_does_not_keep_it(engines, text, detection, pattern, redacted):
    """The unanchored search kept the whole name for a pattern that named only a part of it."""
    engines(("PERSON", detection))
    assert anonymizer.redact_text(text, settings=Settings(allow_patterns=(pattern,))) == redacted


def test_a_detection_is_kept_when_its_whole_text_is_the_pattern():
    text = "Paris Hilton attended the premiere in Berlin."
    results = [_result("PERSON", text, "Paris Hilton"), _result("LOCATION", text, "Berlin")]
    settings = Settings(allow_patterns=("Paris Hilton", r"^Berlin$"))
    assert filter_results(results, text, settings) == []


def test_a_drifted_detection_inside_an_allow_listed_phrase_is_kept():
    """A detector may bound only part of "Nextcloud GmbH": what it cut mid-word is widened, what it cut at a
    word boundary sits inside the match of the pattern in the text."""
    text = "Nextcloud GmbH employs Anna Meyer."
    settings = Settings(allow_patterns=("Nextcloud GmbH",))
    assert filter_results([_result("ORGANIZATION", text, "Nextcloud Gmb", 0)], text, settings) == []
    assert filter_results([_result("ORGANIZATION", text, "Nextcloud", 0)], text, settings) == []
    assert filter_results([_result("ORGANIZATION", text, "GmbH")], text, settings) == []


def test_overlapping_detections_of_an_allow_listed_phrase_are_both_kept():
    """The filter runs before the anonymizer resolves overlaps, so every detection inside the phrase goes."""
    text = "Contact Nextcloud GmbH today."
    results = [_result("ORGANIZATION", text, "Nextcloud GmbH"), _result("PERSON", text, "Nextcloud")]
    assert filter_results(results, text, Settings(allow_patterns=("Nextcloud GmbH",))) == []


@pytest.mark.parametrize("ending", ["'s", "\N{RIGHT SINGLE QUOTATION MARK}s"])
def test_a_detection_that_carries_a_possessive_is_the_name(engines, ending):
    """The recognizer bounds the PERSON as "Anna Meyer's" (seen live with the spaCy model): the possessive is
    grammar, not part of the name, so the pattern of the name keeps the detection."""
    text = f"It is Anna Meyer{ending} office in Berlin."
    engines(("PERSON", f"Anna Meyer{ending}"), ("LOCATION", "Berlin"))
    assert anonymizer.redact_text(text, settings=Settings(allow_patterns=("Anna Meyer",))) == (
        f"It is Anna Meyer{ending} office in <LOCATION>."
    )


def test_spaces_a_detection_carries_around_a_name_are_not_part_of_it():
    text = "It is Anna Meyer's office."
    padded = RecognizerResult("PERSON", 6, 18, 0.85)  # " Anna Meyer's "
    assert filter_results([padded], text, Settings(allow_patterns=("Anna Meyer",))) == []


def test_a_possessive_on_its_own_is_not_a_name():
    bare = RecognizerResult("PERSON", 10, 12, 0.85)  # "'s"
    assert filter_results([bare], "It is Anna Meyer's office.", Settings(allow_patterns=(r"['\u2019]s",))) == [bare]


def test_a_detection_longer_than_the_pattern_is_not_kept():
    text = "Nextcloud GmbH & Co. KG employs Anna Meyer."
    results = [_result("ORGANIZATION", text, "Nextcloud GmbH & Co. KG")]
    assert filter_results(results, text, Settings(allow_patterns=("Nextcloud GmbH",))) == results


@pytest.mark.parametrize("pattern", [r"(Inc\.)?", ".*", "Acme Corp|", "a*", "x?", r"(?:Acme Corp)*"])
def test_a_pattern_that_can_match_nothing_is_refused(caplog, pattern):
    """Such a pattern is a mistake (an optional group or a trailing bar) and used to keep every detection."""
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        assert parse_allow_patterns(pattern) == ()
    assert repr(pattern) in caplog.text and "it can match nothing" in caplog.text


@pytest.mark.parametrize("pattern", [r"Nextcloud( GmbH)?", "Acme Corp", r"\w+"])
def test_a_pattern_that_needs_at_least_one_character_is_accepted(pattern):
    assert parse_allow_patterns(pattern)[0][0] == pattern


def test_a_pattern_with_zero_length_matches_keeps_nothing(engines):
    """Word boundaries and lookaheads match no text, so no detection has the text of such a match."""
    text = "Anna Meyer works in Berlin."
    engines(("PERSON", "Anna Meyer"), ("LOCATION", "Berlin"))
    settings = Settings(allow_patterns=(r"\b", r"(?=Anna)", r"(?<=A)"))
    assert anonymizer.redact_text(text, settings=settings) == "<PERSON> works in <LOCATION>."


def test_a_detection_without_text_is_never_suppressed_by_the_allow_list():
    text = "Anna Meyer works in Berlin."
    empty = RecognizerResult("PERSON", 5, 5, 0.85)
    for pattern in (r"\b", r"(?=M)", r"Berlin", r"M*"):
        assert filter_results([empty], text, Settings(allow_patterns=(pattern,))) == [empty]


def test_an_optional_group_no_longer_leaves_the_redacted_copy_unredacted(engines):
    r"""The reviewer's example: (Inc\.)? used to keep every entity, so the redacted copy came back as the original."""
    text = "Anna Meyer works at Acme Corp in Berlin."
    engines(("PERSON", "Anna Meyer"), ("ORGANIZATION", "Acme Corp"), ("LOCATION", "Berlin"))
    assert anonymizer.redact_text(text, settings=Settings(allow_patterns=(r"(Inc\.)?",))) == (
        "<PERSON> works at <ORGANIZATION> in <LOCATION>."
    )


def test_an_alternation_on_one_line_behaves_like_several_patterns_longest_first():
    """The admin form takes one line: several patterns go in as an alternation. The match of the pattern in the
    text starts with the first alternative that fits, so a shorter one placed first hides the longer ones."""
    text = "Contact Nextcloud GmbH today."
    whole = _result("ORGANIZATION", text, "Nextcloud GmbH")
    part = _result("ORGANIZATION", text, "Nextcloud")
    tail = _result("ORGANIZATION", text, "GmbH")
    assert filter_results([whole, part, tail], text, Settings(allow_patterns=("Nextcloud GmbH|Nextcloud",))) == []
    assert filter_results([whole, part], text, Settings(allow_patterns=("Nextcloud|Nextcloud GmbH",))) == []
    assert filter_results([tail], text, Settings(allow_patterns=("Nextcloud|Nextcloud GmbH",))) == [tail]


def test_a_value_saved_with_an_alternation_is_one_pattern():
    from config import apply_admin_config

    layered = apply_admin_config(Settings(), {ENV_ALLOW_PATTERNS: "Nextcloud GmbH|Acme Corp"})
    assert layered.allow_patterns == ("Nextcloud GmbH|Acme Corp",)


def test_line_breaks_are_still_separators_for_the_environment_variable():
    sources = tuple(source for source, _ in parse_allow_patterns("Nextcloud GmbH\nAcme Corp\n"))
    assert sources == ("Nextcloud GmbH", "Acme Corp")


def test_a_literal_pattern_keeps_its_detections_on_a_long_text(caplog):
    """The maintainer's case: a 230 KB text with 15,000 detections and the pattern Berlin.

    For every detection that the two whole-text matches did not keep, the lookup went through all the
    regions of the pattern, 7,500 x 7,500 times here, until the budget ran out and nothing was kept.
    """
    text, results = _long_text(7500)
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        started = time.perf_counter()
        assert filter_results(results, text, Settings(allow_patterns=("Berlin",))) == results[0::2]
        assert time.perf_counter() - started < 1.0
    assert caplog.text == ""


def test_the_enclosing_region_is_found_as_by_looking_through_every_one():
    """The regions of a pattern do not overlap and come in order, so the one with the greatest start at or
    before a detection is the only one that can enclose it. That lookup must answer as a look through every
    region does, whatever the pattern and the detections."""
    shapes = ["a", "ab", "a+?", "a+?b*", r"(?=a)|ab", r"\b|ab", r"(?<=a)b|ab", "a(?:b|)a", r"^a|b$", r"\ba\b", "b+a?"]
    generator = random.Random(31)
    for _ in range(300):
        pattern = regex.compile(generator.choice(shapes))
        text = "".join(generator.choice("aab ") for _ in range(generator.randrange(0, 14)))
        results = []
        for _ in range(generator.randrange(0, 6)):
            start = generator.randrange(0, len(text) + 1)
            results.append(RecognizerResult("PERSON", start, start + generator.randrange(0, 4), 0.85))
        assert config._allow_listed(pattern, results, text) == _kept_by_scanning(pattern, results, text), (
            pattern.pattern,
            text,
            [(result.start, result.end) for result in results],
        )
