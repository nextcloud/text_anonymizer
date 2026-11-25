# SPDX-FileCopyrightText: 2026 Nextcloud GmbH and Nextcloud contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""What an allow-list pattern keeps: a detection whose whole text it matches, or one inside its match in the text."""

import pytest
from presidio_analyzer import RecognizerResult
from presidio_anonymizer import AnonymizerEngine

import anonymizer
from config import Settings, filter_results


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


def _result(entity_type, text, value, start=None):
    start = text.index(value) if start is None else start
    return RecognizerResult(entity_type, start, start + len(value), 0.85)


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
