# SPDX-FileCopyrightText: 2026 Nextcloud GmbH and Nextcloud contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

import importlib.util
import itertools
import re

import anonymizer
import pytest
from presidio_analyzer import RecognizerResult
from presidio_anonymizer import AnonymizerEngine


class FakeAnalyzer:
    """Reports every match of the given ``(entity_type, pattern, score)`` rules, recording each analyzed text."""

    supported_languages = ["en"]

    def __init__(self, *rules: tuple[str, str, float]):
        self.rules = rules
        self.calls: list[str] = []

    def analyze(self, text: str, language: str) -> list[RecognizerResult]:
        assert language == "en"
        self.calls.append(text)
        return [
            RecognizerResult(entity_type, m.start(), m.end(), score)
            for entity_type, pattern, score in self.rules
            for m in re.finditer(pattern, text)
        ]


@pytest.fixture
def analyzer(monkeypatch):
    fake = FakeAnalyzer(
        ("PERSON", r"John Smith", 0.85),
        ("EMAIL_ADDRESS", r"[\w.]+@example\.com", 1.0),
    )
    monkeypatch.setattr(anonymizer, "_get_engines", lambda: (fake, AnonymizerEngine()))
    return fake


@pytest.mark.parametrize(
    ("text", "max_size", "expected"),
    [
        ("", 10, [(0, 0)]),
        ("short", 10, [(0, 5)]),
        ("exactly 10", 10, [(0, 10)]),
        ("line one\nline two", 10, [(0, 9), (9, 17)]),
        ("word word word", 10, [(0, 10), (10, 14)]),
        ("tab\tseparated", 10, [(0, 4), (4, 13)]),
        ("abcdefghijklmnop", 10, [(0, 10), (10, 16)]),
        ("a b\ncd efgh ijk", 10, [(0, 4), (4, 12), (12, 15)]),
    ],
)
def test_chunk_bounds(text, max_size, expected):
    assert anonymizer._chunk_bounds(text, max_size) == expected


@pytest.mark.parametrize("max_size", [1, 3, 7, 50])
def test_chunk_bounds_cover_text_contiguously(max_size):
    text = "Lorem ipsum dolor sit amet,\nconsectetur adipiscing elit.\tSed do eiusmod tempor."
    bounds = anonymizer._chunk_bounds(text, max_size)
    assert bounds[0][0] == 0
    assert bounds[-1][1] == len(text)
    for (_, end), (start, _) in itertools.pairwise(bounds):
        assert end == start
    assert all(0 < end - start <= max_size for start, end in bounds)


def test_analyze_offsets_results_and_reports_progress(monkeypatch):
    monkeypatch.setattr(anonymizer, "CHUNK_SIZE", 20)
    fake = FakeAnalyzer(("PERSON", r"John Smith", 0.85))
    text = "Hello John Smith.\nBye John Smith.\nNothing here."
    progress = []

    results = anonymizer._analyze(fake, text, "en", progress.append)

    assert len(fake.calls) == 3
    assert [text[r.start : r.end] for r in results] == ["John Smith", "John Smith"]
    assert progress == pytest.approx([100 / 3, 200 / 3, 100])


def test_detect_pii_lists_distinct_entities_in_order(analyzer):
    text = "Mail john.smith@example.com. I am John Smith. Again: John Smith, john.smith@example.com."
    assert anonymizer.detect_pii(text) == "EMAIL_ADDRESS: john.smith@example.com\nPERSON: John Smith"


def test_detect_pii_returns_empty_string_without_pii(analyzer):
    assert anonymizer.detect_pii("Nothing to see here.") == ""


def test_detect_pii_resolves_overlaps_like_redact_text(monkeypatch):
    fake = FakeAnalyzer(("PERSON", r"John Smith", 0.85), ("LOCATION", r"Smith", 0.4))
    monkeypatch.setattr(anonymizer, "_get_engines", lambda: (fake, AnonymizerEngine()))
    text = "I am John Smith."

    assert anonymizer.detect_pii(text) == "PERSON: John Smith"
    assert anonymizer.redact_text(text) == "I am <PERSON>."


def test_redact_text_replaces_entities(analyzer):
    text = "I am John Smith, write to john.smith@example.com."
    assert anonymizer.redact_text(text) == "I am <PERSON>, write to <EMAIL_ADDRESS>."


def test_redact_text_across_chunks(analyzer, monkeypatch):
    monkeypatch.setattr(anonymizer, "CHUNK_SIZE", 30)
    text = "First line with John Smith.\nSecond: john.smith@example.com\nThird."
    progress = []

    assert anonymizer.redact_text(text, on_progress=progress.append) == (
        "First line with <PERSON>.\nSecond: <EMAIL_ADDRESS>\nThird."
    )
    assert len(analyzer.calls) == 3
    assert progress[-1] == 100


@pytest.mark.spacy
@pytest.mark.skipif(importlib.util.find_spec("en_core_web_lg") is None, reason="en_core_web_lg is not installed")
class TestWithSpacyModel:
    TEXT = "My name is John Smith and my email address is john.smith@example.com."

    @pytest.fixture(autouse=True)
    def spacy_engine(self, monkeypatch):
        """GLiNER is the default engine here: the spaCy model is asked for explicitly and built anew."""
        monkeypatch.setenv(anonymizer.ENGINE_ENV_VAR, anonymizer.ENGINE_SPACY)
        monkeypatch.setattr(anonymizer, "_ENGINES", None)

    def test_detect_pii(self):
        lines = anonymizer.detect_pii(self.TEXT).splitlines()
        assert "PERSON: John Smith" in lines
        assert "EMAIL_ADDRESS: john.smith@example.com" in lines

    def test_redact_text(self):
        redacted = anonymizer.redact_text(self.TEXT)
        assert "<PERSON>" in redacted
        assert "<EMAIL_ADDRESS>" in redacted
        assert "John Smith" not in redacted
        assert "john.smith@example.com" not in redacted
