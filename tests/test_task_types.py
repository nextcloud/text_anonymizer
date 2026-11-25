# SPDX-FileCopyrightText: 2026 Nextcloud GmbH and Nextcloud contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the TaskProcessing task types and their output formats."""

import asyncio
from types import SimpleNamespace

import pytest
from presidio_analyzer import RecognizerResult
from presidio_anonymizer import AnonymizerEngine

import anonymizer
import main
from nc_py_api.ex_app.providers.task_processing import ShapeType


class FakeAnalyzer:
    """AnalyzerEngine stand-in returning canned results, so no spaCy model is loaded."""

    supported_languages = ["en"]

    def __init__(self, results: list[RecognizerResult]):
        self.results = results

    def analyze(self, text: str, language: str) -> list[RecognizerResult]:
        return self.results


class ChunkAnalyzer:
    """AnalyzerEngine stand-in that finds ``name`` as a PERSON in every chunk holding it, and counts its calls."""

    supported_languages = ["en"]

    def __init__(self, name: str = "Alice") -> None:
        self.name = name
        self.chunks: list[str] = []

    def analyze(self, text: str, language: str) -> list[RecognizerResult]:
        self.chunks.append(text)
        return [entity("PERSON", self.name, text)] if self.name in text else []


@pytest.fixture
def set_engines(monkeypatch: pytest.MonkeyPatch):
    """Patch ``anonymizer._get_engines`` to return a fake analyzer plus a real anonymizer engine."""

    def set_results(results: list[RecognizerResult]) -> None:
        monkeypatch.setattr(anonymizer, "_get_engines", lambda: (FakeAnalyzer(results), AnonymizerEngine()))

    return set_results


def entity(entity_type: str, text: str, whole: str, score: float = 0.9) -> RecognizerResult:
    start = whole.index(text)
    return RecognizerResult(entity_type=entity_type, start=start, end=start + len(text), score=score)


TEXT = "Alice met Bob at bob@example.com."
TEXT_RESULTS = [
    entity("PERSON", "Alice", TEXT),
    entity("PERSON", "Bob", TEXT),
    entity("EMAIL_ADDRESS", "bob@example.com", TEXT),
]


def test_contains_pii_returns_true_for_pii_text(set_engines) -> None:
    set_engines(TEXT_RESULTS)
    assert anonymizer.contains_pii(TEXT) is True
    assert main.TASK_HANDLERS["text_anonymizer:contains_pii"]({"input": TEXT}) == {"contains_pii": "true"}


def test_contains_pii_returns_false_for_clean_text(set_engines) -> None:
    set_engines([])
    assert anonymizer.contains_pii("Just a perfectly ordinary sentence.") is False
    assert main.TASK_HANDLERS["text_anonymizer:contains_pii"]({"input": "Nothing to see."}) == {"contains_pii": "false"}


def test_contains_pii_does_not_count_dates_or_urls(set_engines) -> None:
    cases = [
        ("The weather is fine today.", "DATE_TIME", "today"),
        ("The meeting is on Monday.", "DATE_TIME", "Monday"),
        ("See https://example.com/report for details.", "URL", "https://example.com/report"),
    ]
    for whole, entity_type, value in cases:
        set_engines([entity(entity_type, value, whole)])
        assert anonymizer.contains_pii(whole) is False, whole
        assert main.TASK_HANDLERS["text_anonymizer:contains_pii"]({"input": whole}) == {"contains_pii": "false"}


def test_contains_pii_counts_locations(set_engines) -> None:
    set_engines([entity("LOCATION", "Springfield", "Our office is in Springfield.")])
    assert anonymizer.contains_pii("Our office is in Springfield.") is True
    assert main.TASK_HANDLERS["text_anonymizer:contains_pii"]({"input": "Our office is in Springfield."}) == {
        "contains_pii": "true"
    }


def test_contains_pii_resolves_overlaps_like_the_other_tasks(set_engines) -> None:
    cases = [
        # The date wins the overlap, so the other tasks report nothing identifying.
        ("June was a busy month.", [("DATE_TIME", "June", 0.9), ("PERSON", "June", 0.5)], False),
        ("Geburtsdatum: 03.04.1985", [("DATE_TIME", "03.04.1985", 0.6), ("PHONE_NUMBER", "03.04.1985", 0.4)], False),
        # The identifying entity wins the overlap.
        ("June was a busy month.", [("PERSON", "June", 0.9), ("DATE_TIME", "June", 0.5)], True),
    ]
    for whole, results, expected in cases:
        set_engines([entity(entity_type, value, whole, score) for entity_type, value, score in results])
        reported = [line for line in anonymizer.detect_pii(whole).splitlines() if not line.startswith("DATE_TIME:")]
        assert bool(reported) is expected, whole
        assert anonymizer.contains_pii(whole) is expected, whole
        assert main.TASK_HANDLERS["text_anonymizer:contains_pii"]({"input": whole}) == {
            "contains_pii": "true" if expected else "false"
        }, whole


def test_contains_pii_stops_at_the_first_chunk_with_pii(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(anonymizer, "CHUNK_SIZE", 50)
    lines = [f"Line {i} of an ordinary text.\n" for i in range(4)]
    cases = [
        # The line holding the name and the progress reported until the answer is known.
        (0, [25.0]),
        (2, [25.0, 50.0, 75.0]),
        (None, [25.0, 50.0, 75.0, 100.0]),
    ]
    for line_number, progress in cases:
        chunked = list(lines)
        if line_number is not None:
            chunked[line_number] = "Alice can be reached at bob@example.com.\n"
        text = "".join(chunked)
        analyzer = ChunkAnalyzer()
        monkeypatch.setattr(anonymizer, "_get_engines", lambda: (analyzer, AnonymizerEngine()))
        reported = []
        assert anonymizer.contains_pii(text, on_progress=reported.append) is (line_number is not None), line_number
        assert reported == progress, line_number
        # Every chunk ends at a line break, so the chunk holding the name is the last one analyzed.
        analyzed = text if line_number is None else "".join(chunked[: line_number + 1])
        assert "".join(analyzer.chunks) == analyzed, line_number


def test_detect_pii_and_redact_text_reach_every_chunk(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(anonymizer, "CHUNK_SIZE", 50)
    analyzer = ChunkAnalyzer()
    monkeypatch.setattr(anonymizer, "_get_engines", lambda: (analyzer, AnonymizerEngine()))
    text = "".join(f"Line {i} of an ordinary text.\n" for i in range(4))
    text = text[:56] + "Alice can be reached at bob@example.com.\n" + text[84:]
    assert anonymizer.detect_pii(text) == "PERSON: Alice"
    assert len(analyzer.chunks) == 4
    analyzer.chunks.clear()
    assert anonymizer.redact_text(text) == text.replace("Alice", "<PERSON>")
    assert len(analyzer.chunks) == 4


def test_redact_handler_without_mode_is_unchanged(set_engines) -> None:
    set_engines(TEXT_RESULTS)
    output = main.TASK_HANDLERS["text_anonymizer:redact_pii"]({"input": TEXT})
    assert output == {"output": anonymizer.redact_text(TEXT)}
    assert output["output"] == "<PERSON> met <PERSON> at <EMAIL_ADDRESS>."


def test_detect_handler_reports_output_slot(set_engines) -> None:
    set_engines(TEXT_RESULTS)
    output = main.TASK_HANDLERS["text_anonymizer:detect_pii"]({"input": TEXT})
    assert output == {"output": anonymizer.detect_pii(TEXT)}


class FakeTaskProcessing:
    def __init__(self) -> None:
        self.results: list[dict] = []

    async def set_progress(self, task_id: int, progress: float) -> None:
        pass

    async def report_result(self, task_id: int, output=None, error_message=None) -> None:
        self.results.append({"task_id": task_id, "output": output, "error_message": error_message})


class FakeNc:
    def __init__(self) -> None:
        self.providers = SimpleNamespace(task_processing=FakeTaskProcessing())

    async def log(self, level: int, message: str) -> None:
        pass


def test_process_task_reports_handler_output_unwrapped(set_engines) -> None:
    set_engines([])
    nc = FakeNc()
    task = {"id": 42, "type": "text_anonymizer:contains_pii", "input": {"input": "Nothing to see."}}
    asyncio.run(main.process_task(nc, task))
    assert nc.providers.task_processing.results == [
        {"task_id": 42, "output": {"contains_pii": "false"}, "error_message": None}
    ]


def test_task_types_are_registered() -> None:
    assert {task_type.id for _, task_type in main.TASK_PROCESSING_PROVIDERS} == {
        "text_anonymizer:detect_pii",
        "text_anonymizer:redact_pii",
        "text_anonymizer:contains_pii",
    }
    assert set(main.TASK_HANDLERS) == {task_type.id for _, task_type in main.TASK_PROCESSING_PROVIDERS}
    for provider, task_type in main.TASK_PROCESSING_PROVIDERS:
        assert provider.id == task_type.id
        assert provider.task_type == task_type.id


def test_contains_pii_task_shape() -> None:
    assert [(d.name, d.shape_type) for d in main.CONTAINS_PII_TASK_TYPE.input_shape] == [("input", ShapeType.TEXT)]
    assert [(d.name, d.shape_type) for d in main.CONTAINS_PII_TASK_TYPE.output_shape] == [
        ("contains_pii", ShapeType.TEXT)
    ]


def test_contains_pii_task_name_matches_the_other_tasks() -> None:
    names = {
        main.DETECT_PII_TASK_TYPE.name,
        main.REDACT_PII_TASK_TYPE.name,
        main.CONTAINS_PII_TASK_TYPE.name,
    }
    assert names == {"Detect sensitive text", "Redact sensitive text", "Contains sensitive text?"}


def test_contains_pii_task_type_says_true_false() -> None:
    assert "yes/no" not in main.CONTAINS_PII_TASK_TYPE.description
    assert "true/false" in main.CONTAINS_PII_TASK_TYPE.description
    assert "yes/no" not in main.CONTAINS_PII_TASK_TYPE.output_shape[0].description
    assert '"true" or "false"' in main.CONTAINS_PII_TASK_TYPE.output_shape[0].description
