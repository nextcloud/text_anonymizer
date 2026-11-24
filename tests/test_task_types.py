# SPDX-FileCopyrightText: 2026 Nextcloud GmbH and Nextcloud contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the TaskProcessing task types and their output formats."""

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


def test_redact_handler_without_mode_is_unchanged(set_engines) -> None:
    set_engines(TEXT_RESULTS)
    output = main.TASK_HANDLERS["text_anonymizer:redact_pii"]({"input": TEXT})
    assert output == {"output": anonymizer.redact_text(TEXT)}
    assert output["output"] == "<PERSON> met <PERSON> at <EMAIL_ADDRESS>."


def test_detect_handler_reports_output_slot(set_engines) -> None:
    set_engines(TEXT_RESULTS)
    output = main.TASK_HANDLERS["text_anonymizer:detect_pii"]({"input": TEXT})
    assert output == {"output": anonymizer.detect_pii(TEXT)}


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
