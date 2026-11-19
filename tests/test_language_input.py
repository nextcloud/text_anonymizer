# SPDX-FileCopyrightText: 2026 Nextcloud GmbH and Nextcloud contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the language input of the TaskProcessing tasks."""

import sys
from pathlib import Path

from presidio_analyzer import RecognizerResult
from presidio_anonymizer import AnonymizerEngine

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ex_app" / "lib"))

import anonymizer  # noqa: E402
import main  # noqa: E402


class RecordingAnalyzer:
    """AnalyzerEngine stand-in recording the language of each analyze call, so the language
    forwarded by the task handlers is observable without loading any model."""

    supported_languages = ["en", "de"]

    def __init__(self) -> None:
        self.languages: list[str] = []

    def analyze(self, text: str, language: str) -> list[RecognizerResult]:
        self.languages.append(language)
        return []


def _recording_analyzer(monkeypatch) -> RecordingAnalyzer:
    analyzer = RecordingAnalyzer()
    monkeypatch.setattr(anonymizer, "_get_engines", lambda: (analyzer, AnonymizerEngine()))
    return analyzer


def test_detect_task_forwards_the_language_input(monkeypatch) -> None:
    analyzer = _recording_analyzer(monkeypatch)
    main.TASK_HANDLERS["text_anonymizer:detect_pii"]({"input": "Anna Müller wohnt in Berlin.", "language": "de"})
    assert analyzer.languages == ["de"]


def test_detect_task_defaults_to_english_without_a_language(monkeypatch) -> None:
    analyzer = _recording_analyzer(monkeypatch)
    main.TASK_HANDLERS["text_anonymizer:detect_pii"]({"input": "Nothing to see."})
    assert analyzer.languages == ["en"]


def test_redact_task_forwards_the_language_input(monkeypatch) -> None:
    analyzer = _recording_analyzer(monkeypatch)
    output = main.TASK_HANDLERS["text_anonymizer:redact_pii"](
        {"input": "Anna Müller wohnt in Berlin.", "language": "de"}
    )
    assert analyzer.languages == ["de"]
    assert output == {"output": "Anna Müller wohnt in Berlin."}


def test_unsupported_task_language_falls_back_to_english(monkeypatch) -> None:
    analyzer = _recording_analyzer(monkeypatch)
    monkeypatch.setattr(analyzer, "supported_languages", ["en"])
    main.TASK_HANDLERS["text_anonymizer:detect_pii"]({"input": "Anna Müller wohnt in Berlin.", "language": "de"})
    assert analyzer.languages == ["en"]
