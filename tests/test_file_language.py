# SPDX-FileCopyrightText: 2026 Nextcloud GmbH and Nextcloud contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the language of a file in the Files action."""

from types import SimpleNamespace

import anonymizer
import main
import pytest
from presidio_analyzer import RecognizerResult
from presidio_anonymizer import AnonymizerEngine
from spacy.util import is_package

GERMAN = "Bitte prüfen Sie den beigefügten Vertrag vor unserem Termin am Montag."
ENGLISH = "Please review the attached contract before our meeting on Monday."
# An English letter with a German identity block: no detector calls this text German.
LETTER = (
    "Dear Mr Smith,\n\nThank you for your inquiry. We will get back to you next week.\n\n"
    "Best regards,\nAnna Müller\n"
    "Personalausweis: L01X00T44\n"
    "Rentenversicherungsnummer: 15070649C103\n"
)
GERMAN_IDENTIFIERS = ("L01X00T44", "15070649C103")


def test_language_of_a_text_is_detected_deterministically():
    """The detector runs without a download and gives the same answer for the same text."""
    assert anonymizer.detect_language(GERMAN, ["en", "de"]) == "de"
    assert anonymizer.detect_language(ENGLISH, ["en", "de"]) == "en"
    assert anonymizer.detect_language(GERMAN, ["en", "de"]) == "de"


@pytest.mark.parametrize("text", ["Merci beaucoup pour votre commande.", "42", "L01X00T44 15070649C103"])
def test_a_text_in_no_supported_language_has_no_language(text):
    """French, digits and identifiers are not assigned to a language the app supports."""
    assert anonymizer.detect_language(text, ["en", "de"]) is None


def test_german_text_without_the_german_model_has_no_language():
    """Without the German pipeline a text can only be English."""
    assert anonymizer.detect_language(GERMAN, ["en"]) is None


class IdentifierAnalyzer:
    """AnalyzerEngine stand-in: the entities of the text's language and, when asked, of another."""

    supported_languages = ["en", "de"]
    entities = {"en": {"PERSON"}, "de": {"PERSON", "DE_PASSPORT", "DE_SOCIAL_SECURITY"}}
    found = {"L01X00T44": "DE_PASSPORT", "15070649C103": "DE_SOCIAL_SECURITY", "Anna Müller": "PERSON"}

    def __init__(self):
        self.calls = []

    def get_supported_entities(self, language=None):
        return sorted(self.entities[language])

    def analyze(self, text, language, entities=None):
        self.calls.append((language, entities))
        return [
            RecognizerResult(kind, text.index(value), text.index(value) + len(value), 1.0)
            for value, kind in self.found.items()
            if value in text and kind in self.entities[language] and (entities is None or kind in entities)
        ]


@pytest.fixture
def analyzer(monkeypatch):
    fake = IdentifierAnalyzer()
    monkeypatch.setattr(anonymizer, "_get_engines", lambda: (fake, AnonymizerEngine()))
    return fake


def test_identifiers_of_another_language_are_redacted_in_a_file(analyzer):
    """An English letter with a German identity block: both identifiers go, each with its type."""
    redacted = anonymizer.redact_text(LETTER, language=None, foreign_identifiers=True)

    for identifier in GERMAN_IDENTIFIERS:
        assert identifier not in redacted
    assert "<DE_PASSPORT>" in redacted
    assert "<DE_SOCIAL_SECURITY>" in redacted
    assert "<PERSON>" in redacted
    assert analyzer.calls == [("en", None), ("de", ["DE_PASSPORT", "DE_SOCIAL_SECURITY"])]


def test_without_the_identifiers_of_other_languages_the_letter_keeps_them(analyzer):
    """A text analyzed as English alone, which is what the Files action did before."""
    redacted = anonymizer.redact_text(LETTER, language=None)

    for identifier in GERMAN_IDENTIFIERS:
        assert identifier in redacted
    assert analyzer.calls == [("en", None)]


def test_an_identifier_the_text_language_found_keeps_its_type(monkeypatch):
    """A number that German typed as a tax id is not re-typed as a US bank number."""
    tax_id = "26954327001"

    class OverlapAnalyzer(IdentifierAnalyzer):
        entities = {"en": {"PERSON", "US_BANK_NUMBER"}, "de": {"PERSON", "DE_TAX_ID"}}
        found = {tax_id: "US_BANK_NUMBER"}

        def analyze(self, text, language, entities=None):
            kind = "DE_TAX_ID" if language == "de" else "US_BANK_NUMBER"
            return [RecognizerResult(kind, text.index(tax_id), text.index(tax_id) + len(tax_id), 1.0)]

    monkeypatch.setattr(anonymizer, "_get_engines", lambda: (OverlapAnalyzer(), AnonymizerEngine()))

    redacted = anonymizer.redact_text(f"Steuer-IdNr. {tax_id}.", language="de", foreign_identifiers=True)

    assert "<DE_TAX_ID>" in redacted
    assert "US_BANK_NUMBER" not in redacted


def test_an_identifier_scored_higher_than_what_the_text_language_found_wins(monkeypatch):
    """Asked for every English entity name, the model may type a German ID as a UUID.

    The identifier that passed its checksum has the higher score and the better type.
    """
    identifier = "L01X00T44"

    class GuessingAnalyzer(IdentifierAnalyzer):
        entities = {"en": {"PERSON", "UUID"}, "de": {"PERSON", "DE_PASSPORT"}}

        def analyze(self, text, language, entities=None):
            if identifier not in text:
                return []
            kind, score = ("UUID", 0.5) if language == "en" else ("DE_PASSPORT", 1.0)
            return [RecognizerResult(kind, text.index(identifier), text.index(identifier) + len(identifier), score)]

    monkeypatch.setattr(anonymizer, "_get_engines", lambda: (GuessingAnalyzer(), AnonymizerEngine()))

    redacted = anonymizer.redact_text(f"Personalausweis: {identifier}", language="en", foreign_identifiers=True)

    assert "<DE_PASSPORT>" in redacted
    assert "UUID" not in redacted


def test_files_action_lets_the_text_decide_the_language(monkeypatch):
    """The Files action passes no language and asks for the identifiers of every language."""
    calls, uploaded = [], []
    monkeypatch.setattr(main, "redact_text", lambda text, **kwargs: calls.append(kwargs) or "redacted")
    node = SimpleNamespace(name="letter.txt", user_path="/letter.txt", is_dir=False)
    nc = SimpleNamespace(
        files=SimpleNamespace(
            by_id=lambda file_id: node,
            download=lambda node: LETTER.encode(),
            upload=lambda path, text: uploaded.append((path, text)),
        ),
        notifications=SimpleNamespace(create=lambda *args: None),
    )
    monkeypatch.setattr(main, "path_exists", lambda nc, path: False)

    main.redact_file(nc, SimpleNamespace(fileId=7, name="letter.txt"))

    assert calls == [{"language": None, "foreign_identifiers": True}]
    assert uploaded == [("/letter (redacted).txt", "redacted")]


@pytest.mark.skipif(
    not all(is_package(model) for model in anonymizer.LANGUAGE_MODELS.values()),
    reason="the English and German spaCy models are not installed",
)
def test_german_identifiers_in_an_english_letter_are_redacted(monkeypatch):
    """End to end with the spaCy engine: neither identifier stays in the redacted letter."""
    monkeypatch.setenv(anonymizer.ENGINE_ENV_VAR, anonymizer.ENGINE_SPACY)

    redacted = anonymizer.redact_text(LETTER, language=None, foreign_identifiers=True)

    for identifier in GERMAN_IDENTIFIERS:
        assert identifier not in redacted
