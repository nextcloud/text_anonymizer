# SPDX-FileCopyrightText: 2026 Nextcloud GmbH and Nextcloud contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# ruff: noqa: S101

"""Tests for the detection engine: GLiNER setup, spaCy fallback, and language routing.

The NLP engine and the GLiNER model are replaced by fakes so that no spaCy model is loaded
and no model is downloaded. Tests that need real models skip themselves when those are not
installed, and they pin the engine to spaCy so that no network access happens either.
"""

import logging

import anonymizer
import pytest
from presidio_analyzer import RecognizerResult
from spacy.util import is_package

ENGLISH_TEXT = "John Smith called from +1-415-555-0192."
# A Steuer-IdNr. that passes the recognizer's ISO 7064 checksum.
VALID_TAX_ID = "26954327001"
GERMAN_TEXT = f"Anna Müller zahlt mit der Steuer-IdNr. {VALID_TAX_ID} ihre Steuern."

ENGLISH_MODEL_INSTALLED = is_package(anonymizer.LANGUAGE_MODELS["en"])
GERMAN_MODEL_INSTALLED = is_package(anonymizer.LANGUAGE_MODELS["de"])


class FakeNlpEngine:
    """Stand-in for the spaCy engine, so that no model is loaded for the tests."""

    def __init__(self, languages):
        self.languages = list(languages)

    def is_loaded(self):
        return True

    def load(self):
        return None

    def get_supported_languages(self):
        return list(self.languages)

    def get_supported_entities(self):
        return ["DATE_TIME", "LOCATION", "NRP", "ORGANIZATION", "PERSON"]

    def process_text(self, text, language):
        return None


class FakeNlpEngineProvider:
    """Stand-in for NlpEngineProvider that records the configuration it was given."""

    last_configuration = None

    def __init__(self, nlp_configuration=None):
        self.nlp_configuration = nlp_configuration

    def create_engine(self):
        FakeNlpEngineProvider.last_configuration = self.nlp_configuration
        return FakeNlpEngine([model["lang_code"] for model in self.nlp_configuration["models"]])


def make_fake_gliner(spans_by_language, load_error=None):
    """Build a GLiNERRecognizer stand-in that returns canned spans per language, with no model."""
    from presidio_analyzer import EntityRecognizer

    class FakeGlinerRecognizer(EntityRecognizer):
        def __init__(self, **kwargs):
            super().__init__(
                supported_entities=sorted(
                    {entity_type for spans in spans_by_language.values() for entity_type, _, _ in spans}
                ),
                name="GLiNERRecognizer",
                supported_language=kwargs.get("supported_language", "en"),
                version="0.0.1",
            )
            self.__dict__.update(kwargs)
            self.is_loaded = False
            self.gliner = None

        def load(self):
            if load_error is not None:
                raise load_error
            # Stands in for the loaded model instance shared between languages.
            self.gliner = object()
            self.is_loaded = True

        def analyze(self, text, entities=None, nlp_artifacts=None):
            results = []
            for entity_type, value, score in spans_by_language.get(self.supported_language, []):
                if value not in text or (entities and entity_type not in entities):
                    continue
                start = text.index(value)
                results.append(
                    RecognizerResult(entity_type=entity_type, start=start, end=start + len(value), score=score)
                )
            return results

    return FakeGlinerRecognizer


def build_engines(monkeypatch, *, german_model=False, gliner=None):
    """Build the engines through the production path, with faked model loading.

    The returned analyzer is also cached as the module engines, so that detect_pii and
    redact_text use it directly.
    """
    monkeypatch.setattr(anonymizer, "NlpEngineProvider", FakeNlpEngineProvider)
    monkeypatch.setattr(
        anonymizer, "is_package", lambda model: german_model and model == anonymizer.LANGUAGE_MODELS["de"]
    )
    if gliner is not None:
        monkeypatch.setattr(anonymizer, "GLiNERRecognizer", gliner)
    engines = anonymizer._get_engines()
    assert engines is anonymizer._ENGINES
    return engines


@pytest.fixture(autouse=True)
def isolated_engines(monkeypatch):
    """Reset the engine cache and the engine selection around every test."""
    anonymizer._ENGINES = None
    FakeNlpEngineProvider.last_configuration = None
    monkeypatch.delenv(anonymizer.ENGINE_ENV_VAR, raising=False)
    yield
    anonymizer._ENGINES = None


def recognizer_names(analyzer, language=None):
    recognizers = (
        analyzer.registry.recognizers
        if language is None
        else analyzer.registry.get_recognizers(language=language, all_fields=True)
    )
    return [recognizer.name for recognizer in recognizers]


def test_gliner_engine_loads_and_detects_english_pii(monkeypatch):
    """GLiNER is the default engine, replaces the spaCy NER recognizer, and its hits are redacted."""
    fake_gliner = make_fake_gliner({"en": [("PERSON", "John Smith", 0.9)]})
    analyzer, _ = build_engines(monkeypatch, gliner=fake_gliner)

    assert recognizer_names(analyzer).count("GLiNERRecognizer") == 1
    assert "SpacyRecognizer" not in recognizer_names(analyzer)
    assert "PERSON: John Smith" in anonymizer.detect_pii(ENGLISH_TEXT)
    redacted = anonymizer.redact_text(ENGLISH_TEXT)
    assert "<PERSON>" in redacted
    assert "John Smith" not in redacted


def test_gliner_model_is_loaded_once_for_all_languages(monkeypatch):
    """The model is loaded once: every further language is a copy of the first recognizer.

    EntityRecognizer's constructor loads the model of each instance, so this runs the real
    GLiNERRecognizer and only replaces the GLiNER class its load() calls.
    """
    from presidio_analyzer.predefined_recognizers.ner import gliner_recognizer

    loaded = []

    class CountingGLiNER:
        @staticmethod
        def from_pretrained(model_name, **kwargs):
            model = object()
            loaded.append(model)
            return model

    monkeypatch.setattr(gliner_recognizer, "GLiNER", CountingGLiNER)
    analyzer, _ = build_engines(monkeypatch, german_model=True)

    assert len(loaded) == 1
    gliner_recognizers = [r for r in analyzer.registry.recognizers if r.name == "GLiNERRecognizer"]
    assert {r.supported_language for r in gliner_recognizers} == {"en", "de"}
    # The copy is a working recognizer of its own language that shares the loaded model.
    assert all(r.is_loaded for r in gliner_recognizers)
    assert all(r.gliner is loaded[0] for r in gliner_recognizers)
    assert [r.supported_language for r in analyzer.registry.get_recognizers("de", all_fields=True) if r.name == "GLiNERRecognizer"] == ["de"]


def test_gliner_load_failure_falls_back_to_spacy(monkeypatch, caplog):
    """A GLiNER load failure keeps the spaCy NER recognizer and logs a warning."""
    fake_gliner = make_fake_gliner({}, load_error=ImportError("the gliner package is not installed"))
    with caplog.at_level(logging.WARNING, logger="anonymizer"):
        analyzer, _ = build_engines(monkeypatch, gliner=fake_gliner)

    names = recognizer_names(analyzer)
    assert names.count("GLiNERRecognizer") == 0
    assert "SpacyRecognizer" in names
    assert "spaCy detection engine" in caplog.text


# A number that passes the Luhn checksum, as the checksums on real cards do.
VALID_CARD = "4111 1111 1111 1111"
GERMAN_CARD_TEXT = f"Kartennummer: {VALID_CARD}."


def test_german_card_number_is_detected(monkeypatch):
    """German has its own credit card recognizer, with German context words.

    Presidio's default configuration registers CreditCardRecognizer for en, es, it and pl only.
    """
    monkeypatch.setenv(anonymizer.ENGINE_ENV_VAR, anonymizer.ENGINE_SPACY)
    analyzer, _ = build_engines(monkeypatch, german_model=True)

    card = next(
        r for r in analyzer.registry.get_recognizers("de", all_fields=True) if r.name == "CreditCardRecognizer"
    )
    assert card.supported_language == "de"
    assert "kreditkarte" in card.context
    assert f"CREDIT_CARD: {VALID_CARD}" in anonymizer.detect_pii(GERMAN_CARD_TEXT, language="de")
    redacted = anonymizer.redact_text(GERMAN_CARD_TEXT, language="de")
    assert VALID_CARD not in redacted
    assert "<CREDIT_CARD>" in redacted


def test_german_card_number_failing_the_checksum_is_not_detected(monkeypatch):
    """The German recognizer keeps the checksum: a number that fails it is not a card."""
    monkeypatch.setenv(anonymizer.ENGINE_ENV_VAR, anonymizer.ENGINE_SPACY)
    build_engines(monkeypatch, german_model=True)

    detected = anonymizer.detect_pii(GERMAN_CARD_TEXT.replace("1111.", "1112."), language="de")
    assert "CREDIT_CARD" not in detected


def test_gliner_model_download_failure_falls_back_to_spacy(monkeypatch):
    """A model download failure (e.g. no network) is also caught and falls back to spaCy."""
    fake_gliner = make_fake_gliner({}, load_error=OSError("model download failed"))
    analyzer, _ = build_engines(monkeypatch, gliner=fake_gliner)

    names = recognizer_names(analyzer)
    assert names.count("GLiNERRecognizer") == 0
    assert "SpacyRecognizer" in names


def test_engine_env_var_selects_spacy(monkeypatch):
    """TEXT_ANONYMIZER_ENGINE=spacy keeps spaCy NER and never touches GLiNER."""
    monkeypatch.setenv(anonymizer.ENGINE_ENV_VAR, anonymizer.ENGINE_SPACY)
    called = []
    fake_gliner = make_fake_gliner({})
    fake_gliner_load = fake_gliner.load

    def spy_load(self):
        called.append(self.supported_language)
        return fake_gliner_load(self)

    fake_gliner.load = spy_load
    analyzer, _ = build_engines(monkeypatch, gliner=fake_gliner)

    names = recognizer_names(analyzer)
    assert "SpacyRecognizer" in names
    assert "GLiNERRecognizer" not in names
    assert called == []


def test_unknown_engine_value_defaults_to_gliner(monkeypatch):
    """An unknown engine value is rejected with a warning and GLiNER stays the default."""
    monkeypatch.setenv(anonymizer.ENGINE_ENV_VAR, "transformers")
    fake_gliner = make_fake_gliner({"en": [("PERSON", "John Smith", 0.9)]})
    analyzer, _ = build_engines(monkeypatch, gliner=fake_gliner)

    assert recognizer_names(analyzer).count("GLiNERRecognizer") == 1


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, anonymizer.ENGINE_GLINER),
        ("", anonymizer.ENGINE_GLINER),
        ("gliner", anonymizer.ENGINE_GLINER),
        ("Spacy", anonymizer.ENGINE_SPACY),
        (" GLINER ", anonymizer.ENGINE_GLINER),
        ("nonsense", anonymizer.ENGINE_GLINER),
    ],
)
def test_requested_engine_parsing(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv(anonymizer.ENGINE_ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(anonymizer.ENGINE_ENV_VAR, value)
    assert anonymizer._requested_engine() == expected


def test_german_pack_registered_when_model_installed(monkeypatch):
    """German support adds the DE_* recognizers and one GLiNER instance per language."""
    fake_gliner = make_fake_gliner({"en": [], "de": []})
    analyzer, _ = build_engines(monkeypatch, german_model=True, gliner=fake_gliner)

    de_entities = set()
    for recognizer in analyzer.registry.recognizers:
        if recognizer.supported_language == "de" and hasattr(recognizer, "supported_entities"):
            de_entities |= set(recognizer.supported_entities)

    assert "DE_TAX_ID" in de_entities
    assert "DE_HEALTH_INSURANCE" in de_entities
    assert "DE_SOCIAL_SECURITY" in de_entities
    # German postal codes are excluded on purpose: they match any five digits.
    assert "DE_PLZ" not in de_entities

    gliner_recognizers = [r for r in analyzer.registry.recognizers if r.name == "GLiNERRecognizer"]
    assert {r.supported_language for r in gliner_recognizers} == {"en", "de"}
    # One model copy is shared between the language instances.
    assert gliner_recognizers[0].gliner is gliner_recognizers[1].gliner


def test_german_pack_absent_when_model_missing(monkeypatch, caplog):
    """Without the German spaCy model, no German pipeline is built and a warning is logged."""
    analyzer, _ = build_engines(monkeypatch)
    assert all(recognizer.supported_language != "de" for recognizer in analyzer.registry.recognizers)
    assert "analyzed with the English pipeline" in caplog.text


def test_nlp_configuration_covers_installed_languages(monkeypatch):
    """The NLP engine gets one model entry per supported language."""
    build_engines(monkeypatch, german_model=True)
    models = FakeNlpEngineProvider.last_configuration["models"]
    assert models == [
        {"lang_code": "en", "model_name": anonymizer.LANGUAGE_MODELS["en"]},
        {"lang_code": "de", "model_name": anonymizer.LANGUAGE_MODELS["de"]},
    ]


def test_german_text_detects_german_pii(monkeypatch):
    """German analysis runs the DE_* recognizers: a valid Steuer-IdNr. is found and redacted."""
    fake_gliner = make_fake_gliner({"de": [("PERSON", "Anna Müller", 0.85)]})
    build_engines(monkeypatch, german_model=True, gliner=fake_gliner)

    detected = anonymizer.detect_pii(GERMAN_TEXT, language="de")
    assert f"DE_TAX_ID: {VALID_TAX_ID}" in detected
    assert "PERSON: Anna Müller" in detected
    redacted = anonymizer.redact_text(GERMAN_TEXT, language="de")
    assert VALID_TAX_ID not in redacted
    assert "<DE_TAX_ID>" in redacted


def test_language_routes_to_language_specific_recognizers(monkeypatch):
    """The language picks the recognizer set: DE_* run for de, not for en."""
    fake_gliner = make_fake_gliner({"en": [], "de": []})
    analyzer, _ = build_engines(monkeypatch, german_model=True, gliner=fake_gliner)

    assert "DeTaxIdRecognizer" in recognizer_names(analyzer, language="de")
    assert "DeTaxIdRecognizer" not in recognizer_names(analyzer, language="en")


@pytest.mark.parametrize(
    ("language", "supported", "expected"),
    [("de", ["en", "de"], "de"), ("en", ["en", "de"], "en"), ("de", ["en"], "en"), ("fr", ["en", "de"], "en")],
)
def test_resolve_language(language, supported, expected):
    assert anonymizer._resolve_language(language, supported) == expected


def test_unsupported_language_falls_back_to_english(monkeypatch, caplog):
    """A language without an installed model is analyzed as English, with a warning."""
    fake_gliner = make_fake_gliner({"en": [("PERSON", "John Smith", 0.9)]})
    build_engines(monkeypatch, gliner=fake_gliner)

    with caplog.at_level(logging.WARNING, logger="anonymizer"):
        anonymizer.redact_text("Anna Müller wohnt in Berlin.", language="de")
    assert "analyzing the text as English" in caplog.text


@pytest.mark.skipif(not ENGLISH_MODEL_INSTALLED, reason="the English spaCy model is not installed")
def test_spacy_engine_detects_english_pii(monkeypatch):
    """The spaCy engine detects English PII end to end, without GLiNER and without network."""
    monkeypatch.setenv(anonymizer.ENGINE_ENV_VAR, anonymizer.ENGINE_SPACY)
    detected = anonymizer.detect_pii(ENGLISH_TEXT)
    assert "PERSON: John Smith" in detected


@pytest.mark.skipif(
    not (ENGLISH_MODEL_INSTALLED and GERMAN_MODEL_INSTALLED),
    reason="the English and German spaCy models are not installed",
)
def test_spacy_engine_detects_german_pii(monkeypatch):
    """The spaCy fallback engine detects German PII end to end when German is requested."""
    monkeypatch.setenv(anonymizer.ENGINE_ENV_VAR, anonymizer.ENGINE_SPACY)
    detected = anonymizer.detect_pii(GERMAN_TEXT, language="de")
    assert f"DE_TAX_ID: {VALID_TAX_ID}" in detected
    assert "PERSON: Anna Müller" in detected


@pytest.mark.skipif(not ENGLISH_MODEL_INSTALLED, reason="the English spaCy model is not installed")
def test_gliner_failure_falls_back_to_working_spacy_detection(monkeypatch):
    """After a GLiNER load failure, detection still works through the real spaCy pipeline."""
    fake_gliner = make_fake_gliner({}, load_error=ImportError("the gliner package is not installed"))
    monkeypatch.setattr(anonymizer, "GLiNERRecognizer", fake_gliner)

    detected = anonymizer.detect_pii(ENGLISH_TEXT)
    assert "PERSON: John Smith" in detected
