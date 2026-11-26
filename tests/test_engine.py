# SPDX-FileCopyrightText: 2026 Nextcloud GmbH and Nextcloud contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
# ruff: noqa: S101

"""Tests for the detection engine: GLiNER setup, spaCy fallback, and language routing.

The NLP engine and the GLiNER model are replaced by fakes so that no spaCy model is loaded
and no model is downloaded. Tests that need real models skip themselves when those are not
installed, and they pin the engine to spaCy so that no network access happens either.
"""

import logging
import threading
import time

import anonymizer
import pytest
from presidio_analyzer import RecognizerResult
from presidio_analyzer.nlp_engine import NlpEngineProvider
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


class FakeNlpEngineProvider(NlpEngineProvider):
    """Stand-in for NlpEngineProvider that records the configuration it is given and loads no model.

    Its constructor is Presidio's own, so that the configuration is validated and Presidio's default
    configuration is read as in production; only ``create_engine`` is replaced.
    """

    last_configuration = None

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


def fake_models(monkeypatch, german_model=False):
    """Replace the spaCy models by FakeNlpEngine; with ``german_model`` the German pipeline exists."""
    monkeypatch.setattr(anonymizer, "NlpEngineProvider", FakeNlpEngineProvider)
    monkeypatch.setattr(
        anonymizer, "is_package", lambda model: german_model and model == anonymizer.LANGUAGE_MODELS["de"]
    )


def build_engines(monkeypatch, *, german_model=False, gliner=None):
    """Build the engines through the production path, with faked model loading.

    The returned analyzer is also cached as the module engines, so that detect_pii and
    redact_text use it directly.
    """
    fake_models(monkeypatch, german_model)
    if gliner is not None:
        monkeypatch.setattr(anonymizer, "GLiNERRecognizer", gliner)
    engines = anonymizer._get_engines()
    assert engines is anonymizer._ENGINES
    return engines


def reset_engine_state():
    anonymizer._ENGINES = None
    anonymizer._UNKNOWN_ENGINE_REPORTED = False
    anonymizer._ENGINE_STATE = (anonymizer.ENGINE_SPACY, None)
    anonymizer._NEXT_GLINER_ATTEMPT = None
    anonymizer._GLINER_RETRY_RUNNING = False


@pytest.fixture(autouse=True)
def isolated_engines(monkeypatch):
    """Reset the engine cache, the engine selection and the retry state around every test."""
    reset_engine_state()
    FakeNlpEngineProvider.last_configuration = None
    monkeypatch.delenv(anonymizer.ENGINE_ENV_VAR, raising=False)
    yield
    reset_engine_state()


def spacy_entities(analyzer, language):
    """The entity types the spaCy NER recognizer of a language delivers."""
    return {entity for r in analyzer.registry.get_recognizers(language, all_fields=True) if r.name == "SpacyRecognizer" for entity in r.supported_entities}


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
    assert spacy_entities(analyzer, "en") == {"NRP", "DATE_TIME"}
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


def flaky_gliner(monkeypatch, permanent_error=None, on_success=None):
    """Make the real GLiNERRecognizer fail its first download and succeed afterwards."""
    from presidio_analyzer.predefined_recognizers.ner import gliner_recognizer

    attempts = []

    class FlakyGLiNER:
        @staticmethod
        def from_pretrained(model_name, **kwargs):
            attempts.append(model_name)
            if len(attempts) == 1:
                raise permanent_error or OSError("the hub is unreachable")
            if on_success is not None:
                on_success()
            return object()

    monkeypatch.setattr(gliner_recognizer, "GLiNER", FlakyGLiNER)
    monkeypatch.setattr(anonymizer, "GLINER_RETRY_DELAYS", (0, 0))
    return attempts


def wait_until(condition, timeout=10.0):
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "the background retry did not finish"
        time.sleep(0.01)


def test_failed_gliner_load_is_retried_without_blocking_requests(monkeypatch):
    """After a failed load the spaCy engine answers at once, and the model is loaded again later."""
    fake_models(monkeypatch, german_model=True)
    release = threading.Event()
    attempts = flaky_gliner(monkeypatch, on_success=release.wait)
    fallback_analyzer, _ = anonymizer._get_engines()

    assert "SpacyRecognizer" in recognizer_names(fallback_analyzer)
    assert anonymizer.engine_state() == (anonymizer.ENGINE_SPACY, "OSError: the hub is unreachable")

    # The cooldown has passed: the request gets the fallback engines while the load is still running.
    engines, _start = anonymizer._get_engines(), time.monotonic()
    assert time.monotonic() - _start < 5
    assert "SpacyRecognizer" in recognizer_names(engines[0])
    release.set()
    wait_until(lambda: anonymizer.engine_state()[0] == anonymizer.ENGINE_GLINER)

    analyzer, _ = anonymizer._get_engines()
    assert attempts == [anonymizer.GLINER_MODEL, anonymizer.GLINER_MODEL]
    assert "GLiNERRecognizer" in recognizer_names(analyzer)
    assert spacy_entities(analyzer, "en") == {"NRP", "DATE_TIME"}
    # The loaded spaCy models are kept, and German keeps its recognizers.
    assert analyzer.nlp_engine is fallback_analyzer.nlp_engine
    assert "DeTaxIdRecognizer" in recognizer_names(analyzer, language="de")
    assert anonymizer.engine_state() == (anonymizer.ENGINE_GLINER, None)


def test_gliner_retry_waits_for_the_cooldown(monkeypatch):
    """A failed load is not retried on every request: the next attempt waits for the delay."""
    fake_models(monkeypatch)
    attempts = flaky_gliner(monkeypatch)
    monkeypatch.setattr(anonymizer, "GLINER_RETRY_DELAYS", (3600,))
    anonymizer._get_engines()
    anonymizer._get_engines()

    assert attempts == [anonymizer.GLINER_MODEL]
    assert anonymizer._NEXT_GLINER_ATTEMPT is not None


def test_missing_gliner_package_is_not_retried(monkeypatch):
    """Without the gliner package no attempt can succeed, so none is made."""
    fake_models(monkeypatch)
    attempts = flaky_gliner(monkeypatch, permanent_error=ImportError("GLiNER is not installed"))
    analyzer, _ = anonymizer._get_engines()
    assert "SpacyRecognizer" in recognizer_names(analyzer)

    anonymizer._get_engines()
    time.sleep(0.1)

    assert attempts == [anonymizer.GLINER_MODEL]
    assert anonymizer._NEXT_GLINER_ATTEMPT is None


def test_failing_retry_backs_off_and_keeps_the_first_error(monkeypatch):
    """Every failed attempt waits longer, and the administrator sees the first error only."""
    from presidio_analyzer.predefined_recognizers.ner import gliner_recognizer

    fake_models(monkeypatch)
    monkeypatch.setattr(anonymizer, "GLINER_RETRY_DELAYS", (60, 300, 1800))
    clock = [1000.0]
    monkeypatch.setattr(anonymizer.time, "monotonic", lambda: clock[0])
    errors = iter([OSError("first failure"), OSError("second failure"), OSError("third failure")])

    class FailingGLiNER:
        @staticmethod
        def from_pretrained(model_name, **kwargs):
            raise next(errors)

    monkeypatch.setattr(gliner_recognizer, "GLiNER", FailingGLiNER)
    anonymizer._get_engines()
    assert anonymizer.engine_state() == (anonymizer.ENGINE_SPACY, "OSError: first failure")

    anonymizer._retry_gliner(anonymizer._ENGINES)
    assert anonymizer._NEXT_GLINER_ATTEMPT == (1300.0, 1)
    anonymizer._retry_gliner(anonymizer._ENGINES)
    assert anonymizer._NEXT_GLINER_ATTEMPT == (2800.0, 2)
    assert anonymizer.engine_state() == (anonymizer.ENGINE_SPACY, "OSError: first failure")


@pytest.mark.parametrize(("storage", "expected"), [(None, None), ("/nc_app", "/nc_app")])
def test_gliner_model_is_loaded_from_the_persistent_storage(monkeypatch, storage, expected):
    """The model is loaded from where the download put it: the storage that survives updates."""
    from presidio_analyzer.predefined_recognizers.ner import gliner_recognizer

    monkeypatch.setattr(anonymizer, "model_cache_dir", lambda: storage)
    loads = []

    class RecordingGLiNER:
        @staticmethod
        def from_pretrained(model_name, **kwargs):
            loads.append(kwargs)
            return object()

    monkeypatch.setattr(gliner_recognizer, "GLiNER", RecordingGLiNER)
    build_engines(monkeypatch, german_model=True)

    assert [load["cache_dir"] for load in loads] == [expected]
    # A request must never wait for a download: the load on the request path reads the storage only.
    assert loads[0]["local_files_only"] is True


def test_missing_model_is_fetched_in_the_background_at_once(monkeypatch):
    """A model that is not in the storage never holds a request up.

    The load on the request path reads the storage only and fails there. The background retry then asks the
    Hub without waiting for a further request or for the first delay, and the engines switch once it is done.
    """
    from presidio_analyzer.predefined_recognizers.ner import gliner_recognizer

    fake_models(monkeypatch)
    loads = []

    class StorageOrHub:
        @staticmethod
        def from_pretrained(model_name, **kwargs):
            loads.append(kwargs.get("local_files_only"))
            if kwargs.get("local_files_only"):
                raise FileNotFoundError("the model is not in the storage")
            return object()

    monkeypatch.setattr(gliner_recognizer, "GLiNER", StorageOrHub)
    monkeypatch.setattr(anonymizer, "GLINER_RETRY_DELAYS", (3600,))
    analyzer, _ = anonymizer._get_engines()

    assert anonymizer.engine_state() == (
        anonymizer.ENGINE_SPACY,
        "FileNotFoundError: the model is not in the storage",
    )
    assert "SpacyRecognizer" in recognizer_names(analyzer)
    wait_until(lambda: anonymizer.engine_state() == (anonymizer.ENGINE_GLINER, None))
    assert loads == [True, False]


NRP_DATE_TEXT = "Maria Weber is a British citizen of Muslim faith who supports the Labour Party and will visit us next Tuesday."


def fake_gliner_model(monkeypatch, canned):
    """Stand in for the loaded GLiNER model: it returns a span for every label it is asked for."""
    from presidio_analyzer.predefined_recognizers.ner import gliner_recognizer

    class FakeModel:
        asked = []

        def predict_entities(self, text, labels, **kwargs):
            FakeModel.asked.append(list(labels))
            return [
                {"start": text.index(value), "end": text.index(value) + len(value), "text": value, "label": label, "score": 0.9}
                for label, value in canned
                if label in labels and value in text
            ]

    monkeypatch.setattr(gliner_recognizer, "GLiNER", type("GLiNER", (), {"from_pretrained": staticmethod(lambda *a, **kw: FakeModel())}))
    return FakeModel


def test_gliner_is_asked_for_nrp_and_date_labels(monkeypatch):
    """The model is asked for the labels behind NRP and DATE_TIME, in every language.

    Before the GLiNER engine, English texts got both from the spaCy NER.
    """
    model = fake_gliner_model(
        monkeypatch, [("nationality", "British"), ("religion", "Muslim faith"), ("date", "next Tuesday"), ("date", "nächsten Dienstag")]
    )
    build_engines(monkeypatch, german_model=True)

    detected = anonymizer.detect_pii(NRP_DATE_TEXT)
    assert "NRP: British" in detected
    assert "NRP: Muslim faith" in detected
    assert "DATE_TIME: next Tuesday" in detected
    assert "DATE_TIME: nächsten Dienstag" in anonymizer.detect_pii("Anna kommt nächsten Dienstag.", language="de")
    asked_en, asked_de = model.asked
    for asked in (asked_en, asked_de):
        assert {"nationality", "religion", "political group", "date", "time"} <= set(asked)
        # The Presidio names of the two types are never sent as raw labels.
        assert not {"NRP", "DATE_TIME"} & set(asked)


def test_gliner_engine_keeps_spacy_ner_for_nrp_and_date_time_only(monkeypatch):
    """Next to the model, the spaCy NER of English delivers only the two types of its labels.

    de_core_news_lg emits PER, LOC, ORG and MISC only, so German gets no reduced recognizer: it would never
    return anything.
    """
    fake_gliner = make_fake_gliner({"en": [], "de": []})
    analyzer, _ = build_engines(monkeypatch, german_model=True, gliner=fake_gliner)

    assert spacy_entities(analyzer, "en") == {"NRP", "DATE_TIME"}
    assert spacy_entities(analyzer, "de") == set()


@pytest.mark.spacy
@pytest.mark.skipif(not ENGLISH_MODEL_INSTALLED, reason="the English spaCy model is not installed")
def test_spacy_ner_delivers_nrp_and_date_time_with_the_gliner_engine(monkeypatch):
    """With the real spaCy model, the two types come from the spaCy NER even if the model finds nothing."""
    monkeypatch.setattr(anonymizer, "GLiNERRecognizer", make_fake_gliner({"en": []}))

    detected = anonymizer.detect_pii(NRP_DATE_TEXT)
    assert "NRP: British" in detected
    assert "DATE_TIME: next Tuesday" in detected


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


def test_empty_engine_value_is_no_choice_and_needs_no_warning(monkeypatch, caplog):
    """An empty value means the variable was not set: GLiNER is used without a warning about it."""
    monkeypatch.setenv(anonymizer.ENGINE_ENV_VAR, "")
    assert anonymizer._requested_engine() == anonymizer.ENGINE_GLINER
    assert caplog.text == ""


def test_an_unknown_engine_value_is_reported_once(monkeypatch, caplog):
    """An unknown value is reported once per process, not at every download and every engine build."""
    monkeypatch.setenv(anonymizer.ENGINE_ENV_VAR, "transformers")

    assert [anonymizer._requested_engine() for _ in range(3)] == [anonymizer.ENGINE_GLINER] * 3
    assert len(caplog.records) == 1


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


def test_nlp_configuration_keeps_presidio_default_ner_settings(monkeypatch):
    """The spaCy engine keeps the NER settings of ``AnalyzerEngine()``: Presidio's defaults, and MISC ignored.

    Without them the labels Presidio ignores for their false positives (ORGANIZATION, CARDINAL, MONEY, ...)
    come through, FAC is no longer a LOCATION, and each unmapped label logs a warning.
    """
    build_engines(monkeypatch, german_model=True)
    ner = FakeNlpEngineProvider.last_configuration["ner_model_configuration"]
    defaults = NlpEngineProvider().nlp_configuration["ner_model_configuration"]

    assert ner["model_to_presidio_entity_mapping"]["FAC"] == "LOCATION"
    assert {"ORGANIZATION", "CARDINAL", "MONEY", "EVENT"} <= set(ner["labels_to_ignore"])
    # de_core_news_lg emits MISC for events, products and the like: a label without a Presidio type.
    assert ner == {**defaults, "labels_to_ignore": [*defaults["labels_to_ignore"], "MISC"]}


def test_gliner_engine_still_redacts_organizations(monkeypatch):
    """The ignored spaCy label stays ignored: an organization the GLiNER model finds is redacted."""
    fake_gliner = make_fake_gliner({"en": [("ORGANIZATION", "Apple", 0.9)]})
    build_engines(monkeypatch, gliner=fake_gliner)

    assert anonymizer.redact_text("I work for Apple.") == "I work for <ORGANIZATION>."


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


@pytest.mark.spacy
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
@pytest.mark.spacy
def test_spacy_engine_detects_german_pii(monkeypatch):
    """The spaCy fallback engine detects German PII end to end when German is requested."""
    monkeypatch.setenv(anonymizer.ENGINE_ENV_VAR, anonymizer.ENGINE_SPACY)
    detected = anonymizer.detect_pii(GERMAN_TEXT, language="de")
    assert f"DE_TAX_ID: {VALID_TAX_ID}" in detected
    assert "PERSON: Anna Müller" in detected


@pytest.mark.spacy
@pytest.mark.skipif(not ENGLISH_MODEL_INSTALLED, reason="the English spaCy model is not installed")
def test_gliner_failure_falls_back_to_working_spacy_detection(monkeypatch):
    """After a GLiNER load failure, detection still works through the real spaCy pipeline."""
    fake_gliner = make_fake_gliner({}, load_error=ImportError("the gliner package is not installed"))
    monkeypatch.setattr(anonymizer, "GLiNERRecognizer", fake_gliner)

    detected = anonymizer.detect_pii(ENGLISH_TEXT)
    assert "PERSON: John Smith" in detected


AIRPORT_TEXT = "My name is John Smith and I work for Apple. Tomorrow I fly from Tegel airport to Paris."
GERMAN_NATIONALITY_TEXT = "Anna Müller ist Deutsche und arbeitet bei Siemens in Berlin."


@pytest.mark.spacy
@pytest.mark.skipif(
    not (ENGLISH_MODEL_INSTALLED and GERMAN_MODEL_INSTALLED),
    reason="the English and German spaCy models are not installed",
)
def test_spacy_engine_keeps_presidio_default_ner_settings(monkeypatch, caplog):
    """With the real models, the spaCy engine finds what ``AnalyzerEngine()`` found.

    An airport is a location, an organization and a German MISC entity are left alone, and no label is
    reported as unmapped, in either language.
    """
    monkeypatch.setenv(anonymizer.ENGINE_ENV_VAR, anonymizer.ENGINE_SPACY)
    with caplog.at_level(logging.WARNING, logger="presidio-analyzer"):
        english = anonymizer.redact_text(AIRPORT_TEXT)
        german = anonymizer.redact_text(GERMAN_NATIONALITY_TEXT, language="de")

    assert "Tegel" not in english
    assert "<LOCATION> airport" in english
    assert "Apple" in english
    assert "<ORGANIZATION>" not in english
    assert "<PERSON>" in german
    assert "Deutsche" in german
    assert "Siemens" in german
    assert "<LOCATION>" in german
    assert [message for message in caplog.messages if "is not mapped" in message] == []
