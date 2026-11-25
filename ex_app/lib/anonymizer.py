# SPDX-FileCopyrightText: 2026 Nextcloud GmbH and Nextcloud contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""PII detection and redaction using Presidio.

Entities are detected by the GLiNER model by default, with the spaCy NER recognizer kept as a
fallback engine when GLiNER is disabled or fails to load. English and German are supported;
German adds its own spaCy model and the DE_* recognizer pack.
"""

import copy
import logging
import os
import threading
from collections.abc import Callable

from presidio_analyzer import AnalyzerEngine, RecognizerResult
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_analyzer.predefined_recognizers import (
    CreditCardRecognizer,
    DeBsnrRecognizer,
    DeFuehrerscheinRecognizer,
    DeHandelsregisterRecognizer,
    DeHealthInsuranceRecognizer,
    DeIdCardRecognizer,
    DeKfzRecognizer,
    DeLanrRecognizer,
    DePassportRecognizer,
    DeSocialSecurityRecognizer,
    DeTaxIdRecognizer,
    DeTaxNumberRecognizer,
    DeVatIdRecognizer,
    GLiNERRecognizer,
)
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from spacy.util import is_package

ProgressCallback = Callable[[float], None]

LOGGER = logging.getLogger(__name__)

# Detection engine selection: ENGINE_GLINER (default) or ENGINE_SPACY.
ENGINE_ENV_VAR = "TEXT_ANONYMIZER_ENGINE"
ENGINE_GLINER = "gliner"
ENGINE_SPACY = "spacy"

# Zero-shot PII model, Apache-2.0, multilingual, runs on CPU. Its entity labels are mapped to
# Presidio types; labels without a Presidio equivalent (e.g. "passport number") are omitted
# because the country-specific regex recognizers with checksums cover those IDs more precisely.
GLINER_MODEL = "urchade/gliner_multi_pii-v1"
GLINER_ENTITY_MAPPING = {
    "person": "PERSON",
    "name": "PERSON",
    "organization": "ORGANIZATION",
    "location": "LOCATION",
    "address": "LOCATION",
    "date of birth": "DATE_TIME",
    "phone number": "PHONE_NUMBER",
    "mobile phone number": "PHONE_NUMBER",
    "landline phone number": "PHONE_NUMBER",
    "fax number": "PHONE_NUMBER",
    "email": "EMAIL_ADDRESS",
    "email address": "EMAIL_ADDRESS",
    "iban": "IBAN",
    "credit card number": "CREDIT_CARD",
    "bank account number": "BANK_ACCOUNT",
    "ip address": "IP_ADDRESS",
    "username": "USERNAME",
    "social media handle": "USERNAME",
    "medication": "MEDICATION",
    "medical condition": "MEDICAL_CONDITION",
    "blood type": "BLOOD_TYPE",
}

LANGUAGE_MODELS = {"en": "en_core_web_lg", "de": "de_core_news_lg"}
# German recognizer pack shipped with Presidio 2.2.363+. DePlzRecognizer is left out: a German
# postal code matches any five digits, and the official Presidio German recipe flags it as a
# false-positive risk unless detection thresholds are tuned.
GERMAN_RECOGNIZERS = (
    DeBsnrRecognizer,
    DeFuehrerscheinRecognizer,
    DeHandelsregisterRecognizer,
    DeHealthInsuranceRecognizer,
    DeIdCardRecognizer,
    DeKfzRecognizer,
    DeLanrRecognizer,
    DePassportRecognizer,
    DeSocialSecurityRecognizer,
    DeTaxIdRecognizer,
    DeTaxNumberRecognizer,
    DeVatIdRecognizer,
)
# Presidio's default configuration registers the credit card recognizer for en, es, it and pl.
GERMAN_CARD_CONTEXT = [
    "kreditkarte",
    "kreditkartennummer",
    "kartennummer",
    "ec-karte",
    "girocard",
    "maestro",
    "visa",
    "mastercard",
    "amex",
]

_LOCK = threading.Lock()
_ENGINES: tuple[AnalyzerEngine, AnonymizerEngine] | None = None
_KEEP_OPERATORS = {"DEFAULT": OperatorConfig("keep")}
# Analysis time grows faster than text length, and spaCy rejects texts over 1,000,000 characters.
CHUNK_SIZE = 10_000


def _requested_engine() -> str:
    """Read the detection engine choice from the environment, defaulting to GLiNER."""
    engine = os.environ.get(ENGINE_ENV_VAR, ENGINE_GLINER).strip().lower()
    if engine not in (ENGINE_GLINER, ENGINE_SPACY):
        LOGGER.warning("Unknown %s value %r, using the %s engine", ENGINE_ENV_VAR, engine, ENGINE_GLINER)
        return ENGINE_GLINER
    return engine


def _supported_languages() -> list[str]:
    """List the languages that have an installed spaCy model, always including English."""
    languages = ["en"]
    if is_package(LANGUAGE_MODELS["de"]):
        languages.append("de")
    else:
        LOGGER.warning(
            "The %s model is not installed, German texts are analyzed with the English pipeline",
            LANGUAGE_MODELS["de"],
        )
    return languages


def _build_gliner_recognizers(languages: list[str]) -> list[GLiNERRecognizer]:
    """One GLiNER recognizer per language. The first loads the model, the others share it.

    The constructor of every recognizer loads the model itself (``EntityRecognizer.__init__`` calls
    ``load()``), so the other languages are shallow copies of the first recognizer: apart from the
    language all of their attributes are equal, and they share the loaded model.
    """
    first = GLiNERRecognizer(
        supported_language=languages[0],
        model_name=GLINER_MODEL,
        entity_mapping=GLINER_ENTITY_MAPPING,
        flat_ner=False,
        multi_label=True,
        map_location="cpu",
    )
    recognizers = [first]
    for language in languages[1:]:
        recognizer = copy.copy(first)
        recognizer.supported_language = language
        recognizers.append(recognizer)
    return recognizers


def _enable_gliner(analyzer: AnalyzerEngine, languages: list[str]) -> bool:
    """Add a GLiNER recognizer per language and drop the spaCy NER recognizers.

    All language instances share one loaded model. If loading fails for any reason (gliner
    package missing, model download unavailable), the spaCy recognizers are kept as the
    fallback engine.
    """
    try:
        recognizers = _build_gliner_recognizers(languages)
    except Exception:
        LOGGER.warning(
            "The GLiNER model %s could not be loaded, using the spaCy detection engine",
            GLINER_MODEL,
            exc_info=True,
        )
        return False

    for recognizer in recognizers:
        analyzer.registry.add_recognizer(recognizer)
    analyzer.registry.remove_recognizer("SpacyRecognizer")
    return True


def _build_engines() -> tuple[AnalyzerEngine, AnonymizerEngine]:
    """Create the analyzer and anonymizer with the recognizer set for every supported language."""
    languages = _supported_languages()
    nlp_engine = NlpEngineProvider(
        nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": language, "model_name": LANGUAGE_MODELS[language]} for language in languages],
        }
    ).create_engine()
    analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=languages)
    if "de" in languages:
        for recognizer in GERMAN_RECOGNIZERS:
            analyzer.registry.add_recognizer(recognizer(supported_language="de"))
        analyzer.registry.add_recognizer(CreditCardRecognizer(supported_language="de", context=GERMAN_CARD_CONTEXT))
    if _requested_engine() == ENGINE_GLINER:
        engine = ENGINE_GLINER if _enable_gliner(analyzer, languages) else ENGINE_SPACY
    else:
        # The fallback keeps the multi-language recognizer configuration built above: the German
        # recognizers stay active whenever the German spaCy model is installed, so the fallback
        # engine differs from the GLiNER engine only by the missing zero-shot model.
        engine = ENGINE_SPACY
    LOGGER.info("Detection engine: %s (languages: %s)", engine, languages)
    return analyzer, AnonymizerEngine()


def _get_engines() -> tuple[AnalyzerEngine, AnonymizerEngine]:
    """Create the Presidio engines on first use, as loading the models takes several seconds."""
    global _ENGINES
    with _LOCK:
        if _ENGINES is None:
            _ENGINES = _build_engines()
        return _ENGINES


def _resolve_language(language: str, supported: list[str]) -> str:
    """Fall back to English for languages without an installed model, so requests never fail."""
    if language not in supported:
        LOGGER.warning(
            "Language %r is not supported (installed: %s), analyzing the text as English",
            language,
            supported,
        )
        return "en"
    return language


def _chunk_bounds(text: str, max_size: int) -> list[tuple[int, int]]:
    """Split ``text`` into ``(start, end)`` ranges of at most ``max_size`` characters.

    Ranges end after the last line break in the window, or after the last space or tab if there is none, so that
    entities are rarely cut in two.
    """
    bounds = []
    start = 0
    while len(text) - start > max_size:
        window = text[start : start + max_size]
        cut = window.rfind("\n") + 1 or max(window.rfind(" "), window.rfind("\t")) + 1 or max_size
        bounds.append((start, start + cut))
        start += cut
    bounds.append((start, len(text)))
    return bounds


def _analyze(
    analyzer: AnalyzerEngine, text: str, language: str, on_progress: ProgressCallback | None
) -> list[RecognizerResult]:
    """Analyze ``text`` chunk by chunk, calling ``on_progress`` with a percentage after each chunk.

    The language is resolved against the analyzer's supported languages, so a request for an
    unsupported language falls back to English instead of failing the task.
    """
    language = _resolve_language(language, analyzer.supported_languages)
    bounds = _chunk_bounds(text, CHUNK_SIZE)
    results = []
    for i, (start, end) in enumerate(bounds, 1):
        for result in analyzer.analyze(text=text[start:end], language=language):
            result.start += start
            result.end += start
            results.append(result)
        if on_progress:
            on_progress(i / len(bounds) * 100)
    return results


def detect_pii(text: str, language: str = "en", on_progress: ProgressCallback | None = None) -> str:
    """List each distinct PII entity in ``text`` as a ``TYPE: value`` line, in order of first appearance.

    Overlapping entities are resolved the same way as in :func:`redact_text`, so the list matches what would be
    redacted. Returns an empty string if no PII was found.
    """
    analyzer, anonymizer = _get_engines()
    results = _analyze(analyzer, text, language, on_progress)
    items = anonymizer.anonymize(text=text, analyzer_results=results, operators=_KEEP_OPERATORS).items
    entities = dict.fromkeys((item.entity_type, item.text) for item in sorted(items, key=lambda item: item.start))
    return "\n".join(f"{entity_type}: {value}" for entity_type, value in entities)


def redact_text(text: str, language: str = "en", on_progress: ProgressCallback | None = None) -> str:
    """Replace every PII entity in ``text`` with a placeholder naming its type, e.g. ``<PERSON>``."""
    analyzer, anonymizer = _get_engines()
    results = _analyze(analyzer, text, language, on_progress)
    return anonymizer.anonymize(text=text, analyzer_results=results).text
