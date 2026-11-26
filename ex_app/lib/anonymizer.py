# SPDX-FileCopyrightText: 2026 Nextcloud GmbH and Nextcloud contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""PII detection and redaction using Presidio.

Entities are detected by the GLiNER model by default, with the spaCy NER recognizer kept as a
fallback engine when GLiNER is disabled or fails to load. English and German are supported;
German adds its own spaCy model and the DE_* recognizer pack.
"""

import copy
import json
import logging
import os
import threading
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path

from huggingface_hub import snapshot_download
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
    SpacyRecognizer,
)
from presidio_analyzer.recognizer_registry import RecognizerRegistry
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from py3langid.langid import MODEL_FILE, LanguageIdentifier
from spacy.util import is_package
from tqdm import tqdm
from transformers import AutoTokenizer

ProgressCallback = Callable[[float], None]

LOGGER = logging.getLogger(__name__)

# Detection engine selection: ENGINE_GLINER (default) or ENGINE_SPACY.
ENGINE_ENV_VAR = "TEXT_ANONYMIZER_ENGINE"
ENGINE_GLINER = "gliner"
ENGINE_SPACY = "spacy"

# Zero-shot PII model, Apache-2.0, multilingual, runs on CPU. Its entity labels are mapped to
# Presidio types; labels without a Presidio equivalent (e.g. "passport number") are omitted
# because the country-specific regex recognizers with checksums cover those IDs more precisely.
# The labels of NRP and DATE_TIME stand for two types that the spaCy NER delivers as well.
GLINER_MODEL = "urchade/gliner_multi_pii-v1"
GLINER_ENTITY_MAPPING = {
    "person": "PERSON",
    "name": "PERSON",
    "organization": "ORGANIZATION",
    "location": "LOCATION",
    "address": "LOCATION",
    "date of birth": "DATE_TIME",
    "date": "DATE_TIME",
    "time": "DATE_TIME",
    "nationality": "NRP",
    "religion": "NRP",
    "political group": "NRP",
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
# The types the spaCy NER keeps delivering next to the GLiNER model.
GLINER_SPACY_ENTITIES = ("NRP", "DATE_TIME")
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

# Seconds to wait before the GLiNER model is loaded again after a failed attempt; the last
# value then repeats.
GLINER_RETRY_DELAYS = (60, 300, 1800, 3600)

_LOCK = threading.Lock()
# Held for as long as a download of the model runs.
_DOWNLOAD_LOCK = threading.Lock()
_ENGINES: tuple[AnalyzerEngine, AnonymizerEngine] | None = None
# The engine in use and, while GLiNER is the requested engine but could not be loaded, the error.
_ENGINE_STATE: tuple[str, str | None] = (ENGINE_SPACY, None)
_NEXT_GLINER_ATTEMPT: tuple[float, int] | None = None
_GLINER_RETRY_RUNNING = False
# The language detector is loaded once, under a lock of its own.
_IDENTIFIER_LOCK = threading.Lock()
_IDENTIFIER: LanguageIdentifier | None = None
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


class _DownloadProgress(tqdm):
    """A progress bar of huggingface_hub that hands its percentage to a callback."""

    def __init__(self, *args, on_progress: Callable[[float], None], **kwargs):
        # huggingface_hub names the bar, which tqdm does not accept.
        kwargs.pop("name", None)
        self._on_progress = on_progress
        super().__init__(*args, **kwargs)

    def display(self, msg=None, pos=None) -> None:
        if self.total:
            self._on_progress(self.n / self.total * 100)
        super().display(msg, pos)


def model_cache_dir() -> str | None:
    """The persistent storage of the app, which survives updates, or None for huggingface_hub's cache.

    AppAPI mounts the storage and passes its path in the environment; without it, for example in
    a development setup, the model is cached where huggingface_hub puts it by default.
    """
    return os.environ.get("APP_PERSISTENT_STORAGE")


def _tokenizer_name(model_dir: str) -> str:
    """The repository of the tokenizer, which the configuration of the model names."""
    config = Path(model_dir, "gliner_config.json")
    return json.loads(config.read_text(encoding="utf-8"))["model_name"]


def _model_in_storage(cache_dir: str | None) -> bool:
    """Whether the model and its tokenizer are complete in the storage; this asks the Hub nothing."""
    try:
        model_dir = snapshot_download(GLINER_MODEL, cache_dir=cache_dir, local_files_only=True)
        AutoTokenizer.from_pretrained(_tokenizer_name(model_dir), cache_dir=cache_dir, local_files_only=True)
    except Exception:
        return False
    return True


def download_model(on_progress: Callable[[float], None]) -> bool:
    """Download the GLiNER model and its tokenizer into the persistent storage of the app.

    Only what is missing is downloaded: a model that is complete in the storage is left alone and costs no
    network. A later load of the model then needs none either. ``on_progress`` gets the percentage of the model
    weights that are downloaded. Nothing is downloaded when the spaCy engine is selected.
    """
    if _requested_engine() != ENGINE_GLINER:
        return False
    # One download at a time: the init, the start of the app and a retry may ask for the model together.
    with _DOWNLOAD_LOCK:
        cache_dir = model_cache_dir()
        if not _model_in_storage(cache_dir):
            model_dir = snapshot_download(
                GLINER_MODEL, cache_dir=cache_dir, tqdm_class=partial(_DownloadProgress, on_progress=on_progress)
            )
            # The tokenizer is a repository of its own, named in the model's configuration.
            AutoTokenizer.from_pretrained(_tokenizer_name(model_dir), cache_dir=cache_dir)
    return True


def _build_gliner_recognizers(languages: list[str], download: bool = False) -> list[GLiNERRecognizer]:
    """One GLiNER recognizer per language. The first loads the model, the others share it.

    The constructor of every recognizer loads the model itself (``EntityRecognizer.__init__`` calls
    ``load()``), so the other languages are shallow copies of the first recognizer: apart from the
    language all of their attributes are equal, and they share the loaded model.

    The model is read from the storage only, so that no request waits for a download. With ``download``
    it is fetched from the Hub when it is missing there, which only the background retry does.
    """
    first = GLiNERRecognizer(
        supported_language=languages[0],
        model_name=GLINER_MODEL,
        entity_mapping=GLINER_ENTITY_MAPPING,
        flat_ner=False,
        multi_label=True,
        map_location="cpu",
        cache_dir=model_cache_dir(),
        local_files_only=not download,
    )
    recognizers = [first]
    for language in languages[1:]:
        recognizer = copy.copy(first)
        recognizer.supported_language = language
        recognizers.append(recognizer)
    return recognizers


def _use_gliner(analyzer: AnalyzerEngine, recognizers: list[GLiNERRecognizer]) -> None:
    """Register the GLiNER recognizers and reduce the spaCy NER recognizers to NRP and DATE_TIME.

    The spaCy pipeline runs for every engine, so its spans of these two types come at no cost as a
    second source next to the labels of the model.
    """
    registry = analyzer.registry
    registry.remove_recognizer("SpacyRecognizer")
    for language in analyzer.supported_languages:
        registry.add_recognizer(SpacyRecognizer(supported_language=language, supported_entities=list(GLINER_SPACY_ENTITIES)))
    for recognizer in recognizers:
        registry.add_recognizer(recognizer)


def _enable_gliner(analyzer: AnalyzerEngine, languages: list[str]) -> bool:
    """Switch the analyzer to the GLiNER engine: one recognizer per language, sharing one loaded model.

    If loading fails for any reason (gliner package missing, model download unavailable), the
    spaCy recognizers are kept as the fallback engine and the load is tried again later.
    """
    try:
        recognizers = _build_gliner_recognizers(languages)
    except Exception as error:
        LOGGER.warning(
            "The GLiNER model %s could not be loaded, using the spaCy detection engine",
            GLINER_MODEL,
            exc_info=True,
        )
        _schedule_gliner_retry(error)
        return False

    _use_gliner(analyzer, recognizers)
    return True


def _nlp_configuration(languages: list[str]) -> dict:
    """The configuration of the spaCy engine for ``languages``, built on Presidio's default one.

    A configuration given to ``NlpEngineProvider`` takes the place of Presidio's ``default.yaml`` as a whole,
    the file ``AnalyzerEngine()`` reads. Its NER settings are kept: the labels with many false positives are
    ignored and the others are mapped to Presidio's types. MISC is ignored as well: ``de_core_news_lg`` emits it
    for events, products, languages and the like, and it has no Presidio type.
    """
    configuration = NlpEngineProvider().nlp_configuration
    configuration["models"] = [
        {"lang_code": language, "model_name": LANGUAGE_MODELS[language]} for language in languages
    ]
    labels_to_ignore = configuration["ner_model_configuration"]["labels_to_ignore"]
    if "MISC" not in labels_to_ignore:
        labels_to_ignore.append("MISC")
    return configuration


def _build_engines() -> tuple[AnalyzerEngine, AnonymizerEngine]:
    """Create the analyzer and anonymizer with the recognizer set for every supported language."""
    global _ENGINE_STATE
    languages = _supported_languages()
    nlp_engine = NlpEngineProvider(nlp_configuration=_nlp_configuration(languages)).create_engine()
    analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=languages)
    if "de" in languages:
        for recognizer in GERMAN_RECOGNIZERS:
            analyzer.registry.add_recognizer(recognizer(supported_language="de"))
        analyzer.registry.add_recognizer(CreditCardRecognizer(supported_language="de", context=GERMAN_CARD_CONTEXT))
    # The spaCy engine keeps the recognizer configuration built above: the German recognizers
    # stay active whenever the German spaCy model is installed, so it differs from the GLiNER
    # engine only by the missing zero-shot model.
    engine = ENGINE_SPACY
    if _requested_engine() == ENGINE_GLINER and _enable_gliner(analyzer, languages):
        engine = ENGINE_GLINER
        _ENGINE_STATE = (ENGINE_GLINER, None)
    LOGGER.info("Detection engine: %s (languages: %s)", engine, languages)
    return analyzer, AnonymizerEngine()


def _schedule_gliner_retry(error: Exception) -> None:
    """Remember a failed GLiNER load and when to try again. Called with ``_LOCK`` held."""
    global _NEXT_GLINER_ATTEMPT, _ENGINE_STATE
    error_class = type(error).__name__
    recorded = _ENGINE_STATE[1]
    if recorded is None or recorded.split(":", 1)[0] != error_class:
        _ENGINE_STATE = (ENGINE_SPACY, f"{error_class}: {error}")
    if isinstance(error, ImportError):
        # Without the gliner package no attempt can succeed.
        _NEXT_GLINER_ATTEMPT = None
        return
    if isinstance(error, FileNotFoundError):
        # The model is not in the storage. Fetching it is the first attempt, which is due at once.
        _NEXT_GLINER_ATTEMPT = (time.monotonic(), 0)
        return
    index = 0 if _NEXT_GLINER_ATTEMPT is None else min(_NEXT_GLINER_ATTEMPT[1] + 1, len(GLINER_RETRY_DELAYS) - 1)
    _NEXT_GLINER_ATTEMPT = (time.monotonic() + GLINER_RETRY_DELAYS[index], index)


def _get_engines() -> tuple[AnalyzerEngine, AnonymizerEngine]:
    """The Presidio engines, built on first use as loading the models takes several seconds."""
    global _ENGINES
    with _LOCK:
        if _ENGINES is None:
            _ENGINES = _build_engines()
        engines = _ENGINES
        _start_gliner_retry()
    return engines


def _start_gliner_retry() -> None:
    """Load the GLiNER model in the background if a failed load is due. Called with ``_LOCK`` held."""
    global _GLINER_RETRY_RUNNING
    if _NEXT_GLINER_ATTEMPT is None or _GLINER_RETRY_RUNNING or time.monotonic() < _NEXT_GLINER_ATTEMPT[0]:
        return
    _GLINER_RETRY_RUNNING = True
    threading.Thread(target=_retry_gliner, args=(_ENGINES,), daemon=True).start()


def _retry_gliner(fallback: tuple[AnalyzerEngine, AnonymizerEngine]) -> None:
    """Load the GLiNER model while requests keep using the fallback, then replace it in one step.

    This is the only place that may fetch the model from the Hub.
    """
    global _NEXT_GLINER_ATTEMPT, _ENGINE_STATE, _GLINER_RETRY_RUNNING, _ENGINES
    fallback_analyzer, anonymizer_engine = fallback
    languages = fallback_analyzer.supported_languages
    try:
        recognizers = _build_gliner_recognizers(languages, download=True)
        # A new analyzer over the loaded spaCy models: a request that already holds the fallback
        # analyzer finishes with it, and no recognizer is removed from an analyzer in use.
        registry = RecognizerRegistry(
            recognizers=[r for r in fallback_analyzer.registry.recognizers if r.name != "SpacyRecognizer"],
            supported_languages=languages,
        )
        analyzer = AnalyzerEngine(nlp_engine=fallback_analyzer.nlp_engine, registry=registry, supported_languages=languages)
        _use_gliner(analyzer, recognizers)
    except Exception as error:
        LOGGER.warning(
            "The GLiNER model %s could not be loaded, using the spaCy detection engine",
            GLINER_MODEL,
            exc_info=True,
        )
        with _LOCK:
            _schedule_gliner_retry(error)
    else:
        LOGGER.info("The GLiNER model %s is loaded, using the GLiNER detection engine", GLINER_MODEL)
        with _LOCK:
            _ENGINES = (analyzer, anonymizer_engine)
            _NEXT_GLINER_ATTEMPT = None
            _ENGINE_STATE = (ENGINE_GLINER, None)
    finally:
        with _LOCK:
            _GLINER_RETRY_RUNNING = False


def engine_state() -> tuple[str, str | None]:
    """The engine in use and, while GLiNER could not be loaded, the error of its first failed load."""
    return _ENGINE_STATE


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


def detect_language(text: str, supported: list[str]) -> str | None:
    """The language of ``text`` if it is one of ``supported``, else None.

    A file has no language input, so its text decides which recognizers analyze it. The detector
    knows 139 languages: a text in a language the app does not support gets None like a text
    without words, and is analyzed as English.
    """
    global _IDENTIFIER
    with _IDENTIFIER_LOCK:
        if _IDENTIFIER is None:
            _IDENTIFIER = LanguageIdentifier.from_model_file(MODEL_FILE, norm_probs=True)
        language = _IDENTIFIER.rank(text)[0][0]
    return language if language in supported else None


def _analyze(
    analyzer: AnalyzerEngine,
    text: str,
    language: str,
    on_progress: ProgressCallback | None,
    foreign_identifiers: bool = False,
) -> list[RecognizerResult]:
    """Analyze ``text`` chunk by chunk, calling ``on_progress`` with a percentage after each chunk.

    The language is resolved against the analyzer's supported languages, so a request for an
    unsupported language falls back to English instead of failing the task. With
    ``foreign_identifiers``, every other supported language adds the identifiers that only its
    recognizers know, except over a span that the text's own language found at least as confidently.
    """
    language = _resolve_language(language, analyzer.supported_languages)
    own_entities = set(analyzer.get_supported_entities(language)) if foreign_identifiers else set()
    foreign = (
        {
            other: [entity for entity in analyzer.get_supported_entities(other) if entity not in own_entities]
            for other in analyzer.supported_languages
            if other != language
        }
        if foreign_identifiers
        else {}
    )
    bounds = _chunk_bounds(text, CHUNK_SIZE)
    results = []
    for i, (start, end) in enumerate(bounds, 1):
        chunk_results = analyzer.analyze(text=text[start:end], language=language)
        for other, entities in foreign.items():
            if entities:
                chunk_results += [
                    result
                    for result in analyzer.analyze(text=text[start:end], language=other, entities=entities)
                    if all(result.score > found.score for found in chunk_results if result.intersects(found))
                ]
        for result in chunk_results:
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


def redact_text(
    text: str, language: str | None = "en", on_progress: ProgressCallback | None = None, foreign_identifiers: bool = False
) -> str:
    """Replace every PII entity in ``text`` with a placeholder naming its type, e.g. ``<PERSON>``.

    ``language=None`` takes the language of the text itself, for a text that comes without one.
    With ``foreign_identifiers``, the identifiers of the other supported languages are replaced
    as well.
    """
    analyzer, anonymizer = _get_engines()
    if language is None:
        language = detect_language(text, analyzer.supported_languages) or "en"
    results = _analyze(analyzer, text, language, on_progress, foreign_identifiers)
    return anonymizer.anonymize(text=text, analyzer_results=results).text
