# SPDX-FileCopyrightText: 2026 Nextcloud GmbH and Nextcloud contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Admin configuration tests.

Precedence (default < env < appconfig), parsing fail-soft behavior, the result filter
semantics and the declarative settings form.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import pytest
from presidio_analyzer import RecognizerResult

from config import (
    ADMIN_CONFIG_KEYS,
    DEFAULT_SCORE_THRESHOLD,
    ENV_ALLOW_PATTERNS,
    ENV_IGNORED_ENTITY_TYPES,
    ENV_SCORE_THRESHOLD,
    Settings,
    apply_admin_config,
    filter_results,
    parse_allow_patterns,
    parse_ignored_entity_types,
)

#: One value per knob per precedence level; the admin value beats env, env beats default.
ENV_LEVEL = {
    ENV_SCORE_THRESHOLD: "0.7",
    ENV_IGNORED_ENTITY_TYPES: "url, DATE_TIME",
    ENV_ALLOW_PATTERNS: "Env Corp",
}
ADMIN_LEVEL = {
    ENV_SCORE_THRESHOLD: "0.3",
    ENV_IGNORED_ENTITY_TYPES: "URL",
    ENV_ALLOW_PATTERNS: "Admin Corp",
}


def test_admin_config_keys_are_the_environment_variable_names() -> None:
    assert ADMIN_CONFIG_KEYS == (
        "TEXT_ANONYMIZER_SCORE_THRESHOLD",
        "TEXT_ANONYMIZER_IGNORED_ENTITY_TYPES",
        "TEXT_ANONYMIZER_ALLOW_PATTERNS",
    )


def test_precedence_admin_beats_env_beats_default_per_knob() -> None:
    from_env = Settings.from_env(ENV_LEVEL)
    assert from_env.score_threshold == 0.7
    assert from_env.ignored_entity_types == frozenset({"URL", "DATE_TIME"})
    assert from_env.allow_patterns == ("Env Corp",)
    layered = apply_admin_config(from_env, ADMIN_LEVEL)
    assert layered.score_threshold == 0.3
    assert layered.ignored_entity_types == frozenset({"URL"})
    assert layered.allow_patterns == ("Admin Corp",)


def test_defaults_apply_without_env_or_admin() -> None:
    settings = Settings.from_env({})
    assert settings == Settings()
    assert settings.score_threshold is None
    assert settings.ignored_entity_types == frozenset()
    assert settings.allow_patterns == ()


def test_no_score_threshold_until_the_threshold_itself_is_set() -> None:
    """Only a threshold saved by the admin or set in the environment filters by score.

    Saving an unrelated knob must not silently start dropping low-score detections.
    """
    results = [_result("PHONE_NUMBER", 0, 12, 0.40)]
    settings = Settings(allow_patterns=("ZZZ-never-matches",))
    assert filter_results(results, "a phone 555-0134", settings) == results
    settings = Settings(ignored_entity_types=frozenset({"URL"}))
    assert filter_results(results, "a phone 555-0134", settings) == results
    assert Settings.from_env({}).score_threshold is None
    assert Settings.from_env({ENV_SCORE_THRESHOLD: "0.7"}).score_threshold == 0.7


def test_unset_or_empty_admin_key_keeps_the_env_value() -> None:
    base = Settings.from_env({ENV_SCORE_THRESHOLD: "0.7"})
    assert apply_admin_config(base, {}).score_threshold == 0.7
    assert apply_admin_config(base, {ENV_SCORE_THRESHOLD: ""}).score_threshold == 0.7
    assert apply_admin_config(base, {ENV_SCORE_THRESHOLD: "   "}).score_threshold == 0.7


@pytest.mark.parametrize("bad", ["abc", "1.5", "-0.1", "nan", "inf"])
def test_invalid_env_threshold_raises_at_startup(bad: str) -> None:
    with pytest.raises(ValueError, match=ENV_SCORE_THRESHOLD):
        Settings.from_env({ENV_SCORE_THRESHOLD: bad})


@pytest.mark.parametrize("bad", ["abc", "1.5", "-0.1", "nan", "inf"])
def test_invalid_admin_threshold_fails_soft_to_the_lower_level(caplog, bad: str) -> None:
    base = Settings.from_env({ENV_SCORE_THRESHOLD: "0.7"})
    with caplog.at_level(logging.WARNING, logger="text_anonymizer.config"):
        layered = apply_admin_config(base, {ENV_SCORE_THRESHOLD: bad})
    assert layered.score_threshold == 0.7
    assert ENV_SCORE_THRESHOLD in caplog.text


def test_ignored_entity_types_are_normalized() -> None:
    assert parse_ignored_entity_types(" url ,  DATE_TIME ,") == frozenset({"URL", "DATE_TIME"})
    assert parse_ignored_entity_types("") == frozenset()
    assert parse_ignored_entity_types(" , ,") == frozenset()


def test_allow_patterns_drop_invalid_lines_with_a_warning(caplog) -> None:
    with caplog.at_level(logging.WARNING, logger="text_anonymizer.config"):
        patterns = parse_allow_patterns("Valid Corp\n([unclosed\n  \nAlso Valid\n")
    assert tuple(source for source, _ in patterns) == ("Valid Corp", "Also Valid")
    assert "([unclosed" in caplog.text


@pytest.mark.parametrize(
    "pattern",
    [
        "Nextcloud GmbH",
        r"[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}",
        r"\+1 \(\d{3}\) \d{3}-\d{4}",
        r"example\.(com|org)",
        r"Apple Inc\.|Nextcloud GmbH",
    ],
)
def test_allow_patterns_accept_ordinary_patterns(pattern) -> None:
    assert tuple(parse_allow_patterns(pattern)) == ((pattern, parse_allow_patterns(pattern)[0][1]),)


def _result(entity_type: str, start: int, end: int, score: float) -> RecognizerResult:
    return RecognizerResult(entity_type=entity_type, start=start, end=end, score=score)


def test_filter_results_applies_the_confidence_threshold() -> None:
    results = [
        _result("PERSON", 0, 4, 0.85),
        _result("PHONE_NUMBER", 10, 22, 0.40),
        _result("URL", 30, 48, 0.50),
    ]
    kept = filter_results(results, "some text", Settings(score_threshold=0.5))
    assert [r.entity_type for r in kept] == ["PERSON", "URL"]
    # the boundary itself is kept: "minimum score" semantics, like Presidio's score_threshold
    assert filter_results(results, "some text", Settings(score_threshold=0.85))[0].score == 0.85


def test_filter_results_removes_ignored_entity_types() -> None:
    text = "Visit https://example.com or write to jane@example.org."
    results = [
        _result("URL", text.index("https://example.com"), text.index("https://example.com") + 18, 0.8),
        _result("EMAIL_ADDRESS", text.index("jane@example.org"), text.index("jane@example.org") + 16, 0.9),
        _result("PERSON", text.index("jane@example.org"), text.index("jane@example.org") + 4, 0.85),
    ]
    settings = Settings(ignored_entity_types=frozenset({"URL"}))
    assert [r.entity_type for r in filter_results(results, text, settings)] == [
        "EMAIL_ADDRESS",
        "PERSON",
    ]


def test_filter_results_allow_list_prevents_redaction_of_matching_spans() -> None:
    text = "Nextcloud GmbH employs Anna Meyer in Berlin."

    def span(value: str) -> tuple[int, int]:
        start = text.index(value)
        return start, start + len(value)

    results = [
        _result("ORGANIZATION", *span("Nextcloud GmbH"), 0.85),
        _result("PERSON", *span("Anna Meyer"), 0.85),
        _result("LOCATION", *span("Berlin"), 0.85),
    ]
    settings = Settings(allow_patterns=(r"^Nextcloud GmbH$", "Berlin"))
    kept = filter_results(results, text, settings)
    # the company name (anchored pattern) and the city are kept as-is; the person is not
    assert [r.entity_type for r in kept] == ["PERSON"]


def test_filter_results_allow_list_matches_a_pattern_containing_the_drifted_span() -> None:
    """A span truncated at a word boundary still hits an unanchored allow-list pattern.

    The recognizer may bound only "Nextcloud" out of "Nextcloud GmbH"; the pattern's match
    in the text contains the drifted span, so the detection is suppressed.
    """
    text = "Nextcloud GmbH employs Anna Meyer."
    person_start = text.index("Anna Meyer")
    results = [
        _result("ORGANIZATION", 0, len("Nextcloud"), 0.85),
        _result("PERSON", person_start, person_start + len("Anna Meyer"), 0.85),
    ]
    settings = Settings(allow_patterns=("Nextcloud GmbH",))
    assert [r.entity_type for r in filter_results(results, text, settings)] == ["PERSON"]


def test_filter_results_allow_list_widens_a_span_cut_mid_word() -> None:
    """A span cutting an allow-listed word in half is widened to word boundaries before matching."""
    text = "Nextcloud GmbH employs Anna Meyer."
    person_start = text.index("Anna Meyer")
    results = [
        _result("ORGANIZATION", 0, len("Nextcloud Gmb"), 0.85),
        _result("PERSON", person_start, person_start + len("Anna Meyer"), 0.85),
    ]
    settings = Settings(allow_patterns=(r"^Nextcloud GmbH$",))
    assert [r.entity_type for r in filter_results(results, text, settings)] == ["PERSON"]


def test_filter_results_preserves_order_and_input() -> None:
    results = [_result("PERSON", 0, 4, 0.9), _result("URL", 5, 9, 0.2)]
    kept = filter_results(results, "text", Settings(score_threshold=0.5))
    assert kept == [results[0]]
    assert len(results) == 2  # the input list is not mutated


# --- declarative admin form (the AppAPI admin surface) -----------------------


def _form():
    from main import admin_settings_form

    return admin_settings_form()


def test_admin_form_fields_are_the_appconfig_keys_with_builtin_defaults() -> None:
    form = _form()
    assert form.id == "detection_settings"
    assert form.section_type == "admin"
    assert form.section_id == "declarative_settings"
    assert [f.id for f in form.fields] == list(ADMIN_CONFIG_KEYS)
    assert [f.default for f in form.fields] == [DEFAULT_SCORE_THRESHOLD, "", ""]


def test_admin_form_threshold_field_is_a_number() -> None:
    from nc_py_api.ex_app.ui.settings import SettingsFieldType

    threshold_field = next(f for f in _form().fields if f.id == ENV_SCORE_THRESHOLD)
    assert threshold_field.type == SettingsFieldType.NUMBER
    text_fields = [f for f in _form().fields if f.id in (ENV_IGNORED_ENTITY_TYPES, ENV_ALLOW_PATTERNS)]
    assert all(f.type == SettingsFieldType.TEXT for f in text_fields)


def test_admin_form_id_survives_the_settings_ui_prefix_strip() -> None:
    """The settings UI submits form.id with the app prefix stripped.

    An id containing the prefix would make every real-UI save address an unregistered
    form (silently dropped).
    """
    assert "text_anonymizer_" not in _form().id


def test_enabled_handler_registers_the_admin_form() -> None:
    from main import ADMIN_SETTINGS_FORM_ID, APP_ENABLED, enabled_handler

    class _FakeTaskProcessing:
        def __init__(self) -> None:
            self.unregistered: list[str] = []

        def register(self, provider, task_type) -> None:
            pass

        def unregister(self, provider_id: str) -> None:
            self.unregistered.append(provider_id)

    class _FakeFilesMenu:
        def register_ex(self, *args, **kwargs) -> None:
            pass

        def unregister(self, name: str) -> None:
            pass

    class _FakeSettingsUI:
        def __init__(self) -> None:
            self.forms: list[object] = []

        def register_form(self, form) -> None:
            self.forms.append(form)

    settings_ui = _FakeSettingsUI()
    nc = SimpleNamespace(
        providers=SimpleNamespace(task_processing=_FakeTaskProcessing()),
        ui=SimpleNamespace(
            files_dropdown_menu=_FakeFilesMenu(),
            settings=settings_ui,
        ),
    )
    try:
        assert enabled_handler(True, nc) == ""
        assert [form.id for form in settings_ui.forms] == [ADMIN_SETTINGS_FORM_ID]
        assert enabled_handler(False, nc) == ""
        assert len(settings_ui.forms) == 1  # disablement keeps the form registered
    finally:
        APP_ENABLED.clear()


def test_info_xml_declares_the_environment_fallbacks() -> None:
    """AppAPI only passes declared variable names into the container.

    Every env knob must therefore be declared, and the display names must match the
    admin form field titles.
    """
    tree = ET.parse(Path(__file__).resolve().parents[1] / "appinfo" / "info.xml")
    variables = tree.getroot().find("external-app").find("environment-variables")
    declared = {v.findtext("name"): v.findtext("display-name") for v in variables.findall("variable")}
    assert set(ADMIN_CONFIG_KEYS) <= set(declared)
    by_id = {f.id: f.title for f in _form().fields}
    for key in ADMIN_CONFIG_KEYS:
        assert declared[key] == by_id[key], key


def test_info_xml_declares_no_variable_the_code_does_not_read() -> None:
    """Every declared variable must be a knob the pipeline resolves; extras are dead UI."""
    tree = ET.parse(Path(__file__).resolve().parents[1] / "appinfo" / "info.xml")
    variables = tree.getroot().find("external-app").find("environment-variables")
    declared = {v.findtext("name") for v in variables.findall("variable")}
    assert declared == set(ADMIN_CONFIG_KEYS)


def test_readme_documents_every_admin_setting() -> None:
    """The README configuration list stays in sync with the settings the form exposes."""
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    for key in ADMIN_CONFIG_KEYS:
        assert key in readme, key
