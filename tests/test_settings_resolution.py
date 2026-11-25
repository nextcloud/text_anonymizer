# SPDX-FileCopyrightText: 2026 Nextcloud GmbH and Nextcloud contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""How a task and a file action get their detection settings: the environment layer, the admin values on top."""

import asyncio
import logging
from types import SimpleNamespace

import pytest
from presidio_analyzer import RecognizerResult

import main
from config import (
    ADMIN_CONFIG_KEYS,
    ENV_ALLOW_PATTERNS,
    ENV_IGNORED_ENTITY_TYPES,
    ENV_SCORE_THRESHOLD,
    Settings,
    filter_results,
)


@pytest.fixture(autouse=True)
def environment(monkeypatch):
    """A clean environment layer per test: no TEXT_ANONYMIZER_* variable and nothing remembered from another test."""
    for key in ADMIN_CONFIG_KEYS:
        monkeypatch.delenv(key, raising=False)
    main._env_settings.cache_clear()
    yield
    main._env_settings.cache_clear()


def make_nc(values=None, error=None):
    """A Nextcloud client whose ``appconfig_ex.get_values`` answers ``values`` ({key: value}) or raises ``error``.

    The sync and the async variants share one counter of the calls, so a test can tell whether the server was
    asked.
    """
    calls = []
    values = dict(values or {})

    def answer():
        calls.append(1)
        if error is not None:
            raise error
        return [SimpleNamespace(key=key, value=value) for key, value in values.items()]

    async def answer_async():
        return answer()

    async_appconfig = SimpleNamespace(get_values=lambda keys: answer_async())
    return SimpleNamespace(
        calls=calls,
        appconfig_ex=SimpleNamespace(get_values=lambda keys: answer()),
        async_appconfig_ex=async_appconfig,
    )


def resolve_task(nc):
    return asyncio.run(main._resolve_task_settings(SimpleNamespace(appconfig_ex=nc.async_appconfig_ex)))


def resolve_action(nc):
    return main._resolve_action_settings(SimpleNamespace(appconfig_ex=nc.appconfig_ex))


def test_an_invalid_threshold_in_the_environment_keeps_the_other_environment_settings(monkeypatch):
    """One bad value used to throw out the whole environment layer, here the allow-list next to it."""
    monkeypatch.setenv(ENV_SCORE_THRESHOLD, "0,5")
    monkeypatch.setenv(ENV_ALLOW_PATTERNS, "Jane Doe")
    expected = Settings(allow_patterns=("Jane Doe",))
    assert resolve_task(make_nc()) == expected
    assert resolve_action(make_nc()) == expected

    text = "Jane Doe lives at 10 Downing Street."
    results = [RecognizerResult("PERSON", 0, 8, 0.85), RecognizerResult("LOCATION", 18, 36, 0.85)]
    assert filter_results(results, text, expected) == [results[1]]


def test_a_warning_about_the_environment_is_logged_once_not_on_every_task(monkeypatch, caplog):
    monkeypatch.setenv(ENV_SCORE_THRESHOLD, "0,5")
    monkeypatch.setenv(ENV_ALLOW_PATTERNS, "Jane Doe")
    calls = []
    real_from_env = Settings.from_env.__func__

    def counting_from_env(cls, env=None):
        calls.append(1)
        return real_from_env(cls, env)

    monkeypatch.setattr(Settings, "from_env", classmethod(counting_from_env))
    with caplog.at_level(logging.WARNING, logger="text_anonymizer.config"):
        nc = make_nc()
        for _ in range(3):
            resolve_task(nc)
            resolve_action(nc)
    assert len(calls) == 1
    assert caplog.text.count(ENV_SCORE_THRESHOLD) == 1
    assert len(nc.calls) == 6  # the admin values are still read for every task and every file action


@pytest.mark.parametrize("resolve", [resolve_task, resolve_action], ids=["task", "action"])
def test_admin_values_are_layered_over_the_environment(resolve, monkeypatch):
    monkeypatch.setenv(ENV_ALLOW_PATTERNS, "Env Corp")
    monkeypatch.setenv(ENV_SCORE_THRESHOLD, "0.7")
    nc = make_nc({ENV_ALLOW_PATTERNS: "Admin Corp", ENV_IGNORED_ENTITY_TYPES: "URL"})
    assert resolve(nc) == Settings(
        score_threshold=0.7, ignored_entity_types=frozenset({"URL"}), allow_patterns=("Admin Corp",)
    )
    assert len(nc.calls) == 1


@pytest.mark.parametrize("resolve", [resolve_task, resolve_action], ids=["task", "action"])
def test_a_failed_read_keeps_the_environment_values_with_a_warning(resolve, monkeypatch, caplog):
    monkeypatch.setenv(ENV_ALLOW_PATTERNS, "Env Corp")
    nc = make_nc(error=RuntimeError("server unreachable"))
    with caplog.at_level(logging.WARNING, logger="main"):
        assert resolve(nc) == Settings(allow_patterns=("Env Corp",))
    assert "Admin detection settings could not be read, environment values stay in force: server unreachable" in caplog.text


@pytest.mark.parametrize("resolve", [resolve_task, resolve_action], ids=["task", "action"])
def test_nothing_set_anywhere_resolves_to_none_so_the_pipeline_runs_unfiltered(resolve):
    assert resolve(make_nc()) is None


@pytest.mark.parametrize("resolve", [resolve_task, resolve_action], ids=["task", "action"])
def test_the_task_and_the_file_action_share_one_layering(resolve, monkeypatch):
    """The two resolvers differ by the await of the read only: the layering is one function."""
    seen = []
    layer = main._resolve_settings
    monkeypatch.setattr(main, "_resolve_settings", lambda records: seen.append([r.value for r in records]) or layer(records))
    assert resolve(make_nc({ENV_ALLOW_PATTERNS: "Admin Corp"})) == Settings(allow_patterns=("Admin Corp",))
    assert seen == [["Admin Corp"]]
