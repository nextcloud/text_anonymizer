# SPDX-FileCopyrightText: 2026 Nextcloud GmbH and Nextcloud contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The time bound on allow-list patterns: no administrator's pattern can hold the task worker."""

import inspect
import logging
import re
import time

import pytest
import regex
from presidio_analyzer import RecognizerResult

import config
from config import Settings, filter_results, parse_allow_patterns

LOGGER_NAME = "text_anonymizer.config"


def _result(start, end, score=0.85):
    return RecognizerResult(entity_type="PERSON", start=start, end=end, score=score)


@pytest.mark.parametrize(
    ("pattern", "text"),
    [
        pytest.param(r"(a|aa)+$", "a" * 33 + "b", id="quantified-alternation"),
        pytest.param(r"(?=(a+)+b)", "a" * 30, id="nested-quantifier-in-a-lookahead"),
        pytest.param(r"(a+)+$", "a" * 30 + "b", id="nested-quantifier"),
        pytest.param(r"\s+$", " " * 20000 + "x", id="one-quantifier-quadratic-over-a-long-text"),
    ],
)
def test_no_pattern_holds_the_worker(pattern, text):
    """The shapes that never end answer within the budget and leave the entity as not allow-listed."""
    started = time.perf_counter()
    kept = filter_results([_result(0, len(text))], text, Settings(allow_patterns=(pattern,)))
    assert time.perf_counter() - started < config.ALLOW_PATTERN_TIME_BUDGET * 4
    assert kept == [_result(0, len(text))]


def test_a_pattern_that_runs_out_of_time_is_dropped_whole_with_one_warning(monkeypatch, caplog):
    """No verdict of the pattern survives, and the pattern is warned about once per process."""
    monkeypatch.setattr(config, "ALLOW_PATTERN_TIME_BUDGET", 0.0)
    config._TIMED_OUT_PATTERNS.clear()
    text = "Anna Meyer works in Berlin."
    results = [_result(0, 10), _result(20, 26)]
    settings = Settings(allow_patterns=("Berlin",))
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        for _ in range(2):
            assert filter_results(results, text, settings) == results
    assert caplog.text.count("Berlin") == 1
    assert "Ignoring allow-list pattern 'Berlin'" in caplog.text


def test_a_sound_pattern_is_unaffected_by_the_budget():
    """The budget is never spent on a pattern that names what it keeps."""
    config._TIMED_OUT_PATTERNS.clear()
    text = "Nextcloud GmbH employs Anna Meyer in Berlin."
    results = [_result(0, 14, 0.9), _result(23, 33), _result(37, 43)]
    kept = filter_results(results, text, Settings(allow_patterns=("Berlin", "Nextcloud GmbH")))
    assert kept == [results[1]]
    assert config._TIMED_OUT_PATTERNS == set()


def test_the_patterns_run_under_an_engine_that_bounds_the_time():
    assert isinstance(parse_allow_patterns("Berlin")[0][1], regex.Pattern)


@pytest.mark.parametrize(
    ("pattern", "text"),
    [
        pytest.param("Ann{e<=1}", "Anm", id="errors"),
        pytest.param("Acme{i<=2}", "Acmeee", id="insertions"),
        pytest.param("Nextcloud {d<=1}GmbH", "NextcloudGmbH", id="deletions"),
        pytest.param("a{e}", "x", id="unbounded"),
    ],
)
def test_a_brace_that_is_no_counted_repeat_is_refused(caplog, pattern, text):
    """re reads the brace as a literal; the engine would read a fuzzy match, which keeps texts the pattern does not name."""
    assert re.compile(pattern).fullmatch(text) is None
    assert regex.compile(pattern).fullmatch(text) is not None
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        assert parse_allow_patterns(pattern) == ()
    assert pattern in caplog.text
    assert "write it as '\\{'" in caplog.text


@pytest.mark.parametrize(
    "pattern",
    [
        "a{1,3}",
        "a{2,}",
        "a{,3}",
        "a{,}",
        "a{}",
        r"\{2}",
        "[{]",
        "[{]x",
        r"[\]]{2}",
        "x{2,3}?",
        "(?:ab|cd){2}",
        "Nextcloud GmbH",
    ],
)
def test_counted_repeats_and_braces_in_classes_are_accepted(pattern):
    assert parse_allow_patterns(pattern)[0][0] == pattern


def test_a_pattern_the_engine_refuses_is_ignored_with_a_warning(monkeypatch, caplog):
    """re decides the syntax; should the engine refuse a pattern all the same, it is ignored and never a failure."""

    def refuse(_source):
        raise regex.error("unterminated character set")

    monkeypatch.setattr(config.regex, "compile", refuse)
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        assert parse_allow_patterns("Nextcloud GmbH") == ()
    assert "Ignoring invalid allow-list pattern 'Nextcloud GmbH'" in caplog.text


def test_no_private_module_of_re_is_read():
    source = inspect.getsource(config)
    assert "re._parser" not in source
    assert "_sre_parse" not in source
