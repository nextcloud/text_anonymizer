# SPDX-FileCopyrightText: 2026 Nextcloud GmbH and Nextcloud contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The time bound on allow-list patterns: no administrator's pattern can hold the task worker."""

import inspect
import itertools
import logging
import random
import re
import threading
import time
import warnings

import pytest
import regex
from presidio_analyzer import RecognizerResult

import config
from config import Settings, filter_results, parse_allow_patterns

LOGGER_NAME = "text_anonymizer.config"


def _result(start, end, score=0.85):
    return RecognizerResult(entity_type="PERSON", start=start, end=end, score=score)


def _text_with_detections(lines):
    """``lines`` lines of "Anna Meyer grew up in Berlin.": a detection of "Anna Meyer", then one of "Berlin",
    per line, so that ``results[0::2]`` are the ones a pattern Berlin does not keep."""
    text = "Anna Meyer grew up in Berlin.\n" * lines
    return text, [_result(match.start(), match.end()) for match in re.finditer("Anna Meyer|Berlin", text)]


class Around:
    """A pattern for ``_allow_listed`` that runs ``before(name)`` and then answers as the ``inner`` pattern."""

    def __init__(self, inner, before):
        self.inner = inner
        self.before = before
        self.calls = []

    def finditer(self, text, timeout, concurrent):
        self.calls.append("finditer")
        self.before("finditer")
        return self.inner.finditer(text, timeout=timeout, concurrent=concurrent)

    def fullmatch(self, string, timeout, concurrent):
        self.calls.append("fullmatch")
        self.before("fullmatch")
        return self.inner.fullmatch(string, timeout=timeout, concurrent=concurrent)


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
    assert time.perf_counter() - started < config._time_budget(text) * 4
    assert kept == [_result(0, len(text))]


def test_a_pattern_that_runs_out_of_time_is_dropped_whole_and_logged_every_time(monkeypatch, caplog):
    """No verdict of the pattern survives, and every analysis that loses it says so: a skip that is logged
    once per process leaves an administrator who reads the log later without a word about it."""
    monkeypatch.setattr(config, "ALLOW_PATTERN_TIME_BUDGET", 0.0)
    monkeypatch.setattr(config, "ALLOW_PATTERN_SECONDS_PER_CHARACTER", 0.0)
    text = "Anna Meyer works in Berlin."
    results = [_result(0, 10), _result(20, 26)]
    settings = Settings(allow_patterns=("Berlin",))
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        for _ in range(2):
            assert filter_results(results, text, settings) == results
    assert caplog.text.count("Ignoring allow-list pattern 'Berlin'") == 2
    assert "on a text of 27 characters with 2 detections" in caplog.text
    assert "0.00 s" in caplog.text
    assert "backtrack" not in caplog.text


def test_a_sound_pattern_is_unaffected_by_the_budget(caplog):
    """The budget is never spent on a pattern that names what it keeps."""
    text = "Nextcloud GmbH employs Anna Meyer in Berlin."
    results = [_result(0, 14, 0.9), _result(23, 33), _result(37, 43)]
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        kept = filter_results(results, text, Settings(allow_patterns=("Berlin", "Nextcloud GmbH")))
    assert kept == [results[1]]
    assert caplog.text == ""


def test_the_patterns_run_under_an_engine_that_bounds_the_time():
    assert isinstance(parse_allow_patterns("Berlin")[0][1], regex.Pattern)


@pytest.mark.parametrize(
    ("pattern", "text"),
    [
        pytest.param("Ann{e<=1}", "Anm", id="errors"),
        pytest.param("Acme{i<=2}", "Acmeee", id="insertions"),
        pytest.param("Nextcloud {d<=1}GmbH", "NextcloudGmbH", id="deletions"),
        pytest.param("a{e}", "x", id="unbounded"),
        pytest.param("a{e<=0}", "a", id="a-constraint-that-permits-nothing"),
        pytest.param("(?x)a{ e}", "a", id="a-space-in-the-constraint"),
        pytest.param("(?x)a{1, 2}", "a", id="a-spaced-counted-repeat"),
    ],
)
def test_a_brace_that_is_no_counted_repeat_is_a_literal_for_the_engine_too(pattern, text):
    """re reads the brace as a literal. Read by the engine as written, it would be a fuzzy match or a
    repeat, which keeps texts the pattern does not name; so the engine gets the brace escaped."""
    assert re.compile(pattern).fullmatch(text) is None
    assert regex.compile(pattern).fullmatch(text) is not None
    ((source, compiled),) = parse_allow_patterns(pattern)
    assert source == pattern
    assert compiled.fullmatch(text) is None
    plain = re.compile(pattern)
    assert [compiled.fullmatch(letter) is not None for letter in ("a", "b", "x", "{", "}")] == [
        plain.fullmatch(letter) is not None for letter in ("a", "b", "x", "{", "}")
    ]


def _strings():
    """Every string of at most four characters from "ab", and of at most three from "ab{}"."""
    for letters, length in (("ab", 4), ("ab{}", 3)):
        for size in range(length + 1):
            for characters in itertools.product(letters, repeat=size):
                yield "".join(characters)


def _reads_the_same(compiled, plain):
    return all(
        (compiled.fullmatch(string) is not None) == (plain.fullmatch(string) is not None) for string in _strings()
    )


def test_an_accepted_pattern_is_read_by_the_engine_as_re_reads_it():
    """What the engine is given must mean what Python's re says it means, over patterns put together at
    random: braces, escapes, classes, groups, flags, comments, counted repeats and fuzzy constraints."""
    tokens = ["a", "b", r"\N{LATIN SMALL LETTER A}", "[ab]", "[{]", "[^{]", r"\{", "}", "{", "e", "i", "s", "d",
              "<=", "<", "1", "2", ",", "(", ")", "(?:", "(?x)", "(?x:", "(?-x:", "(?#c)", "#", " ", "*", "+", "?",
              "|", "^", "$", r"\b", "\\"]
    generator = random.Random(31)
    accepted = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)  # what re says of a nested set is its own reading
        for _ in range(3000):
            source = "".join(generator.choice(tokens) for _ in range(generator.randrange(1, 7))).strip()
            if not source:
                continue
            try:
                plain = re.compile(source)
            except re.error:
                continue
            parsed = parse_allow_patterns(source)
            if plain.fullmatch("") is not None:
                assert parsed == (), source
                continue
            assert len(parsed) == 1, source
            assert _reads_the_same(parsed[0][1], plain), source
            accepted += 1
    assert accepted > 1000


@pytest.mark.parametrize(
    ("pattern", "reason"),
    [
        pytest.param(r"[[:alpha:]]+", "POSIX class", id="posix-class"),
        pytest.param(r"[^[:digit:]]x", "POSIX class", id="negated-posix-class"),
        pytest.param(r"[a-z[:alpha:]]", "POSIX class", id="posix-class-among-members"),
        pytest.param("(?x) Anna \u3000Marie", "Unicode space", id="verbose-mode"),
        pytest.param("(?x: Acme\u3000GmbH )", "Unicode space", id="scoped-verbose-mode"),
    ],
)
def test_what_the_engine_reads_wider_than_re_is_refused(caplog, pattern, reason):
    """Two constructs mean one thing to the engine and another to re, and no rewriting can mend that."""
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        assert parse_allow_patterns(pattern) == ()
    assert repr(pattern) in caplog.text
    assert reason in caplog.text


@pytest.mark.parametrize(
    "pattern",
    ["[[]", "[a[b]]", "[a&&b]", "[a||b]", "[a~~b]", "[:alpha:]", "Acme\u3000GmbH"],
)
def test_what_both_engines_read_alike_is_accepted(pattern):
    assert parse_allow_patterns(pattern)[0][0] == pattern


@pytest.mark.parametrize(
    "pattern",
    [
        "a{1,3}",
        "a{2,}",
        "a{,3}b",
        "a{,}b",
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


def test_the_budget_grows_with_the_text_and_stops_at_its_cap():
    assert config._time_budget("") == config.ALLOW_PATTERN_TIME_BUDGET
    assert config._time_budget("x" * 100_000) == pytest.approx(config.ALLOW_PATTERN_TIME_BUDGET + 1.0)
    assert config._time_budget("x" * 10_000_000) == config.ALLOW_PATTERN_TIME_CAP


def test_a_sound_pattern_gets_the_time_a_long_text_needs(monkeypatch):
    """A literal needs tens of milliseconds on a long text with many detections: with a budget that does not
    grow with the text it is skipped there and all the names it covers are redacted."""
    monkeypatch.setattr(config, "ALLOW_PATTERN_TIME_BUDGET", 0.0)
    text, results = _text_with_detections(10_000)
    assert filter_results(results, text, Settings(allow_patterns=("Berlin",))) == results[0::2]


def test_a_pattern_of_the_engine_alone_is_refused_by_re(caplog):
    """(?r) searches from the end, so the matches would not come in order; re does not know the flag."""
    assert regex.compile(r"(?r)Berlin").search("Berlin") is not None
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        assert parse_allow_patterns(r"(?r)Berlin") == ()
    assert "Ignoring invalid allow-list pattern '(?r)Berlin'" in caplog.text


def test_the_budget_counts_the_processor_time_of_the_thread_not_the_wall_clock():
    """A file action and a task worker wait on each other for the GIL: waiting is not work. A pattern whose
    matches wait is not skipped, where the wall clock made it run out of its budget."""
    pause = 0.15  # the pass makes three engine calls, and this text has less than three pauses in its budget
    berlin = regex.compile("Berlin")
    around = Around(berlin, before=lambda name: time.sleep(pause))
    text = "Anna Meyer works in Berlin."
    results = [_result(0, 10), _result(20, 26)]
    assert config._time_budget(text) < pause * 3
    assert config._allow_listed(around, results, text) == {1}


def test_a_match_the_engine_stops_for_the_work_of_other_threads_is_started_again():
    """The engine counts the processor time of the whole process, so work that other threads do without the
    GIL can stop a match on which this thread spent little: that match is run again, not treated as slow."""
    berlin = regex.compile("Berlin")
    text = "Anna Meyer works in Berlin."
    results = [_result(0, 10), _result(20, 26)]

    def stop_once(_name):
        if around.calls == ["finditer"]:
            raise TimeoutError

    around = Around(berlin, before=stop_once)
    assert config._allow_listed(around, results, text) == {1}
    assert around.calls.count("finditer") == 2


def test_a_match_that_keeps_stopping_is_given_up_after_a_few_starts():
    def stop_always(_name):
        raise TimeoutError

    around = Around(regex.compile("Berlin"), before=stop_always)
    with pytest.raises(TimeoutError):
        config._allow_listed(around, [_result(20, 26)], "Anna Meyer works in Berlin.")
    assert len(around.calls) == config._ENGINE_ATTEMPTS


def _burn_the_gil(stop):
    while not stop.is_set():
        sum(i * i for i in range(1000))


def test_a_sound_pattern_is_not_skipped_next_to_threads_that_work_at_the_same_time(caplog):
    """The case of the maintainer: a file action and a task worker run CPU-heavy work at the same time. The
    budget was counted on the wall clock, so a sound pattern ran out of it while it waited for the GIL. The
    engine also gives the GIL up during a match and queues to take it back, which stretched the pass over
    many seconds for a few hundredths of a second of work."""
    text, results = _text_with_detections(4000)
    settings = Settings(allow_patterns=("Berlin",))
    assert filter_results(results, text, settings) == results[0::2]
    stop = threading.Event()
    threads = [threading.Thread(target=_burn_the_gil, args=(stop,), daemon=True) for _ in range(3)]
    for thread in threads:
        thread.start()
    try:
        time.sleep(0.1)
        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            started = time.perf_counter()
            assert filter_results(results, text, settings) == results[0::2]
            assert time.perf_counter() - started < 2.0
    finally:
        stop.set()
        for thread in threads:
            thread.join()
    assert caplog.text == ""
