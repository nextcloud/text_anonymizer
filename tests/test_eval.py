# SPDX-FileCopyrightText: 2026 Nextcloud GmbH and Nextcloud contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Evaluation harness tests: dataset validity, matching semantics, metric math and a live run.

The live test needs the spaCy model (en_core_web_lg) and is skipped when it is not
installed, e.g. in lint-only environments; the metric and dataset tests always run.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import anonymizer
import pytest
from presidio_analyzer import RecognizerResult
from presidio_anonymizer import AnonymizerEngine

EVALS_DIR = Path(__file__).resolve().parents[1] / "evals"
if str(EVALS_DIR) not in sys.path:
    sys.path.insert(0, str(EVALS_DIR))

import run as eval_run  # noqa: E402
from config import Settings  # noqa: E402

#: The ten entity types the starter dataset must cover.
REQUIRED_TYPES = {
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "LOCATION",
    "CREDIT_CARD",
    "IBAN_CODE",
    "US_SSN",
    "URL",
    "IP_ADDRESS",
    "DATE_TIME",
}


def test_starter_dataset_is_valid_and_covers_all_required_types() -> None:
    samples = eval_run.load_dataset(eval_run.DEFAULT_DATASET)
    assert len(samples) >= 25
    annotated = {entity.type for sample in samples for entity in sample.truth}
    assert annotated >= REQUIRED_TYPES
    traps = [sample for sample in samples if not sample.truth]
    assert len(traps) >= 5, "the set must keep false-positive traps (no-PII texts)"


def test_starter_dataset_spans_slice_to_the_annotated_text() -> None:
    """Every stored span must slice to the exact annotated value (annotation rot guard)."""
    with eval_run.DEFAULT_DATASET.open(encoding="utf-8") as fp:
        records = [json.loads(line) for line in fp if line.strip()]
    for record in records:
        for entity in record["entities"]:
            assert record["text"][entity["start"] : entity["end"]] == entity["text"]


@pytest.mark.parametrize(
    ("line", "reason"),
    [
        ('{"text": "abc", "entities": [{"type": "X", "start": 0, "end": 99}]}', "span out of range"),
        ('{"text": "abc", "entities": [{"type": "X", "start": 2, "end": 1}]}', "start after end"),
        (
            '{"text": "abcd", "entities": [{"type": "X", "start": 0, "end": 3},'
            ' {"type": "Y", "start": 2, "end": 4}]}',
            "overlapping spans",
        ),
        ('{"entities": []}', "missing text key"),
        ("not json", "malformed json"),
    ],
)
def test_malformed_records_are_rejected(tmp_path: Path, line: str, reason: str) -> None:
    dataset = tmp_path / "bad.jsonl"
    dataset.write_text(line + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        eval_run.load_dataset(dataset)


def test_empty_dataset_is_rejected(tmp_path: Path) -> None:
    dataset = tmp_path / "empty.jsonl"
    dataset.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no records"):
        eval_run.load_dataset(dataset)


GOOD_RECORD = {
    "text": "Call Maria Schmidt.",
    "entities": [{"type": "PERSON", "start": 5, "end": 18, "text": "Maria Schmidt"}],
}
GOOD_LINE = json.dumps(GOOD_RECORD)


def _run_main(tmp_path: Path, capsys, record: object) -> tuple[int, str]:
    """``main`` on a dataset of one valid line and ``record``: its exit code and its stderr."""
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(GOOD_LINE + "\n" + json.dumps(record) + "\n", encoding="utf-8")
    code = eval_run.main(["--dataset", str(dataset)])
    return code, capsys.readouterr().err


BAD_RECORDS = [
    ("string-start", {"text": "abc", "entities": [{"type": "X", "start": "0", "end": 3}]}),
    ("string-end", {"text": "abc", "entities": [{"type": "X", "start": 0, "end": "3"}]}),
    ("null-start", {"text": "abc", "entities": [{"type": "X", "start": None, "end": 3}]}),
    ("null-end", {"text": "abc", "entities": [{"type": "X", "start": 0, "end": None}]}),
    ("float-offsets", {"text": "abc", "entities": [{"type": "X", "start": 0.0, "end": 3.0}]}),
    ("boolean-offset", {"text": "abc", "entities": [{"type": "X", "start": True, "end": 3}]}),
    ("text-a-number", {"text": 12345, "entities": []}),
    ("text-null", {"text": None, "entities": []}),
    ("text-a-list", {"text": ["abc"], "entities": []}),
    ("type-a-number", {"text": "abc", "entities": [{"type": 5, "start": 0, "end": 3}]}),
    ("type-empty", {"text": "abc", "entities": [{"type": "", "start": 0, "end": 3}]}),
    ("entities-a-dict", {"text": "abc", "entities": {}}),
    ("entity-a-string", {"text": "abc", "entities": ["X"]}),
]


@pytest.mark.parametrize("record", [record for _, record in BAD_RECORDS], ids=[name for name, _ in BAD_RECORDS])
def test_a_field_of_the_wrong_type_stops_the_run_with_a_clean_error(tmp_path: Path, capsys, record) -> None:
    """A string or null offset, a text that is not a string: a message naming the record, exit code 2."""
    code, error = _run_main(tmp_path, capsys, record)

    assert code == 2
    assert error.startswith(f"error: {tmp_path / 'dataset.jsonl'}:2: ")
    assert "Traceback" not in error


@pytest.mark.parametrize("shift", [-1, 1], ids=["one-to-the-left", "one-to-the-right"])
def test_a_span_that_is_not_its_annotated_text_stops_the_run(tmp_path: Path, capsys, shift: int) -> None:
    """Offsets shifted by one character are refused: the message names the record and the two texts."""
    shifted = {"type": "PERSON", "start": 5 + shift, "end": 18 + shift, "text": "Maria Schmidt"}

    code, error = _run_main(tmp_path, capsys, {"text": "Call Maria Schmidt.", "entities": [shifted]})

    assert code == 2
    start = f"error: {tmp_path / 'dataset.jsonl'}:2: entity 0: span PERSON {5 + shift}-{18 + shift} is "
    assert error.startswith(start)
    assert error.endswith(", not its annotated text 'Maria Schmidt'\n")


def test_a_span_without_annotated_text_loads(tmp_path: Path) -> None:
    """Only a span that names its text is checked against it."""
    dataset = tmp_path / "dataset.jsonl"
    span = {"type": "PERSON", "start": 5, "end": 18}
    dataset.write_text(json.dumps({"text": "Call Maria Schmidt.", "entities": [span]}) + "\n", encoding="utf-8")

    assert eval_run.load_dataset(dataset) == [
        eval_run.Sample("Call Maria Schmidt.", (eval_run.Entity("PERSON", 5, 18),))
    ]


def test_exact_matching_counts_duplicates_one_for_one() -> None:
    truth = [eval_run.Entity("PERSON", 0, 4), eval_run.Entity("URL", 10, 14)]
    detected = [
        eval_run.Entity("PERSON", 0, 4),
        eval_run.Entity("PERSON", 0, 4),  # same span twice: one match, one false positive
        eval_run.Entity("PERSON", 20, 24),  # no matching truth: false positive
    ]
    matched, missed = eval_run.match_exact(detected, truth)
    assert matched == [eval_run.Entity("PERSON", 0, 4)]
    assert missed == [eval_run.Entity("URL", 10, 14)]


def test_partial_matching_prefers_the_largest_overlap() -> None:
    truth = [eval_run.Entity("LOCATION", 10, 30)]
    detected = [eval_run.Entity("LOCATION", 28, 40), eval_run.Entity("LOCATION", 10, 18)]
    matched, missed = eval_run.match_partial(detected, truth)
    assert matched == [eval_run.Entity("LOCATION", 10, 18)]
    assert missed == []
    # the wide-overlap detection was consumed by the only truth; a different type never matches
    matched, missed = eval_run.match_partial(detected, [eval_run.Entity("PERSON", 10, 30)])
    assert matched == []
    assert missed == [eval_run.Entity("PERSON", 10, 30)]


def _score(matcher, text: str, detected: list[eval_run.Entity], truth: list[eval_run.Entity]) -> eval_run.Counts:
    """The overall counts of one sample, as a run collects them."""
    table = {eval_run.OVERALL: eval_run.Counts()}
    examples = {"missed (FN)": {}, "spurious (FP)": {}}
    eval_run.accumulate(table, matcher, detected, truth, eval_run.Sample(text, tuple(truth)), examples, 3)
    return table[eval_run.OVERALL]


@pytest.mark.parametrize("matcher", [eval_run.match_exact, eval_run.match_partial])
def test_every_detection_is_a_hit_or_a_false_positive(matcher) -> None:
    """A second copy of a matched span is consumed by no annotation: it is a false positive."""
    text = "I am John Smith."
    truth = [eval_run.Entity("PERSON", 5, 15)]
    detected = [
        eval_run.Entity("PERSON", 5, 15, 0.9),
        eval_run.Entity("PERSON", 5, 15, 0.9),  # the same span a second time
        eval_run.Entity("LOCATION", 0, 3, 0.4),  # no annotation of that type
    ]

    counts = _score(matcher, text, detected, truth)

    assert (counts.detected, counts.tp, counts.fp, counts.fn) == (3, 1, 2, 0)


def test_an_address_split_into_street_and_city_scores_one_hit_and_one_false_positive() -> None:
    """The annotation policy of the README: both matchings are one-to-one."""
    text = "The package should be delivered to 221B Baker Street, London before the weekend."
    truth = [eval_run.Entity("LOCATION", 35, 60)]
    split = [eval_run.Entity("LOCATION", 35, 52, 0.85), eval_run.Entity("LOCATION", 54, 60, 0.85)]

    exact = _score(eval_run.match_exact, text, split, truth)
    partial = _score(eval_run.match_partial, text, split, truth)

    assert (exact.tp, exact.fp, exact.fn) == (0, 2, 1)
    assert (partial.tp, partial.fp, partial.fn) == (1, 1, 0)


FOUR_NAMES = "Anna Berta Carla Dora"
FOUR_SPANS = ((0, 4), (5, 10), (11, 16), (17, 21))


def _recorded_examples(samples: list[tuple[str, list, list]], limit: int) -> dict[str, dict[str, list[str]]]:
    """The examples the exact pass records for ``(text, truth, detected)`` triples, in order."""
    table = {eval_run.OVERALL: eval_run.Counts()}
    examples = {"missed (FN)": {}, "spurious (FP)": {}}
    for text, truth, detected in samples:
        sample = eval_run.Sample(text, tuple(truth))
        eval_run.accumulate(table, eval_run.match_exact, detected, truth, sample, examples, limit)
    return examples


def _bracketed(entries: list[str]) -> list[str]:
    return [entry.partition("[")[2].partition("]")[0] for entry in entries]


@pytest.mark.parametrize("limit", [1, 3])
def test_examples_of_one_sample_are_capped_not_lost(limit: int) -> None:
    """A sample with four misses, or four spurious detections, still yields the first ones up to the limit."""
    four = [eval_run.Entity("PERSON", start, end) for start, end in FOUR_SPANS]

    missed = _recorded_examples([(FOUR_NAMES, four, [])], limit)["missed (FN)"]["PERSON"]
    spurious = _recorded_examples([(FOUR_NAMES, [], four)], limit)["spurious (FP)"]["PERSON"]

    assert _bracketed(missed) == ["Anna", "Berta", "Carla", "Dora"][:limit]
    assert _bracketed(spurious) == ["Anna", "Berta", "Carla", "Dora"][:limit]


def test_examples_of_a_later_sample_use_the_room_the_earlier_ones_left() -> None:
    four = [eval_run.Entity("PERSON", start, end) for start, end in FOUR_SPANS]
    later = [eval_run.Entity("PERSON", start, end) for start, end in ((0, 5), (6, 11), (12, 18), (19, 24))]

    samples = [(FOUR_NAMES, four[:2], []), ("Erich Frank Gustav Heinz", later, [])]
    missed = _recorded_examples(samples, 3)["missed (FN)"]["PERSON"]

    assert _bracketed(missed) == ["Anna", "Berta", "Erich"]


def test_main_prints_the_examples_up_to_the_limit(monkeypatch, tmp_path: Path, capsys) -> None:
    """``--examples 1`` on a sample with four misses prints one example of them, not none."""
    monkeypatch.setattr(eval_run, "_get_engines", lambda: (None, None))
    monkeypatch.setattr(eval_run, "detect", lambda text, settings: [])
    dataset = tmp_path / "four.jsonl"
    entities = [{"type": "PERSON", "start": start, "end": end} for start, end in FOUR_SPANS]
    dataset.write_text(json.dumps({"text": FOUR_NAMES, "entities": entities}) + "\n", encoding="utf-8")

    assert eval_run.main(["--dataset", str(dataset), "--examples", "1"]) == 0

    printed = [line for line in capsys.readouterr().out.splitlines() if line.startswith("  missed (FN)")]
    assert printed == ["  missed (FN) PERSON: ...[Anna] Berta Carla Dora..."]


def test_precision_recall_favors_recall_in_f2() -> None:
    counts = eval_run.Counts(truth=3, detected=4, tp=2, fp=2, fn=1)
    precision, recall, f1, f2 = eval_run.precision_recall(counts)
    assert precision == pytest.approx(0.5)
    assert recall == pytest.approx(2 / 3)
    assert f1 == pytest.approx(4 / 7)
    assert f2 == pytest.approx(5 * 0.5 * (2 / 3) / (4 * 0.5 + 2 / 3))
    assert f2 > f1  # recall-weighted


def test_precision_recall_zero_guards() -> None:
    for precision, recall, f1, f2 in (
        eval_run.precision_recall(eval_run.Counts()),
        eval_run.precision_recall(eval_run.Counts(truth=2, detected=3)),
    ):
        assert (precision, recall, f1, f2) == (0.0, 0.0, 0.0, 0.0)


def test_build_settings_resolves_flags_through_the_config_parsers() -> None:
    assert eval_run.build_settings(eval_run.parse_args([])) == Settings()
    args = eval_run.parse_args(["--threshold", "0.7", "--ignore", "url, date", "--allow", "Nextcloud GmbH"])
    settings = eval_run.build_settings(args)
    assert settings.score_threshold == 0.7
    assert settings.ignored_entity_types == frozenset({"URL", "DATE"})
    assert settings.allow_patterns == ("Nextcloud GmbH",)


def test_build_settings_rejects_out_of_range_threshold() -> None:
    args = eval_run.parse_args(["--threshold", "1.5"])
    with pytest.raises(ValueError, match="threshold"):
        eval_run.build_settings(args)


class FakeAnalyzer:
    """Reports every match of the given ``(entity_type, pattern, score)`` rules."""

    def __init__(self, *rules: tuple[str, str, float]) -> None:
        self.rules = rules

    def analyze(self, text: str, language: str) -> list[RecognizerResult]:
        assert language == "en"
        return [
            RecognizerResult(entity_type, match.start(), match.end(), score)
            for entity_type, pattern, score in self.rules
            for match in re.finditer(pattern, text)
        ]


def _use_fake_analyzer(monkeypatch, *rules: tuple[str, str, float]) -> None:
    """Give the benchmark and the app the same fake analyzer, with a real anonymizer."""
    engines = (FakeAnalyzer(*rules), AnonymizerEngine())
    monkeypatch.setattr(eval_run, "_get_engines", lambda: engines)
    monkeypatch.setattr(anonymizer, "_get_engines", lambda: engines)


MAIL_RULES = (
    ("EMAIL_ADDRESS", r"[\w.]+@example\.com", 1.0),
    ("URL", r"example\.com", 0.5),
    ("PERSON", r"Jane", 0.85),
)
MAIL_TEXT = "Mail Jane.Doe@example.com from Jane."


def test_detect_scores_the_entities_the_app_reports(monkeypatch) -> None:
    """A detection the app's overlap resolution drops, such as a URL inside an e-mail address, is not scored."""
    _use_fake_analyzer(monkeypatch, *MAIL_RULES)

    detected = eval_run.detect(MAIL_TEXT, Settings())

    assert [(e.type, MAIL_TEXT[e.start : e.end], e.score) for e in detected] == [
        ("EMAIL_ADDRESS", "Jane.Doe@example.com", 1.0),
        ("PERSON", "Jane", 0.85),
    ]
    assert anonymizer.detect_pii(MAIL_TEXT) == "EMAIL_ADDRESS: Jane.Doe@example.com\nPERSON: Jane"


def test_detect_applies_the_settings_before_the_overlap_resolution(monkeypatch) -> None:
    """As in the app: an ignored e-mail address leaves visible the entities that were hidden inside it."""
    _use_fake_analyzer(monkeypatch, *MAIL_RULES)
    settings = Settings(ignored_entity_types=frozenset({"EMAIL_ADDRESS"}))

    detected = eval_run.detect(MAIL_TEXT, settings)

    assert [(e.type, MAIL_TEXT[e.start : e.end]) for e in detected] == [
        ("PERSON", "Jane"),
        ("URL", "example.com"),
        ("PERSON", "Jane"),
    ]
    assert anonymizer.detect_pii(MAIL_TEXT, settings=settings) == "PERSON: Jane\nURL: example.com"


def test_detect_merges_entities_of_one_type_as_the_app_does(monkeypatch) -> None:
    """Intersecting entities of one type and neighbours of one type separated by spaces become one span."""
    _use_fake_analyzer(
        monkeypatch,
        ("PERSON", r"John Sm", 0.85),
        ("PERSON", r"Smith", 0.4),
        ("LOCATION", r"New", 0.6),
        ("LOCATION", r"York", 0.9),
    )
    text = "John Smith lives in New York."

    detected = eval_run.detect(text, Settings())

    assert [(e.type, text[e.start : e.end]) for e in detected] == [("PERSON", "John Smith"), ("LOCATION", "New York")]


def test_detect_refuses_a_text_where_two_entity_types_overlap_partly(monkeypatch) -> None:
    """After a partial overlap of two types, the offsets Presidio reports do not address the text."""
    _use_fake_analyzer(
        monkeypatch,
        ("PERSON", r"Anna Berlin", 0.85),
        ("LOCATION", r"Berlin Street", 0.6),
        ("EMAIL_ADDRESS", r"a@b\.com", 1.0),
    )

    with pytest.raises(ValueError, match="overlap partly"):
        eval_run.detect("Anna Berlin Street, mail a@b.com", Settings())


def test_detect_filters_the_results_through_the_helper_of_the_app(monkeypatch) -> None:
    """The settings apply through ``anonymizer._apply_settings``, the helper the task types use."""
    _use_fake_analyzer(monkeypatch, *MAIL_RULES)
    applied = []

    def spy(_results, _text, settings):
        applied.append(settings)
        return []

    monkeypatch.setattr(eval_run, "_apply_settings", spy)
    settings = Settings(score_threshold=0.5)

    assert eval_run.detect(MAIL_TEXT, settings) == []
    assert applied == [settings]


def test_the_runner_keeps_the_line_limit_of_the_repository() -> None:
    """CI lints ex_app/lib only, so the limit of 120 characters is checked here for the runner."""
    lines = (EVALS_DIR / "run.py").read_text(encoding="utf-8").splitlines()
    assert [number for number, line in enumerate(lines, 1) if len(line) > 120] == []


def _require_spacy_model() -> None:
    try:
        import en_core_web_lg  # noqa: F401
    except ImportError:
        pytest.skip("spaCy model en_core_web_lg is not installed")


@pytest.mark.spacy
def test_live_run_on_the_starter_dataset_produces_metrics() -> None:
    """The harness end to end: real analyzer, real chunking, numbers out."""
    _require_spacy_model()
    samples = eval_run.load_dataset(eval_run.DEFAULT_DATASET)
    result = eval_run.run(samples, Settings(), 0)
    truth_total = sum(len(sample.truth) for sample in samples)
    overall = result["by_exact"][eval_run.OVERALL]
    assert overall.truth == truth_total
    assert overall.detected > 0
    assert 0 < overall.tp <= min(overall.detected, overall.truth)
    for counts in result["by_exact"].values():
        precision, recall, f1, f2 = eval_run.precision_recall(counts)
        assert all(0.0 <= value <= 1.0 for value in (precision, recall, f1, f2))
    # partial match can only equal or beat exact match on the same detections
    assert result["by_partial"][eval_run.OVERALL].tp >= overall.tp


@pytest.mark.spacy
def test_live_detection_applies_the_admin_filter_path() -> None:
    """A tuned settings object must flow through the same filter the admin knobs use."""
    _require_spacy_model()
    samples = eval_run.load_dataset(eval_run.DEFAULT_DATASET)
    sensitive = next(sample for sample in samples if any(e.type == "CREDIT_CARD" for e in sample.truth))
    baseline = eval_run.detect(sensitive.text, Settings())
    assert any(e.type == "CREDIT_CARD" for e in baseline)
    ignored = eval_run.detect(sensitive.text, Settings(ignored_entity_types=frozenset({"CREDIT_CARD"})))
    assert not any(e.type == "CREDIT_CARD" for e in ignored)
    assert len(ignored) < len(baseline)


@pytest.mark.spacy
def test_live_detect_scores_what_the_detect_task_lists() -> None:
    """On every sample of the dataset, the benchmark scores the entities the detect task lists."""
    _require_spacy_model()
    for sample in eval_run.load_dataset(eval_run.DEFAULT_DATASET):
        detected = eval_run.detect(sample.text, Settings())
        scored = {f"{entity.type}: {sample.text[entity.start : entity.end]}" for entity in detected}
        assert scored == set(anonymizer.detect_pii(sample.text).splitlines())
