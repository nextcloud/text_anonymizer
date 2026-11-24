# SPDX-FileCopyrightText: 2026 Nextcloud GmbH and Nextcloud contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Evaluation harness tests: dataset validity, matching semantics, metric math and a live run.

The live test needs the spaCy model (en_core_web_lg) and is skipped when it is not
installed, e.g. in lint-only environments; the metric and dataset tests always run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

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


def _require_spacy_model() -> None:
    try:
        import en_core_web_lg  # noqa: F401
    except ImportError:
        pytest.skip("spaCy model en_core_web_lg is not installed")


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
