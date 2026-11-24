#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Nextcloud GmbH and Nextcloud contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

r"""Benchmark runner: precision, recall and F-scores of the PII detector against a labeled dataset.

The dataset is JSONL, one ``{"text": ..., "entities": [{"type", "start", "end", "text"}, ...]}``
object per line. Each text is run through the app's own analysis pipeline (the same Presidio
engine and chunking the task types use) and the detected spans are compared with the
annotated ground truth:

* exact match: same entity type and identical offsets
* partial match: same entity type and overlapping spans (preserves credit when the detector
  bounds an entity slightly differently)

Results are reported per entity type and overall, as precision, recall, F1 and F2 (recall
weighted higher, the metric recommended for redaction). ``--threshold``, ``--ignore`` and
``--allow`` apply the same filtering the admin settings apply (see ex_app/lib/config.py), so
configuration changes can be measured before they are deployed; until ``--threshold`` is
given, no score threshold filters, exactly as in the app until the threshold is set.

Usage: ``python evals/run.py [--dataset evals/dataset.jsonl] [--threshold 0.4] [--ignore URL,DATE]
[--allow 'Nextcloud GmbH|Apple Inc\\.'] [--json out.json]``
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "ex_app" / "lib"))

# The public detect/redact API returns formatted text without offsets, but a span-based
# benchmark needs the raw results of the app's real chunked analysis pipeline.
from anonymizer import _analyze, _get_engines  # noqa: E402
from config import Settings, filter_results, parse_allow_patterns, parse_ignored_entity_types  # noqa: E402

DEFAULT_DATASET = Path(__file__).resolve().parent / "dataset.jsonl"
OVERALL = "OVERALL"


@dataclass(frozen=True)
class Entity:
    """One annotated or detected entity span; ``score`` is the Presidio confidence if known."""

    type: str
    start: int
    end: int
    score: float | None = None


@dataclass(frozen=True)
class Sample:
    """One dataset line: a text and its annotated ground-truth entities."""

    text: str
    truth: tuple[Entity, ...]


@dataclass
class Counts:
    """Confusion counts for one entity type (or the overall row)."""

    truth: int = 0
    detected: int = 0
    tp: int = 0
    fp: int = 0
    fn: int = 0


def load_dataset(path: Path) -> list[Sample]:
    """Load and validate a JSONL dataset, raising ``ValueError`` on any malformed record.

    Validation catches annotation rot at benchmark time: spans must be in range, ground-truth
    spans must not overlap, and each annotated span must be present exactly once.
    """
    samples = []
    with path.open(encoding="utf-8") as fp:
        for line_no, line in enumerate(fp, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                text = record["text"]
                entities = [Entity(type=e["type"], start=e["start"], end=e["end"]) for e in record["entities"]]
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                raise ValueError(f"{path}:{line_no}: malformed record: {e}") from e
            previous_end = -1
            for entity in sorted(entities, key=lambda e: e.start):
                if not 0 <= entity.start < entity.end <= len(text):
                    raise ValueError(f"{path}:{line_no}: span {entity} out of range (length {len(text)})")
                if entity.start < previous_end:
                    raise ValueError(f"{path}:{line_no}: overlapping ground-truth spans at {entity.start}")
                previous_end = entity.end
            samples.append(Sample(text=text, truth=tuple(sorted(entities, key=lambda e: e.start))))
    if not samples:
        raise ValueError(f"{path}: no records found")
    return samples


def detect(text: str, settings: Settings) -> list[Entity]:
    """Run the app's analyzer pipeline on ``text`` and apply the configured filters.

    The unfiltered baseline is what the task types produce; a non-default settings
    object goes through the production predicate (config.filter_results).
    """
    analyzer, _ = _get_engines()
    results = _analyze(analyzer, text, "en", None)
    if settings != Settings():
        results = filter_results(results, text, settings)
    return [Entity(type=r.entity_type, start=r.start, end=r.end, score=r.score) for r in results]


def match_exact(detected: list[Entity], truth: list[Entity]) -> tuple[list[Entity], list[Entity]]:
    """Greedy one-to-one exact matching; returns (matched detections, missed truths).

    Detections and truths match when type, start and end are all equal. Duplicate spans on
    one side are consumed one-for-one, so a name detected twice against one annotation
    counts one match plus one false positive.
    """
    remaining = Counter((e.type, e.start, e.end) for e in truth)
    matched = []
    for entity in detected:
        key = (entity.type, entity.start, entity.end)
        if remaining[key] > 0:
            remaining[key] -= 1
            matched.append(entity)
    missed = []
    for entity in truth:
        key = (entity.type, entity.start, entity.end)
        if remaining[key] > 0:
            remaining[key] -= 1
            missed.append(entity)
    return matched, missed


def match_partial(detected: list[Entity], truth: list[Entity]) -> tuple[list[Entity], list[Entity]]:
    """Greedy one-to-one matching by span overlap; returns (matched detections, missed truths).

    A detection and a truth match when they share an entity type and their spans overlap;
    candidate pairs are consumed largest overlap first, so the closest pair wins.
    """
    candidates = sorted(
        (
            max(d.start, t.start) - min(d.end, t.end),
            i,
            j,
        )
        for i, d in enumerate(detected)
        for j, t in enumerate(truth)
        if d.type == t.type and min(d.end, t.end) > max(d.start, t.start)
    )
    used_detected: set[int] = set()
    used_truth: set[int] = set()
    for _, i, j in candidates:
        if i in used_detected or j in used_truth:
            continue
        used_detected.add(i)
        used_truth.add(j)
    matched = [detected[i] for i in sorted(used_detected)]
    missed = [truth[j] for j in range(len(truth)) if j not in used_truth]
    return matched, missed


def precision_recall(counts: Counts) -> tuple[float, float, float, float]:
    """Compute precision, recall, F1 and F2 (recall weighted double) with zero guards."""

    def f_beta(beta: int) -> float:
        denominator = beta**2 * precision + recall
        return (1 + beta**2) * precision * recall / denominator if denominator else 0.0

    precision = counts.tp / (counts.tp + counts.fp) if counts.tp + counts.fp else 0.0
    recall = counts.tp / (counts.tp + counts.fn) if counts.tp + counts.fn else 0.0
    return precision, recall, f_beta(1), f_beta(2)


def accumulate(
    table: dict[str, Counts],
    matcher,
    detected: list[Entity],
    truth: list[Entity],
    sample: Sample,
    examples: dict[str, dict[str, list[str]]],
    example_limit: int,
) -> None:
    """Update the per-type and overall counts; on the exact pass, record mismatch examples.

    Examples are capped per entity type and category; the exact pass is the one that feeds
    them because partial-match mismatches are dominated by boundary drift.
    """
    matched, missed = matcher(detected, list(truth))
    matched_keys = {(e.type, e.start, e.end) for e in matched}
    spurious = [e for e in detected if (e.type, e.start, e.end) not in matched_keys]
    for entity_type in {e.type for e in detected} | {e.type for e in truth}:
        counts = table.setdefault(entity_type, Counts())
        counts.truth += sum(1 for e in truth if e.type == entity_type)
        counts.detected += sum(1 for e in detected if e.type == entity_type)
        counts.tp += sum(1 for e in matched if e.type == entity_type)
        counts.fp += sum(1 for e in spurious if e.type == entity_type)
        counts.fn += sum(1 for e in missed if e.type == entity_type)
        if matcher is match_exact and counts.fn <= example_limit:
            examples["missed (FN)"].setdefault(entity_type, []).extend(
                _context(sample.text, e) for e in missed if e.type == entity_type
            )
        if matcher is match_exact and counts.fp <= example_limit:
            examples["spurious (FP)"].setdefault(entity_type, []).extend(
                _context(sample.text, e) + (f" (score {e.score:.2f})" if e.score is not None else "")
                for e in spurious
                if e.type == entity_type
            )
    overall = table.setdefault(OVERALL, Counts())
    overall.truth += len(truth)
    overall.detected += len(detected)
    overall.tp += len(matched)
    overall.fp += len(spurious)
    overall.fn += len(missed)


def _context(text: str, entity: Entity, width: int = 24) -> str:
    """The entity rendered in its surrounding text, for mismatch examples."""
    left = text[max(0, entity.start - width) : entity.start]
    right = text[entity.end : entity.end + width]
    return f"...{left}[{text[entity.start:entity.end]}]{right}...".replace("\n", " ")


def format_table(title: str, by_type: dict[str, Counts]) -> list[str]:
    """Render one metric table: per-type rows sorted by support, then the overall row."""
    lines = [f"== {title} =="]
    header = f"{'type':<18}{'truth':>7}{'detected':>10}{'TP':>6}{'FP':>6}{'FN':>6}{'P':>8}{'R':>8}{'F1':>8}{'F2':>8}"
    lines.append(header)
    lines.append("-" * len(header))
    rows = [entry for entry in by_type.items() if entry[0] != OVERALL and (entry[1].truth or entry[1].detected)]
    for entity_type, counts in sorted(rows, key=lambda item: (-item[1].truth, item[0])):
        lines.append(_format_row(entity_type, counts))
    lines.append("-" * len(header))
    lines.append(_format_row(OVERALL, by_type[OVERALL]))
    return lines


def _format_row(label: str, counts: Counts) -> str:
    precision, recall, f1, f2 = precision_recall(counts)
    return (
        f"{label:<18}{counts.truth:>7}{counts.detected:>10}{counts.tp:>6}{counts.fp:>6}{counts.fn:>6}"
        f"{precision:>8.3f}{recall:>8.3f}{f1:>8.3f}{f2:>8.3f}"
    )


def run(samples: list[Sample], settings: Settings, example_limit: int) -> dict:
    """Detect every sample and aggregate the metric tables and mismatch examples."""
    by_exact: dict[str, Counts] = {OVERALL: Counts()}
    by_partial: dict[str, Counts] = {OVERALL: Counts()}
    examples: dict[str, dict[str, list[str]]] = {"missed (FN)": {}, "spurious (FP)": {}}
    started = time.perf_counter()
    for sample in samples:
        detected = detect(sample.text, settings)
        accumulate(by_exact, match_exact, detected, list(sample.truth), sample, examples, example_limit)
        accumulate(by_partial, match_partial, detected, list(sample.truth), sample, examples, example_limit)
    return {
        "by_exact": by_exact,
        "by_partial": by_partial,
        "examples": examples,
        "analysis_seconds": time.perf_counter() - started,
    }


def build_settings(args: argparse.Namespace) -> Settings:
    """Resolve CLI flags into Settings through the same parsers the config module uses."""
    settings = Settings()
    if args.threshold is not None:
        if not 0.0 <= args.threshold <= 1.0:
            raise ValueError("--threshold must be between 0.0 and 1.0")
        settings = replace(settings, score_threshold=args.threshold)
    if args.ignore is not None:
        settings = replace(settings, ignored_entity_types=parse_ignored_entity_types(args.ignore))
    if args.allow is not None:
        settings = replace(settings, allow_patterns=tuple(s for s, _ in parse_allow_patterns(args.allow)))
    return settings


def dump_json(path: Path, dataset: Path, samples: list[Sample], settings: Settings, result: dict) -> None:
    """Write the machine-readable report, for CI regression gates and trend tracking."""
    payload = {
        "dataset": str(dataset),
        "samples": len(samples),
        "settings": {
            "score_threshold": settings.score_threshold,
            "ignored_entity_types": sorted(settings.ignored_entity_types),
            "allow_patterns": list(settings.allow_patterns),
        },
        "analysis_seconds": round(result["analysis_seconds"], 3),
        "exact": {k: vars(v) for k, v in result["by_exact"].items()},
        "partial": {k: vars(v) for k, v in result["by_partial"].items()},
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command line surface; every flag is optional so a bare ``python evals/run.py`` works."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="JSONL dataset path")
    parser.add_argument("--threshold", type=float, default=None, help="confidence threshold; no score threshold applies until one is given (the settings form suggests 0.5)")
    parser.add_argument("--ignore", default=None, help="comma-separated entity types to ignore, e.g. URL,DATE")
    parser.add_argument("--allow", default=None, help="newline-separated regex; matches are never redacted")
    parser.add_argument("--examples", type=int, default=3, help="mismatch examples per type (default 3)")
    parser.add_argument("--json", type=Path, default=None, help="write the full result as JSON here")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        settings = build_settings(args)
        samples = load_dataset(args.dataset)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print("Loading the analyzer (the spaCy model load takes a few seconds)...")
    engine_started = time.perf_counter()
    _get_engines()
    engine_seconds = time.perf_counter() - engine_started
    result = run(samples, settings, max(0, args.examples))
    truth_total = sum(len(sample.truth) for sample in samples)
    print()
    print(f"Dataset: {args.dataset} ({len(samples)} texts, {truth_total} annotated entities)")
    print(
        f"Settings: threshold={settings.score_threshold}"
        f" ignored={','.join(sorted(settings.ignored_entity_types)) or '-'}"
        f" allow={settings.allow_patterns or '-'}"
    )
    print(f"Engine load: {engine_seconds:.1f}s, analysis: {result['analysis_seconds']:.1f}s")
    print()
    for line in format_table("Exact span match (type + identical offsets)", result["by_exact"]):
        print(line)
    print()
    for line in format_table("Partial span match (type + any overlap)", result["by_partial"]):
        print(line)
    if args.examples > 0:
        print()
        print("== Mismatch examples (exact match) ==")
        for category, per_type in result["examples"].items():
            for entity_type, entries in sorted(per_type.items()):
                for entry in entries[: args.examples]:
                    print(f"  {category} {entity_type}: {entry}")
    if args.json:
        dump_json(args.json, args.dataset, samples, settings, result)
        print(f"\nJSON report written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
