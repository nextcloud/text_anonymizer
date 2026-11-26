<!--
SPDX-FileCopyrightText: 2026 Nextcloud GmbH and Nextcloud contributors
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Evaluation harness

Measures the PII detector against a labeled dataset: precision, recall, F1 and F2
(F2 weights recall higher, the recommended metric for redaction) per entity type and
overall, under exact span matching (identical offsets and type) and partial span
matching (same type, any overlap).

```bash
python evals/run.py            # or: make eval
```

Tuning flags apply the same filtering the admin settings apply (see
`ex_app/lib/config.py`), so a configuration change can be measured before it is
deployed:

```bash
python evals/run.py --threshold 0.4 --ignore URL,DATE_TIME --allow 'Nextcloud GmbH|Apple Inc\.'
```

## What is scored

The numbers describe what the app reports. Each text goes through the pipeline of the
task types: the analyzer, the settings given by the flags, then the overlap resolution
the redaction applies (Presidio's anonymizer with the keep operators, as `detect_pii`
calls it). The entities left are compared with the annotations:

- An entity contained in a longer one of another type is dropped, for example a URL
  recognized inside an e-mail address. It is still redacted, as part of the longer
  entity, but an annotation of it counts as a miss of its type.
- Entities of one type that intersect, or that only a space separates, become one span.
- The settings apply before the resolution: ignoring a type or raising the threshold
  brings back the entities that a dropped one contained.

Every occurrence counts, while the detect task lists a value that occurs twice once.
The score shown with a false positive is the confidence Presidio gives the entity after
the resolution; a threshold applies to the analyzer's results before it. The texts are
analyzed as English, as the tasks do. A text where two entities of different types
overlap partly is refused: Presidio then reports offsets that do not address the text.

## Dataset format

`dataset.jsonl` holds one JSON object per line:

```json
{"text": "Call Maria Schmidt at +1 (212) 555-0134.",
 "entities": [{"type": "PERSON", "start": 5, "end": 18, "text": "Maria Schmidt"},
              {"type": "PHONE_NUMBER", "start": 23, "end": 40, "text": "+1 (212) 555-0134"}]}
```

Entity types follow Presidio's names (`PERSON`, `EMAIL_ADDRESS`, `PHONE_NUMBER`,
`LOCATION`, `CREDIT_CARD`, `IBAN_CODE`, `US_SSN`, `URL`, `IP_ADDRESS`, `DATE_TIME`).
The runner validates every line at load time and refuses to run, with
`error: <file>:<line>: ...` and exit code 2, on a record whose fields have the wrong
types (a string or null offset, a text that is not a string), a span out of range or
overlapping another, or a span whose `text` is not the slice between its offsets. The
`text` of an entity is optional: only a span that names it is checked against it.

## Annotation policy

The starter set (30 texts, 38 entities) mixes clear cases with edge cases and
false-positive traps:

- Only explicit calendar dates are annotated as `DATE_TIME` ("March 3, 2025"), not
  weekday or vague temporal phrases ("Friday", "the weekend").
- A full address is one `LOCATION` span, even when the detector splits it into
  street and city. Both matchings are one-to-one, so a split address scores one
  true positive and one false positive: the partial matching recovers the recall,
  not the precision.
- Organization and product names are not annotated: public company names are not
  personal data ("Nextcloud GmbH and Apple Inc. announced ...").
- Long digit strings that fail the Luhn check are not annotated (a 16-digit order
  reference is not a credit card number).
- Fictional values are used throughout: RFC 2606 / example domains, the 555-01xx
  North American number range, Ofcom's +44 20 7946 09xx range, Luhn-valid test card
  numbers and randomly constructed identifiers. The textbook placeholder SSN
  123-45-6789 is deliberately not used: Presidio's recognizer rejects it as a known
  example value.

Adding samples: append lines to `dataset.jsonl`, keeping the offsets exact and naming
each span's `text` (the load-time validation refuses a span that does not slice to
it), or regenerate the file and keep it reviewed like any other source.
