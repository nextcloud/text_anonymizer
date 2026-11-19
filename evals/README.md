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
python evals/run.py --threshold 0.4 --deny URL,DATE_TIME --allow 'Nextcloud GmbH|Apple Inc\.'
```

## Dataset format

`dataset.jsonl` holds one JSON object per line:

```json
{"text": "Call Maria Schmidt at +1 (212) 555-0134.",
 "entities": [{"type": "PERSON", "start": 5, "end": 18, "text": "Maria Schmidt"},
              {"type": "PHONE_NUMBER", "start": 23, "end": 40, "text": "+1 (212) 555-0134"}]}
```

Entity types follow Presidio's names (`PERSON`, `EMAIL_ADDRESS`, `PHONE_NUMBER`,
`LOCATION`, `CREDIT_CARD`, `IBAN_CODE`, `US_SSN`, `URL`, `IP_ADDRESS`, `DATE_TIME`).
The runner validates every line at load time and refuses to run on malformed,
out-of-range or overlapping annotations.

## Annotation policy

The starter set (30 texts, 38 entities) mixes clear cases with edge cases and
false-positive traps:

- Only explicit calendar dates are annotated as `DATE_TIME` ("March 3, 2025"), not
  weekday or vague temporal phrases ("Friday", "the weekend").
- A full address is one `LOCATION` span, even when the detector splits it into
  street and city; the partial-match table credits such boundary drift.
- Organization and product names are not annotated: public company names are not
  personal data ("Nextcloud GmbH and Apple Inc. announced ...").
- Long digit strings that fail the Luhn check are not annotated (a 16-digit order
  reference is not a credit card number).
- Fictional values are used throughout: RFC 2606 / example domains, the 555-01xx
  North American number range, Ofcom's +44 20 7946 09xx range, Luhn-valid test card
  numbers and randomly constructed identifiers. The textbook placeholder SSN
  123-45-6789 is deliberately not used: Presidio's recognizer rejects it as a known
  example value.

Adding samples: append lines to `dataset.jsonl` keeping spans exact (the load-time
validation will catch mistakes), or regenerate the file and keep it reviewed like
any other source.
