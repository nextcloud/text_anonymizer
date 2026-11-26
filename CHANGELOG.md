<!--
SPDX-FileCopyrightText: 2026 Nextcloud GmbH and Nextcloud contributors
SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- "Redact text" and "Redact all text files" skip a file whose redacted copy is up to date, i.e. was made from the
  content the file still has. A copy made by 0.1.0 is not recognised as up to date: the first action on such a file
  writes one more numbered copy next to it, as 0.1.0 did on every action. The app keeps one small record per
  redacted file in Nextcloud's app configuration; uninstalling the app does not remove these records.

## [0.1.0] - 2026-11-10

### Added

- "Detect sensitive text" (`text_anonymizer:detect_pii`) TaskProcessing task type, which lists each distinct PII entity
  found in the input text.
- "Redact sensitive text" (`text_anonymizer:redact_pii`) TaskProcessing task type, which replaces each PII entity in
  the input text with a placeholder naming its type.
- "Redact text" file action for text files, which saves a redacted copy next to the original and notifies the user
  when it is ready.
- PII detection with [Presidio](https://presidio.dataprivacystack.org/) and the `en_core_web_lg` spaCy model.
- Chunked analysis of long texts, with task progress reported to Nextcloud after each chunk.
- Support for Nextcloud 31 to 36.

<!-- Links -->

[0.1.0]: https://github.com/nextcloud/text_anonymizer/releases/tag/v0.1.0
