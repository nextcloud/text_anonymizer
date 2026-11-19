# SPDX-FileCopyrightText: 2026 Nextcloud GmbH and Nextcloud contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Test configuration: import the ExApp modules the same way the app runtime does."""

import sys
from pathlib import Path

LIB_DIR = Path(__file__).resolve().parents[1] / "ex_app" / "lib"
if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))
