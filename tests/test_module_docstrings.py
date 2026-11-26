# SPDX-FileCopyrightText: 2026 Nextcloud GmbH and Nextcloud contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The module docstrings of the app name what is there."""

import ast
import re
from pathlib import Path

import config

LIB = Path(config.__file__).resolve().parent
ROOT = LIB.parent.parent


def test_a_directory_a_module_docstring_names_holds_code():
    """The docstring of one module named the evaluation harness, a directory of another pull request."""
    for module in sorted(LIB.glob("*.py")):
        docstring = ast.get_docstring(ast.parse(module.read_text(encoding="utf-8"))) or ""
        for name in re.findall(r"\b([A-Za-z_][A-Za-z_0-9]*)/", docstring):
            assert any((ROOT / name).rglob("*.py")), f"{module.name} names the directory {name}/"
