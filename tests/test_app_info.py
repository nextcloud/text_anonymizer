# SPDX-FileCopyrightText: 2026 Nextcloud GmbH and Nextcloud contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the environment variables that appinfo/info.xml declares to AppAPI."""

import xml.etree.ElementTree as ET
from pathlib import Path

import anonymizer

INFO_XML = Path(__file__).resolve().parents[1] / "appinfo" / "info.xml"


def declared_variables() -> dict:
    """The environment variables AppAPI reads from info.xml, by name."""
    root = ET.parse(INFO_XML).getroot()
    return {
        variable.findtext("name"): variable
        for variable in root.findall("./external-app/environment-variables/variable")
    }


def test_engine_variable_is_declared_for_administrators():
    """The engine choice is a declared variable, so an administrator can set it in AppAPI."""
    variable = declared_variables()[anonymizer.ENGINE_ENV_VAR]

    assert variable.findtext("display-name")
    description = variable.findtext("description")
    assert f"{anonymizer.ENGINE_GLINER} (default)" in description
    assert anonymizer.ENGINE_SPACY in description
