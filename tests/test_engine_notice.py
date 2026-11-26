# SPDX-FileCopyrightText: 2026 Nextcloud GmbH and Nextcloud contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the report of the detection engine to the Nextcloud log."""

import asyncio
from types import SimpleNamespace

import anonymizer
import main
from nc_py_api.ex_app import LogLvl

FALLBACK = (anonymizer.ENGINE_SPACY, "OSError: the hub is unreachable")
LOADED = (anonymizer.ENGINE_GLINER, None)
CHOSEN = (anonymizer.ENGINE_SPACY, None)


def test_a_failed_load_is_reported_once_and_its_recovery_once(monkeypatch):
    """The administrator hears of a fallback once and of the recovery once, not on every request."""
    monkeypatch.setattr(main, "_LAST_ENGINE_STATE", None)
    monkeypatch.setattr(main, "engine_state", lambda: FALLBACK)

    notice = main.engine_notice()
    assert "The GLiNER model could not be loaded (OSError: the hub is unreachable)" in notice
    assert main.engine_notice() is None

    monkeypatch.setattr(main, "engine_state", lambda: LOADED)
    assert "The GLiNER model is loaded" in main.engine_notice()
    assert main.engine_notice() is None


def test_the_spacy_engine_chosen_by_the_administrator_is_not_reported(monkeypatch):
    """spaCy chosen on purpose is not a fallback."""
    monkeypatch.setattr(main, "_LAST_ENGINE_STATE", None)
    monkeypatch.setattr(main, "engine_state", lambda: CHOSEN)

    assert main.engine_notice() is None


def test_files_action_reports_a_failed_load_to_the_nextcloud_log(monkeypatch):
    monkeypatch.setattr(main, "_LAST_ENGINE_STATE", None)
    monkeypatch.setattr(main, "engine_state", lambda: FALLBACK)
    monkeypatch.setattr(main, "redact_file", lambda nc, file_info: None)
    logged = []
    nc = SimpleNamespace(set_user=lambda user: None, log=lambda level, message: logged.append((level, message)))
    monkeypatch.setattr(main, "NextcloudApp", lambda: nc)

    main.redact_files("admin", [object(), object()])

    assert len(logged) == 1
    assert logged[0][0] == LogLvl.WARNING
    assert "could not be loaded" in logged[0][1]


def test_task_loop_reports_a_failed_load_to_the_nextcloud_log(monkeypatch):
    """The task loop reports the engine after a task, with the nc it already has."""
    monkeypatch.setattr(main, "_LAST_ENGINE_STATE", None)
    monkeypatch.setattr(main, "engine_state", lambda: FALLBACK)
    monkeypatch.setattr(main, "SHUTDOWN", asyncio.Event())
    monkeypatch.setattr(main, "TRIGGER", asyncio.Event())
    main.APP_ENABLED.set()

    async def process_task(nc, task):
        pass

    monkeypatch.setattr(main, "process_task", process_task)

    class FakeNextcloud:
        def __init__(self):
            self.logs = []
            self.served = False
            self.providers = SimpleNamespace(task_processing=self)

        async def next_task(self, provider_ids, task_types):
            if self.served:
                main.SHUTDOWN.set()
                main.TRIGGER.set()
                return None
            self.served = True
            return {"task": {"id": 1}}

        async def log(self, level, message):
            self.logs.append((level, message))

    nc = FakeNextcloud()
    monkeypatch.setattr(main, "AsyncNextcloudApp", lambda: nc)

    asyncio.run(main.process_tasks())

    assert len(nc.logs) == 1
    assert nc.logs[0][0] == LogLvl.WARNING
    assert "could not be loaded" in nc.logs[0][1]
