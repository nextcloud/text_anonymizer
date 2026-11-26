# SPDX-FileCopyrightText: 2026 Nextcloud GmbH and Nextcloud contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import main
import pytest
from config import ADMIN_CONFIG_KEYS, ENV_ALLOW_PATTERNS, Settings
from nc_py_api import NextcloudException
from nc_py_api.files import ActionFileInfo


def make_nc(existing_paths=()):
    """Return a mocked ``NextcloudApp`` whose ``files.by_path`` only finds ``existing_paths``."""
    nc = MagicMock()

    def by_path(path):
        if path not in existing_paths:
            raise NextcloudException(404, "Not Found")
        return MagicMock()

    nc.files.by_path.side_effect = by_path
    return nc


def make_file_info(file_id=42, name="notes.txt"):
    return ActionFileInfo(
        fileId=file_id,
        name=name,
        directory="/",
        etag="",
        mime="text/plain",
        fileType="file",
        size=10,
        favorite="false",
        permissions=27,
        mtime=0,
        userId="admin",
    )


@pytest.fixture
def events(monkeypatch):
    """Replace the module-level events, as asyncio events must not be shared between event loops."""
    monkeypatch.setattr(main, "TRIGGER", asyncio.Event())
    monkeypatch.setattr(main, "SHUTDOWN", asyncio.Event())
    monkeypatch.setattr(main, "APP_ENABLED", main.threading.Event())


def test_path_exists():
    nc = make_nc({"/a.txt"})
    assert main.path_exists(nc, "/a.txt")
    assert not main.path_exists(nc, "/b.txt")


def test_path_exists_reraises_other_errors():
    nc = MagicMock()
    nc.files.by_path.side_effect = NextcloudException(500, "Server Error")
    with pytest.raises(NextcloudException):
        main.path_exists(nc, "/a.txt")


@pytest.mark.parametrize(
    ("user_path", "existing", "expected"),
    [
        ("/notes.txt", (), "/notes (redacted).txt"),
        ("/dir/notes.md", (), "/dir/notes (redacted).md"),
        ("/README", (), "/README (redacted)"),
        ("/notes.txt", ("/notes (redacted).txt",), "/notes (redacted 2).txt"),
        (
            "/notes.txt",
            ("/notes (redacted).txt", "/notes (redacted 2).txt"),
            "/notes (redacted 3).txt",
        ),
        ("/notes (redacted).txt", ("/notes (redacted).txt",), "/notes (redacted 2).txt"),
        (
            "/notes (redacted 2).txt",
            ("/notes (redacted).txt", "/notes (redacted 2).txt"),
            "/notes (redacted 3).txt",
        ),
    ],
)
def test_redacted_copy_path(user_path, existing, expected):
    assert main.redacted_copy_path(make_nc(existing), user_path) == expected


def test_redact_file_uploads_redacted_copy(monkeypatch):
    monkeypatch.setattr(main, "redact_text", lambda text, settings: text.replace("John", "<PERSON>"))
    nc = make_nc()
    node = nc.files.by_id.return_value
    node.is_dir = False
    node.name = "notes.txt"
    node.user_path = "/dir/notes.txt"
    nc.files.download.return_value = b"Hi John"

    main.redact_file(nc, make_file_info())

    nc.files.by_id.assert_called_once_with(42)
    nc.files.upload.assert_called_once_with("/dir/notes (redacted).txt", "Hi <PERSON>")
    nc.notifications.create.assert_called_once_with("Redacted copy of notes.txt saved as notes (redacted).txt")


def test_redact_file_hands_the_settings_over_when_nothing_is_set(monkeypatch):
    """With nothing set anywhere the action makes the same call as with settings: what the built-in defaults
    mean is decided in the pipeline, not once more in the action."""
    for key in ADMIN_CONFIG_KEYS:
        monkeypatch.delenv(key, raising=False)
    main._env_settings.cache_clear()
    redact_text = MagicMock(return_value="Hi")
    monkeypatch.setattr(main, "redact_text", redact_text)
    nc = make_nc()
    node = nc.files.by_id.return_value
    node.is_dir = False
    node.name = "notes.txt"
    node.user_path = "/notes.txt"
    nc.files.download.return_value = b"Hi"

    main.redact_file(nc, make_file_info())

    redact_text.assert_called_once_with("Hi", settings=Settings())


@pytest.mark.parametrize("is_dir", [None, True])
def test_redact_file_missing_file(is_dir):
    nc = make_nc()
    if is_dir is None:
        nc.files.by_id.return_value = None
    else:
        nc.files.by_id.return_value.is_dir = True

    main.redact_file(nc, make_file_info())

    nc.files.upload.assert_not_called()
    nc.notifications.create.assert_called_once_with("Could not redact notes.txt", "The file could not be found.")


def test_redact_file_non_utf8():
    nc = make_nc()
    node = nc.files.by_id.return_value
    node.is_dir = False
    node.name = "image.txt"
    nc.files.download.return_value = b"\xff\xfe\x00"

    main.redact_file(nc, make_file_info(name="image.txt"))

    nc.files.upload.assert_not_called()
    nc.notifications.create.assert_called_once_with(
        "Could not redact image.txt", "The file is not a UTF-8 encoded text file."
    )


def test_redact_file_unexpected_error():
    nc = make_nc()
    nc.files.by_id.side_effect = RuntimeError("boom")

    main.redact_file(nc, make_file_info())

    nc.log.assert_called_once()
    nc.notifications.create.assert_called_once_with("Could not redact notes.txt", "An unexpected error occurred.")


def test_enabled_handler_registers_features(events):
    nc = MagicMock()

    assert main.enabled_handler(True, nc) == ""

    registered = [call.args[0].id for call in nc.providers.task_processing.register.call_args_list]
    assert registered == ["text_anonymizer:detect_pii", "text_anonymizer:redact_pii"]
    nc.ui.files_dropdown_menu.register_ex.assert_called_once()
    assert nc.ui.files_dropdown_menu.register_ex.call_args.args[0] == main.REDACT_FILE_ACTION_NAME
    assert main.APP_ENABLED.is_set()


def test_enabled_handler_unregisters_features(events):
    main.APP_ENABLED.set()
    nc = MagicMock()

    assert main.enabled_handler(False, nc) == ""

    unregistered = [call.args[0] for call in nc.providers.task_processing.unregister.call_args_list]
    assert unregistered == ["text_anonymizer:detect_pii", "text_anonymizer:redact_pii"]
    nc.ui.files_dropdown_menu.unregister.assert_called_once_with(main.REDACT_FILE_ACTION_NAME)
    assert not main.APP_ENABLED.is_set()


def test_enabled_handler_returns_error(events):
    nc = MagicMock()
    nc.providers.task_processing.register.side_effect = RuntimeError("boom")

    assert main.enabled_handler(True, nc) == "boom"
    assert not main.APP_ENABLED.is_set()


def test_wait_for_trigger(events):
    async def run():
        assert not await main.wait_for_trigger(0.01)
        main.TRIGGER.set()
        assert await main.wait_for_trigger(10)

    asyncio.run(run())


def make_async_nc(admin_settings=None):
    """Return a mocked ``AsyncNextcloudApp`` whose admin settings read answers ``admin_settings``
    ({key: value}; nothing saved by default)."""
    nc = MagicMock()
    nc.appconfig_ex.get_values = AsyncMock(
        return_value=[SimpleNamespace(key=key, value=value) for key, value in (admin_settings or {}).items()]
    )
    nc.providers.task_processing.set_progress = AsyncMock()
    nc.providers.task_processing.report_result = AsyncMock()
    nc.providers.task_processing.next_task = AsyncMock()
    nc.log = AsyncMock()
    return nc


def test_process_task_reports_output_and_progress(monkeypatch):
    seen = []

    def handler(text, on_progress, settings):
        seen.append(settings)
        on_progress(50.0)
        on_progress(100.0)
        return text.upper()

    monkeypatch.setitem(main.TASK_HANDLERS, "text_anonymizer:redact_pii", handler)
    nc = make_async_nc({ENV_ALLOW_PATTERNS: "Admin Corp"})
    task = {"id": 7, "type": "text_anonymizer:redact_pii", "input": {"input": "hello"}}

    asyncio.run(main.process_task(nc, task))

    assert [call.args for call in nc.providers.task_processing.set_progress.await_args_list] == [(7, 50.0), (7, 100.0)]
    nc.providers.task_processing.report_result.assert_awaited_once_with(7, {"output": "HELLO"})
    # the handler runs with the settings resolved for the task: the admin values over the environment
    assert [resolved.allow_patterns for resolved in seen] == [("Admin Corp",)]


def test_process_task_reports_error(monkeypatch):
    def handler(text, on_progress, settings):
        raise ValueError("bad input")

    monkeypatch.setitem(main.TASK_HANDLERS, "text_anonymizer:detect_pii", handler)
    nc = make_async_nc()
    task = {"id": 7, "type": "text_anonymizer:detect_pii", "input": {"input": "hello"}}

    asyncio.run(main.process_task(nc, task))

    nc.log.assert_awaited_once()
    nc.providers.task_processing.report_result.assert_awaited_once_with(
        7, error_message="bad input", user_facing_error_message="The text could not be processed."
    )


def test_process_tasks_fetches_and_processes_until_shutdown(monkeypatch, events):
    nc = make_async_nc()
    task = {"id": 1, "type": "text_anonymizer:detect_pii", "input": {"input": "hello"}}

    async def next_task(provider_ids, task_types):
        assert provider_ids == ["text_anonymizer:detect_pii", "text_anonymizer:redact_pii"]
        assert task_types == ["text_anonymizer:detect_pii", "text_anonymizer:redact_pii"]
        if nc.providers.task_processing.next_task.await_count == 1:
            return {"task": task}
        main.SHUTDOWN.set()
        main.TRIGGER.set()
        return None

    nc.providers.task_processing.next_task.side_effect = next_task
    process_task = AsyncMock()
    monkeypatch.setattr(main, "AsyncNextcloudApp", lambda: nc)
    monkeypatch.setattr(main, "process_task", process_task)
    main.APP_ENABLED.set()

    asyncio.run(asyncio.wait_for(main.process_tasks(), 5))

    process_task.assert_awaited_once_with(nc, task)
    assert nc.providers.task_processing.next_task.await_count == 2
