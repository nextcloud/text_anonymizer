# SPDX-FileCopyrightText: 2026 Nextcloud GmbH and Nextcloud contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""ExApp server providing TaskProcessing tasks and a Files action for detecting and redacting PII."""

import asyncio
import functools
import logging
import os
import posixpath
import re
import threading
from collections.abc import Iterable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from anonymizer import detect_pii, redact_text
from config import (
    ADMIN_CONFIG_KEYS,
    DEFAULT_SCORE_THRESHOLD,
    ENV_ALLOW_PATTERNS,
    ENV_IGNORED_ENTITY_TYPES,
    ENV_SCORE_THRESHOLD,
    Settings,
    apply_admin_config,
)
from fastapi import BackgroundTasks, Depends, FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from nc_py_api import AsyncNextcloudApp, NextcloudApp, NextcloudException
from nc_py_api.ex_app import (
    AppAPIAuthMiddleware,
    LogLvl,
    anc_app,
    run_app,
    set_handlers,
)
from nc_py_api.ex_app.providers.task_processing import (
    ShapeDescriptor,
    ShapeType,
    TaskProcessingProvider,
    TaskType,
)
from nc_py_api.ex_app.ui.settings import SettingsField, SettingsFieldType, SettingsForm
from nc_py_api.files import ActionFileInfo, ActionFileInfoEx

LOGGER = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    set_handlers(app, enabled_handler, trigger_handler=trigger_handler)
    # The environment layer is read here, at startup, so a bad value is reported once and not on the first task.
    _env_settings()
    if NextcloudApp().enabled_state:
        APP_ENABLED.set()
    worker = asyncio.create_task(process_tasks())
    yield
    SHUTDOWN.set()
    TRIGGER.set()
    await worker


APP = FastAPI(lifespan=lifespan)
APP.add_middleware(AppAPIAuthMiddleware)  # set global AppAPI authentication middleware
APP.mount("/img", StaticFiles(directory=Path(__file__).parent.parent / "img"), name="img")

DETECT_PII_TASK_TYPE = TaskType(
    id="text_anonymizer:detect_pii",
    name="Detect sensitive text",
    description="Check whether a text contains personally identifiable information (PII).",
    input_shape=[ShapeDescriptor(name="input", description="Text to check", shape_type=ShapeType.TEXT)],
    output_shape=[
        ShapeDescriptor(
            name="output",
            description="List of PII found in the text, or empty if none was found",
            shape_type=ShapeType.TEXT,
        )
    ],
)
DETECT_PII_PROVIDER = TaskProcessingProvider(
    id=DETECT_PII_TASK_TYPE.id,
    name="TextAnonymizer PII Detector",
    task_type=DETECT_PII_TASK_TYPE.id,
)

REDACT_PII_TASK_TYPE = TaskType(
    id="text_anonymizer:redact_pii",
    name="Redact sensitive text",
    description="Replace all personally identifiable information (PII) in a text with placeholders.",
    input_shape=[ShapeDescriptor(name="input", description="Text to redact", shape_type=ShapeType.TEXT)],
    output_shape=[ShapeDescriptor(name="output", description="Redacted text", shape_type=ShapeType.TEXT)],
)
REDACT_PII_PROVIDER = TaskProcessingProvider(
    id=REDACT_PII_TASK_TYPE.id,
    name="TextAnonymizer PII Redactor",
    task_type=REDACT_PII_TASK_TYPE.id,
)

TASK_PROCESSING_PROVIDERS = [
    (DETECT_PII_PROVIDER, DETECT_PII_TASK_TYPE),
    (REDACT_PII_PROVIDER, REDACT_PII_TASK_TYPE),
]
TASK_HANDLERS = {
    DETECT_PII_TASK_TYPE.id: detect_pii,
    REDACT_PII_TASK_TYPE.id: redact_text,
}

# The declarative settings section AppAPI ships for ExApps ("ExApps Settings").
ADMIN_SETTINGS_SECTION_ID = "declarative_settings"
# The Nextcloud settings UI submits the form id with the app id prefix stripped, so an id
# containing the prefix would make every real-UI save address an unregistered form and the
# saved value would be silently dropped.
ADMIN_SETTINGS_FORM_ID = "detection_settings"


def admin_settings_form() -> SettingsForm:
    """The admin-editable detection settings form (AppAPI declarative settings).

    Field ids are the ``oc_appconfig_ex`` keys the pipeline resolves per task and file
    action, so a value saved here overrides the environment variable of the same name (see
    the config module). Field defaults show the built-in defaults: an environment override
    stays in force until an admin saves a value, and the form cannot see the container
    environment.
    """
    return SettingsForm(
        id=ADMIN_SETTINGS_FORM_ID,
        section_type="admin",
        section_id=ADMIN_SETTINGS_SECTION_ID,
        title="Text Anonymizer",
        description=(
            "Detection settings for every task and file action. A value saved here overrides"
            " the environment variable of the same name and applies to work started after the"
            " save; an empty field keeps the environment value. The defaults shown are the"
            " built-in defaults, not the current environment overrides."
        ),
        fields=[
            SettingsField(
                id=ENV_SCORE_THRESHOLD,
                title="Detection confidence threshold",
                type=SettingsFieldType.NUMBER,
                default=DEFAULT_SCORE_THRESHOLD,
                description=(
                    "Minimum confidence score, from 0.0 to 1.0, for an entity to be reported"
                    " or redacted. Until a value is saved here or set in the environment,"
                    " entities are reported regardless of confidence; the pre-filled 0.5 is"
                    " a suggestion, not an active filter. Recognizer scores cluster in"
                    " coarse bands (checksum-backed recognizers such as IBANs and credit"
                    " cards near 1.0, NER models near 0.85, URLs and phone numbers around"
                    " 0.4 to 0.6), so raising the threshold drops whole entity types at a"
                    " step instead of weakening detections one by one."
                ),
            ),
            SettingsField(
                id=ENV_IGNORED_ENTITY_TYPES,
                title="Ignored entity types",
                type=SettingsFieldType.TEXT,
                default="",
                description=(
                    "Comma-separated entity types that are never reported or redacted, for"
                    " example URL,DATE_TIME. Default: none."
                ),
                placeholder="URL,DATE_TIME",
            ),
            SettingsField(
                id=ENV_ALLOW_PATTERNS,
                title="Allow-list patterns",
                type=SettingsFieldType.TEXT,
                default="",
                description=(
                    "Regular expressions; a detected entity is kept as-is when one matches its"
                    " whole text, or when the entity sits inside a match of one, for example public"
                    " company names. A pattern that is only part of a longer name keeps nothing:"
                    " Paris keeps Paris, not Paris Hilton. The field takes one line, so join several"
                    " patterns with | and put the longest first: Nextcloud GmbH|Nextcloud|Acme"
                    " Corp (not with commas: a comma is part of the expression). A pattern that is"
                    " invalid, that can match nothing (such as (Inc\\.)? or .*) or that takes too"
                    " long on a text is ignored; the ExApp log says which. Default: none."
                ),
                placeholder="Nextcloud GmbH|Acme Corp",
            ),
        ],
    )


@functools.cache
def _env_settings() -> Settings:
    """The environment layer of the detection settings, read once per process: the environment of
    the container does not change while the app runs. An invalid value falls back to the
    default of its setting with a warning, instead of failing every task."""
    return Settings.from_env()


def _resolve_settings(records: Iterable[Any]) -> Settings:
    """Detection settings in force: the environment values with the admin-saved appconfig ``records``
    (what ``appconfig_ex.get_values`` returns) layered on top; the built-in defaults when nothing
    is set, which the pipeline leaves the detections alone for."""
    return apply_admin_config(_env_settings(), {record.key: record.value for record in records})


def _environment_settings(error: Exception) -> Settings:
    """The environment values alone, for when the admin-saved appconfig could not be read."""
    LOGGER.warning("Admin detection settings could not be read, environment values stay in force: %s", error)
    return _env_settings()


async def _resolve_task_settings(nc: AsyncNextcloudApp) -> Settings:
    """Detection settings for one task; a failed server read keeps the environment values."""
    try:
        records = await nc.appconfig_ex.get_values(list(ADMIN_CONFIG_KEYS))
    except Exception as e:
        return _environment_settings(e)
    return _resolve_settings(records)


def _resolve_action_settings(nc: NextcloudApp) -> Settings:
    """Detection settings for a Files action, through the synchronous client; a failed server read keeps the
    environment values."""
    try:
        records = nc.appconfig_ex.get_values(list(ADMIN_CONFIG_KEYS))
    except Exception as e:
        return _environment_settings(e)
    return _resolve_settings(records)


# Set by `enabled_handler`, which runs in a worker thread.
APP_ENABLED = threading.Event()
# Set when Nextcloud announces a new task (Nextcloud 33+) and on shutdown.
TRIGGER = asyncio.Event()
SHUTDOWN = asyncio.Event()
POLL_INTERVAL = 5
# Once Nextcloud has sent a trigger, polling is only a fallback for missed triggers.
POLL_INTERVAL_WITH_TRIGGER = 5 * 60
POLL_INTERVAL_ON_ERROR = 10


async def trigger_handler(_provider_id: str) -> None:
    TRIGGER.set()


async def wait_for_trigger(timeout: float) -> bool:
    """Wait until a trigger arrives or ``timeout`` seconds pass, and return whether a trigger arrived."""
    try:
        await asyncio.wait_for(TRIGGER.wait(), timeout)
    except TimeoutError:
        return False
    return True


async def process_task(nc: AsyncNextcloudApp, task: dict[str, Any]) -> None:
    loop = asyncio.get_running_loop()

    def on_progress(percent: float) -> None:
        # Wait for each update, as Nextcloud marks a task as running again if its progress arrives after its result.
        asyncio.run_coroutine_threadsafe(nc.providers.task_processing.set_progress(task["id"], percent), loop).result()

    try:
        handler = TASK_HANDLERS[task["type"]]
        settings = await _resolve_task_settings(nc)
        # Presidio is CPU-bound, so it runs in a worker thread to keep the event loop responsive.
        output = await asyncio.to_thread(handler, task["input"]["input"], on_progress=on_progress, settings=settings)
    except Exception as e:
        await nc.log(LogLvl.ERROR, f"Failed to process task {task['id']}: {e}")
        await nc.providers.task_processing.report_result(
            task["id"], error_message=str(e), user_facing_error_message="The text could not be processed."
        )
        return
    await nc.providers.task_processing.report_result(task["id"], {"output": output})


async def process_tasks() -> None:
    """Fetch and process tasks for all providers one at a time until shutdown."""
    nc = AsyncNextcloudApp()
    provider_ids = [provider.id for provider, _ in TASK_PROCESSING_PROVIDERS]
    task_types = list(TASK_HANDLERS)
    poll_interval = POLL_INTERVAL
    while not SHUTDOWN.is_set():
        TRIGGER.clear()
        if not APP_ENABLED.is_set():
            await wait_for_trigger(POLL_INTERVAL)
            continue
        try:
            response = await nc.providers.task_processing.next_task(provider_ids, task_types)
            if response:
                await process_task(nc, response["task"])
            elif await wait_for_trigger(poll_interval):
                poll_interval = POLL_INTERVAL_WITH_TRIGGER
        except Exception:
            LOGGER.exception("Failed to fetch or process the next task")
            await wait_for_trigger(POLL_INTERVAL_ON_ERROR)


REDACT_FILE_ACTION_NAME = "redact_pii_file"
REDACT_FILE_ACTION_HANDLER = "/redact_file"
# AppAPI matches by substring, so "text/" matches all text/* types but not e.g. ODT documents.
REDACT_FILE_ACTION_MIME = "text/"
# AppAPI passes an unset default action to the Files app as `null`, which the Files app treats as a default action.
# Ordering after the built-in "Download" action (30) keeps any real default action, e.g. "View", ahead of this one.
REDACT_FILE_ACTION_ORDER = 31
REDACTED_SUFFIX_RE = re.compile(r" \(redacted(?: \d+)?\)$")


def path_exists(nc: NextcloudApp, user_path: str) -> bool:
    try:
        nc.files.by_path(user_path)
    except NextcloudException as e:
        # WebDAV lookups raise a plain NextcloudException for a missing path, not NextcloudExceptionNotFound.
        if e.status_code == 404:
            return False
        raise
    return True


def redacted_copy_path(nc: NextcloudApp, user_path: str) -> str:
    """Return a free path next to ``user_path`` for the redacted copy, e.g. ``notes (redacted).txt``.

    Redacting an existing copy such as ``notes (redacted).txt`` yields ``notes (redacted 2).txt``.
    """
    directory, name = posixpath.split(user_path)
    stem, ext = posixpath.splitext(name)
    stem = REDACTED_SUFFIX_RE.sub("", stem)
    candidate = posixpath.join(directory, f"{stem} (redacted){ext}")
    counter = 2
    while path_exists(nc, candidate):
        candidate = posixpath.join(directory, f"{stem} (redacted {counter}){ext}")
        counter += 1
    return candidate


def redact_file(nc: NextcloudApp, file_info: ActionFileInfo) -> None:
    """Write a redacted copy of the file next to the original and notify the user of the outcome."""
    try:
        node = nc.files.by_id(file_info.fileId)
        if node is None or node.is_dir:
            nc.notifications.create(f"Could not redact {file_info.name}", "The file could not be found.")
            return
        try:
            text = nc.files.download(node).decode("utf-8")
        except UnicodeDecodeError:
            nc.notifications.create(f"Could not redact {node.name}", "The file is not a UTF-8 encoded text file.")
            return
        copy_path = redacted_copy_path(nc, node.user_path)
        redacted = redact_text(text, settings=_resolve_action_settings(nc))
        nc.files.upload(copy_path, redacted)
        nc.notifications.create(f"Redacted copy of {node.name} saved as {posixpath.basename(copy_path)}")
    except Exception as e:
        nc.log(LogLvl.ERROR, f"Failed to redact file {file_info.fileId}: {e}")
        nc.notifications.create(f"Could not redact {file_info.name}", "An unexpected error occurred.")


def redact_files(user_id: str, files: list[ActionFileInfo]) -> None:
    nc = NextcloudApp()
    nc.set_user(user_id)
    for file_info in files:
        redact_file(nc, file_info)


@APP.post(REDACT_FILE_ACTION_HANDLER)
async def redact_file_action(
    files: ActionFileInfoEx,
    nc: Annotated[AsyncNextcloudApp, Depends(anc_app)],
    background_tasks: BackgroundTasks,
):
    # Files are accessed as the user who triggered the action, so Nextcloud enforces their permissions.
    background_tasks.add_task(redact_files, await nc.user, files.files)
    return JSONResponse(content={})


def enabled_handler(enabled: bool, nc: NextcloudApp) -> str:
    # This will be called each time application is `enabled` or `disabled`
    # NOTE: `user` is unavailable on this step, so all NC API calls that require it will fail as unauthorized.
    try:
        if enabled:
            for provider, task_type in TASK_PROCESSING_PROVIDERS:
                nc.providers.task_processing.register(provider, task_type)
            nc.ui.files_dropdown_menu.register_ex(
                REDACT_FILE_ACTION_NAME,
                "Redact text",
                REDACT_FILE_ACTION_HANDLER,
                mime=REDACT_FILE_ACTION_MIME,
                icon="img/icon.svg",
                order=REDACT_FILE_ACTION_ORDER,
            )
            # Registering the form on every enable also refreshes its schema after upgrades
            # (a re-registration edits the existing form in place).
            nc.ui.settings.register_form(admin_settings_form())
            APP_ENABLED.set()
        else:
            APP_ENABLED.clear()
            for provider, _ in TASK_PROCESSING_PROVIDERS:
                nc.providers.task_processing.unregister(provider.id)
            nc.ui.files_dropdown_menu.unregister(REDACT_FILE_ACTION_NAME)
    except Exception as e:
        nc.log(LogLvl.ERROR, f"Failed to {'register' if enabled else 'unregister'} app features: {e}")
        # A non-empty string is shown to the Nextcloud administrator as an error.
        return str(e)
    return ""


if __name__ == "__main__":
    # Wrapper around `uvicorn.run`.
    # You are free to call it directly, with just using the `APP_HOST` and `APP_PORT` variables from the environment.
    os.chdir(Path(__file__).parent)
    run_app("main:APP", log_level="trace")
