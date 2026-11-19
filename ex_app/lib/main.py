# SPDX-FileCopyrightText: 2026 Nextcloud GmbH and Nextcloud contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""ExApp server providing TaskProcessing tasks and a Files action for detecting and redacting PII."""

import asyncio
import logging
import os
import posixpath
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from anonymizer import detect_pii, redact_text
from batch import BatchReport, oversize_detail, redacted_copy_is_fresh, redact_folder, redacted_copy_path
from fastapi import BackgroundTasks, Depends, FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from nc_py_api import AsyncNextcloudApp, NextcloudApp
from nc_py_api.ex_app import (
    AppAPIAuthMiddleware,
    LogLvl,
    anc_app,
    run_app,
    set_handlers,
    setup_nextcloud_logging,
)
from nc_py_api.ex_app.providers.task_processing import (
    ShapeDescriptor,
    ShapeType,
    TaskProcessingProvider,
    TaskType,
)
from nc_py_api.files import ActionFileInfo, ActionFileInfoEx

LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    """Route application and library logs to nextcloud.log.

    The handler is attached to the root logger, so module loggers such as the batch progress
    logger propagate into it; the root level is lowered to INFO for the same reason, as it
    defaults to WARNING and would drop the per-file progress lines.
    """
    setup_nextcloud_logging()
    logging.getLogger().setLevel(logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    set_handlers(app, enabled_handler, trigger_handler=trigger_handler)
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
        # Presidio is CPU-bound, so it runs in a worker thread to keep the event loop responsive.
        output = await asyncio.to_thread(handler, task["input"]["input"], on_progress=on_progress)
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
REDACT_FOLDER_ACTION_NAME = "redact_pii_folder"
REDACT_FILE_ACTION_HANDLER = "/redact_file"
# AppAPI matches by substring, so "text/" matches all text/* types but not e.g. ODT documents.
REDACT_FILE_ACTION_MIME = "text/"
# Folders report the DAV collection type "httpd/unix-directory", which the substring match catches on "dir";
# no file MIME type contains "dir".
REDACT_FOLDER_ACTION_MIME = "dir"
# AppAPI passes an unset default action to the Files app as `null`, which the Files app treats as a default action.
# Ordering after the built-in "Download" action (30) keeps any real default action, e.g. "View", ahead of this one.
REDACT_FILE_ACTION_ORDER = 31


def redact_file(nc: NextcloudApp, file_info: ActionFileInfo) -> None:
    """Write a redacted copy of the file next to the original and notify the user of the outcome."""
    try:
        node = nc.files.by_id(file_info.fileId)
        if node is None or node.is_dir:
            nc.notifications.create(f"Could not redact {file_info.name}", "The file could not be found.")
            return
        reason = oversize_detail(node.info.size)
        if reason is not None:
            nc.notifications.create(f"Could not redact {node.name}", f"The file is {reason} and was skipped.")
            return
        if redacted_copy_is_fresh(nc, node):
            nc.notifications.create(f"A redacted copy of {node.name} is already up to date")
            return
        try:
            text = nc.files.download(node).decode("utf-8")
        except UnicodeDecodeError:
            nc.notifications.create(f"Could not redact {node.name}", "The file is not a UTF-8 encoded text file.")
            return
        copy_path = redacted_copy_path(nc, node.user_path)
        # The Files action has no language input surface; English is redact_text's default.
        nc.files.upload(copy_path, redact_text(text))
        nc.notifications.create(f"Redacted copy of {node.name} saved as {posixpath.basename(copy_path)}")
    except Exception as e:
        nc.log(LogLvl.ERROR, f"Failed to redact file {file_info.fileId}: {e}")
        nc.notifications.create(f"Could not redact {file_info.name}", "An unexpected error occurred.")


def _log_batch_progress(index: int, total: int, name: str) -> None:
    LOGGER.info("Redacting %s (%d of %d)", name, index, total)


def redact_folder_action(nc: NextcloudApp, dir_info: ActionFileInfo) -> None:
    """Redact every text file under the folder recursively and notify the user with one summary."""
    try:
        node = nc.files.by_id(dir_info.fileId)
        if node is None or not node.is_dir:
            nc.notifications.create(f"Could not redact text files in {dir_info.name}", "The folder could not be found.")
            return
        # A Files action has no task ID, so task progress cannot be reported through TaskProcessing;
        # each file is reported through the progress callback instead. Files actions also have no
        # language input surface, so the batch runs the default English pipeline.
        report = redact_folder(nc, node, on_progress=_log_batch_progress)
        notify_batch_summary(nc, report)
    except Exception as e:
        nc.log(LogLvl.ERROR, f"Failed to redact text files in {dir_info.fileId}: {e}")
        nc.notifications.create(f"Could not redact text files in {dir_info.name}", "An unexpected error occurred.")


def notify_batch_summary(nc: NextcloudApp, report: BatchReport) -> None:
    """Send a single notification summarizing how many files were processed, redacted, failed and skipped."""
    if not report.found_any:
        nc.notifications.create(
            f"No text files in {report.folder_name}",
            f"No redactable text files were found in {report.folder_name}.",
        )
        return
    message = f"{report.processed_count} files processed, {len(report.redacted)} redacted, {len(report.failed)} failed."
    if report.skipped:
        skipped = ", ".join(f"{result.node.name} ({result.detail})" for result in report.skipped)
        message += f" Skipped: {skipped}."
    if report.failed:
        failed = ", ".join(f"{result.node.name} ({result.detail})" for result in report.failed)
        message += f" Failed: {failed}."
    nc.notifications.create(f"Redacted text files in {report.folder_name}", message)


# Files actions run as background tasks, and a double click can dispatch the same action twice
# within about a second; duplicate dispatches are serialized per node, so one pass redacts and
# the repeat hits the fresh-copy guard instead of racing it into duplicate copies.
_ACTION_LOCKS: dict[int, threading.Lock] = {}
_ACTION_LOCKS_GUARD = threading.Lock()


def _action_lock(file_id: int) -> threading.Lock:
    """The process-wide lock serializing redaction actions for one node."""
    with _ACTION_LOCKS_GUARD:
        if file_id not in _ACTION_LOCKS:
            _ACTION_LOCKS[file_id] = threading.Lock()
        return _ACTION_LOCKS[file_id]


def redact_files(user_id: str, files: list[ActionFileInfo]) -> None:
    nc = NextcloudApp()
    nc.set_user(user_id)
    for file_info in files:
        with _action_lock(file_info.fileId):
            # AppAPI's Files payload reports directories as "folder" (current files plugin)
            # or "dir" (older payloads); both dispatch to the folder action.
            if file_info.fileType.lower() in ("dir", "folder"):
                redact_folder_action(nc, file_info)
            else:
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
            nc.ui.files_dropdown_menu.register_ex(
                REDACT_FOLDER_ACTION_NAME,
                "Redact all text files",
                REDACT_FILE_ACTION_HANDLER,
                mime=REDACT_FOLDER_ACTION_MIME,
                icon="img/icon.svg",
                order=REDACT_FILE_ACTION_ORDER,
            )
            APP_ENABLED.set()
        else:
            APP_ENABLED.clear()
            for provider, _ in TASK_PROCESSING_PROVIDERS:
                nc.providers.task_processing.unregister(provider.id)
            nc.ui.files_dropdown_menu.unregister(REDACT_FILE_ACTION_NAME)
            nc.ui.files_dropdown_menu.unregister(REDACT_FOLDER_ACTION_NAME)
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
