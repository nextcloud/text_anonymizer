# SPDX-FileCopyrightText: 2026 Nextcloud GmbH and Nextcloud contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the download of the GLiNER model while AppAPI initialises the app."""

import asyncio
import json
import logging
import os
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import anonymizer
import main
import pytest
from fastapi import BackgroundTasks

TOKENIZER = "microsoft/mdeberta-v3-base"


class FakeNC:
    def __init__(self):
        self.status = []

    def set_init_status(self, progress, error=""):
        self.status.append((progress, error))


@pytest.fixture
def downloads(monkeypatch, tmp_path):
    """Record what download_model fetches, and where, instead of the network.

    The storage starts empty. With ``storage_model_complete`` the model and its tokenizer are complete there,
    so a fetch that may read the storage only (``local_files_only``) succeeds without one.
    """
    calls = SimpleNamespace(
        snapshot=[], tokenizer=[], storage=str(tmp_path / "nc_app"), storage_model_complete=False
    )
    monkeypatch.setattr(anonymizer, "model_cache_dir", lambda: calls.storage, raising=False)

    def missing_from_storage(kwargs):
        return kwargs.get("local_files_only") and not calls.storage_model_complete

    def snapshot_download(model, **kwargs):
        calls.snapshot.append((model, kwargs))
        if missing_from_storage(kwargs):
            raise FileNotFoundError("LocalEntryNotFoundError: the model is not in the storage")
        model_dir = tmp_path / "model"
        model_dir.mkdir(exist_ok=True)
        (model_dir / "gliner_config.json").write_text(json.dumps({"model_name": TOKENIZER}), encoding="utf-8")
        return model_dir

    def from_pretrained(cls, model, **kwargs):
        calls.tokenizer.append((model, kwargs))
        if missing_from_storage(kwargs):
            raise OSError("the tokenizer is not in the storage")
        return None

    monkeypatch.setattr(anonymizer, "snapshot_download", snapshot_download, raising=False)
    monkeypatch.setattr(anonymizer.AutoTokenizer, "from_pretrained", classmethod(from_pretrained), raising=False)
    return calls


def test_model_and_tokenizer_are_downloaded_into_the_persistent_storage(monkeypatch, downloads):
    """Both repositories of the model go to the app's persistent storage, which survives updates."""
    progress = []

    assert anonymizer.download_model(progress.append) is True
    (download,) = [options for _, options in downloads.snapshot if not options.get("local_files_only")]
    assert download["cache_dir"] == downloads.storage
    assert downloads.tokenizer[-1] == (TOKENIZER, {"cache_dir": downloads.storage})
    # The progress bar class reports its percentage, as huggingface_hub drives it.
    with open(os.devnull, "w") as sink:
        bar = download["tqdm_class"](total=100, file=sink)
        bar.n = 50
        bar.display()
        bar.n = 100
        bar.display()
    assert progress == [0.0, 50.0, 100.0]


def test_nothing_is_downloaded_for_the_spacy_engine(monkeypatch, downloads):
    monkeypatch.setenv(anonymizer.ENGINE_ENV_VAR, anonymizer.ENGINE_SPACY)

    assert anonymizer.download_model(lambda percent: None) is False
    assert downloads.snapshot == []
    assert downloads.tokenizer == []


def test_model_and_tokenizer_in_the_storage_are_not_downloaded_again(monkeypatch, downloads):
    """A start or an update with the model in place costs no download: the storage is checked without the Hub.

    Only a model or a tokenizer that cannot be loaded from the storage alone is fetched.
    """
    checked = (TOKENIZER, {"cache_dir": downloads.storage, "local_files_only": True})
    downloads.storage_model_complete = True

    assert anonymizer.download_model(lambda percent: None) is True
    assert [options.get("local_files_only") for _, options in downloads.snapshot] == [True]
    assert downloads.tokenizer == [checked]

    downloads.storage_model_complete = False
    assert anonymizer.download_model(lambda percent: None) is True
    assert [options.get("local_files_only") for _, options in downloads.snapshot] == [True, True, None]
    assert downloads.tokenizer == [checked, (TOKENIZER, {"cache_dir": downloads.storage})]


def test_init_downloads_the_model_and_then_completes(monkeypatch):
    """AppAPI sees the progress of the download and enables the app when it is done."""
    nc = FakeNC()

    def download_model(on_progress):
        on_progress(50)
        on_progress(100)

    monkeypatch.setattr(main, "download_model", download_model)
    main.download_model_task(nc)

    assert nc.status == [(50, ""), (99, ""), (100, "")]


def test_init_completes_even_when_the_download_fails(monkeypatch, caplog):
    """A Hub that cannot be reached must not make the installation fail."""
    nc = FakeNC()

    def download_model(on_progress):
        raise OSError("the hub is unreachable")

    monkeypatch.setattr(main, "download_model", download_model)
    with caplog.at_level(logging.WARNING, logger="main"):
        main.download_model_task(nc)

    assert nc.status == [(100, "")]
    assert "could not be downloaded" in caplog.text


def test_init_endpoint_answers_at_once_and_downloads_in_the_background():
    """The download is a background task, as in the default init handler of nc_py_api."""
    background_tasks = BackgroundTasks()
    nc = FakeNC()

    response = asyncio.run(main.init(background_tasks, nc))

    assert response.body == b"{}"
    assert len(background_tasks.tasks) == 1


def test_start_of_an_enabled_app_resumes_the_model_download(monkeypatch):
    """AppAPI sends its init once, so a start of an enabled app finishes the download of the model itself.

    An app that is not enabled at its start is left to the init AppAPI is about to send, so that nothing
    is downloaded twice.
    """
    started = threading.Event()
    monkeypatch.setattr(main, "set_handlers", lambda *args, **kwargs: None)
    monkeypatch.setattr(main, "resume_init", lambda nc: started.set())
    monkeypatch.setattr(main, "SHUTDOWN", asyncio.Event())
    monkeypatch.setattr(main, "APP_ENABLED", threading.Event())

    async def idle_worker():
        await main.SHUTDOWN.wait()

    monkeypatch.setattr(main, "process_tasks", idle_worker)
    nc = MagicMock()
    monkeypatch.setattr(main, "NextcloudApp", lambda: nc)

    async def start():
        async with main.lifespan(main.APP):
            pass

    nc.enabled_state = False
    asyncio.run(start())
    assert not started.wait(0.3)
    assert not main.APP_ENABLED.is_set()

    nc.enabled_state = True
    asyncio.run(start())
    assert started.wait(5)
    assert main.APP_ENABLED.is_set()


def test_resume_init_reports_the_progress_and_never_raises(monkeypatch, caplog):
    """The download of a resumed init reports to AppAPI as the init does, and a start survives its failure."""
    def download_model(on_progress):
        on_progress(40)
        raise OSError("the hub is unreachable")

    monkeypatch.setattr(main, "download_model", download_model)
    nc = FakeNC()
    main.resume_init(nc)
    # The download failed after one report of progress: the initialization is completed all the same.
    assert nc.status == [(40, ""), (100, "")]

    nc = FakeNC()
    nc.set_init_status = MagicMock(side_effect=RuntimeError("Nextcloud is not reachable"))
    monkeypatch.setattr(main, "download_model", lambda on_progress: on_progress(10))
    with caplog.at_level(logging.WARNING, logger="main"):
        main.resume_init(nc)

    assert "after the restart" in caplog.text
