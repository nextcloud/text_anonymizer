# SPDX-FileCopyrightText: 2026 Nextcloud GmbH and Nextcloud contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the download of the GLiNER model while AppAPI initialises the app."""

import asyncio
import json
import logging
import os
from types import SimpleNamespace

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
    """Record what download_model fetches, and where, instead of the network."""
    calls = SimpleNamespace(snapshot=[], tokenizer=[], storage=str(tmp_path / "nc_app"))
    monkeypatch.setattr(anonymizer, "model_cache_dir", lambda: calls.storage, raising=False)

    def snapshot_download(model, **kwargs):
        calls.snapshot.append((model, kwargs))
        model_dir = tmp_path / "model"
        model_dir.mkdir(exist_ok=True)
        (model_dir / "gliner_config.json").write_text(json.dumps({"model_name": TOKENIZER}), encoding="utf-8")
        return model_dir

    monkeypatch.setattr(anonymizer, "snapshot_download", snapshot_download, raising=False)
    monkeypatch.setattr(
        anonymizer.AutoTokenizer,
        "from_pretrained",
        classmethod(lambda cls, model, **kwargs: calls.tokenizer.append((model, kwargs))),
        raising=False,
    )
    return calls


def test_model_and_tokenizer_are_downloaded_into_the_persistent_storage(monkeypatch, downloads):
    """Both repositories of the model go to the app's persistent storage, which survives updates."""
    progress = []

    assert anonymizer.download_model(progress.append) is True
    ((model, options),) = downloads.snapshot
    assert model == anonymizer.GLINER_MODEL
    assert options["cache_dir"] == downloads.storage
    assert downloads.tokenizer == [(TOKENIZER, {"cache_dir": downloads.storage})]
    # The progress bar class reports its percentage, as huggingface_hub drives it.
    with open(os.devnull, "w") as sink:
        bar = options["tqdm_class"](total=100, file=sink)
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
