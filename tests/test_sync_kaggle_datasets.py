"""Kaggle dataset sync must refuse foreign ids/paths and scope the proxy to the kaggle command only."""
import os
import sys

import pytest

import scripts.sync_kaggle_datasets as kd


def test_only_project_dataset_ids_are_allowed():
    assert kd.ALLOWED_DATASETS == {
        "thermo-agent-code", "thermo-agent-corpus", "thermo-agent-resume",
        "thermo-teacher-prompts", "thermo-teacher-prompts-big", "thermo-agent-eval-replays",
    }
    for slug in kd.ALLOWED_DATASETS:
        assert kd.resolve_dataset(slug) == f"rezanadimi/{slug}"
        assert kd.resolve_dataset(f"rezanadimi/{slug}") == f"rezanadimi/{slug}"
    for bad in ("thermo-kernel-wheels", "rezanadimi/thermo-kernel-wheels",
                "someone-else/thermo-agent-code", "thermo-agent-code/evil", ""):
        with pytest.raises(ValueError, match="refusing Kaggle dataset"):
            kd.resolve_dataset(bad)


def test_dataset_dir_refuses_non_project_dataset_and_outside_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="refusing Kaggle dataset"):
        kd.dataset_dir("rezanadimi/thermo-kernel-wheels")
    # A non-existent local dir for an allowed id is refused (never silently created).
    with pytest.raises(SystemExit, match="does not exist locally"):
        kd.plan("upload", "thermo-agent-corpus")
    # The legacy big-prompts folder maps to the allow-listed slug.
    (tmp_path / "data" / "kaggle_datasets" / "teacher_big_prompts").mkdir(parents=True)
    item = kd.plan("upload", "thermo-teacher-prompts-big")
    assert item["dir"] == "data/kaggle_datasets/teacher_big_prompts"


def test_dry_run_scopes_proxy_to_kaggle_and_never_mutates_global_env(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "kaggle_datasets" / "thermo-agent-code").mkdir(parents=True)
    monkeypatch.delenv("HTTP_PROXY", raising=False)
    monkeypatch.delenv("HTTPS_PROXY", raising=False)
    monkeypatch.delenv("http_proxy", raising=False)
    monkeypatch.delenv("https_proxy", raising=False)

    item = kd.plan("upload", "thermo-agent-code", message="rebuild", tag="2026-11-19")
    assert item["dataset"] == "rezanadimi/thermo-agent-code"
    assert item["command"][:3] == [kd.kaggle_bin(), "datasets", "version"]
    assert "-p" in item["command"] and "rebuild [tag 2026-11-19]" in item["command"]
    assert "$env:HTTP_PROXY='socks5h://localhost:10808'; $env:HTTPS_PROXY='socks5h://localhost:10808'" in item["display"]

    env = kd.kaggle_env("SECRET_TOKEN")
    assert env["HTTP_PROXY"] == env["HTTPS_PROXY"] == "socks5h://localhost:10808"
    assert env["KAGGLE_API_TOKEN"] == "SECRET_TOKEN"
    assert "HTTP_PROXY" not in os.environ and "HTTPS_PROXY" not in os.environ, "proxy must not be global"

    def no_network(*a, **k):
        pytest.fail("dry-run must not invoke kaggle or touch the network")

    monkeypatch.setattr(kd.subprocess, "run", no_network)
    assert kd.run_plan(item, dry_run=True) == 0
    assert "proxy scoped to the kaggle command" in capsys.readouterr().out


def test_cli_refuses_foreign_dataset_before_any_command(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["sync_kaggle_datasets.py", "upload",
                                      "--dataset", "thermo-kernel-wheels", "--dry-run"])
    with pytest.raises(ValueError, match="refusing Kaggle dataset"):
        kd.main()


def test_download_plan_is_additive_local_and_uses_proxy_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    d = tmp_path / "data" / "kaggle_datasets" / "thermo-agent-corpus"
    d.mkdir(parents=True)
    item = kd.plan("download", "thermo-agent-corpus", overwrite=False)
    assert item["command"][1:4] == ["datasets", "download", "rezanadimi/thermo-agent-corpus"]
    assert "--unzip" in item["command"] and "-w" not in item["command"]
    over = kd.plan("download", "thermo-agent-corpus", overwrite=True)
    assert "-w" in over["command"]
