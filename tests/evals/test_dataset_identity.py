"""A released corpus fingerprint ignores runtime files but detects changed evidence."""

from __future__ import annotations

import subprocess

import pytest

from kyb.evals import runner


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    repo = tmp_path / "checkout"
    root = repo / "datasets"
    root.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    record = root / "a record.json"
    record.write_text('{"verified": true}\n')
    subprocess.run(["git", "-C", str(repo), "add", "datasets"], check=True)
    monkeypatch.setattr(runner, "REPO", repo)
    monkeypatch.setattr(runner, "datasets", lambda: root)
    return repo, root, record


def test_runtime_locks_and_untracked_records_do_not_change_released_corpus(corpus):
    _, root, _ = corpus
    before = runner._inputs()
    (root / ".catalog.lock").write_text("runtime lock")
    (root / "uncommitted.json").write_text('{"verified": false}')
    assert runner._inputs() == before


def test_changed_tracked_content_and_new_index_members_change_identity(corpus):
    repo, root, record = corpus
    before = runner._inputs()
    record.write_text('{"verified": false}\n')
    changed = runner._inputs()
    assert changed["datasets_hash"] != before["datasets_hash"]
    assert changed["datasets_files"] == before["datasets_files"] == 1
    added = root / "new.json"
    added.write_text("{}")
    subprocess.run(["git", "-C", str(repo), "add", str(added)], check=True)
    after = runner._inputs()
    assert after["datasets_files"] == 2
    assert after["datasets_hash"] != changed["datasets_hash"]


def test_missing_tracked_evidence_fails_instead_of_silently_shrinking_corpus(corpus):
    _, _, record = corpus
    record.unlink()
    with pytest.raises(FileNotFoundError):
        runner._inputs()


def test_explicit_external_corpus_remains_content_addressed(tmp_path, monkeypatch):
    root = tmp_path / "external"
    root.mkdir()
    record = root / "record.json"
    record.write_text("{}")
    monkeypatch.setattr(runner, "datasets", lambda: root)
    before = runner._inputs()
    record.write_text('{"changed": true}')
    after = runner._inputs()
    assert before["datasets_files"] == after["datasets_files"] == 1
    assert before["datasets_hash"] != after["datasets_hash"]
