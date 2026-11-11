"""`kyb init` copies the watchlist into `policy/` beside the policy, hashed and pinned (spec)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from kyb.audit.chain import verify_workspace
from kyb.util.hashing import sha256_file
from kyb.workspace.init import init_workspace

ASSETS = Path(__file__).resolve().parents[1] / "examples"


def lock_of(root: Path) -> dict:
    return json.loads((root / "template.lock").read_text())


def test_init_copies_the_watchlist_beside_the_policy(tmp_path):
    root = tmp_path / "ws"
    lock = init_workspace(root, tenant="t", policy_src=ASSETS / "policy/compliance_rules.md")
    copied = root / "policy/watchlists.json"
    assert copied.read_bytes() == (ASSETS / "policy/watchlists.json").read_bytes()
    assert lock["watchlist_at_init"] == sha256_file(copied) == lock_of(root)["watchlist_at_init"]
    assert lock["policy_at_init"] == sha256_file(root / "policy/compliance_rules.md")
    assert verify_workspace(root).ok


def test_a_given_watchlist_wins_and_is_pinned(tmp_path):
    other = tmp_path / "list.json"
    other.write_text(json.dumps({"version": "wl-test", "sanctions": [], "pep": []}))
    root = tmp_path / "ws"
    lock = init_workspace(root, tenant="t", policy_src=ASSETS / "policy/compliance_rules.md",
                          watchlist_src=other)  # fmt: skip
    assert (root / "policy/watchlists.json").read_bytes() == other.read_bytes()
    assert lock["watchlist_at_init"] == sha256_file(other)


def test_no_watchlist_beside_the_policy_is_not_an_error(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    shutil.copyfile(ASSETS / "policy/compliance_rules.md", src / "compliance_rules.md")
    root = tmp_path / "ws"
    lock = init_workspace(root, tenant="t", policy_src=src / "compliance_rules.md")
    assert lock["watchlist_at_init"] is None and not (root / "policy/watchlists.json").exists()
    # a forced re-init without a list does not leave the old one behind, looking pinned
    init_workspace(root, tenant="t", policy_src=ASSETS / "policy/compliance_rules.md", force=True)
    assert (root / "policy/watchlists.json").exists()
    init_workspace(root, tenant="t", policy_src=src / "compliance_rules.md", force=True)
    assert not (root / "policy/watchlists.json").exists()
    assert lock_of(root)["watchlist_at_init"] is None


@pytest.mark.parametrize("content", [None, "not json", "[1, 2]"])
def test_a_bad_given_watchlist_fails_before_anything_is_created(tmp_path, content):
    given = tmp_path / "list.json"
    if content is not None:
        given.write_text(content)
    root = tmp_path / "ws"
    with pytest.raises((FileNotFoundError, ValueError)):
        init_workspace(root, tenant="t", policy_src=ASSETS / "policy/compliance_rules.md",
                       watchlist_src=given)  # fmt: skip
    assert not root.exists()
