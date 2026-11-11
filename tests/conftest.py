"""Tests run as if outside any harness session: no project dir, no pinned workspace."""

import pytest


@pytest.fixture(autouse=True)
def _no_harness_env(monkeypatch):
    for var in (
        "CLAUDE_PROJECT_DIR",
        "KYB_WORKSPACE",
        "KYB_DISCOVERY_ROOT",
        "KYB_VERTEX_QUOTA_PROJECT",
        "KYB_VERTEX_LOCATION",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("KYB_VERTEX_PROJECT", "kyb-vertex-project")


@pytest.fixture
def qa_ws(tmp_path):
    """A workspace holding case C's recorded FactSet (a real derivation) and its run, for the
    copilot tests."""
    import json

    from decision_fixtures import REPO, workspace

    from kyb.audit.chain import workspace_action
    from kyb.contracts.b4 import FACTS_REL
    from kyb.signing.ledger import write_record
    from kyb.workspace.runs import start_run

    case = "sg-strait-capital"
    root = workspace(tmp_path)
    facts = REPO / "tests" / "fixtures" / "facts" / "sg-strait-capital.factset.json"
    with workspace_action(root):
        start_run(root, case, as_of="2026-10-25", reason="test")
        write_record(root, FACTS_REL.format(case=case, run="r1"), json.loads(facts.read_text()),
                     action="derive_facts", tier="derived", as_of="2026-10-25", inputs=[],
                     detail={})  # fmt: skip
    return root
