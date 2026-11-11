"""The callback uses real core checks without publishing model proposals to the workspace."""
# ruff: noqa: F811 -- imported pytest fixtures are injected by parameter name

import asyncio
import copy
import inspect
import json
import stat

import pytest
from test_read_contract import CASE, _built, _payload, ws  # noqa: F401

from kyb.codex import preflight
from kyb.read.schema import ClaimsSubmission
from kyb.read.submit import _check, claims_rel
from kyb.util.hashing import sha256_json


@pytest.fixture
def job(ws, tmp_path, monkeypatch):
    directory = tmp_path / "private-job"
    directory.mkdir(mode=0o700)
    payload = _payload(ws)
    scope = {k: payload[k] for k in ("case", "run", "doc", "doc_sha256")}
    scope.update(version=1, role="kyb-reader", root=str(ws))
    path = directory / preflight.SCOPE_FILE
    path.write_text(json.dumps(scope))
    path.chmod(0o600)
    monkeypatch.setenv("KYB_CODEX_RECEIPT_DIR", str(directory))
    return directory, scope


def receipts(job):
    return [
        json.loads(line) for line in (job[0] / preflight.RECEIPTS_FILE).read_text().splitlines()
    ]


@pytest.mark.parametrize("change", [None, "subject", "page", "bbox", "text_layer", "schema"])
def test_preflight_runs_exact_core_validation_and_does_not_record_claims(ws, job, change):
    payload = _payload(ws)
    claim = payload["claims"][0]
    if change == "subject":
        claim["subject"] = "Unquoted proposed subject"
    elif change == "page":
        claim["page"] = 99
    elif change == "bbox":
        claim["bbox"] = [1710, 760, 2600, 835]
    elif change == "text_layer":
        claim["source"] = "text_layer"
    elif change == "schema":
        claim["confidence"] = 2
    before = {name: (ws / name).read_bytes() for name in ("catalog.json", "ledger/audit.jsonl")}
    out = preflight.validate_claims(payload["claims"])
    assert out["attempt"] == 1 and out["claims_hash"] == sha256_json(payload["claims"])
    if change == "schema":
        assert not out["valid"] and out["errors"][0].startswith("schema:")
    else:
        expected = _check(ClaimsSubmission.model_validate(payload), ws)
        assert out["errors"] == expected
        assert out["valid"] == (not expected)
    assert not (ws / claims_rel(CASE, "r1", "05")).exists()
    assert {name: (ws / name).read_bytes() for name in before} == before


def test_callback_scope_cannot_be_supplied_as_model_arguments(ws, job):
    assert list(inspect.signature(preflight.validate_claims).parameters) == ["claims"]
    with pytest.raises(TypeError):
        preflight.validate_claims(_payload(ws)["claims"], case="other", role="officer")
    tools = asyncio.run(preflight.create_server().list_tools())
    assert [tool.name for tool in tools] == ["validate_claims"]
    assert set(tools[0].input_schema["properties"]) == {"claims"}


def test_receipt_binds_final_claims_hash_without_retaining_proposals(ws, job):
    claims = _payload(ws)["claims"]
    claims[0]["value"] = {"private-proposed-component": "never-persist-this-proposal"}
    result = preflight.validate_claims(claims)
    assert not result["valid"]
    stored = receipts(job)[0]
    assert stored == {"attempt": 1, "claims_hash": sha256_json(claims), "valid": False,
                      "errors": ["excerpt_missing"], "scope_hash": sha256_json(job[1]),
                      "role": "kyb-reader"}  # fmt: skip
    raw = (job[0] / preflight.RECEIPTS_FILE).read_text()
    assert "never-persist-this-proposal" not in raw and "private-proposed-component" not in raw
    for name in (preflight.SCOPE_FILE, preflight.RECEIPTS_FILE, "preflight.lock"):
        assert stat.S_IMODE((job[0] / name).stat().st_mode) == 0o600


def test_only_three_calls_validate_and_over_budget_is_logged(ws, job, monkeypatch):
    calls = []
    real_check = preflight._check

    def checking(sub, root):
        calls.append(sub)
        return real_check(sub, root)

    monkeypatch.setattr(preflight, "_check", checking)
    claims = _payload(ws)["claims"]
    for attempt in range(1, 4):
        out = preflight.validate_claims(claims)
        assert out["valid"] and out["attempt"] == attempt
    denied = preflight.validate_claims(claims)
    assert denied["attempt"] == 4 and denied["errors"] == ["budget_exhausted"]
    assert len(calls) == 3
    assert receipts(job)[-1]["errors"] == ["budget_exhausted"]


@pytest.mark.parametrize(
    "change",
    [
        "role",
        "extra",
        "version",
        "relative",
        "run",
        "hash",
        "scope_mode",
        "job_mode",
        "symlink",
        "duplicate_key",
    ],
)
def test_invalid_scope_fails_without_checking_or_receipting(ws, job, monkeypatch, change):
    directory, original = job
    scope = copy.deepcopy(original)
    path = directory / preflight.SCOPE_FILE
    if change == "role":
        scope["role"] = "kyb-predicates"
    elif change == "extra":
        scope["model"] = "other"
    elif change == "version":
        scope["version"] = True
    elif change == "relative":
        scope["root"] = "relative-workspace"
    elif change == "run":
        scope["run"] = "../r1"
    elif change == "hash":
        scope["doc_sha256"] = "not-a-hash"
    elif change == "scope_mode":
        path.chmod(0o644)
    elif change == "job_mode":
        directory.chmod(0o755)
    elif change == "symlink":
        path.rename(directory / "other-scope")
        path.symlink_to(directory / "other-scope")
    elif change == "duplicate_key":
        path.write_text(json.dumps(scope)[:-1] + ', "role":"kyb-reader"}')
    if change not in {"scope_mode", "job_mode", "symlink", "duplicate_key"}:
        path.write_text(json.dumps(scope))
    monkeypatch.setattr(preflight, "_check", lambda *_: pytest.fail("scope must refuse first"))
    out = preflight.validate_claims(_payload(ws)["claims"])
    assert not out["valid"] and out["attempt"] == 0
    assert not (directory / preflight.RECEIPTS_FILE).exists()


def test_missing_scope_or_environment_refuses_without_validation(ws, tmp_path, monkeypatch):
    monkeypatch.delenv("KYB_CODEX_RECEIPT_DIR", raising=False)
    assert preflight.validate_claims([])["attempt"] == 0
    monkeypatch.setenv("KYB_CODEX_RECEIPT_DIR", str(tmp_path / "missing"))
    assert preflight.validate_claims([])["attempt"] == 0


def test_scope_hash_change_or_corrupt_receipts_cannot_reset_budget(ws, job):
    claims = _payload(ws)["claims"]
    assert preflight.validate_claims(claims)["valid"]
    path = job[0] / preflight.SCOPE_FILE
    scope = copy.deepcopy(job[1])
    scope["doc"] = "04"
    path.write_text(json.dumps(scope))
    assert preflight.validate_claims(claims)["attempt"] == 0
    path.write_text(json.dumps(job[1]))
    log = job[0] / preflight.RECEIPTS_FILE
    log.write_text("broken-json\n")
    assert preflight.validate_claims(claims)["attempt"] == 0


def test_workspace_integrity_failure_is_receipted_without_validating_or_submitting(ws, job):
    with (ws / "ledger/audit.jsonl").open("a") as stream:
        stream.write("{}\n")
    out = preflight.validate_claims(_payload(ws)["claims"])
    assert not out["valid"] and out["attempt"] == 1
    assert receipts(job)[0]["errors"] == ["workspace_unavailable"]
    assert not (ws / claims_rel(CASE, "r1", "05")).exists()
