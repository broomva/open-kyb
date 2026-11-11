"""The `derive_facts` action: read a run's verification, write `gold/facts/<case>/<run>.json`.

Call under the workspace action lock (`workspace_action`). Facts are built only on a
verification that still matches its inputs (the same check `kyb review --status` makes), and
per-document assurance is read through `assurance_now`, never from the witness record's snapshot.
Ownership-conflict materiality is read from the workspace's policy file. The record is written once:
re-deriving the same run must give the same bytes.

Completion is resumable (as `read.submit._finish`): the record is published, then audited (once,
found by its output hash), then given its catalog receipt last. A retry after any failure
completes exactly the missing steps, and nothing counts until the receipt exists.
"""

from __future__ import annotations

import json
from pathlib import Path

from kyb.audit.chain import AuditLog, chain_events
from kyb.contracts.b4 import FACTS_REL, FactSet
from kyb.facts.derive import METHOD, DocInput, derive_factset, materiality
from kyb.policy.provenance import approved_policy
from kyb.read.plan import _documents, read_plan
from kyb.read.status import _verification_stale
from kyb.read.submit import claims_rel
from kyb.util.fsutil import contained, publish_exclusive, safe_id
from kyb.util.hashing import sha256_bytes, sha256_json
from kyb.witness.verify import assurance_now
from kyb.workspace.catalog import Catalog
from kyb.workspace.runs import FRESH_RUN, run_meta, safe_run_id

POLICY_REL = "policy/compliance_rules.md"


def audited(root: Path, action: str, output_hash: str) -> bool:
    """Whether this action already recorded this exact output (a retry must not append twice)."""
    return any(e["action"] == "tool.call" and e["detail"].get("action") == action
               and e["output_hash"] == output_hash for e in chain_events(root))  # fmt: skip


def facts_rel(case: str, run: str) -> str:
    return FACTS_REL.format(case=safe_id(case, "case id"), run=safe_run_id(run))


def _witness_rel(case: str, run: str, doc: str) -> str:
    return f"silver/witness/{case}/{run}/{doc}.json"


def load_documents(root: Path, case: str, run: str) -> list[DocInput]:
    """Every document of the run as verification left it. Refuses a run whose verification is
    missing or stale."""
    catalog = Catalog(root)
    summary = f"silver/verified_claims/{case}/{run}.json"
    if catalog.get(summary) is None:
        raise LookupError(f"{case} {run}: not verified yet (run `kyb claims verify {case}`)")
    docs = [e["doc"] for e in read_plan(root, case, run)]
    if stale := _verification_stale(root, catalog, case, run, docs, summary):
        raise PermissionError(f"{case} {run}: verification is stale ({stale}); verify a new run")
    names = {key: f["name"] for key, f in _documents(root, case)}
    out = []
    for doc in docs:
        sub = json.loads(contained(root, claims_rel(case, run, doc)).read_text("utf-8"))
        rec = json.loads(contained(root, _witness_rel(case, run, doc)).read_text("utf-8"))
        now = assurance_now(root, case, run, doc)
        out.append(DocInput(
            doc=doc, document_name=names[doc], claims=sub["claims"],
            verdicts={r["claim_id"]: bool(r["verified"]) for r in rec["results"]},
            assurance=now["assurance"], isolated=now["context_now"] == "isolated",
            coverage=rec.get("coverage") or [],
            findings=[f["kind"] for f in rec.get("findings") or []],
        ))  # fmt: skip
    return out


def derive_facts(root: Path, case: str, run: str) -> tuple[FactSet, str, bool]:
    """Derive and record the run's `FactSet`. Returns (facts, path, written): `written` is False
    when the identical record was already complete (nothing new is appended to the audit log)."""
    case = safe_id(case, "case id")
    run = safe_run_id(run)
    as_of = run_meta(root, case, run)["as_of"]
    docs = load_documents(root, case, run)
    policy = approved_policy(root)
    points = materiality(policy.decode("utf-8"))
    fs = derive_factset(case, run, as_of, docs, materiality=points)
    rel = facts_rel(case, run)
    body = (json.dumps(fs.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n").encode()
    out_hash = sha256_bytes(body)
    catalog = Catalog(root)
    if catalog.get(rel) is not None:
        if sha256_bytes(contained(root, rel).read_bytes()) != out_hash:
            raise PermissionError(f"{rel}: already derived with different facts; facts from "
                                  "changed inputs or another policy are a new run")  # fmt: skip
        return fs, rel, False
    publish_exclusive(contained(root, rel), body)  # identical bytes from an interrupted try: ok
    inputs = [catalog.ref(f"silver/verified_claims/{case}/{run}.json")]
    for d in docs:
        inputs += [catalog.ref(claims_rel(case, run, d.doc)),
                   catalog.ref(_witness_rel(case, run, d.doc))]  # fmt: skip
    if not audited(root, "derive_facts", out_hash):
        AuditLog(root / "ledger" / "audit.jsonl").append(
            "facts", "tool.call", input_hash=sha256_json(sorted(inputs)), output_hash=out_hash,
            detail={"action": "derive_facts", "case": case, "run": run,
                    "fact_set_hash": fs.fact_set_hash, "policy_hash": sha256_bytes(policy),
                    "entities": len(fs.entities), "holdings": len(fs.holdings),
                    "conflicts": len(fs.conflicts), "unverified": len(fs.unverified)},
        )  # fmt: skip
    catalog.register(rel, tier="derived", as_of=as_of, inputs=inputs,
                     produced_by={"action": "derive_facts", "method": METHOD,
                                  "contracts": fs.contracts_version,
                                  "policy_hash": sha256_bytes(policy),
                                  "materiality_points": format(points, "f")})  # fmt: skip
    return fs, rel, True


def load_facts(root: Path, case: str, run: str) -> FactSet:
    """A recorded `FactSet`, only while it still stands: its bytes match its receipt and its own
    hash, every input still hashes to the receipt it was derived from, the policy still states the
    materiality they were derived under, the verification is not stale, and each document's
    assurance is still what the facts recorded (later evidence can
    lower assurance; facts built on the higher one are refused, and a new run re-derives them)."""
    case = safe_id(case, "case id")
    run = safe_run_id(run)
    rel = facts_rel(case, run)
    catalog = Catalog(root)
    entry = catalog.get(rel)
    if entry is None:
        raise LookupError(f"{case} {run}: no facts yet (run `kyb facts {case}`)")
    raw = contained(root, rel).read_bytes()
    if sha256_bytes(raw) != entry["content_hash"]:
        raise PermissionError(f"{rel}: bytes differ from its catalog receipt")
    fs = FactSet.model_validate_json(raw)
    if sha256_json(fs.model_dump(mode="json", exclude={"fact_set_hash"})) != fs.fact_set_hash:
        raise PermissionError(f"{rel}: fact_set_hash does not match its content")
    for ref in entry["inputs"]:
        path, _, want = ref.partition("#")
        now = catalog.get(path)
        if now is None or now["content_hash"] != want:
            raise PermissionError(f"{rel}: input {path} changed since the facts were derived")
        if sha256_bytes(contained(root, path).read_bytes()) != want:
            raise PermissionError(f"{rel}: input {path} differs from its receipt")
    # the facts depend on a policy parameter (materiality): a policy that now states another
    # value needs facts derived under it (Codex r2 #5); cosmetic policy edits do not
    points = format(materiality(approved_policy(root).decode("utf-8")), "f")
    if entry["produced_by"].get("materiality_points") != points:
        raise PermissionError(f"{rel}: derived with conflict materiality "
                              f"{entry['produced_by'].get('materiality_points')}, the policy now "
                              f"states {points}; facts are derived once per run. "
                              f"{FRESH_RUN}")  # fmt: skip
    load_documents(root, case, run)  # refuses a stale verification
    for v in fs.verification:
        now = assurance_now(root, case, run, v.doc)
        if (now["assurance"], now["context_now"] == "isolated") != (v.assurance, v.isolated):
            raise PermissionError(f"{rel}: document {v.doc}'s assurance changed since the facts "
                                  "were derived; derive them in a new run")  # fmt: skip
    return fs
