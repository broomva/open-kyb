"""propose_predicates: scope, evidence universe and one-time harness attestation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from kyb.audit.chain import AuditLog, chain_events
from kyb.contracts.b4 import PredicateAnswer
from kyb.read.harness import PREDICATES_TYPES, attest, split_token
from kyb.run.action import load_work_order, work_order_rel
from kyb.run.records import read_record, write_record
from kyb.util.fsutil import contained, publish_exclusive, safe_id
from kyb.util.hashing import canonical_json, sha256_json
from kyb.workspace.catalog import Catalog


class Proposal(BaseModel):
    model_config = ConfigDict(extra="forbid")
    work_order_id: str
    case: str
    run: str
    fact_set_hash: str
    rules_version: str
    catalog_version: str
    answers: list[PredicateAnswer]


def answers_rel(work_order_id: str) -> str:
    return f"gold/predicates/answers/{safe_id(work_order_id, 'work order id')}.json"


def pin_rel(work, instance) -> str:
    kinds = {
        "same_party": (
            "sanctions_corroborated",
            "sanctions_fuzzy",
            "pep_active",
            "pep_former",
        ),
        "natural_person_directs_business": ("nominee_control", "nominee_sole_controller"),
        "amendment_effective": ("conflicting_filings",),
        "doc_is_proof_of_address": ("required_documents",),
    }
    relevant = kinds[instance.predicate]
    matched: dict[str, str] = {}
    for k in relevant:
        hits = {
            rid: text for rid, text in work.policy_text.items() if rid == k or f"kind: {k}" in text
        }
        if not hits:
            matched = {}
            break
        matched.update(hits)
    policy_text = (
        matched
        if matched
        else {"unmapped_policy_version": work.rules_version, "policy_text": work.policy_text}
    )
    key = sha256_json(
        {
            "facts": work.fact_set_hash,
            "catalog": work.catalog_version,
            "instance": instance.model_dump(mode="json"),
            "policy_text": policy_text,
        }
    )
    return f"gold/predicates/pins/{key.removeprefix('sha256:')}.json"


def load_answers(root: Path, work_order_id: str) -> dict | None:
    work = load_work_order(root, work_order_id)
    sources = {}
    answers = []
    for instance in work.instances:
        rel = pin_rel(work, instance)
        if Catalog(root).get(rel):
            pin = read_record(root, rel)
            answers.append(pin["answer"])
            sources[instance.instance_id] = pin["source"]
    if not sources:
        return None
    first = sources[sorted(sources)[0]]
    return {
        "work_order_id": work_order_id,
        "answers": answers,
        "sources": sources,
        "pending": [i.instance_id for i in work.instances if i.instance_id not in sources],
        "produced_by": {"predicates": first["who"]},
    }


def propose_predicates(root: Path, payload: dict[str, Any]) -> dict:
    args, token = split_token(payload)
    sub = Proposal.model_validate(args)
    work = load_work_order(root, sub.work_order_id)
    for key in ("case", "run", "fact_set_hash", "rules_version", "catalog_version"):
        if getattr(sub, key) != getattr(work, key):
            raise ValueError(f"{key} differs from work order")
    instances = {i.instance_id: i for i in work.instances}
    cached = load_answers(root, sub.work_order_id) or {}
    pending = set(instances) - set(cached.get("sources", {}))
    rel = answers_rel(sub.work_order_id)
    input_hash = sha256_json(args)
    path = contained(root, rel)
    ids = [a.instance_id for a in sub.answers]
    if len(ids) != len(set(ids)) or set(ids) - set(instances):
        raise ValueError("answer every pending instance exactly once")
    if not path.exists() and set(ids) != pending:
        raise ValueError("answer every pending instance exactly once")
    for raw, answer in zip(args["answers"], sub.answers, strict=True):
        if type(raw.get("value")) is not bool and raw.get("value") != "UNKNOWN":
            raise ValueError("predicate value must be a boolean or UNKNOWN")
        if set(answer.evidence) - set(instances[answer.instance_id].evidence_universe):
            raise ValueError(f"{answer.instance_id}: evidence outside its universe")
        if len(answer.evidence) != len(set(answer.evidence)):
            raise ValueError(f"{answer.instance_id}: duplicate evidence ids")
    if path.exists():
        import json

        record = json.loads(path.read_bytes())
        if record["produced_by"]["input_hash"] != input_hash:
            raise PermissionError(
                "predicate answers are pinned; different answers require a new run"
            )
    else:
        who = attest(
            root,
            tool="propose_predicates",
            input_hash=input_hash,
            token=token,
            allowed_types=PREDICATES_TYPES,
            target={"work_order": sub.work_order_id},
        )
        record = {
            **sub.model_dump(mode="json"),
            "produced_by": {"input_hash": input_hash, "predicates": who.identity()},
        }
    who = record["produced_by"]["predicates"]
    publish_exclusive(path, canonical_json(record) + b"\n")
    # Consume before the catalog receipt, also when completing an interrupted publication.
    if who["context"] == "isolated" and not any(
        e["action"] == "harness.attestation_consumed"
        and e["detail"].get("token_hash") == who["token_hash"]
        for e in chain_events(root)
    ):
        AuditLog(root / "ledger/audit.jsonl").append(
            "harness",
            "harness.attestation_consumed",
            input_hash=input_hash,
            output_hash=sha256_json(record),
            detail={
                "token_hash": who["token_hash"],
                "tool_use_id": who["tool_use_id"],
                "agent_id": who["agent_id"],
                "session": who["session"],
                "target": {"work_order": sub.work_order_id},
            },
        )
    write_record(
        root,
        rel,
        record,
        action="propose_predicates",
        tier="claimed",
        as_of=Catalog(root).get(work_order_rel(sub.work_order_id))["as_of"],
        inputs=[Catalog(root).ref(work_order_rel(sub.work_order_id))],
    )
    for answer in sub.answers:
        pinned_rel = pin_rel(work, instances[answer.instance_id])
        if Catalog(root).get(pinned_rel):
            prior = read_record(root, pinned_rel)
            if prior["answer"] != answer.model_dump(mode="json"):
                raise PermissionError("a predicate revision is already pinned")
            continue
        write_record(
            root,
            pinned_rel,
            {
                "answer": answer.model_dump(mode="json"),
                "source": {"work_order_id": work.work_order_id, "who": who, "record": rel},
            },
            action="pin_predicate",
            tier="claimed",
            as_of=Catalog(root).get(work_order_rel(sub.work_order_id))["as_of"],
            inputs=[Catalog(root).ref(rel)],
        )
    return {"recorded": rel, "answers": len(sub.answers), "context": who["context"]}
