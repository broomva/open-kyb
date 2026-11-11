"""Workspace recording for L4j's Jev port. Imported only when L4j is installed."""

from __future__ import annotations

import os
from pathlib import Path

from kyb.audit.chain import AuditLog, chain_events
from kyb.run.records import write_record
from kyb.util.fsutil import contained
from kyb.util.hashing import sha256_json


def default_support_port(root: Path, *, answerer_model=None):
    from kyb.witness.jev import MODEL_ALIAS, PROMPT_VERSION, JevClient, JevPort

    live = bool(os.environ.get("TYPESAFE_API_KEY", "").strip())
    port = JevPort(
        contained(root, "gold/judgements/support-recordings"),
        JevClient() if live else None,
        mode="auto",
    )

    class WorkspacePort:
        family = port.family
        identity = {
            "family": family,
            "model_alias": MODEL_ALIAS,
            "prompt_version": PROMPT_VERSION,
            "answerer_guard": "attestation-stop-v1",
        }

        def answerer_model(self, instance, answer):
            return answerer_model(instance, answer) if answerer_model else None

        def ask(self, body):
            response = port.ask(body)
            key = response.request_hash.removeprefix("sha256:")
            rel = f"gold/judgements/support-calls/{key}.json"
            record = {
                "request": body,
                "response": {
                    "model": response.model,
                    "answers": response.answers,
                    "usage": response.usage,
                },
                "request_hash": response.request_hash,
            }
            write_record(
                root, rel, record, action="record_support_call", tier="judged", as_of="recorded"
            )
            log_key = sha256_json(record)
            if not any(
                e["action"] == "model.call" and e["input_hash"] == log_key
                for e in chain_events(root)
            ):
                usage = response.usage
                AuditLog(root / "ledger/audit.jsonl").append(
                    "support",
                    "model.call",
                    input_hash=log_key,
                    output_hash=sha256_json(response.answers),
                    tokens_in=int(usage.get("input_tokens", usage.get("prompt_tokens", 0))),
                    tokens_out=int(usage.get("output_tokens", usage.get("completion_tokens", 0))),
                    cost_usd=float(usage.get("cost_usd", 0)),
                    detail={
                        "model": response.model,
                        "request_hash": response.request_hash,
                        "replayed": response.replayed,
                        "cost_reported": "cost_usd" in usage,
                    },
                )
            return response

    return WorkspacePort()


def answerer_resolver(root: Path, work, proposal: dict | None):
    """Resolve only the exact pinned answer, with current harness-attested provenance."""
    from kyb.predicates.lifecycle import lifecycle_reasons
    from kyb.read.harness import model_family, stopped_model
    from kyb.run.action import load_work_order
    from kyb.run.records import read_record

    def resolve(instance, answer):
        if proposal is None or instance not in work.instances:
            return None
        source = proposal.get("sources", {}).get(instance.instance_id)
        if not source:
            return None
        recorded = read_record(root, source["record"])
        if answer.model_dump(mode="json") not in recorded["answers"]:
            return None
        source_work = load_work_order(root, source["work_order_id"])
        if instance not in source_work.instances:
            return None
        who = source["who"]
        if who != recorded["produced_by"]["predicates"]:
            return None
        events = chain_events(root)
        if lifecycle_reasons(
            events,
            who,
            f"kyb-predicates {source_work.work_order_id}",
            {"predicate_work_order", "propose_predicates"},
        ):
            return None
        model = stopped_model(events, who.get("session"), who.get("agent_id"))
        return model if model_family(model) != "unknown" else None

    return resolve
