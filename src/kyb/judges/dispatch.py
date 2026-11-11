"""Optional J-SUPPORT boundary. No silent fallback to a different model."""

from __future__ import annotations

from pathlib import Path

from kyb.contracts.b4 import JudgeOutcome
from kyb.judges.core import outcome
from kyb.run.records import read_record, write_record
from kyb.util.hashing import sha256_file, sha256_json
from kyb.workspace.catalog import Catalog


def checker_identity(checker) -> dict:
    return getattr(checker, "identity", {"available": checker is not None})


def support_checker(root: Path, *, port=None):
    try:
        from kyb.judges.support import j_support
    except ImportError:
        return None
    if port is None:
        from kyb.decide.support import default_support_port

        port = default_support_port(
            root,
            answerer_model=lambda i, a: getattr(check, "answerer_model", lambda *_: None)(i, a),
        )
    identity = {
        "available": True,
        "cache_version": "support-dispatch-v2-permanent-veto",
        "judge_code": sha256_file(Path(j_support.__code__.co_filename)),
        "port": getattr(
            port,
            "identity",
            {
                "family": getattr(port, "family", "unknown"),
                "revision": getattr(port, "revision", "pinned"),
            },
        ),
    }

    def check(instance, answer, evidence):
        resolver = getattr(port, "answerer_model", None)
        try:
            answerer = resolver(instance, answer) if callable(resolver) else None
        except RuntimeError, ValueError, LookupError, OSError:
            answerer = None
        # A content veto survives judge/configuration revisions and answer paraphrases.
        # Evidence or the typed question must actually change before it can be reconsidered.
        semantic_key = sha256_json(
            {
                "instance": instance.model_dump(mode="json"),
                "value": answer.value,
                "citations": sorted(answer.evidence),
                "evidence": [
                    e.model_dump(mode="json") for e in sorted(evidence, key=lambda e: e.evidence_id)
                ],
            }
        ).removeprefix("sha256:")
        veto_rel = f"gold/judgements/support-vetoes/{semantic_key}.json"
        key = sha256_json(
            {
                "identity": identity,
                "answerer_model": answerer,
                "instance": instance.model_dump(mode="json"),
                "answer": answer.model_dump(mode="json"),
                "evidence": [e.model_dump(mode="json") for e in evidence],
            }
        )
        rel = f"gold/judgements/support-outcomes/{key.removeprefix('sha256:')}.json"
        if Catalog(root).get(veto_rel):
            rel = veto_rel
            result = JudgeOutcome.model_validate(read_record(root, rel)["outcome"])
        elif Catalog(root).get(rel):
            result = JudgeOutcome.model_validate(read_record(root, rel)["outcome"])
        else:
            try:
                result = j_support(instance, answer, evidence, port=port)
            except (RuntimeError, ValueError, LookupError, OSError) as exc:
                result = outcome(
                    "J-SUPPORT",
                    instance.instance_id,
                    f"support unavailable: {type(exc).__name__}",
                    unknown=True,
                )
            if result.judge_id != "J-SUPPORT" or result.target != instance.instance_id:
                result = outcome(
                    "J-SUPPORT",
                    instance.instance_id,
                    "invalid support judge response",
                    unknown=True,
                )
            content = bool(result.model and result.request_hash) and result.effect != "unknown"
            if content and result.verdict == "veto":
                rel = veto_rel
            elif not content:
                # A failed/unavailable attempt is audit evidence, never a semantic outcome.
                # Do not link it as a permanent cache hit; the next decide retries it.
                prefix = f"gold/judgements/support-attempts/{key.removeprefix('sha256:')}/"
                attempt = 0
                while Catalog(root).get(f"{prefix}{attempt}.json"):
                    attempt += 1
                write_record(
                    root,
                    f"{prefix}{attempt}.json",
                    {"identity": identity, "outcome": result.model_dump(mode="json")},
                    action="record_support_attempt",
                    tier="judged",
                    as_of="recorded",
                )
                return result
            write_record(
                root,
                rel,
                {"identity": identity, "outcome": result.model_dump(mode="json")},
                action="record_support_outcome",
                tier="judged",
                as_of="recorded",
            )
        if work_order_id := getattr(check, "work_order_id", None):
            digest = sha256_json({"request": key, "outcome": rel}).removeprefix("sha256:")
            link = f"gold/judgements/support-links/{work_order_id}/{digest}.json"
            write_record(
                root,
                link,
                {"outcome": rel},
                action="consume_support",
                tier="judged",
                as_of="recorded",
                inputs=[Catalog(root).ref(rel)],
            )
        return result

    check.identity = identity
    return check
