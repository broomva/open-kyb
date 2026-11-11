"""`kyb claims verify` (`verify_quotes`): every submitted claim is witnessed from pixels (spec).

Per document, code cuts each claim's crop (and its linked context blocks) from the verified PDF
bytes; one blind witness request transcribes and re-reads them all. The witness never sees the
claim: crops are sent with opaque positional ids (`k1`, `k2`, ...), never the reader's ids.
Then, in code:
- transcription: `norm-v2` exact containment of the excerpt, on token boundaries (a text-layer
  excerpt whose table cell wraps is rebuilt from the layer line below, `contains_wrapped`);
- text layer: a `text_layer` claim must also occur in the page's text layer; a layer claim the
  pixels do not confirm is an integrity finding (hidden text);
- binding (`bind-v6`, `kyb.witness.bind`): every component of the claim binds to one of the
  crop's re-read `{subject, attribute, value}` tuples (same subject, a field label of the document's
  own entity, or a role in either orientation; a compatible attribute; the same typed value); a
  claim flagged struck-through must bind a struck value, any other claim must not. Every required
  context block must be linked *and* its crop must come back readable; context qualifiers are
  recorded with the result;
- coverage: every material fact from the claim-blind layout must be matched by a verified claim
  for the same canonical attribute; a different value of that attribute is a recorded conflict,
  a missing one is `coverage: incomplete`;
- assurance: W2 only when the reader's and the witness's families are both known and differ.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from kyb.audit.chain import AuditLog, chain_events
from kyb.docintel import poppler
from kyb.read.harness import eligibility, model_family, stopped_model
from kyb.read.plan import _documents, load_layout, read_plan
from kyb.read.submit import claims_rel, submitted
from kyb.util.fsutil import contained, publish_exclusive
from kyb.util.hashing import sha256_bytes, sha256_json
from kyb.witness.bind import (
    BIND,
    Scope,
)
from kyb.witness.bind import binding as bind_claim
from kyb.witness.bind import canon_attr as canon_attr
from kyb.witness.bind import (
    subject_match as subject_match,
)
from kyb.witness.bind import (
    value_match as value_match,
)
from kyb.witness.coverage import VERSION as COVERAGE
from kyb.witness.coverage import check_record as check_coverage_record
from kyb.witness.coverage import coverage as match_coverage
from kyb.witness.errors import ROUTING_VERSION, WitnessAnomalous
from kyb.witness.layout import record_usage
from kyb.witness.norm import VERSION as NORM
from kyb.witness.norm import contains, contains_wrapped
from kyb.workspace.catalog import Catalog
from kyb.workspace.runs import run_meta

PAD = 12
W1_REASON = "W2 needs reader and witness families that are both known and different"


def _clip(b, size):
    w, h = size
    return (max(0, b[0] - PAD), max(0, b[1] - PAD), min(w, b[2] + PAD), min(h, b[3] + PAD))


def _readable(r: dict) -> bool:
    return bool(r.get("readable")) and bool((r.get("transcription") or "").strip())


def _verify_document(root: Path, case: str, run: str, doc: str, witness, pdf: Path) -> dict:
    sub = json.loads(contained(root, claims_rel(case, run, doc)).read_text("utf-8"))
    layout = load_layout(root, case, run, doc)
    blocks = {b.block_id: (pl.page, b, pl.size_px) for pl in layout.pages for b in pl.blocks}
    page_text = {}
    for pl in layout.pages:
        meta = contained(root, f"silver/pages/{case}/{run}/{doc}/p{pl.page}.json").read_text()
        page_text[pl.page] = json.loads(meta)["text_layer"] or ""
    crops: list[bytes] = []
    slot: dict[str, str] = {}  # opaque crop id -> claim id, or ctx:<claim>:<block>
    with tempfile.TemporaryDirectory(prefix="kyb-crops-") as tmp:

        def cut(page: int, bbox, size, what: str) -> None:
            k = f"k{len(crops) + 1}"
            out = poppler.render_region(pdf, page, _clip(bbox, size), Path(tmp) / f"{k}.png")
            crops.append(out.read_bytes())
            slot[k] = what

        for c in sub["claims"]:
            cut(c["page"], c["bbox"], blocks[c["block_id"]][2], c["claim_id"])
            for lb in c.get("linked_blocks", []):
                page, b, size = blocks[lb]
                cut(page, b.bbox, size, f"ctx:{c['claim_id']}:{lb}")
    resp = witness.ask("transcribe", crops, {"crop_ids": list(slot)}) if crops else None
    got: dict[str, dict] = {}
    for r in resp.data.get("crops", []) if resp else []:
        if isinstance(r, dict) and r.get("crop_id") in slot:
            got[slot[r["crop_id"]]] = r
    # what the document says about its own entity, read blind: layout headings and facts, the
    # witnessed transcription of claimed title blocks, and every crop's re-read tuples
    titles = [got[c["claim_id"]]["transcription"] for c in sub["claims"]
              if blocks[c["block_id"]][1].kind in ("title", "heading")
              and _readable(got.get(c["claim_id"], {}))]  # fmt: skip
    headings = [b.heading or "" for pl in layout.pages for b in pl.blocks]
    scope = Scope(headings + titles, layout.material_facts, got)
    results, findings = [], []
    for c in sub["claims"]:
        w = got.get(c["claim_id"], {})
        layer = "n/a"
        t_ok = _readable(w) and contains(w["transcription"], c["excerpt"])
        if c["source"] == "text_layer":
            t_ok = _readable(w) and contains_wrapped(
                w["transcription"], c["excerpt"], page_text.get(c["page"], "")
            )
            layer = "match" if contains(page_text.get(c["page"], ""), c["excerpt"]) else "mismatch"
            if layer == "match" and not t_ok:
                findings.append({"kind": "layer_vs_pixel", "claim_id": c["claim_id"],
                                 "detail": "the layer says it; the pixels do not"})  # fmt: skip
        block = blocks[c["block_id"]][1]
        linked = c.get("linked_blocks", [])
        missing = [r for r in block.refers_to if r not in linked]
        unread = [lb for lb in linked if not _readable(got.get(f"ctx:{c['claim_id']}:{lb}", {}))]
        qualifiers = sorted({q for b in [block, *(blocks[lb][1] for lb in linked)]
                             for q in b.qualifiers})  # fmt: skip
        if missing or unread:
            binding = "incomplete"
        elif bind_claim(c, w.get("reread") or {}, w.get("transcription") or "", scope):
            binding = "match"
        else:
            binding = "mismatch"
        ok = t_ok and binding == "match" and layer in ("n/a", "match")
        results.append({"claim_id": c["claim_id"],
                        "transcription": "witnessed" if t_ok else "failed", "text_layer": layer,
                        "binding": binding, "missing_context": missing, "unread_context": unread,
                        "context_qualifiers": qualifiers, "verified": ok})  # fmt: skip
    ok_ids = {r["claim_id"] for r in results if r["verified"]}
    verified = [c for c in sub["claims"] if c["claim_id"] in ok_ids]
    coverage = match_coverage([f.model_dump() for f in layout.material_facts], verified)
    uncovered = [x for x in coverage if x["status"] == "uncovered"]
    if uncovered:
        findings.append({"kind": "coverage_incomplete", "doc": doc, "uncovered": uncovered})
    if not layout.material_facts:
        findings.append({"kind": "coverage_unestablished", "doc": doc,
                         "detail": "the layout lists no material facts"})  # fmt: skip
    witness_family = model_family(resp.model) if resp else "unknown"
    now = assurance(root, sub, witness_family)
    record = {
        "document_id": f"{case}/{doc}", "model": resp.model if resp else None,
        "request": resp.request_hash if resp else None, "comparator": NORM, "binding": BIND,
        "coverage_version": COVERAGE,
        "saw_claim": False, "saw_text_layer": False, "crop_ids": "opaque positional",
        # the assurance when witnessed: a snapshot. Read it with `assurance_now`, which
        # re-derives it from the chain (later evidence can only lower it)
        "assurance": now["assurance"], "assurance_reason": now["assurance_reason"],
        "reader_model": now["reader_model"], "reader_family": now["reader_family"],
        "witness_family": witness_family,
        "provenance": getattr(resp, "provenance", {}) if resp else {},
        "crops": [{"crop_id": k, "for": slot[k], **got.get(slot[k], {})} for k in slot],
        "results": results, "coverage": coverage, "findings": findings,
    }  # fmt: skip
    if resp:
        record_usage(root, "witness", case, run, doc, resp)
    return record


def _blocked(root: Path, case: str, run: str, doc: str, reason: str) -> dict:
    """An explicit immutable refusal is visible to facts/judges, never a hidden skipped claim."""
    sub = json.loads(contained(root, claims_rel(case, run, doc)).read_text("utf-8"))
    layout = load_layout(root, case, run, doc)
    now = assurance(root, sub, "unknown")
    return {"document_id": f"{case}/{doc}", "model": None, "request": None,
            "comparator": NORM, "binding": BIND, "coverage_version": COVERAGE, "saw_claim": False,
            "saw_text_layer": False, **now, "witness_family": "unknown", "crops": [],
            "results": [{"claim_id": c["claim_id"], "verified": False,
                         "self_referential": False, "transcription": "not_witnessed",
                         "binding": "incomplete", "text_layer": "not_checked"}
                        for c in sub["claims"]],
            "coverage": [{**f.model_dump(), "status": "uncovered"}
                         for f in layout.material_facts],
            "findings": [{"kind": "witness_routing_refused", "detail": reason}],
            "self_referential": False, "routing_refused": True}  # fmt: skip


def verify_document(root: Path, case: str, run: str, doc: str, witness, pdf: Path) -> dict:
    """Route only after attestation, retaining both strata and all refused attempts."""
    if not hasattr(witness, "for_document"):
        return _verify_document(root, case, run, doc, witness, pdf)
    from kyb.witness.strata import RoutingRefused

    routing_config = {"version": ROUTING_VERSION, "stratum": witness.config.stratum,
                      "a_backends": list(witness.config.a_backends)}  # fmt: skip
    sub = json.loads(contained(root, claims_rel(case, run, doc)).read_text("utf-8"))
    try:
        routes = witness.for_document(sub)
    except RoutingRefused as exc:
        return {**_blocked(root, case, run, doc, str(exc)), "stratum": witness.config.stratum,
                "routing_config": routing_config}  # fmt: skip
    records = {}
    for stratum, port in routes.items():
        try:
            rec = _verify_document(root, case, run, doc, port, pdf)
        except WitnessAnomalous as exc:
            # Unknown envelope shapes never verify a claim: every claim stays CLAIMED.
            rec = _blocked(root, case, run, doc, str(exc))
            rec["findings"] = [{"kind": "witness_envelope_anomalies", "detail": str(exc),
                                "count": len(exc.anomalies)}]  # fmt: skip
            rec["envelope_anomalous"] = True
        except (RoutingRefused, RuntimeError, LookupError, ValueError) as exc:
            rec = _blocked(root, case, run, doc, str(exc))
        rec |= {"stratum": stratum, "self_referential": stratum == "B",
                "attempts": port.attempts, "routing_config": routing_config}  # fmt: skip
        for r in rec["results"]:
            r["self_referential"] = stratum == "B"
        if stratum == "B":
            rec["assurance"] = "W1"
            if not rec.get("routing_refused"):
                rec["assurance_reason"] = "same actual model as reader (spec stratum B)"
            if any(r["verified"] for r in rec["results"]):
                rec["findings"].append({"kind": "self_referential_verification"})
        records[stratum] = rec
    if len(records) == 1:
        return next(iter(records.values()))
    a, b = records["A"], records["B"]
    out = {**a, "stratum": "A+B", "strata": records, "self_referential": False,
           "results": [], "findings": [*a["findings"],
               *(f for f in b["findings"]
                 if f["kind"] != "self_referential_verification")]}  # fmt: skip
    for ar, br in zip(a["results"], b["results"], strict=True):
        # Both strata must independently verify the same claim. Different blind re-reads
        # are also surfaced; they cannot quietly rescue each other's unsupported reading.
        disagree = ar["verified"] != br["verified"]
        if disagree:
            out["findings"].append({"kind": "witness_strata_disagreement",
                                     "claim_id": ar["claim_id"]})  # fmt: skip
        out["results"].append({**ar, "verified": ar["verified"] and br["verified"],
                               "self_referential": False,
                               "strata": {"A": ar, "B": br}})  # fmt: skip
    # Compare blind normalized readings, beyond the boolean acceptance verdict.
    from kyb.witness.norm import norm

    def readings(rec):
        def normalize(value):
            if isinstance(value, str):
                return norm(value)
            if isinstance(value, dict):
                return {k: normalize(v) for k, v in value.items()}
            if isinstance(value, list):
                return sorted((normalize(v) for v in value), key=sha256_json)
            return value

        out = {}
        for crop in rec["crops"]:
            reread = crop.get("reread") or {}
            values = [{"struck": False, **v} for v in reread.get("values", [])]
            out[crop["crop_id"]] = normalize(
                {
                    "transcription": crop.get("transcription", ""),
                    "readable": crop.get("readable"),
                    "values": values,
                    "struck": reread.get("struck") or [],
                    "annotations": reread.get("annotations") or [],
                }
            )
        return out

    if readings(a) != readings(b):
        out["findings"].append(
            {"kind": "witness_strata_disagreement", "detail": "blind readings differ"}
        )
    if a.get("routing_refused") or b.get("routing_refused"):
        out["routing_refused"] = True
        out["assurance"] = "W1"
    return out


def assurance(root: Path, sub: dict, witness_family: str) -> dict:
    """W1/W2 for a document from the chain as it stands now (spec, spec).

    W2 needs a submission attested `isolated` that is still eligible (one known hook
    registration in its session, no resume: `harness.eligibility`), a reader model taken only
    from complete SubagentStop transcripts (`harness.stopped_model`, never the session's), and
    reader and witness families that are known and different. Evidence recorded later (a second
    registration, a resume, a mixed or unreadable stop snapshot) lowers it; nothing raises it.
    """
    events = chain_events(root)
    reader = sub["produced_by"]["reader"]
    session, agent = reader.get("session"), reader.get("agent_id")
    reasons = [] if reader.get("context") == "isolated" else ["the submission is unattested"]
    reasons += eligibility(events, session, agent) if not reasons else []
    reader_model = stopped_model(events, session, agent) if not reasons else None
    reader_family = model_family(reader_model)
    from kyb.codex.managed import identity

    managed = identity(events, session, agent) if not reasons else None
    managed_reader = managed is not None and not managed["reasons"]
    w2 = (not reasons and reader_family != witness_family
          and "unknown" not in (reader_family, witness_family))  # fmt: skip
    return {"assurance": "W2" if w2 else "W1",
            "assurance_reason": "" if w2 else "; ".join(reasons) or W1_REASON,
            "reader_model": reader_model, "reader_family": reader_family,
            "reader_model_scope": "served-response" if managed_reader else "served-model",
            "reader_origin": "codex-managed" if managed_reader else "harness-subagent",
            "context_now": "isolated" if not reasons else "unattested"}  # fmt: skip


def anomaly_summary(rec: dict) -> dict:
    """Envelope anomalies across every recorded attempt of a document (count and severities)."""
    found: list[dict] = []
    records = (rec.get("strata") or {}).values() if rec.get("strata") else [rec]
    for r in records:
        for attempt in r.get("attempts") or []:
            found.extend(a for a in attempt.get("anomalies") or [] if isinstance(a, dict))
    severities: dict[str, int] = {}
    for a in found:
        severities[a.get("severity", "unknown")] = (
            severities.get(a.get("severity", "unknown"), 0) + 1
        )
    return {"count": len(found), "severities": severities}


def assurance_now(root: Path, case: str, run: str, doc: str) -> dict:
    """The document's assurance as of now: every consumer (decide, dogfood) reads this, not the
    witness record's snapshot."""
    sub = json.loads(contained(root, claims_rel(case, run, doc)).read_text("utf-8"))
    rec = json.loads(contained(root, f"silver/witness/{case}/{run}/{doc}.json").read_text("utf-8"))
    check_coverage_record(rec)
    if rec.get("stratum") and (rec.get("routing_config") or {}).get("version") != ROUTING_VERSION:
        raise ValueError("routed witness protocol changed; verification needs a new run")
    now = assurance(root, sub, rec["witness_family"])
    if rec.get("stratum") == "B" or rec.get("routing_refused"):
        now |= {"assurance": "W1", "assurance_reason": rec.get("assurance_reason", "route refused")}
    return {**now, "at_witness": rec["assurance"],
            "self_referential": bool(rec.get("self_referential")),
            "anomalies": anomaly_summary(rec)}  # fmt: skip


def _reuse(root: Path, catalog: Catalog, rel: str, claims: str, layout: str) -> dict:
    """A completed result counts only if it and every input still hash to their receipts."""
    entry = catalog.get(rel)
    for path in (rel, claims, layout):
        recorded = catalog.get(path)["content_hash"]
        if sha256_bytes(contained(root, path).read_bytes()) != recorded:
            raise PermissionError(f"{path}: bytes differ from its catalog receipt; not reusing")
    want = sorted([catalog.ref(claims), catalog.ref(layout)])
    if sorted(entry["inputs"]) != want:
        raise PermissionError(f"{rel}: was witnessed against different inputs; not reusing")
    record = json.loads(contained(root, rel).read_text("utf-8"))
    check_coverage_record(record)
    return record


def verify_case(root: Path, case: str, run: str, witness) -> dict:
    """Witness every document's claims. Call under the workspace action lock."""
    plan = read_plan(root, case, run)
    missing = [e["doc"] for e in plan if not submitted(root, case, run, e["doc"])]
    if missing:
        raise LookupError(f"{case}: no submission yet for {missing}")
    files = dict(_documents(root, case))
    catalog, as_of = Catalog(root), run_meta(root, case, run)["as_of"]
    summary = {"case": case, "run": run, "documents": {}}
    with tempfile.TemporaryDirectory(prefix="kyb-verify-") as tmp:
        for e in plan:
            src = f"bronze/packets/{case}/{files[e['doc']]['name']}"
            data = contained(root, src).read_bytes()
            if sha256_bytes(data) != e["doc_sha256"]:
                raise PermissionError(f"{src}: bronze bytes do not match the manifest")
            pdf = Path(tmp) / f"{e['doc']}.pdf"
            pdf.write_bytes(data)
            rel = f"silver/witness/{case}/{run}/{e['doc']}.json"
            if catalog.get(rel) is not None:  # already witnessed: reuse only if nothing changed
                if witness.mode == "live":
                    raise ValueError(f"{case} {run}: already verified; a live re-read is a new run")
                rec = _reuse(root, catalog, rel, claims_rel(case, run, e["doc"]), e["layout"])
                if hasattr(witness, "config") and rec.get("routing_config") != {
                    "version": ROUTING_VERSION,
                    "stratum": witness.config.stratum,
                    "a_backends": list(witness.config.a_backends),
                }:
                    raise ValueError("witness routing config changed; verification needs a new run")
            else:
                rec = verify_document(root, case, run, e["doc"], witness, pdf)
                body = (json.dumps(rec, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
                publish_exclusive(contained(root, rel), body)
                by = {"action": "verify_quotes", "model": rec["model"], "comparator": NORM,
                      "binding": BIND, "coverage_version": COVERAGE}  # fmt: skip
                inputs = [catalog.ref(claims_rel(case, run, e["doc"])), catalog.ref(e["layout"])]
                catalog.register(rel, tier="verified", produced_by=by, as_of=as_of, inputs=inputs)
            uncovered = any(x["status"] == "uncovered" for x in rec["coverage"])
            summary["documents"][e["doc"]] = {
                "verified": sum(r["verified"] for r in rec["results"]),
                "claims": len(rec["results"]),
                "coverage": "incomplete" if uncovered else "complete",
                "conflicts": sum(x["status"] == "conflict" for x in rec["coverage"]),
                "findings": len(rec["findings"]),
                "assurance": assurance_now(root, case, run, e["doc"])["assurance"],
                "assurance_at_witness": rec["assurance"],
                "stratum": rec.get("stratum"),
                "self_referential": bool(rec.get("self_referential")),
                "anomalies": anomaly_summary(rec),
            }  # fmt: skip
    rel = f"silver/verified_claims/{case}/{run}.json"
    # the receipt records what verification established then; assurance can only be lowered
    # later, so it is kept as `assurance_at_verify` and read through `assurance_now`
    record = {**summary, "assurance_note": "historical; read current assurance via assurance_now",
              "documents": {d: {k: v for k, v in r.items() if k != "assurance"}
                            | {"assurance_at_verify": r["assurance"]}
                            for d, r in summary["documents"].items()}}  # fmt: skip
    body = (json.dumps(record, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    if catalog.get(rel) is None:
        publish_exclusive(contained(root, rel), body)
        inputs = [catalog.ref(f"silver/witness/{case}/{run}/{e['doc']}.json") for e in plan]
        catalog.register(rel, tier="verified", produced_by={"action": "verify_quotes"},
                         as_of=as_of, inputs=inputs)  # fmt: skip
        AuditLog(root / "ledger" / "audit.jsonl").append(
            "witness", "tool.call", input_hash=sha256_json(plan), output_hash=sha256_bytes(body),
            detail={"action": "verify_quotes", "case": case, "run": run,
                    "verified": sum(d["verified"] for d in summary["documents"].values())},
        )  # fmt: skip
    return summary
