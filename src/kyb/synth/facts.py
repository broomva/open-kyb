"""CaseSpec → typed FactSet (b4 contract), so `kyb.policy` can evaluate a synthetic case directly.

This is the fixture path the policy tests use for A/B/C, generalised: no reader, no witness, no
model. Each spec fact becomes an entity, holding, role, document or conflict with a verified
evidence reference into the document that states it. Whatever the policy engine then decides is the
real engine's reading of those facts (`FixtureDecider`); the label engine never sees this module.
"""

from __future__ import annotations

from typing import Any

from kyb.contracts import b4
from kyb.synth.case import DOC_KINDS, CaseSpec, Holder, Person
from kyb.synth.label import code_of, conflict_items, region_of

PRESENT: dict[str, Any] = {"state": "present"}
ABSENT: dict[str, Any] = {"state": "absent"}


def doc_key(kind: str) -> str:
    return f"{DOC_KINDS.index(kind) + 1:02d}"


class _Refs:
    """Allocates one claim per (document, ordinal); `table()` resolves them as EvidenceItems."""

    def __init__(self, case: CaseSpec) -> None:
        self.case = case
        self.n: dict[str, int] = {}
        self.items: dict[str, dict] = {}

    def doc_for(self, *prefer: str) -> str:
        for kind in (*prefer, *DOC_KINDS):
            if self.case.doc(kind) is not None:
                return doc_key(kind)
        return "01"

    def ref(self, doc: str) -> dict:
        self.n[doc] = self.n.get(doc, 0) + 1
        cid = f"c-{self.case.case_id}{doc}-{self.n[doc]:02d}"
        self.items[cid] = {"claim_id": cid, "doc": doc, "page": 1, "verified": True}
        return self.items[cid]

    def table(self) -> list[dict]:
        names = {doc_key(k): f"{doc_key(k)}_{k}.pdf" for k in DOC_KINDS}
        return [
            {
                "evidence_id": cid,
                "doc": r["doc"],
                "document_name": names.get(r["doc"], "x.pdf"),
                "page": 1,
                "bbox": (0, 0, 10, 10),
                "excerpt": cid,
                "verified": True,
            }
            for cid, r in sorted(self.items.items())
        ]


def _person_entity(p: Person, refs: _Refs, doc: str) -> dict:
    d: dict[str, Any] = {
        "entity_id": f"e-{p.pid}",
        "kind": "person",
        "names": [p.name],
        "nationality": [p.nationality],
        "dob": [p.dob.isoformat()],
        "jurisdiction": [code_of(p.domicile)],
        "evidence": [refs.ref(doc)],
    }
    if region_of(p.domicile):
        d["regions"] = [region_of(p.domicile)]
    return d


def _holder_entities(
    h: Holder, refs: _Refs, doc: str, ents: dict[str, dict], holdings: list, roles: list
) -> None:
    """The holder, its chain, and the person it resolves to (if any), as entities and holdings."""
    app = "e-app"
    kind = {"natural": "person", "legal": "company", "foundation": "foundation", "trust": "trust"}[
        h.kind
    ]
    nominee = h.nominee
    top = f"e-{h.hid}"
    if h.kind == "natural" and h.person is not None and not nominee:
        ents.setdefault(f"e-{h.person.pid}", _person_entity(h.person, refs, doc))
        holdings.append(_holding(f"hd-{h.hid}", f"e-{h.person.pid}", app, h.pct, refs, doc))
        return
    ents[top] = {
        "entity_id": top,
        "kind": "nominee" if nominee else kind,
        "names": [h.name],
        "jurisdiction": [code_of(h.domicile)],
        "evidence": [refs.ref(doc)],
        **({"regions": [region_of(h.domicile)]} if region_of(h.domicile) else {}),
        **({"identifiers": {"reg_no": [h.reg_no]}} if h.reg_no else {}),
    }
    holdings.append(_holding(f"hd-{h.hid}", top, app, h.pct, refs, doc))
    if nominee:
        roles.append(_role(f"rl-{h.hid}", top, app, "nominee_shareholder", refs, doc))
    below = top
    for i, name in enumerate(h.chain):
        eid = f"e-{h.hid}-c{i}"
        ents[eid] = {
            "entity_id": eid,
            "kind": "company",
            "names": [name],
            "jurisdiction": [code_of(h.domicile)],
            "evidence": [refs.ref(doc)],
        }
        holdings.append(_holding(f"hd-{h.hid}-c{i}", eid, below, 100, refs, doc))
        below = eid
    if h.person is not None:
        pid = f"e-{h.person.pid}"
        ents.setdefault(pid, _person_entity(h.person, refs, doc))
        if kind in ("foundation", "trust"):
            roles.append(_role(f"rl-{h.hid}-b", pid, below, "beneficiary", refs, doc))
        else:
            holdings.append(_holding(f"hd-{h.hid}-p", pid, below, 100, refs, doc))


def _holding(hid: str, owner: str, owned: str, pct: int, refs: _Refs, doc: str) -> dict:
    return {
        "holding_id": hid,
        "owner": owner,
        "owned": owned,
        "pct": pct,
        "status": "current",
        "evidence": [refs.ref(doc)],
    }


def _role(rid: str, holder: str, of: str, kind: str, refs: _Refs, doc: str) -> dict:
    return {"role_id": rid, "holder": holder, "of": of, "role": kind, "evidence": [refs.ref(doc)]}


def _tri(present: bool, refs: _Refs, doc: str) -> dict:
    return {**(PRESENT if present else ABSENT), "evidence": [refs.ref(doc)]}


def to_factset(case: CaseSpec) -> b4.FactSet:
    c = case.company
    refs = _Refs(case)
    d_inc = refs.doc_for("incorporation")
    d_reg = refs.doc_for("share_register", "ubo_declaration")
    ents: dict[str, dict] = {}
    holdings: list[dict] = []
    roles: list[dict] = []

    addresses = [
        {
            "kind": "registered",
            "text": c.registered_address,
            "evidence": [refs.ref(d_inc)],
            "mass_registration": _tri(c.mass_registration_address, refs, d_inc),
        }
    ]
    if c.operating_address and c.operating_address != c.registered_address:
        addresses.append(
            {
                "kind": "operating",
                "text": c.operating_address,
                "evidence": [refs.ref(d_inc)],
                "mass_registration": ABSENT,
            }
        )
    ents["e-app"] = {
        "entity_id": "e-app",
        "kind": "company",
        "names": [c.name],
        "jurisdiction": [c.jurisdiction],
        "incorporated": [c.incorporated.isoformat()],
        "identifiers": {"reg_no": [c.tax_id]},
        "addresses": addresses,
        "evidence": [refs.ref(d_inc)],
        **({"regions": [c.region]} if c.region else {}),
    }

    for h in case.holders:
        _holder_entities(h, refs, d_reg, ents, holdings, roles)
    for i, o in enumerate(case.officers):
        if o.person is not None:
            ents.setdefault(f"e-{o.person.pid}", _person_entity(o.person, refs, d_inc))
            holder = f"e-{o.person.pid}"
        else:
            holder = f"e-off-{abs(hash_name(o.name))}"
            ents.setdefault(
                holder,
                {
                    "entity_id": holder,
                    "kind": "agent" if o.role == "registered_agent" else "nominee",
                    "names": [o.name],
                    "evidence": [refs.ref(d_inc)],
                },
            )
        roles.append(_role(f"rl-o{i}", holder, "e-app", o.role, refs, d_inc))

    by_name = {
        h.name: (
            f"e-{h.person.pid}"
            if h.kind == "natural" and h.person and not h.nominee
            else f"e-{h.hid}"
        )
        for h in case.holders
    }
    conflicts = []
    for i, (subject, attr, vals) in enumerate(conflict_items(case), start=1):
        sid = "e-app" if subject == "applicant" else by_name.get(subject, "e-app")
        conflicts.append(
            {
                "conflict_id": f"x-{i}",
                "subject": sid,
                "attribute": attr,
                "values": [{"value": v, "evidence": [refs.ref(doc_key(k))]} for k, v in vals],
            }
        )

    docs = []
    for d in case.documents:
        if not d.present:
            continue
        key = doc_key(d.kind)
        docs.append(
            {
                "doc": key,
                "document_name": f"{key}_{d.kind}.pdf",
                "doc_type": d.kind,
                "issued": d.issued.isoformat() if d.issued else None,
                "signed": d.signed,
                "stated_validity_days": d.stated_validity_days,
                "evidence": [refs.ref(key)],
            }
        )

    bank = case.doc("bank_reference") is not None
    operating = bool(c.operating_address) and c.operating_address != c.registered_address
    body = {
        "case": case.case_id,
        "run": "r1",
        "as_of": case.as_of.isoformat(),
        "fact_set_hash": f"sha256:{case.case_id}",
        "applicant": "e-app",
        "entities": list(ents.values()),
        "holdings": holdings,
        "roles": roles,
        "documents": docs,
        "conflicts": conflicts,
        "operations": {
            "distinct_operating_address": _tri(operating, refs, d_inc),
            "bank_reference_or_financials": _tri(bank, refs, d_inc),
            "employees_or_website": _tri(c.employees_or_website, refs, d_inc),
        },
        "corporate_object": list(c.activities),
        "evidence": [],
        "verification": [],
    }
    body["evidence"] = refs.table()
    return b4.FactSet.model_validate(body)


def hash_name(name: str) -> int:
    """A stable small id for an officer entity (builtin hash() is salted per process)."""
    return sum((i + 1) * ord(ch) for i, ch in enumerate(name))
