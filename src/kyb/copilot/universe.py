"""What the copilot may cite: facts and verified claims, witnessed snippets, recorded decisions.

Everything here is read from receipted records and recomputed on each call. `record_answer`
never trusts the bundle an agent was shown: it rebuilds the universe and checks against that.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from kyb.contracts.b4 import FACTS_REL, EvidenceItem, EvidenceRef, FactSet
from kyb.read.schema import LayoutRecord
from kyb.signing.ledger import read_record
from kyb.util.fsutil import safe_id
from kyb.util.hashing import sha256_json
from kyb.witness.bind import value_text
from kyb.witness.norm import norm
from kyb.workspace.catalog import Catalog
from kyb.workspace.runs import safe_run_id

EXCERPTS_SHOWN = 2
SPANS_PREFIX = "silver/spans/{case}/{run}/"  # text-layer lines the witness read (spans.py)


@dataclass(frozen=True)
class Cite:
    """One place in a document: what the bundle and a rendered answer show for a citation."""

    evidence_id: str
    document_name: str
    page: int
    bbox: tuple[int, int, int, int]
    excerpt: str

    def view(self) -> dict:
        return {"evidence_id": self.evidence_id, "document": self.document_name,
                "page": self.page, "bbox": list(self.bbox), "excerpt": self.excerpt}  # fmt: skip


@dataclass(frozen=True)
class Fact:
    """A fact (or verified claim) with the subject spellings it may be stated under."""

    fact_id: str
    subjects: tuple[str, ...]  # every spelling the documents use; subjects[0] is displayed
    attribute: str
    value: str
    note: str = ""
    cites: tuple[Cite, ...] = ()

    def text(self) -> str:
        return " ".join([*self.subjects, self.attribute, self.value, self.note])

    def index_text(self) -> str:
        """What retrieval matches a question against: the fact, plus the verified lines it cites
        (a question often uses the page's words, not the fact's)."""
        return " ".join([self.text(), *(c.excerpt for c in self.cites)])

    def index_where(self) -> str:
        """The documents it cites: a weaker signal (`lexicon.WHERE_WEIGHT`) than its content."""
        return " ".join(dict.fromkeys(c.document_name for c in self.cites))


@dataclass(frozen=True)
class Snippet:
    """A span of a page that a recorded pixel transcription contains verbatim (norm-v2)."""

    snippet_id: str
    doc: str
    document_name: str
    page: int
    bbox: tuple[int, int, int, int]
    page_hash: str
    text: str
    witness_request: str

    def index_text(self) -> str:
        return self.text

    def index_where(self) -> str:
        return self.document_name

    def view(self) -> dict:
        return {"snippet_id": self.snippet_id, "document": self.document_name, "page": self.page,
                "bbox": list(self.bbox), "page_hash": self.page_hash, "text": self.text,
                "witness_request": self.witness_request}  # fmt: skip


@dataclass
class Universe:
    case: str
    run: str
    facts: dict[str, Fact] = field(default_factory=dict)
    snippets: dict[str, Snippet] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)  # why part of the record was not used
    integrity: list[dict] = field(default_factory=list)  # layer-vs-pixel disagreements
    as_of: str | None = None  # the facts' frozen as_of, when facts exist


def _fact_id(kind: str, *parts: str) -> str:
    return f"f-{sha256_json([kind, *parts])[7:17]}"


def _cite(item: EvidenceItem) -> Cite:
    return Cite(item.evidence_id, item.document_name, item.page, item.bbox, item.excerpt)


def _load_facts(root: Path, case: str, run: str) -> FactSet | None:
    rel = FACTS_REL.format(case=case, run=run)
    if Catalog(root).get(rel) is None:
        return None
    facts = FactSet.model_validate(read_record(root, rel))
    body = facts.model_dump(mode="json", exclude={"fact_set_hash"})
    if sha256_json(body) != facts.fact_set_hash:
        raise PermissionError(f"{rel}: fact_set_hash does not match its content")
    return facts


class _Builder:
    def __init__(self, facts: FactSet) -> None:
        self.by_id = {e.evidence_id: e for e in facts.evidence if e.verified}
        self.by_field: dict[str, list[EvidenceItem]] = {}
        for e in self.by_id.values():
            if e.field:
                self.by_field.setdefault(e.field, []).append(e)
        self.names = {e.entity_id: tuple(e.names) or (e.entity_id,) for e in facts.entities}
        self.out: dict[str, Fact] = {}

    def cites(self, refs: list[EvidenceRef] | None = None, *fields: str) -> tuple[Cite, ...]:
        """Verified evidence for a fact: its references, else everything filed under its field.
        Unverified claims are never cited (spec)."""
        items = [self.by_id[r.claim_id] for r in refs or [] if r.claim_id in self.by_id]
        for f in fields:
            items += self.by_field.get(f, [])
        seen: dict[str, EvidenceItem] = {i.evidence_id: i for i in items}
        return tuple(_cite(seen[k]) for k in sorted(seen))

    def add(self, kind: str, subjects: tuple[str, ...], attribute: str, value: str, cites,
            note: str = "", key: str = "") -> None:  # fmt: skip
        if not cites or not value.strip():  # an uncited fact is not an answer
            return
        fid = _fact_id(kind, key or subjects[0], attribute, value)
        self.out[fid] = Fact(fid, subjects, attribute, value, note, cites)

    def name(self, eid: str) -> str:
        return self.names.get(eid, (eid,))[0]


def _entity_facts(b: _Builder, facts: FactSet) -> None:
    for e in facts.entities:
        subj, ev = b.names[e.entity_id], f"entity[{e.entity_id}]"
        for n in e.names:
            b.add("name", subj, "name", n, b.cites(None, f"{ev}.names"), key=e.entity_id)
        for attr in ("jurisdiction", "nationality", "dob", "incorporated", "regions"):
            for v in getattr(e, attr):
                b.add(attr, subj, attr, v, b.cites(None, f"{ev}.{attr}"), key=e.entity_id)
        for kind, values in e.identifiers.items():
            for v in values:
                cites = b.cites(None, f"{ev}.identifiers.{kind}")
                b.add("id", subj, f"identifier:{kind}", v, cites, key=e.entity_id)
        for a in e.addresses:
            b.add("address", subj, f"address:{a.kind}", a.text, b.cites(a.evidence),
                  note=a.mass_registration.note, key=e.entity_id)  # fmt: skip


def _relation_facts(b: _Builder, facts: FactSet) -> None:
    for h in facts.holdings:
        pct = f"{h.pct.normalize():f}%" if h.pct is not None else "an undisclosed share"
        shares = f", {h.shares} shares" if h.shares is not None else ""
        note = f"status {h.status}" + (f", as of {h.as_of}" if h.as_of else "")
        b.add("holding", b.names[h.owner], "owns", f"{pct} of {b.name(h.owned)}{shares}",
              b.cites(h.evidence, f"holding[{h.holding_id}].pct", f"holding[{h.holding_id}]"),
              note=note, key=h.holding_id)  # fmt: skip
    for r in facts.roles:
        b.add("role", b.names[r.holder], "role", f"{r.role} of {b.name(r.of)}",
              b.cites(r.evidence, f"role[{r.role_id}]"), key=r.role_id)  # fmt: skip
    for c in facts.conflicts:
        refs = [ref for v in c.values for ref in v.evidence]
        b.add("conflict", b.names[c.subject], f"conflict:{c.attribute}",
              " | ".join(v.value for v in c.values), b.cites(refs),
              note="the documents disagree; unresolved (spec)", key=c.conflict_id)  # fmt: skip


def _document_facts(b: _Builder, facts: FactSet) -> None:
    for d in facts.documents:
        subj = (d.document_name, d.doc)
        for attr in ("doc_type", "issued", "signed_on", "stated_validity_days"):
            v = getattr(d, attr)
            if v is not None:
                b.add("doc", subj, attr, str(v), b.cites(d.evidence, f"document[{d.doc}].{attr}"),
                      key=d.doc)  # fmt: skip
    app = b.names[facts.applicant]
    for text in facts.corporate_object:
        b.add("object", app, "corporate_object", text, b.cites(None, "corporate_object"),
              key=facts.applicant)  # fmt: skip
    for name, tri in facts.operations:
        b.add("ops", app, f"operations:{name}", tri.state, b.cites(tri.evidence), note=tri.note,
              key=facts.applicant)  # fmt: skip


def _claim_facts(root: Path, case: str, run: str, out: dict[str, Fact]) -> None:
    """Verified claims, by their own subject, attribute and value (the witness's verdict)."""
    catalog = Catalog(root)
    for rel in sorted(catalog.entries()):
        pre = f"silver/claims/{case}/{run}/"
        if not (rel.startswith(pre) and rel.endswith(".json")):
            continue
        doc = Path(rel).stem
        wrel = f"silver/witness/{case}/{run}/{doc}.json"
        if catalog.get(wrel) is None:
            continue
        verified = {r["claim_id"] for r in read_record(root, wrel)["results"] if r["verified"]}
        name = _doc_name(root, case, doc)
        for c in read_record(root, rel)["claims"]:
            if c["claim_id"] not in verified:
                continue
            cite = Cite(c["claim_id"], name, c["page"], tuple(c["bbox"]), c["excerpt"])
            attr = c.get("attribute") or c["type"]
            out[c["claim_id"]] = Fact(c["claim_id"], (c["subject"],), attr, value_text(c["value"]),
                                      cites=(cite,))  # fmt: skip


def _doc_name(root: Path, case: str, doc: str) -> str:
    from kyb.read.plan import _documents

    return dict(_documents(root, case)).get(doc, {}).get("name", doc)


def _snippets(root: Path, case: str, run: str) -> dict[str, Snippet]:
    """The lines of every readable crop transcription the witness recorded: pixel readings,
    each bound to its page by the claim's (or the context block's) location and the layout's
    page hash. Nothing here comes from a text layer."""
    from kyb.read.plan import load_layout

    out: dict[str, Snippet] = {}
    catalog = Catalog(root)
    for rel in sorted(catalog.entries()):
        if not (rel.startswith(f"silver/witness/{case}/{run}/") and rel.endswith(".json")):
            continue
        doc = Path(rel).stem
        layout, rec = load_layout(root, case, run, doc), read_record(root, rel)
        crel = f"silver/claims/{case}/{run}/{doc}.json"
        if layout is None or catalog.get(crel) is None:
            continue
        claims = {c["claim_id"]: c for c in read_record(root, crel)["claims"]}
        out.update(_doc_snippets(root, case, doc, rec, claims, layout))
    return out


def _doc_snippets(root: Path, case: str, doc: str, rec: dict, claims: dict,
                  layout: LayoutRecord) -> dict[str, Snippet]:  # fmt: skip
    hashes = {p.page: p.page_hash for p in layout.pages}
    blocks = {b.block_id: (p.page, b.bbox) for p in layout.pages for b in p.blocks}
    name, out = _doc_name(root, case, doc), {}
    for crop in rec["crops"]:
        text = (crop.get("transcription") or "").strip()
        if not crop.get("readable") or not text:
            continue
        where = crop["for"]
        if where.startswith("ctx:"):
            page, bbox = blocks[where.rsplit(":", 1)[1]]
        else:
            page, bbox = claims[where]["page"], tuple(claims[where]["bbox"])
        for line in (ln.strip() for ln in text.splitlines()):
            if line:
                sid = "sn-" + sha256_json([doc, page, list(bbox), norm(line)])[7:17]
                out.setdefault(sid, Snippet(sid, doc, name, page, tuple(bbox), hashes[page], line,
                                            rec["request"]))  # fmt: skip
    return out


def _span_snippets(root: Path, case: str, run: str) -> tuple[dict[str, Snippet], list[dict]]:
    """Text-layer lines whose exact text occurs in the witness's reading of their crop, plus the
    integrity findings for the lines the pixels read differently."""
    prefix, out, findings = SPANS_PREFIX.format(case=case, run=run), {}, []
    for rel in sorted(Catalog(root).entries()):
        if not rel.startswith(prefix):
            continue
        rec = read_record(root, rel)
        name = _doc_name(root, case, rec["doc"])
        findings += [{**f, "document": name} for f in rec["findings"]]
        for sp in rec["spans"]:
            if sp["matched"]:
                sid = (
                    "sn-"
                    + sha256_json([rec["doc"], sp["page"], sp["bbox"], norm(sp["text"])])[7:17]
                )
                out[sid] = Snippet(sid, rec["doc"], name, sp["page"], tuple(sp["bbox"]),
                                   sp["page_hash"], sp["text"], rec["request"])  # fmt: skip
    return out, findings


def load_universe(root: Path, case: str, run: str) -> Universe:
    case, run = safe_id(case, "case id"), safe_run_id(run)
    u = Universe(case, run)
    facts = _load_facts(root, case, run)
    if facts is None:
        u.notes.append(f"{case} {run} has no facts yet; only witnessed snippets are available")
    else:
        u.as_of = facts.as_of
        b = _Builder(facts)
        _entity_facts(b, facts)
        _relation_facts(b, facts)
        _document_facts(b, facts)
        u.facts.update(b.out)
    _claim_facts(root, case, run, u.facts)
    u.snippets = _snippets(root, case, run)
    spans, u.integrity = _span_snippets(root, case, run)
    u.snippets.update(spans)
    return u


def entity_stems(u: Universe) -> frozenset[str]:
    """Stems of every name the facts give a subject: the words that say whom, not what."""
    from kyb.copilot import lexicon

    return frozenset(t for f in u.facts.values() for s in f.subjects for t in lexicon.stems(s))


def decisions(root: Path, case: str) -> list[dict]:
    """The case's recorded decisions, newest first, as the pointers `explain` needs."""
    from kyb.signing.decisions import decision_paths, load_decision, signature

    rows = []
    for rel in decision_paths(root):
        try:
            d = load_decision(root, Path(rel).stem)
        except LookupError, PermissionError, ValueError:
            continue
        if d.case == case:
            sig = signature(root, d.decision_id)
            rows.append({"decision_id": d.decision_id, "run": d.run,
                         "status": d.kyb_decision.status, "risk_score": d.kyb_decision.risk_score,
                         "emitted_at": d.kyb_decision.emitted_at.isoformat(),
                         "signature_state": "signed" if sig else "pending_signature",
                         "rule_ids": [r.rule_id for r in d.rule_results]})  # fmt: skip
    return sorted(rows, key=lambda r: (r["emitted_at"], r["decision_id"]), reverse=True)
