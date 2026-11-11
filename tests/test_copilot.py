"""L5 copilot (issue): `ask` bundles and `record_answer` statement checks (agent-driven-review
§5). The property under test: nothing the record does not support is ever rendered (0
fabricated), and what it does support is rendered by code, from the record."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from decision_fixtures import decision, record, rule, workspace

from kyb.audit.chain import verify_workspace, workspace_action
from kyb.contracts.kyb_decision_schema import Citation, Violation
from kyb.copilot import lexicon
from kyb.copilot.answer import record_answer
from kyb.copilot.bundle import ask
from kyb.copilot.universe import load_universe
from kyb.workspace.catalog import Catalog
from kyb.workspace.runs import start_run

REPO = Path(__file__).resolve().parents[1]
C = "sg-strait-capital"
FIXTURE = REPO / "tests" / "fixtures" / "facts" / "sg-strait-capital.factset.json"


@pytest.fixture
def ws(qa_ws):
    return qa_ws


def answer(ws, statements, question="q"):
    with workspace_action(ws):
        return record_answer(ws, C, question, statements, "r1")


def fact_for(ws, attribute, containing=""):
    u = load_universe(ws, C, "r1")
    hits = [
        f
        for f in u.facts.values()
        if f.attribute == attribute and containing.upper() in f.value.upper()
    ]
    return min(hits, key=lambda f: f.fact_id)


def stmt(f, **over):
    return {"type": "fact", "subject": f.subjects[0], "attribute": f.attribute, "value": f.value,
            "cites": [f.fact_id], **over}  # fmt: skip


# --- ask: facts first ---------------------------------------------------------------------------


def test_ask_returns_the_cited_fact_with_its_document_page_and_excerpt(ws):
    out = ask(ws, C, "Who owns STRAIT CAPITAL PTE. LTD.?", "r1")
    top = out["facts"][0]
    assert top["subject"] in ("Sovereign Trust", "Li Wei") and top["attribute"] == "owns"
    assert top["value"].upper().startswith("100% OF STRAIT CAPITAL PTE. LTD.")
    (cite,) = top["cites"]
    assert cite["document"] == "02_register_of_members.pdf" or cite["document"].endswith(".pdf")
    assert cite["page"] >= 1 and cite["excerpt"] and len(cite["bbox"]) == 4
    assert out["empty"] is False and out["snippets"] == []


def test_the_bundle_is_deterministic(ws):
    q = "registered agent of STRAIT"
    assert ask(ws, C, q, "r1") == ask(ws, C, q, "r1")


def test_a_question_nothing_in_the_record_matches_gives_an_empty_bundle(ws):
    out = ask(ws, C, "What is the CEO's favourite colour?", "r1")
    assert out["empty"] is True and out["facts"] == [] and out["snippets"] == []


def test_a_lone_shared_word_does_not_make_an_answer(ws):
    # "company" is in the record, but the question is about something else entirely
    assert ask(ws, C, "Does the company sponsor a football team in Madagascar?", "r1")["empty"]


def test_accents_and_case_are_folded(ws):
    assert lexicon.terms("Duración DE la Sociedad") == ["duracion", "sociedad"]


def test_a_case_without_a_run_is_refused(ws):
    with pytest.raises(LookupError, match="no run yet"):
        ask(ws, "Z_unknown", "anything")


# --- record_answer: facts -----------------------------------------------------------------------


def test_a_true_fact_statement_is_rendered_by_code_with_its_evidence(ws):
    f = fact_for(ws, "owns", "STRAIT")
    out = answer(ws, [stmt(f)])
    assert out["accepted"] and out["rejected"] == []
    assert f"• {f.subjects[0]} — owns: {f.value}" in out["text"]
    assert f'    ↳ {f.cites[0].document_name} p.{f.cites[0].page}: "{f.cites[0].excerpt}"' in out[
        "text"]  # fmt: skip
    assert out["text"].endswith(f"[kyb-answer {out['answer_id']}]")
    rec = json.loads((ws / "ledger" / "qa" / f"{out['answer_id']}.json").read_text())
    assert rec["accepted"] and rec["text"] == out["text"]
    assert verify_workspace(ws, at_boundary=True).ok


def test_the_agent_cannot_add_prose_to_a_fact_statement(ws):
    f = fact_for(ws, "owns", "STRAIT")
    out = answer(ws, [stmt(f, text="and it is definitely not sanctioned")])
    assert not out["accepted"] and "text" in out["rejected"][0]["reason"]


@pytest.mark.parametrize("change", [
    {"value": "49% of STRAIT CAPITAL PTE. LTD."},  # another value
    {"subject": "Sebastián Paredes Vega"},  # another subject
    {"attribute": "role"},  # another attribute
    {"cites": ["f-0000000000"]},  # a citation that is not in the record
    {"cites": []},  # no citation at all
])  # fmt: skip
def test_a_fact_that_does_not_match_its_citation_is_rejected_with_the_reason(ws, change):
    f = fact_for(ws, "owns", "STRAIT")
    out = answer(ws, [stmt(f, **change)])
    assert not out["accepted"] and out["text"] is None and out["rejected"][0]["reason"]
    rec = json.loads((ws / "ledger" / "qa" / f"{out['answer_id']}.json").read_text())
    assert rec["accepted"] is False and rec["results"][0]["ok"] is False


def test_one_failing_statement_rejects_the_whole_answer(ws):
    good = stmt(fact_for(ws, "owns", "STRAIT"))
    bad = stmt(fact_for(ws, "role", "registered_agent"), value="chairman")
    out = answer(ws, [good, bad])
    assert not out["accepted"] and out["text"] is None
    assert [r["index"] for r in out["rejected"]] == [1]


def test_values_compare_under_norm_not_loosely(ws):
    f = fact_for(ws, "identifier:reg_no")
    assert answer(ws, [stmt(f, value=f.value.lower().upper())])["accepted"]  # case-folded
    assert not answer(ws, [stmt(f, value=f.value + "0")])["accepted"]  # one digit more


def test_every_recorded_fact_is_accepted_and_every_mutation_of_it_is_rejected(ws):
    """The fabrication property: a true statement renders, any altered one does not."""
    u = load_universe(ws, C, "r1")
    assert len(u.facts) >= 20
    fabricated = 0
    for f in u.facts.values():
        assert answer(ws, [stmt(f)])["accepted"], f
        for bad in (
            stmt(f, value=f.value + " 1"),
            stmt(f, value="x " + f.value),
            stmt(f, attribute=f.attribute + "-x"),
            stmt(f, subject="Nobody Ltd"),
        ):
            fabricated += answer(ws, [bad])["accepted"]
    assert fabricated == 0


def test_an_unverified_claim_is_never_citable(ws):
    u = load_universe(ws, C, "r1")
    fs = json.loads(FIXTURE.read_text())
    unverified = {e["evidence_id"] for e in fs["evidence"] if not e["verified"]}
    cited = {c.evidence_id for f in u.facts.values() for c in f.cites}
    assert not unverified & cited


# --- record_answer: NOT_IN_DOCUMENTS ------------------------------------------------------------


UNANSWERABLE = [
    "the CEO's favourite colour",
    "annual revenue in 2025",
    "number of employees",
    "auditor of the company",
    "tax residency in Singapore",
    "pet owned by director",
]


@pytest.mark.parametrize("topic", UNANSWERABLE)
def test_unanswerable_questions_get_the_literal_not_in_documents(ws, topic):
    out = answer(ws, [{"type": "NOT_IN_DOCUMENTS", "topic": topic}], f"What is the {topic}?")
    assert out["accepted"] and f"Not in the documents: {topic}." in out["text"]


def test_not_in_documents_is_refused_when_the_record_covers_the_topic(ws):
    out = answer(ws, [{"type": "NOT_IN_DOCUMENTS", "topic": "registered agent"}])
    assert not out["accepted"] and "covers" in out["rejected"][0]["reason"]


def test_not_in_documents_must_name_a_topic(ws):
    assert not answer(ws, [{"type": "NOT_IN_DOCUMENTS", "topic": "the of"}])["accepted"]


def test_not_in_documents_about_a_recorded_decision_must_use_explain(ws):
    record(ws, decision(C, status="MANUAL_REVIEW", rules=[rule("DOC-01")]))
    out = answer(ws, [{"type": "NOT_IN_DOCUMENTS", "topic": "why manual review"}])
    assert not out["accepted"] and "explain" in out["rejected"][0]["reason"]


def test_a_fabricated_statement_never_comes_out_of_an_unanswerable_question(ws):
    """An agent that answers anyway, with the nearest thing in the record, is rejected."""
    near = fact_for(ws, "role", "registered_agent")
    wrong = [stmt(near, value="auditor of STRAIT CAPITAL PTE. LTD."),
             {"type": "snippet", "question_attribute": "auditor", "value": "Deloitte",
              "cites": ["sn-0000000000"]}]  # fmt: skip
    for s in wrong:
        assert not answer(ws, [s], "Who is the auditor?")["accepted"]


# --- record_answer: shape ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "statements",
    [[], [{"type": "opinion", "text": "looks fine"}], [{"type": "fact"}], ["free text"], [{}]],
)
def test_malformed_answers_are_rejected_and_recorded(ws, statements):
    out = answer(ws, statements)
    assert not out["accepted"] and out["text"] is None and out["rejected"]
    assert (ws / "ledger" / "qa" / f"{out['answer_id']}.json").exists()


def test_the_same_answer_is_the_same_bytes_and_is_audited_once(ws):
    f = fact_for(ws, "owns", "STRAIT")
    first, second = answer(ws, [stmt(f)], "who owns it"), answer(ws, [stmt(f)], "who owns it")
    assert first == second
    audit = (ws / "ledger" / "audit.jsonl").read_text()
    assert audit.count('"action": "record_answer"') == 1
    assert verify_workspace(ws, at_boundary=True).ok
    assert answer(ws, [stmt(f)], "a different question")["answer_id"] != first["answer_id"]


# --- record_answer: explain ---------------------------------------------------------------------


def _decision(ws):
    cite = Citation(document_id=f"{C}/02_register.pdf", locator="p.1 bbox[1,2,3,4]@300dpi",
                    excerpt="Sovereign Trust 100%")  # fmt: skip
    v = Violation(rule_id="DOC-01", severity="REVIEW", description="DOC-01 yes", evidence=[cite])
    rules = [rule("DOC-01"), rule("GOV-01", "no"), rule("DOC-02", "unknown")]
    return record(ws, decision(C, rules=rules,
                               violations=[v], vetoes=["J-READER-ISOLATION"],
                               considered=["co-1234"]))  # fmt: skip


def test_an_explanation_is_rendered_from_the_recorded_decision(ws):
    did = _decision(ws)
    out = answer(
        ws, [{"type": "explain", "decision_id": did, "rule_ids": ["DOC-01", "GOV-01", "DOC-02"]}]
    )
    text = out["text"]
    assert out["accepted"] and f"Decision {did} for ACME: MANUAL_REVIEW" in text
    assert "pending signature" in text and "DOC-01 [REVIEW] fired" in text
    assert 'Sovereign Trust 100%"' in text  # the violation's cited excerpt

    assert "GOV-01 did not fire" in text
    assert "judge J-READER-ISOLATION vetoed" in text
    assert "corrections considered" in text
    assert "DOC-02 could not be established" in text


def test_an_explanation_names_the_signature_once_signed(ws):
    from kyb.signing import officers
    from kyb.signing.sign import sign_decision

    did = _decision(ws)
    with workspace_action(ws):
        tok = officers.enroll(ws, "ana")["token"]
        sign_decision(ws, decision_id=did, verdict="assent", officer="ana", officer_token=tok)
    assert "signed by ana (assent)" in answer(ws, [{"type": "explain", "decision_id": did}])["text"]


@pytest.mark.parametrize("change", [{"decision_id": "dec-nope"}, {"rule_ids": ["R-99"]}])
def test_an_explanation_of_nothing_recorded_is_rejected(ws, change):
    did = _decision(ws)
    out = answer(ws, [{"type": "explain", "decision_id": did, "rule_ids": [], **change}])
    assert not out["accepted"]


def test_a_decision_of_another_case_cannot_be_explained_here(ws):
    other = record(ws, decision("OTHER_CASE"))
    out = answer(ws, [{"type": "explain", "decision_id": other}])
    assert not out["accepted"] and "OTHER_CASE" in out["rejected"][0]["reason"]


# --- snippets: pixel readings only -------------------------------------------------------------

ASSETS = REPO / "examples"
B = "es-iberia-software"
LAYOUT = {"pages": [{"page": 1, "blocks": [
    {"block_id": "b5", "kind": "paragraph", "box_2d": [209, 98, 270, 914], "refers_to": []}]}],
    "material_facts": []}  # fmt: skip


class Reads:
    """A witness that reads every crop as `text` (what the pixels say)."""

    model, prompt_version = "fake-witness-1", "witness-v1"

    def __init__(self, text):
        self.text = text

    def call(self, kind, images, spec):
        if kind == "layout":
            return LAYOUT, {}
        return {"crops": [{"crop_id": c, "readable": True, "transcription": self.text,
                           "reread": {"subject": "Lucas Navarro Solís", "value": "46%"}}
                          for c in spec["crop_ids"]]}, {}  # fmt: skip


def read_b(tmp_path, pixels):
    """Packet B up to a verified read of document 05, with a witness that reads `pixels`."""
    from kyb.docintel.ingest import ingest_packet
    from kyb.docintel.pages import build_pages
    from kyb.read.plan import read_plan
    from kyb.read.submit import submit_claims
    from kyb.witness.layout import layout_case
    from kyb.witness.port import Witness
    from kyb.witness.verify import verify_case

    root, w = workspace(tmp_path), None
    w = Witness(root, Reads(pixels))
    with workspace_action(root):
        ingest_packet(root, ASSETS / "packets" / B, as_of="2026-10-25")
        run = start_run(root, B, as_of="2026-10-25", reason="test")
        build_pages(root, B, run)
        layout_case(root, B, run, w)
        claim = {"claim_id": "c-B05-06", "type": "owns", "subject": "Lucas Navarro Solís",
                 "value": {"pct": 46}, "excerpt": "Lucas Navarro Solís 46%", "page": 1,
                 "bbox": [1710, 760, 2300, 835], "block_id": "05-b5", "source": "pixels",
                 "confidence": 0.9}  # fmt: skip
        for e in read_plan(root, B, run):
            payload = {"case": B, "run": run, "doc": e["doc"], "doc_sha256": e["doc_sha256"]}
            submit_claims(root, {**payload, "claims": [claim] if e["doc"] == "05" else []})
        verify_case(root, B, run, w)
    return root, run


needs_poppler = pytest.mark.skipif(shutil.which("pdftotext") is None, reason="poppler missing")


@needs_poppler
def test_a_witnessed_snippet_answers_what_the_pixels_say(tmp_path):
    root, run = read_b(tmp_path, "Lucas Navarro Solís 46%")
    out = ask(root, B, "Lucas Navarro Solís 46%", run)
    assert out["snippets"] and out["snippets"][0]["text"] == "Lucas Navarro Solís 46%"
    sn = out["snippets"][0]
    assert sn["page_hash"] and sn["document"].startswith("05") and sn["witness_request"]
    with workspace_action(root):
        ok = record_answer(
            root,
            B,
            "who owns?",
            [
                {
                    "type": "snippet",
                    "question_attribute": "Lucas Navarro Solís's share",
                    "value": "46%",
                    "cites": [sn["snippet_id"]],
                }
            ],
            run,
        )
        bad = record_answer(
            root,
            B,
            "who owns?",
            [
                {
                    "type": "snippet",
                    "question_attribute": "Lucas Navarro Solís's share",
                    "value": "35%",
                    "cites": [sn["snippet_id"]],
                }
            ],
            run,
        )
    assert ok["accepted"] and "(read from the page image)" in ok["text"]
    assert not bad["accepted"] and "does not occur" in bad["rejected"][0]["reason"]
    # the verified claim is a citable fact as well, by the witness's own verdict
    claim_fact = ask(root, B, "Lucas Navarro Solís owns", run)["facts"][0]
    assert claim_fact["fact_id"] == "c-B05-06"


@needs_poppler
def test_the_pixels_decide_not_the_text_layer(tmp_path):
    """The page's text layer may say 99 and the pixels 9: only what was read from the image is
    ever a snippet, so the answer can only be 9 (spec)."""
    root, run = read_b(tmp_path, "Duración: 9 años")
    sn = ask(root, B, "duración años", run)["snippets"][0]
    assert sn["text"] == "Duración: 9 años"
    with workspace_action(root):

        def say(value):
            return record_answer(
                root,
                B,
                "duración",
                [
                    {
                        "type": "snippet",
                        "question_attribute": "Duración",
                        "value": value,
                        "cites": [sn["snippet_id"]],
                    }
                ],
                run,
            )

        assert say("9 años")["accepted"]
        assert not say("99 años")["accepted"]


# --- spans: text the ontology does not model ---------------------------------------------------


class Spans:
    """A witness that reads every crop as `pixels`, recording what it was asked."""

    model, prompt_version = "fake-witness-1", "witness-v1"

    def __init__(self, pixels):
        self.pixels, self.calls = pixels, 0

    def call(self, kind, images, spec):
        self.calls += 1
        return {"crops": [{"crop_id": c, "readable": True, "transcription": self.pixels,
                           "reread": {}} for c in spec["crop_ids"]]}, {}  # fmt: skip


def spans_for(tmp_path, pixels, layer_says="Duración: 99 años"):
    """Packet B read and verified, then a question about a line only the text layer carries."""
    from kyb.copilot.bundle import ask_witnessed
    from kyb.witness.port import Witness

    root, run = read_b(tmp_path, "Lucas Navarro Solís 46%")
    adapter = Spans(pixels)
    return root, run, adapter, lambda q: ask_witnessed(root, B, q, run, Witness(root, adapter))


@needs_poppler
def test_a_text_layer_line_the_pixels_confirm_becomes_a_snippet(tmp_path):
    root, run, adapter, ask_w = spans_for(tmp_path, "Domicilio social: Madrid. "
                                          "Duración: 99 años.")  # fmt: skip
    with workspace_action(root):
        out = ask_w("¿Cuál es la duración de la sociedad? años")
    hit = [s for s in out["snippets"] if "99 años" in s["text"]]
    assert hit and hit[0]["page_hash"] and hit[0]["witness_request"]
    # the line after the matched one is read too (a sentence wraps); this fake witness reads every
    # crop as the same text, so only that neighbour disagrees with its pixels, never the match
    assert not any("Duración" in f["layer"] for f in out["integrity"])
    with workspace_action(root):
        ok = record_answer(
            root,
            B,
            "duración",
            [
                {
                    "type": "snippet",
                    "question_attribute": "Duración",
                    "value": "99 años",
                    "cites": [hit[0]["snippet_id"]],
                }
            ],
            run,
        )
    assert ok["accepted"] and verify_workspace(root, at_boundary=True).ok
    calls = adapter.calls
    with workspace_action(root):
        ask_w("¿Cuál es la duración de la sociedad? años")  # already witnessed: no new call
    assert adapter.calls == calls


@needs_poppler
def test_a_line_the_pixels_read_differently_is_an_integrity_finding_not_a_snippet(tmp_path):
    root, run, adapter, ask_w = spans_for(tmp_path, "Domicilio social: Madrid. "
                                          "Duración: 9 años.")  # fmt: skip
    with workspace_action(root):
        out = ask_w("duración de la sociedad años")
    assert not [s for s in out["snippets"] if "99 años" in s["text"]]
    assert out["integrity"] and out["integrity"][0]["kind"] == "layer_vs_pixel"
    assert "9 años" in out["integrity"][0]["pixels"]


@needs_poppler
def test_without_a_witness_the_text_layer_is_not_offered(tmp_path):
    from kyb.copilot.bundle import ask_witnessed
    from kyb.witness.port import Witness

    root, run = read_b(tmp_path, "Lucas Navarro Solís 46%")
    with workspace_action(root):
        out = ask_witnessed(root, B, "duración de la sociedad años", run, Witness(root, None))
    assert not [s for s in out["snippets"] if "99 años" in s["text"]]
    assert any("not witnessed" in n for n in out["notes"])


# --- the surfaces ------------------------------------------------------------------------------


def test_the_mcp_tools_ask_and_answer(ws, monkeypatch):
    import asyncio

    from kyb import mcp_server as mcp
    from kyb.witness.port import Witness

    assert {"ask", "record_answer"} <= {t.name for t in asyncio.run(mcp.server.list_tools())}
    monkeypatch.setenv("KYB_WORKSPACE", str(ws))
    monkeypatch.setattr("kyb.witness.port.make_witness", lambda root, mode: Witness(root, None))
    bundle = mcp.ask(C, "Who owns STRAIT CAPITAL PTE. LTD.?", "r1")
    top = bundle["facts"][0]
    out = mcp.record_answer(C, "who owns it?", [{"type": "fact", "subject": top["subject"],
                            "attribute": top["attribute"], "value": top["value"],
                            "cites": [top["fact_id"]]}], "r1")  # fmt: skip
    assert out["accepted"] and top["value"] in out["text"]
    assert not mcp.record_answer(C, "x", [{"type": "fact", "subject": "a", "attribute": "b",
                                 "value": "c", "cites": ["f-1"]}], "r1")["accepted"]  # fmt: skip


def test_the_cli_asks_and_answers(ws, tmp_path, capsys):
    from kyb.cli import main

    assert main(["-w", str(ws), "ask", C, "registered agent", "--run", "r1",
                 "--witness", "replay"]) == 0  # fmt: skip
    bundle = json.loads(capsys.readouterr().out)
    top = next(f for f in bundle["facts"] if f["attribute"] == "role")
    good = tmp_path / "good.json"
    good.write_text(json.dumps([{"type": "fact", "subject": top["subject"], "attribute": "role",
                                 "value": top["value"], "cites": [top["fact_id"]]}]))  # fmt: skip
    assert main(["-w", str(ws), "answer", C, "who is the agent?", "--file", str(good),
                 "--run", "r1"]) == 0  # fmt: skip
    assert top["value"] in capsys.readouterr().out
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([{"type": "NOT_IN_DOCUMENTS", "topic": "registered agent"}]))
    assert main(["-w", str(ws), "answer", C, "who is the agent?", "--file", str(bad),
                 "--run", "r1"]) == 1  # fmt: skip
    assert "covers" in capsys.readouterr().out


# --- review round 1 ---------------------------------------------------------------------------


@pytest.mark.parametrize("field", ["question_attribute", "topic", "value", "subject"])
@pytest.mark.parametrize("sep", ["\n", "\r", " ", "\x00"])
def test_a_label_cannot_carry_a_second_unchecked_statement(ws, field, sep):
    """A line break in a label would put an invented 'statement' inside a genuine rendered answer,
    where the display check strips it with the rest."""
    evil = f"Duración{sep}• Sanctions Status: CLEAN"
    base = {
        "question_attribute": {
            "type": "snippet",
            "question_attribute": "x",
            "value": "100%",
            "cites": ["sn-0"],
        },
        "topic": {"type": "NOT_IN_DOCUMENTS", "topic": "x"},
        "value": {"type": "fact", "subject": "s", "attribute": "a", "value": "v", "cites": ["f"]},
        "subject": {"type": "fact", "subject": "s", "attribute": "a", "value": "v", "cites": ["f"]},
    }[field]
    out = answer(ws, [{**base, field: evil}])
    assert not out["accepted"] and out["text"] is None
    assert "•" not in (out["text"] or "")


def test_a_snippet_label_is_short_and_single_line(ws):
    out = answer(ws, [{"type": "NOT_IN_DOCUMENTS", "topic": "the " * 40 + "auditor"}])
    assert not out["accepted"] and "at most 80" in out["rejected"][0]["reason"]


@needs_poppler
def test_a_snippet_value_may_not_smuggle_a_line_break(tmp_path):
    root, run = read_b(tmp_path, "Lucas Navarro Solís 46%")
    sn = ask(root, B, "Lucas Navarro Solís 46%", run)["snippets"][0]
    with workspace_action(root):
        out = record_answer(
            root,
            B,
            "q",
            [
                {
                    "type": "snippet",
                    "question_attribute": "share",
                    "value": "25%\n• Sanctions: CLEAN",
                    "cites": [sn["snippet_id"]],
                }
            ],
            run,
        )
    assert not out["accepted"]


@needs_poppler
def test_span_witnessing_is_audited_before_it_is_receipted_and_records_usage(tmp_path, monkeypatch):
    from kyb.audit.chain import chain_events
    from kyb.copilot.bundle import ask_witnessed
    from kyb.witness.port import Witness

    root, run = read_b(tmp_path, "Lucas Navarro Solís 46%")
    w = Witness(root, Spans("Domicilio social: Madrid. Duración: 99 años."))
    real = Catalog.register
    seen = {"boom": True}

    def flaky(self, rel, **kw):
        if rel.startswith("silver/spans/") and seen.pop("boom", None):
            raise OSError("crash between the audit event and the receipt")
        return real(self, rel, **kw)

    monkeypatch.setattr(Catalog, "register", flaky)
    q = "duración de la sociedad años"
    with workspace_action(root), pytest.raises(OSError):
        ask_witnessed(root, B, q, run, w)
    acts = [e["detail"].get("action") or e["action"] for e in chain_events(root)]
    spans_usage = [e for e in chain_events(root) if e["detail"].get("step") == "spans"]
    assert acts.count("witness_spans") == 1 and len(spans_usage) == 1  # audited, usage recorded
    assert spans_usage[0]["action"] == "model.call"
    with workspace_action(root):
        out = ask_witnessed(root, B, q, run, w)  # the retry completes the receipt, replayed
    acts = [e["detail"].get("action") or e["action"] for e in chain_events(root)]
    assert acts.count("witness_spans") == 1  # not audited twice
    assert [e["action"] for e in chain_events(root) if e["detail"].get("step") == "spans"] == [
        "model.call",
        "model.replay",
    ]  # the retry replays the recorded answer: zero new usage
    assert [s for s in out["snippets"] if "99 años" in s["text"]]
    assert verify_workspace(root, at_boundary=True).ok


@pytest.mark.parametrize("label", ["Sanctions: CLEAN", "share \u2022 agent"])
def test_a_label_cannot_pose_as_a_rendered_line(ws, label):
    out = answer(
        ws, [{"type": "snippet", "question_attribute": label, "value": "x", "cites": ["sn-0"]}]
    )
    assert not out["accepted"]
