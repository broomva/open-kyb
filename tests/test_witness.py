"""The witness (spec, spec): claim-blind layout, blind transcription, norm-v2, bind-v6, coverage.

A fake adapter stands in for Gemini, so these run offline and deterministically. It answers
transcription requests by crop position, because the crop ids it sees are opaque (k1, k2, ...).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from kyb.audit.chain import chain_events, verify_workspace, workspace_action
from kyb.docintel.ingest import ingest_packet
from kyb.docintel.pages import build_pages
from kyb.read.plan import load_layout, read_plan
from kyb.read.status import review_status
from kyb.read.submit import submit_claims
from kyb.witness.layout import LayoutRejected, layout_case, to_px
from kyb.witness.norm import contains, norm
from kyb.witness.port import Witness
from kyb.witness.verify import subject_match, value_match, verify_case
from kyb.workspace import guard
from kyb.workspace.init import init_workspace
from kyb.workspace.runs import start_run

REPO = Path(__file__).resolve().parents[1]
ASSETS = REPO / "examples"
CASE = "es-iberia-software"
pytestmark = pytest.mark.skipif(shutil.which("pdftotext") is None, reason="poppler missing")

ROW = [209, 98, 270, 914]  # 05's row block, 0–1000 units: x 250–2330, y 690–890 px
NOTE = [362, 106, 394, 914]  # a footnote block below it: y ~1195–1300 px

LAYOUT = {
    "pages": [
        {
            "page": 1,
            "blocks": [
                {
                    "block_id": "b3",
                    "kind": "paragraph",
                    "box_2d": ROW,
                    "markers": ["strike"],
                    "refers_to": [],
                },
                {"block_id": "b6", "kind": "footnote", "box_2d": NOTE, "qualifiers": ["pendiente"]},
            ],
        }
    ],
    "material_facts": [
        {
            "block_id": "b3",
            "kind": "ownership",
            "subject": "V. Ortega Méndez",
            "attribute": "pct",
            "value": "25%",
        },
        {
            "block_id": "b3",
            "kind": "ownership",
            "subject": "Inversiones Altiplano, S.A.",
            "attribute": "shareholding_percentage",
            "value": "25%",
        },
    ],
}


def crop(transcription, subject="", value="", struck=(), readable=True):
    return {
        "transcription": transcription,
        "readable": readable,
        "reread": {"subject": subject, "value": value, "struck": list(struck)},
    }


class FakeWitness:
    prompt_version = "witness-v1"

    def __init__(self, crops=(), layout=None, model="fake-witness-1"):
        self.crops, self.model = list(crops), model
        self.layout = LAYOUT if layout is None else layout
        self.calls: list[tuple[str, dict]] = []

    def call(self, kind, images, spec):
        self.calls.append((kind, spec))
        if kind == "layout":
            return self.layout, {"promptTokenCount": 100, "candidatesTokenCount": 20}
        out = [
            {"crop_id": cid, **(self.crops[i] if i < len(self.crops) else crop(""))}
            for i, cid in enumerate(spec["crop_ids"])
        ]
        return {"crops": out}, {"promptTokenCount": 50, "candidatesTokenCount": 10}


@pytest.fixture(scope="module")
def _built(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("built") / "ws"
    init_workspace(root, tenant="t", policy_src=ASSETS / "policy/compliance_rules.md")
    with workspace_action(root):
        ingest_packet(root, ASSETS / "packets" / CASE, as_of="2026-10-25")
        build_pages(root, CASE, start_run(root, CASE, as_of="2026-10-25", reason="test"))
        layout_case(root, CASE, "r1", Witness(root, FakeWitness()))
    return root


@pytest.fixture
def ws(_built, tmp_path) -> Path:
    root = tmp_path / "ws"
    shutil.copytree(_built, root, symlinks=True)
    return root


def _claim(cid, subject, value, excerpt, bbox, block="05-b3", **kw):
    return {
        "claim_id": cid,
        "type": "owns",
        "subject": subject,
        "value": value,
        "excerpt": excerpt,
        "page": 1,
        "bbox": bbox,
        "block_id": block,
        "source": "pixels",
        "confidence": 0.9,
        **kw,
    }


ORTEGA = (
    "c-B05-06",
    "V. Ortega Méndez",
    {"pct": 25},
    "V. Ortega Méndez 25%",
    [1710, 760, 2300, 835],
)


def _submit_all(ws, claims_05, doc="05"):
    for e in read_plan(ws, CASE, "r1"):
        with workspace_action(ws):
            submit_claims(
                ws,
                {
                    "case": CASE,
                    "run": "r1",
                    "doc": e["doc"],
                    "doc_sha256": e["doc_sha256"],
                    "claims": claims_05 if e["doc"] == doc else [],
                },
            )


def _verify(ws, fake, doc="05") -> dict:
    with workspace_action(ws):
        verify_case(ws, CASE, "r1", Witness(ws, fake))
    return json.loads((ws / f"silver/witness/{CASE}/r1/{doc}.json").read_text())


def _result(rec, cid):
    return next(r for r in rec["results"] if r["claim_id"] == cid)


def test_norm_v1_is_exact_on_token_boundaries():
    assert contains("L. Navarro Solís  50 %", "l. navarro solís 50 %")
    assert contains("recibe 10% — pendiente", "10% - pendiente")
    assert not contains("Duración: 9 años", "99 años")
    assert not contains("V. Ortega Méndez 146%", "46%")  # no matching inside a number
    assert norm("ＡＢＣ") == "abc"


def test_bind_v2_comparators():
    assert subject_match("L. Navarro Solís", "Lucas Navarro Solís")
    assert not subject_match("S. Navarro Solís", "Lucas Navarro Solís")  # no token reuse
    assert not subject_match("L. Navarro Solís", "V. Ortega Méndez")
    assert value_match({"pct": 25}, "25%") and not value_match({"pct": 46}, "146%")
    assert value_match("19 de agosto de 2026", "2026-08-19", "issue_date")
    assert not value_match("19 de agosto de 2026", "2026-08-16", "issue_date")


def test_box_2d_converts_to_pixels_on_the_page():
    assert to_px(ROW, (2550, 3300)) == (250, 690, 2331, 891)
    assert to_px([1000, 1000, 1000, 1000], (2550, 3300)) == (2549, 3299, 2550, 3300)


def test_layout_is_claim_blind_recorded_and_replayable(ws):
    lay = load_layout(ws, CASE, "r1", "05")
    assert [b.block_id for b in lay.pages[0].blocks] == ["05-b3", "05-b6"]
    assert len(lay.material_facts) == 2
    fake = FakeWitness()
    Witness(ws, fake).ask("layout", [b"x"], {"kinds": ["k"]})
    w = Witness(ws, None, mode="replay", model="fake-witness-1", prompt_version="witness-v1")
    assert w.ask("layout", [b"x"], {"kinds": ["k"]}).replayed
    with pytest.raises(LookupError):
        w.ask("layout", [b"never-seen"], {"kinds": ["k"]})


@pytest.mark.parametrize(
    "bad",
    [
        {},
        {"pages": [], "material_facts": []},
        {
            "pages": [{"page": 1, "blocks": [{"block_id": "b1", "box_2d": [1, 2]}]}],
            "material_facts": [],
        },
        {
            "pages": [{"page": 1, "blocks": []}],
            "material_facts": [
                {"block_id": "nope", "kind": "k", "subject": "s", "attribute": "a", "value": "v"}
            ],
        },
    ],
)
def test_an_unusable_layout_answer_is_refused_and_not_recorded(tmp_path, bad):
    root = tmp_path / "ws"
    init_workspace(root, tenant="t", policy_src=ASSETS / "policy/compliance_rules.md")
    with workspace_action(root):
        ingest_packet(root, ASSETS / "packets" / CASE, as_of="2026-10-25")
        build_pages(root, CASE, start_run(root, CASE, as_of="2026-10-25", reason="test"))
    with pytest.raises(LayoutRejected), workspace_action(root):
        layout_case(root, CASE, "r1", Witness(root, FakeWitness(layout=bad)))
    assert load_layout(root, CASE, "r1", "01") is None


def test_layout_before_pages_is_refused(ws):
    with workspace_action(ws):
        start_run(ws, CASE, as_of="2026-10-25", reason="test")
    with pytest.raises(LookupError, match="not rendered"), workspace_action(ws):
        layout_case(ws, CASE, "r2", Witness(ws, FakeWitness()))


def test_verify_witnesses_binds_and_checks_coverage(ws):
    wrong = _claim(
        "c-B05-03", "L. Navarro Solís", {"pct": 68}, "L. Navarro Solís 68%", [870, 730, 1700, 830]
    )
    _submit_all(ws, [_claim(*ORTEGA), wrong])
    fake = FakeWitness(
        [
            crop("V. Ortega Méndez 25%;", "V. Ortega Méndez", "25%"),
            crop("L. Navarro Solís 46%", "L. Navarro Solís", "46%"),
        ]
    )
    rec = _verify(ws, fake)
    assert _result(rec, "c-B05-06")["verified"] is True
    assert _result(rec, "c-B05-03")["transcription"] == "failed"
    status = {f["subject"]: f["status"] for f in rec["coverage"]}
    assert status == {"V. Ortega Méndez": "matched", "Inversiones Altiplano, S.A.": "uncovered"}
    assert rec["assurance"] == "W1" and rec["saw_claim"] is False
    sent = [s for k, s in fake.calls if k == "transcribe"][0]["crop_ids"]
    assert sent == ["k1", "k2"]  # the witness sees opaque ids, never the reader's
    assert verify_workspace(ws, at_boundary=True).ok


def test_verified_structured_holding_covers_its_percentage_component(ws):
    claim = _claim(*ORTEGA)
    claim.update(value={"pct": 25, "shares": 1250}, excerpt="V. Ortega Méndez 1.250 acciones 25%")
    _submit_all(ws, [claim])
    rec = _verify(
        ws, FakeWitness([crop(claim["excerpt"], "V. Ortega Méndez", "1.250 acciones 25%")])
    )
    assert _result(rec, claim["claim_id"])["verified"] is True
    assert rec["coverage"][0]["status"] == "matched"
    assert rec["coverage"][1]["status"] == "uncovered"  # another subject
    assert rec["coverage_version"] == "coverage-v3"


def test_unsafe_coverage_record_requires_a_new_run_without_rewriting_it(ws, monkeypatch):
    from kyb.witness import coverage as coverage_module
    from kyb.witness import verify as verify_module

    _submit_all(ws, [_claim(*ORTEGA)])
    fake = FakeWitness([crop("V. Ortega Méndez 25%", "V. Ortega Méndez", "25%")])
    with monkeypatch.context() as old:
        old.setattr(coverage_module, "VERSION", "coverage-v2")
        old.setattr(verify_module, "COVERAGE", "coverage-v2")
        _verify(ws, fake)
    path = ws / f"silver/witness/{CASE}/r1/05.json"
    before = path.read_bytes()
    with pytest.raises(ValueError, match="coverage matcher changed"):
        verify_module.assurance_now(ws, CASE, "r1", "05")
    with pytest.raises(ValueError, match="coverage matcher changed"), workspace_action(ws):
        verify_case(ws, CASE, "r1", Witness(ws, fake))
    status = review_status(ws, CASE)
    assert status["next"] == "stale" and "coverage matcher changed" in status["detail"]
    assert path.read_bytes() == before


def test_a_claim_id_cannot_smuggle_the_answer_to_the_witness(ws):
    leaky = _claim("c-B05-OrtegaMendez-25pct", *ORTEGA[1:])
    _submit_all(ws, [leaky])
    fake = FakeWitness([crop("V. Ortega Méndez 25%", "V. Ortega Méndez", "25%")])
    _verify(ws, fake)
    assert "OrtegaMéndez" not in json.dumps([s for _, s in fake.calls])


@pytest.mark.parametrize(
    "claim, witness_crop, why",
    [
        (
            (
                "c-B05-01",
                "V. Ortega Méndez",
                {"pct": 46},
                "V. Ortega Méndez 46%",
                [1710, 760, 2300, 835],
            ),
            crop("V. Ortega Méndez 146%; folio 50", "V. Ortega Méndez", "146%"),
            "inside 146%",
        ),
        (
            ORTEGA,
            crop("V. Ortega Méndez 25% tachado; ahora 68%", "V. Ortega Méndez", "68%"),
            "the re-read says 68%",
        ),
        (
            (
                "c-B05-02",
                "S. Navarro Solís",
                {"pct": 25},
                "S. Navarro Solís 25%",
                [870, 730, 1700, 830],
            ),
            crop("Lucas Navarro Solís 25%", "Lucas Navarro Solís", "25%"),
            "initial reuse",
        ),
        (
            ORTEGA,
            crop("V. Ortega Méndez 25%", "V. Ortega Méndez", "68%", struck=["25%"]),
            "a struck value presented as current",
        ),
    ],
)
def test_false_accepts_are_closed(ws, claim, witness_crop, why):
    _submit_all(ws, [_claim(*claim)])
    rec = _verify(ws, FakeWitness([witness_crop]))
    assert rec["results"][0]["verified"] is False, why


def test_a_struck_value_verifies_only_when_flagged(ws):
    struck = _claim(
        "c-B05-03",
        "L. Navarro Solís",
        {"pct": 46},
        "L. Navarro Solís 46%",
        [870, 730, 1700, 830],
        flags=["struck-through"],
    )
    _submit_all(ws, [struck])
    rec = _verify(
        ws,
        FakeWitness(
            [crop("L. Navarro Solís 46% 68% ver nota", "L. Navarro Solís", "68%", struck=["46%"])]
        ),
    )
    assert rec["results"][0]["verified"] is True


def test_required_context_must_be_read_not_just_listed(ws):
    linked = _claim(*ORTEGA, linked_blocks=["05-b6"])
    _submit_all(ws, [linked])
    rec = _verify(ws, FakeWitness([crop("V. Ortega Méndez 25%", "V. Ortega Méndez", "25%")]))
    r = rec["results"][0]  # the witness answered the main crop and not the context crop
    assert r["binding"] == "incomplete" and r["unread_context"] == ["05-b6"]
    assert r["context_qualifiers"] == ["pendiente"]


def test_coverage_needs_the_same_attribute(tmp_path):
    block = {"block_id": "b3", "kind": "p", "box_2d": ROW}
    fact = {
        "block_id": "b3",
        "kind": "date",
        "subject": "V. Ortega Méndez",
        "attribute": "date_of_birth",
        "value": "2026-08-19",
    }
    date_fact = {"pages": [{"page": 1, "blocks": [block]}], "material_facts": [fact]}
    root = tmp_path / "w2"
    init_workspace(root, tenant="t", policy_src=ASSETS / "policy/compliance_rules.md")
    with workspace_action(root):
        ingest_packet(root, ASSETS / "packets" / CASE, as_of="2026-10-25")
        build_pages(root, CASE, start_run(root, CASE, as_of="2026-10-25", reason="test"))
        layout_case(root, CASE, "r1", Witness(root, FakeWitness(layout=date_fact)))
    issued = _claim(
        "c-B05-09",
        "V. Ortega Méndez",
        "2026-08-19",
        "V. Ortega Méndez 2026-08-19",
        [1710, 760, 2300, 835],
        type="issued_on",
        attribute="issue_date",
    )
    _submit_all(root, [issued])
    rec = _verify(
        root, FakeWitness([crop("V. Ortega Méndez 2026-08-19", "V. Ortega Méndez", "2026-08-19")])
    )
    assert rec["coverage"][0]["status"] == "uncovered"  # an issue date is not a birth date


def test_a_value_the_witness_reads_differently_is_a_recorded_conflict(ws):
    altiplano = _claim(
        "c-B05-07",
        "Inversiones Altiplano, S.A.",
        {"pct": 30},
        "Inversiones Altiplano, S.A. 30%",
        [250, 815, 920, 885],
    )
    _submit_all(ws, [altiplano])
    rec = _verify(
        ws, FakeWitness([crop("Inversiones Altiplano, S.A. 30%.", "Inversiones Altiplano", "30%")])
    )
    cov = {f["subject"]: f for f in rec["coverage"]}
    assert cov["Inversiones Altiplano, S.A."]["status"] == "conflict"  # the layout says 25%


def test_verify_refuses_until_every_document_has_a_submission(ws):
    with pytest.raises(LookupError, match="no submission"), workspace_action(ws):
        verify_case(ws, CASE, "r1", Witness(ws, FakeWitness()))


def test_live_rereads_become_revisions_and_replays_cost_nothing(ws):
    fake = FakeWitness()
    first = Witness(ws, fake, mode="live").ask("layout", [b"z"], {"kinds": ["k"]})
    fake.layout = {"pages": [], "material_facts": [{"changed": True}]}
    second = Witness(ws, fake, mode="live").ask("layout", [b"z"], {"kinds": ["k"]})
    assert first.request_hash == second.request_hash and second.data != first.data
    pinned = Witness(ws, None, mode="replay", model="fake-witness-1", prompt_version="witness-v1")
    assert pinned.ask("layout", [b"z"], {"kinds": ["k"]}).data == first.data
    _submit_all(ws, [_claim(*ORTEGA)])
    witness_crop = crop("V. Ortega Méndez 25%", "V. Ortega Méndez", "25%")
    _verify(ws, FakeWitness([witness_crop]))
    with workspace_action(ws):  # a second verify reuses the recorded witness result
        verify_case(ws, CASE, "r1", Witness(ws, FakeWitness([witness_crop])))
    calls = [
        e
        for e in chain_events(ws)
        if e["action"] == "model.call" and e["detail"]["step"] == "witness"
    ]
    assert len(calls) == 1


def _attested_reader(ws, plan, payload, reader_models=("claude-opus-5-5",)):
    ev = {"cwd": str(ws), "session_id": "s1"}
    guard.session_start({**ev, "model": "claude-opus-5-5"})
    launch = {"subagent_type": "kyb:kyb-reader", "prompt": f"kyb-reader {CASE} r1 05"}
    assert (
        guard.pre_tool_use({**ev, "tool_name": "Agent", "tool_use_id": "L1", "tool_input": launch})[
            0
        ]
        == 0
    )
    guard.subagent_start({**ev, "agent_id": "a5", "agent_type": "kyb:kyb-reader"})
    bind = {"case": CASE, "run": "r1", "doc": "05"}
    assert (
        guard.pre_tool_use(
            {
                **ev,
                "tool_name": "mcp__kyb__read_plan",
                "agent_id": "a5",
                "agent_type": "kyb:kyb-reader",
                "tool_use_id": "r1",
                "tool_input": bind,
            }
        )[0]
        == 0
    )
    out = guard.pre_tool_use(
        {
            **ev,
            "tool_name": "mcp__kyb__read_document",
            "agent_id": "a5",
            "agent_type": "kyb:kyb-reader",
            "tool_use_id": "u1",
            "tool_input": payload,
        }
    )
    executed = out[2]["hookSpecificOutput"]["updatedInput"]
    with workspace_action(ws):
        assert submit_claims(ws, executed)["context"] == "isolated"
    if reader_models is not None:  # the harness-written subagent transcript, read at SubagentStop
        snapshots = reader_models if isinstance(reader_models, list) else [reader_models]
        for n, models in enumerate(snapshots):  # a stop can be vetoed and the agent continue
            transcript = ws.parent / f"agent-a5-{n}.jsonl"
            transcript.write_text(
                "".join(
                    json.dumps({"type": "assistant", "message": {"model": m}}) + "\n"
                    for m in models
                )
            )
            guard.subagent_stop(
                {
                    **ev,
                    "agent_id": "a5",
                    "agent_type": "kyb:kyb-reader",
                    "agent_transcript_path": str(transcript),
                }
            )
    for d, e in plan.items():
        if d != "05":
            with workspace_action(ws):
                submit_claims(
                    ws,
                    {
                        "case": CASE,
                        "run": "r1",
                        "doc": d,
                        "doc_sha256": e["doc_sha256"],
                        "claims": [],
                    },
                )


@pytest.mark.parametrize(
    "witness_model, expected",
    [("gemini-3.1-pro-preview", "W2"), ("claude-opus-5-5", "W1"), ("fake-witness-1", "W1")],
)
def test_w2_needs_known_different_families(ws, witness_model, expected):
    plan = {e["doc"]: e for e in read_plan(ws, CASE, "r1")}
    payload = {
        "case": CASE,
        "run": "r1",
        "doc": "05",
        "doc_sha256": plan["05"]["doc_sha256"],
        "claims": [_claim(*ORTEGA)],
    }
    _attested_reader(ws, plan, payload)
    witness = FakeWitness(
        [crop("V. Ortega Méndez 25%", "V. Ortega Méndez", "25%")], model=witness_model
    )
    assert _verify(ws, witness)["assurance"] == expected


@pytest.mark.parametrize(
    "reader_models",
    [
        None,  # no SubagentStop record: the session's model is the parent's, not evidence
        ("gemini-3.1-pro-preview",),  # the reader ran on the witness's family
        ("claude-opus-5-5", "gemini-3.1-pro-preview"),  # more than one model
        [("claude-opus-5-5",), ("claude-opus-5-5", "gpt-5")],  # a vetoed stop, then a changed one
        [("claude-opus-5-5",), ()],  # a later stop whose transcript was empty or unreadable
    ],
)
def test_w2_rests_on_the_readers_own_model(ws, reader_models):
    plan = {e["doc"]: e for e in read_plan(ws, CASE, "r1")}
    payload = {
        "case": CASE,
        "run": "r1",
        "doc": "05",
        "doc_sha256": plan["05"]["doc_sha256"],
        "claims": [_claim(*ORTEGA)],
    }
    _attested_reader(ws, plan, payload, reader_models)
    witness = FakeWitness(
        [crop("V. Ortega Méndez 25%", "V. Ortega Méndez", "25%")], model="gemini-3.1-pro-preview"
    )
    assert _verify(ws, witness)["assurance"] == "W1"


# --- review round 2 -----------------------------------------------------------------------------


def test_round2_typed_comparisons():
    assert not contains("V. Ortega Méndez 146.46%", "46%")  # not cut from a decimal
    assert not subject_match("N. Solís", "Lucas Navarro Solís")  # an initial stays in place
    assert subject_match("Lucas Solís", "Lucas Navarro Solís")  # a full token may skip
    assert subject_match("Valeria Ortega Méndez", "Ortega Méndez, Valeria")  # surname-first form
    assert not value_match("VG/3184920", "AE/3184920", "registry_id")
    assert not value_match("Calle Sur 10", "Calle Norte 10", "address")
    assert not value_match("August 15, 2026", "September 15, 2026", "issue_date")
    assert not value_match("2026-09-08", "MM/DD/YYYY: 08/09/2026", "issue_date")
    assert value_match("2026-08-09", "MM/DD/YYYY: 08/09/2026", "issue_date")
    assert not value_match("15/9", "15/9/2026", "issue_date")  # an incomplete date never matches


def test_distinct_attributes_stay_distinct():
    from kyb.witness.verify import canon_attr

    assert canon_attr("meeting_date") != canon_attr("issue_date")
    assert canon_attr("registered_address") != canon_attr("operating_address")
    assert canon_attr("shareholding_percentage") == canon_attr("pct")


def test_an_all_empty_layout_is_refused_unless_pages_are_marked_blank(tmp_path):
    root = tmp_path / "ws"
    init_workspace(root, tenant="t", policy_src=ASSETS / "policy/compliance_rules.md")
    with workspace_action(root):
        ingest_packet(root, ASSETS / "packets" / CASE, as_of="2026-10-25")
        build_pages(root, CASE, start_run(root, CASE, as_of="2026-10-25", reason="test"))
    empty = {"pages": [{"page": 1, "blocks": []}], "material_facts": []}
    with pytest.raises(LayoutRejected, match="blank"), workspace_action(root):
        layout_case(root, CASE, "r1", Witness(root, FakeWitness(layout=empty)))


def test_a_changed_claims_file_is_not_reused_and_live_mode_refuses(ws):
    _submit_all(ws, [_claim(*ORTEGA)])
    witness_crop = crop("V. Ortega Méndez 25%", "V. Ortega Méndez", "25%")
    _verify(ws, FakeWitness([witness_crop]))
    with pytest.raises(ValueError, match="new run"), workspace_action(ws):
        verify_case(ws, CASE, "r1", Witness(ws, FakeWitness([witness_crop]), mode="live"))
    path = ws / f"silver/claims/{CASE}/r1/05.json"
    path.chmod(0o644)
    path.write_text(path.read_text().replace('"pct": 25', '"pct": 99'))
    with pytest.raises(PermissionError, match="receipt"), workspace_action(ws):
        verify_case(ws, CASE, "r1", Witness(ws, FakeWitness([witness_crop])))


# --- review round 3 -----------------------------------------------------------------------------


def test_round3_kind_first_typed_comparison():
    from kyb.witness.verify import canon_attr

    assert not value_match("Calle Sur 10 bis", "Calle Sur 10", "address")  # full, not contained
    assert value_match("Calle Sur 10", "calle sur, 10", "address")
    assert not value_match("25% al 2026-08-19", "68% al 2026-08-19", "pct")  # no date takeover
    assert not value_match("VG/2026-08-19/A", "AE/2026-08-19/B", "registry_id")
    period = canon_attr("stated_validity")
    assert period != canon_attr("valid_until")
    assert value_match("válido por 30 días", "válido por 30 días", period)
    assert value_match({"pct": 25}, "25.0%", "pct") and value_match({"pct": 25.5}, "25,5%", "pct")
    assert value_match("1.250 acciones", "1250", "shares")


# --- review round 4 -----------------------------------------------------------------------------


def test_round4_transfers_are_quoted_relationships():
    from kyb.witness.bind import transfer_binding as _transfer_binding

    note = "* L. Navarro recibe 10% extra de Altiplano el 3/8 - pendiente de asentar."
    quoted = {"value": "recibe 10% extra de Altiplano el 3/8 - pendiente de asentar"}
    assert _transfer_binding(quoted, {"value": "10%"}, note)
    for wrong in ("+10% to Altiplano", "recibe 10% extra de Ortega", "cede 10% a Altiplano"):
        assert not _transfer_binding({"value": wrong}, {"value": "10%"}, note), wrong
    assert not _transfer_binding(quoted, {"value": "15%"}, note)


def test_round4_numbers_keep_their_role_and_typed_value():
    from kyb.witness.verify import canon_attr

    assert not value_match("25% al 2026-08-19", "19% al 2026-08-25", "pct")
    assert not value_match(125.0, "1250 acciones", "shares")
    assert value_match(1250.0, "1250 acciones", "shares")
    assert value_match({"shares": 1250000}, "1.250.000 acciones", "shares")
    assert canon_attr("shares_acquired") != canon_attr("shares")


def test_round4_a_corroborating_claim_does_not_hide_a_conflict(ws):
    sixty = _claim(
        "c-B05-08", "V. Ortega Méndez", {"pct": 68}, "V. Ortega Méndez 68%", [1710, 760, 2300, 835]
    )
    _submit_all(ws, [_claim(*ORTEGA), sixty])
    answers = [
        crop("V. Ortega Méndez 25%", "V. Ortega Méndez", "25%"),
        crop("V. Ortega Méndez 68%", "V. Ortega Méndez", "68%"),
    ]
    rec = _verify(ws, FakeWitness(answers))
    conflicts = [x for x in rec["coverage"] if x["status"] == "conflict"]
    assert any("c-B05-08" in x.get("claims", []) for x in conflicts)


# --- review round 4: assurance is re-derived from the chain whenever it is read ------------------


def _w2_case(ws):
    plan = {e["doc"]: e for e in read_plan(ws, CASE, "r1")}
    payload = {
        "case": CASE,
        "run": "r1",
        "doc": "05",
        "doc_sha256": plan["05"]["doc_sha256"],
        "claims": [_claim(*ORTEGA)],
    }
    _attested_reader(ws, plan, payload)
    witness = FakeWitness(
        [crop("V. Ortega Méndez 25%", "V. Ortega Méndez", "25%")], model="gemini-3.1-pro-preview"
    )
    assert _verify(ws, witness)["assurance"] == "W2"
    return witness


def _summary_assurance(ws, witness):
    with workspace_action(ws):
        return verify_case(ws, CASE, "r1", Witness(ws, witness))["documents"]["05"]["assurance"]


def test_a_second_registration_found_after_submission_lowers_assurance(ws):
    from kyb.witness.verify import assurance_now

    witness = _w2_case(ws)
    tok = guard.HOOK_SOURCE.set("workspace")  # a workspace hook shows up later in the session
    try:
        guard.append_hook(
            {
                "cwd": str(ws),
                "session_id": "s1",
                "tool_name": "Read",
                "tool_input": {},
                "tool_use_id": "late-1",
            }
        )
    finally:
        guard.HOOK_SOURCE.reset(tok)
    now = assurance_now(ws, CASE, "r1", "05")
    assert now["assurance"] == "W1" and now["at_witness"] == "W2"
    assert "more than once" in now["assurance_reason"]
    assert _summary_assurance(ws, witness) == "W1"  # the cached verification is re-derived too


def test_a_later_mixed_model_stop_lowers_a_cached_w2(ws):
    """Codex's round-4 prediction: verify -> a later mixed-model stop -> verify must not return
    the cached W2."""
    witness = _w2_case(ws)
    later = ws.parent / "agent-a5-late.jsonl"
    later.write_text(
        "".join(
            json.dumps({"type": "assistant", "message": {"model": m}}) + "\n"
            for m in ("claude-opus-5-5", "gemini-3.1-pro-preview")
        )
    )
    stop = {
        "cwd": str(ws),
        "session_id": "s1",
        "agent_id": "a5",
        "agent_type": "kyb:kyb-reader",
        "agent_transcript_path": str(later),
    }
    guard.subagent_stop(stop)
    assert _summary_assurance(ws, witness) == "W1"


def test_untagged_hooks_beside_tagged_ones_are_ambiguous(ws):
    from kyb.audit.chain import chain_events
    from kyb.read.harness import hook_sources, registration_problem

    ev = {"cwd": str(ws), "session_id": "s7"}
    guard.subagent_start({**ev, "agent_id": "x", "agent_type": "kyb:kyb-reader"})  # untagged
    tok = guard.HOOK_SOURCE.set("plugin")
    try:
        guard.subagent_start({**ev, "agent_id": "x", "agent_type": "kyb:kyb-reader"})
    finally:
        guard.HOOK_SOURCE.reset(tok)
    events = chain_events(ws)
    assert hook_sources(events, "s7") == {"plugin", "untagged"}
    assert "more than once" in registration_problem(events, "s7")


def test_an_unsupported_hook_source_is_refused():
    from kyb.cli import main

    with pytest.raises(SystemExit) as exc:
        main(["guard", "subagent-start", "--source", "forged"])
    assert exc.value.code == 2


# --- review round 5 ------------------------------------------------------------------------------


def _deliver_post(ws, source, tool_use_id):
    tok = guard.HOOK_SOURCE.set(source)
    try:
        guard.append_hook(
            {
                "cwd": str(ws),
                "session_id": "s1",
                "tool_name": "Read",
                "tool_input": {},
                "tool_use_id": tool_use_id,
            }
        )
    finally:
        guard.HOOK_SOURCE.reset(tok)


def test_the_same_call_from_a_second_registration_is_still_evidence(ws):
    from kyb.witness.verify import assurance_now

    tok = guard.HOOK_SOURCE.set("plugin")  # the reader runs under the plugin's hooks
    try:
        witness = _w2_case(ws)
    finally:
        guard.HOOK_SOURCE.reset(tok)
    _deliver_post(ws, "plugin", "dup-1")
    assert assurance_now(ws, CASE, "r1", "05")["assurance"] == "W2"  # same source: a re-delivery
    _deliver_post(ws, "workspace", "dup-1")  # the same call, delivered by the other registration
    assert assurance_now(ws, CASE, "r1", "05")["assurance"] == "W1"
    assert _summary_assurance(ws, witness) == "W1"


def test_a_turn_without_a_model_makes_the_snapshot_incomplete(ws, tmp_path):
    t = tmp_path / "partial.jsonl"
    t.write_text(
        json.dumps({"type": "assistant", "message": {"model": "claude-opus-5-5"}})
        + "\n"
        + json.dumps({"type": "assistant", "message": {}})
        + "\n"
    )
    assert guard._transcript_models(str(t)) == (["claude-opus-5-5"], False)
    synthetic = tmp_path / "synthetic.jsonl"
    turns = [
        {"type": "assistant", "message": {"model": "claude-opus-5-5"}},
        {"type": "assistant", "message": {"model": "<synthetic>"}},
    ]  # harness-inserted
    synthetic.write_text("".join(json.dumps(t) + "\n" for t in turns))
    assert guard._transcript_models(str(synthetic)) == (["claude-opus-5-5"], True)


def test_the_persisted_summary_is_a_receipt_and_status_reports_current_assurance(ws):
    """Codex's round-5 prediction: verify W2 -> late disqualifying evidence -> verify must not
    leave a W2 that a consumer of the summary file or of review_status would read as current."""
    witness = _w2_case(ws)
    _deliver_post(ws, "workspace", "late-2")  # a second registration shows up later
    assert _summary_assurance(ws, witness) == "W1"
    persisted = json.loads((ws / f"silver/verified_claims/{CASE}/r1.json").read_text())
    doc = persisted["documents"]["05"]
    assert "assurance" not in doc and doc["assurance_at_verify"] == "W2"
    assert "assurance_now" in persisted["assurance_note"]
    status = review_status(ws, CASE)
    assert status["next"] == "decide" and status["assurance"]["05"] == "W1"


def test_a_summary_in_the_old_schema_is_stale_not_current(ws):
    """Round 6: a summary written before `assurance_at_verify` (a bare `assurance`) is never
    presented as current; status reports it stale and the receipt itself is not rewritten."""
    from kyb.util.hashing import sha256_bytes

    _w2_case(ws)
    rel = f"silver/verified_claims/{CASE}/r1.json"
    legacy = json.loads((ws / rel).read_text())
    doc = legacy["documents"]["05"]
    doc["assurance"] = doc.pop("assurance_at_verify")
    legacy.pop("assurance_note")
    body = (json.dumps(legacy, indent=2) + "\n").encode()
    (ws / rel).write_bytes(body)  # stands in for a receipt the earlier writer produced
    catalog = json.loads((ws / "catalog.json").read_text())
    catalog[rel]["content_hash"] = sha256_bytes(body)  # ...and its catalog entry
    (ws / "catalog.json").write_text(json.dumps(catalog, indent=2, sort_keys=True) + "\n")
    status = review_status(ws, CASE)
    assert status["next"] == "stale" and "pre-receipt summary schema" in status["detail"]


# --- L1 comparator hardening (issue): the binding classes of docs/analysis/b3-dogfood.md ---
# Every relaxation is paired with a negative: the same row with another party's value still fails.


def _t(subject, attribute, value, struck=False):
    return {"subject": subject, "attribute": attribute, "value": value, "struck": struck}


def _rr(*tuples, struck=()):
    return {"values": list(tuples), "struck": list(struck)}


def _c(subject, value, attribute=None, type_="owns", block="02-b5", **kw):
    return {
        "claim_id": "c-X-01",
        "type": type_,
        "subject": subject,
        "attribute": attribute,
        "value": value,
        "block_id": block,
        **kw,
    }


def _fact(block, subject, attribute, value):
    from kyb.read.schema import MaterialFact

    return MaterialFact(block_id=block, kind="k", subject=subject, attribute=attribute, value=value)


# B04 / B05 rows: three holders in one crop, re-read as one tuple each
REGISTER = _rr(
    _t("Lucas Navarro Solís", "Acciones Serie A", "2.300"),
    _t("Lucas Navarro Solís", "%", "46%"),
    _t("Valeria Ortega Méndez", "Acciones Serie A", "1.250"),
    _t("Valeria Ortega Méndez", "%", "25%"),
)


def test_class1_every_tuple_in_a_crop_is_a_candidate():
    from kyb.witness.bind import binding

    assert binding(_c("Lucas Navarro Solís", {"pct": 46}), REGISTER)
    assert binding(_c("Lucas Navarro Solís", {"shares": 2300}, "shares_held"), REGISTER)
    assert binding(_c("Valeria Ortega Méndez", {"pct": 25}), REGISTER)
    # negative: the same row, another holder's value
    assert not binding(_c("Valeria Ortega Méndez", {"pct": 46}), REGISTER)
    assert not binding(_c("Lucas Navarro Solís", {"shares": 1250}, "shares_held"), REGISTER)
    # a count never stands in for a percentage, nor a percentage for a count
    assert not binding(_c("Lucas Navarro Solís", {"pct": 2500}), REGISTER)
    assert not binding(_c("Lucas Navarro Solís", {"shares": 50}, "shares_held"), REGISTER)


def test_class1_a_labelled_tuple_binds_only_a_compatible_attribute():
    from kyb.witness.bind import binding

    row = _rr(
        _t("Lucas Navarro Solís", "Nacionalidad", "Mexicana"),
        _t("Lucas Navarro Solís", "Fecha de nacimiento", "1969-10-15"),
        _t("Valeria Ortega Méndez", "Fecha de nacimiento", "1976-10-15"),
    )
    assert binding(_c("Lucas Navarro Solís", "1969-10-15", "date_of_birth", "date"), row)
    assert binding(_c("Lucas Navarro Solís", "Mexicana", "nationality", "identifier"), row)
    assert not binding(_c("Lucas Navarro Solís", "1969-10-15", "incorporation_date"), row)
    assert not binding(_c("Lucas Navarro Solís", "1976-10-15", "date_of_birth", "date"), row)


def test_class1_live_rows_verify_each_value_end_to_end(ws):
    ortega_50 = _claim(
        "c-B05-08",
        "V. Ortega Méndez",
        {"pct": 46},
        "L. Navarro Solís 46%; V. Ortega Méndez 25%;",
        [1710, 760, 2300, 835],
    )
    _submit_all(ws, [_claim(*ORTEGA), ortega_50])
    row = {
        "transcription": "L. Navarro Solís 46%; V. Ortega Méndez 25%;",
        "readable": True,
        "reread": _rr(_t("L. Navarro Solís", "%", "46%"), _t("V. Ortega Méndez", "%", "25%")),
    }
    rec = _verify(ws, FakeWitness([row, row]))
    assert _result(rec, "c-B05-06")["verified"] is True
    assert _result(rec, "c-B05-08")["verified"] is False  # Navarro's 46% is not Ortega's


def test_class2_a_field_label_binds_for_the_documents_own_entity():
    from kyb.witness.bind import Scope, binding

    scope = Scope(
        ["SERVICIO DE ADMINISTRACIÓN TRIBUTARIA - CONSTANCIA DE SITUACIÓN FISCAL"],
        [_fact("02-b5", "CONSTANCIA DE SITUACIÓN FISCAL", "Fecha", "2026-10-25")],
        {"k1": {"reread": _rr(_t("Denominación", "Denominación", "IBERIA SOFTWARE SL"))}},
    )
    fecha = _rr(_t("Fecha", "Fecha", "2026-10-25"))
    estatus = _rr(_t("", "Estatus", "ACTIVO."))
    doc = _c("Constancia de Situación Fiscal", "2026-10-25", "issue_date", "document_date")
    assert binding(doc, fecha, scope=scope)
    entity = _c("Iberia Software, S.L.", "ACTIVO", "tax_status", "status")
    assert binding(entity, estatus, scope=scope)
    # negatives: not the document's entity; another value; a different attribute; no scope
    assert not binding(
        _c("Inversiones Altiplano, S.A.", "ACTIVO", "tax_status"), estatus, scope=scope
    )
    assert not binding(
        _c("Constancia de Situación Fiscal", "2026-09-11", "issue_date"), fecha, scope=scope
    )
    assert not binding(_c("Iberia Software", "ACTIVO", "tax_regime"), estatus, scope=scope)
    assert not binding(doc, fecha)


def test_class2_a_field_label_never_binds_another_partys_value():
    from kyb.witness.bind import Scope, binding

    # the same row names Valeria Ortega Méndez: her RFC is not the company's
    row = _rr(
        _t("Valeria Ortega Méndez", "RFC", "MADE761015M55"), _t("RFC", "RFC", "MADE761015M55")
    )
    scope = Scope(["ACTA CONSTITUTIVA — IBERIA SOFTWARE S.L."], [], {})
    company = _c(
        "Iberia Software, S.L.",
        "MADE761015M55",
        "rfc",
        "identifier",
        block="01-b6",
    )
    assert not binding(company, row, scope=scope)
    # the blind layout gives the block's RFC to a person: a bare label does not move it
    label_only = _rr(_t("RFC", "RFC", "MADE761015M55"))
    layout_says = Scope(
        scope.entities, [_fact("01-b6", "VALERIA ORTEGA MÉNDEZ", "RFC", "MADE761015M55")], {}
    )
    assert not binding(company, label_only, scope=layout_says)
    assert binding(company, label_only, scope=scope)  # the control: nobody else holds it


def test_class3_role_claims_bind_in_either_orientation():
    from kyb.witness.bind import binding

    board = _rr(
        _t("Presidente", "", "Lucas Navarro Solís"), _t("Secretario", "", "Valeria Ortega Méndez")
    )
    person_first = _rr(_t("Lucas Navarro Solís", "role", "Presidente"))
    lucas = _c("Lucas Navarro Solís", "Presidente", "board_role", "role")
    assert binding(lucas, board) and binding(lucas, person_first)
    assert binding(_c("Valeria Ortega Méndez", "Secretario", "board_role", "role"), board)
    # negatives: the same rows, the other person's role
    assert not binding(_c("Valeria Ortega Méndez", "Presidente", "board_role", "role"), board)
    assert not binding(_c("Lucas Navarro Solís", "Secretario", "board_role", "role"), board)
    assert not binding(_c("Valeria Ortega Méndez", "Presidente", "role", "role"), person_first)
    # an inversion is for roles only: a value is never read back as a subject
    assert not binding(_c("Lucas Navarro Solís", "Presidente", "tax_id", "identifier"), board)


def test_class4_text_compares_after_trailing_punctuation():
    from kyb.witness.bind import binding

    row = _rr(
        _t("Iberia Software", "Estatus", "ACTIVO."),
        _t("Inversiones Altiplano", "Estatus", "SUSPENDIDO."),
    )
    assert value_match("ACTIVO", "ACTIVO.", "tax_status")
    assert value_match("Servicios (tenedora de acciones)", "Servicios (tenedora de acciones).")
    assert binding(_c("Iberia Software", "ACTIVO", "tax_status", "status"), row)
    assert not value_match("ACTIVO", "INACTIVO.", "tax_status")
    assert not binding(_c("Iberia Software", "SUSPENDIDO", "tax_status", "status"), row)


def test_class5_structured_values_compare_by_component():
    from kyb.witness.bind import binding

    capital = _rr(_t("Capital social fijo", "Capital social fijo", "MXN 6.000.000"))
    shares = _rr(_t("", "Acciones", "5.000 acciones Serie C de MXN 1.000"))
    scope_free = {"subject": "Iberia Software"}
    money = _c(
        **scope_free,
        value={"amount": 6000000, "currency": "MXN"},
        attribute="fixed_share_capital",
        type_="capital",
    )
    from kyb.witness.bind import Scope

    scope = Scope(["ACTA — IBERIA SOFTWARE"], [], {})
    assert binding(money, capital, scope=scope)
    deal = _c(
        **scope_free,
        value={"shares": 5000, "series": "C", "par_value": 1000, "currency": "MXN"},
        attribute="total_shares",
        type_="capital",
    )
    assert binding(deal, shares, scope=scope)
    # negatives: a swapped count and nominal value; another amount; another currency; a series
    swapped = {
        **deal,
        "value": {"shares": 1000, "series": "C", "par_value": 5000, "currency": "MXN"},
    }
    assert not binding(swapped, shares, scope=scope)
    assert not binding(
        {**money, "value": {"amount": 1000, "currency": "MXN"}}, capital, scope=scope
    )
    assert not binding(
        {**money, "value": {"amount": 6000000, "currency": "USD"}}, capital, scope=scope
    )
    assert not binding({**deal, "value": {**deal["value"], "series": "B"}}, shares, scope=scope)
    # a register row: every component must be the same holder's
    row = _c("Lucas Navarro Solís", {"pct": 46, "shares": 2300}, "shareholding")
    assert binding(row, REGISTER)
    assert not binding({**row, "value": {"pct": 50, "shares": 1250}}, REGISTER)
    assert not binding({**row, "subject": "Valeria Ortega Méndez"}, REGISTER)


def test_a_struck_tuple_binds_only_a_flagged_claim():
    from kyb.witness.bind import binding

    row = _rr(_t("L. Navarro Solís", "%", "46%", struck=True), _t("L. Navarro Solís", "%", "68%"))
    assert binding(_c("L. Navarro Solís", {"pct": 68}), row)
    assert not binding(_c("L. Navarro Solís", {"pct": 46}), row)
    assert binding(_c("L. Navarro Solís", {"pct": 46}, flags=["struck-through"]), row)
    assert not binding(_c("L. Navarro Solís", {"pct": 68}, flags=["struck-through"]), row)
    assert not binding(_c("V. Ortega Méndez", {"pct": 46}, flags=["struck-through"]), row)


def test_names_fold_accents_but_not_letters():
    assert subject_match("Iberia Software, S.L.", "IBERIA SOFTWARE SL")
    assert not subject_match("Iberia Software", "GRUPO ALBORADA CAPITAL SA DE CV")


def test_layer_artifacts_are_not_pixel_mismatches():
    from kyb.witness.norm import contains_wrapped

    layer = (
        "Lucas Navarro Solís 2.300 46%Transmisión de 1.100 acciones a Altiplano\n"
        "Solís                                (asiento 4)\n"
    )
    pixels = "Lucas Navarro\nSolís\n2.300\n46%\nTransmisión de 1.100 acciones a Altiplano"
    assert contains_wrapped(pixels, "Lucas Navarro Solís 2.300 46%", layer)
    assert contains(pixels, "46%Transmisión de 1.100 acciones")  # a glued cell
    assert contains("SA DE CV . RFC: MCH180416AB1", "SA DE CV · RFC: MCH180416AB1")
    # negatives: a different continuation, another value, a continuation from another line
    assert not contains_wrapped(
        pixels.replace("Solís", "Ortega"), "Lucas Navarro Solís 2.300 46%", layer
    )
    assert not contains_wrapped(pixels, "Lucas Navarro Solís 2.300 68%", layer)
    assert not contains_wrapped(
        "Lucas Navarro (asiento 4) 2.300 46%", "Lucas Navarro Solís 2.300 46%", layer
    )
    assert not contains("SA DE CV . RFC: MCH180416AB1", "SA DE CV RFC: MCH180416AB1")
    assert not contains("146%Transmisión", "46% Transmisión")


def test_a_wrapped_text_layer_row_is_witnessed_without_a_finding(ws):
    row = _claim(
        "c-B04-03",
        "Lucas Navarro",
        {"pct": 46, "shares": 2300},
        "Lucas Navarro Solís 2.300 46%",
        [300, 700, 2300, 880],
        block="04-b3",
        source="text_layer",
        attribute="shareholding",
    )
    _submit_all(ws, [row], doc="04")
    pixels = {
        "transcription": "Lucas Navarro\nSolís\n2.300\n46%",
        "readable": True,
        "reread": _rr(
            _t("Lucas Navarro Solís", "Acciones", "2.300"), _t("Lucas Navarro Solís", "%", "46%")
        ),
    }
    rec = _verify(ws, FakeWitness([pixels]), doc="04")
    r = rec["results"][0]
    assert r["text_layer"] == "match" and r["transcription"] == "witnessed", r
    assert r["verified"] is True and not [
        f for f in rec["findings"] if f["kind"] == "layer_vs_pixel"
    ]


def test_a_transfer_binds_only_its_own_holders_row():
    from kyb.witness.bind import binding

    two_rows = (
        "Lucas Navarro Solís 2.300 46% Transmisión de 1.100 acciones a Altiplano\n"
        "Valeria Ortega Méndez 1.250 25% Transmisión de 900 acciones a Altiplano"
    )
    rr = _rr(
        _t("Lucas Navarro Solís", "notes", "Transmisión de 1.100 acciones a Altiplano"),
        _t("Lucas Navarro Solís", "%", "46%"),
        _t("Valeria Ortega Méndez", "notes", "Transmisión de 900 acciones a Altiplano"),
    )
    mine = _c("Lucas Navarro Solís", "Transmisión de 1.100 acciones a Altiplano", "share_transfer")
    assert binding(mine, rr, two_rows)
    # negatives: Valeria's transfer claimed for Lucas; a transfer the pixels do not quote
    assert not binding({**mine, "value": "Transmisión de 900 acciones a Altiplano"}, rr, two_rows)
    assert not binding({**mine, "value": "Transmisión de 1.100 acciones a Ortega"}, rr, two_rows)
    note = _rr(_t("L. Navarro", "nota", "recibe 10% extra de Altiplano - pendiente"))
    pending = _c(
        "L. Navarro",
        "L. Navarro recibe 10% extra de Altiplano - pendiente",
        "pending_share_acquisition",
        "amendment",
    )
    assert binding(pending, note, "* L. Navarro recibe 10% extra de Altiplano - pendiente")
    assert not binding(
        {**pending, "subject": "V. Ortega Méndez"},
        note,
        "* L. Navarro recibe 10% extra de Altiplano - pendiente",
    )


def test_review_hardening_the_layout_and_the_crop_guard_a_field_label():
    from kyb.witness.bind import Scope, binding

    # the layout names the claimant for the block's date, but with another value
    scope = Scope(
        [], [_fact("02-b5", "Lucas Navarro Solís", "Fecha de nacimiento", "1969-10-15")], {}
    )
    lucas = _c("Lucas Navarro Solís", "1976-10-15", "date_of_birth", "date")
    assert not binding(lucas, _rr(_t("", "Fecha de nacimiento", "1976-10-15")), scope=scope)
    assert binding(
        {**lucas, "value": "1969-10-15"},
        _rr(_t("", "Fecha de nacimiento", "1969-10-15")),
        scope=scope,
    )
    # a named party in the crop holds the very value, under no label
    company = _c("Iberia Software", "MADE761015M55", "rfc", "identifier")
    entity = Scope(["ACTA — IBERIA SOFTWARE"], [], {})
    unlabelled = _rr(
        _t("Valeria Ortega Méndez", "", "MADE761015M55"), _t("", "RFC", "MADE761015M55")
    )
    assert not binding(company, unlabelled, scope=entity)


def test_review_hardening_a_struck_role_line_withdraws_the_role():
    from kyb.witness.bind import binding

    rr = _rr(
        _t("Presidente", "", "Lucas Navarro Solís"),
        _t("Presidente", "", "Valeria Ortega Méndez"),
        struck=["Presidente: Lucas Navarro Solís"],
    )
    assert not binding(_c("Lucas Navarro Solís", "Presidente", "role", "role"), rr)
    assert binding(_c("Valeria Ortega Méndez", "Presidente", "role", "role"), rr)


def test_review_hardening_a_lone_mark_between_numbers_is_content():
    assert not contains("folio 1 . 250", "folio 1 250")
    assert contains("SA DE CV . RFC", "SA DE CV · RFC")


# --- Codex review round 1: five false accepts, each now a negative test -------------------------


def test_r1_another_named_company_owns_the_label_value():
    from kyb.witness.bind import Scope, binding

    a, b = "Iberia Software", "Inversiones Altiplano"
    scope = Scope([a, b], [_fact("02-b5", b, "RFC", "ALTIPLANO123")], {})
    row = _rr(_t(b, "RFC", "ALTIPLANO123"), _t("RFC", "RFC", "ALTIPLANO123"))
    assert not binding(_c(a, "ALTIPLANO123", "rfc"), row, scope=scope)
    # two companies with company-only facts: no unambiguous own entity, a bare label binds neither
    objects = [_fact("01-b7", a, "Objeto social", "x"), _fact("01-b8", b, "Objeto social", "y")]
    two = Scope([], objects, {})
    assert not binding(
        _c(a, "ALTIPLANO123", "rfc"), _rr(_t("RFC", "RFC", "ALTIPLANO123")), scope=two
    )


def test_r1_a_transfer_needs_the_claimants_own_transfer_tuple():
    from kyb.witness.bind import binding

    row = _rr(
        _t("Lucas Navarro Solís", "%", "46%"),
        _t("Valeria Ortega Méndez", "notes", "Transmisión de 46% a Altiplano"),
    )
    pixels = "Lucas Navarro Solís 46%\nValeria Ortega Méndez: Transmisión de 46% a Altiplano"
    claim = _c("Lucas Navarro Solís", "Transmisión de 46% a Altiplano", "share_transfer")
    assert not binding(claim, row, pixels)
    numberless = _rr(
        _t("Lucas Navarro Solís", "name", "Lucas Navarro Solís"),
        _t("Valeria Ortega Méndez", "notes", "Transmisión a Altiplano"),
    )
    both = "Lucas Navarro Solís\nValeria Ortega Méndez Transmisión a Altiplano"
    assert not binding({**claim, "value": "Transmisión a Altiplano"}, numberless, both)
    legacy = {"subject": "Lucas Navarro Solís", "value": "Lucas Navarro Solís"}
    assert not binding(
        {**claim, "value": "Transmisión a Altiplano"},
        legacy,
        "Lucas Navarro Solís Transmisión a Altiplano",
    )


def test_r1_a_struck_value_belongs_to_its_own_subject():
    from kyb.witness.bind import binding

    row = _rr(
        _t("Lucas Navarro Solís", "%", "46%"),
        _t("Valeria Ortega Méndez", "%", "25%", struck=True),
        struck=["25%"],
    )
    assert not binding(_c("Lucas Navarro Solís", {"pct": 25}, flags=["struck-through"]), row)
    assert binding(_c("Valeria Ortega Méndez", {"pct": 25}, flags=["struck-through"]), row)


def test_r1_a_structured_key_never_replaces_the_claimed_attribute():
    from kyb.witness.bind import binding

    row = _rr(_t("Lucas Navarro Solís", "Fecha de nacimiento", "1969-10-15"))
    wrong = _c("Lucas Navarro Solís", {"date_of_birth": "1969-10-15"}, "incorporation_date")
    assert not binding(wrong, row)
    assert binding({**wrong, "attribute": "date_of_birth"}, row)
    assert not binding(
        _c("Lucas Navarro Solís", {"pct": 46}, "date_of_birth"),
        _rr(_t("Lucas Navarro Solís", "%", "46%")),
    )


def test_r1_money_keeps_its_sign():
    from kyb.witness.bind import binding

    claim = _c(
        "Iberia Software",
        {"amount": 6000000, "currency": "MXN"},
        "fixed_share_capital",
        "capital",
    )
    for written in ("-MXN 6.000.000", "(MXN 6.000.000)", "MXN -6.000.000"):
        row = _rr(_t("Iberia Software", "Capital social fijo", written))
        assert not binding(claim, row), written
    assert binding(claim, _rr(_t("Iberia Software", "Capital social fijo", "MXN 6.000.000")))


def test_a_document_date_binds_for_the_document_and_never_moves_to_another_owner():
    from kyb.witness.bind import Scope, binding

    facts = [_fact("02-b5", "CONSTANCIA DE SITUACIÓN FISCAL", "Fecha", "2026-10-25")]
    scope = Scope(
        ["SERVICIO DE ADMINISTRACIÓN TRIBUTARIA - CONSTANCIA DE SITUACIÓN FISCAL"], facts, {}
    )
    fecha = _rr(_t("", "Fecha", "2026-10-25"))
    doc = _c("Constancia de Situación Fiscal", "2026-10-25", "issue_date", "date")
    assert binding(doc, fecha, scope=scope)
    # the blind layout gives the date to the document: it does not move to any company
    assert not binding({**doc, "subject": "IBERIA SOFTWARE SL"}, fecha, scope=scope)
    assert not binding({**doc, "subject": "Inversiones Altiplano, S.A."}, fecha, scope=scope)


def test_a_value_that_states_its_own_label():
    from kyb.witness.bind import Scope, binding

    scope = Scope(["ACTA — IBERIA SOFTWARE"], [], {})
    rr = _rr(_t("", "folio_mercantil", "N-2017101543"))
    folio = _c("Iberia Software", "N-2017101543", "folio_mercantil", "identifier")
    assert binding(folio, rr, scope=scope)
    assert not binding({**folio, "value": "N-2017051244"}, rr, scope=scope)
    assert not binding({**folio, "attribute": "rfc"}, rr, scope=scope)  # the label is a folio
    named = _rr(
        _t("Valeria Ortega Méndez", "", "N-2017101543"),
        _t("", "folio_mercantil", "N-2017101543"),
    )
    assert not binding(folio, named, scope=scope)  # another party holds the very value


# --- Codex review round 2: five more false accepts and one false reject, each now a test --------


def test_r2_a_value_never_overwrites_its_explicit_label():
    from kyb.witness.bind import binding

    named = _rr(_t("Acme", "legal_name", "Domicile Holdings S.A."))
    assert not binding(_c("Acme", "Holdings S.A.", "registered_address"), named)
    unlabelled_one_word = _rr(_t("Acme", "", "Domicile Holdings S.A."))
    assert not binding(_c("Acme", "Holdings S.A.", "registered_address"), unlabelled_one_word)
    assert binding(
        _c("Acme", "2026-10-25", "issue_date", "date"), _rr(_t("Acme", "", "Fecha: 2026-10-25"))
    )


def test_r2_claimed_amounts_keep_their_sign_and_decimals():
    from kyb.witness.bind import binding

    pos = _rr(_t("Acme", "capital", "MXN 6.000.000"))
    for amount in ("-5000000", "-6.000.000", "(6.000.000)", -5000000):
        claim = _c("Acme", {"amount": amount, "currency": "MXN"}, "capital")
        assert not binding(claim, pos), amount
    neg = _rr(_t("Acme", "capital", "-MXN 6.000.000"))
    assert binding(_c("Acme", {"amount": "-6.000.000", "currency": "MXN"}, "capital"), neg)
    cents = _rr(_t("Acme", "capital", "MXN 1.000,50"))
    for amount in (1000.50, "1000.50", "1.000,50"):
        assert binding(_c("Acme", {"amount": amount, "currency": "MXN"}, "capital"), cents), amount
    assert not binding(_c("Acme", {"amount": "1000.51", "currency": "MXN"}, "capital"), cents)


def test_r2_a_parent_and_its_subsidiary_are_two_companies():
    from kyb.witness.bind import Scope, binding, same_company

    scope = Scope(
        [],
        [
            _fact("01-b1", "Acme Holdings", "legal_name", "Acme Holdings"),
            _fact("01-b2", "Acme Holdings Colombia", "legal_name", "Acme Holdings Colombia"),
        ],
        {},
    )
    assert not binding(
        _c("Acme Holdings", "COL123", "rfc"), _rr(_t("RFC", "RFC", "COL123")), scope=scope
    )
    assert same_company("Iberia Software, S.L.", "IBERIA SOFTWARE SL")
    assert not same_company("Acme Holdings", "Acme Holdings Colombia")


def test_r2_a_struck_role_line_withdraws_the_role_in_every_orientation():
    from kyb.witness.bind import binding

    rr = _rr(_t("Alice Jones", "role", "Presidente"), struck=["Presidente: Alice Jones"])
    assert not binding(_c("Alice Jones", "Presidente", "role", "role"), rr)
    assert binding(
        _c("Alice Jones", "Presidente", "role", "role"),
        _rr(_t("Alice Jones", "role", "Presidente")),
    )


def test_r2_a_transfer_agrees_with_its_strike_state():
    from kyb.witness.bind import binding

    text = "Transmisión de 46% a Altiplano"
    current = _rr(_t("Alice Jones", "notes", text))
    struck = _rr(_t("Alice Jones", "notes", text, struck=True))
    flagged = _c("Alice Jones", text, "share_transfer", flags=["struck-through"])
    plain = _c("Alice Jones", text, "share_transfer")
    pixels = "Alice Jones " + text
    assert not binding(flagged, current, pixels)  # flagged struck, the pixels show it current
    assert binding(flagged, struck, pixels)
    assert not binding(plain, struck, pixels)  # presented as current, the pixels strike it
    assert not binding(plain, _rr(_t("Alice Jones", "notes", text), struck=[text]), pixels)
    assert binding(plain, current, pixels)
    assert not binding({**flagged, "subject": "Bob Smith"}, struck, pixels)


def test_r2_signs_hold_for_counts_and_percentages():
    from kyb.witness.bind import binding

    row = _rr(_t("Alice Jones", "%", "46%"), _t("Alice Jones", "acciones", "2.300"))
    assert not binding(_c("Alice Jones", "-46%", "pct"), row)
    assert not binding(_c("Alice Jones", {"shares": "-2500"}, "shares_held"), row)
    assert not binding(_c("Alice Jones", {"pct": 46}), _rr(_t("Alice Jones", "%", "-46%")))
    deal = _rr(_t("", "Acciones", "-5.000 acciones Serie A"))
    assert not binding(_c("Alice Jones", {"shares": 5000, "series": "A"}, "shares"), deal)
    assert value_match("2020-04-12", "2020-04-12", "issue_date")  # a date's hyphens are not signs
    assert value_match("N-2017101543", "N-2017101543", "registry_id")


def test_r7_a_cut_name_still_counts_as_a_rival_company():
    from kyb.witness.bind import Scope, binding

    facts = [_fact("01-b7", "IBERIA SOFTWARE S.L.", "Objeto social", "x")]
    cut = {
        "k1": {
            "transcription": "y acuerdan fundar una Sociedad Limitada denominada GRUPO",
            "reread": _rr(_t("GRUPO", "denominada", "GRUPO")),
        }
    }
    scope = Scope(["ESCRITURA PÚBLICA No. 48.223"], facts, cut)
    label = _rr(_t("", "Domicilio social", "Ciudad de México"))
    claim = _c(
        "Iberia Software, S.L.",
        "Ciudad de México",
        "registered_address",
        "address",
        block="01-b8",
    )
    assert "GRUPO" in scope.companies
    assert not binding(claim, label, scope=scope)  # deliberate loss under l1-a5 / spec
    # Without the ambiguous second name, an ordinary field-label control still binds.
    assert binding(claim, label, scope=Scope(scope.titles, facts, {}))
    # the layout names a second company: a bare label binds for neither
    two = Scope(
        scope.titles,
        [*facts, _fact("01-b9", "Inversiones Altiplano, S.A.", "Objeto social", "y")],
        cut,
    )
    assert not binding(claim, label, scope=two)


def test_a_persons_registered_address_does_not_make_them_a_company():
    from kyb.witness.bind import Scope, binding

    # as the live layout read 01: persons with a birth date and a "registered address"
    facts = [
        _fact("01-b6", "LUCAS NAVARRO SOLÍS", "registered address", "Av. Chapultepec 480"),
        _fact("01-b6", "LUCAS NAVARRO SOLÍS", "date of birth", "1969-10-15"),
        _fact("01-b6", "VALERIA ORTEGA MÉNDEZ", "registered address", "Av. Chapultepec 480"),
        _fact("01-b6", "VALERIA ORTEGA MÉNDEZ", "date of birth", "1976-10-15"),
        _fact("01-b7", "IBERIA SOFTWARE S.L.", "declared corporate object", "x"),
    ]
    scope = Scope(["ESCRITURA PÚBLICA No. 48.223 — ACTA CONSTITUTIVA"], facts, {})
    duration = _c("IBERIA SOFTWARE S.L.", "99 años", "duration", block="01-b8")
    assert binding(duration, _rr(_t("", "Duración", "99 años")), scope=scope)
    # negatives: a person is not the document's entity; a second company still blocks
    assert not binding(
        {**duration, "subject": "Lucas Navarro Solís"},
        _rr(_t("", "Duración", "99 años")),
        scope=scope,
    )
    two = Scope(
        scope.titles,
        [*facts, _fact("01-b9", "Inversiones Altiplano, S.A.", "declared corporate object", "y")],
        {},
    )
    assert not binding(duration, _rr(_t("", "Duración", "99 años")), scope=two)


# --- Codex review round 3: four more false accepts and one should-fix, each now a test ----------


def test_r3_a_complete_rival_name_read_in_a_crop_still_counts():
    from kyb.witness.bind import Scope, binding

    crops = {
        "k1": {
            "transcription": "Inversiones Altiplano",
            "reread": _rr(_t("Inversiones Altiplano", "legal_name", "Inversiones Altiplano")),
        }
    }
    scope = Scope([], [_fact("01-b1", "Acme Holdings", "Objeto social", "investments")], crops)
    claim = _c("Acme Holdings", "ALTIPLANO123", "rfc")
    assert not binding(claim, _rr(_t("RFC", "RFC", "ALTIPLANO123")), scope=scope)
    # a cut name that is not at the crop's edge is not a cut
    mid = {
        "k1": {
            "transcription": "denominada GRUPO IBERIA y otra",
            "reread": _rr(_t("", "denominada", "GRUPO")),
        }
    }
    facts = [_fact("01-b7", "IBERIA SOFTWARE", "Objeto social", "x")]
    assert not binding(
        _c("IBERIA SOFTWARE", "X1", "rfc"),
        _rr(_t("RFC", "RFC", "X1")),
        scope=Scope([], facts, mid),
    )


def test_r3_ranges_never_bind_a_single_percentage_and_signs_survive_punctuation():
    from kyb.witness.bind import binding

    for written in ("10 -20%", "10-20%", "10 - 20%", "10% - 20%", "10 a 20%"):
        row = _rr(_t("Alice Jones", "pct", written))
        for claimed in (-20, 20, 10):
            assert not binding(_c("Alice Jones", claimed, "pct"), row), (written, claimed)
    assert not binding(
        _c("Alice Jones", 46, "pct"), _rr(_t("Alice Jones", "%", "Participación:-46%"))
    )
    assert binding(_c("Alice Jones", -46, "pct"), _rr(_t("Alice Jones", "%", "Participación:-46%")))
    assert binding(_c("Alice Jones", -20, "pct"), _rr(_t("Alice Jones", "%", "-20%")))
    assert binding(_c("Alice Jones", -20, "pct"), _rr(_t("Alice Jones", "%", "total -20%")))


def test_r3_a_struck_transfer_agrees_on_signed_percentages():
    from kyb.witness.bind import binding

    flagged = _c("Alice Jones", "46% a Altiplano", "share_transfer", flags=["struck-through"])
    negative = _rr(_t("Alice Jones", "notes", "-46% a Altiplano", struck=True))
    assert not binding(flagged, negative, "Alice Jones -46% a Altiplano")
    same = _rr(_t("Alice Jones", "notes", "46% a Altiplano", struck=True))
    assert binding(flagged, same, "Alice Jones 46% a Altiplano")


def test_r3_a_struck_role_tuple_withdraws_the_role_in_either_orientation():
    from kyb.witness.bind import binding

    claim = _c("Alice Jones", "Presidente", "role", "role")
    for struck in (
        _t("Presidente", "", "Alice Jones", struck=True),
        _t("Alice Jones", "role", "Presidente", struck=True),
    ):
        assert not binding(claim, _rr(struck, _t("Alice Jones", "role", "Presidente"))), struck
    other = _t("Presidente", "", "Bob Smith", struck=True)
    assert binding(claim, _rr(other, _t("Alice Jones", "role", "Presidente")))


def test_r3_a_claimed_amount_must_be_written_whole():
    from kyb.witness.bind import binding

    row = _rr(_t("Acme", "capital", "MXN 6.000.000"))
    for amount in ("(5000000", "5000000)", "5000000-", "--5000000", "6.000.000x"):
        assert not binding(_c("Acme", {"amount": amount, "currency": "MXN"}, "capital"), row), (
            amount
        )
    assert binding(_c("Acme", {"amount": "6.000.000", "currency": "MXN"}, "capital"), row)


# --- Codex review round 4: prediction held; two identity false accepts, each now a test ---------


def test_r4_a_whole_legal_name_is_never_a_cut():
    from kyb.witness.bind import Scope, binding

    whole, rival = "Acme Holdings, S.L.", "Acme, S.L."
    crops = {
        "k1": {
            "transcription": "Razón social: " + rival,
            "reread": _rr(_t(rival, "legal_name", rival)),
        }
    }
    scope = Scope([], [_fact("01-b1", whole, "Objeto social", "investments")], crops)
    assert rival in scope.companies
    assert not binding(_c(whole, "ACM123", "rfc"), _rr(_t("RFC", "RFC", "ACM123")), scope=scope)


def test_r4_a_company_known_only_by_its_address_still_counts():
    from kyb.witness.bind import Scope, binding

    facts = [
        _fact("01-b1", "Acme Holdings", "Objeto social", "investments"),
        _fact("01-b9", "Inversiones Altiplano, S.A.", "registered_address", "Av. Chapultepec 480"),
    ]
    scope = Scope([], facts, {})
    assert not binding(
        _c("Acme Holdings", "ALTIPLANO123", "rfc"),
        _rr(_t("RFC", "RFC", "ALTIPLANO123")),
        scope=scope,
    )
    # the person control: a registered address given to someone with a birth date is a domicile
    persons = [
        facts[0],
        _fact("01-b6", "Lucas Navarro Solís", "registered address", "x"),
        _fact("01-b6", "Lucas Navarro Solís", "date of birth", "1969-10-15"),
    ]
    assert binding(
        _c("Acme Holdings", "AH123", "rfc"),
        _rr(_t("RFC", "RFC", "AH123")),
        scope=Scope([], persons, {}),
    )


# --- Codex review round 5: prediction held; one boundary false accept, now a test ---------------


def test_r5_a_legal_form_with_its_dot_is_still_a_legal_form():
    from kyb.witness.bind import Scope, binding

    whole, rival = "Acme Inc. Holdings, LLC", "Acme Inc."
    facts = [_fact("01-b1", whole, "registered_address", "Calle 1")]
    crops = {
        "k1": {
            "transcription": "Razón social: " + rival,
            "reread": _rr(_t(rival, "legal_name", rival)),
        }
    }
    scope = Scope([], facts, crops)
    assert rival in scope.companies
    assert not binding(_c(whole, "ACME123", "rfc"), _rr(_t("RFC", "RFC", "ACME123")), scope=scope)
    for form in ("Acme S.A.", "Acme SA", "Acme Ltda.", "Acme C.V."):
        assert (
            form
            in Scope(
                [],
                facts,
                {"k1": {"transcription": form, "reread": _rr(_t(form, "legal_name", form))}},
            ).companies
        ), form


def test_r5_a_company_with_a_nationality_is_still_a_company():
    from kyb.witness.bind import Scope, binding

    facts = [
        _fact("03-b1", "Iberia Software", "Objeto social", "x"),
        _fact("03-b7", "Inversiones Altiplano, S.A.", "registered_address", "Belice"),
        _fact("03-b7", "Inversiones Altiplano, S.A.", "Nacionalidad", "Belice"),
    ]
    scope = Scope([], facts, {})
    assert not binding(
        _c("Iberia Software", "ALTIPLANO123", "rfc"),
        _rr(_t("RFC", "RFC", "ALTIPLANO123")),
        scope=scope,
    )


# --- Codex review round 6: prediction held; one anchor false accept, now a test -----------------


def test_r6_an_address_known_company_never_anchors_a_cut():
    from kyb.witness.bind import Scope, binding

    for whole, rival in (
        ("Acme Holdings", "Acme"),
        ("Inversiones Altiplano Holdings", "Inversiones Altiplano"),
    ):
        facts = [
            _fact("01-b1", whole, "registered_address", "Calle 1"),
            _fact("01-b1", whole, "nationality", "Belice"),
        ]
        row = _rr(_t("RFC", "RFC", "RIVAL123"), _t(rival, "legal_name", rival))
        crops = {"k1": {"transcription": "RFC: RIVAL123\nRazón social: " + rival, "reread": row}}
        scope = Scope([], facts, crops)
        assert rival in scope.companies and whole in scope.companies, scope.companies
        assert not binding(_c(whole, "RIVAL123", "rfc"), row, scope=scope), whole


@pytest.mark.parametrize("source", ["legal_name", "Objeto social", "registered_address"])
@pytest.mark.parametrize(
    "rival", ["Acme", "Acme Inc.", "Acme S.A.", "Acme SA", "Acme Ltda.", "Acme C.V."]
)
def test_r7_every_crop_name_counts_regardless_of_layout_anchor(source, rival):
    from kyb.witness.bind import Scope, binding

    whole = rival + " Holdings"
    facts = [_fact("01-b1", whole, source, whole if source == "legal_name" else "x")]
    row = _rr(_t("RFC", "RFC", "RIVAL123"), _t(rival, "legal_name", rival))
    scope = Scope([], facts, {"k1": {"transcription": "Razón social: " + rival, "reread": row}})
    assert rival in scope.companies and whole in scope.companies
    assert not binding(_c(whole, "RIVAL123", "rfc"), row, scope=scope)
