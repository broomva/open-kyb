"""Future-layout vocabulary and actual cache/coverage boundaries; zero model inference.

Page bytes and served envelopes are explicit doubles. No test establishes autonomous layout
quality, independent witness assurance or permission to reinterpret historical recordings.
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import pytest

from kyb.witness import gemini
from kyb.witness.adapters import request
from kyb.witness.coverage import coverage
from kyb.witness.errors import WitnessRefused, WitnessUnavailable
from kyb.witness.port import Witness
from kyb.witness.vertex import VertexWitness

REPO = Path(__file__).resolve().parents[1]
OLD_LAYOUT = "Legacy layout prompt v0 (pre-jurisdiction-attribute)"
LAYOUT_SPEC = {"kinds": ["jurisdiction"], "doc_sha256": "sha256:offline-page"}
LAYOUT_DATA = {"pages": [{"page": 1, "blank": True, "blocks": []}], "material_facts": []}


class OfflineIO:
    """Real Vertex request shaping/decoding with both credential and HTTP I/O replaced."""

    def __init__(self):
        self.bodies = []
        self.token_calls = 0

    def token(self):
        self.token_calls += 1
        return "offline-fixture-" * 2

    def open(self, req, timeout):
        body = json.loads(req.data)
        self.bodies.append(body)
        schema = body["generationConfig"]["responseSchema"]
        data = LAYOUT_DATA if "pages" in schema["properties"] else {"crops": []}
        envelope = {
            "modelVersion": "gemini-3.1-pro-preview",
            "candidates": [
                {
                    "finishReason": "STOP",
                    "content": {"parts": [{"text": json.dumps(data)}]},
                }
            ],
        }
        response = io.BytesIO(json.dumps(envelope).encode())
        response.status = 200
        return response

    def adapter(self):
        return VertexWitness(
            model="gemini-3.1-pro-preview", retries=0, token_source=self.token, urlopen=self.open
        )


def test_real_transport_request_has_only_generic_method_and_supplied_page():
    fake = OfflineIO()
    image = b"offline-page-image"
    data, _ = fake.adapter().call("layout", [image], LAYOUT_SPEC)
    assert data == LAYOUT_DATA and fake.token_calls == 1
    (body,) = fake.bodies
    assert set(body) == {"contents", "generationConfig"}
    (content,) = body["contents"]
    assert content["role"] == "user"
    prompt, marker, pixels = content["parts"]
    assert prompt["text"] == request("layout", LAYOUT_SPEC)[0]
    assert marker == {"text": "[image 1]"}
    assert base64.b64decode(pixels["inline_data"]["data"]) == image
    text = prompt["text"]
    assert "use attribute jurisdiction" in text
    assert "retain the source label/description and ambiguity" in text
    assert "nationality column is not automatically proof" in text
    assert not any(s in text for s in ("Belice", "Altiplano", "B_grupo", "45", "70"))
    assert "attribute and value exactly as written" not in text


@pytest.mark.parametrize(
    "extra",
    [
        "claims",
        "failed_claims",
        "expected_country",
        "repair_targets",
        "target_score",
        "entity_ids",
        "text_layer",
    ],
)
def test_modern_request_identity_refuses_non_blind_layout_inputs(tmp_path, extra):
    fake = OfflineIO()
    with pytest.raises(ValueError, match="non-blind"):
        Witness(tmp_path, fake.adapter()).ask(
            "layout", [b"page"], {**LAYOUT_SPEC, extra: "ANSWER-SENTINEL"}
        )
    assert fake.bodies == [] and fake.token_calls == 0


def test_modern_record_replay_separates_layout_and_preserves_transcription(tmp_path, monkeypatch):
    fake = OfflineIO()
    adapter = fake.adapter()
    pages, crops = [b"offline-page"], [b"offline-crop"]
    transcribe_spec = {"crop_ids": ["k1"]}
    with monkeypatch.context() as patch:
        patch.setattr(gemini, "LAYOUT_PROMPT", OLD_LAYOUT)
        old = Witness(tmp_path, adapter)
        layout = old.ask("layout", pages, LAYOUT_SPEC)
        transcription = old.ask("transcribe", crops, transcribe_spec)
        _, old_folder = old._request("layout", pages, LAYOUT_SPEC)
        old_path = old_folder / "0.json"
        old_bytes = old_path.read_bytes()
    assert len(fake.bodies) == 2
    replay = Witness(tmp_path, adapter, mode="replay")
    new_key, new_folder = replay._request("layout", pages, LAYOUT_SPEC)
    assert new_key != layout.request_hash and new_folder != old_folder
    with pytest.raises(WitnessUnavailable, match="no recorded witness response"):
        replay.ask("layout", pages, LAYOUT_SPEC)
    same = replay.ask("transcribe", crops, transcribe_spec)
    assert same.replayed and same.request_hash == transcription.request_hash
    assert len(fake.bodies) == 2
    # The actual port also refuses an intact old record transplanted to the new key.
    transplant = Witness(tmp_path / "transplant", adapter, mode="replay")
    _, folder = transplant._request("layout", pages, LAYOUT_SPEC)
    folder.mkdir(parents=True)
    (folder / "0.json").write_bytes(old_bytes)
    with pytest.raises(WitnessRefused, match="different request"):
        transplant.ask("layout", pages, LAYOUT_SPEC)
    current = Witness(tmp_path, adapter).ask("layout", pages, LAYOUT_SPEC)
    assert current.request_hash == new_key and not current.replayed
    assert Witness(tmp_path, adapter, mode="replay").ask("layout", pages, LAYOUT_SPEC).replayed
    assert len(fake.bodies) == 3 and old_path.read_bytes() == old_bytes


def test_legacy_adapter_none_identity_collision_remains_a_held_limitation(tmp_path, monkeypatch):
    legacy = Witness(tmp_path, adapter=None)
    current_key, _ = legacy._request("layout", [b"page"], LAYOUT_SPEC)
    with monkeypatch.context() as patch:
        patch.setattr(gemini, "LAYOUT_PROMPT", OLD_LAYOUT)
        old_key, _ = legacy._request("layout", [b"page"], LAYOUT_SPEC)
    assert current_key == old_key  # Characterization only: this route is not admitted.


CLAIM = {
    "claim_id": "c-fixture-01",
    "type": "identity",
    "attribute": "jurisdiction",
    "subject": "Example Ventures Ltd",
    "block_id": "b1",
    "value": "Canada",
}
FACT = {
    "kind": "jurisdiction",
    "subject": "Example Ventures Ltd",
    "block_id": "b1",
    "attribute": "jurisdiction",
    "value": "Canada",
}


def test_predeclared_future_layout_key_matches_only_verified_source_components():
    # Both are hand-authored fixtures, not predictions from the new prompt.
    assert coverage([FACT], [CLAIM])[0]["status"] == "matched"
    assert coverage([FACT], [])[0]["status"] == "uncovered"


@pytest.mark.parametrize(
    "change,status",
    [
        ({"block_id": "foreign-block"}, "uncovered"),
        ({"subject": "Unrelated Holdings Ltd"}, "uncovered"),
        ({"attribute": "nationality"}, "uncovered"),
        ({"attribute": "residence"}, "uncovered"),
        ({"attribute": "registered_address"}, "uncovered"),
        ({"attribute": "constituida en"}, "uncovered"),
        ({"value": "Chile"}, "conflict"),
        ({"value": "Republic of Canada"}, "conflict"),
        ({"value": "Canada (proposed)"}, "conflict"),
    ],
)
def test_alignment_never_erases_subject_scope_field_or_value(change, status):
    assert coverage([{**FACT, **change}], [CLAIM])[0]["status"] == status
