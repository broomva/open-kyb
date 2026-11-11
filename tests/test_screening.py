"""`screen` (L2; spec, spec, spec): golden screening for A/B/C, and the normalisers' properties.

The property tests are seeded and stdlib-only: candidates ⊇ triggers over perturbed list names,
idempotent normalisation, and recall never below the policy's own score.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from kyb.contracts.b4 import Entity, FactSet, Operations, ScreeningResult, TriFact
from kyb.facts.derive import DocInput, derive_factset, materiality
from kyb.screening.normalize import (
    d044,
    jaro_winkler,
    policy_score,
    recall_score,
    transliterate,
    variants,
)
from kyb.screening.screen import (
    RECALL_CUTOFF,
    entries,
    identifiers,
    policy_threshold,
    screen_factset,
)

REPO = Path(__file__).resolve().parents[1]
ASSETS = REPO / "examples"
WATCHLIST = json.loads((ASSETS / "policy/watchlists.json").read_text("utf-8"))
POLICY = (ASSETS / "policy/compliance_rules.md").read_text("utf-8")
FIXTURES = Path(__file__).parent / "fixtures" / "facts"


def _facts(name: str) -> FactSet:
    raw = json.loads((FIXTURES / f"{name}.json").read_text("utf-8"))
    docs = []
    for d in raw["documents"]:
        if "claim_fields" in raw:
            claims = []
            for row in d["claims"]:
                c = {k: v for k, v in zip(raw["claim_fields"], row, strict=False) if v is not None}
                base = {"page": 1, "bbox": [100, 100, 2000, 200], "block_id": "b1",
                        "source": "text_layer", "confidence": 0.9, "flags": []}  # fmt: skip
                claims.append({**base, **c})
            verdicts = {c["claim_id"]: True for c in claims}
        else:
            claims, verdicts = d["claims"], d["verdicts"]
        docs.append(DocInput(doc=d["doc"], document_name=d["document_name"], claims=claims,
                             verdicts=verdicts, coverage=d.get("coverage", [])))  # fmt: skip
    return derive_factset(raw["case"], raw["run"], raw["as_of"], docs,
                          materiality=materiality(POLICY))  # fmt: skip


def _screen(name: str, threshold: float = 0.85) -> ScreeningResult:
    return screen_factset(_facts(name), WATCHLIST, watchlist_version="2026-11-01#test",
                          threshold=threshold)  # fmt: skip


def _by_entry(result: ScreeningResult) -> dict[str, list]:
    out: dict[str, list] = {}
    for c in result.candidates:
        out.setdefault(c.entry.entry_id, []).append(c)
    return out


# --- golden -----------------------------------------------------------------------------------


def test_c_fires_scr02_on_san_0001_with_conflicting_identifiers():
    res = _screen("sg-strait-capital")
    (san,) = _by_entry(res)["SAN-0001"]
    assert 0.86 <= san.jw_score <= 0.93 and san.jw_score == pytest.approx(0.923232)
    assert san.matched_name == "Strait Capital FZE" and not san.alias_exact
    assert san.triggers_fuzzy
    assert san.identifiers.corroborating == []  # no SCR-01
    assert sorted(san.identifiers.conflicting) == ["jurisdiction", "reg_no"]  # VG/3184920 vs AE/...
    assert san.entry.registration_number == "DMCC-417733" and san.entry.jurisdiction == "AE"
    assert res.method == "candidates-v1+jw-0.85+d054"


def test_c_pep_0101_is_exact_and_corroborated():
    (pep,) = _by_entry(_screen("sg-strait-capital"))["PEP-0101"]
    assert pep.jw_score == 1.0 and pep.alias_exact
    assert not pep.triggers_fuzzy  # SCR-01 screens sanctions; PEP-01 reads PEPs (spec)
    assert pep.identifiers.corroborating == ["dob", "nationality+dob"]
    assert pep.identifiers.conflicting == []
    assert pep.entry.left_office == "2024-08-14" and pep.entry.in_office is False  # PEP-01 window


def test_b_ortega_vs_pep_0103_is_a_non_firing_note():
    res = _screen("es-iberia-software")
    (pep,) = _by_entry(res)["PEP-0103"]
    assert pep.jw_score >= 0.85 and not pep.triggers_fuzzy
    assert sorted(pep.identifiers.conflicting) == ["dob"]
    assert pep.identifiers.corroborating == []


def test_d054_b_sat_spelling_does_not_fire_and_c_still_does():
    """Normalisation rule: abbreviation letters are joined after the
    punctuation pass, so "SA DE CV" and "S.L." are one string. Literally, B's SAT
    spelling would score 0.828968 against SAN-0003 and fire; now every spelling scores 0.819981,
    and C keeps 0.928407 against SAN-0001."""
    from kyb.screening.normalize import d044_literal, jaro_winkler

    sat, deed = (
        "IBERIA SOFTWARE SL",
        "IBERIA SOFTWARE S.L.",
    )
    for spelling in (sat, deed):
        assert policy_score(spelling, "Iberia Fluvial Austral S.A.") == pytest.approx(0.833333)
    literal = jaro_winkler(d044_literal(sat), d044_literal("Iberia Fluvial Austral S.A."))
    assert literal == pytest.approx(0.82871794, abs=1e-6)  # what spec changes
    (san,) = _by_entry(_screen("es-iberia-software"))["SAN-0003"]
    assert not san.triggers_fuzzy and san.jw_score == pytest.approx(0.833333)
    assert sorted(san.identifiers.conflicting) == ["jurisdiction"]
    assert not any(c.triggers_fuzzy for c in _screen("es-iberia-software").candidates)  # golden B
    (c_san,) = _by_entry(_screen("sg-strait-capital"))["SAN-0001"]
    assert c_san.triggers_fuzzy and c_san.jw_score == pytest.approx(0.923232)


def test_a_has_no_hits():
    res = _screen("cl-andes-agro")
    assert not any(c.triggers_fuzzy for c in res.candidates)
    assert all(c.jw_score < 0.85 for c in res.candidates)


@pytest.mark.parametrize("name", ["cl-andes-agro", "es-iberia-software", "sg-strait-capital"])
def test_screening_is_deterministic_and_bound_to_its_facts(name):
    one, two = _screen(name), _screen(name)
    assert one.model_dump_json() == two.model_dump_json()
    assert one.fact_set_hash == _facts(name).fact_set_hash
    entity_ids = {e.entity_id for e in _facts(name).entities}
    assert {c.entity_id for c in one.candidates} <= entity_ids
    assert len({c.candidate_id for c in one.candidates}) == len(one.candidates)


# --- the policy's threshold is read, never hard-coded (invariant 3) ---------------------------


def test_threshold_comes_from_the_policy_text():
    assert policy_threshold(POLICY) == 0.88
    edited = POLICY.replace("score ≥ 0.88", "score ≥ 0.95")
    assert policy_threshold(edited) == 0.95
    res = _screen("sg-strait-capital", threshold=policy_threshold(edited))
    (san,) = _by_entry(res)["SAN-0001"]
    assert not san.triggers_fuzzy and res.method.startswith("candidates-v1+jw-0.95")
    with pytest.raises(ValueError, match="threshold"):
        policy_threshold(POLICY.replace("similarity score ≥ 0.88 where", "a close name"))


def test_a_lower_threshold_still_keeps_every_trigger_a_candidate():
    res = _screen("es-iberia-software", threshold=0.5)
    hits = [c for c in res.candidates if c.triggers_fuzzy]
    assert hits and all(c.jw_score >= 0.5 for c in hits)


# --- identifiers -------------------------------------------------------------------------------


def _entity(names, kind="company", **kw) -> Entity:
    return Entity(entity_id=f"e-{d044(names[0]).replace(' ', '-')}", kind=kind, names=names,
                  evidence=[], **kw)  # fmt: skip


def test_r1a_corroboration_is_the_policys_own():
    listed = {e.entry_id: e for e in entries(WATCHLIST)}
    delta = _entity(["Fluvial Austral"], identifiers={"reg_no": ["172260845"]}, jurisdiction=["PA"])
    assert identifiers(delta, listed["SAN-0003"]).corroborating == ["reg_no"]
    volkov = _entity(["Dimitri Volkov"], kind="person", nationality=["RU"], dob=["1968-07-21"])
    assert identifiers(volkov, listed["SAN-0002"]).corroborating == ["jurisdiction+dob"]
    # a date of birth alone does not corroborate a sanctions match (SCR-01)
    alone = _entity(["Dimitri Volkov"], kind="person", dob=["1968-07-21"])
    assert identifiers(alone, listed["SAN-0002"]).corroborating == []
    # missing values are never conflicts
    assert identifiers(_entity(["Fluvial Austral"]), listed["SAN-0003"]).conflicting == []
    # a registration number is compared only with registration numbers (Codex r1 #4): a
    # passport that happens to equal it neither corroborates nor hides the real disagreement
    ids = {"reg_no": ["DIFFERENT"], "passport": ["172260845"]}
    check = identifiers(_entity(["Fluvial Austral"], identifiers=ids), listed["SAN-0003"])
    assert check.corroborating == [] and check.conflicting == ["reg_no"]
    # one matching and one different value: both are reported (spec decides; L3 reads)
    ids = {"reg_no": ["172260845", "999"]}
    check = identifiers(_entity(["Fluvial Austral"], identifiers=ids), listed["SAN-0003"])
    assert check.corroborating == ["reg_no"] and check.conflicting == ["reg_no"]
    # each geographic field against its own (Codex r2 #4): a nationality never stands in for a
    # jurisdiction the entry states
    from kyb.contracts.b4 import WatchlistEntry

    entry = WatchlistEntry(
        entry_id="SAN-9",
        list_kind="sanctions",
        name="Juan Gomez",
        entity_type="individual",
        jurisdiction="CO",
        dob="1970-01-01",
    )
    juan = _entity(
        ["Juan Gomez"], kind="person", jurisdiction=["MX"], nationality=["CO"], dob=["1970-01-01"]
    )
    check = identifiers(juan, entry)
    assert check.corroborating == [] and check.conflicting == ["jurisdiction"]
    # an individual entry that states only a nationality reads it (the explicit fallback)
    assert identifiers(volkov, listed["SAN-0002"]).corroborating == ["jurisdiction+dob"]


def test_alias_exact_and_transliteration_fire():
    fs = _facts("cl-andes-agro")
    extra = [_entity(["Fluvial Austral"]), _entity(["Димитри Волков"], kind="person"),
             _entity(["Tareq Al-Mansoor"], kind="person")]  # fmt: skip
    res = screen_factset(fs.model_copy(update={"entities": [*fs.entities, *extra]}), WATCHLIST,
                         watchlist_version="t", threshold=0.85)  # fmt: skip
    by = {(c.entity_id, c.entry.entry_id): c for c in res.candidates}
    assert by[("e-fluvial-austral", "SAN-0003")].alias_exact
    assert by[("e-fluvial-austral", "SAN-0003")].triggers_fuzzy
    cyrillic = next(c for (e, k), c in by.items() if k == "SAN-0002")
    assert cyrillic.triggers_fuzzy and cyrillic.jw_score == 1.0
    assert ("e-tareq-al-mansoor", "SAN-0004") in by  # a recall candidate for the agent


def test_jaro_winkler_reference_values():
    assert jaro_winkler("martha", "marhta") == pytest.approx(0.961111, abs=1e-6)
    assert jaro_winkler("dwayne", "duane") == pytest.approx(0.84, abs=1e-6)
    assert jaro_winkler("dixon", "dicksonx") == pytest.approx(0.813333, abs=1e-6)
    assert jaro_winkler("", "") == 0.0 and jaro_winkler("abc", "abc") == 1.0


# --- properties ----------------------------------------------------------------------------------

_ACCENTS = {"a": "á", "e": "é", "i": "í", "o": "ó", "u": "ú", "n": "ñ", "c": "ç"}
_SUFFIXES = ["", " S.A.", " SA", " Ltd", " LLC", " S.L.", " FZE", " Inc."]


def _perturb(rng: random.Random, name: str) -> str:
    """A spelling variant a document might use: accents, case, punctuation, suffixes, token
    order, a dropped or doubled letter."""
    toks = name.split()
    op = rng.randrange(7)
    if op == 0:
        toks = [("".join(_ACCENTS.get(ch, ch) if rng.random() < 0.3 else ch for ch in t))
                for t in toks]  # fmt: skip
    elif op == 1:
        toks = [t.upper() if rng.random() < 0.5 else t.lower() for t in toks]
    elif op == 2 and len(toks) > 1:
        rng.shuffle(toks)
    elif op == 3:
        i = rng.randrange(len(toks))
        if len(toks[i]) > 3:
            j = rng.randrange(len(toks[i]))
            toks[i] = toks[i][:j] + toks[i][j + 1 :]
    elif op == 4:
        i = rng.randrange(len(toks))
        j = rng.randrange(len(toks[i]))
        toks[i] = toks[i][:j] + toks[i][j] + toks[i][j:]
    elif op == 5:
        toks = [t + rng.choice([",", ".", "-", ""]) for t in toks]
    out = " ".join(toks)
    return out + rng.choice(_SUFFIXES) if op == 6 else out


def _corpus() -> list[str]:
    names = [n for kind in ("sanctions", "pep") for e in WATCHLIST[kind]
             for n in [e["name"], *e["aliases"]]]  # fmt: skip
    rng = random.Random(20261027)
    out = []
    for n in names:
        out += [_perturb(rng, n) for _ in range(25)]
    return out + [
        "Iberia Software, S.L.",
        "Strait Capital Pte. Ltd.",
        "Valeria Ortega Méndez",
        "Andes Agropecuaria SpA",
        "Lucas Navarro Solís",
        "Άννα Παπαδοπούλου",
        "Νίκος Καζαντζάκης",
        "Димитри Волков",
        "Ёлка Ўзбек",
    ]


def test_normalisers_are_idempotent():
    for name in _corpus():
        once = d044(name)
        assert d044(once) == once
        for key, form in variants(name).items():
            if key in ("d044", "suffix", "token_order"):
                assert variants(form)[key] == form, (name, key)


def test_spacing_inside_abbreviations_is_one_name():
    from kyb.screening.normalize import d044_literal

    assert d044("S.L.") == "sl" and d044("SA DE CV") == "sa de cv"  # spec
    assert d044_literal("S.L.") != d044_literal("SA DE CV")  # spec before it
    assert d044("Paredes Vega") == d044("PAREDES VEGA")
    assert d044("Άννα") == d044(d044("Άννα")) == "anna"  # Codex r1 #11
    assert transliterate("Волков") == "volkov"


def test_recall_is_never_below_the_policy_score():
    listed = [n for kind in ("sanctions", "pep") for e in WATCHLIST[kind]
              for n in [e["name"], *e["aliases"]]]  # fmt: skip
    for name in _corpus():
        for x in listed:
            assert recall_score(name, x) + 1e-6 >= policy_score(name, x), (name, x)


def test_candidates_contain_every_trigger():
    """Over perturbed list names screened as persons and as companies, at the policy's
    threshold and at lower ones: every pair the policy's comparison hits is a candidate."""
    corpus = _corpus()
    ents = []
    for i, name in enumerate(corpus):
        ents.append(Entity(entity_id=f"e-p{i}", kind="person", names=[name], evidence=[]))
        ents.append(Entity(entity_id=f"e-c{i}", kind="company", names=[name], evidence=[]))
    unknown = TriFact(state="unknown")
    fs = FactSet(case="P", run="r1", as_of="2026-10-25", fact_set_hash="sha256:x",
                 applicant="e-c0", entities=ents, holdings=[], roles=[], documents=[],
                 conflicts=[], operations=Operations(distinct_operating_address=unknown,
                                                     bank_reference_or_financials=unknown,
                                                     employees_or_website=unknown),
                 evidence=[], verification=[])  # fmt: skip
    listed = entries(WATCHLIST)
    for threshold in (0.85, 0.75, RECALL_CUTOFF - 0.1):
        res = screen_factset(fs, WATCHLIST, watchlist_version="t", threshold=threshold)
        got = {(c.entity_id, c.entry.entry_id) for c in res.candidates}
        hits = 0
        for e in ents:
            for entry in listed:
                names = [entry.name, *entry.aliases]
                aliases = {d044(a) for a in entry.aliases}
                if (max(policy_score(e.names[0], x) for x in names) >= threshold
                        or d044(e.names[0]) in aliases):  # fmt: skip
                    hits += 1
                    assert (e.entity_id, entry.entry_id) in got, (e.names[0], entry.entry_id)
        assert hits > 100
        assert all(c.triggers_fuzzy == (c.entry.list_kind == "sanctions"
                                      and (c.jw_score >= threshold or c.alias_exact))
                   for c in res.candidates)  # fmt: skip
