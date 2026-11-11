"""`load_policy`: `compliance_rules.md` → `PolicyModel` (invariant 3: policy is configuration).

Every value the engine applies (thresholds, windows, lists, severities, points) is read from the
text at decision time. Nothing the file states is hard-coded here; what *is* hard-coded is the
shape of each sentence the engine knows how to apply.

**The guard.** Each section is reduced to a skeleton: its text with every parsed value replaced
by a slot, then folded cosmetically (`text.cosmetic`). The skeleton's hash is compared with the
one this engine implements. So:
- a parameter edit (3 → 5 years, a new country, a new threshold, a new severity) changes a slot
  value, keeps the skeleton, and applies;
- a cosmetic edit (whitespace, bold, `->` vs `→`, `2` vs `two`) folds to the same skeleton and
  applies;
- any other wording change alters the skeleton, is recorded in `PolicyModel.unrepresentable`, and
  evaluates the affected rules (or `POLICY-GUARD` for a global section) as `unknown` →
  `MANUAL_REVIEW`, never a silent default.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, get_args

from kyb.contracts.b4 import PolicyModel, RuleSpec, Severity
from kyb.policy import industry, regions
from kyb.policy.text import NUM, cosmetic, fold, number
from kyb.util.hashing import sha256_bytes

B = r"(?:\*\*)?"
SEV = r"(?P<sev>REJECT|REVIEW|SCORE)"
PTS = r"(?:\s*\+?\s*(?P<pts>\d+))?"
GE = r"(?:≥|>=)"
LE = r"(?:≤|<=)"
ARROW = r"(?:→|->)"

SECTION_KINDS: dict[str, tuple[str, ...]] = {
    "sanctions": ("sanctions_corroborated", "sanctions_fuzzy"),
    "prohibited_jurisdictions": ("jurisdiction_prohibited",),
    "monitored_jurisdictions": ("jurisdiction_monitored",),
    "ownership": ("ubo_unresolved", "ubo_high_risk_unresolved"),
    "governance": ("nominee_control", "nominee_sole_controller", "dormant_or_layered"),
    "pep": ("pep_active", "pep_former"),
    "conflicts": ("conflicting_filings",),
    "documents": ("required_documents",),
    "operations": ("shell_indicators", "shell_severe"),
    "industry": ("industry_prohibited", "industry_elevated"),
}
SECTION_RULES = SECTION_KINDS
GLOBAL_SECTIONS = ("preamble", "protocol", "scoring", "overrides", "citations")
_TITLED = {
    "decision protocol": "protocol",
    "sanctions watchlist screening": "sanctions",
    "embargoed jurisdictions": "prohibited_jurisdictions",
    "monitored jurisdictions": "monitored_jurisdictions",
    "beneficial ownership resolution": "ownership",
    "governance nominees dormancy": "governance",
    "political exposure pep": "pep",
    "documentary integrity discrepancies": "conflicts",
    "mandatory filings freshness": "documents",
    "operational footprint shell posture": "operations",
    "sector activity risk": "industry",
    "scoring modifiers threshold bands": "scoring",
    "officer overrides precedent": "overrides",
    "witness grounding citations": "citations",
}
SECTION_ORDER = ("preamble", "protocol", *SECTION_KINDS, "scoring", "overrides", "citations")

KNOWN_SKELETONS: dict[str, str] = {
    "preamble": "cbdebd9f1bbfa853",
    "protocol": "f7c722d653451408",
    "sanctions": "267ba6fdc0f5d37a",
    "prohibited_jurisdictions": "467ce0413726e099",
    "monitored_jurisdictions": "07eb880f97db0177",
    "ownership": "b06b487cf1e3fd02",
    "governance": "a45a189acfa1c2a2",
    "pep": "e84afa00033dbb3c",
    "conflicts": "389bf381fb7a86bf",
    "documents": "db4b01fe81bb0096",
    "operations": "8cfda0303a2693fe",
    "industry": "8ef0bd220ec89f8a",
    "scoring": "126c99795fffd608",
    "overrides": "94717d231671e779",
    "citations": "603af9e32fec2343",
}

PARAM_KEYS = (
    "prohibited_jurisdictions",
    "prohibited_regions",
    "high_risk_jurisdictions",
    "high_risk_regions",
    "ownership_threshold_pct",
    "high_risk_unresolved_pct",
    "name_similarity_threshold",
    "pep_lookback_years",
    "pep_former_points",
    "max_holding_layers",
    "shell_review_min",
    "shell_reject_min",
    "ubo_declaration_max_age_months",
    "proof_of_address_max_age_months",
    "young_company_months",
    "auto_approve_max_score",
    "score_cap",
    "prohibited_industries",
    "high_risk_industries",
    "conflict_extra_points",
    "conflict_ownership_points",
)


class _Section:
    def __init__(self, key: str, text: str) -> None:
        self.key, self.text = key, text
        self.slots: list[tuple[int, int]] = []
        self.problems: list[str] = []

    def find(
        self, pat: str, *groups: str, what: str, flags: int = re.IGNORECASE | re.MULTILINE
    ) -> re.Match[str] | None:
        ms = list(re.finditer(pat, self.text, flags))
        if len(ms) != 1:
            self.problems.append(f"{what} could not be read (matched {len(ms)} times)")
            return None
        m = ms[0]
        for g in groups:
            if m.group(g) is not None:
                self.slots.append(m.span(g))
        return m

    def skeleton_hash(self) -> str:
        s = self.text
        for a, b in sorted(set(self.slots), reverse=True):
            s = f"{s[:a]}\x00{s[b:]}"
        return sha256_bytes(cosmetic(s).replace("\x00", "<v>").encode("utf-8"))[7:23]


def _split(text: str) -> dict[str, Any]:
    parts = re.split(r"(?m)^(?=##\s+)", text)
    out: dict[str, Any] = {"preamble": _Section("preamble", parts[0].strip()), "unknown": []}
    for raw in parts[1:]:
        body = raw.strip()
        heading = body.splitlines()[0].removeprefix("##").strip()
        title = re.sub(r"\s*[—–-]\s*\*\*.*$", "", re.sub(r"^\d+\.\s*", "", heading)).strip()
        key = _TITLED.get(fold(title))
        if key is None or key in out:
            out["unknown"].append(heading)
            continue
        sec = _Section(key, body)
        num = re.match(r"##\s+(\d+)\.", sec.text)
        if num:
            sec.slots.append(num.span(1))
        out[key] = sec
    return out


def _codes(sec: _Section, m: re.Match[str] | None) -> tuple[list[str], dict[str, str]]:
    """A jurisdiction list: "`KP`, `IR`, or the Crimea/... regions of `UA`" → codes, qualified."""
    if m is None:
        return [], {}
    plain: set[str] = set()
    qualified: dict[str, str] = {}
    for raw in m.group("list").split(","):
        item = re.sub(r"^\s*(?:or|and)\s+", "", raw.strip(), flags=re.IGNORECASE)
        codes = re.findall(r"`([A-Z]{2})`", item)
        if len(codes) != 1:
            sec.problems.append(f"cannot read jurisdiction list item {item!r}")
            continue
        if codes[0] in plain or codes[0] in qualified:
            sec.problems.append(
                f"`{codes[0]}` is listed more than once; state all of its regions in one entry"
            )
            continue
        rest = cosmetic(re.sub(r"`[A-Z]{2}`", " ", item))
        rest = re.sub(r"^the\s+", "", rest).strip()
        if rest:
            qualified[codes[0]] = rest
            if missing := regions.unmatched(rest):
                sec.problems.append(
                    f"`{codes[0]}` {rest!r}: no region lexicon for {missing}, "
                    "so it cannot be matched"
                )
        else:
            plain.add(codes[0])
    return sorted(plain), dict(sorted(qualified.items()))


def _industries(sec: _Section, m: re.Match[str] | None) -> list[str]:
    if m is None:
        return []
    out = []
    for raw in m.group("list").split(","):
        label = re.sub(r"^\s*(?:or|and)\s+", "", raw.strip(), flags=re.IGNORECASE).rstrip(". ")
        if not label:
            continue
        if industry.key(label) not in industry.LEXICON:
            sec.problems.append(f"industry {label!r} has no lexicon entry, so it cannot be matched")
        out.append(label)
    return out


_ONE = r"(?:REJECT|REVIEW|SCORE)(?:\s*\+\d+)?"
_SUMMARY = re.compile(rf"\s*{_ONE}(?:\s*/\s*{_ONE})*\s*", re.IGNORECASE)
_ONE_FULL = re.compile(rf"\s*{SEV}{PTS}\s*", re.IGNORECASE)


def _heading(sec: _Section) -> tuple[set[str], int | None]:
    """Severities (and optional points) from `## … — **…**`."""
    m = sec.find(
        r"\A##\s[^\x0a]*?[—–-]\s*\*\*(?P<sum>[^*\x0a]+)\*\*(?P<tail>[^\x0a]*)$",
        what="heading severity",
    )
    if m is None:
        return set(), None
    if not _SUMMARY.fullmatch(m.group("sum")):
        sec.problems.append(f"heading severity {m.group('sum')!r} is not SEVERITY [+points] / …")
        return set(), None
    sec.slots.append(m.span("sum"))
    sev = set(re.findall(r"REJECT|REVIEW|SCORE", m.group("sum").upper()))
    pts = re.findall(r"\+(\d+)", m.group("sum"))
    tail = re.search(r"\(\+(\d+)\)", m.group("tail"))
    if tail:
        sec.slots.append((m.start("tail") + tail.start(1), m.start("tail") + tail.end(1)))
        pts.append(tail.group(1))
    if len(pts) > 1:
        sec.problems.append("heading states more than one set of points")
        return sev, None
    return sev, int(pts[0]) if pts else None


def _agree(sec: _Section, what: str, *vals: Any) -> Any:
    def _norm(v: Any) -> Any:
        return tuple(sorted(v)) if isinstance(v, (set, frozenset)) else v

    present = [_norm(v) for v in vals if v is not None]
    if len(present) != len(vals) or len(set(present)) != 1:
        sec.problems.append(f"{what} is stated inconsistently: {sorted(map(str, set(present)))}")
        return None
    return present[0]


def _single(sec: _Section, hs: set[str], pts: int | None) -> tuple[str, int] | None:
    """A section whose heading states its single rule's severity and points."""
    if len(hs) != 1:
        sec.problems.append("heading must state exactly one severity")
        return None
    sev = next(iter(hs))
    if sev == "SCORE" and pts is None:
        sec.problems.append(f"{sec.key} is SCORE but states no points")
        return None
    return sev, pts or 0


def _sev(m: re.Match[str] | None) -> str | None:
    return m.group("sev").upper() if m else None


def _int(m: re.Match[str] | None, group: str = "n") -> int | None:
    return number(m.group(group)) if m else None


_VALID_SEVS: frozenset[str] = frozenset(get_args(Severity))


def _as_sev(raw: str) -> Severity:
    if raw not in _VALID_SEVS:
        raise ValueError(f"invalid severity: {raw!r}")
    return raw  # type: ignore[return-value]


def _composite(sec: _Section) -> set[str]:
    """A heading that summarises several sub-rules. Their severities and points come from the
    bullets; points in the composite heading are ambiguous and guarded."""
    hs, pts = _heading(sec)
    if pts is not None:
        sec.problems.append(f"{sec.key} heading states points on a composite rule")
    return hs


def _decl_single(sec: _Section, kind: str) -> tuple[str, str, int] | None:
    m = sec.find(
        rf"\*\*(?P<id>[A-Z0-9-]+)\s*·\s*`kind:\s*{kind}`:\*\*",
        "id",
        what=f"{kind} declaration",
    )
    spec = _single(sec, *_heading(sec))
    if m is None:
        return None
    rid = m.group("id")
    if spec is None:
        return rid, "REVIEW", 0
    return rid, spec[0], spec[1]


def _decl_sub(sec: _Section, kind: str) -> tuple[str, str, int] | None:
    m = sec.find(
        rf"\*\*(?P<id>[A-Z0-9-]+)\s*\((?P<sum>[^)\x0a]+)\)\s*·\s*`kind:\s*{kind}`:\*\*",
        "id",
        "sum",
        what=f"{kind} declaration",
    )
    if m is None:
        return None
    rid = m.group("id")
    raw_sum = m.group("sum")
    one = _ONE_FULL.fullmatch(raw_sum.strip())
    if not one:
        sec.problems.append(f"{rid} severity {raw_sum!r} is not SEVERITY [+points]")
        return rid, "REVIEW", 0
    sev = one.group("sev").upper()
    pts = int(one.group("pts")) if one.group("pts") else None
    if sev == "SCORE" and pts is None:
        sec.problems.append(f"{rid} is SCORE but states no points")
        return rid, "SCORE", 0
    return rid, sev, pts or 0


def _parse_sections(
    secs: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, tuple[str, str, int]]]:
    """Params and {kind: (rule_id, severity, score_delta)} for every section that parses."""
    p: dict[str, Any] = {}
    r: dict[str, tuple[str, str, int]] = {}
    thresholds: list[int | None] = []

    def sec(k: str) -> _Section | None:
        return secs.get(k)

    if s := sec("sanctions"):
        hs = _composite(s)
        a = _decl_sub(s, "sanctions_corroborated")
        b = _decl_sub(s, "sanctions_fuzzy")
        t = s.find(
            rf"score\s*{GE}\s*{B}(?P<t>\d*\.\d+|\d+){B}",
            "t",
            what="name similarity threshold",
        )
        if t:
            p["name_similarity_threshold"] = float(t.group("t"))
        if a and b:
            _agree(s, "heading severities", frozenset(hs), frozenset({a[1], b[1]}))
            r["sanctions_corroborated"], r["sanctions_fuzzy"] = a, b
    if s := sec("prohibited_jurisdictions"):
        m = s.find(r"resident in:\s*(?P<list>[^.\x0a]+)\.", "list", what="prohibited list")
        p["prohibited_jurisdictions"], p["prohibited_regions"] = _codes(s, m)
        if spec := _decl_single(s, "jurisdiction_prohibited"):
            r["jurisdiction_prohibited"] = spec
    if s := sec("monitored_jurisdictions"):
        m = s.find(r"domiciled in:\s*(?P<list>[^.\x0a]+)\.", "list", what="monitored list")
        p["high_risk_jurisdictions"], p["high_risk_regions"] = _codes(s, m)
        t = s.find(
            rf"a\s*{GE}\s*{B}(?P<t>{NUM})%{B}\s*owner", "t", what="monitored owner threshold"
        )
        thresholds.append(_int(t, "t"))
        if spec := _decl_single(s, "jurisdiction_monitored"):
            r["jurisdiction_monitored"] = spec
    if s := sec("ownership"):
        hs = _composite(s)
        t1 = s.find(rf"owner of\s*{B}(?P<t>{NUM})%\s*or more", "t", what="ownership threshold")
        a = _decl_sub(s, "ubo_unresolved")
        ta = s.find(
            rf"`kind:\s*ubo_unresolved`:\*\*\s*Any\s*{GE}\s*{B}(?P<t>{NUM})%{B}",
            "t",
            what="ubo_unresolved threshold",
        )
        b = _decl_sub(s, "ubo_high_risk_unresolved")
        tb = s.find(
            rf"`kind:\s*ubo_high_risk_unresolved`:\*\*\s*{GE}\s*{B}(?P<t>{NUM})%{B}"
            r"\s*of ownership unresolved",
            "t",
            what="unresolved high-risk threshold",
        )
        thresholds += [_int(t1, "t"), _int(ta, "t")]
        if tb:
            p["high_risk_unresolved_pct"] = _int(tb, "t")
        if a and b:
            _agree(s, "heading severities", frozenset(hs), frozenset({a[1], b[1]}))
            r["ubo_unresolved"], r["ubo_high_risk_unresolved"] = a, b
    if s := sec("governance"):
        hs = _composite(s)
        a = _decl_sub(s, "nominee_control")
        b = _decl_sub(s, "nominee_sole_controller")
        c = _decl_sub(s, "dormant_or_layered")
        layers = s.find(
            rf"with\s*{B}(?P<n>{NUM}){B}\s*or more intermediate corporate holding layers",
            "n",
            what="max holding layers",
        )
        if layers:
            p["max_holding_layers"] = _int(layers, "n")
        if a and b and c:
            _agree(s, "heading severities", frozenset(hs), frozenset({a[1], b[1], c[1]}))
            r["nominee_control"], r["nominee_sole_controller"], r["dormant_or_layered"] = a, b, c
    if s := sec("pep"):
        hs = _composite(s)
        a = _decl_sub(s, "pep_active")
        b = _decl_sub(s, "pep_former")
        t = s.find(
            rf"or\s*{GE}\s*{B}(?P<t>{NUM})%{B}\s*shareholder", "t", what="PEP holder threshold"
        )
        y1 = s.find(rf"within the last\s*{B}(?P<y>{NUM})\s*years", "y", what="PEP window")
        y2 = s.find(rf"more than\s*{B}(?P<y>{NUM}){B}\s*years ago", "y", what="former PEP window")
        thresholds.append(_int(t, "t"))
        years = _agree(s, "the PEP look-back", _int(y1, "y"), _int(y2, "y"))
        if years is not None:
            p["pep_lookback_years"] = years
        if b:
            p["pep_former_points"] = b[2]
        if a and b:
            _agree(s, "heading severities", frozenset(hs), frozenset({a[1], b[1]}))
            r["pep_active"], r["pep_former"] = a, b
    owner_sec = sec("monitored_jurisdictions") or sec("ownership") or sec("pep")
    if owner_sec and thresholds:
        t = _agree(owner_sec, "the ≥ owner threshold", *thresholds)
        if t is not None:
            p["ownership_threshold_pct"] = t
    if s := sec("conflicts"):
        m = s.find(
            rf"differing by more than\s*{B}(?P<n>{NUM}){B}\s*points",
            "n",
            what="ownership materiality",
        )
        if m:
            p["conflict_ownership_points"] = _int(m, "n")
        if spec := _decl_single(s, "conflicting_filings"):
            r["conflicting_filings"] = spec
    if s := sec("documents"):
        u = s.find(
            rf"UBO declaration dated within\s*{B}(?P<n>{NUM}){B}\s*months",
            "n",
            what="UBO declaration window",
        )
        a = s.find(
            rf"address dated within\s*{B}(?P<n>{NUM}){B}\s*months",
            "n",
            what="proof-of-address window",
        )
        if u:
            p["ubo_declaration_max_age_months"] = _int(u, "n")
        if a:
            p["proof_of_address_max_age_months"] = _int(a, "n")
        if spec := _decl_single(s, "required_documents"):
            r["required_documents"] = spec
    if s := sec("operations"):
        hs = _composite(s)
        y = s.find(
            rf"less than\s*{B}(?P<n>{NUM}){B}\s*months before application",
            "n",
            what="young-company window",
        )
        a = _decl_sub(s, "shell_indicators")
        b = _decl_sub(s, "shell_severe")
        rv = s.find(
            rf"`kind:\s*shell_indicators`:\*\*\s*{B}(?P<n>{NUM})\s+or more indicators"
            rf"\s*{ARROW}\s*{B}{SEV}",
            "n",
            "sev",
            what="shell REVIEW count",
        )
        rj = s.find(
            rf"`kind:\s*shell_severe`:\*\*\s*{B}(?P<n>{NUM})\s+or more indicators"
            rf" in a monitored jurisdiction\s*{ARROW}\s*{B}{SEV}",
            "n",
            "sev",
            what="shell REJECT count",
        )
        if y:
            p["young_company_months"] = _int(y, "n")
        if rv:
            p["shell_review_min"] = _int(rv, "n")
        if rj:
            p["shell_reject_min"] = _int(rj, "n")
        if a and b and rv and rj:
            _agree(s, "shell_indicators severity", a[1], _sev(rv))
            _agree(s, "shell_severe severity", b[1], _sev(rj))
            _agree(s, "heading severities", frozenset(hs), frozenset({a[1], b[1]}))
        if a:
            r["shell_indicators"] = a
        if b:
            r["shell_severe"] = b
    if s := sec("industry"):
        hs = _composite(s)
        a = _decl_sub(s, "industry_prohibited")
        b = _decl_sub(s, "industry_elevated")
        pr = s.find(
            r"\*\*Prohibited:\*\*\s*(?P<list>[^\x0a]+)$", "list", what="prohibited industries"
        )
        hr = s.find(r"\*\*Elevated:\*\*\s*(?P<list>[^\x0a]+)$", "list", what="elevated industries")
        p["prohibited_industries"] = _industries(s, pr)
        p["high_risk_industries"] = _industries(s, hr)
        if a and b:
            _agree(s, "heading severities", frozenset(hs), frozenset({a[1], b[1]}))
            r["industry_prohibited"], r["industry_elevated"] = a, b
    if s := sec("scoring"):
        mod = _decl_sub(s, "conflict_multiplier")
        cap = s.find(rf"capped at\s*{B}(?P<n>\d+)", "n", what="score cap")
        lo = s.find(rf"risk_score\s*{LE}\s*(?P<n>\d+)", "n", what="auto-approve threshold")
        hi = s.find(rf"risk_score\s*{GE}\s*(?P<n>\d+)", "n", what="review threshold")
        if mod:
            p["conflict_extra_points"] = mod[2]
            r["conflict_multiplier"] = mod
        if cap:
            p["score_cap"] = int(cap.group("n"))
        if lo and hi:
            if int(hi.group("n")) != int(lo.group("n")) + 1:
                s.problems.append(
                    "the APPROVED and MANUAL_REVIEW score bands leave a gap or overlap"
                )
            else:
                p["auto_approve_max_score"] = int(lo.group("n"))
    if (s := sec("preamble")) and (
        v := s.find(r"\*\*Version:\*\*\s*`(?P<v>[^`\x0a]+)`", "v", what="version")
    ):
        s.find(r"\*\*Effective:\*\*\s*(?P<d>\d{4}-\d{2}-\d{2})", "d", what="effective date")
        p["_version"] = v.group("v").strip()
    return p, r


def load_policy(path: str | Path) -> PolicyModel:
    return parse_policy(Path(path).read_bytes())


def parse_policy(raw: bytes) -> PolicyModel:
    text = raw.decode("utf-8")
    secs = _split(text)
    params, rules = _parse_sections(secs)
    version = params.pop("_version", "unknown")
    unrep: list[str] = [f"policy: unrecognised section {h!r}" for h in secs["unknown"]]
    specs: list[RuleSpec] = []
    for key, kinds in SECTION_KINDS.items():
        if key not in secs:
            unrep.append(f"{key}: section missing")
            continue
        body = secs[key].text
        for kind in kinds:
            if kind in rules:
                rid, sev, delta = rules[kind]
                specs.append(
                    RuleSpec(
                        rule_id=rid,
                        kind=kind,
                        severity=_as_sev(sev),
                        score_delta=delta,
                        text=body,
                    )
                )
            else:
                unrep.append(f"{key}: rule {kind} could not be read")
    sc = secs.get("scoring")
    for kind in ("conflict_multiplier",):
        if isinstance(sc, _Section) and kind in rules:
            rid, sev, delta = rules[kind]
            specs.append(
                RuleSpec(
                    rule_id=rid,
                    kind=kind,
                    severity=_as_sev(sev),
                    score_delta=delta,
                    text=sc.text,
                )
            )
        elif isinstance(sc, _Section):
            unrep.append(f"scoring: rule {kind} could not be read")
    for key in SECTION_ORDER:
        s = secs.get(key)
        if not isinstance(s, _Section):
            if key in GLOBAL_SECTIONS:
                unrep.append(f"{key}: section missing")
            continue
        for prob in s.problems:
            unrep.append(f"{key}: {prob}")
        if not s.problems and key in KNOWN_SKELETONS and s.skeleton_hash() != KNOWN_SKELETONS[key]:
            unrep.append(f"{key}: wording differs from the rule the engine implements")
    for k in PARAM_KEYS:
        if k not in params and not unrep:
            unrep.append(f"policy: parameter {k} could not be read")
    return PolicyModel(
        rules_version=version,
        content_hash=sha256_bytes(raw),
        params=params,
        rules=specs,
        unrepresentable=sorted(dict.fromkeys(unrep)),
    )


def version_id(policy: PolicyModel) -> str:
    """`rules_version#content_hash`: what a decision record and a work order pin (invariant 3)."""
    return f"{policy.rules_version}#{policy.content_hash}"


def section_texts(raw: bytes) -> dict[str, str]:
    """Each recognised section's raw text, keyed by section id, for human-facing diffs."""
    secs = _split(raw.decode("utf-8"))
    out = {k: s.text for k, s in secs.items() if isinstance(s, _Section)}
    return out | {f"unrecognised: {h}": h for h in secs["unknown"]}


def skeleton_hashes(raw: bytes) -> dict[str, str]:
    """For maintainers: the skeleton hash of each section, to review and pin in KNOWN_SKELETONS."""
    secs = _split(raw.decode("utf-8"))
    _parse_sections(secs)
    return {k: s.skeleton_hash() for k in SECTION_ORDER if isinstance(s := secs.get(k), _Section)}
