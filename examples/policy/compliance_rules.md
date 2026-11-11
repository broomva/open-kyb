# Corporate Onboarding Risk Policy

**Version:** `2026.11-r1`
**Effective:** 2026-11-01

## 0. Decision Protocol

- **0.1 · Outcome Categories:** Onboarding evaluations resolve to `APPROVED`, `MANUAL_REVIEW`, or `REJECTED`. Every `REJECTED` or `MANUAL_REVIEW` determination must record the specific rule identifiers that triggered.
- **0.2 · Precedence & Escalation:**
  - Any triggered `REJECT` rule → `REJECTED`.
  - Any triggered `REVIEW` rule, or any rule whose evaluation remains `unknown` because required evidence was unverified or omitted → `MANUAL_REVIEW` unless a `REJECT` rule already applies.
  - Numeric risk weights (`+N`) sum into `risk_score`, bounded at a ceiling of `90` (`uncapped_score` preserves the raw total).
- **0.3 · Evidence Standard:** Only claims confirmed by the blind pixel witness enter the derived fact graph. Unverified claims and uncovered layout regions force dependent rules to `unknown`, never to clean absence.

---

## 1. Sanctions & Watchlist Screening — **REJECT / REVIEW**

- **SCR-01 (REJECT) · `kind: sanctions_corroborated`:** Any screened party (applicant, beneficial owner, director, legal representative, signatory, or intermediate holding vehicle) matching a sanctions list entry by normalized name or alias AND corroborated by a secondary identifier (registry code, tax ID, passport, or date of birth + jurisdiction).
- **SCR-02 (REVIEW) · `kind: sanctions_fuzzy`:** Any screened party matching a sanctions entry with Jaro-Winkler similarity score ≥ 0.88 where no secondary identifier confirms or refutes the hit.

## 2. Embargoed Jurisdictions — **REJECT**

- **JUR-01 · `kind: jurisdiction_prohibited`:** Incorporation or principal operations in, or a UBO resident in: `KP`, `IR`, `SY`, `AF`, `BY`, or the Crimea/Donetsk/Luhansk/Zaporizhzhia/Kherson regions of `UA`.

## 3. Monitored Jurisdictions — **SCORE +20**

- **JUR-02 · `kind: jurisdiction_monitored`:** Applicant, holding-chain vehicle, or a ≥ 20% owner domiciled in: `MM`, `VU`, `VG`, `BZ`, `WS`, `SC`, or free zones of `AE`.
- **UAE Free-Zone Carve-out:** `AE` is not monitored nationwide; only offshore and free-zone registrations (RAKEZ, IFZA, DMCC, JAFZA, Ajman Free Zone, UAQ FTZ, Hamriyah) trigger this rule. Mainland Dubai DED and Abu Dhabi DED/ADGM entities with audited statements are exempt.

## 4. Beneficial Ownership Resolution — **REVIEW / REJECT**

Every beneficial owner of **20% or more** direct or indirect equity or voting control must be identified with full legal name, nationality, and date of birth.

- **OWN-01 (REVIEW +25) · `kind: ubo_unresolved`:** Any ≥ 20% stake held by a corporate vehicle, trust, or foundation whose ultimate natural-person controllers are not resolved in the packet.
- **OWN-02 (REJECT) · `kind: ubo_high_risk_unresolved`:** ≥ 40% of ownership unresolved where the applicant or that holder is domiciled in a monitored jurisdiction.
- **Foundation & Trust Resolution:** A private foundation or trust is resolved only when a charter or declaration names a natural-person beneficiary or council controller with nationality and date of birth.

## 5. Governance, Nominees & Dormancy — **SCORE / REVIEW**

- **GOV-01 (SCORE +15) · `kind: nominee_control`:** Presence of a nominee director, nominee shareholder, or corporate service provider acting as officer.
- **GOV-02 (REVIEW) · `kind: nominee_sole_controller`:** When a nominee is the sole controller on record and the packet names no individual who actually runs the company.
- **GOV-03 (REVIEW +15) · `kind: dormant_or_layered`:** An applicant with **2** or more intermediate corporate holding layers or a declared dormant/inactive status.

## 6. Political Exposure (PEP) — **REVIEW / SCORE**

- **PEP-01 (REVIEW +30) · `kind: pep_active`:** Any UBO, director, or ≥ 20% shareholder who holds or held a prominent public function within the last **3 years** of `as_of`, or is an immediate family member or close associate of such an official.
- **PEP-02 (SCORE +15) · `kind: pep_former`:** A PEP whose public function ended more than 3 years ago with no other high-risk triggers.

## 7. Documentary Integrity & Discrepancies — **REVIEW +20**

- **DOC-01 · `kind: conflicting_filings`:** Material disagreement across packet documents on ownership stakes (differing by more than **2** points), share counts, legal name, addresses, registry identifiers, or incorporation date, OR an unexecuted/handwritten amendment noted in corporate records.

## 8. Mandatory Filings & Freshness — **REVIEW +15**

- **DOC-02 · `kind: required_documents`:** Onboarding requires: (1) current certificate of incorporation or registry extract (not expired by its stated validity); (2) executed UBO declaration dated within **6** months of `as_of`; (3) evidence of the registered or operating address dated within **3** months of `as_of`.

## 9. Operational Footprint & Shell Posture — **REVIEW / REJECT**

Evaluate the following shell-entity indicators:
1. Registered address is a mass-registration agent or PO Box;
2. No distinct operating address evidenced;
3. Incorporated less than **12** months before application;
4. Neither a bank reference nor financial statements on file;
5. No employees or operational website evidenced.

- **OPS-01 (REVIEW +20) · `kind: shell_indicators`:** **2 or more indicators → REVIEW.**
- **OPS-02 (REJECT) · `kind: shell_severe`:** **3 or more indicators in a monitored jurisdiction → REJECT.**

## 10. Sector & Activity Risk — **REJECT / SCORE**

- **ACT-01 (REJECT) · `kind: industry_prohibited`:** **Prohibited:** unlicensed banking or money transmission, shell bank, adult entertainment, online gambling, weapons or defence brokerage.
- **ACT-02 (SCORE +10) · `kind: industry_elevated`:** **Elevated:** money service business, virtual asset or crypto exchange, precious metals or stones trading, maritime vessel chartering, trade finance, charitable foundation administration.

## 11. Scoring Modifiers & Threshold Bands

- **MOD-01 (SCORE +10) · `kind: conflict_multiplier`:** Applied for each unresolved document discrepancy beyond the first.
- `risk_score` is capped at **90** (`uncapped_score` records the pre-cap total).
- `risk_score ≤ 25` with zero `REJECT` or `REVIEW` triggers → `APPROVED`.
- `risk_score ≥ 26` or any `REVIEW` trigger → `MANUAL_REVIEW`.

## 12. Officer Overrides & Precedent

An enrolled compliance officer may record a cryptographically signed override with written rationale into the hash-chained ledger; the signed record supersedes the unsigned proposal without mutating prior entries.

## 13. Witness Grounding & Citations

Every rule outcome and derived fact in a decision record must reference the source document, page index, crop coordinates, and pixel-verified text span.
