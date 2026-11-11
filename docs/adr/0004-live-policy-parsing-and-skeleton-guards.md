# ADR-0004: Live Policy Parsing with Section Skeleton Guards

- **Status:** Accepted
- **Date:** 2026-11-01

## Context

Compliance teams need to adjust policy parameters (PEP look-back windows, prohibited/high-risk jurisdiction lists, ownership thresholds, document recency windows, and score cutoffs) in `compliance_rules.md` without redeploying code, while preventing silent misinterpretation if someone rewrites a rule's logic in prose the engine does not implement.

## Decision

`src/kyb/policy/parse.py` parses every numeric parameter, code list, industry lexicon entry, and severity directly from `compliance_rules.md` at decision time, replacing each parsed slot with a placeholder (`\x00`) to form a normalized section skeleton. Each section's skeleton SHA-256 prefix is compared against `KNOWN_SKELETONS`:
- Parameter edits (e.g. `5 years` -> `3 years`, adding a country code, changing `risk_score <= 30` -> `20`) preserve the skeleton hash and take effect immediately.
- Prose changes that alter rule semantics outside parameter slots change the skeleton hash, mark the affected rule `unknown`, and append a `POLICY-GUARD` result that forces `MANUAL_REVIEW`.

Policy updates in a workspace follow a two-step `propose_policy` -> officer-signed `apply_policy` workflow (`src/kyb/policy/change.py`).

## Consequences

- Parameter changes require zero code changes or redeploys.
- Unrepresentable prose edits fail safe to `MANUAL_REVIEW` with an explicit reason.
