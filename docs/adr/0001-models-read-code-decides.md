# ADR-0001: Models Read and Escalate; Deterministic Code Decides

- **Status:** Accepted
- **Date:** 2026-11-01

## Context

In regulated KYB onboarding, delegating final compliance judgement to a language model introduces unacceptable risks: prompt injection from applicant PDFs, non-deterministic threshold drift, and unexplainable approvals.

## Decision

Restrict models to two narrow roles:
1. Extracting candidate claims with bounding-box citations from individual documents (`kyb-reader`) and transcribing cropped pixel regions (`kyb.witness`).
2. Answering narrow boolean predicates (`kyb-predicates`) when a deterministic rule requires semantic disambiguation (e.g., whether a lease covers an operating address).

All entity resolution, ownership percentage rollups, policy rule evaluations (`SCR-`, `JUR-`, `IND-`, `OWN-`, `PEP-`, `GOV-`, `OPS-`, `DOC-`), risk-score arithmetic (`BANDS`), and status composition (`APPROVED`, `MANUAL_REVIEW`, `REJECTED`) execute in deterministic Python (`src/kyb/facts/`, `src/kyb/policy/`, `src/kyb/decide/`).

## Consequences

- Same verified facts, screening results, policy text, and predicate answers always produce a byte-identical `KYBDecision`.
- No model output can waive a triggered `REJECT` or `REVIEW` rule.
