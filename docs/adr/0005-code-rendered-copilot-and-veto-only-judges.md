# ADR-0005: Code-Rendered Copilot Q&A and Veto-Only Judges

- **Status:** Accepted
- **Date:** 2026-11-01

## Context

When compliance officers ask follow-up questions about an applicant packet, free-form LLM prose risks fabricating citations, paraphrasing numbers inaccurately, or implying an application is approved when it requires review.

## Decision

1. **Code-rendered answers (`src/kyb/copilot/answer.py`).** The agent selects structured statement descriptors (`fact`, `snippet`, `explain`, or `NOT_IN_DOCUMENTS`) referencing verified IDs from the `ask` bundle. Deterministic code verifies that every cited value appears in the referenced record and renders the final text and citations verbatim. A `Stop` hook (`src/kyb/copilot/display.py`) audits the transcript to flag any unverified factual text outside rendered answers.
2. **Veto-only judges (`src/kyb/judges/`).** Post-evaluation judges (`J-ARITH`, `J-PRECEDENCE`, `J-VERIFICATION`, `J-READER-ISOLATION`, `J-APPROVAL-ASSURANCE`, `J-SUPPORT`) inspect the assembled decision and evidence. A judge can only veto a decision into `MANUAL_REVIEW`; no judge can convert a `MANUAL_REVIEW` or `REJECTED` outcome into `APPROVED`.

## Consequences

- Fabricated values or forged citations are deterministically rejected by `record_answer`.
- Judge failures or unavailable judge transports fail safe to `MANUAL_REVIEW`.
