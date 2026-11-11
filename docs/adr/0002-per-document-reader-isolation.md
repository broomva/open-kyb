# ADR-0002: Hook-Attested Per-Document Reader Isolation

- **Status:** Accepted
- **Date:** 2026-11-01

## Context

When a single agent reads an entire multi-document KYB packet at once, cross-document context contaminates extraction: the agent may silently reconcile contradictory ownership percentages across filings or follow instructions embedded in one document while reading another.

## Decision

Enforce one-document-per-subagent isolation (`kyb-reader <case> <run> <doc>`) through runtime hooks (`src/kyb/guard/guard.py`):
- `SubagentStart` records the canonical reader launch prompt and binds the subagent ID to a single `(case, run, doc)` tuple.
- `PreToolUse` blocks `Bash` and restricts `Read`, `Glob`, `read_plan`, `render_crop`, and `read_document` to that single document's rendered pages and crops.
- `SubagentStop` and `read_document` verify the session transcript and record a cryptographic isolation attestation.

## Consequences

- Cross-document contradictions (such as a 60%/40% incorporation deed vs. a 50%/25%/25% share register) are preserved faithfully in silver claims and surfaced deterministically by `OWN-02`.
- Any unattested or non-isolated claim submission is flagged by `J-READER-ISOLATION` and blocked from supporting an automatic approval.
