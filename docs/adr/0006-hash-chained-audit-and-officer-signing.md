# ADR-0006: Hash-Chained Audit Ledger and Officer Signing Gate

- **Status:** Accepted
- **Date:** 2026-11-01

## Context

Regulatory audits require cryptographic proof that onboarding records were not altered retroactively and that every final disposition (especially an override of a system recommendation) was authorized by an accountable human compliance officer.

## Decision

1. **SHA-256 hash chain (`src/kyb/audit/chain.py`).** Every workspace action appends an `AuditEvent` to `ledger/audit.jsonl` whose `hash` covers `(seq, timestamp, actor, action, input_hash, output_hash, detail, prev_hash)`. Periodic checkpoints (`ledger/checkpoint.json`) pin the chain head.
2. **Content-addressed artifact catalog (`src/kyb/workspace/catalog.py`).** Every bronze, silver, and gold artifact is registered in `catalog.json` with its SHA-256 digest, data tier, lineage inputs, and producer metadata; files are set read-only after publication.
3. **Officer signing gate (`src/kyb/signing/`).** Decisions emitted by `kyb decide` have `signature_state = "pending_signature"`. Recording a signature requires an enrolled officer's PBKDF2-HMAC-SHA256 verified token (`KYB_OFFICER_TOKEN`) and, for any status override, a non-empty written rationale.

## Consequences

- Any post-hoc edit, deletion, or reordering in `ledger/audit.jsonl` or `catalog.json` is detected by `kyb audit verify`.
- Unsigned agent recommendations are structurally distinct from officer-signed dispositions.
