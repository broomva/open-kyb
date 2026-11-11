# Architecture

`open-kyb` is an auditable KYB compliance review engine designed for high-assurance document intake, beneficial ownership resolution, watchlist screening, and deterministic policy evaluation.

## 1. Core Invariants

1. **Models read; code decides.** Extraction and predicate answering use language/vision models under strict structural constraints, but rule evaluation, risk scoring, and status assignment (`APPROVED`, `MANUAL_REVIEW`, `REJECTED`) are 100% deterministic Python functions (`src/kyb/policy/`).
2. **Per-document isolation.** During extraction, each PDF in an applicant packet is assigned to a separate `kyb-reader` subagent invocation. Runtime hooks (`src/kyb/guard/`) block any reader from accessing other documents, workspace facts, or policy rules.
3. **Blind cross-family pixel verification.** Every extracted claim must cite a page and 300-dpi pixel bounding box (`bbox`). The witness port (`src/kyb/witness/`) crops that pixel region and sends it to a vision model from a different model family without the claim text or PDF text layer.
4. **Policy is live configuration with a structural guard.** `src/kyb/policy/parse.py` parses thresholds, windows, jurisdiction lists, industry lexicons, and point weights from `compliance_rules.md` at decision time. Each section's prose skeleton (text with parsed parameter slots removed) is hashed against `KNOWN_SKELETONS`; any unrepresentable structural edit triggers a `POLICY-GUARD` manual-review veto rather than silently misapplying a rule.
5. **Veto-only judges.** Safety judges (`src/kyb/judges/`) check arithmetic, precedence, reader isolation attestations, witness assurance (`W2` required for `APPROVED`), and evidence support (`J-SUPPORT`). Judges can only escalate to `MANUAL_REVIEW`.
6. **Tamper-evident hash-chained ledger.** Every workspace mutation appends a signed SHA-256 hash-chained event to `ledger/audit.jsonl` (`src/kyb/audit/chain.py`) and registers content-addressed artifacts in `catalog.json` (`src/kyb/workspace/catalog.py`).

## 2. Workspace Data Tiers

A tenant workspace initialized via `kyb init` materializes a strict directory hierarchy governed by `ontology/kyb.ontology.toml`:

- **`bronze/`**: Immutable raw inputs (`manifest.json` pinning SHA-256 digests of ingested packet PDFs).
- **`silver/`**: Intermediate extraction and verification records:
  - `silver/pages/<case>/<run>/` — 300-dpi rendered PNG pages and `pdftotext` text layers.
  - `silver/layout/<case>/<run>/` — claim-blind page layout blocks and material fact regions.
  - `silver/claims/<case>/<run>/` — per-document reader claim submissions (`agent_proposal` tier).
  - `silver/witness/<case>/<run>/` — blind pixel witness transcriptions and `bind-v6` verification verdicts.
- **`gold/`**: Derived compliance records:
  - `gold/facts/<case>/<run>.json` — reconciled `FactSet` (entities, ownership holdings graph, roles, documents, cross-document conflicts).
  - `gold/screening/<case>/<run>.json` — `ScreeningResult` (normalized Jaro-Winkler + Cyrillic/Greek/Arabic transliteration matches and identifier corroboration/conflict checks).
  - `gold/predicates/<case>/<run>.work.json` & `.answers.json` — bounded semantic predicate work orders and hook-attested subagent answers.
  - `gold/decisions/<case>/<run>/<decision_id>.json` — `DecisionRecord` containing the schema-validated `KYBDecision`, rule evaluations, judge outcomes, and signature state.
- **`ledger/`**: `ledger/audit.jsonl` (append-only SHA-256 hash chain), `ledger/checkpoint.json`, `ledger/officers.json`, and `ledger/policy/` (content-addressed policy proposals and officer approvals).

## 3. Witness Transport & Credential Isolation

- **Vertex AI via ADC only (`src/kyb/witness/vertex.py`).** Gemini witness calls use GCP Vertex AI (`https://aiplatform.googleapis.com/v1/projects/{project}/locations/{location}/publishers/google/models/{model}:generateContent`) configured via environment variables (`KYB_VERTEX_PROJECT`, `KYB_VERTEX_QUOTA_PROJECT`, `KYB_VERTEX_LOCATION`, default `global`).
- Short-lived OAuth tokens are obtained per request via `gcloud auth application-default print-access-token` with a scrubbed environment (`_GCLOUD_ENV`) and are never logged or written to disk.
- Direct developer API keys (`GEMINI_API_KEY`, `GOOGLE_API_KEY`) are refused (`src/kyb/witness/gemini.py`).

## 4. Copilot Q&A (`src/kyb/copilot/`)

- `kyb ask` retrieves relevant verified facts and pixel-witnessed page snippets using concept-aware BM25 (`src/kyb/copilot/lexicon.py`, `src/kyb/copilot/bundle.py`).
- `kyb record-answer` (`src/kyb/copilot/answer.py`) renders the final prose answer in deterministic code from the cited fact/snippet IDs. Any attempt by an agent to supply an unverified value, a mismatched citation, or an unanswerable claim is rejected before rendering.
