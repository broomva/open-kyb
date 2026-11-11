# ADR-0003: Blind Cross-Family Pixel Witness

- **Status:** Accepted
- **Date:** 2026-11-01

## Context

PDF text layers can omit scanned pages, miss handwritten amendments or strike-throughs, or contain hidden text-layer spoofing that differs from visible pixels. Furthermore, if the same model family both extracts and verifies a claim, correlated blind spots can pass undetected.

## Decision

Verify every document through a two-stage blind pixel witness (`src/kyb/witness/`):
1. `layout(pages)`: a claim-blind pass over 300-dpi page images that identifies layout blocks and material facts without seeing the PDF text layer.
2. `transcribe(crops)`: a blind pass over 300-dpi cropped bounding boxes (`k1`, `k2`, ...) that transcribes visible text, struck-through values, and handwritten annotations without seeing the reader's proposed claim.

When the reader runs on an Anthropic or OpenAI model, Stratum A routes witness calls to Google Gemini on GCP Vertex AI (`vertex-gcp`) via Application Default Credentials.

## Consequences

- Struck-through numbers and handwritten margin notes (such as on scanned board minutes) are captured as multi-valued claims and surface ownership conflicts.
- Claims that fail pixel binding (`bind-v6`) remain `unverified` and cannot clear compliance rules.
