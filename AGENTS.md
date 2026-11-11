# Agent Instructions (`open-kyb`)

## Purpose

`open-kyb` is an autonomous KYB compliance review system. Models read documents and escalate risks; deterministic code evaluates `compliance_rules.md` and computes the `KYBDecision`; an enrolled compliance officer signs the final disposition.

## Key Commands

- `uv sync` — install dependencies
- `make lint` — run `ruff check` and `ruff format --check`
- `uv run pytest -q` — run the offline test suite
- `uv run kyb verify` — run the verification suites and write `artifacts/verify/scorecard.json`
- `make demo` — run the offline end-to-end demonstration over `examples/packets/`
- `uv run python scripts/make_examples.py` — regenerate synthetic PDF packets in `examples/packets/`

## Architectural Rules

1. **Never bypass hooks or workspace guards.** Workspace artifacts under `bronze/`, `silver/`, `gold/`, `policy/`, and `ledger/` are managed exclusively through `kyb` CLI / MCP actions.
2. **Reader isolation.** Launch `kyb-reader` subagents with the exact canonical prompt `kyb-reader <case> <run> <doc>` — one subagent per document.
3. **No hardcoded credentials or project IDs.** Vertex AI uses ADC and reads `KYB_VERTEX_PROJECT`, `KYB_VERTEX_QUOTA_PROJECT`, and `KYB_VERTEX_LOCATION` (default `global`) from the environment.
