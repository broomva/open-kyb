# Security Policy

## Security & Trust Model

`open-kyb` processes untrusted applicant PDF documents and evaluates them against compliance rules. Its security posture relies on defense in depth:

1. **Untrusted Document Isolation:** Applicant PDFs are treated as untrusted inputs that may contain prompt-injection attempts or spoofed text layers. Reader subagents run in isolated per-document contexts without shell access (`Bash` is blocked by `PreToolUse` hooks) and can only read their assigned document's rendered pages and crops.
2. **Cross-Family Pixel Verification:** Every claim is checked against 300-dpi cropped page pixels by a blind witness from a separate model family, neutralizing hidden text-layer injections.
3. **Credential Hygiene:**
   - No API keys, tokens, or secrets are stored in the repository or workspace ledger.
   - Direct Gemini API-key routes (`GEMINI_API_KEY` / `GOOGLE_API_KEY`) are refused in code.
   - GCP Vertex AI authentication uses short-lived OAuth tokens from Application Default Credentials (`gcloud auth application-default print-access-token`) with a minimal scrubbed environment; tokens are never logged or persisted.
   - Officer signing tokens are verified against salted PBKDF2-HMAC-SHA256 digests (`210,000` iterations) in `ledger/officers.json`.
4. **Tamper-Evident Audit Trail:** All workspace writes are guarded by `PreToolUse` / `PostToolUse` hooks, registered in `catalog.json` with SHA-256 digests, and chained in `ledger/audit.jsonl`.

## Reporting a Vulnerability

If you discover a security vulnerability in `open-kyb`, please report it privately via GitHub Security Advisories on [`https://github.com/broomva/open-kyb/security/advisories`](https://github.com/broomva/open-kyb/security/advisories) rather than opening a public issue.
