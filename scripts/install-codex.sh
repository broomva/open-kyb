#!/bin/sh
# Install kyb into Codex (spec): the skills, the reader instructions, and the kyb MCP server.
# Run it from a clone of broomva/open-kyb (it copies the skills from ./plugin).
#
#   scripts/install-codex.sh                 # skills to ~/.agents/skills, server from GitHub
#   KYB_DEV_REPO=$PWD scripts/install-codex.sh   # server from this checkout (development)
#   KYB_SKILLS_DIR=.agents/skills scripts/install-codex.sh   # skills for one repo only
#
# Managed Codex processes bind actual served-model attestation; native encrypted subagent
# launches stay unattested. The managed runtime hooks require normal Codex trust review.
set -eu

here=$(cd "$(dirname "$0")/.." && pwd)
skills_dir=${KYB_SKILLS_DIR:-"$HOME/.agents/skills"}

command -v codex >/dev/null 2>&1 || { echo "codex is not on PATH" >&2; exit 1; }
command -v uv >/dev/null 2>&1 || { echo "uv is not on PATH (https://docs.astral.sh/uv/)" >&2; exit 1; }

[ -d "$here/plugin/skills" ] || { echo "run this from a clone of broomva/open-kyb" >&2; exit 1; }

mkdir -p "$skills_dir"
for skill in "$here"/plugin/skills/*/; do
  name=$(basename "$skill")
  rm -rf "${skills_dir:?}/$name"
  cp -R "$skill" "$skills_dir/$name"
  echo "skill  $name -> $skills_dir/$name"
done
# Codex has no kyb-reader agent type: kyb-review hands the reader's instructions to a generic
# subagent from here
cp "$here/plugin/agents/kyb-reader.md" "$skills_dir/kyb-review/reader.md"
cp "$here/plugin/agents/kyb-predicates.md" "$skills_dir/kyb-review/predicates.md"
echo "reader kyb-reader instructions -> $skills_dir/kyb-review/reader.md"

codex mcp remove kyb >/dev/null 2>&1 || true
if [ -n "${KYB_DEV_REPO:-}" ]; then
  codex mcp add kyb --env KYB_WITNESS_STRATUM=A --env KYB_WITNESS_A_BACKENDS=vertex-gcp:gemini-3.1-pro-preview,vertex-gcp:gemini-3.8-flash -- uv run --quiet --project "$KYB_DEV_REPO" kyb-mcp
else
  codex mcp add kyb --env KYB_WITNESS_STRATUM=A --env KYB_WITNESS_A_BACKENDS=vertex-gcp:gemini-3.1-pro-preview,vertex-gcp:gemini-3.8-flash -- uvx --quiet --from "${KYB_SOURCE:-git+https://github.com/broomva/open-kyb@main}" kyb-mcp
fi
echo "mcp    kyb registered (codex mcp list)"
echo "note   The witness uses gcloud application-default credentials (Vertex AI)."
echo "Open Codex in a project with a packet and ask: review the packet at <dir> with kyb-review."
echo "Dashboard: ask 'open the open-kyb dashboard with kyb-dashboard'. Reload the host after install."
echo "The dashboard skill requires a matching MCP build; unreleased UI code needs KYB_DEV_REPO."

echo "Codex: call codex_runtime, review/trust its managed hooks, then use codex_read_document/codex_predicates."
echo "Run its returned probe_command after trusting hooks (one explicit Codex model call); require ready=true."
