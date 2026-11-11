"""Exercise skill installation with a stub CLI; never touch shared Codex registration."""

import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("development", [True, False])
def test_installer_copies_ui_skill_and_keeps_server_source_explicit(tmp_path, development):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    codex = bin_dir / "codex"
    codex.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$KYB_TEST_CODEX_LOG"\n')
    codex.chmod(0o755)
    uv = bin_dir / "uv"
    uv.write_text("#!/bin/sh\nexit 0\n")
    uv.chmod(0o755)
    skills = tmp_path / "skills"
    calls = tmp_path / "codex-calls"
    env = {
        **os.environ,
        "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
        "KYB_SKILLS_DIR": str(skills),
        "KYB_TEST_CODEX_LOG": str(calls),
    }
    for name in ("KYB_DEV_REPO", "KYB_SOURCE"):
        env.pop(name, None)
    if development:
        env["KYB_DEV_REPO"] = str(REPO)
    else:
        env["KYB_SOURCE"] = "git+https://example.invalid/kyb@matching-build"
    result = subprocess.run(
        ["sh", str(REPO / "scripts/install-codex.sh")],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    for name in ("kyb-dashboard", "kyb-review"):
        source = (REPO / "plugin/skills" / name / "SKILL.md").read_text()
        assert (skills / name / "SKILL.md").read_text() == source
        assert (REPO / ".claude/skills" / name / "SKILL.md").read_text() == source
    assert "kyb-dashboard" in (skills / "kyb-review/SKILL.md").read_text()
    assert (skills / "kyb-review/reader.md").is_file()
    assert (skills / "kyb-review/predicates.md").is_file()
    commands = calls.read_text().splitlines()
    assert commands[0] == "mcp remove kyb"
    prefix = (
        "mcp add kyb --env KYB_WITNESS_STRATUM=A --env "
        "KYB_WITNESS_A_BACKENDS=vertex-gcp:gemini-3.1-pro-preview,"
        "vertex-gcp:gemini-3.8-flash -- "
    )
    if development:
        assert commands[1] == prefix + f"uv run --quiet --project {REPO} kyb-mcp"
    else:
        assert commands[1] == (
            prefix + "uvx --quiet --from git+https://example.invalid/kyb@matching-build kyb-mcp"
        )
    assert "matching MCP build" in result.stdout
    assert "Reload the host" in result.stdout
