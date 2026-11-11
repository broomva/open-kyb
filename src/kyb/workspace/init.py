"""`kyb init`: materialise a tenant workspace from the versioned template.

See spec, spec, spec and spec. The template is pinned by `template.lock`. The policy is copied
in as Compliance's file, and its hash is recorded whenever it is loaded. The watchlist sits beside
it in `policy/`, hashed and pinned the same way (spec), so `kyb screen` reads the list the
workspace was made with. Nothing about any applicant exists yet.
"""

from __future__ import annotations

import json
import shlex
import shutil
import sys
from importlib import resources
from pathlib import Path

from kyb import ontology
from kyb.audit.chain import AuditLog, workspace_action, write_checkpoint
from kyb.util.hashing import sha256_bytes, sha256_file

# 0.2.0: the ontology is TOML (spec). Older workspaces keep their pinned YAML (kyb.ontology).
TEMPLATE_VERSION = "0.2.0"
WATCHLIST = "watchlists.json"
LAYOUT = (
    "bronze/packets",
    "silver",
    "gold",
    "ledger",
    "kg",
    "cases",
    "ontology",
    "policy",
    ".claude",
)


def _template(name: str) -> bytes:
    return resources.files("kyb.workspace.template").joinpath(name).read_bytes()


def _cmd(event: str, *, block: bool = True) -> dict:
    """A command hook; `block` makes a hook failure block the action (fail closed, spec)."""
    h = {"type": "command", "command": _hook(event)}
    if block:
        h["onFailure"] = "block"
    return h


def _hook(event: str) -> str:
    # The interpreter that ran `kyb init`, so hooks work outside an activated venv; `kyb.hook` is
    # the hooks' fast lane (spec).
    return f"{shlex.quote(sys.executable)} -m kyb.hook {event} --source workspace"


def _settings() -> dict:
    # The same guards also run inside every action, so a harness without these hooks gains nothing.
    return {
        "permissions": {
            "allow": [
                "Bash(kyb *)", "Bash(pdftotext *)", "Bash(pdftoppm *)", "Bash(pdfinfo *)",
                "Read(**)", "Write(cases/**)", "Edit(cases/**)",
            ],
            "deny": [
                "Write(bronze/**)", "Write(silver/**)", "Write(gold/**)", "Write(ledger/**)",
                "Write(kg/**)", "Edit(bronze/**)", "Edit(silver/**)", "Edit(gold/**)",
                "Edit(ledger/**)", "Edit(kg/**)", "Write(policy/**)", "Edit(policy/**)",
                # the hook entry points mint approvals; policy applies via MCP or an officer
                "Bash(kyb guard *)", "Bash(kyb-hook *)", "Bash(kyb audit append-hook *)",
                "Bash(kyb policy apply *)", "Bash(kyb init *)", "WebFetch", "WebSearch",
            ],
        },
        "hooks": {
            # Every tool: write guards, reader/predicates confinement, launches, attestation.
            "PreToolUse": [{"matcher": "*", "hooks": [_cmd("pre")]}],
            "SessionStart": [{"hooks": [_cmd("session-start")]}],
            "SubagentStart": [{"hooks": [_cmd("subagent-start")]}],
            "SubagentStop": [{"hooks": [_cmd("subagent-stop", block=False)]}],
            "PostToolUse": [{"matcher": "*", "hooks": [_cmd("post")]}],
            "Stop": [{"hooks": [{"type": "command", "command": _hook("stop")}]}],
            # the display check's last chance to read what was shown (L5)
            "SessionEnd": [{"hooks": [{"type": "command", "command": _hook("ingest-session")}]}],
        },
    }  # fmt: skip


def init_workspace(
    root: Path,
    *,
    tenant: str,
    policy_src: Path,
    watchlist_src: Path | None = None,
    force: bool = False,
    officer: str | None = None,
    officer_token: str | None = None,
) -> dict:
    """`watchlist_src` defaults to the `watchlists.json` beside the policy (the sample assets
    ship them together). Without either, the workspace has no list and `kyb screen` asks for one.

    A forced re-init that would change the workspace's approved policy is an officer's act, as
    `kyb policy apply` is: `officer` and their token must authenticate (PR #80 review). A re-init
    that keeps the approved text needs no one."""
    root = root.resolve()
    if not Path(policy_src).is_file():  # before anything is created
        raise FileNotFoundError(f"policy {policy_src} not found")
    watchlist = _watchlist(Path(policy_src), watchlist_src)
    root.mkdir(parents=True, exist_ok=True)
    with workspace_action(root):  # the existence check happens inside the lock
        if (root / "template.lock").exists() and not force:
            raise FileExistsError(f"{root} is already a workspace (template.lock exists)")
        # a lock or chain removed, compressed or hidden by any means does not make a workspace
        # new: re-initialising one could install another policy (PR #80 review)
        existing = _holds_workspace(root)
        if existing and not force:
            raise FileExistsError(f"{root} already holds workspace data ({existing}): it is a "
                                  "workspace; re-initialising it needs --force")  # fmt: skip
        approver = (
            _policy_change_approver(root, Path(policy_src), officer, officer_token)
            if existing
            else None
        )
        return _init_locked(root, tenant=tenant, policy_src=policy_src, watchlist=watchlist,
                            approver=approver)  # fmt: skip


_DATA_FILES = ("ledger/audit.jsonl", "catalog.json")
_DATA_DIRS = ("ledger", "bronze", "silver", "gold", "kg", "cases")


def _holds_workspace(root: Path) -> str | None:
    """What makes `root` an existing workspace (a lock, a chain, a catalog, or any file in a
    layer), or None for a directory a first init may use. Only kyb's own lock file is ignored."""
    for rel in _DATA_FILES:
        if (root / rel).exists():
            return "ledger/audit.jsonl (an audit chain)" if rel.startswith("ledger") else rel
    for d in _DATA_DIRS:
        if (root / d).is_dir() and any(p.is_file() for p in (root / d).rglob("*")):
            return f"{d}/"
    return None


def _policy_change_approver(root: Path, policy_src: Path, officer: str | None,
                            token: str | None) -> str | None:  # fmt: skip
    """The officer approving a re-init of an existing workspace that changes its approved
    policy; None when the text stays the approved one. A workspace whose chain cannot be read
    (hidden, compressed) has no approved text, so only an officer can re-initialise it. No
    wrapper or shell form gets around this."""
    from kyb.policy.provenance import approved_hash

    if sha256_file(policy_src) == approved_hash(root):
        return None
    if officer is None:
        raise PermissionError(
            "re-initialising this workspace would replace its approved policy: that is an "
            "officer's act. Run `kyb init … --force --officer <id>` in the officer's own "
            "terminal, or change the policy with propose_policy and apply_policy."
        )
    from kyb.signing.officers import authenticate

    return authenticate(root, officer, token)


def _watchlist(policy_src: Path, given: Path | None) -> bytes | None:
    """The watchlist's bytes, checked before anything is written: a given path must exist and hold
    a JSON object; the default beside the policy is optional."""
    path = Path(given) if given is not None else policy_src.with_name(WATCHLIST)
    if not path.is_file():
        if given is not None:
            raise FileNotFoundError(f"watchlist {given} not found")
        return None
    raw = path.read_bytes()
    try:
        ok = isinstance(json.loads(raw), dict)
    except ValueError:
        ok = False
    if not ok:
        raise ValueError(f"watchlist {path} is not a JSON object")
    return raw


def _init_locked(root: Path, *, tenant: str, policy_src: Path, watchlist: bytes | None,
                 approver: str | None = None) -> dict:  # fmt: skip
    for d in LAYOUT:
        (root / d).mkdir(parents=True, exist_ok=True)

    onto = ontology.template_bytes()
    ontology.parse(onto)  # the schema is the gate: a template failing validation is never written
    (root / ontology.WORKSPACE_PATH).write_bytes(onto)
    shutil.copyfile(policy_src, root / "policy" / "compliance_rules.md")
    if watchlist is not None:
        (root / "policy" / WATCHLIST).write_bytes(watchlist)
    else:  # a forced re-init without a list must not leave the old one looking pinned
        (root / "policy" / WATCHLIST).unlink(missing_ok=True)
    claude_md = _template("CLAUDE.md").decode("utf-8")
    claude_md = claude_md.replace("{tenant}", tenant).replace(
        "{template_version}", TEMPLATE_VERSION
    )
    (root / "CLAUDE.md").write_text(claude_md, "utf-8")
    (root / ".claude" / "settings.json").write_text(
        json.dumps(_settings(), indent=2) + "\n", "utf-8"
    )
    mcp = {
        "mcpServers": {
            "kyb": {
                "command": "kyb-mcp",
                "args": ["--workspace", "."],
                "env": {"KYB_TENANT": tenant},
            }
        }
    }
    (root / ".mcp.json").write_text(json.dumps(mcp, indent=2) + "\n", "utf-8")
    if not (root / "catalog.json").exists():
        (root / "catalog.json").write_text("{}\n", "utf-8")

    lock = {
        "template": "kyb-template",
        "version": TEMPLATE_VERSION,
        "ontology": sha256_bytes(onto),
        "ontology_path": ontology.WORKSPACE_PATH,
        "tenant": tenant,
        "policy_at_init": sha256_file(root / "policy" / "compliance_rules.md"),
        "watchlist_at_init": sha256_bytes(watchlist) if watchlist is not None else None,
    }
    (root / "template.lock").write_text(json.dumps(lock, indent=2) + "\n", "utf-8")
    AuditLog(root / "ledger" / "audit.jsonl").append(
        "kyb",
        "workspace.init",
        input_hash=sha256_bytes(onto),
        output_hash=sha256_file(root / "template.lock"),
        detail={
            "template": f"kyb-template@{TEMPLATE_VERSION}",
            "tenant": tenant,
            **({"policy_change_approved_by": approver} if approver else {}),
        },
    )
    write_checkpoint(root)
    return lock
