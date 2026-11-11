"""Harness hooks (spec): `kyb guard pre`, `kyb guard stop`, `kyb audit append-hook`.

These are a convenience layer for Claude Code-style harnesses. The same rules run inside every `kyb`
action, so a harness without hooks gains no extra rights. Locally this is trusted development
(spec): the Bash check is a best-effort pattern match, not an OS boundary.
"""

from __future__ import annotations

import contextvars
import functools
import json
import re
import sys
from pathlib import Path
from typing import Any

# Lazy (PEP 810): every agent tool call runs a hook, and most never reach these (spec, spec).
lazy import shlex
lazy import subprocess

lazy from kyb.audit.chain import AuditCorruption, AuditLog, verify_workspace, workspace_action
lazy from kyb.util.fsutil import action_lock
lazy from kyb.util.hashing import sha256_json
lazy from kyb.workspace import shell

PROTECTED = ("bronze", "silver", "gold", "ledger", "kg", "ontology", "policy", "catalog.json",
             "template.lock", ".claude", ".mcp.json", "CLAUDE.md")  # fmt: skip
# PDF tools the agent may run: their *output* operand must land in an agent-writable path.
_PDF_OUTPUT_TOOLS = {"pdftotext", "pdftoppm", "pdfimages", "pdfseparate", "pdftocairo"}
_AGENT_WRITABLE = re.compile(r"^cases/[^/]+/(scratch|notes|proposals)/")
_MUTATORS = r"(>|>>|\btee\b|\bmv\b|\bcp\b|\brm\b|\bsed\s+-i|\btouch\b|\bmkdir\b|\bchmod\b|\bln\b)"
_NETWORK = re.compile(r"\b(curl|wget|nc|ncat|ssh|scp|rsync|ftp)\b")


def find_root(start: Path) -> Path | None:
    """The kyb workspace for a session, or None when none (or more than one) resolves."""
    return discover(start)[0]


def discover(start: Path) -> tuple[Path | None, str]:
    """`KYB_WORKSPACE`, else the nearest enclosing workspace, else the single workspace under
    `workspaces/` of the nearest enclosing directory that has one, never above
    `KYB_DISCOVERY_ROOT` when it is set. Returns the root, or None and why: `none` or
    `ambiguous` (two workspaces in reach: set KYB_WORKSPACE)."""
    import os

    env = os.environ.get("KYB_WORKSPACE")
    if env:
        p = Path(env).expanduser().resolve()
        return (
            (p, "")
            if (p / "template.lock").exists()
            else (None, f"KYB_WORKSPACE={env} is not a workspace")
        )
    # KYB_DISCOVERY_ROOT bounds the search: nothing above it is considered (a harness run in a
    # fresh project must never pick up a workspace from an enclosing directory)
    bound = os.environ.get("KYB_DISCOVERY_ROOT")
    top = Path(bound).expanduser().resolve() if bound else None
    dirs = [p for p in (start, *start.parents) if top is None or p.is_relative_to(top)]
    for p in dirs:
        if (p / "template.lock").exists():
            return p, ""
    for p in dirs:
        found = sorted(w.parent for w in p.glob("workspaces/*/template.lock"))
        if len(found) == 1:
            return found[0], ""
        if len(found) > 1:
            return (
                None,
                f"ambiguous: {len(found)} workspaces under {p / 'workspaces'}; set KYB_WORKSPACE",
            )
    return None, "none"


def session_start_dir(event: dict[str, Any]) -> Path:
    """Where a hook looks for the workspace: the session's project directory (what the MCP
    server starts in), else the tool's cwd, which a `cd` can move."""
    import os

    return Path(os.environ.get("CLAUDE_PROJECT_DIR") or event.get("cwd") or ".").resolve()


def check_write(root: Path, target: str | Path, cwd: Path | None = None) -> str | None:
    """Return a denial reason, or None if the agent may write `target`.

    Relative paths resolve against the tool's working directory (`cwd`), as the tool would.
    """
    path = Path(target)
    path = ((cwd or root) / path) if not path.is_absolute() else path
    try:
        rel = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return f"{target} is outside the workspace"
    if _AGENT_WRITABLE.match(rel):
        return None
    return f"{rel}: agents may write only cases/<case>/{{scratch,notes,proposals}}/ (spec)"


# Fallback when a tool's own `-h` cannot be read: options known to take a value.
_POPPLER_VALUE_OPTS = {
    "-f", "-l", "-r", "-rx", "-ry", "-x", "-y", "-W", "-H", "-sz", "-scale-to", "-scale-to-x",
    "-scale-to-y", "-enc", "-eol", "-fixed", "-opw", "-upw", "-jpegopt", "-tiffcompression",
    "-thinlinemode", "-aa", "-aaVector", "-freetype", "-paper", "-icc", "-antialias",
    "-colspacing",
}  # fmt: skip
_OPT_LINE = re.compile(r"^\s+(-{1,2}[\w?-]+)\s*(<[^>]*>)?\s*:")
# Only pdftotext treats `-` as stdout; for the others `-` is a real output path or prefix.
_STDOUT_DASH = {"pdftotext"}


@functools.cache
def _poppler_options(tool: str) -> dict[str, bool] | None:
    """{option: takes_a_value}, read from the installed tool's own usage text."""
    try:
        p = subprocess.run([tool, "-h"], capture_output=True, text=True, timeout=5)
    except OSError, subprocess.TimeoutExpired:
        return None
    opts = {}
    for line in (p.stdout + p.stderr).splitlines():
        if m := _OPT_LINE.match(line):
            opts[m[1]] = m[2] is not None
    return opts or None


def _pdf_output_operand(argv: list[str]) -> tuple[str | None, str | None]:
    """(output path, refusal) for a supported PDF tool invocation.

    Options are parsed from the tool's own usage text, so a value is never taken for an operand;
    an option the tool does not list is refused rather than guessed.
    """
    tool = Path(argv[0]).name
    known = _poppler_options(tool)
    positional: list[str] = []
    args = iter(argv[1:])
    for a in args:
        if a == "--":
            positional.extend(args)
        elif a.startswith("-") and a != "-":
            if known is None:
                takes_value = a in _POPPLER_VALUE_OPTS
            elif a in known:
                takes_value = known[a]
            else:
                return None, f"{tool}: unsupported option {a}"
            if takes_value:
                next(args, None)
        else:
            positional.append(a)
    if len(positional) >= 2:
        out = positional[1] if tool == "pdftotext" else positional[-1]
        return (None if out == "-" and tool in _STDOUT_DASH else out), None
    if len(positional) == 1 and tool == "pdftotext":  # writes <input>.txt beside the input
        return str(Path(positional[0]).with_suffix(".txt")), None
    if len(positional) == 1 and tool == "pdftocairo":  # writes <stem>… in the working directory
        return Path(positional[0]).stem, None
    return None, None


def check_bash(command: str, root: Path | None = None, cwd: Path | None = None) -> str | None:
    if _NETWORK.search(command):
        return "network tools are not allowed; egress is kyb actions and the model provider only"
    if root is not None:
        try:
            segments = [shlex.split(seg) for seg in re.split(r"[;&|]+", command) if seg.strip()]
        except ValueError:
            return "could not parse the command; quote it plainly"
        for argv in segments:
            if argv and Path(argv[0]).name in _PDF_OUTPUT_TOOLS:
                out, refusal = _pdf_output_operand(argv)
                if refusal:
                    return refusal
                if out is not None and (reason := check_write(root, out, cwd)):
                    return f"{argv[0]} output: {reason}"
    for name in PROTECTED:
        if re.search(
            _MUTATORS + r"[^;&|]*(?:^|[\s'\"=/>])" + re.escape(name) + r"(?=$|[\s/'\";&|])", command
        ):
            return f"the command appears to modify {name}; use a kyb action instead"
    if root is not None:  # the same, read from the tokens: quoting and `cd` do not hide a write
        try:
            written = shell.protected_write(command, root, cwd or root, PROTECTED)
        except ValueError:
            return "could not parse the command; quote it plainly"
        if written:
            return f"the command writes {written}; use a kyb action instead"
    return None


def check_invocations(command: str, *, strict: bool = True) -> str | None:
    """A denial for what an agent's shell may never run, workspace or not (PR #80 review, F1).
    Outside a workspace (`strict=False`) a line that cannot be tokenised is let through: there is
    nothing of kyb's there to protect, and a heredoc must not be refused for an apostrophe."""
    try:
        found = shell.forbidden(command, strict=strict)
    except ValueError:
        return "could not parse the command; quote it plainly" if strict else None
    if found is None:
        return None
    kind, what = found
    why = {
        shell.HOOK: f"`{what}` is a kyb hook entry point: the hooks run only from the harness, "
        "never from the agent's shell",
        shell.APPLY: "policy is applied only through the MCP apply_policy, which the harness asks "
        "the human for, or by an officer in their own terminal; never from the agent's shell",
        shell.REINIT: "`kyb init --force` re-installs the workspace's policy: an officer does that "
        "in their own terminal, never the agent's shell",
        shell.COMPUTED: f"`{what}`: what this command runs cannot be read from it; run kyb "
        "commands literally",
    }
    return why.get(kind, f"`{what}` reaches kyb's hook or policy-change internals; use the kyb "
                         "actions (propose_policy, apply_policy)")  # fmt: skip


# Submit tools whose calls are attested (spec). MCP names arrive prefixed (mcp__<server>__<tool>).
ATTESTED_TOOLS = ("read_document", "propose_predicates")
READER_TOOLS = {"Read", "Glob", "read_plan", "render_crop", "read_document"}
PREDICATES_TOOLS = {"predicate_work_order", "propose_predicates"}


def _base_tool(name: str) -> str:
    return name.rsplit("__", 1)[-1] if name.startswith("mcp__") else name


# Which registration delivered the running hook: "plugin" (the kyb plugin's hooks.json) or
# "workspace" (the hooks `kyb init` writes into a workspace's .claude/settings.json). Both can be
# loaded in one session; attestation then fails safe (harness.attest), instead of guessing which
# deliveries were duplicates.
HOOK_SOURCE: contextvars.ContextVar[str] = contextvars.ContextVar("kyb_hook_source", default="")


def _harness_event(root: Path, action: str, detail: dict[str, Any], tin: dict,
                   duplicate=None) -> None:  # fmt: skip
    """Append a harness event under the action lock, tagged with the hook source. `duplicate`
    skips only an exact re-delivery of the same call (same session and tool_use_id); lifecycle
    events (SubagentStart/Stop) are always recorded, because a count cannot tell a duplicate from
    a resume (PR #13 review round 3)."""
    from kyb.audit.chain import chain_events

    if HOOK_SOURCE.get():
        detail = {**detail, "hook_source": HOOK_SOURCE.get()}
    with workspace_action(root):
        if duplicate is not None and duplicate(chain_events(root)):
            return
        AuditLog(root / "ledger" / "audit.jsonl").append_record(
            "harness",
            action,
            input_hash=sha256_json(tin),
            output_hash=sha256_json(detail),
            detail=detail,
        )


def _reader_scope(root: Path, bind: dict) -> list[Path]:
    from kyb.read.plan import layout_rel, pages_rel  # local: guard stays importable without read

    c, r, d = bind["case"], bind["run"], bind["doc"]
    return [(root / pages_rel(c, r, d)).resolve(), (root / layout_rel(c, r, d)).resolve(),
            (root / f"silver/crops/{c}/{r}/{d}").resolve()]  # fmt: skip


def _in_scope(root: Path, event: dict, raw: str, scope: list[Path]) -> bool:
    if not raw:
        return False
    target = (Path(event.get("cwd") or root) / raw).resolve()
    return any(target == s or target.is_relative_to(s) for s in scope)


def _glob_root(pattern: str) -> str:
    """The directory an absolute glob pattern searches: its prefix before the first wildcard."""
    cut = min((i for i, ch in enumerate(pattern) if ch in "*?[{"), default=len(pattern))
    head = pattern[:cut]
    return head if head.endswith("/") else str(Path(head).parent)


def _confine(root: Path, event: dict[str, Any], base: str, tin: dict) -> str | None:
    """Reader and predicates subagents may use only their own tools, on their own document."""
    from kyb.audit.chain import chain_events
    from kyb.read.harness import PREDICATES_TYPES, READER_TYPES, bound_document

    agent_type, agent_id = event.get("agent_type") or "", event.get("agent_id") or ""
    session = event.get("session_id") or ""
    if agent_type in PREDICATES_TYPES:
        if base not in PREDICATES_TOOLS:
            return f"{agent_type} may use only {sorted(PREDICATES_TOOLS)}"
        return _bind_predicates(root, agent_id, session, tin, base)
    if agent_type not in READER_TYPES:
        return None
    if base not in READER_TOOLS:
        return f"{agent_type} may use only {sorted(READER_TOOLS)}"
    events = chain_events(root)
    bind = bound_document(events, session, agent_id) if agent_id else None
    if base == "read_plan":
        want = {k: tin.get(k) for k in ("case", "run", "doc")}
        if not all(want.values()) or not agent_id:
            return "a reader must call read_plan with its case, run and doc"
        return _bind(root, agent_id, session, want, tin)
    if bind is None:
        return "a reader must call read_plan for its document first"
    if base == "render_crop" and any(
            str(tin.get(k)) != str(bind[k]) for k in ("case", "run", "doc")):  # fmt: skip
        return f"{base} must target this reader's document ({bind['doc']})"
    scope = _reader_scope(root, bind)
    if base == "Read" and not _in_scope(root, event, tin.get("file_path") or "", scope):
        return f"{tin.get('file_path')!r} is outside this reader's document ({bind['doc']})"
    if base == "Glob":
        pattern, path = tin.get("pattern") or "", tin.get("path") or ""
        if ".." in pattern.split("/"):
            return "glob patterns may not contain '..'"
        if pattern.startswith("/") and not _in_scope(root, event, _glob_root(pattern), scope):
            return f"glob pattern {pattern!r} searches outside this reader's document"
        if not pattern.startswith("/") and not _in_scope(root, event, path, scope):
            return f"glob path {path!r} is outside this reader's document ({bind['doc']})"
    return None


def _bind_predicates(root: Path, agent_id: str, session: str, tin: dict, tool: str) -> str | None:
    from kyb.audit.chain import chain_events
    from kyb.read.harness import PREDICATES_TYPES, bound_predicates, by_action

    want = tin.get("work_order_id")
    if not want or not agent_id:
        return "predicates agent must name its work_order_id"
    with workspace_action(root):
        events = chain_events(root)
        bind = bound_predicates(events, session, agent_id)
        if bind:
            return None if bind.get("work_order") == want else "agent bound to another work order"
        if tool != "predicate_work_order":
            return "call predicate_work_order first"
        taken = {d.get("launch_id") for d in by_action(events, "harness.predicate_bind")}
        launch = next(
            (
                d
                for d in by_action(events, "harness.launch")
                if d.get("session") == session
                and d.get("work_order") == want
                and d.get("subagent_type") in PREDICATES_TYPES
                and d.get("launch_id") not in taken
            ),
            None,
        )
        if launch is None:
            return "no free canonical predicates launch"
        detail = {
            "agent_id": agent_id,
            "session": session,
            "launch_id": launch["launch_id"],
            "work_order": want,
        }
        if HOOK_SOURCE.get():
            detail["hook_source"] = HOOK_SOURCE.get()
        AuditLog(root / "ledger/audit.jsonl").append_record(
            "harness",
            "harness.predicate_bind",
            input_hash=sha256_json(tin),
            output_hash=sha256_json(detail),
            detail=detail,
        )
    return None


def _bind(root: Path, agent_id: str, session: str, want: dict, tin: dict) -> str | None:
    """Bind a reader to a free canonical launch: re-read, check and append under one lock."""
    from kyb.audit.chain import chain_events
    from kyb.read.harness import bound_document, free_launch

    with workspace_action(root):
        events = chain_events(root)
        bind = bound_document(events, session, agent_id)
        if bind is not None:
            same = {k: bind.get(k) for k in want} == want
            return None if same else f"this reader is bound to {bind['doc']}; not {want['doc']}"
        launch = free_launch(events, session, want)
        if launch is None:
            return f"no canonical kyb-reader launch for {want} in this session is waiting to bind"
        detail = {"agent_id": agent_id, "session": session, "launch_id": launch["launch_id"],
                  **want}  # fmt: skip
        if HOOK_SOURCE.get():
            detail["hook_source"] = HOOK_SOURCE.get()
        AuditLog(root / "ledger" / "audit.jsonl").append_record(
            "harness",
            "harness.reader_bind",
            input_hash=sha256_json(tin),
            output_hash=sha256_json(detail),
            detail=detail,
        )
    return None


def _launch(root: Path, event: dict[str, Any], tin: dict) -> str | None:
    """A kyb-reader or kyb-predicates launch must use the canonical prompt; it is recorded."""
    from kyb.read.harness import PREDICATES_PROMPT, PREDICATES_TYPES, READER_PROMPT, READER_TYPES

    kind = tin.get("subagent_type") or ""
    if kind not in READER_TYPES | PREDICATES_TYPES:
        return None
    prompt = (tin.get("prompt") or "").strip()
    rx = READER_PROMPT if kind in READER_TYPES else PREDICATES_PROMPT
    m = rx.fullmatch(prompt)
    if m is None or not event.get("tool_use_id"):
        return f"{kind} must be launched with its canonical prompt only (got {prompt[:80]!r})"
    if tin.get("name"):
        return f"{kind} may not be launched with a name: kyb agents are never messaged"
    detail = {"session": event.get("session_id") or "", "launch_id": event["tool_use_id"],
              "subagent_type": kind, "prompt": prompt}  # fmt: skip
    if kind in READER_TYPES:
        detail.update(case=m[1], run=m[2], doc=m[3])
    else:
        detail.update(work_order=m[1])
    from kyb.read.harness import by_action

    source = HOOK_SOURCE.get() or None
    _harness_event(root, "harness.launch", detail, tin, duplicate=lambda ev: any(
        d.get("launch_id") == detail["launch_id"] and d.get("session") == detail["session"]
        and d.get("hook_source") == source
        for d in by_action(ev, "harness.launch")))  # fmt: skip
    return None


def pre_tool_use(event: dict[str, Any]) -> tuple[int, str] | tuple[int, str, dict]:
    """Guards every tool call. For submit tools, mints the one-time attestation token and returns
    it as `updatedInput`, so the action can tell which hook call it is executing."""
    import secrets

    from kyb.read.harness import (
        PREDICATES_TYPES,
        READER_TYPES,
        TOKEN_FIELD,
        split_token,
        token_hash,
    )

    tool = event.get("tool_name", "")
    base = _base_tool(tool)
    tin = event.get("tool_input") or {}
    root, why = discover(session_start_dir(event))
    command = str(tin.get("command") or "")
    if tool == "Bash" and (refusal := check_invocations(command, strict=root is not None)):
        return 2, refusal
    if root is None:
        kyb = READER_TYPES | PREDICATES_TYPES
        if (event.get("agent_type") or "") in kyb or (
                tool in ("Agent", "Task") and (tin.get("subagent_type") or "") in kyb):  # fmt: skip
            return 2, f"no kyb workspace resolves for this session ({why}); kyb agents fail closed"
        return 0, ""
    cwd = Path(event.get("cwd") or ".").resolve()
    reason = None
    if tool in ("Write", "Edit", "MultiEdit", "NotebookEdit"):
        reason = check_write(root, tin.get("file_path") or tin.get("notebook_path") or "", cwd)
    elif tool == "Bash":
        reason = check_bash(tin.get("command", ""), root, cwd)
    elif tool == "SendMessage":
        reason = _deny_message_to_kyb_agents(root, tin)
    elif tool in ("Agent", "Task"):
        reason = _launch(root, event, tin)
    reason = reason or _confine(root, event, base, tin)
    if not reason:
        # signing (spec) and policy changes need the human's own approval
        from kyb.signing.gate import ASK, check, policy_input

        if verdict := check(base, tin, event):
            if verdict[0] == ASK:  # the harness's own permission prompt: the human's approval
                out = {"hookEventName": "PreToolUse", "permissionDecision": "ask",
                       "permissionDecisionReason": verdict[1]}  # fmt: skip
                if (wanted := policy_input(base, tin)) is not None:
                    lines, problem = _proposal_lines(root, wanted)
                    if problem:  # nothing to approve: never put the human in front of it
                        return 2, problem
                    out["permissionDecisionReason"] += lines
                    out["updatedInput"] = _policy_confirmation(root, event, base, tin, wanted)
                return 0, "", {"hookSpecificOutput": out}
            reason = verdict[1]
    if reason:
        return 2, reason
    if base in ATTESTED_TOOLS and event.get("tool_use_id"):
        clean, _ = split_token(tin)
        token = secrets.token_hex(16)
        _harness_event(root, "harness.attestation", {
            "tool": base, "tool_use_id": event["tool_use_id"], "input_hash": sha256_json(clean),
            "token_hash": token_hash(token), "agent_id": event.get("agent_id") or "",
            "agent_type": event.get("agent_type") or "", "session": event.get("session_id") or "",
        }, clean)  # fmt: skip
        out = {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "allow",
                                      "updatedInput": {**clean, TOKEN_FIELD: token}}}  # fmt: skip
        return 0, "", out
    return 0, ""


def _proposal_lines(root: Path, wanted: dict) -> tuple[str, str | None]:
    """What the human approves, in the harness's own prompt: the recorded parameter and rule
    diff of the receipted proposal, for exactly the hash being approved (PR #80 review, F2/F6).
    Returns (lines, None), or ("", why there is nothing to approve)."""
    from kyb.policy.change import load_proposal, summary

    try:
        record = load_proposal(root, wanted["proposal_id"])
        shown = summary(record)
    except (OSError, ValueError, LookupError, PermissionError, KeyError) as exc:
        return "", f"apply_policy: proposal {wanted['proposal_id']} cannot be read: {exc}"
    if record["current_hash"] != wanted["expected_current_hash"]:
        return "", (f"apply_policy: stale: proposal {wanted['proposal_id']} was made against "
                    f"{record['current_hash']}, not {wanted['expected_current_hash']}")  # fmt: skip
    changes = "; ".join([*shown["parameter_diff"], *shown["rule_diff"]])
    rules = ", ".join(shown["changed_rules"]) or "none"
    changes = changes or "no parameter, severity or points change"
    return f": {changes}; rule text changed: {rules}", None


def _policy_confirmation(root: Path, event: dict, base: str, tin: dict, wanted: dict) -> dict:
    """The asked call's input plus a one-time token. The harness delivers it only if the human
    approves the prompt; the chain keeps only its hash, bound to exactly this input."""
    import secrets

    from kyb.policy.change import ASK_EVENT, CONFIRMATION_FIELD, confirmation_hash

    token = secrets.token_hex(16)
    detail = {"tool": base, "input_hash": sha256_json(wanted),
              "token_hash": confirmation_hash(token), "tool_use_id": event["tool_use_id"],
              "session": event.get("session_id") or ""}  # fmt: skip
    _harness_event(root, ASK_EVENT, detail, tin)
    return {**tin, CONFIRMATION_FIELD: token}  # replaces any value already there


def _deny_message_to_kyb_agents(root: Path, tin: dict) -> str | None:
    """Readers and the predicates agent are never resumed or messaged after launch (spec).

    The structured recipient is checked against every kyb agent id and kyb launch name; a message
    whose recipient cannot be read is refused while any kyb agent exists.
    """
    from kyb.audit.chain import chain_events
    from kyb.read.harness import PREDICATES_TYPES, READER_TYPES

    kinds = READER_TYPES | PREDICATES_TYPES
    events = chain_events(root)
    protected = {e["detail"].get("agent_id") for e in events
                 if e.get("action") == "harness.subagent_start"
                 and e["detail"].get("agent_type") in kinds}  # fmt: skip
    protected |= {e["detail"].get("name") for e in events if e.get("action") == "harness.launch"}
    protected -= {None, ""}
    if not protected:
        return None
    to = tin.get("to") or tin.get("recipient") or tin.get("agent_id") or tin.get("name")
    if not isinstance(to, str) or not _route_key(to):
        return "messages are refused while kyb agents exist unless their recipient is explicit"
    keys = {_route_key(p) for p in protected}
    blob = _route_key(json.dumps(tin, ensure_ascii=False))
    # compare the way the harness routes: normalised, and never only by exact spelling
    if _route_key(to) in keys or any(k and k in blob for k in keys):
        return f"messaging or resuming kyb agents is not allowed: {to!r}"
    return None


def _route_key(name: str) -> str:
    """A recipient reduced to its letters and digits (after NFKC and casefolding).

    Stricter than the harness's own routing normalisation, whatever separators it rewrites
    (spaces, tabs, hyphens, control or format characters): any spelling that can resolve to a kyb
    agent reduces to the same key as that agent's id.
    """
    import unicodedata

    t = unicodedata.normalize("NFKC", name).casefold()
    return "".join(ch for ch in t if ch.isalnum())


def session_start(event: dict[str, Any]) -> tuple[int, str]:
    """Record the harness-reported model (Claude Code's SessionStart may carry `model`)."""
    root = find_root(session_start_dir(event))
    if root is None:
        return 0, ""
    detail = {"session": event.get("session_id", ""), "model": event.get("model") or "",
              "source": event.get("source", "")}  # fmt: skip
    _harness_event(root, "harness.session_start", detail, detail)
    return 0, ""


def subagent_start(event: dict[str, Any]) -> tuple[int, str]:
    root = find_root(session_start_dir(event))
    if root is None:
        return 0, ""
    detail = {"agent_id": event.get("agent_id") or "", "agent_type": event.get("agent_type") or "",
              "session": event.get("session_id", "")}  # fmt: skip
    try:
        # every start is recorded: a second one is a resume, or a second registration, and
        # attestation treats both as unattested
        _harness_event(root, "harness.subagent_start", detail, detail)
    except AuditCorruption as exc:
        return 2, str(exc)
    return 0, ""


def _transcript_models(path: str) -> tuple[list[str], bool]:
    """The models the harness recorded for a subagent's own turns, read from the transcript the
    harness wrote (not from anything the model said), and whether that reading is complete: a
    missing or unreadable file, an unparsable line, or no assistant turn makes it incomplete,
    and an incomplete snapshot leaves the reader's model unknown (harness.stopped_model)."""
    p = Path(path) if path else None
    if p is None or p.suffix != ".jsonl" or not p.is_file():
        return [], False
    models: set[str] = set()
    complete, turns = True, 0
    try:
        with p.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    complete = False
                    continue
                if not isinstance(rec, dict) or rec.get("type") != "assistant":
                    continue
                turns += 1
                model = (rec.get("message") or {}).get("model")
                if not isinstance(model, str) or not model:
                    complete = False  # a turn whose model the transcript does not name
                elif not model.startswith("<"):  # "<synthetic>": harness-inserted, not a model
                    models.add(model)
    except OSError, UnicodeDecodeError:
        return [], False
    return sorted(models), complete and turns > 0 and bool(models)


def subagent_stop(event: dict[str, Any]) -> tuple[int, str]:
    root = find_root(session_start_dir(event))
    if root is None:
        return 0, ""
    last = sha256_json(event.get("last_assistant_message") or "")
    transcript = event.get("agent_transcript_path") or ""
    detail = {"agent_id": event.get("agent_id") or "", "agent_type": event.get("agent_type") or "",
              "session": event.get("session_id") or "", "agent_transcript_path": transcript,
              "last_message_sha256": last}  # fmt: skip
    detail["models"], detail["complete"] = _transcript_models(transcript)
    from kyb.predicates.lifecycle import transcript_summary

    detail["lifecycle"] = transcript_summary(transcript)
    try:
        # every stop snapshot is recorded (a stop can be vetoed and the agent go on): the
        # reader's model is the union over all of them (harness.stopped_model)
        _harness_event(root, "harness.subagent_stop", detail, detail)
        from kyb.audit.chain import chain_events

        with workspace_action(root):
            events = chain_events(root)
            for turn in detail["lifecycle"]["turns"]:
                usage = turn["usage"]
                usage_id = sha256_json(
                    {"session": detail["session"], "agent_id": detail["agent_id"], "turn": turn}
                )
                if any(e["action"] == "model.call" and e["input_hash"] == usage_id for e in events):
                    continue
                AuditLog(root / "ledger/audit.jsonl").append_record(
                    "harness",
                    "model.call",
                    input_hash=usage_id,
                    output_hash=sha256_json(usage),
                    tokens_in=usage["input_tokens"],
                    tokens_out=usage["output_tokens"],
                    detail={
                        "session": detail["session"],
                        "agent_id": detail["agent_id"],
                        "model": turn["model"],
                        "cost_basis": "transcript_usage",
                        "cost_reported": False,
                        "cache_read_input_tokens": usage.get("cache_read_input_tokens", 0),
                        "cache_creation_input_tokens": usage.get("cache_creation_input_tokens", 0),
                    },
                )
    except AuditCorruption as exc:
        return 2, str(exc)
    return 0, ""


def stop(event: dict[str, Any]) -> tuple[int, str]:
    root = find_root(session_start_dir(event))
    if root is None:
        return 0, ""
    with action_lock(root):
        result = verify_workspace(root, at_boundary=True)
    if not result.ok:
        return 2, f"audit chain does not verify: {result.error}"
    return ingest_session(event)


def ingest_session(event: dict[str, Any]) -> tuple[int, str]:
    """The display check (Stop and SessionEnd): capture what was shown and flag unverified
    answers. It flags; it never blocks the harness, and a failure to read is reported, not fatal."""
    if not event.get("transcript_path"):
        return 0, ""
    from kyb.copilot import display

    try:
        out = display.run(event)
    except Exception as exc:
        return 0, f"kyb display check could not read the transcript: {type(exc).__name__}: {exc}"
    flagged = (out or {}).get("unverified_answer", 0)
    return 0, f"kyb: {flagged} unverified answer(s) flagged this session" if flagged else ""


def append_hook(event: dict[str, Any]) -> tuple[int, str]:
    root = find_root(session_start_dir(event))
    if root is None:
        return 0, ""
    from kyb.audit.chain import chain_events

    detail = {"tool": event.get("tool_name", ""), "session": event.get("session_id", ""),
              "tool_use_id": event.get("tool_use_id") or ""}  # fmt: skip
    if HOOK_SOURCE.get():
        detail["hook_source"] = HOOK_SOURCE.get()
    try:
        with workspace_action(root):  # the hook event is an action: checked before, pinned after
            if detail["tool_use_id"] and any(
                    e["component"] == "harness" and e["action"] == "tool.call"
                    and e["detail"].get("tool_use_id") == detail["tool_use_id"]
                    and e["detail"].get("session") == detail["session"]
                    and e["detail"].get("hook_source") == detail.get("hook_source")
                    for e in chain_events(root)):  # fmt: skip
                # the same call re-delivered by the same registration; another registration's
                # delivery is recorded, because it is evidence of a second registration
                return 0, ""
            AuditLog(root / "ledger" / "audit.jsonl").append_record(
                "harness", "tool.call",
                input_hash=sha256_json(event.get("tool_input") or {}),
                output_hash=sha256_json(event.get("tool_response") or {}),
                detail=detail,
            )  # fmt: skip
    except AuditCorruption as exc:
        return 2, str(exc)
    return 0, ""


def run_hook(fn, source: str = "") -> int:
    """Run a hook. Unexpected failures block (exit 2): the hooks are a trust basis (spec).
    `source` names the registration that delivered it (`--source plugin|workspace`)."""
    try:
        event = json.load(sys.stdin)
    except json.JSONDecodeError:
        event = {}
    token = HOOK_SOURCE.set(source)  # for this hook only: an in-process caller runs others after
    try:
        result = fn(event)
    except Exception as exc:  # fail closed, never open
        print(f"kyb hook failed closed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    finally:
        HOOK_SOURCE.reset(token)
    code, msg = result[0], result[1]
    if len(result) > 2 and result[2]:
        print(json.dumps(result[2]))
    if msg:
        print(msg, file=sys.stderr)
    return code
