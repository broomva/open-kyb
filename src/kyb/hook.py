"""`kyb-hook`: the harness hooks' own entry point, the fast lane (spec).

Every agent tool call runs `kyb-hook pre` and `kyb-hook post`, so start-up is paid on each one.
This entry point parses its few arguments by hand (no Typer) and imports lazily (PEP 810): an
event pays only for the modules it reaches, and recording one appends with hashing and canonical
JSON only (`AuditLog.append_record`), never the pydantic model.

    kyb-hook <event> --source plugin|workspace      (the hook event JSON on stdin)

`--source` names the registration that delivered the hook: the plugin's `hooks.json` or the
hooks `kyb init` writes into a workspace. It is recorded with every event, so a session with
both registrations is detected (`harness.registration_problem`). A usage error exits 2, which a
PreToolUse hook treats as a block: a misconfigured hook fails closed.
"""

from __future__ import annotations

import sys

lazy from kyb.workspace import guard

# event -> the guard function that handles it
EVENTS = {
    "pre": "pre_tool_use",
    "post": "append_hook",
    "session-start": "session_start",
    "subagent-start": "subagent_start",
    "subagent-stop": "subagent_stop",
    "stop": "stop",
    "ingest-session": "ingest_session",
}
SOURCES = ("plugin", "workspace")
USAGE = f"usage: kyb-hook {{{','.join(EVENTS)}}} --source {{{'|'.join(SOURCES)}}}"


class UsageError(ValueError):
    pass


def parse(argv: list[str]) -> tuple[str, str]:
    """(event, source) from `<event> --source <s>` or `<event> --source=<s>`, in either order."""
    event = source = None
    args = iter(argv)
    for a in args:
        if a == "--source":
            value = next(args, None)
            if value is None:
                raise UsageError("--source needs a value")
            source = value
        elif a.startswith("--source="):
            source = a.removeprefix("--source=")
        elif a.startswith("-"):
            raise UsageError(f"unknown option {a}")
        elif event is None:
            event = a
        else:
            raise UsageError(f"unexpected argument {a}")
    if event not in EVENTS:
        raise UsageError(f"unknown hook event {event!r}" if event else "no hook event given")
    if source not in SOURCES:
        raise UsageError(f"--source must be one of {', '.join(SOURCES)} (got {source!r})")
    return event, source


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv in (["-h"], ["--help"]):
        print(USAGE)
        return 0
    try:
        event, source = parse(argv)
    except UsageError as exc:
        print(f"kyb-hook: {exc}\n{USAGE}", file=sys.stderr)
        return 2
    return guard.run_hook(getattr(guard, EVENTS[event]), source)


if __name__ == "__main__":
    raise SystemExit(main())
