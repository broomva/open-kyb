"""What a shell command line runs and writes, read from its tokens (cross-review of PR #80, F1/F3).

The guard's older checks match the raw string, which quoting (`pol"icy"`), escapes (`gu\\ard`) and
a `cd` defeat. Here the command is tokenised as the shell would (`shlex`, operators split), each
simple command is unwrapped (`VAR=x`, `env`, `uv run`, `python -m`, `sh -c`, `eval`) and read
for two things:

- **forbidden invocations** from the agent's shell: the hook entry points (`kyb guard`,
  `kyb-hook`, `python -m kyb.hook`, `kyb audit append-hook`), which would let an agent mint its
  own approval, `kyb policy apply` (an agent applies only through MCP `apply_policy`; an officer
  in their own terminal), `kyb init --force` (a re-init installs a policy), and Python code that
  names those internals. Inside a workspace (`strict`), a command whose name is computed (`$K`,
  `$(…)`, backticks) or a shell reading commands from a pipe is refused too: what it runs
  cannot be read;
- **writes into protected paths**, with `cd` followed across the line: redirections, and the
  operands of the commands that write.

Still a best-effort reading, not an OS boundary (spec): a script file, or an interpreter other
than `sh -c`/`python -c`, is not followed, and same-user Python can forge any record.
"""

from __future__ import annotations

import glob
import re
import shlex
from pathlib import Path

_OPERATORS = set(";&|()")
_REDIRECT_CHARS = set("<>&|")
# wrappers that run the rest of the line, with the options of each that take a value (a wrapper's
# value must not be read as the command: `nice -n 5 kyb …`, `timeout -s KILL 9 kyb …`)
_WRAPPERS: dict[str, frozenset[str]] = {
    "env": frozenset({"-u", "--unset", "-C", "--chdir", "-P"}),
    "nice": frozenset({"-n", "--adjustment"}),
    "timeout": frozenset({"-s", "--signal", "-k", "--kill-after"}),
    "sudo": frozenset({"-u", "-g", "-C", "-h", "-p", "-U", "-r", "-t", "-D", "-R", "-T"}),
    "doas": frozenset({"-u", "-C"}),
    "stdbuf": frozenset({"-i", "-o", "-e", "--input", "--output", "--error"}),
    "xargs": frozenset({"-I", "-n", "-P", "-L", "-d", "-E", "-s", "-a", "-J", "-R", "-S"}),
    "exec": frozenset({"-a"}),
    "caffeinate": frozenset({"-t", "-w"}),
    "time": frozenset({"-f", "-o"}),
    "script": frozenset({"-t", "-T"}),
    "arch": frozenset(), "setsid": frozenset(), "chrt": frozenset(),
    "ionice": frozenset({"-c", "-n"}),
    "nohup": frozenset(), "command": frozenset(), "builtin": frozenset(),
}  # fmt: skip
# positional values before the command: timeout's duration, script's typescript file
_WRAPPER_OPERANDS = {"timeout": 1, "script": 1}
_SHELLS = {"sh", "bash", "zsh", "dash", "ksh", "fish"}
_UV_VALUE_OPTS = {"--with", "--python", "-p", "--project", "--directory", "--package",
                  "--from", "--env-file", "--group", "--extra"}  # fmt: skip
# commands that write, replace or remove every operand (archivers replace a file by its archive)
_WRITES_ALL = {"rm", "rmdir", "unlink", "touch", "mkdir", "chmod", "chown", "chgrp", "truncate",
               "tee", "shred", "mv", "patch", "gzip", "gunzip", "bzip2", "bunzip2", "xz", "unxz",
               "zstd", "lz4", "zip", "trash", "srm"}  # fmt: skip
_WRITES_LAST = {"cp", "install", "ln", "rsync"}
# module names, not paths: reading `src/kyb/policy/...` is fine, importing the internals is not
_INTERNALS = re.compile(
    r"\bkyb\.(?:hook|policy|audit|workspace\.(?:guard|init))\b"
    r"|\bfrom\s+kyb(?:\.workspace)?\s+import\b[^\n;]*\b(?:hook|guard|init|policy|audit)\b"
    r"|\bkyb-hook\b|policy_ask"
)
_FIND_EXEC = ("-exec", "-execdir", "-ok", "-okdir")
HOOK = "hook"
APPLY = "apply"
REINIT = "reinit"
INTERNALS = "internals"
COMPUTED = "computed"


def _tokens(command: str) -> list[str]:
    # a newline separates commands, as `;` does; shlex alone would treat it as a space. An escaped
    # parenthesis (`find … \( … \)`) is a word, but shlex's operator splitting ignores the escape.
    line = command.replace("\n", " ; ").replace("\\(", "\x01").replace("\\)", "\x02")
    lex = shlex.shlex(line, posix=True, punctuation_chars=True)
    lex.whitespace_split = True
    # kept as the words `\(` and `\)`: a bare parenthesis would read as an operator below
    return [t.replace("\x01", "\\(").replace("\x02", "\\)") for t in lex]


def simple_commands(command: str) -> list[tuple[list[str], list[str]]]:
    """`(argv, redirection targets)` for each simple command. Raises ValueError if unparseable."""
    tokens = _tokens(command)
    out: list[tuple[list[str], list[str]]] = []
    argv: list[str] = []
    redirs: list[str] = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        if t and set(t) <= _REDIRECT_CHARS and ("<" in t or ">" in t):
            if ">" in t and nxt:
                redirs.append(nxt)
            i += 2
            continue
        if t and set(t) <= _OPERATORS:
            if argv or redirs:
                out.append((argv, redirs))
            argv, redirs = [], []
        elif t.isdigit() and nxt and set(nxt) <= _REDIRECT_CHARS:
            pass  # a file descriptor number (`2>`), not an operand
        else:
            argv.append(t)
        i += 1
    if argv or redirs:
        out.append((argv, redirs))
    return out


def _unwrap(argv: list[str]) -> list[str]:
    """The argv that actually runs, past assignments, wrappers (and their values), `env -S` and
    `uv run`/`uvx`."""
    while argv:
        name = Path(argv[0]).name
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", argv[0]):
            argv = argv[1:]
        elif name in _WRAPPERS:
            takes, argv = _WRAPPERS[name], argv[1:]
            while argv and argv[0].startswith("-") and argv[0] != "-":
                if name == "env" and argv[0] in ("-S", "--split-string") and len(argv) > 1:
                    argv = shlex.split(argv[1]) + argv[2:]  # the string is the command line
                    break
                argv = argv[2:] if argv[0] in takes else argv[1:]
            else:
                argv = argv[_WRAPPER_OPERANDS.get(name, 0) :]
        elif name == "uv" and argv[1:2] in (["run"], ["tool"]) or name == "uvx":
            skip = 1 if name == "uvx" else 3 if argv[1:3] == ["tool", "run"] else 2
            argv = argv[skip:]
            while argv and argv[0].startswith("-"):
                argv = argv[2:] if argv[0] in _UV_VALUE_OPTS else argv[1:]
        else:
            return argv
    return argv


def _kyb_subcommand(args: list[str]) -> list[str]:
    """`kyb`'s arguments past its global `-w/--workspace` option."""
    while args and (args[0] in ("-w", "--workspace") or args[0].startswith("--workspace=")):
        args = args[1:] if "=" in args[0] else args[2:]
    return args


def _shell_script(argv: list[str]) -> str | None:
    """The command string of `sh -c …`, `bash -lc …` and the like, else None."""
    for i, a in enumerate(argv[1:-1], 1):
        if a.startswith("-") and not a.startswith("--") and "c" in a[1:]:
            return argv[i + 1]
    return None


def _invocation(argv: list[str]) -> tuple[str, str] | None:
    """`(kind, what)` when `argv` is a forbidden invocation, else None."""
    name = Path(argv[0]).name
    if name == "kyb-hook":
        return HOOK, "kyb-hook"
    if re.fullmatch(r"python3?(\.\d+)?", name):
        if "-c" in argv[1:-1] and _INTERNALS.search(argv[argv.index("-c") + 1]):
            return INTERNALS, "python -c"
        if "-m" in argv[1:-1]:
            module = argv[argv.index("-m") + 1]
            if module.startswith(("kyb.hook", "kyb.workspace.guard")):
                return HOOK, f"python -m {module}"
            if module in ("kyb", "kyb.cli"):
                return _invocation(["kyb", *argv[argv.index("-m") + 2 :]])
        return None
    if name != "kyb":
        return None
    sub = _kyb_subcommand(argv[1:])
    if sub[:1] == ["guard"] or sub[:2] == ["audit", "append-hook"]:
        return HOOK, "kyb " + " ".join(sub[:2])
    if sub[:2] == ["policy", "apply"]:
        return APPLY, "kyb policy apply"
    if sub[:1] == ["init"] and "--force" in sub:
        return REINIT, "kyb init --force"
    return None


def forbidden(command: str, *, strict: bool = True, depth: int = 0) -> tuple[str, str] | None:
    """The first forbidden invocation in `command`, following `sh -c`, `eval` and `find -exec`.
    `strict` (inside a workspace) also refuses what cannot be read: a computed command name, a
    shell fed from a pipe, and Python on a line that names kyb's internals (a heredoc). Raises
    ValueError if the line cannot be tokenised."""
    if depth > 3:
        return COMPUTED, "nested shells"
    commands = simple_commands(command)
    if strict and _INTERNALS.search(command) and any(
            re.fullmatch(r"python3?(\.\d+)?", Path(_unwrap(a)[0]).name)
            for a, _ in commands if _unwrap(a)):  # fmt: skip
        return INTERNALS, "python"
    for argv, _ in commands:
        if found := _check(_unwrap(argv), strict, depth):
            return found
    return None


def _check(argv: list[str], strict: bool, depth: int) -> tuple[str, str] | None:
    if not argv:
        return None
    name = Path(argv[0]).name
    if strict and any(c in argv[0] for c in "$`"):
        return COMPUTED, argv[0]
    if name in _SHELLS:
        script = _shell_script(argv)
        if script is not None:
            return forbidden(script, strict=strict, depth=depth + 1)
        if strict and not [a for a in argv[1:] if not a.startswith("-")]:
            return COMPUTED, f"{name} reading its commands from a pipe"  # `… | bash -s`
        return None
    if name == "eval":
        return forbidden(" ".join(argv[1:]), strict=strict, depth=depth + 1)
    if name == "find":
        for i, a in enumerate(argv):
            if a in _FIND_EXEC and (found := _check(_unwrap(argv[i + 1 :]), strict, depth + 1)):
                return found
        return None
    return _invocation(argv)


def _targets(argv: list[str]) -> list[str]:
    name = Path(argv[0]).name
    operands = [a for a in argv[1:] if not a.startswith("-")]
    if name in _WRITES_ALL:
        return operands
    if name in _WRITES_LAST:
        named = [argv[i + 1] for i, a in enumerate(argv[:-1]) if a in ("-t", "--target-directory")]
        return operands[-1:] + named
    if name in ("sed", "perl") and any(a.startswith(("-i", "--in-place")) for a in argv[1:]):
        return operands
    if name == "dd":
        return [a[3:] for a in argv[1:] if a.startswith("of=")]
    return []


def _find_targets(argv: list[str]) -> list[str]:
    """What `find` writes: under its start paths when it deletes, or runs a writer per match."""
    starts = []
    for a in argv[1:]:
        if a.startswith(("-", "(", "!", "\\")):
            break
        starts.append(a)
    starts = starts or ["."]
    if "-delete" in argv:
        return starts
    for i, a in enumerate(argv):
        if a in _FIND_EXEC:
            sub = _unwrap(argv[i + 1 :])
            if sub and Path(sub[0]).name in _WRITES_ALL | _WRITES_LAST | {"sed", "perl", "dd"}:
                return starts
    return []


def protected_write(command: str, root: Path, cwd: Path, protected: tuple[str, ...],
                    depth: int = 0) -> str | None:  # fmt: skip
    """The first workspace path under `protected` the command writes, following `cd`, or None."""
    root = root.resolve()
    here: Path | None = cwd
    for argv, redirs in simple_commands(command):
        argv = _unwrap(argv)
        name = Path(argv[0]).name if argv else ""
        if name in ("cd", "pushd"):
            dest = next((a for a in argv[1:] if not a.startswith("-")), "~")
            here = None if dest == "-" or here is None else (here / Path(dest).expanduser())
            continue
        if name in _SHELLS and (script := _shell_script(argv)) is not None and depth < 3:
            inner = protected_write(script, root, here or cwd, protected, depth + 1)
            if inner:
                return inner
        writes = _find_targets(argv) if name == "find" else _targets(argv) if argv else []
        for target in [*redirs, *writes]:
            path = Path(target).expanduser()
            if not path.is_absolute():
                if here is None:
                    return f"{target} (after a `cd` whose destination cannot be read)"
                path = here / path
            # a glob names whatever it matches (`rm template.l*`), and its own literal form
            for match in [path, *map(Path, glob.glob(str(path)))]:
                if root.is_relative_to(match.resolve()) and match.resolve() != root:
                    return f"{target} (holds the workspace)"  # moving or removing an ancestor
                try:
                    rel = match.resolve().relative_to(root)
                except ValueError:
                    continue
                if not rel.parts or rel.parts[0] in protected:
                    return rel.as_posix() or "."
    return None
