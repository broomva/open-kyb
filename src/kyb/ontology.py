"""The ontology's schema (spec, spec): a pydantic model over `kyb.ontology.toml`.

The ontology is authored in TOML and read with the stdlib `tomllib`. This model is its schema: a
load fails on an unknown key, an unknown role, transport or tier, a write path outside the
declared layers, or a path written by both a proposer and a judge (proposer ⊥ judge, Kinetic's
simulator ⊥ verifier rule). `python -m kyb.ontology schema` prints the JSON Schema and
`python -m kyb.ontology json` the validated ontology as JSON, for the MCP/CLI generator.

Workspaces made before template 0.2.0 hold `ontology/kyb.ontology.yaml`, pinned by hash in their
`template.lock`. They stay as they are: kyb never parses YAML. A legacy file whose bytes are the
0.1.0 template's is the same ontology as the 0.1.0 TOML (same content, re-encoded), so it loads
as that; any other YAML is refused rather than guessed at.
"""

from __future__ import annotations

import fnmatch
import json
import sys
import tomllib
from importlib import resources
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from kyb.util.hashing import sha256_bytes

TEMPLATE = "kyb.ontology.toml"
WORKSPACE_PATH = "ontology/kyb.ontology.toml"
LEGACY_PATH = "ontology/kyb.ontology.yaml"
# sha256 of the YAML that templates up to 0.1.0 wrote; its content is TEMPLATE's, re-encoded
LEGACY_SHA256 = "sha256:324bfaf13ccb2a94c26b5e8168910cafd694d709ddd2834ac9f09c69c68646ae"

Role = Literal["pipeline", "officer", "agent", "auditor", "ci"]
Transport = Literal["cli", "mcp", "api"]
Tier = Literal["untrusted", "claimed", "verified", "corroborated", "derived", "judged", "decided"]
# where an action may write: the medallion layers and the ledger (spec); `**/current` is the
# kg projection's pointer files
LAYERS = ("bronze/", "silver/", "gold/", "ledger/", "kg/")
PROJECTION_PATHS = ("**/current",)
# the one configuration file an action may replace: apply_policy, behind the human's approval
POLICY_PATHS = ("policy/compliance_rules.md",)
PROPOSERS = frozenset({"propose", "propose_predicates"})
JUDGES = frozenset({"run_judges", "emit_decision"})


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ObjectType(_Strict):
    key: list[str]
    props: list[str] = []
    kinds: list[str] = []
    pii: bool = False


class LinkType(_Strict):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    from_: str | list[str] = Field(alias="from")
    to: str | list[str]
    props: list[str] = []


class Action(_Strict):
    invoker: list[Role]
    transport: list[Transport] = []
    writes: list[str] = Field(min_length=1)
    reads: list[str] = []
    model: Literal["witness"] | None = None
    claim_blind: bool = False
    reader: Literal["harness"] | None = None
    per: Literal["document"] | None = None
    tier_out: Tier | None = None
    checks: list[str] = []
    cap: Literal["MANUAL_REVIEW"] | None = None
    auth: Literal["officer_token"] | None = None
    deterministic: bool = False

    @model_validator(mode="after")
    def _writes_in_layers(self) -> Action:
        for path in self.writes:
            if "\\" in path or any(part in ("", ".", "..") for part in path.split("/")):
                raise ValueError(f"write path {path!r} must be a plain relative pattern")
            if not (path.startswith(LAYERS) or path in PROJECTION_PATHS + POLICY_PATHS):
                raise ValueError(f"write path {path!r} is outside the declared layers")
        return self


class ReadTool(_Strict):
    invoker: list[Role]
    transport: list[Transport]


class Registry(_Strict):
    """The action registry: what `workspace-materialization.md` §3 shows, and what the hooks, the
    sandbox mounts and the MCP/CLI/API surfaces derive from."""

    actions: dict[str, Action]
    agent_writable: list[str]
    reads: dict[str, ReadTool]

    @model_validator(mode="after")
    def _proposer_not_judge(self) -> Registry:
        proposed = {
            p for name in PROPOSERS & self.actions.keys() for p in self.actions[name].writes
        }
        judged = {p for name in JUDGES & self.actions.keys() for p in self.actions[name].writes}
        if both := sorted((p, j) for p in proposed for j in judged if may_overlap(p, j)):
            raise ValueError(f"written by both a proposer and a judge: {both}")
        if overlap := self.actions.keys() & self.reads.keys():
            raise ValueError(f"declared as both an action and a read: {sorted(overlap)}")
        return self


def may_overlap(a: str, b: str) -> bool:
    """Whether two write patterns could match one path. Conservative: `**` matches any rest, and
    two wildcard segments are assumed to meet, so only provably disjoint patterns are False."""

    def wild(seg: str) -> bool:
        return any(c in seg for c in "*?[")

    sa, sb = a.split("/"), b.split("/")
    for x, y in zip(sa, sb, strict=False):
        if x == "**" or y == "**":
            return True
        if wild(x) and wild(y):
            continue
        if wild(x) or wild(y):
            literal, pattern = (y, x) if wild(x) else (x, y)
            if not fnmatch.fnmatchcase(literal, pattern):
                return False
        elif x != y:
            return False
    rest = sa[len(sb) :] or sb[len(sa) :]
    return all(seg == "**" for seg in rest)  # a trailing `**` may match nothing


class Ontology(Registry):
    version: str
    tiers: list[Tier]
    objects: dict[str, ObjectType]
    links: dict[str, LinkType]


def parse(data: bytes) -> Ontology:
    return Ontology.model_validate(tomllib.loads(data.decode("utf-8")))


def template_bytes() -> bytes:
    return resources.files("kyb.workspace.template").joinpath(TEMPLATE).read_bytes()


def template() -> Ontology:
    return parse(template_bytes())


def workspace_ontology(root: Path) -> Ontology:
    """A workspace's own ontology: its TOML, or a legacy 0.1.0 YAML recognised by its hash."""
    path = root / WORKSPACE_PATH
    if path.is_file():
        return parse(path.read_bytes())
    legacy = root / LEGACY_PATH
    if legacy.is_file():
        if sha256_bytes(legacy.read_bytes()) == LEGACY_SHA256:
            return template()
        raise ValueError(f"{legacy}: a YAML ontology that is not the 0.1.0 template's")
    raise FileNotFoundError(f"{root} has no ontology ({WORKSPACE_PATH})")


def json_schema() -> dict:
    return Ontology.model_json_schema(by_alias=True)


def main(argv: list[str] | None = None) -> int:
    what = (sys.argv[1:] if argv is None else argv) or ["json"]
    if what == ["schema"]:
        print(json.dumps(json_schema(), indent=2, sort_keys=True))
    elif what == ["json"]:
        print(json.dumps(template().model_dump(mode="json", by_alias=True), indent=2))
    else:
        print("usage: python -m kyb.ontology [json|schema]", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
