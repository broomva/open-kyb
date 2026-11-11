"""Witness envelope drift report: aggregate `envelope_anomalies` across recorded witness responses.

    uv run python scripts/witness_drift.py [--ws WORKSPACE | --path DIR] [--json]

Reads every `silver/witness/raw/*/{0.json,failed/*.json}` record under the workspace (or any
`--path`), and prints one row per (path, key): count, providers, first-seen. A maintainer adds the
benign keys to the allow-lists (`kyb.witness.gemini_outcome._ROOT_SPEC`, `claude_stream._*_SPECS`)
in a one-line PR; `severity: error` rows are real failure signals and are never allow-listed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def collect(base: Path) -> list[dict]:
    """Every anomaly with the record it came from, in a stable (path-sorted) order."""
    found = []
    for record in sorted(base.rglob("*.json")):
        if "raw" not in record.parts:
            continue
        try:
            body = json.loads(record.read_text("utf-8"))
        except OSError, ValueError:
            continue
        prov = body.get("provenance") if isinstance(body, dict) else None
        for item in (prov or {}).get("envelope_anomalies") or []:
            if isinstance(item, dict):
                found.append({**item, "record": str(record.relative_to(base))})
    return found


def aggregate(anomalies: list[dict]) -> list[dict]:
    rows: dict[tuple, dict] = {}
    for a in anomalies:
        key = (a.get("path"), a.get("key"), a.get("severity"))
        row = rows.setdefault(
            key,
            {
                "path": key[0],
                "key": key[1],
                "severity": key[2],
                "count": 0,
                "providers": set(),
                "first_seen": a["record"],
            },
        )
        row["count"] += 1
        row["providers"].add(a.get("provider"))
    return sorted(
        ({**r, "providers": sorted(map(str, r["providers"]))} for r in rows.values()),
        key=lambda r: (-r["count"], str(r["path"])),
    )


def render(rows: list[dict]) -> str:
    if not rows:
        return "no envelope anomalies recorded\n"
    head = ("path", "key", "severity", "count", "providers", "first_seen")
    table = [head] + [(str(r["path"]), str(r["key"]), str(r["severity"]), str(r["count"]),
                       ",".join(r["providers"]), str(r["first_seen"])) for r in rows]  # fmt: skip
    widths = [max(len(row[i]) for row in table) for i in range(len(head))]
    return "".join(
        "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)).rstrip() + "\n"
        for row in table
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--ws", default="workspaces/kyb-demo", help="workspace root")
    parser.add_argument("--path", help="directory to scan instead of the workspace")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    base = Path(args.path or args.ws)
    if not base.is_dir():
        print(f"not a directory: {base}", file=sys.stderr)
        return 2
    rows = aggregate(collect(base))
    sys.stdout.write(json.dumps(rows, indent=2) + "\n" if args.json else render(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
