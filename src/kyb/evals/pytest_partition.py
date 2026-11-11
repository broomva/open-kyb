"""Actual pytest node/phase receipts, never an aggregate or native attestation.

The fixed host runner supplies this plugin to pytest. Marker selection schedules
work; only complete selected execution is evidence for that partition. Native
source, renderer binding, union coverage and strict verification remain separate.
"""

from __future__ import annotations

import os
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

_OFFLINE_ENV = {
    "HOME",
    "PATH",
    "TMPDIR",
    "TEMP",
    "TMP",
    "LANG",
    "PYTHONPATH",
    "GITHUB_ACTIONS",
    "GITHUB_REPOSITORY",
    "RUNNER_NAME",
    "RUNNER_OS",
    "RUNNER_ARCH",
    "GITHUB_JOB",
    "GITHUB_WORKFLOW",
    "GITHUB_WORKSPACE",
    "GITHUB_SHA",
    "GITHUB_WORKFLOW_SHA",
    "GITHUB_EVENT_NAME",
    "GITHUB_EVENT_PATH",
    "GITHUB_WORKFLOW_REF",
    "GITHUB_RUN_ID",
    "GITHUB_RUN_ATTEMPT",
}


@contextmanager
def _offline_environment():
    """Process-local test isolation; restore the parent, persist no environment."""
    original = dict(os.environ)
    allowed = {k: v for k, v in original.items() if k in _OFFLINE_ENV or k.startswith("LC_")}
    allowed.update(PYTHONDONTWRITEBYTECODE="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    try:
        os.environ.clear()
        os.environ.update(allowed)
        yield
    finally:
        os.environ.clear()
        os.environ.update(original)


class PartitionRecorder:
    """Record full collection before marker selection and actual phase outcomes."""

    def __init__(self, partition: str) -> None:
        if partition not in {"portable", "renderer"}:
            raise ValueError("unknown pytest partition")
        self.partition = partition
        self.collected: list[str] | None = None
        self.renderer: list[str] = []
        self.selected: list[str] | None = None
        self.deselected: list[str] = []
        self.reports: list[dict[str, str]] = []
        self.collection_failed = False
        self.exit_code: int | None = None

    @pytest.hookimpl(hookwrapper=True, tryfirst=True)
    def pytest_collection_modifyitems(self, items):
        # Wrapping observes the unfiltered collection and the final selected set.
        self.collected = [item.nodeid for item in items]
        self.renderer = [
            item.nodeid for item in items if item.get_closest_marker("renderer_bound") is not None
        ]
        yield
        self.selected = [item.nodeid for item in items]

    def pytest_deselected(self, items) -> None:
        self.deselected.extend(item.nodeid for item in items)

    def pytest_collectreport(self, report) -> None:
        self.collection_failed |= report.failed or report.skipped

    def pytest_runtest_logreport(self, report) -> None:
        self.reports.append(
            {
                "nodeid": report.nodeid,
                "phase": report.when,
                "outcome": "xfail" if getattr(report, "wasxfail", None) else report.outcome,
            }
        )

    def pytest_sessionfinish(self, exitstatus) -> None:
        self.exit_code = int(exitstatus)

    def receipt(self) -> dict[str, Any]:
        """Require every selected node to have one passing setup/call/teardown."""
        errors = []
        collected, selected = self.collected or [], self.selected or []
        all_ids, selected_ids, renderer_ids = set(collected), set(selected), set(self.renderer)
        if self.collected is None or self.selected is None:
            errors.append("collection-incomplete")
        if any(type(n) is not str or not n for n in collected + selected + self.renderer):
            errors.append("invalid-node-id")
        if len(all_ids) != len(collected) or len(selected_ids) != len(selected):
            errors.append("duplicate-collected-or-selected-node")
        if len(renderer_ids) != len(self.renderer) or not renderer_ids <= all_ids:
            errors.append("invalid-renderer-marker-set")
        expected = renderer_ids if self.partition == "renderer" else all_ids - renderer_ids
        if not expected or selected_ids != expected:
            errors.append("selected-partition-incomplete-or-unexpected")
        if set(self.deselected) != all_ids - selected_ids or len(set(self.deselected)) != len(
            self.deselected
        ):
            errors.append("deselection-manifest-inconsistent")
        if self.collection_failed:
            errors.append("collection-failed")
        if self.exit_code != 0:
            errors.append("pytest-not-successful-or-unfinished")
        phases: dict[str, list[dict[str, str]]] = {}
        for report in self.reports:
            phases.setdefault(report["nodeid"], []).append(report)
        if set(phases) != selected_ids:
            errors.append("execution-node-set-incomplete-or-unexpected")
        complete = []
        for node, reports in phases.items():
            if Counter(r["phase"] for r in reports) != Counter(
                {"setup": 1, "call": 1, "teardown": 1}
            ) or any(r["outcome"] != "passed" for r in reports):
                errors.append("node-skipped-failed-or-phase-count-invalid")
            else:
                complete.append(node)
        return {
            "schema": "kyb-pytest-partition-v1",
            "partition": self.partition,
            "status": "failed" if errors else "pass",
            "collected": sorted(collected),
            "renderer_bound": sorted(self.renderer),
            "selected": sorted(selected),
            "deselected": sorted(self.deselected),
            "executed_passing": sorted(complete),
            "reports": [dict(r) for r in self.reports],
            "exit_code": self.exit_code,
            "errors": sorted(set(errors)),
            "product_pass": False,
        }


def execute_partition(root: Path, partition: str) -> dict[str, Any]:
    """Fixed selection; the native caller must own host/source/heavy admission."""
    recorder = PartitionRecorder(partition)
    with _offline_environment():
        pytest.main(
            [
                "-q",
                "-p",
                "no:cacheprovider",
                "--rootdir",
                str(root),
                str(root / "tests"),
                "-m",
                "renderer_bound" if partition == "renderer" else "not renderer_bound",
            ],
            plugins=[recorder],
        )
    return recorder.receipt()
