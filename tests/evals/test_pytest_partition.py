"""Finite receipt negatives plus actual tiny pytest runs, with no renderer."""

import os
from types import SimpleNamespace

import pytest

from kyb.evals.pytest_partition import PartitionRecorder, execute_partition


def item(name, bound=False):
    return SimpleNamespace(nodeid=name, get_closest_marker=lambda _: object() if bound else None)


def collected(recorder, *, renderer=True):
    all_items = [item("test_a"), item("test_b", bound=True)]
    hook = recorder.pytest_collection_modifyitems(all_items)
    next(hook)
    chosen = [all_items[1 if renderer else 0]]
    recorder.pytest_deselected([all_items[0 if renderer else 1]])
    all_items[:] = chosen
    with pytest.raises(StopIteration):
        next(hook)
    for phase in ("setup", "call", "teardown"):
        recorder.pytest_runtest_logreport(
            SimpleNamespace(nodeid=chosen[0].nodeid, when=phase, outcome="passed")
        )
    recorder.pytest_sessionfinish(0)
    return recorder


@pytest.mark.parametrize("partition", ["renderer", "portable"])
def test_valid_receipt_is_only_partition_proof(partition):
    recorder = collected(PartitionRecorder(partition), renderer=partition == "renderer")
    result = recorder.receipt()
    assert result["status"] == "pass" and result["product_pass"] is False
    assert result["selected"] == result["executed_passing"]
    assert set(result["selected"]) | set(result["deselected"]) == set(result["collected"])
    result["reports"][0]["outcome"] = "changed"
    assert recorder.receipt()["status"] == "pass"  # fresh output, no input aliases


@pytest.mark.parametrize(
    "attack",
    [
        "unfinished",
        "wrong-selection",
        "duplicate",
        "omitted",
        "unexpected",
        "deselected",
        "skipped",
        "failed",
        "xfail",
        "duplicate-phase",
        "missing-phase",
        "collection",
        "exit",
    ],
)
def test_incomplete_or_nonpassing_execution_never_passes(attack):
    recorder = collected(PartitionRecorder("renderer"))
    if attack == "unfinished":
        recorder.exit_code = None
    elif attack == "wrong-selection":
        recorder.partition = "portable"
    elif attack == "duplicate":
        recorder.collected.append("test_b")
    elif attack == "omitted":
        recorder.selected.clear()
    elif attack == "unexpected":
        recorder.reports[0]["nodeid"] = "foreign"
    elif attack == "deselected":
        recorder.deselected.clear()
    elif attack in {"skipped", "failed", "xfail"}:
        recorder.reports[1]["outcome"] = attack
    elif attack == "duplicate-phase":
        recorder.reports.append(dict(recorder.reports[1]))
    elif attack == "missing-phase":
        recorder.reports.pop()
    elif attack == "collection":
        recorder.collection_failed = True
    else:
        recorder.exit_code = 1
    assert recorder.receipt()["status"] == "failed"


def test_no_collection_and_unknown_partition_do_not_pass():
    assert PartitionRecorder("portable").receipt()["status"] == "failed"
    with pytest.raises(ValueError):
        PartitionRecorder("other")


def test_skipped_collection_cannot_shrink_the_required_universe():
    recorder = collected(PartitionRecorder("renderer"))
    recorder.pytest_collectreport(SimpleNamespace(failed=False, skipped=True))
    assert recorder.receipt()["status"] == "failed"


@pytest.mark.parametrize("partition", ["renderer", "portable"])
@pytest.mark.parametrize("skipped_module", [False, True])
def test_actual_tiny_pytest_collection_and_execution(
    tmp_path, partition, skipped_module, monkeypatch
):
    monkeypatch.setenv("UNTRUSTED_TEST_ENV", "synthetic-private-sentinel")
    (tmp_path / "pytest.ini").write_text("[pytest]\nmarkers = renderer_bound: synthetic marker\n")
    tests = tmp_path / "tests"
    tests.mkdir()
    # Each in-process pytest session must import a distinct synthetic module.
    source = tests / f"test_tiny_{tmp_path.name}.py"
    helper = f"support_{tmp_path.name}"
    (tests / f"{helper}.py").write_text("VALUE = True\n")
    source.write_text(
        f"from {helper} import VALUE\nassert VALUE\n"
        "import pytest, os\ndef test_portable(): assert 'UNTRUSTED_TEST_ENV' not in os.environ\n"
        "@pytest.mark.renderer_bound\ndef test_renderer():\n"
        "    assert 'UNTRUSTED_TEST_ENV' not in os.environ\n"
    )
    if skipped_module:
        (tests / f"test_module_skip_{tmp_path.name}.py").write_text(
            "import pytest\npytest.skip('synthetic omission', allow_module_level=True)\n"
        )
    receipt = execute_partition(tmp_path, partition)
    assert os.environ["UNTRUSTED_TEST_ENV"] == "synthetic-private-sentinel"
    assert receipt["exit_code"] == 0
    assert receipt["status"] == ("failed" if skipped_module else "pass")
    assert len(receipt["collected"]) == 2
    assert len(receipt["selected"]) == len(receipt["executed_passing"]) == 1
    assert len(receipt["renderer_bound"]) == len(receipt["deselected"]) == 1
