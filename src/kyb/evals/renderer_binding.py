"""Pure renderer-candidate validation module for the open-kyb KYB pipeline.

Implements the reusable strict consumer contract (D070/D072) to validate
a revalidation candidate against separately supplied expected job context,
input manifest, and current profile.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from kyb.evals.renderer import (
    COMPONENTS,
    Comparison,
    RendererIncomparable,
    check_historical_maps,
    compare,
)
from kyb.evals.witness import fullrun

_SHA = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_POSITIVE = re.compile(r"^[1-9][0-9]*$")

# List of expected job fields
EXPECTED_JOB_FIELDS = {
    "head",
    "event_sha",
    "workflow_sha",
    "event",
    "pr",
    "run_id",
    "attempt",
    "workflow_ref",
    "runner",
}


def _same_types(actual: Any, expected: Any) -> bool:
    """JSON booleans and integers are distinct evidence, despite Python equality."""
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            _same_types(actual[key], value) for key, value in expected.items()
        )
    return actual == expected


def _hash_profile(profile: dict[str, str]) -> str:
    raw = json.dumps(profile, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _valid_profile_structure(profile: Any) -> bool:
    """Rigorous check that profile structure matches expectations strictly."""
    if not isinstance(profile, dict) or set(profile) != set(COMPONENTS):
        return False
    if any(not isinstance(profile[k], str) or not profile[k] for k in COMPONENTS):
        return False
    return all(
        _SHA256.fullmatch(profile[k])
        for k in ("executable_sha256", "font_set_sha256", "font_config_sha256")
    )


def validate_candidate(
    candidate: Any,
    *,
    expected_job: dict[str, Any],
    current_profile: dict[str, Any],
    input_manifest: dict[str, str],
) -> dict[str, Any]:
    """Validate a revalidation candidate against independent caller contexts.

    Performs strict schema/status/type validation, native job checks,
    input manifest equivalence, historical verdict map checks, request count
    consistency, and profile comparison.

    Returns a kyb-renderer-binding-check-v1 validation record.
    """
    # Check 1: Strict candidate schema/kind and scalar check.
    if not isinstance(candidate, dict):
        raise ValueError("Candidate must be a dictionary")
    if "incomparability_stage" not in candidate:
        raise ValueError("Candidate incomparable stage must be explicitly present")

    # Guard against summary/observation-only object or wrong schema
    if (
        candidate.get("kind") != "revalidation-candidate"
        or candidate.get("schema") != "kyb-renderer-revalidation-candidate-v1"
    ):
        raise ValueError("Invalid candidate kind or schema")

    if candidate.get("status") != "revalidated":
        # It must not be rescued or incomparable/pass if failed/malformed
        raise ValueError("Candidate status is not revalidated")

    # Strict type checks on metadata/errors/flags
    errors = candidate.get("errors")
    if not isinstance(errors, list) or len(errors) > 0:
        raise ValueError("Candidate must have zero errors")

    # Ensure product_pass is exactly False and of type bool
    product_pass = candidate.get("product_pass")
    if type(product_pass) is not bool or product_pass is not False:
        raise ValueError("Candidate product_pass must be strictly False")

    if candidate.get("legacy_binding") != "candidate-unadmitted":
        raise ValueError("Candidate legacy_binding must be candidate-unadmitted")

    if candidate.get("recording_time_identity") != "unknown":
        raise ValueError("Candidate recording_time_identity must be unknown")

    # Ensure no previous ordinary pytest-partition execution claim
    if candidate.get("pytest_renderer_partition") != "not executed by this candidate collector":
        raise ValueError("Candidate pytest_renderer_partition claims execution")

    # Check 2: Require complete native job fields and strict strings
    if not isinstance(expected_job, dict) or set(expected_job) != EXPECTED_JOB_FIELDS:
        raise ValueError("Expected job fields missing or extra")

    # Validate types of expected_job values (must be strict strings)
    for field, val in expected_job.items():
        if type(val) is not str:
            raise ValueError(f"Job field {field} must be a strict string")

    # Validate formats of expected_job fields
    if not _SHA.fullmatch(expected_job["head"]):
        raise ValueError("Job head must be a full 40-hex SHA")
    if not _SHA.fullmatch(expected_job["event_sha"]):
        raise ValueError("Job event_sha must be a full 40-hex SHA")
    if not _SHA.fullmatch(expected_job["workflow_sha"]):
        raise ValueError("Job workflow_sha must be a full 40-hex SHA")

    if not _POSITIVE.fullmatch(expected_job["run_id"]):
        raise ValueError("Job run_id must be a positive integer string")
    if not _POSITIVE.fullmatch(expected_job["attempt"]):
        raise ValueError("Job attempt must be a positive integer string")

    if expected_job["runner"] != "kyb-self-hosted-runner-1":
        raise ValueError("Job runner must be the approved Mac runner")

    event = expected_job["event"]
    if event == "pull_request":
        if not _POSITIVE.fullmatch(expected_job["pr"]):
            raise ValueError("Job pr must be positive integer string for pull_request event")
    elif event == "workflow_dispatch":
        if expected_job["pr"] != "":
            raise ValueError("Job pr must be empty for workflow_dispatch event")
    else:
        raise ValueError("Job event must be pull_request or workflow_dispatch")

    workflow_ref = expected_job["workflow_ref"]
    if not workflow_ref.startswith("broomva/open-kyb/.github/workflows/ci.yml@") or any(
        c.isspace() for c in workflow_ref
    ):
        raise ValueError("Job workflow_ref is malformed or unapproved")
    ref = workflow_ref.split("@", 1)[1]
    is_branch = ref.startswith("refs/heads/") and bool(ref[len("refs/heads/") :])
    is_pr_merge = event == "pull_request" and ref == f"refs/pull/{expected_job['pr']}/merge"
    if not (is_branch or is_pr_merge):
        raise ValueError("Job workflow_ref does not match its event and PR")

    # Compare every candidate job field exactly against expected_job
    candidate_job = candidate.get("job")
    if not isinstance(candidate_job, dict) or candidate_job != expected_job:
        raise ValueError("Candidate job does not match expected job context")

    # Check 3: Manifest Validation
    if not isinstance(input_manifest, dict) or not input_manifest:
        raise ValueError("Expected input manifest is empty or not a dict")

    # Check that keys are string and values are lowercase64-hex SHA256 maps
    for k, v in input_manifest.items():
        if type(k) is not str or type(v) is not str or not _SHA256.fullmatch(v):
            raise ValueError("Manifest keys/values must be string to lowercase 64-hex SHA256")

    candidate_manifest = candidate.get("input_manifest")
    if not isinstance(candidate_manifest, dict) or candidate_manifest != input_manifest:
        raise ValueError("Candidate input_manifest mismatch")

    # Canonical compact sorted-JSON digest match
    encoded = json.dumps(input_manifest, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(encoded).hexdigest()
    if candidate.get("input_manifest_sha256") != digest:
        raise ValueError("Candidate input_manifest_sha256 mismatch")

    # Check 4: Historical Maps Validation
    candidate_map_evidence = candidate.get("map_evidence")
    # This will raise ValueError on any internal maps malformations/mismatches/alterations
    recomputed_map_check = check_historical_maps(candidate_map_evidence)

    # Check exact equality to the retained map_check
    if not _same_types(candidate.get("map_check"), recomputed_map_check):
        raise ValueError("Candidate map_check mismatch")

    # Check 5: Provider & Seed Request Counts Validation
    new_provider_calls = candidate.get("new_provider_calls")
    if type(new_provider_calls) is not int or new_provider_calls != 0:
        raise ValueError("Candidate new_provider_calls must be strict integer 0")

    product_seed_requests = candidate.get("product_seed_requests")
    if not isinstance(product_seed_requests, dict):
        raise ValueError("Candidate product_seed_requests must be a dict")

    runs = fullrun.runs()
    expected_run_names = {run.name for run in runs}
    if set(product_seed_requests) != expected_run_names:
        raise ValueError("Candidate product_seed_requests keys must match expected run names")

    for run_name in expected_run_names:
        val = product_seed_requests[run_name]
        if type(val) is not int or val <= 0:
            raise ValueError("Product seed request count must be strictly positive integer")

    # Check 6: Profile Comparison & Matching Validation
    profile_before = candidate.get("profile_before")
    profile_after = candidate.get("profile_after")

    if not isinstance(profile_before, dict) or not isinstance(profile_after, dict):
        raise ValueError("Candidate profiles must be dictionaries")

    if not _valid_profile_structure(profile_before) or not _valid_profile_structure(profile_after):
        raise ValueError("Candidate profiles are structurally invalid or incomplete")

    if not isinstance(current_profile, dict) or not _valid_profile_structure(current_profile):
        # Current profile is incomplete/invalid, hence incomparable/unavailable
        raise RendererIncomparable(
            "Current profile is structurally incomplete or unavailable", category="profile-schema"
        )

    # Compare profiles using compare from renderer
    before_after_comp = compare(profile_before, profile_after)
    before_current_comp = compare(profile_before, current_profile)
    after_current_comp = compare(profile_after, current_profile)

    if (
        before_after_comp.state is not Comparison.MATCH
        or before_current_comp.state is not Comparison.MATCH
        or after_current_comp.state is not Comparison.MATCH
    ):
        raise RendererIncomparable("Renderer profile mismatch or drift detected", category="drift")

    if candidate.get("renderer_comparison") != "MATCH":
        raise RendererIncomparable("Candidate renderer_comparison is not MATCH", category="drift")

    if candidate.get("incomparability_stage") is not None:
        raise RendererIncomparable("Candidate incomparable stage is not null", category="drift")

    # Return only a fresh kyb-renderer-binding-check-v1 validation record
    return {
        "schema": "kyb-renderer-binding-check-v1",
        "binding_eligible": True,
        "eligible_for_admission": True,
        "job": dict(expected_job),
        "profile": dict(current_profile),
        "profile_hash": _hash_profile(current_profile),
        "input_manifest_sha256": digest,
        "map_check": recomputed_map_check,
        "recording_time_identity": "unknown",
        "binding_admitted": False,
        "product_pass": False,
    }
