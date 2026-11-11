"""spec: the decision profile, chosen explicitly at decide time. Absent, it is `strict`.

`KYB_DECISION_PROFILE=approve-measured` (or `kyb decide --profile approve-measured`) scopes the
fact-gap vetoes (J-EVIDENCE on failed claims, J-VERIFICATION on uncovered facts) to the gaps a
live rule reads. It never touches unknown rules, REJECTED/MANUAL_REVIEW, J-ANOMALY, the W1 cap,
the fixture-input veto, J-SUPPORT or reader isolation (spec holds: models never clear).
"""

from __future__ import annotations

import os

from kyb.judges.core import PROFILES, STRICT

ENV = "KYB_DECISION_PROFILE"


def resolve_profile(explicit: str | None = None) -> str:
    """The flag, else the environment, else `strict`. An unknown name is an error, not a fallback:
    a misspelt opt-in must not silently decide under another policy."""
    name = (explicit or os.environ.get(ENV) or STRICT).strip().lower()
    if name not in PROFILES:
        raise ValueError(f"unknown decision profile {name!r}; use one of {', '.join(PROFILES)}")
    return name
