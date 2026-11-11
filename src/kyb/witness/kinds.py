"""Material-fact kinds the claim-blind layout pass must list.

They are the attributes the policy reads. When the policy module evaluates a run, these kinds map
uncovered layout blocks and unverified claims to the rule kinds that depend on them.
"""

from __future__ import annotations

MATERIAL_FACT_KINDS = {
    "ownership": "shareholders and percentages, direct or through entities",
    "identifier": "registry numbers, tax ids, passports, dates of birth",
    "date": "issue, incorporation, signing and validity dates",
    "address": "registered and operating addresses",
    "role": "directors, legal representatives, nominees, signatories",
    "amendment": "struck-through, handwritten or pending changes",
    "corporate_activity": "every activity in the declared corporate object",
    "jurisdiction": "countries of incorporation and residence",
    "financial_reference": "bank references and financial statements",
    "operations": "employees, website, operating evidence",
    "stated_validity": "validity periods stated by an issuer",
}
KINDS_2026_10_R3 = MATERIAL_FACT_KINDS


def kinds(rules_version: str = "2026.11-r1") -> list[str]:
    return [f"{k}: {v}" for k, v in MATERIAL_FACT_KINDS.items()]
