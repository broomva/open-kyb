"""Calendar arithmetic for the policy's windows. Months are calendar months, clamped to the last
day of a shorter month, and every window is measured back from the frozen `as_of` (spec)."""

from __future__ import annotations

import calendar
from datetime import date, timedelta

from kyb.contracts.b4 import Document


def parse(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def add_months(d: date, months: int) -> date:
    y, m = divmod(d.month - 1 + months, 12)
    year, month = d.year + y, m + 1
    return date(year, month, min(d.day, calendar.monthrange(year, month)[1]))


def months_before(as_of: date, months: int) -> date:
    return add_months(as_of, -months)


def years_before(as_of: date, years: int) -> date:
    return add_months(as_of, -12 * years)


def doc_date(doc: Document) -> date | None:
    """A lease is dated by its signature (spec); anything else by its issue date, else by its
    signature."""
    if doc.doc_type == "lease":
        return parse(doc.signed_on)
    return parse(doc.issued) or parse(doc.signed_on)


def lapsed(doc: Document, as_of: date) -> bool | None:
    """An issuer-stated validity that has run out by `as_of` (spec). None when a validity is
    stated but the issue date it runs from is not known."""
    if doc.stated_validity_days is None:
        return False
    issued = parse(doc.issued)
    if issued is None:
        return None
    return as_of > issued + timedelta(days=doc.stated_validity_days)
