"""The structured statements an agent may answer with (agent-driven-review §5).

An agent answers in these four shapes only. A factual statement carries no prose: its text is
rendered by code from the record it cites.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_STATEMENTS = 20
# a label names what was asked; a colon or bullet would let it pose as a rendered "label: value"
# line or a second statement
_LABEL = r"^[^:\u2022\x00-\x1f\x7f\x85\u2028\u2029]+$"
MAX_LABEL = 80
# a label or topic is one printable line: a line break could carry a second, unchecked "statement"
# inside a genuine rendered answer, past the display check
_ONE_LINE = r"^[^\x00-\x1f\x7f\x85\u2028\u2029]+$"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _text(v: object) -> str:
    # an agent may pass 25 or 2026-01-09 as a bare value; the check compares text
    if isinstance(v, bool) or not isinstance(v, str | int | float):
        raise ValueError("a value is plain text or a number")
    return str(v)


class FactStatement(_Strict):
    """A fact or verified claim, stated under its own subject and attribute."""

    type: Literal["fact"]
    subject: str = Field(min_length=1, pattern=_ONE_LINE)
    attribute: str = Field(min_length=1, pattern=_ONE_LINE)
    value: str = Field(min_length=1, pattern=_ONE_LINE)
    cites: list[str] = Field(min_length=1)  # fact ids and verified claim ids from the bundle

    _value = field_validator("value", mode="before")(_text)


class SnippetStatement(_Strict):
    """Document content the ontology does not model: a value inside a witnessed snippet."""

    type: Literal["snippet"]
    question_attribute: str = Field(min_length=1, max_length=MAX_LABEL, pattern=_LABEL)
    value: str = Field(min_length=1, pattern=_ONE_LINE)
    cites: list[str] = Field(min_length=1)  # snippet ids from the bundle

    _value = field_validator("value", mode="before")(_text)


class ExplainStatement(_Strict):
    """Why a recorded decision is what it is, rendered from the decision record alone."""

    type: Literal["explain"]
    decision_id: str = Field(min_length=1)
    rule_ids: list[str] = []  # empty: every fired rule


class NotInDocuments(_Strict):
    """The literal answer when the record has no support for what was asked."""

    type: Literal["NOT_IN_DOCUMENTS"]
    topic: str = Field(
        min_length=1, max_length=MAX_LABEL, pattern=_ONE_LINE
    )  # what could not be found, in the documents' own terms


Statement = Annotated[
    FactStatement | SnippetStatement | ExplainStatement | NotInDocuments,
    Field(discriminator="type"),
]
