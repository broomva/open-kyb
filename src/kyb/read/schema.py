"""The reading contract's shapes (spec, spec): layout records, claims, and a claims submission.

Validated by Pydantic before anything is recorded. Bounding boxes are pixels of the 300-dpi render
of the page, origin top-left, as `[x0, y0, x1, y1]`.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MAX_CROPS_PER_DOC = 60
MAX_PAGES_PER_DOC = 40
ELLIPSIS = re.compile(r"\.\.\.|…")
CLAIM_ID = r"^c-[A-Za-z0-9]{1,8}-[A-Za-z0-9_.-]{1,40}$"

BBox = tuple[int, int, int, int]


def _check_bbox(b: BBox) -> BBox:
    x0, y0, x1, y1 = b
    if min(b) < 0 or x0 >= x1 or y0 >= y1:
        raise ValueError(f"bbox {list(b)} must be [x0, y0, x1, y1] with x0 < x1, y0 < y1, all >= 0")
    return b


def inside(inner: BBox, outer: BBox) -> bool:
    return (
        inner[0] >= outer[0]
        and inner[1] >= outer[1]
        and inner[2] <= outer[2]
        and inner[3] <= outer[3]
    )


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Block(_Strict):
    block_id: str
    kind: str
    bbox: BBox
    heading: str | None = None
    qualifiers: list[str] = []
    markers: list[str] = []
    refers_to: list[str] = []

    @field_validator("bbox")
    @classmethod
    def _bbox_ok(cls, v: BBox) -> BBox:
        return _check_bbox(v)


class MaterialFact(_Strict):
    """A fact the claim-blind layout pass found; each must be matched by a verified claim."""

    block_id: str
    kind: str  # generated from the policy (spec): ownership, identifier, date, activity, ...
    subject: str
    attribute: str
    value: str


class PageLayout(_Strict):
    page: int = Field(ge=1)
    page_hash: str
    size_px: tuple[int, int]
    blocks: list[Block]


class LayoutRecord(_Strict):
    """Written by `layout_pages` (the witness, claim-blind) before any reading."""

    document_id: str
    doc_sha256: str
    model: str
    pages: list[PageLayout]
    material_facts: list[MaterialFact] = []

    @model_validator(mode="after")
    def _blocks_unique(self) -> LayoutRecord:
        ids = [b.block_id for p in self.pages for b in p.blocks]
        if len(ids) != len(set(ids)):
            raise ValueError("block ids must be unique within a document")
        return self


class Claim(_Strict):
    claim_id: str = Field(pattern=CLAIM_ID)
    type: str = Field(min_length=1, max_length=40)
    subject: str = Field(min_length=1)
    attribute: str | None = None
    object: str | None = None
    value: Any
    excerpt: str = Field(min_length=1, max_length=600)
    page: int = Field(ge=1)
    bbox: BBox
    block_id: str
    linked_blocks: list[str] = []
    source: Literal["text_layer", "pixels"]
    document_date: str | None = None
    role: str | None = None
    confidence: float = Field(ge=0, le=1)
    flags: list[str] = []

    @field_validator("bbox")
    @classmethod
    def _bbox_ok(cls, v: BBox) -> BBox:
        return _check_bbox(v)

    @field_validator("excerpt")
    @classmethod
    def _contiguous(cls, v: str) -> str:
        if ELLIPSIS.search(v):
            raise ValueError("excerpts must be contiguous: no ellipses")
        return v


class ClaimsSubmission(_Strict):
    """The arguments of `read_document`, exactly as the reader passes them."""

    case: str
    run: str
    doc: str
    doc_sha256: str
    claims: list[Claim] = Field(max_length=MAX_CROPS_PER_DOC)

    @model_validator(mode="after")
    def _ids_unique(self) -> ClaimsSubmission:
        ids = [c.claim_id for c in self.claims]
        if len(ids) != len(set(ids)):
            raise ValueError("claim ids must be unique within a submission")
        return self
