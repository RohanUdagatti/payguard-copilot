"""Pydantic schema and validation for Golden Set questions."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from payguard.rag.generator import NOT_FOUND_MESSAGE
from payguard.rag.security_checks import contains_luhn_pan

GoldenType = Literal[
    "lookup", "paraphrase", "near_miss", "multi_section", "out_of_scope"
]


class GoldenQuestion(BaseModel):
    """Schema for a golden evaluation dataset question."""

    id: str = Field(min_length=1, description="Unique question identifier.")
    question: str = Field(min_length=1, description="User question.")
    ground_truth: str = Field(default="", description="Factual reference answer.")
    source_doc: str = Field(
        min_length=1, description="Primary document ID (or 'none' for out_of_scope)."
    )
    section_hint: str | None = Field(
        default=None, description="Expected section heading or chapter."
    )
    region: str | None = Field(default="US", description="Regulatory jurisdiction.")
    type: GoldenType = Field(
        description="lookup | paraphrase | near_miss | multi_section | out_of_scope."
    )
    answerable: bool = Field(
        description="Whether the question is answerable from ingested documents."
    )
    verified: bool = Field(
        default=False,
        description="Whether this entry has been human-reviewed and verified.",
    )
    needs_ground_truth: bool = Field(
        default=False,
        description="Flag indicating human must fill in ground_truth.",
    )

    @model_validator(mode="after")
    def _validate_rules(self) -> GoldenQuestion:
        # 1. Reject Luhn-valid card numbers (PANs)
        if contains_luhn_pan(self.question):
            raise ValueError(
                f"PAN detected in question for {self.id}: card numbers are prohibited."
            )
        if self.ground_truth and contains_luhn_pan(self.ground_truth):
            raise ValueError(
                f"PAN detected in ground_truth for {self.id}: "
                "card numbers are prohibited."
            )

        # 2. Out-of-scope validation
        if self.type == "out_of_scope":
            if self.answerable:
                raise ValueError(
                    f"Out-of-scope item {self.id} must have answerable=False."
                )
            if self.ground_truth != NOT_FOUND_MESSAGE:
                raise ValueError(
                    f"Out-of-scope item {self.id} must have "
                    f"ground_truth={NOT_FOUND_MESSAGE!r}."
                )

        # 3. Verified items must have non-empty ground_truth
        if self.verified and not self.ground_truth.strip():
            raise ValueError(
                f"Verified item {self.id} must have non-empty ground_truth."
            )

        return self
