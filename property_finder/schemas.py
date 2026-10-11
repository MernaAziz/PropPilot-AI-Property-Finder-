"""Pydantic models.

The first three are filled by the LLM through PydanticOutputParser.
The last two are built by our own code from the parsed results and the data.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

Feature = Literal["balcony", "furnished", "parking", "elevator", "pets_allowed"]
FEATURES: tuple[str, ...] = Feature.__args__

# Words people use for each feature
_FEATURE_SYNONYMS = {
    "balcony": "balcony", "terrace": "balcony",
    "furnished": "furnished", "furniture": "furnished",
    "parking": "parking", "garage": "parking", "parking spot": "parking",
    "elevator": "elevator", "lift": "elevator",
    "pets_allowed": "pets_allowed", "pets": "pets_allowed", "pet": "pets_allowed",
    "pet friendly": "pets_allowed", "pet-friendly": "pets_allowed",
}


# ---------------------------------------------------------------------------
# Step 1 output: the user's request as structured filters
# ---------------------------------------------------------------------------
class SearchFilters(BaseModel):
    """Structured version of the user's apartment request."""

    max_price: Optional[int] = Field(
        None, description="Maximum monthly rent as a plain number. null if no budget was given."
    )
    min_bedrooms: Optional[int] = Field(
        None, description="Minimum number of bedrooms. A studio is 0. null if not mentioned."
    )
    districts: list[str] = Field(
        default_factory=list,
        description="District names the user explicitly named, copied from the known list. Empty if none.",
    )
    near: Optional[str] = Field(
        None,
        description="A place the user wants to live close to, e.g. 'the university' or 'the metro'. null if none.",
    )
    must_have: list[Feature] = Field(
        default_factory=list,
        description="Hard requirements. Only values from the allowed feature list.",
    )
    preferences: list[str] = Field(
        default_factory=list,
        description="Soft wishes that are not hard filters, e.g. 'quiet', 'modern', 'good for roommates'.",
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_features(cls, data):
        """Map synonyms ('lift' -> 'elevator') and move unknown features to preferences.

        This runs before validation, so an LLM answer like must_have=["Lift", "quiet"]
        becomes must_have=["elevator"], preferences=["quiet"] instead of crashing.
        """
        if not isinstance(data, dict):
            return data
        must_have, extra = [], []
        for item in data.get("must_have") or []:
            key = _FEATURE_SYNONYMS.get(str(item).strip().lower())
            if key and key not in must_have:
                must_have.append(key)
            elif not key:
                extra.append(str(item))
        data["must_have"] = must_have
        data["preferences"] = list(data.get("preferences") or []) + extra
        return data


# ---------------------------------------------------------------------------
# Step 2 output: which districts count as "near" a place
# ---------------------------------------------------------------------------
class AreaChoice(BaseModel):
    districts: list[str] = Field(description="Districts close to the place, closest first, at most 3.")
    reasoning: str = Field(description="One short sentence explaining the choice.")


# ---------------------------------------------------------------------------
# Step 4 output: the LLM's ranking of the candidates
# ---------------------------------------------------------------------------
class RankedListing(BaseModel):
    listing_id: str = Field(description="Exact id of the candidate, e.g. 'L07'.")
    match_score: int = Field(ge=0, le=100, description="How well it fits the request, 0-100.")
    pros: list[str] = Field(description="1-3 short pros, based only on the listing data.")
    cons: list[str] = Field(description="1-3 short cons, based only on the listing data.")
    reason: str = Field(description="One sentence explaining the score.")


class RankingResult(BaseModel):
    summary: str = Field(description="1-2 sentences for the user about the overall results.")
    results: list[RankedListing]


# ---------------------------------------------------------------------------
# Final output, assembled in code (prices and titles come from the data, not the LLM)
# ---------------------------------------------------------------------------
class PropertyResult(BaseModel):
    listing_id: str
    title: str
    district: str
    price: int
    bedrooms: int
    size_sqm: int
    match_score: int
    issues: list[str]
    pros: list[str]
    cons: list[str]
    reason: str


class SearchResponse(BaseModel):
    request: str
    filters: SearchFilters
    districts: list[str]
    area_note: str
    relax_notes: list[str]
    candidate_ids: list[str]
    summary: str
    results: list[PropertyResult]
