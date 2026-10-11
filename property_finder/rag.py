"""RAG layer: two Chroma vector stores (listings and area descriptions).

Listings are searched with a *metadata filter* (price, bedrooms, district,
features) plus *semantic similarity* to the user's request. Areas are searched
by similarity only, to find which districts are near a landmark.
"""

import json
import uuid
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from . import config
from .schemas import FEATURES, SearchFilters


# ---------------------------------------------------------------------------
# Loading and converting data
# ---------------------------------------------------------------------------
def load_json(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def feature_list(listing: dict) -> list[str]:
    return [f.replace("_", " ") for f in FEATURES if listing.get(f)]


def listing_to_document(listing: dict) -> Document:
    """The text is what gets embedded; the metadata is what we filter on."""
    beds = "studio" if listing["bedrooms"] == 0 else f"{listing['bedrooms']}-bedroom"
    features = ", ".join(feature_list(listing)) or "none"
    text = (
        f"{listing['title']}. {beds} apartment in {listing['district']}, "
        f"{listing['size_sqm']} sqm, floor {listing['floor']}. "
        f"Features: {features}. {listing['description']}"
    )
    metadata = {
        "id": listing["id"],
        "district": listing["district"],
        "price": listing["price"],
        "bedrooms": listing["bedrooms"],
        **{f: bool(listing.get(f)) for f in FEATURES},
    }
    return Document(page_content=text, metadata=metadata)


def area_to_document(area: dict) -> Document:
    text = (
        f"{area['district']}: {area['description']} "
        f"Landmarks: {', '.join(area['landmarks'])}. "
        f"To City University: {area['to_university']}. "
        f"To the metro: {area['to_metro']}. Vibe: {area['vibe']}."
    )
    return Document(page_content=text, metadata={"district": area["district"]})


# ---------------------------------------------------------------------------
# Metadata filter
# ---------------------------------------------------------------------------
def build_where(filters: SearchFilters, districts: list[str], price_factor: float = 1.0) -> dict | None:
    """Turn SearchFilters into a Chroma `where` filter."""
    conditions = []
    if filters.max_price is not None:
        conditions.append({"price": {"$lte": round(filters.max_price * price_factor)}})
    if filters.min_bedrooms is not None:
        conditions.append({"bedrooms": {"$gte": filters.min_bedrooms}})
    if districts:
        conditions.append({"district": {"$in": districts}})
    for feature in filters.must_have:
        conditions.append({feature: {"$eq": True}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}  # Chroma needs $and when there are 2+ conditions


# ---------------------------------------------------------------------------
# The index
# ---------------------------------------------------------------------------
class PropertyIndex:
    def __init__(self, embeddings: Embeddings, listings: list[dict], areas: list[dict]):
        self.listings = {l["id"]: l for l in listings}
        self.areas = {a["district"]: a for a in areas}

        suffix = uuid.uuid4().hex[:8]  # unique names, so rebuilding never mixes old data in
        self.listing_store = Chroma.from_documents(
            [listing_to_document(l) for l in listings],
            embedding=embeddings,
            ids=[l["id"] for l in listings],
            collection_name=f"listings_{suffix}",
        )
        self.area_store = Chroma.from_documents(
            [area_to_document(a) for a in areas],
            embedding=embeddings,
            ids=[a["district"] for a in areas],
            collection_name=f"areas_{suffix}",
        )

    @classmethod
    def from_files(cls, embeddings: Embeddings) -> "PropertyIndex":
        return cls(embeddings, load_json(config.LISTINGS_FILE), load_json(config.AREAS_FILE))

    @property
    def district_names(self) -> list[str]:
        return list(self.areas)

    def search_areas(self, query: str, k: int = 3) -> list[Document]:
        return self.area_store.similarity_search(query, k=k)

    def search_listings(
        self,
        query: str,
        filters: SearchFilters,
        districts: list[str],
        k: int = config.TOP_K,
        price_factor: float = 1.0,
    ) -> list[dict]:
        where = build_where(filters, districts, price_factor)
        docs = self.listing_store.similarity_search(query, k=k, filter=where)
        return [self.listings[d.metadata["id"]] for d in docs]
