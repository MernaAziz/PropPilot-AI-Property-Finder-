"""The full pipeline, built with LCEL.

    request
      │
      ├─ Step 1  parse_chain      prompt | llm | PydanticOutputParser(SearchFilters)
      ├─ Step 2  area_step        RAG over area docs -> prompt | llm | PydanticOutputParser(AreaChoice)
      ├─ Step 3  find_candidates  RAG over listings with metadata filters (+ relaxation)
      ├─ Step 4  rank_step        prompt | llm | PydanticOutputParser(RankingResult)
      └─ Step 5  assemble         merge LLM ranking with real listing data -> SearchResponse

Each step adds one key to a shared dict with RunnablePassthrough.assign(), so
later steps can see everything earlier steps produced.
"""

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import (
    Runnable,
    RunnableBranch,
    RunnableConfig,
    RunnableLambda,
    RunnablePassthrough,
)

from . import config
from .prompts import AREA_PROMPT, PARSE_PROMPT, RANK_PROMPT
from .rag import PropertyIndex, feature_list
from .schemas import (
    AreaChoice,
    PropertyResult,
    RankingResult,
    SearchFilters,
    SearchResponse,
)


# ---------------------------------------------------------------------------
# The three LLM chains (prompt | llm | parser)
# ---------------------------------------------------------------------------
def build_parse_chain(llm: BaseChatModel, index: PropertyIndex) -> Runnable:
    parser = PydanticOutputParser(pydantic_object=SearchFilters)
    prompt = PARSE_PROMPT.partial(
        format_instructions=parser.get_format_instructions(),
        district_names=", ".join(index.district_names),
    )
    return prompt | llm | parser


def build_area_chain(llm: BaseChatModel, index: PropertyIndex) -> Runnable:
    parser = PydanticOutputParser(pydantic_object=AreaChoice)
    prompt = AREA_PROMPT.partial(format_instructions=parser.get_format_instructions())

    def retrieve_area_context(x: dict) -> str:
        docs = index.search_areas(x["near"], k=3)
        return "\n\n".join(d.page_content for d in docs)

    # Retrieval (RAG) -> prompt -> llm -> parser
    return RunnablePassthrough.assign(area_context=retrieve_area_context) | prompt | llm | parser


def build_rank_chain(llm: BaseChatModel) -> Runnable:
    parser = PydanticOutputParser(pydantic_object=RankingResult)
    prompt = RANK_PROMPT.partial(format_instructions=parser.get_format_instructions())
    return prompt | llm | parser


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def clean_districts(names: list[str], known: list[str]) -> list[str]:
    """Keep only real district names (case-insensitive), in the original order."""
    lookup = {k.lower(): k for k in known}
    out = []
    for name in names:
        match = lookup.get(name.strip().lower())
        if match and match not in out:
            out.append(match)
    return out


def find_issues(listing: dict, filters: SearchFilters, districts: list[str]) -> list[str]:
    """Requirements this listing breaks. Computed in code, so it is always correct."""
    issues = []
    if districts and listing["district"] not in districts:
        issues.append(f"Outside {' / '.join(districts)}")
    if filters.max_price is not None and listing["price"] > filters.max_price:
        over = listing["price"] - filters.max_price
        issues.append(f"{config.CURRENCY}{over} over budget")
    return issues


def format_candidate(candidate: dict) -> str:
    l = candidate["listing"]
    beds = "studio" if l["bedrooms"] == 0 else f"{l['bedrooms']} bedrooms"
    return (
        f"[{l['id']}] {l['title']}\n"
        f"District: {l['district']} | Rent: {config.CURRENCY}{l['price']}/month | "
        f"{beds}, {l['bathrooms']} bathroom(s), {l['size_sqm']} sqm, floor {l['floor']}\n"
        f"Features: {', '.join(feature_list(l)) or 'none'}\n"
        f"Issues: {'; '.join(candidate['issues']) or 'none'}\n"
        f"Description: {l['description']}"
    )


def format_area_notes(index: PropertyIndex, districts: set[str]) -> str:
    lines = []
    for d in sorted(districts):
        a = index.areas[d]
        lines.append(
            f"- {d}: {a['vibe']}. To City University: {a['to_university']}. "
            f"To the metro: {a['to_metro']}."
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The full pipeline
# ---------------------------------------------------------------------------
def build_pipeline(llm: BaseChatModel, index: PropertyIndex) -> Runnable:
    parse_chain = build_parse_chain(llm, index)
    area_chain = build_area_chain(llm, index)
    rank_chain = build_rank_chain(llm)

    # ---- Step 2: decide which districts to search ------------------------
    def named_districts(state: dict) -> list[str]:
        return clean_districts(state["filters"].districts, index.district_names)

    def use_named_districts(state: dict) -> dict:
        names = named_districts(state)
        return {"districts": names, "note": f"Searching the districts you named: {', '.join(names)}."}

    def resolve_landmark(state: dict, config: RunnableConfig) -> dict:
        near = state["filters"].near
        choice: AreaChoice = area_chain.invoke({"near": near}, config=config)
        names = clean_districts(choice.districts, index.district_names)
        if not names:
            return {"districts": [], "note": f"Couldn't tell which districts are near {near}, so searching the whole city."}
        return {"districts": names, "note": f"Near {near}: {', '.join(names)}. {choice.reasoning}"}

    area_step = RunnableBranch(
        (lambda s: bool(named_districts(s)), RunnableLambda(use_named_districts)),
        (lambda s: bool(s["filters"].near), RunnableLambda(resolve_landmark)),
        RunnableLambda(lambda s: {"districts": [], "note": "No area given, so searching the whole city."}),
    )

    # ---- Step 3: retrieve listings, relaxing the search if too few -------
    def find_candidates(state: dict) -> dict:
        filters: SearchFilters = state["filters"]
        districts: list[str] = state["areas"]["districts"]
        query = state["request"]

        # (districts to use, price multiplier, note to show if this attempt adds listings)
        attempts = [(districts, 1.0, None)]
        if districts:
            attempts.append(([], 1.0, f"Only a few matches in {' / '.join(districts)}, so other districts are included."))
        if filters.max_price is not None:
            pct = round((config.BUDGET_STRETCH - 1) * 100)
            attempts.append(([], config.BUDGET_STRETCH, f"Also showing places up to {pct}% over your budget."))

        found: dict[str, dict] = {}
        notes: list[str] = []
        for attempt_districts, price_factor, note in attempts:
            if len(found) >= config.MIN_RESULTS:
                break
            listings = index.search_listings(query, filters, attempt_districts, price_factor=price_factor)
            new = [l for l in listings if l["id"] not in found]
            if new and note:
                notes.append(note)
            for l in new:
                found[l["id"]] = l

        candidates = [
            {"listing": l, "issues": find_issues(l, filters, districts)}
            for l in list(found.values())[: config.TOP_K]
        ]
        return {"candidates": candidates, "notes": notes}

    # ---- Step 4: rank with the LLM (skipped when nothing was found) ------
    def rank_input(state: dict) -> dict:
        candidates = state["search"]["candidates"]
        return {
            "request": state["request"],
            "filters": state["filters"].model_dump_json(),
            "area_notes": format_area_notes(index, {c["listing"]["district"] for c in candidates}),
            "candidates": "\n\n".join(format_candidate(c) for c in candidates),
        }

    no_results = RankingResult(
        summary="Sorry, nothing matches that request, even after relaxing the area and budget. "
                "Try removing a requirement or raising the budget.",
        results=[],
    )
    rank_step = RunnableBranch(
        (lambda s: not s["search"]["candidates"], RunnableLambda(lambda s: no_results)),
        RunnableLambda(rank_input) | rank_chain,
    )

    # ---- Step 5: merge the ranking with the real data --------------------
    def assemble(state: dict) -> SearchResponse:
        by_id = {c["listing"]["id"]: c for c in state["search"]["candidates"]}
        results: list[PropertyResult] = []
        seen: set[str] = set()
        for ranked in state["ranking"].results:
            listing_id = ranked.listing_id.strip().strip("[]").upper()
            candidate = by_id.get(listing_id)
            if candidate is None or listing_id in seen:
                continue  # ignore ids the LLM invented or repeated
            seen.add(listing_id)
            l = candidate["listing"]
            score = ranked.match_score
            if candidate["issues"]:
                score = min(score, config.MAX_SCORE_WITH_ISSUES)
            results.append(PropertyResult(
                listing_id=l["id"], title=l["title"], district=l["district"],
                price=l["price"], bedrooms=l["bedrooms"], size_sqm=l["size_sqm"],
                match_score=score, issues=candidate["issues"],
                pros=ranked.pros, cons=ranked.cons, reason=ranked.reason,
            ))
        results.sort(key=lambda r: r.match_score, reverse=True)

        return SearchResponse(
            request=state["request"],
            filters=state["filters"],
            districts=state["areas"]["districts"],
            area_note=state["areas"]["note"],
            relax_notes=state["search"]["notes"],
            candidate_ids=list(by_id),
            summary=state["ranking"].summary,
            results=results,
        )

    # ---- Chain everything together ---------------------------------------
    return (
        RunnablePassthrough.assign(filters=parse_chain.with_config(run_name="1_parse_request"))
        | RunnablePassthrough.assign(areas=area_step.with_config(run_name="2_resolve_areas"))
        | RunnablePassthrough.assign(search=RunnableLambda(find_candidates).with_config(run_name="3_retrieve_listings"))
        | RunnablePassthrough.assign(ranking=rank_step.with_config(run_name="4_rank"))
        | RunnableLambda(assemble).with_config(run_name="5_assemble")
    )


def create_finder() -> tuple[Runnable, PropertyIndex]:
    """Build the index and pipeline with the models from config / .env."""
    index = PropertyIndex.from_files(config.get_embeddings())
    return build_pipeline(config.get_llm(), index), index
