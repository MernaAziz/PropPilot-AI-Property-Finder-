"""Offline tests: a fake LLM and fake embeddings, so no API key is needed.

Run with:  pytest -q
"""

import json

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from property_finder import PropertyIndex, SearchFilters, build_pipeline
from property_finder.rag import build_where


@pytest.fixture(scope="module")
def index():
    return PropertyIndex.from_files(DeterministicFakeEmbedding(size=32))


def run(index, request, llm_replies):
    llm = FakeListChatModel(responses=llm_replies)
    return build_pipeline(llm, index).invoke({"request": request})


# ---------------------------------------------------------------------------
# Output parser / schema behaviour
# ---------------------------------------------------------------------------
def test_feature_synonyms_and_unknown_features():
    f = SearchFilters.model_validate({"must_have": ["Balcony", "lift", "pets", "quiet", "balcony"]})
    assert f.must_have == ["balcony", "elevator", "pets_allowed"]
    assert f.preferences == ["quiet"]


def test_build_where_shapes():
    assert build_where(SearchFilters(), []) is None
    assert build_where(SearchFilters(max_price=800), []) == {"price": {"$lte": 800}}
    where = build_where(SearchFilters(max_price=800, must_have=["balcony"]), ["Old Town"], 1.15)
    assert where == {"$and": [
        {"price": {"$lte": 920}},
        {"district": {"$in": ["Old Town"]}},
        {"balcony": {"$eq": True}},
    ]}


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
def test_landmark_search_with_relaxation(index):
    parse_reply = json.dumps({
        "max_price": 800, "min_bedrooms": 2, "districts": [],
        "near": "the university", "must_have": ["Balcony"], "preferences": ["quiet"],
    })
    # Fenced JSON and a lowercase district name, like real models sometimes return
    area_reply = '```json\n{"districts": ["University District", "old town"], "reasoning": "Both are a short walk from campus."}\n```'
    rank_reply = json.dumps({
        "summary": "Two places fit perfectly; the rest are outside the university area.",
        "results": [
            {"listing_id": "L07", "match_score": 85, "pros": ["Cheap"], "cons": ["Weekend noise"], "reason": "Good fit."},
            {"listing_id": "L01", "match_score": 90, "pros": ["5 min to campus"], "cons": ["No elevator"], "reason": "Best fit."},
            {"listing_id": "L22", "match_score": 85, "pros": ["Very cheap"], "cons": ["Far"], "reason": "Cheap but far."},
            {"listing_id": "L17", "match_score": 60, "pros": ["Near metro"], "cons": ["Older"], "reason": "OK."},
            {"listing_id": "L99", "match_score": 99, "pros": ["Invented"], "cons": [], "reason": "Hallucinated."},
        ],
    })

    res = run(index, "2-bedroom near the university, under 800, with a balcony", [parse_reply, area_reply, rank_reply])

    # Step 1: parser + validator
    assert res.filters.max_price == 800 and res.filters.min_bedrooms == 2
    assert res.filters.must_have == ["balcony"]
    # Step 2: area chain output cleaned to real district names
    assert res.districts == ["University District", "Old Town"]
    # Step 3: only 2 exact matches, so other districts were added
    assert set(res.candidate_ids) == {"L01", "L07", "L17", "L22", "L23"}
    assert len(res.relax_notes) == 1
    # Step 5: invented id dropped, sorted by score, price from the data
    ids = [r.listing_id for r in res.results]
    assert ids == ["L01", "L07", "L22", "L17"]
    l22 = res.results[2]
    assert l22.match_score == 69, "listings with issues are capped"
    assert l22.price == 450
    assert l22.issues and l22.issues[0].startswith("Outside")


def test_named_district_skips_area_llm_call(index):
    parse_reply = json.dumps({"max_price": 900, "min_bedrooms": 1, "districts": ["riverside"], "must_have": ["pets"]})
    rank_reply = json.dumps({"summary": "One match in Riverside.", "results": [
        {"listing_id": "L13", "match_score": 88, "pros": ["Garden"], "cons": ["1 bedroom"], "reason": "Fits."},
    ]})
    # Only two replies: if the area chain ran, the rank step would get the wrong reply and fail
    res = run(index, "1-bedroom in Riverside under 900, pet friendly", [parse_reply, rank_reply])

    assert res.districts == ["Riverside"]
    assert res.area_note.startswith("Searching the districts you named")
    assert res.results[0].listing_id == "L13"
    assert all(c in res.candidate_ids for c in ["L13"])


def test_no_results_skips_ranking(index):
    parse_reply = json.dumps({"max_price": 300, "min_bedrooms": 5})
    # The second reply is not valid JSON: the test fails if the rank chain is called
    res = run(index, "5 bedrooms under 300", [parse_reply, "not json"])

    assert res.candidate_ids == []
    assert res.results == []
    assert res.summary.startswith("Sorry")
