"""Streamlit UI.  Run with:  streamlit run app.py"""

import streamlit as st

from property_finder import create_finder
from property_finder.config import CURRENCY

st.set_page_config(page_title="PropPilot (AI Property Finder)", page_icon="🏠", layout="wide")

EXAMPLES = [
    "2-bedroom near the university, under 800, with a balcony",
    "Quiet furnished place close to the metro, max 1000",
    "Pet-friendly 3 bedrooms in Green Hills with parking",
    "Cheapest studio or 1-bedroom anywhere, furnished",
]


@st.cache_resource(show_spinner="Loading models and building the vector index...")
def load_finder():
    return create_finder()


def use_example(text: str) -> None:
    st.session_state.request = text


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------
st.title("🏠 PropPilot (AI Property Finder)")
st.write("Describe the apartment you want in your own words.")

st.text_area("Your request", key="request", height=80, placeholder=EXAMPLES[0])

st.caption("Or try an example:")
for col, example in zip(st.columns(len(EXAMPLES)), EXAMPLES):
    col.button(example, on_click=use_example, args=(example,))

if st.button("🔎 Find apartments", type="primary"):
    request = st.session_state.get("request", "").strip()
    if not request:
        st.warning("Type what you're looking for first.")
        st.stop()
    try:
        finder, _ = load_finder()
    except Exception as e:  # missing API key, missing package, etc.
        st.error(f"Couldn't load the models. Check your .env file and installed packages.\n\n`{e}`")
        st.stop()
    with st.spinner("Searching and ranking..."):
        try:
            st.session_state.result = finder.invoke({"request": request})
        except Exception as e:
            st.error(f"Something went wrong while searching.\n\n`{e}`")
            st.stop()

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
res = st.session_state.get("result")
if res:
    st.subheader("Results")
    st.info(res.summary)
    for note in [res.area_note, *res.relax_notes]:
        st.caption(f"ℹ️ {note}")

    for r in res.results:
        with st.container(border=True):
            left, right = st.columns([3, 1])
            beds = "Studio" if r.bedrooms == 0 else f"{r.bedrooms} bedrooms"
            left.markdown(f"#### {r.title}")
            left.caption(f"{r.district} · {beds} · {r.size_sqm} sqm · ID {r.listing_id}")
            right.metric("Rent / month", f"{CURRENCY}{r.price:,}")

            st.progress(r.match_score / 100, text=f"Match score: {r.match_score}/100")
            if r.issues:
                st.warning(" · ".join(r.issues))

            pros, cons = st.columns(2)
            pros.markdown("**👍 Pros**\n\n" + "\n".join(f"- {p}" for p in r.pros))
            cons.markdown("**👎 Cons**\n\n" + "\n".join(f"- {c}" for c in r.cons))
            st.caption(r.reason)

    with st.expander("🔍 Behind the scenes: what each step produced"):
        st.markdown("**Step 1 · Parsed filters** (Chain + PydanticOutputParser)")
        st.json(res.filters.model_dump())
        st.markdown("**Step 2 · Districts to search** (RAG over area descriptions)")
        st.write(", ".join(res.districts) or "Whole city")
        st.caption(res.area_note)
        st.markdown("**Step 3 · Retrieved candidates** (RAG over listings with metadata filters)")
        st.write(", ".join(res.candidate_ids) or "None")
        for note in res.relax_notes:
            st.caption(note)
        st.markdown("**Steps 4-5 · Final ranked output**")
        st.json([r.model_dump() for r in res.results])
