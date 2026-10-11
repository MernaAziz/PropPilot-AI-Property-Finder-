# 🏠 AI Property Finder

Find apartments by describing what you want in plain language.

> **"2-bedroom near the university, under 800, with a balcony"**
>
> → 1. Bright 2BR with sunny balcony near campus · University District · $750 · **92/100**
> → 2. Historic 2BR with wrought-iron balcony · Old Town · $640 · **84/100**
> → 3. Budget 2BR near the train station · Eastgate · $450 · **69/100** ⚠ outside the university area

Each result comes with a match score, pros, cons and a one-line reason.

---

## 1. Which technique is used where

| Technique | Where it's used |
|---|---|
| **Chaining (LCEL)** | 5 steps connected with `RunnablePassthrough.assign`, `RunnableBranch` and `prompt \| llm \| parser` |
| **Output Parsers** | `PydanticOutputParser` in 3 places: request → `SearchFilters`, landmark → `AreaChoice`, candidates → `RankingResult` |
| **RAG** | 2 Chroma vector stores: **listings** (semantic search + metadata filters) and **area descriptions** (to understand "near the university") |

---

## 2. How it works, step by step

```
User request (free text)
   │
   ▼
Step 1  PARSE ─────────── prompt | llm | PydanticOutputParser(SearchFilters)
   │    {"max_price": 800, "min_bedrooms": 2, "near": "the university", "must_have": ["balcony"]}
   ▼
Step 2  FIND AREAS ────── retrieve area docs (RAG) → prompt | llm | PydanticOutputParser(AreaChoice)
   │    ["University District", "Old Town"]
   ▼
Step 3  RETRIEVE ──────── Chroma similarity search + metadata filter
   │    price ≤ 800 AND bedrooms ≥ 2 AND balcony = true AND district IN [...]
   │    too few results? → drop the district → then allow +15% budget
   │    [L01, L07, L17, L22, L23]
   ▼
Step 4  RANK ──────────── prompt | llm | PydanticOutputParser(RankingResult)
   │    score + pros + cons + reason for each candidate
   ▼
Step 5  ASSEMBLE ──────── plain Python: merge LLM output with the real data
        → SearchResponse
```

### Step 1: Parse the request (Chain + Output Parser)
The LLM turns free text into a `SearchFilters` object. The prompt includes the parser's
`get_format_instructions()`, so the model knows the exact JSON shape.

A Pydantic `model_validator` cleans up the LLM's answer before validation:
`"lift"` becomes `"elevator"`, `"pets"` becomes `"pets_allowed"`, and anything that isn't a
real feature (like `"quiet"`) moves to `preferences` instead of crashing the parser.

### Step 2: Understand "near X" (RAG + Chain + Output Parser)
"Near the university" isn't a district, so we can't filter on it directly. Instead:
1. Search the **area descriptions** vector store for the landmark.
2. Give the retrieved descriptions to the LLM and ask which districts are within ~15 minutes.
3. Parse the answer into `AreaChoice`.

A `RunnableBranch` picks the path: if the user **named** a district, use it directly (no LLM call);
if they mentioned a **landmark**, run the area chain; otherwise search the **whole city**.

### Step 3: Retrieve listings (RAG with metadata filters)
Each listing is stored as a `Document`:
- `page_content`: a text description (this is what gets embedded)
- `metadata`: price, bedrooms, district, balcony, furnished... (this is what we filter on)

Hard requirements become a Chroma `where` filter, and the similarity search ranks what's left
by how close it is to the request's meaning. If fewer than 3 listings come back, the search
**relaxes** step by step (drop the district, then allow 15% over budget), and the UI tells the user.

### Step 4: Rank (Chain + Output Parser)
The candidates, the area notes and the parsed filters go to the LLM, which returns a `RankingResult`:
a summary plus a score, pros, cons and reason for each listing.

### Step 5: Assemble (guardrails in plain Python)
We don't trust the LLM blindly:
- **Prices, titles and districts come from the data**, never from the LLM.
- **IDs the LLM invents** (not in the candidates) are dropped.
- **Issues** ("$40 over budget", "outside University District") are computed in code, and a listing
  with issues can't score above 69, whatever the LLM said.

---

## 3. Project structure

```
property-finder/
├── app.py                    Streamlit UI
├── cli.py                    Command-line version
├── data/
│   ├── listings.json         24 sample apartments
│   └── areas.json            6 district descriptions
├── property_finder/
│   ├── config.py             Settings + LLM/embeddings factory (reads .env)
│   ├── schemas.py            Pydantic models for the parsers and the final output
│   ├── prompts.py            The 3 prompt templates
│   ├── rag.py                Documents, Chroma vector stores, metadata filter
│   └── chains.py             The LCEL pipeline (steps 1-5)
├── tests/test_pipeline.py    Offline tests with a fake LLM (no API key needed)
├── requirements.txt
└── .env.example
```

---

## 4. Setup and run

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env               # Windows: copy .env.example .env
# open .env, pick a provider and paste your API key
```

Using Groq, Gemini or Ollama instead of OpenAI? Install its package too
(`pip install langchain-groq`, `langchain-google-genai` or `langchain-ollama`) and set `LLM_MODEL` in `.env`.

```bash
streamlit run app.py                                        # web app
python cli.py -v "quiet furnished place near the metro"     # terminal, -v shows every step
pytest -q                                                   # tests, no API key needed
```

The first run downloads the free embedding model (~90 MB).

---

## 5. Try these requests

| Request | What it shows |
|---|---|
| 2-bedroom near the university, under 800, with a balcony | Landmark → areas, then relaxation to other districts |
| 1-bedroom in Riverside under 900, pet friendly | Named district (area LLM step skipped) |
| Quiet furnished place close to the metro, max 1000 | "quiet" becomes a soft preference used in ranking |
| 5 bedrooms under 300 | No results: the ranking step is skipped |

---

## 6. Ideas to extend it

- **Use your own data**: scrape or export real listings into `listings.json` (same fields).
- **Persist the index**: pass `persist_directory="chroma_db"` to `Chroma` so embeddings are built once.
- **Self-repairing parser**: on `OutputParserException`, send the error back to the LLM and ask it to fix the JSON.
- **Chat follow-ups**: "cheaper ones?" or "only furnished" using conversation memory.
- **Compare mode**: pick 2-3 results and generate a side-by-side comparison.
- **Map view**: add coordinates to each listing and show results with `st.map`.
- **Deploy**: push to GitHub and deploy free on Streamlit Community Cloud or Hugging Face Spaces
  (put your API key in the platform's secrets, not in the code).
