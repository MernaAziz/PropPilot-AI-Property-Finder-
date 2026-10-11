"""Prompt templates for the three LLM steps."""

from langchain_core.prompts import ChatPromptTemplate

# ---------------------------------------------------------------------------
# Step 1: free text -> SearchFilters
# ---------------------------------------------------------------------------
PARSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You turn apartment search requests into search filters for a rental website.\n\n"
     "Known districts: {district_names}\n"
     "Allowed must-have features: balcony, furnished, parking, elevator, pets_allowed\n\n"
     "Rules:\n"
     "- Only put a district in `districts` if the user names that district.\n"
     "- Places like 'the university', 'the metro', 'the park' go in `near`, not in `districts`.\n"
     "- 'under 800', 'max 800', 'budget 800', 'up to 800' all mean max_price = 800.\n"
     "- '2-bedroom' or '2BR' means min_bedrooms = 2. 'studio' means min_bedrooms = 0.\n"
     "- A feature the user requires (with, must have, need) goes in `must_have`.\n"
     "- Wishes (quiet, modern, cheap, sunny, 'would be nice') go in `preferences`.\n\n"
     "{format_instructions}"),
    ("human", "{request}"),
])

# ---------------------------------------------------------------------------
# Step 2: landmark -> which districts are near it (uses retrieved area docs)
# ---------------------------------------------------------------------------
AREA_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You decide which districts of a city count as close to a place.\n"
     "Use ONLY the district descriptions below. Choose districts within about "
     "15 minutes of the place on foot or by public transport, closest first, at most 3. "
     "Copy district names exactly.\n\n"
     "District descriptions:\n{area_context}\n\n"
     "{format_instructions}"),
    ("human", "Place: {near}"),
])

# ---------------------------------------------------------------------------
# Step 4: candidates -> ranked list with pros/cons
# ---------------------------------------------------------------------------
RANK_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a friendly rental assistant. Rank the candidate apartments for the user.\n\n"
     "Scoring guide:\n"
     "- 90-100: meets every requirement and most wishes\n"
     "- 70-89: meets every requirement but misses some wishes\n"
     "- 40-69: breaks a requirement (see each candidate's Issues line)\n"
     "- below 40: poor fit\n\n"
     "Rules:\n"
     "- Use ONLY facts from the candidates and area notes. Never invent features.\n"
     "- Rank every candidate and use its exact id.\n"
     "- Give 1-3 short pros and 1-3 short cons for each one.\n"
     "- The summary is 1-2 sentences to the user. If some results break a requirement, say so.\n\n"
     "{format_instructions}"),
    ("human",
     "User request: {request}\n\n"
     "Parsed filters: {filters}\n\n"
     "Area notes:\n{area_notes}\n\n"
     "Candidates:\n{candidates}"),
])
