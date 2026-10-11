"""Settings and model factories.

Everything is read from environment variables (or a .env file), so you can
switch LLM provider without touching the code.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LISTINGS_FILE = DATA_DIR / "listings.json"
AREAS_FILE = DATA_DIR / "areas.json"

# "provider:model-name". Examples are in .env.example.
LLM_MODEL = os.getenv("LLM_MODEL", "openai:gpt-4.1-mini")
EMBEDDINGS_MODEL = os.getenv(
    "EMBEDDINGS_MODEL", "huggingface:sentence-transformers/all-MiniLM-L6-v2"
)

CURRENCY = os.getenv("CURRENCY", "$")

# Retrieval settings
TOP_K = 6                 # max listings sent to the ranking step
MIN_RESULTS = 3           # below this, the search is relaxed step by step
BUDGET_STRETCH = 1.15     # last relaxation step: allow up to 15% over budget
MAX_SCORE_WITH_ISSUES = 69  # a listing that breaks a requirement can't score higher


def get_llm():
    """Chat model used by every chain. temperature=0 keeps the JSON output stable."""
    from langchain.chat_models import init_chat_model

    return init_chat_model(LLM_MODEL, temperature=0)


def get_embeddings():
    """Embedding model used by both vector stores."""
    from langchain.embeddings import init_embeddings

    return init_embeddings(EMBEDDINGS_MODEL)
