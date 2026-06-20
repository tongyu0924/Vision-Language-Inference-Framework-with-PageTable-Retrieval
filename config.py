import re
import json
import random
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from openai import OpenAI

API_KEY = " "
client = OpenAI(api_key=API_KEY)

CFG = {
    "vision_model": "gpt-4o-mini",
    "index_model": "gpt-4o-mini",
    "prune_model": "gpt-4o-mini",
    "answer_model": "gpt-4o-mini",

    "corpus_docs": 600,
    "index_max_workers": 20,

    "prune_batch_size": 25,
    "prune_keep_per_batch": 8,
    "final_evidence_top_k": 14,

    "max_plan_steps": 2,

    "split": "validation",
    "n_eval": 50,

    "output_csv": "caip_hashindex_viquae_results.csv",
    "output_json": "caip_hashindex_viquae_results.json",
}

random.seed(42)
np.random.seed(42)

WIKI_USER_AGENT = "CAIPHashIndexBot/2.0 multimodal retrieval research"

INVALID_ANSWERS = {
    "", "unknown", "none", "null", "n/a",
    "section", "references", "reference", "early", "career", "life",
    "history", "background", "details", "overview", "introduction",
    "american", "british", "french", "german", "italian", "english",
    "person", "man", "woman", "people", "together", "although",
    "most", "wood", "lineage", "revolution", "scottish", "united states",
}

STOPWORDS = {
    "what", "which", "who", "where", "when", "why", "how",
    "was", "were", "is", "are", "the", "a", "an", "of", "in",
    "on", "to", "for", "by", "with", "this", "that", "his",
    "her", "he", "she", "it", "its", "did", "does", "do",
    "before", "after", "from", "and", "or", "as", "at",
}


def validate_api_key():
    if not API_KEY or not API_KEY.strip():
        raise RuntimeError(
            "API_KEY is empty. Set API_KEY = 'sk-...' near the top of this script."
        )
    try:
        client.responses.create(
            model=CFG["index_model"],
            input=[{"role": "user", "content": [{"type": "input_text", "text": "ping"}]}],
            max_output_tokens=16,
        )
        print("✓ API key validated — OpenAI connection OK.")
    except Exception as e:
        msg = str(e)
        if "401" in msg or "invalid_api_key" in msg.lower():
            raise RuntimeError(f"API_KEY is invalid (401). Detail: {msg}")
        if "rate_limit_exceeded" in msg.lower() and "requests per day" in msg.lower():
            raise RuntimeError(
                f"Daily request limit (RPD) exhausted — resets at midnight UTC. Detail: {msg}"
            )
        if "insufficient_quota" in msg.lower():
            raise RuntimeError(f"API key has no remaining credit. Detail: {msg}")
        if "429" in msg:
            raise RuntimeError(f"Rate limited (429) — wait a moment and retry. Detail: {msg}")
        raise RuntimeError(f"Could not reach OpenAI API. Original error: {msg}")


@dataclass
class Document:
    doc_id: int
    title: str
    text: str
    source: str = ""


@dataclass
class PageEntry:
    doc_id: int
    title: str
    summary: str
    keywords: List[str]
    text: str


@dataclass
class SubjectIdentity:
    primary_title: str
    alt_titles: List[str]
    doc_id: Optional[int]
    raw_description: str


@dataclass
class RetrievalTrace:
    step: int
    queries: List[str]
    retrieved_doc_ids: List[int]
    matched_evidence: List[Dict[str, Any]]


@dataclass
class CAIPResult:
    answer: str
    confidence: float
    context_state: Dict[str, Any]
    verified_evidence: List[Dict[str, Any]]
    trace: List[RetrievalTrace]
    final_reasoning: str
    raw_final: Dict[str, Any]
    subject: SubjectIdentity


def safe_json_parse(text: str) -> Dict[str, Any]:
    text = str(text).strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        return json.loads(m.group(0))
    raise ValueError(f"Could not parse JSON:\n{text[:1200]}")


def normalize_answer(s: str) -> str:
    s = str(s).lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^a-z0-9\u0370-\u03ff\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_retrieval_text(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9\u0370-\u03ff\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def clean_query(q: str) -> str:
    q = str(q)
    q = re.sub(r"[^A-Za-z0-9\u0370-\u03ff\s\-]", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q


def is_invalid_answer(ans: str) -> bool:
    a = normalize_answer(ans)
    if a in INVALID_ANSWERS:
        return True
    if len(a) <= 1:
        return True
    if len(a.split()) > 12:
        return True
    return False


def extract_keywords(text: str, min_len: int = 3) -> List[str]:
    text = normalize_retrieval_text(text)
    return [t for t in text.split() if len(t) >= min_len and t not in STOPWORDS]


def doc_key(title, text):
    return (str(title).strip().lower(), str(text)[:160].strip().lower())
