import re
import random
from typing import List, Dict, Any, Optional, Tuple

from config import Document, PageEntry, extract_keywords, normalize_retrieval_text, safe_json_parse, CFG
from llm_io import call_openai_text
import state

RELATION_DIRECTION_PATTERNS = {
    "predecessor": [r"\bbefore\b", r"\bprevious(ly)?\b", r"\bformer(ly)?\b",
                    r"\bpredecessor\b", r"\breplaced by\b", r"\bsucceeded by\b",
                    r"\bcame before\b", r"\bprior to\b"],
    "successor": [r"\bafter\b", r"\bnext\b", r"\bsuccessor\b", r"\breplaces\b",
                 r"\bfollow(ed|ing)\b", r"\bsucceeds\b", r"\bcame after\b"],
}

PREDECESSOR_TEXT_SIGNALS = [r"\breplaced by\b", r"\bsucceeded by\b", r"\bformer(ly)?\b",
                           r"\bprevious(ly)?\b", r"\buntil\b"]
SUCCESSOR_TEXT_SIGNALS = [r"\bcurrent(ly)?\b", r"\bnow\b", r"\btoday\b", r"\bsince\b"]


def detect_relation_direction(question: str) -> Optional[str]:
    ql = question.lower()
    for direction, patterns in RELATION_DIRECTION_PATTERNS.items():
        if any(re.search(p, ql) for p in patterns):
            return direction
    return None


def relation_direction_score(direction: Optional[str], doc_text: str) -> float:
    if direction is None:
        return 0.0
    text_n = normalize_retrieval_text(doc_text[:1500])
    signals = PREDECESSOR_TEXT_SIGNALS if direction == "predecessor" else SUCCESSOR_TEXT_SIGNALS
    hits = sum(1 for p in signals if re.search(p, text_n))
    return min(hits * 0.15, 0.45)


def format_page_batch(entries: List[PageEntry]) -> str:
    lines = []
    for e in entries:
        kw = ", ".join(e.keywords)
        lines.append(f"[{e.doc_id}] {e.title} — {e.summary} (keywords: {kw})")
    return "\n".join(lines)


def agent_prune_batch(queries: List[str], context_state: Dict[str, Any],
                      entries: List[PageEntry], keep_top_k: int,
                      question: str = "") -> List[int]:
    batch_text = format_page_batch(entries)
    query_text = "; ".join(queries[:6])
    direction = detect_relation_direction(question) if question else None
    direction_hint = ""
    if direction == "predecessor":
        direction_hint = ("\nNOTE: This question asks about what came BEFORE/"
                          "PREVIOUSLY — prefer documents about the PREDECESSOR "
                          "entity, not the thing currently shown in the image.")
    elif direction == "successor":
        direction_hint = ("\nNOTE: This question asks about what comes AFTER/"
                          "NEXT — prefer documents about the SUCCESSOR entity.")

    prompt = f"""
You are a relevance-judging search agent operating over a hash-indexed
document table (NOT a vector search — judge relevance by reasoning).

Search queries: {query_text}
Target relation: {context_state.get('target_relation')}
Expected answer type: {context_state.get('expected_answer_type')}{direction_hint}

Indexed document summaries (each prefixed with its ID):
{batch_text}

Select the IDs of up to {keep_top_k} entries most likely to contain the
answer. Consider topical/semantic relevance, not just keyword overlap.
Even if nothing is a strong match, select your best {min(3, keep_top_k)}
guesses rather than returning an empty list — partial relevance is still
useful downstream.

Return ONLY valid JSON:
{{"selected_ids": [id1, id2, ...]}}
"""
    def _keyword_fallback() -> List[int]:
        scored = []
        q_tokens = set()
        for q in queries:
            q_tokens |= set(extract_keywords(q))
        for e in entries:
            text_n = normalize_retrieval_text(f"{e.title} {e.summary} {' '.join(e.keywords)}")
            overlap = sum(1 for t in q_tokens if t in text_n)
            scored.append((e.doc_id, overlap))
        scored.sort(key=lambda x: -x[1])
        return [doc_id for doc_id, _ in scored[:keep_top_k]]

    try:
        out = call_openai_text(CFG["prune_model"], prompt, temperature=0.0, max_tokens=150)
        data = safe_json_parse(out)
        ids = [int(i) for i in data.get("selected_ids", [])]
        valid_ids = {e.doc_id for e in entries}
        ids = [i for i in ids if i in valid_ids][:keep_top_k]
        if not ids:
            ids = _keyword_fallback()
        return ids
    except Exception:
        return _keyword_fallback()


def agent_led_pruning(queries: List[str], context_state: Dict[str, Any],
                      page_table: Dict[int, PageEntry],
                      force_include_ids: Optional[List[int]] = None,
                      question: str = "") -> List[Tuple[Document, float, str]]:
    all_ids = list(page_table.keys())
    random.shuffle(all_ids)
    batch_size = CFG["prune_batch_size"]

    shortlist_with_rank: Dict[int, int] = {}
    for i in range(0, len(all_ids), batch_size):
        batch_ids = all_ids[i:i + batch_size]
        batch_entries = [page_table[did] for did in batch_ids]
        kept = agent_prune_batch(queries, context_state, batch_entries,
                                 keep_top_k=CFG["prune_keep_per_batch"], question=question)
        for rank, did in enumerate(kept):
            if did not in shortlist_with_rank or rank < shortlist_with_rank[did]:
                shortlist_with_rank[did] = rank

    for did in (force_include_ids or []):
        if did in page_table:
            shortlist_with_rank[did] = -1

    shortlist = sorted(shortlist_with_rank.keys(), key=lambda d: shortlist_with_rank[d])

    if len(shortlist) > CFG["final_evidence_top_k"]:
        forced = [d for d in shortlist if shortlist_with_rank[d] == -1]
        rest = [d for d in shortlist if shortlist_with_rank[d] != -1]
        if rest:
            entries = [page_table[did] for did in rest]
            keep_n = max(CFG["final_evidence_top_k"] - len(forced), 0)
            if keep_n > 0:
                rest = agent_prune_batch(queries, context_state, entries,
                                         keep_top_k=keep_n, question=question)
            else:
                rest = []
        shortlist = forced + rest

    results = []
    for did in shortlist[:CFG["final_evidence_top_k"]]:
        doc = state.get_doc_by_id(did)
        if doc is None:
            continue
        rank = shortlist_with_rank.get(did, CFG["prune_keep_per_batch"])
        pseudo_score = max(CFG["prune_keep_per_batch"] - max(rank, 0), 1) / CFG["prune_keep_per_batch"]
        best_query = queries[0] if queries else ""
        results.append((doc, pseudo_score, best_query))

    return results


def get_context_terms(context_state: Dict[str, Any], question: str) -> Dict[str, List[str]]:
    relation_terms = extract_keywords(context_state.get("target_relation", ""))
    question_terms = extract_keywords(question)
    subject_terms = extract_keywords(context_state.get("subject_title", ""))
    return {"relation_terms": list(dict.fromkeys(relation_terms)),
           "question_terms": list(dict.fromkeys(question_terms)),
           "subject_terms": list(dict.fromkeys(subject_terms))}


def overlap_score(terms: List[str], text: str) -> float:
    if not terms:
        return 0.0
    text_n = normalize_retrieval_text(text)
    return sum(1 for t in terms if t in text_n) / max(len(terms), 1)


def expected_answer_type_score(expected_type: str, text: str) -> float:
    et = str(expected_type).lower()
    t = str(text)
    if et == "date":
        return 1.0 if re.search(r"\b(1[0-9]{3}|20[0-9]{2})\b", t) else 0.0
    if et == "number":
        return 1.0 if re.search(r"\b\d+\b", t) else 0.0
    if et in {"place", "person", "organization", "object", "event", "short_phrase"}:
        return 0.6 if re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", t) else 0.2
    return 0.2


def score_document(question: str, context_state: Dict[str, Any],
                   doc: Document, retrieval_score: float, query: str) -> Dict[str, Any]:
    terms = get_context_terms(context_state, question)
    title_text = f"{doc.title} {doc.text[:2200]}"

    relation_match = overlap_score(terms["relation_terms"] + terms["question_terms"], title_text)
    type_match = expected_answer_type_score(context_state.get("expected_answer_type", ""), title_text)
    subject_match = overlap_score(terms["subject_terms"], title_text)
    direction = detect_relation_direction(question)
    direction_match = relation_direction_score(direction, doc.text)

    score = (0.30 * relation_match + 0.20 * type_match
            + 0.20 * subject_match + 0.20 * direction_match
            + 0.10 * min(max(retrieval_score, 0.0), 1.0))

    return {
        "doc_id": doc.doc_id, "title": doc.title, "doc_text": doc.text[:2400],
        "retrieved_by_query": query, "retrieval_score": float(retrieval_score),
        "relation_match": round(relation_match, 4), "type_match": round(type_match, 4),
        "subject_match": round(subject_match, 4), "direction_match": round(direction_match, 4),
        "within_tier_score": round(score, 4),
    }


def build_tiered_evidence(question: str, context_state: Dict[str, Any],
                          subject, related_entity_ids: List[int],
                          pruned_candidates: List[Tuple[Document, float, str]]) -> Dict[str, Any]:
    tier0, tier1, tier2 = [], [], []
    seen_ids = set()

    if subject.doc_id is not None:
        doc = state.get_doc_by_id(subject.doc_id)
        if doc is not None:
            e = score_document(question, context_state, doc, 1.0, subject.primary_title)
            e["tier"] = 0
            tier0.append(e)
            seen_ids.add(doc.doc_id)

    for did in related_entity_ids:
        if did in seen_ids:
            continue
        doc = state.get_doc_by_id(did)
        if doc is None:
            continue
        e = score_document(question, context_state, doc, 0.9, context_state.get("related_entity_guess", ""))
        e["tier"] = 1
        tier1.append(e)
        seen_ids.add(did)

    for doc, score, query in pruned_candidates:
        if doc.doc_id in seen_ids:
            continue
        e = score_document(question, context_state, doc, score, query)
        e["tier"] = 2
        tier2.append(e)
        seen_ids.add(doc.doc_id)

    tier0.sort(key=lambda e: -e["within_tier_score"])
    tier1.sort(key=lambda e: -e["within_tier_score"])
    tier2.sort(key=lambda e: -e["within_tier_score"])

    return {"tier0": tier0, "tier1": tier1[:6], "tier2": tier2[:8]}
