import json
from typing import Dict, Any, List

from config import normalize_answer, is_invalid_answer, safe_json_parse, STOPWORDS, CFG
from llm_io import call_openai_text


def answer_supported_by_evidence(answer: str, all_evidence: List[Dict[str, Any]]) -> bool:
    ans = normalize_answer(answer)
    if not ans or ans == "unknown" or is_invalid_answer(ans):
        return False
    if not all_evidence:
        return False
    combined = " ".join(str(e.get("doc_text", "")) + " " + str(e.get("title", ""))
                        for e in all_evidence)
    combined_n = normalize_answer(combined)
    if ans in combined_n:
        return True
    ans_tokens = [t for t in ans.split() if len(t) >= 3 and t not in STOPWORDS]
    if ans_tokens:
        hits = sum(1 for t in ans_tokens if t in combined_n)
        if hits / len(ans_tokens) >= 0.7:
            return True
    return False


def generate_final_answer(question: str, context_state: Dict[str, Any],
                          tiered_evidence: Dict[str, Any]) -> Dict[str, Any]:
    all_evidence = tiered_evidence["tier0"] + tiered_evidence["tier1"] + tiered_evidence["tier2"]
    if not all_evidence:
        return {"answer": "unknown", "confidence": 0.0,
               "reasoning": "No retrieved evidence.", "is_answer_valid": False}

    def _block(e):
        return {"doc_id": e["doc_id"], "title": e["title"], "text": e["doc_text"][:1800]}

    tier0_blocks = [_block(e) for e in tiered_evidence["tier0"]]
    tier1_blocks = [_block(e) for e in tiered_evidence["tier1"]]
    tier2_blocks = [_block(e) for e in tiered_evidence["tier2"]]

    prompt = f"""
You are an evidence-only answer extractor for a knowledge-based VQA system.

Question: {question}
Subject of the image (already confirmed): {context_state.get("subject_title")}
Expected answer type: {context_state.get("expected_answer_type")}

TIER 0 — The subject's OWN article (HIGHEST TRUST — this document is
guaranteed to be about the correct entity in the image; if it answers the
question, use it and ignore conflicting info elsewhere):
{json.dumps(tier0_blocks, ensure_ascii=False, indent=2) if tier0_blocks else "(none — subject article could not be fetched)"}

TIER 1 — Specifically nominated related-entity articles (HIGH TRUST):
{json.dumps(tier1_blocks, ensure_ascii=False, indent=2) if tier1_blocks else "(none)"}

TIER 2 — General search results (LOWER TRUST — topically related but NOT
confirmed to be about the exact entity in the image; use only to fill
gaps Tier 0/1 don't cover, NEVER to override them):
{json.dumps(tier2_blocks, ensure_ascii=False, indent=2) if tier2_blocks else "(none)"}

Instructions:
1. Always check Tier 0 first. If it answers the question, use it.
2. Check Tier 1 next, only if Tier 0 doesn't answer the question.
3. Use Tier 2 only as a last resort, and NEVER let it override a Tier 0/1
   answer — Tier 2 documents may be about a different, unrelated entity
   that merely shares similar keywords (e.g. a more famous person/place
   of the same type).
4. If multiple historical predecessors/candidates are listed in evidence,
   pick the one chronologically closest / most directly relevant using
   any dates given — not an earlier-still one.
5. You may combine TWO facts from evidence (e.g. one document says who was
   named successor, another confirms they were later expelled) as long as
   every fact you use is explicitly present in the evidence shown.
6. Do NOT use outside/world knowledge — every fact must trace back to the
   evidence shown above.
7. Give the shortest, most specific correct answer (a name/date/place/
   phrase) — never a generic word like Section, References, Career, Life.
8. Before returning "unknown", check ALL evidence for partial matches or
   facts that combine to answer the question.

Return ONLY valid JSON:
{{"answer": "short answer span or unknown", "confidence": 0.0,
 "reasoning": "which tier/doc the answer came from", "is_answer_valid": true}}
"""
    out = call_openai_text(CFG["answer_model"], prompt, temperature=0.0, max_tokens=300)
    final = safe_json_parse(out)
    ans = final.get("answer", "")

    if is_invalid_answer(ans):
        return {"answer": "unknown", "confidence": 0.0,
               "reasoning": "Invalid or generic answer.", "is_answer_valid": False}
    if ans != "unknown" and not answer_supported_by_evidence(ans, all_evidence):
        return {"answer": "unknown", "confidence": 0.0,
               "reasoning": "Answer not supported by evidence.", "is_answer_valid": False}
    return final


def detect_viquae_fields(first: Dict[str, Any]):
    img_key = next((k for k in ["image", "img", "picture"] if k in first), None)
    q_key = next((k for k in ["input", "question", "query"] if k in first), None)
    a_key = next((k for k in ["output", "answer", "answers", "original_answer"] if k in first), None)
    if not (img_key and q_key and a_key):
        raise ValueError(f"Could not detect fields: {list(first.keys())}")
    return img_key, q_key, a_key


def parse_viquae_answers(output_field) -> List[str]:
    if isinstance(output_field, dict):
        answers = output_field.get("answer", [])
        orig = output_field.get("original_answer")
        out = list(answers) if isinstance(answers, list) else [answers]
        if orig:
            out.append(orig)
        return [str(a) for a in out if a]
    if isinstance(output_field, list):
        return [str(a) for a in output_field if a]
    if isinstance(output_field, str):
        return [output_field]
    return []


def soft_match(pred: str, golds: List[str]) -> bool:
    pred_n = normalize_answer(pred)
    if not pred_n or pred_n == "unknown":
        return False
    for g in golds:
        g_n = normalize_answer(g)
        if g_n and (pred_n == g_n or pred_n in g_n or g_n in pred_n):
            return True
    return False


def exact_match(pred: str, golds: List[str]) -> bool:
    pred_n = normalize_answer(pred)
    if not pred_n or pred_n == "unknown":
        return False
    return any(pred_n == normalize_answer(g) for g in golds)


def gold_in_evidence(evidence: List[Dict[str, Any]], golds: List[str]) -> bool:
    combined = " ".join(str(e.get("doc_text", "")) + " " + str(e.get("title", "")) for e in evidence)
    combined_n = normalize_answer(combined)
    return any(normalize_answer(g) and normalize_answer(g) in combined_n for g in golds)


def choose_canonical_gold(golds):
    if not golds:
        return ""
    def score(g):
        g = str(g).strip()
        ascii_ratio = sum(1 for c in g if ord(c) < 128) / max(len(g), 1)
        s = 0
        if ascii_ratio > 0.9: s += 3
        if 1 <= len(g.split()) <= 4: s += 1
        if "(" not in g and ")" not in g: s += 1
        if "," not in g: s += 1
        s -= abs(len(g) - 12) * 0.02
        return s
    return sorted(golds, key=score, reverse=True)[0]
