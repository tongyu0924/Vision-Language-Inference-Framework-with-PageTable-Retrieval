import io
import hashlib
from typing import Union, Dict, Any, List
from PIL import Image

from config import SubjectIdentity, safe_json_parse, clean_query, CFG
from llm_io import call_openai_vision
from corpus import ensure_entity_indexed

_subject_cache: Dict[str, SubjectIdentity] = {}


def _image_cache_key(image_source: Union[Image.Image, str]) -> str:
    if isinstance(image_source, str):
        return f"url:{image_source}"
    buf = io.BytesIO()
    image_source.convert("RGB").resize((64, 64)).save(buf, format="PNG")
    return f"hash:{hashlib.md5(buf.getvalue()).hexdigest()}"


def identify_subject(image_source: Union[Image.Image, str],
                     verbose: bool = False) -> SubjectIdentity:
    key = _image_cache_key(image_source)
    if key in _subject_cache:
        return _subject_cache[key]

    prompt = """
Look at this image carefully and identify its SPECIFIC named subject —
a person, place, object, species, artwork, or similar. Read any visible
text, labels, plaques, or distinguishing features to be as precise as
possible (a specific person's full name, a specific species' scientific
or common name, a specific named object/artwork — NOT a generic category
like "a king" or "a plant").

Return ONLY valid JSON:
{
  "description": "what you observe in 1 sentence",
  "primary_title": "your single best-guess Wikipedia article title",
  "alt_titles": ["alternative guess 1", "alternative guess 2"]
}
"""
    try:
        out = call_openai_vision(CFG["vision_model"], image_source, prompt, temperature=0.0)
        data = safe_json_parse(out)
        primary = str(data.get("primary_title", "")).strip()
        alts = [str(t).strip() for t in data.get("alt_titles", []) if str(t).strip()]
        desc = str(data.get("description", ""))
    except Exception as e:
        if verbose:
            print(f"  [subject-id failed: {repr(e)}]")
        primary, alts, desc = "", [], ""

    subject = SubjectIdentity(primary_title=primary, alt_titles=alts,
                              doc_id=None, raw_description=desc)
    _subject_cache[key] = subject
    if verbose:
        print(f"  [subject-id] primary={primary!r}  alts={alts}  desc={desc!r}")
    return subject


def ensure_subject_indexed(subject: SubjectIdentity, verbose: bool = False) -> SubjectIdentity:
    if subject.doc_id is not None:
        return subject
    for title in ([subject.primary_title] + subject.alt_titles):
        if not title:
            continue
        did = ensure_entity_indexed(title, verbose=verbose)
        if did is not None:
            subject.doc_id = did
            break
    return subject


def build_context_state(image_source: Union[Image.Image, str], question: str,
                        subject: SubjectIdentity) -> Dict[str, Any]:
    prompt = f"""
The image's subject has already been identified as: "{subject.primary_title}"
(description: {subject.raw_description})

Given this fixed subject, analyze the following question to plan
retrieval. Do not answer the question. Do not second-guess the subject
identity above.

Question:
{question}

Return ONLY valid JSON:
{{
  "target_relation": "exact relation or attribute asked by the question",
  "expected_answer_type": "person | place | date | number | organization | object | event | currency | short_phrase | yes_no | other",
  "related_entity_guess": "if the question asks about something RELATED to the subject (predecessor, successor, family member, product, location, creator, etc.), your best guess at that related entity's own Wikipedia article title — empty string if not applicable or the question is about the subject itself",
  "related_entity_alt_guesses": ["other plausible related-entity titles, if there could be more than one candidate (e.g. multiple historical predecessors)"],
  "constraints": ["textual constraints implied by the question, e.g. dates, locations"],
  "initial_queries": ["short query 1", "short query 2"]
}}
"""
    try:
        out = call_openai_vision(CFG["vision_model"], image_source, prompt, temperature=0.0)
        ctx = safe_json_parse(out)
    except Exception as e:
        print("Context call failed, using fallback:", repr(e))
        ctx = {"target_relation": question, "expected_answer_type": "short_phrase",
              "related_entity_guess": "", "related_entity_alt_guesses": [],
              "constraints": [], "initial_queries": [question]}

    ctx.setdefault("target_relation", question)
    ctx.setdefault("expected_answer_type", "short_phrase")
    ctx.setdefault("related_entity_guess", "")
    ctx.setdefault("related_entity_alt_guesses", [])
    ctx.setdefault("constraints", [])
    ctx.setdefault("initial_queries", [question])
    ctx["subject_title"] = subject.primary_title
    ctx["subject_description"] = subject.raw_description
    return ctx


def is_noisy_query(q):
    q = clean_query(q)
    toks = q.split()
    if len(toks) > 12:
        return True
    visual_noise = {"banknote", "specimen", "architectural", "elements",
                    "map", "image", "shown", "visible", "photo", "picture"}
    return sum(1 for t in toks if t.lower() in visual_noise) >= 3


def deterministic_query_expansion(question: str, context_state: Dict[str, Any]) -> List[str]:
    q = question.strip()
    queries = [q]
    target_relation = str(context_state.get("target_relation", "")).strip()
    subject = str(context_state.get("subject_title", "")).strip()
    if target_relation:
        queries.append(target_relation)
    if subject:
        queries.append(subject)
        if target_relation:
            queries.append(f"{subject} {target_relation}")
    for init_q in context_state.get("initial_queries", []):
        queries.append(init_q)
    related = context_state.get("related_entity_guess", "")
    if related:
        queries.append(related)
    for alt in context_state.get("related_entity_alt_guesses", []):
        queries.append(alt)

    out, seen = [], set()
    for x in queries:
        x = clean_query(x)
        if not x or is_noisy_query(x):
            continue
        lx = x.lower()
        if lx not in seen:
            out.append(x)
            seen.add(lx)
    return out[:10]


def plan_queries(question: str, context_state: Dict[str, Any], step: int) -> Dict[str, Any]:
    return {"queries": deterministic_query_expansion(question, context_state)}


def resolve_related_entities(context_state: Dict[str, Any], verbose: bool = False) -> List[int]:
    titles = []
    if context_state.get("related_entity_guess"):
        titles.append(context_state["related_entity_guess"])
    titles.extend(context_state.get("related_entity_alt_guesses", []))

    ids = []
    for t in titles[:5]:
        did = ensure_entity_indexed(t, verbose=verbose)
        if did is not None and did not in ids:
            ids.append(did)
    return ids
