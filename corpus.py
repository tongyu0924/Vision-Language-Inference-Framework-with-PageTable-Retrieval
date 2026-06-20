import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, List, Dict, Tuple
import requests
from tqdm.auto import tqdm
from datasets import load_dataset

from config import Document, PageEntry, doc_key, safe_json_parse, WIKI_USER_AGENT, CFG
from llm_io import call_openai_text
import state


def parse_text_field(x) -> str:
    if x is None:
        return ""
    if isinstance(x, dict):
        if "paragraph" in x:
            p = x["paragraph"]
            return "\n".join(str(i) for i in p) if isinstance(p, list) else str(p)
        if "text" in x:
            return parse_text_field(x["text"])
        return " ".join(str(v) for v in x.values())
    if isinstance(x, list):
        return "\n".join(str(i) for i in x)
    return str(x)


def parse_title_from_text(text: str) -> str:
    lines = [l.strip() for l in str(text).splitlines() if l.strip()]
    if lines:
        first = re.sub(r"^Title:\s*", "", lines[0])
        return first[:200]
    return ""


def make_document(row, doc_id: int, source: str) -> Optional[Document]:
    raw_text = row.get("text") or row.get("paragraph") or row.get("content")
    text = parse_text_field(raw_text)
    title = (
        row.get("wikipedia_title")
        or row.get("title")
        or row.get("page_title")
        or parse_title_from_text(text)
        or ""
    )
    title = str(title).strip()[:200]
    text = str(text).strip()
    if not title and text:
        title = parse_title_from_text(text)
    if len(text) < 80:
        return None
    if title and text.lower().startswith(title.lower()):
        text = text[len(title):].strip()
    return Document(doc_id=doc_id, title=title, text=text[:2600], source=source)


def load_viquae_wikipedia_corpus(target_docs: int) -> List[Document]:
    docs = []
    seen = set()
    data_files = {
        "humans_with_faces": "humans_with_faces.jsonl.gz",
        "humans_without_faces": "humans_without_faces.jsonl.gz",
        "non_humans": "non_humans.jsonl.gz",
    }
    ds_dict = load_dataset("PaulLerner/viquae_wikipedia", data_files=data_files, streaming=True)
    iterators = {k: iter(v) for k, v in ds_dict.items()}
    active = set(iterators.keys())
    pbar = tqdm(total=target_docs, desc="Loading ViQuAE Wikipedia corpus")
    while len(docs) < target_docs and active:
        for key in list(active):
            try:
                row = next(iterators[key])
            except StopIteration:
                active.discard(key)
                continue
            doc = make_document(row, len(docs), f"viquae_wikipedia/{key}")
            if doc is None:
                continue
            k = doc_key(doc.title, doc.text)
            if k in seen:
                continue
            seen.add(k)
            docs.append(doc)
            pbar.update(1)
            if len(docs) >= target_docs:
                break
    pbar.close()
    return docs


def build_page_entry(doc: Document) -> PageEntry:
    prompt = f"""
Summarise the following document in ONE sentence (max 25 words).
Then list 4-8 key entities/keywords (people, places, dates, numbers,
named things, currencies) someone searching for this content would use.

Title: {doc.title}
Text: {doc.text[:900]}

Return ONLY valid JSON:
{{"summary": "...", "keywords": ["...", "..."]}}
"""
    try:
        out = call_openai_text(CFG["index_model"], prompt, temperature=0.0, max_tokens=150)
        data = safe_json_parse(out)
        summary = str(data.get("summary", doc.text[:120]))
        keywords = [str(k) for k in data.get("keywords", [])][:8]
    except Exception:
        summary = doc.text[:150]
        keywords = re.findall(r"\b[A-Z][a-zA-Z]+\b", doc.text)[:6]
    return PageEntry(doc_id=doc.doc_id, title=doc.title, summary=summary,
                     keywords=keywords, text=doc.text)


def build_page_table(docs: List[Document], max_workers: int = 20) -> Dict[int, PageEntry]:
    table: Dict[int, PageEntry] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(build_page_entry, d): d for d in docs}
        for fut in tqdm(as_completed(futures), total=len(docs),
                        desc=f"Building Hash-Indexed PageTable ({max_workers} workers)"):
            try:
                entry = fut.result()
                table[entry.doc_id] = entry
            except Exception:
                d = futures[fut]
                table[d.doc_id] = PageEntry(
                    doc_id=d.doc_id, title=d.title, summary=d.text[:150],
                    keywords=re.findall(r"\b[A-Z][a-zA-Z]+\b", d.text)[:6],
                    text=d.text)
    return table


def fetch_wikipedia_extract(title: str) -> Optional[Tuple[str, str]]:
    def _try_exact(t: str) -> Optional[Tuple[str, str]]:
        try:
            r = requests.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query", "prop": "extracts", "explaintext": 1,
                    "exsectionformat": "plain", "titles": t, "format": "json",
                    "redirects": 1,
                },
                headers={"User-Agent": WIKI_USER_AGENT}, timeout=10,
            )
            pages = r.json().get("query", {}).get("pages", {})
            for _, page in pages.items():
                if "missing" in page:
                    continue
                extract = page.get("extract", "")
                resolved_title = page.get("title", t)
                if extract and len(extract) > 80:
                    return resolved_title, extract
        except Exception:
            pass
        return None

    result = _try_exact(title)
    if result is not None:
        return result

    try:
        r = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={"action": "query", "list": "search", "srsearch": title,
                   "srlimit": 1, "format": "json"},
            headers={"User-Agent": WIKI_USER_AGENT}, timeout=10,
        )
        hits = r.json().get("query", {}).get("search", [])
        if hits:
            best_title = hits[0]["title"]
            return _try_exact(best_title)
    except Exception:
        pass
    return None


def ensure_entity_indexed(entity_title: str, verbose: bool = False) -> Optional[int]:
    if not entity_title or len(entity_title.strip()) < 2:
        return None

    fetched = fetch_wikipedia_extract(entity_title)
    if fetched is None:
        if verbose:
            print(f"  [entity-fetch] '{entity_title}' -> NOT FOUND")
        return None

    resolved_title, extract = fetched
    key = doc_key(resolved_title, "")[0]

    if key in state._title_to_doc_id:
        if verbose:
            print(f"  [entity-fetch] '{entity_title}' -> '{resolved_title}' "
                 f"(already indexed as doc_id={state._title_to_doc_id[key]})")
        return state._title_to_doc_id[key]

    doc_id = state._next_doc_id[0]
    state._next_doc_id[0] += 1
    doc = Document(doc_id=doc_id, title=resolved_title, text=extract[:2600],
                   source="entity_grounded_fetch")
    state.corpus.append(doc)
    state._title_to_doc_id[key] = doc_id

    entry = build_page_entry(doc)
    state.page_table[doc_id] = entry
    if verbose:
        print(f"  [entity-fetch] '{entity_title}' -> '{resolved_title}' "
             f"(new doc_id={doc_id})  summary={entry.summary!r}")
    return doc_id
