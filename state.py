from typing import Dict, List, Optional
from config import Document, PageEntry, doc_key

corpus: List[Document] = []
page_table: Dict[int, PageEntry] = {}
_title_to_doc_id: Dict[str, int] = {}
_next_doc_id: List[int] = [0]


def init_state(loaded_corpus: List[Document], loaded_page_table: Dict[int, PageEntry]):
    global corpus, page_table, _title_to_doc_id, _next_doc_id
    corpus = loaded_corpus
    page_table = loaded_page_table
    _title_to_doc_id = {doc_key(d.title, "")[0]: d.doc_id for d in corpus}
    _next_doc_id[0] = len(corpus)


def get_doc_by_id(doc_id) -> Optional[Document]:
    try:
        return corpus[int(doc_id)]
    except Exception:
        return None


def get_page_by_id(doc_id) -> Optional[PageEntry]:
    return page_table.get(int(doc_id))
