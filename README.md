# Vision-Language Inference Framework with PageTable Retrieval

A hash-indexed, agent-led retrieval framework for knowledge-based Visual Question Answering (VQA). Built and evaluated on [ViQuAE](https://github.com/PaulLerner/ViQuAE).

## Overview

Most multimodal RAG systems embed an image+question pair into a single vector and retrieve via semantic similarity matching. This is brittle: the same image asked about with different questions can be re-identified inconsistently, and a single ranked evidence list lets topically-similar-but-wrong documents outrank the correct one.

This framework replaces vector search with an **LLM-summarized hash index** and a **tiered trust model** for evidence, instead of a flat ranked list.

## Contributions

- **One-time, cached subject identification** — each image's subject is identified exactly once and reused across every question asked about it, eliminating identity drift between questions.
- **Hash-Indexed PageTable Construction** — every corpus document is summarized once by an LLM into `(summary, keywords)`, forming a hash table instead of a vector index.
- **Agent-Led Dynamic Pruning** — an LLM agent reads PageTable summaries in batches and reasons about relevance, replacing cosine-similarity retrieval.
- **Tiered Co-linearity Matching** — evidence is partitioned into three explicit trust tiers (subject article / question-specific related entities / general search results) rather than pooled into one ranked list, so a guaranteed-correct document can never be outranked by a more "famous" but wrong one.
- **Evidence-only answer extraction** — answers must be traceable to retrieved text; no reliance on the model's parametric memory.

## Inference Framework

Step 1. One-time Subject Identification (cached per image)

Step 2. Question Understanding (subject is fixed input)

Step 3. Related-Entity Resolution

Step 4. Hash-Index Agent-Led Pruning ★ contribution

Step 5. Tiered Co-linearity Matching ★ contribution
- Tier 0: subject's own article (highest trust)
- Tier 1: question-specific related entities
- Tier 2: general hash-index results (lowest trust)

Step 6. Evidence-Only Answer Extraction

Output: answer + confidence
