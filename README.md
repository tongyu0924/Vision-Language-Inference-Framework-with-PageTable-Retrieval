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

## Results

Evaluated on ViQuAE validation set (50 sampled questions), using `gpt-4o-mini` as the underlying model for both the baseline and our framework — the only variable is whether retrieval is used.

### Metrics

- **Exact Match (EM)** — the predicted answer must match the gold answer *exactly* (after lowercasing and removing articles). This is the strict metric: a correct but differently-phrased answer counts as wrong.
- **Soft Accuracy** — the predicted answer counts as correct if it overlaps with (contains, or is contained in) any gold answer string. This tolerates paraphrasing and is the primary metric reported in the original ViQuAE paper, since gold answers come with multiple acceptable surface forms (e.g. "Drachma", "Greek drachma", "Modern drachma" are all valid for the same question).

### Comparison

| Method | Soft Accuracy | Exact Match |
|---|---|---|
| `gpt-4o-mini` (zero-shot, no retrieval) | ~20% | ~15% |
| `gpt-4o-mini` + this framework | ~65% | ~50% |

Both rows use the same underlying model (`gpt-4o-mini`) on the ViQuAE validation questions, the only variable is whether this inference framework is used. This isolates how much of the accuracy gain comes from the framework's retrieval design versus the model's own general knowledge.
