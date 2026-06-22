# Vision-Language Inference Framework with PageTable Retrieval

A hash-indexed, agent-led retrieval framework for knowledge-based Visual Question Answering (VQA). Built and evaluated on [ViQuAE](https://github.com/PaulLerner/ViQuAE).

## Overview

Most multimodal RAG systems embed an image+question pair into a single vector and retrieve evidence via semantic similarity matching. This is brittle: the same image asked about with different questions can be re-identified inconsistently, and a single ranked evidence list may let topically similar but incorrect documents outrank the correct one.

This framework replaces vector search with a **Structured ExtractionIndex** and **subject-prioritized evidence extraction**. Instead of storing each document as an embedding or a single LLM-generated hash key, each chunk is converted into structured fields such as `topics`, `entities`, `relations`, `evidence`, and `keywords`, then retrieved through inverted lookup tables.

## Contributions

- **One-time, cached subject identification** — each image's subject is identified exactly once and reused across every question asked about it, eliminating identity drift between questions.
- **Structured ExtractionIndex Construction** — each corpus chunk is processed once by an LLM to extract structured fields such as `topics`, `entities`, `concepts`, `methods`, `claims`, `relations`, `evidence`, `questions_answered`, and `keywords`.
- **Inverted Lookup Table Retrieval** — extracted fields are compiled into lookup tables such as `topic_index`, `entity_index`, `concept_index`, `relation_index`, `question_index`, and `keyword_index`, replacing vector similarity search with structured retrieval.
- **Query-side Structured Extraction** — each user question is converted into structured fields such as `intent`, `entities`, `topics`, `relations`, and `keywords`, then matched against the ExtractionIndex.
- **Answer-type hard filtering** — candidate chunks are pruned by a deterministic type-compatibility predicate (e.g., date queries require a four-digit year, person queries require a named entity), requiring no model calls and providing a traceable rejection reason for every discarded document.
- **Subject-prioritized tiered evidence** — candidates are organized into three tiers by provenance: subject article first, entity-grounded documents second, and remaining PageTable candidates ordered by field-hit count, without numerical scoring.
- **Evidence-only answer extraction** — answers must be traceable to retrieved summaries, evidence snippets, or full content; no reliance on the model's parametric memory.

## Inference Framework

Step 1. One-time Subject Identification (cached per image)
Step 2. Question Understanding (subject is fixed input)
Step 3. Query-side Structured Extraction
Extract:
- `intent`
- `entities`
- `topics`
- `relations`
- `keywords`
Step 4. Entity-grounded Augmentation
Step 5. Search Inverted Lookup Tables ★ contribution
Step 6. Candidate Chunk Retrieval
Step 7. Answer-type Hard Filtering ★ contribution
Step 8. Tiered Evidence Organization (Tier 0 / 1 / 2 by provenance)
Step 9. Need More Detail?
- Yes: Read full content
- No: Use extracted evidence
Step 10. Subject-prioritized Evidence-Only Answer Extraction
Output: answer

## Results

### Metrics

- **Exact Match (EM)** — the predicted answer must match the gold answer *exactly* (after lowercasing and removing articles). This is the strict metric: a correct but differently-phrased answer counts as wrong.
- **Soft Accuracy** — the predicted answer counts as correct if it overlaps with (contains, or is contained in) any gold answer string. This tolerates paraphrasing and is the primary metric reported in the original ViQuAE paper, since gold answers come with multiple acceptable surface forms (e.g. "Drachma", "Greek drachma", "Modern drachma" are all valid for the same question).

### Comparison

| Method | Soft Accuracy | Exact Match |
|---|---|---|
| `gpt-4o-mini` (zero-shot, no retrieval) | ~20% | ~15% |
| `gpt-4o-mini` + this framework | ~65% | ~50% |

Both rows use the same underlying model (`gpt-4o-mini`) on the ViQuAE validation questions, the only variable is whether this inference framework is used. This isolates how much of the accuracy gain comes from the framework's retrieval design versus the model's own general knowledge.
