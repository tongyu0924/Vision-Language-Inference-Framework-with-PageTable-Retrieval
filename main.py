import subprocess
import sys


def _pip(*packages):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *packages])


_pip("openai", "datasets", "pandas", "pillow", "requests", "tqdm", "numpy")

import json
from typing import Union
from PIL import Image
import pandas as pd
from tqdm.auto import tqdm
from datasets import load_dataset

from config import validate_api_key, CFG, CAIPResult, RetrievalTrace
import state
from corpus import load_viquae_wikipedia_corpus, build_page_table
from llm_io import get_viquae_image_source
from pipeline_stages import (
    identify_subject, ensure_subject_indexed, build_context_state,
    plan_queries, resolve_related_entities,
)
from retrieval import agent_led_pruning, build_tiered_evidence
from answer_and_eval import (
    generate_final_answer, answer_supported_by_evidence,
    detect_viquae_fields, parse_viquae_answers,
    soft_match, exact_match, gold_in_evidence, choose_canonical_gold,
)


def run_caip_hashindex(image_source: Union[Image.Image, str], question: str,
                       verbose: bool = False) -> CAIPResult:
    subject = identify_subject(image_source, verbose=verbose)
    subject = ensure_subject_indexed(subject, verbose=verbose)

    context_state = build_context_state(image_source, question, subject)

    related_ids = resolve_related_entities(context_state, verbose=verbose)

    if verbose:
        print(f"Subject: {subject.primary_title!r} (doc_id={subject.doc_id})")
        print(f"Related entity doc_ids: {related_ids}")

    plan = plan_queries(question, context_state, step=1)
    pruned = agent_led_pruning(plan["queries"], context_state, state.page_table, question=question)

    tiered = build_tiered_evidence(question, context_state, subject, related_ids, pruned)

    if verbose:
        print(f"Tier0={len(tiered['tier0'])}  Tier1={len(tiered['tier1'])}  Tier2={len(tiered['tier2'])}")
        for e in tiered["tier0"] + tiered["tier1"]:
            print(f"  [T{e['tier']}] {e['title']}: {e['doc_text'][:150]}")

    final = generate_final_answer(question, context_state, tiered)

    all_evidence = tiered["tier0"] + tiered["tier1"] + tiered["tier2"]
    trace = [RetrievalTrace(step=1, queries=plan["queries"],
                            retrieved_doc_ids=[e["doc_id"] for e in all_evidence],
                            matched_evidence=all_evidence)]

    return CAIPResult(
        answer=final.get("answer", ""), confidence=float(final.get("confidence", 0.0)),
        context_state=context_state, verified_evidence=all_evidence, trace=trace,
        final_reasoning=final.get("reasoning", ""), raw_final=final, subject=subject,
    )


def debug_first_sample():
    ds = load_dataset("PaulLerner/viquae_dataset", split=CFG["split"], streaming=True)
    first = next(iter(ds))
    img_key, q_key, a_key = detect_viquae_fields(first)
    image_source = get_viquae_image_source(first, img_key)
    question = str(first.get(q_key))
    golds = parse_viquae_answers(first.get(a_key))

    print("\n" + "=" * 80)
    print("DEBUG FIRST SAMPLE")
    print("=" * 80)
    print("Q:", question)
    print("Gold canonical:", choose_canonical_gold(golds))

    result = run_caip_hashindex(image_source, question, verbose=True)

    supported = answer_supported_by_evidence(result.answer, result.verified_evidence)
    hit = gold_in_evidence(result.verified_evidence, golds)
    soft = soft_match(result.answer, golds) and supported

    print("\nFINAL")
    print("Pred:", result.answer)
    print("Confidence:", result.confidence)
    print("Supported:", supported)
    print("Gold in evidence:", hit)
    print("Soft correct:", soft)
    print("Evidence count:", len(result.verified_evidence))
    return result


def run_viquae_val_hashindex(n_eval: int, split: str) -> pd.DataFrame:
    print(f"\nLoading ViQuAE split={split}...")
    probe = load_dataset("PaulLerner/viquae_dataset", split=split, streaming=True)
    first = next(iter(probe))
    img_key, q_key, a_key = detect_viquae_fields(first)
    print("Detected fields:")
    print(" image:", img_key)
    print(" question:", q_key)
    print(" answer:", a_key)

    ds = load_dataset("PaulLerner/viquae_dataset", split=split, streaming=True)

    rows = []
    loaded = 0
    skipped_image = 0
    skipped_qa = 0
    failed = 0
    scanned = 0

    for row in tqdm(ds, desc="CAIP-HashIndex ViQuAE validation"):
        if loaded >= n_eval:
            break
        scanned += 1
        image_source = get_viquae_image_source(row, img_key)
        if image_source is None:
            skipped_image += 1
            continue
        question = str(row.get(q_key) or "")
        golds = parse_viquae_answers(row.get(a_key))
        if not question or not golds:
            skipped_qa += 1
            continue

        qid = str(row.get("id", loaded))
        gold_canonical = choose_canonical_gold(golds)

        print("\n" + "=" * 80)
        print(f"Sample {loaded + 1}/{n_eval}")
        print("QID:", qid)
        print("Q:", question)
        print("Gold canonical:", gold_canonical)
        print("=" * 80)

        try:
            result = run_caip_hashindex(image_source, question, verbose=False)
            pred = result.answer
            supported = answer_supported_by_evidence(pred, result.verified_evidence)
            hit = gold_in_evidence(result.verified_evidence, golds)
            soft = soft_match(pred, golds) and supported
            em = exact_match(pred, golds) and supported

            row_out = {
                "qid": qid, "question": question, "prediction": pred,
                "confidence": result.confidence, "gold_canonical": gold_canonical,
                "gold_all": golds, "correct_soft": soft, "exact_match": em,
                "gold_in_evidence": hit, "answer_supported": supported,
                "n_evidence": len(result.verified_evidence),
                "reasoning": result.final_reasoning, "image_source": str(image_source),
                "subject_title": result.subject.primary_title,
            }
            rows.append(row_out)

            print("Pred:", pred)
            print("Confidence:", result.confidence)
            print("Answer supported:", supported)
            print("Soft correct:", soft)
            print("Exact match:", em)
            print("Gold in evidence:", hit)
            print("Evidence:", len(result.verified_evidence))

            loaded += 1
            pd.DataFrame(rows).to_csv(CFG["output_csv"], index=False)
            with open(CFG["output_json"], "w", encoding="utf-8") as f:
                json.dump(rows, f, ensure_ascii=False, indent=2)

        except Exception as e:
            failed += 1
            print("Runtime failed:", repr(e))
            continue

    df = pd.DataFrame(rows)

    print("\n" + "=" * 80)
    print("CAIP-HASHINDEX VIQUAE VALIDATION RESULTS")
    print("=" * 80)
    print("Requested N:", n_eval)
    print("Completed N:", len(df))
    print("Scanned rows:", scanned)
    print("Skipped image:", skipped_image)
    print("Skipped QA:", skipped_qa)
    print("Runtime failed:", failed)

    if len(df):
        print(f"Soft Accuracy: {df['correct_soft'].mean():.4f}")
        print(f"Exact Match: {df['exact_match'].mean():.4f}")
        print(f"Gold-in-Evidence Recall: {df['gold_in_evidence'].mean():.4f}")
        print(f"Answer Supported Rate: {df['answer_supported'].mean():.4f}")
        print(f"Average Confidence: {df['confidence'].mean():.4f}")
        print(f"Average Evidence Count: {df['n_evidence'].mean():.2f}")

    df.to_csv(CFG["output_csv"], index=False)
    with open(CFG["output_json"], "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print("\nSaved CSV:", CFG["output_csv"])
    print("Saved JSON:", CFG["output_json"])

    return df


def main():
    validate_api_key()

    print("Loading ViQuAE Wikipedia corpus...")
    corpus = load_viquae_wikipedia_corpus(CFG["corpus_docs"])
    print("Corpus docs:", len(corpus))
    assert len(corpus) > 0

    print("\nBuilding Hash-Indexed PageTable...")
    page_table = build_page_table(corpus, max_workers=CFG["index_max_workers"])
    print("PageTable entries:", len(page_table))
    example = next(iter(page_table.values()))
    print(f"Example: title={example.title!r}  summary={example.summary!r}")

    state.init_state(corpus, page_table)

    debug_first_sample()

    eval_df = run_viquae_val_hashindex(n_eval=CFG["n_eval"], split=CFG["split"])

    try:
        from IPython.display import display
        display_cols = [
            "qid", "question", "prediction", "confidence", "gold_canonical",
            "correct_soft", "exact_match", "gold_in_evidence", "answer_supported",
            "n_evidence", "subject_title",
        ]
        display(eval_df[display_cols])
    except Exception:
        print(eval_df.head())

    return eval_df


if __name__ == "__main__":
    main()
