#!/usr/bin/env python3
"""
Evaluate a local MLX model's directed sentiment extraction against ground truth.

Schema:
  entity   - company / body name
  ticker   - stock ticker or null
  polarity - "+", "-", "0", "~" (mixed/ambiguous)
  category - event type (Legal, Business, Performance, …)

"""

import os
import re
from pathlib import Path

import json
import argparse
from mlx_lm import load, generate

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# JSONL_PATH = Path("data/sentiment_eval.jsonl")
# MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit" #"mlx-community/gemma-3-4b-it-4bit"
# ADAPTER_PATH = "adapters" #"adapters_llama"

# SYSTEM_PROMPT = (
#     "You are a financial analyst specializing in directed sentiment extraction. "
#     "Given a financial news text, identify all mentioned entities and determine "
#     "the sentiment directed toward each one. Return your answer as a JSON array "
#     "where each element has: \"entity\" (name), \"polarity\" (+ positive, - negative, "
#     "0 neutral, ~ context-dependent), and \"category\" (one of: Legal, Business, "
#     "Performance, Recruitment, NewsRelease, Bankruptcy).\n\n"
#     "Valid polarities: \"+\", \"-\", \"0\", \"~\"\n"
# )

# VALID_POLARITIES = {"+", "-", "0", "~"}

# SYSTEM_PROMPT = """You are a financial news analyst specialising in directed (entity-level) sentiment analysis.

# Given a financial news sentence, extract EVERY named entity — companies, regulators, government bodies, individuals — and assign a sentiment polarity from that entity's perspective:

#   +   Positive  — entity wins, gains, benefits, succeeds
#   -   Negative  — entity loses, is penalised, harmed, fails
#   0   Neutral   — entity is mentioned but the outcome is clearly neutral
#   ~   Mixed     — outcome is genuinely ambiguous or has both positive and negative aspects for the entity

# Also provide:
#   ticker   — stock-exchange ticker (e.g. "AAPL") if the entity is a publicly-traded company; otherwise null
#   category — event type, e.g. Legal, Business, Performance, Recruitment, Regulatory, Financial, M&A, Product

# Return ONLY a JSON object in this exact format, with no commentary before or after:
# {"extractions": [{"entity": "...", "ticker": "..." or null, "polarity": "+"/"-"/"0"/"~", "category": "..."}, ...]}"""


# ---------------------------------------------------------------------------
# MLX inference + JSON parsing
# ---------------------------------------------------------------------------
# def _parse_json(raw: str) -> list[dict]:
#     """Extract and validate the extractions list from the model's raw output."""
#     # Try the whole string first, then look for the first {...} block
#     candidates = [raw]
#     m = re.search(r"\{.*\}", raw, re.DOTALL)
#     if m:
#         candidates.append(m.group())

#     for candidate in candidates:
#         try:
#             data = json.loads(candidate)
#             if isinstance(data, list):
#                 data = {'extractions':data}
#             extractions = data.get("extractions", [])
#             # Keep only entries with required keys and valid polarity
#             valid = [
#                 e for e in extractions
#                 if isinstance(e, dict)
#                 and all(k in e for k in ("entity", "ticker", "polarity", "category"))
#                 and e["polarity"] in VALID_POLARITIES
#             ]
#             if valid:
#                 return valid
#         except json.JSONDecodeError:
#             continue
#     return []

def extract_json(raw: str) -> list:
    """Parse a JSON array from model output, stripping markdown fences if present."""
    # Strip markdown code fences (```json ... ``` or ``` ... ```)
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip()
    cleaned = cleaned.rstrip("`").strip()

    # Find the first '[' and last ']' to isolate the array
    start = cleaned.find("[")
    end = cleaned.rfind("]")

    if start == -1 or end == -1:
        raise ValueError(f"No JSON array found in model output:\n{raw}")

    candidate = cleaned[start : end + 1]

    # Fix trailing commas before ] or } (common LLM output issue)
    candidate = re.sub(r",\s*([\]}])", r"\1", candidate)

    try:
        return json.loads(candidate)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON from model output:\n{raw}\n\nExtracted candidate:\n{candidate}\n\nError: {e}") from e


def extract_entities(model, tokenizer, text: str, max_tokens) -> list[dict]:
    """Run the local MLX model and return parsed extractions."""
    
    SYSTEM_PROMPT = (
        "You are a financial analyst specializing in directed sentiment extraction. "
        "Given a financial news text, identify all mentioned entities and determine "
        "the sentiment directed toward each one. Return your answer as a JSON array "
        "where each element has: \"entity\" (name), \"polarity\" (+ positive, - negative, "
        "0 neutral, ~ context-dependent), and \"category\" (one of: Legal, Business, "
        "Performance, Recruitment, NewsRelease, Bankruptcy).\n\n"
        "Valid polarities: \"+\", \"-\", \"0\", \"~\"\n"
    )

    user_content = f"Extract the directed financial sentiment from the following text:\n\n{text}"

    # messages = [
    #     {"role": "system", "content": SYSTEM_PROMPT},
    #     {
    #         "role": "user",
    #         "content": user_content
    #         # "content": (
    #         #     "Extract all entities and their sentiment polarities "
    #         #     f"from this financial news sentence:\n\n{text}"
    #         # ),
    #     },
    # ]

    # prompt = tokenizer.apply_chat_template(
    #     messages, tokenize=False, add_generation_prompt=True
    # )

    if tokenizer.chat_template is not None:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True
        )
    else:
        prompt = f"{SYSTEM_PROMPT}\n\nUser: {user_content}\nAssistant:"

    raw = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
#        temp=temp,
    )

    return extract_json(raw)
    # raw = generate(model, tokenizer, prompt=prompt, max_tokens=1024, verbose=False)
    # return extract_json(raw)


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------
def _norm(name: str) -> str:
    """Normalise entity name for fuzzy matching."""
    return re.sub(r"\s+", " ", name.strip().lower())


def compare(predicted: list[dict], ground_truth: list[dict]) -> dict:
    gt_map   = {_norm(e["entity"]): e for e in ground_truth}
    pred_map = {_norm(e["entity"]): e for e in predicted}

    tp = sum(1 for k in gt_map if k in pred_map)
    fp = sum(1 for k in pred_map if k not in gt_map)
    fn = sum(1 for k in gt_map if k not in pred_map)

    polarity_correct = sum(
        1 for k in gt_map
        if k in pred_map and gt_map[k]["polarity"] == pred_map[k]["polarity"]
    )
    polarity_total = tp  # only count entity matches

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall    = tp / (tp + fn) if tp + fn else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if precision + recall else 0.0)
    pol_acc   = polarity_correct / polarity_total if polarity_total else 0.0

    return dict(
        tp=tp, fp=fp, fn=fn,
        precision=precision, recall=recall, f1=f1,
        polarity_correct=polarity_correct,
        polarity_total=polarity_total,
        polarity_accuracy=pol_acc,
    )


# ---------------------------------------------------------------------------
# Pretty print helpers
# ---------------------------------------------------------------------------
def _row(polarity: str, entity: str, ticker, category: str) -> str:
    t = ticker or "N/A"
    return f"  [{polarity:1s}] {entity:<42s} ({t:<12s}) [{category}]"


def _print_extractions(extractions: list[dict]) -> None:
    for e in extractions:
        print(_row(e["polarity"], e["entity"], e.get("ticker"), e["category"]))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract directed financial sentiment evaluation"
    )
    parser.add_argument(
        "--dataset_path",
        default="data/sentiment_eval.jsonl",
        help="Path to the input JSONL dataset",
    )
    parser.add_argument(
        "--model",
        default="mlx-community/Llama-3.2-3B-Instruct-4bit",
        help="Base model name or local path",
    )
    parser.add_argument(
        "--adapter",
        default="adapters",
        help="Path to trained LoRA adapter (omit or set to '' to use base model)",
    )
    parser.add_argument(
        "--max_tokens", type=int, default=512, help="Maximum tokens to generate"
    )

    # JSONL_PATH = Path("data/sentiment_eval.jsonl")
    # MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit" #"mlx-community/gemma-3-4b-it-4bit"
    # ADAPTER_PATH = "adapters" #"adapters_llama"

    args = parser.parse_args()

    JSONL_PATH = Path(args.dataset_path)

    records = [
        json.loads(line)
        for line in JSONL_PATH.read_text().splitlines()
        if line.strip()
    ]
    n = len(records)
    print(f"Loaded {n} records from {args.dataset_path}\n{'=' * 80}")

    print(f"Loading model: {args.model}")
    # Load with adapter if path exists, otherwise base model only
    if args.adapter and os.path.exists(args.adapter):
        model, tokenizer = load(args.model, adapter_path=args.adapter)
    else:
        if args.adapter:
            print(f"Warning: adapter path '{args.adapter}' not found, using base model.")
        model, tokenizer = load(args.model)

    # print(f"Loading model: {MODEL}")
    # model, tokenizer = load(MODEL) #, adapter_path=ADAPTER_PATH)
    # print("Model loaded.\n")

    totals  = dict(tp=0, fp=0, fn=0, polarity_correct=0, polarity_total=0)
    correct_records = 0  # records where every entity + polarity is perfect

    for i, record in enumerate(records, 1):
        rid  = record["id"]
        text = record["text"]
        gt   = record["extractions"]

        print(f"\n[{i:02d}/{n}] {rid}")
        print(f"Text: {text}\n")

        print("Ground truth:")
        _print_extractions(gt)

        predicted = extract_entities(model, tokenizer, text, args.max_tokens)

        print("\nPredicted:")
        _print_extractions(predicted)

        m = compare(predicted, gt)
        for k in ("tp", "fp", "fn", "polarity_correct", "polarity_total"):
            totals[k] += m[k]

        if m["fp"] == 0 and m["fn"] == 0 and m["polarity_accuracy"] == 1.0:
            correct_records += 1
            perfect = " ✓"
        else:
            perfect = ""

        print(
            f"\nRecord: P={m['precision']:.0%}  R={m['recall']:.0%}  "
            f"F1={m['f1']:.0%}  "
            f"Polarity={m['polarity_accuracy']:.0%} "
            f"({m['polarity_correct']}/{m['polarity_total']} matched){perfect}"
        )
        print("-" * 80)

    # ---- Overall report ----
    tp, fp, fn = totals["tp"], totals["fp"], totals["fn"]
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec  = tp / (tp + fn) if tp + fn else 0.0
    f1   = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    pol_acc = (totals["polarity_correct"] / totals["polarity_total"]
               if totals["polarity_total"] else 0.0)

    print(f"\n{'=' * 80}")
    print("OVERALL ACCURACY REPORT")
    print(f"{'=' * 80}")
    print(f"Records evaluated :  {n}")
    print(f"Perfect records   :  {correct_records}/{n}  ({correct_records/n:.1%})")
    print()
    print(f"Entity Precision  :  {prec:.1%}  ({tp} TP, {fp} FP)")
    print(f"Entity Recall     :  {rec:.1%}  ({tp} TP, {fn} FN)")
    print(f"Entity F1         :  {f1:.1%}")
    print()
    print(f"Polarity Accuracy :  {pol_acc:.1%}  "
          f"({totals['polarity_correct']}/{totals['polarity_total']} "
          f"matched-entity pairs)")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
