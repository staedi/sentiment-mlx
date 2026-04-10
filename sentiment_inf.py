#!/usr/bin/env python3
"""
Inference script for directed financial sentiment extraction.
Returns extracted entities as a JSON array.
"""

import os
import re
import json
import argparse
from mlx_lm import load, generate


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


def extract_sentiment(
    text: str,
    model_name: str = "mlx-community/Llama-3.2-3B-Instruct-4bit",
    adapter_path: str = "adapters",
    max_tokens: int = 512,
    # temp: float = 0.1,
) -> list:
    """
    Extract directed financial sentiment from text.

    Returns a list of dicts with keys: entity, polarity, category.
    """
    print(f"Loading model: {model_name}")
    # Load with adapter if path exists, otherwise base model only
    if adapter_path and os.path.exists(adapter_path):
        model, tokenizer = load(model_name, adapter_path=adapter_path)
    else:
        if adapter_path:
            print(f"Warning: adapter path '{adapter_path}' not found, using base model.")
        model, tokenizer = load(model_name)

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


def main():
    parser = argparse.ArgumentParser(
        description="Extract directed financial sentiment"
    )
    parser.add_argument("--text", required=True, help="Financial news text to analyse")
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
    # parser.add_argument(
    #     "--temp", type=float, default=0.1, help="Sampling temperature (0.0 = greedy)"
    # )
    args = parser.parse_args()

    print(f"Input: {args.text}\n")

    try:
        results = extract_sentiment(
            text=args.text,
            model_name=args.model,
            adapter_path=args.adapter,
            max_tokens=args.max_tokens,
            # temp=args.temp,
        )
        print("Extractions:")
        print(json.dumps(results, indent=2, ensure_ascii=False))
    except ValueError as e:
        print(f"Error: {e}")
        raise
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
