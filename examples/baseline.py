"""Baseline: single model answers questions with context concatenated to the query."""

import argparse
import os
import sys
import time
import traceback
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from orchestrator.models import OpenAIHandler


def main():
    p = argparse.ArgumentParser(description="Single-model baseline on a benchmark dataset")
    p.add_argument("--task", default="qasper", help="Task name (default: qasper)")
    p.add_argument("-n", "--num-examples", type=int, default=20,
                   help="Number of examples (-1 for all, default: 20)")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument("--model",
                   default=os.environ.get("MODEL",
                                          "meta-llama/Llama-3.3-70B-Instruct-Turbo"))
    p.add_argument("--judge-model",
                   default="Qwen/Qwen3-235B-A22B-Instruct-2507-tput")
    p.add_argument("--base-url",
                   default=os.environ.get("BASE_URL", "https://api.together.xyz/v1"))
    p.add_argument("--num-distractors", type=int, default=0,
                   help="Number of distractor papers to add to context (default: 0)")
    p.add_argument("--out-dir", default=None,
                   help="Override output directory (default: out/<task>/<timestamp>)")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.3)
    args = p.parse_args()

    # --- Load task ---
    limit = args.num_examples if args.num_examples > 0 else None

    if args.task == "qasper":
        from tasks.qasper import QasperTask
        task = QasperTask(
            seed=args.seed,
            limit=limit,
            num_distractors=args.num_distractors,
            judge_model=args.judge_model,
            base_url=args.base_url,
        )
    else:
        print(f"Unknown task: {args.task}", file=sys.stderr)
        sys.exit(1)

    examples = task.build_dataset()
    print(f"Task: {args.task} | Examples: {len(examples)} | Seed: {args.seed}")
    print(f"Model: {args.model}")
    print(f"Judge: {args.judge_model}")

    # --- Setup model ---
    handler = OpenAIHandler(
        model=args.model,
        base_url=args.base_url,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    # --- Setup output directory ---
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.out_dir:
        out_dir = os.path.join(args.out_dir, timestamp)
    else:
        out_dir = os.path.join("out", args.task, timestamp)
    os.makedirs(out_dir, exist_ok=True)

    # --- Run ---
    rows = []
    for i, example in enumerate(examples):
        print(f"\n[{i+1}/{len(examples)}] {example.query[:80]}...")

        try:
            messages = [
                {
                    "role": "system",
                    "content": "Answer the question about the following document. "
                               "Be concise and specific.",
                },
                {
                    "role": "user",
                    "content": f"Document:\n{example.context}\n\n"
                               f"Question: {example.query}",
                },
            ]

            resp = handler.chat(messages)
            prediction = resp.text
            sc = task.score(prediction, example)
            print(f"  Score: {sc} | "
                  f"In: {resp.usage.input_tokens:,} | Out: {resp.usage.output_tokens:,} | "
                  f"{resp.elapsed:.1f}s")

        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()
            prediction = f"ERROR: {e}"
            resp = None
            sc = 0

        rows.append({
            "id": example.id,
            "doc_id": example.metadata.get("doc_id", ""),
            "question": example.query,
            "label": example.target,
            "prediction": prediction,
            "score": sc,
            "input_tokens": resp.usage.input_tokens if resp else 0,
            "output_tokens": resp.usage.output_tokens if resp else 0,
        })

    # --- Save results ---
    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "results.csv")
    df.to_csv(csv_path, index=False)

    total_in = df["input_tokens"].sum()
    total_out = df["output_tokens"].sum()

    print(f"\n{'='*60}")
    print(f"Results saved to: {csv_path}")
    print(f"Accuracy: {df['score'].mean():.2%} ({int(df['score'].sum())}/{len(df)})")
    print(f"Tokens — input: {total_in:,} | output: {total_out:,} | total: {total_in + total_out:,}")


if __name__ == "__main__":
    main()
