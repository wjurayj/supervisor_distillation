"""Run a supervisor-worker experiment on a benchmark dataset."""

import argparse
import os
import sys
import traceback
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from distill import OpenAIHandler, run


def main():
    p = argparse.ArgumentParser(description="Run experiment on benchmark dataset")
    p.add_argument("--task", default="qasper", help="Task name (default: qasper)")
    p.add_argument("-n", "--num-examples", type=int, default=20,
                   help="Number of examples (-1 for all, default: 20)")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument("--supervisor-model",
                   default=os.environ.get("SUPERVISOR_MODEL",
                                          "meta-llama/Llama-3.3-70B-Instruct-Turbo"))
    p.add_argument("--worker-model",
                   default=os.environ.get("WORKER_MODEL",
                                          "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"))
    p.add_argument("--judge-model",
                   default="Qwen/Qwen3-235B-A22B-Instruct-2507-tput")
    p.add_argument("--base-url",
                   default=os.environ.get("BASE_URL", "https://api.together.xyz/v1"))
    p.add_argument("--max-iterations", type=int, default=15)
    args = p.parse_args()

    # --- Load task ---
    limit = args.num_examples if args.num_examples > 0 else None

    if args.task == "qasper":
        from tasks.qasper import QasperTask
        task = QasperTask(
            seed=args.seed,
            limit=limit,
            judge_model=args.judge_model,
            base_url=args.base_url,
        )
    # elif args.task == "other_task":
    #     from tasks.other_task import OtherTask
    #     task = OtherTask(...)
    else:
        print(f"Unknown task: {args.task}", file=sys.stderr)
        sys.exit(1)

    examples = task.build_dataset()
    print(f"Task: {args.task} | Examples: {len(examples)} | Seed: {args.seed}")
    print(f"Supervisor: {args.supervisor_model}")
    print(f"Worker: {args.worker_model}")
    print(f"Judge: {args.judge_model}")

    # --- Setup experiment directory ---
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_dir = os.path.join("logs", args.task, timestamp)
    os.makedirs(experiment_dir, exist_ok=True)

    # --- Setup models ---
    supervisor = OpenAIHandler(
        model=args.supervisor_model, base_url=args.base_url,
        temperature=0.7, max_tokens=2048,
    )
    worker = OpenAIHandler(
        model=args.worker_model, base_url=args.base_url,
        temperature=0.6, max_tokens=512,
    )

    # --- Run ---
    rows = []
    for i, example in enumerate(examples):
        print(f"\n[{i+1}/{len(examples)}] {example.query[:80]}...")
        example_dir = os.path.join(experiment_dir, f"{i:03d}")

        try:
            result = run(
                query=example.query,
                context=example.context,
                supervisor=supervisor,
                worker=worker,
                max_iterations=args.max_iterations,
                log_dir=example_dir,
                label=example.target,
            )
            prediction = str(result.answer)
            sc = task.score(prediction, example)
            print(f"  Score: {sc} | Iters: {result.iterations} | {result.elapsed:.1f}s")

        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()
            prediction = f"ERROR: {e}"
            sc = 0

        rows.append({
            "id": example.id,
            "doc_id": example.metadata.get("doc_id", ""),
            "question": example.query,
            "log_path": os.path.relpath(example_dir, experiment_dir),
            "label": example.target,
            "prediction": prediction,
            "score": sc,
        })

    # --- Save results ---
    df = pd.DataFrame(rows)
    csv_path = os.path.join(experiment_dir, "results.csv")
    df.to_csv(csv_path, index=False)

    print(f"\n{'='*60}")
    print(f"Results saved to: {csv_path}")
    print(f"Accuracy: {df['score'].mean():.2%} ({int(df['score'].sum())}/{len(df)})")


if __name__ == "__main__":
    main()
