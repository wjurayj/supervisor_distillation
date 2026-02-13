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

from orchestrator import FeatureFlags, OpenAIHandler, run
from orchestrator.prompts import MINIONS_PROMPT


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
    p.add_argument("--num-distractors", type=int, default=0,
                   help="Number of distractor papers to add to context (default: 0)")
    p.add_argument("--max-iterations", type=int, default=15)
    p.add_argument("--prompt", default=None, choices=["default", "minions"],
                   help="Supervisor prompt template (default: default)")
    p.add_argument("--log-dir", default=None,
                   help="Override log directory (default: logs/<task>/<timestamp>)")
    # Feature flags for ablation experiments
    p.add_argument("--structured-jobs", action="store_true")
    p.add_argument("--structured-output", action="store_true")
    p.add_argument("--builtin-chunking", action="store_true")
    p.add_argument("--explicit-convergence", action="store_true")
    p.add_argument("--synthesis-cot", action="store_true")
    p.add_argument("--all-minions", action="store_true",
                   help="Turn on all 5 Minions-style feature flags")
    args = p.parse_args()

    # --- Build feature flags ---
    any_flag = (
        args.all_minions or args.structured_jobs or args.structured_output
        or args.builtin_chunking or args.explicit_convergence or args.synthesis_cot
    )
    if any_flag:
        if args.all_minions:
            features = FeatureFlags.all_on()
        else:
            features = FeatureFlags(
                structured_jobs=args.structured_jobs,
                structured_output=args.structured_output,
                builtin_chunking=args.builtin_chunking,
                explicit_convergence=args.explicit_convergence,
                synthesis_cot=args.synthesis_cot,
            )
    else:
        features = None

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
    if features:
        print(f"Features: {features.label()}")

    # --- Resolve prompt template ---
    prompt_template = None
    if args.prompt == "minions":
        prompt_template = MINIONS_PROMPT

    # --- Setup experiment directory ---
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.log_dir:
        experiment_dir = os.path.join(args.log_dir, timestamp)
    else:
        task_label = args.task if not args.prompt or args.prompt == "default" else f"{args.task}_{args.prompt}"
        if features:
            task_label = f"{task_label}_feat-{features.label()}"
        experiment_dir = os.path.join("logs", task_label, timestamp)
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
                prompt_template=prompt_template,
                features=features,
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
