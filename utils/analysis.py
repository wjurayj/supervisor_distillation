#!/usr/bin/env python3
"""Aggregate and display experiment results from log directories.

Usage:
    # Pass a parent directory (each subfolder is a run):
    python utils/analysis.py logs/qasper

    # Pass individual run directories:
    python utils/analysis.py logs/qasper/20260211-163519 logs/qasper/20260211-164854
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass, field
from glob import glob
from pathlib import Path


@dataclass
class TokenStats:
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class RunStats:
    run_dir: str
    supervisor_model: str = ""
    worker_model: str = ""
    accuracy: float = 0.0
    correct: int = 0
    total: int = 0
    total_steps: int = 0
    supervisor: TokenStats = field(default_factory=TokenStats)
    worker: TokenStats = field(default_factory=TokenStats)

    @property
    def avg_steps(self) -> float:
        return self.total_steps / self.total if self.total else 0.0


def _read_jsonl_usage(path: str) -> TokenStats:
    """Sum token usage across all entries in a JSONL log file."""
    stats = TokenStats()
    if not os.path.exists(path):
        return stats
    with open(path) as f:
        for line in f:
            usage = json.loads(line).get("usage", {})
            stats.input_tokens += usage.get("input_tokens", 0)
            stats.output_tokens += usage.get("output_tokens", 0)
    return stats


def _read_model_name(path: str) -> str:
    """Read the model name from the first entry of a JSONL log file."""
    if not os.path.exists(path):
        return ""
    with open(path) as f:
        first = f.readline()
        if first:
            return json.loads(first).get("model", "")
    return ""


def _count_steps(path: str) -> int:
    """Count supervisor steps (one per JSONL line)."""
    if not os.path.exists(path):
        return 0
    with open(path) as f:
        return sum(1 for _ in f)


def analyze_run(run_dir: str) -> RunStats | None:
    """Analyze a single run directory containing results.csv and example subdirs."""
    csv_path = os.path.join(run_dir, "results.csv")
    if not os.path.exists(csv_path):
        return None

    stats = RunStats(run_dir=run_dir)

    # Read accuracy from results.csv
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    stats.total = len(rows)
    stats.correct = sum(1 for r in rows if float(r["score"]) > 0)
    stats.accuracy = stats.correct / stats.total if stats.total else 0.0

    # Aggregate from example subdirectories
    example_dirs = sorted(glob(os.path.join(run_dir, "[0-9][0-9][0-9]")))
    for ex_dir in example_dirs:
        sup_file = os.path.join(ex_dir, "supervisor.jsonl")
        wrk_file = os.path.join(ex_dir, "worker.jsonl")

        # Model names from first example that has them
        if not stats.supervisor_model:
            stats.supervisor_model = _read_model_name(sup_file)
        if not stats.worker_model:
            stats.worker_model = _read_model_name(wrk_file)

        sup_usage = _read_jsonl_usage(sup_file)
        wrk_usage = _read_jsonl_usage(wrk_file)
        stats.supervisor.input_tokens += sup_usage.input_tokens
        stats.supervisor.output_tokens += sup_usage.output_tokens
        stats.worker.input_tokens += wrk_usage.input_tokens
        stats.worker.output_tokens += wrk_usage.output_tokens
        stats.total_steps += _count_steps(sup_file)

    return stats


def resolve_run_dirs(paths: list[str]) -> list[str]:
    """Given CLI paths, return a flat list of run directories.

    If a path itself contains results.csv, treat it as a run dir.
    Otherwise treat it as a parent and use its immediate subdirectories.
    """
    run_dirs: list[str] = []
    for p in paths:
        if os.path.isfile(os.path.join(p, "results.csv")):
            run_dirs.append(p)
        elif os.path.isdir(p):
            for child in sorted(os.listdir(p)):
                child_path = os.path.join(p, child)
                if os.path.isdir(child_path) and os.path.isfile(
                    os.path.join(child_path, "results.csv")
                ):
                    run_dirs.append(child_path)
    return run_dirs


def shorten_model(name: str) -> str:
    """Strip org prefix for display (e.g. 'meta-llama/Llama-3.3-70B-Instruct-Turbo' -> 'Llama-3.3-70B-Instruct-Turbo')."""
    return name.split("/")[-1] if "/" in name else name


def fmt_tokens(n: int) -> str:
    """Format token count with comma separators."""
    return f"{n:,}"


def print_table(all_stats: list[RunStats]) -> None:
    headers = [
        "Supervisor",
        "Worker",
        "Accuracy",
        "Avg Steps",
        "Total Steps",
        "Sup In",
        "Sup Out",
        "Wrk In",
        "Wrk Out",
    ]

    rows: list[list[str]] = []
    for s in all_stats:
        rows.append([
            shorten_model(s.supervisor_model),
            shorten_model(s.worker_model),
            f"{s.accuracy:.0%} ({s.correct}/{s.total})",
            f"{s.avg_steps:.1f}",
            str(s.total_steps),
            fmt_tokens(s.supervisor.input_tokens),
            fmt_tokens(s.supervisor.output_tokens),
            fmt_tokens(s.worker.input_tokens),
            fmt_tokens(s.worker.output_tokens),
        ])

    # Compute column widths
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    def fmt_row(cells: list[str]) -> str:
        parts = []
        for i, cell in enumerate(cells):
            # Right-align numeric columns (index 2+), left-align model names
            if i < 2:
                parts.append(cell.ljust(col_widths[i]))
            else:
                parts.append(cell.rjust(col_widths[i]))
        return " | ".join(parts)

    sep = "-+-".join("-" * w for w in col_widths)

    print(fmt_row(headers))
    print(sep)
    for row in rows:
        print(fmt_row(row))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze experiment run directories and print a results table."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Run directories (with results.csv) or parent directories containing them.",
    )
    args = parser.parse_args()

    run_dirs = resolve_run_dirs(args.paths)
    if not run_dirs:
        print("No run directories found.", file=sys.stderr)
        sys.exit(1)

    all_stats: list[RunStats] = []
    for d in run_dirs:
        s = analyze_run(d)
        if s:
            all_stats.append(s)

    if not all_stats:
        print("No valid results found.", file=sys.stderr)
        sys.exit(1)

    print_table(all_stats)


if __name__ == "__main__":
    main()
