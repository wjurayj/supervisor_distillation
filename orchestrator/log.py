"""Streaming JSONL logger writing events to 3 separate files as they happen."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from orchestrator.models import LMResponse
    from orchestrator.repl import ExecResult


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write(f, obj: dict) -> None:
    f.write(json.dumps(obj) + "\n")
    f.flush()


class RunLogger:
    """Writes events to supervisor.jsonl, worker.jsonl, repl.jsonl, and task.jsonl."""

    def __init__(self, log_dir: str):
        os.makedirs(log_dir, exist_ok=True)
        self._log_dir = log_dir
        self._supervisor_f = open(os.path.join(log_dir, "supervisor.jsonl"), "a")
        self._worker_f = open(os.path.join(log_dir, "worker.jsonl"), "a")
        self._repl_f = open(os.path.join(log_dir, "repl.jsonl"), "a")
        self._task_f = open(os.path.join(log_dir, "task.jsonl"), "a")

    def log_task_input(
        self, query: str, context: Any, label: str | None = None, *, features: Any = None,
    ) -> None:
        """Log the task input (query, context, optional label and features)."""
        from dataclasses import asdict
        entry: dict[str, Any] = {
            "type": "input",
            "query": query,
            "context": context if isinstance(context, str) else str(context),
            "label": label or "",
            "timestamp": _now(),
        }
        if features is not None:
            entry["features"] = asdict(features)
        _write(self._task_f, entry)

    def log_task_output(self, answer: Any) -> None:
        """Log the task output (FINAL answer)."""
        _write(self._task_f, {
            "type": "output",
            "answer": answer if isinstance(answer, str) else str(answer),
            "timestamp": _now(),
        })

    def log_supervisor(self, step: int, messages: list[dict], response: LMResponse) -> None:
        _write(self._supervisor_f, {
            "step": step,
            "timestamp": _now(),
            "model": response.model,
            "messages": messages,
            "response": response.text,
            "usage": {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens},
            "elapsed": response.elapsed,
        })

    def log_worker(self, step: int, prompt: str, response: LMResponse) -> None:
        _write(self._worker_f, {
            "step": step,
            "timestamp": _now(),
            "model": response.model,
            "prompt": prompt,
            "response": response.text,
            "usage": {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens},
            "elapsed": response.elapsed,
        })

    def log_worker_batch(self, step: int, prompts: list[str], responses: list[LMResponse]) -> None:
        for prompt, response in zip(prompts, responses):
            self.log_worker(step, prompt, response)

    def log_repl(self, step: int, code: str, result: ExecResult) -> None:
        _write(self._repl_f, {
            "step": step,
            "timestamp": _now(),
            "code": code,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "elapsed": result.elapsed,
        })

    def close(self) -> None:
        for f in (self._supervisor_f, self._worker_f, self._repl_f, self._task_f):
            f.close()
