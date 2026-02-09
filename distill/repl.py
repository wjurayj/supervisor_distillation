"""Sandboxed REPL with worker primitives for supervisor-generated code."""

from __future__ import annotations

import io
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Any, Callable


class FinalSignal(Exception):
    """Raised when FINAL() is called to signal completion."""

    def __init__(self, answer: Any):
        self.answer = answer


@dataclass
class ExecResult:
    stdout: str
    stderr: str
    variables: dict[str, Any]
    elapsed: float


_BLOCKED_BUILTINS = {"eval", "exec", "compile", "input"}


def _make_safe_builtins() -> dict:
    import builtins
    safe = {k: v for k, v in vars(builtins).items() if k not in _BLOCKED_BUILTINS}
    return safe


class REPL:
    """Sandboxed Python REPL with worker model primitives.

    Primitives available in the namespace:
        context   — the long document (str, list, or dict)
        query     — the user's question
        worker(prompt) -> str — call worker model
        worker_batch(prompts) -> list[str] — batch call worker model
        FINAL(answer) — signal completion
    """

    def __init__(
        self,
        context: Any,
        query: str,
        worker_fn: Callable[[str], str],
        worker_batch_fn: Callable[[list[str]], list[str]],
        output_limit: int = 2000,
    ):
        self.output_limit = output_limit
        self._namespace: dict[str, Any] = {
            "__builtins__": _make_safe_builtins(),
            "context": context,
            "query": query,
            "worker": worker_fn,
            "worker_batch": worker_batch_fn,
            "FINAL": self._final,
        }
        self._final_answer: Any = None

    def _final(self, answer: Any) -> None:
        raise FinalSignal(answer)

    @property
    def final_answer(self) -> Any:
        return self._final_answer

    def execute(self, code: str) -> ExecResult:
        """Execute code in the sandboxed namespace. Returns ExecResult with full output."""
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = stdout_buf, stderr_buf

        t0 = time.perf_counter()
        try:
            exec(code, self._namespace)
        except FinalSignal as fs:
            self._final_answer = fs.answer
        except Exception:
            stderr_buf.write(traceback.format_exc())
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr
            elapsed = time.perf_counter() - t0

        return ExecResult(
            stdout=stdout_buf.getvalue(),
            stderr=stderr_buf.getvalue(),
            variables={k: v for k, v in self._namespace.items() if not k.startswith("_")},
            elapsed=elapsed,
        )

    def truncate_output(self, result: ExecResult) -> str:
        """Return combined stdout+stderr truncated to output_limit for LM context."""
        parts = []
        if result.stdout:
            parts.append(result.stdout)
        if result.stderr:
            parts.append(f"[stderr]\n{result.stderr}")
        combined = "\n".join(parts) if parts else "(no output)"
        if len(combined) > self.output_limit:
            combined = combined[: self.output_limit] + f"\n... [truncated to {self.output_limit} chars]"
        return combined
