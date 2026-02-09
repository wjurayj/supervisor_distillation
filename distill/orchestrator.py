"""Main orchestrator loop: supervisor generates code, REPL executes it with worker calls."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any

from distill.log import RunLogger
from distill.models import LMResponse, ModelHandler, Usage
from distill.prompts import build_system_prompt, build_user_prompt
from distill.repl import REPL

_CODE_BLOCK_RE = re.compile(r"```repl\s*\n(.*?)```", re.DOTALL)


@dataclass
class RunResult:
    answer: Any
    iterations: int
    supervisor_usage: Usage
    worker_usage: Usage
    elapsed: float


def run(
    query: str,
    context: Any,
    supervisor: ModelHandler,
    worker: ModelHandler,
    *,
    max_iterations: int = 15,
    log_dir: str | None = None,
    output_limit: int = 2000,
    worker_ctx_k: int = 8,
) -> RunResult:
    """Run a supervisor-worker deep research session.

    Args:
        query: The user's question.
        context: The long document (str, list, or dict).
        supervisor: ModelHandler for the large supervisor model.
        worker: ModelHandler for the small worker model.
        max_iterations: Maximum supervisor iterations before forcing a final answer.
        log_dir: Directory for JSONL logs. None disables logging.
        output_limit: Max chars of REPL output included in LM context.
        worker_ctx_k: Worker context window size in thousands of tokens (for prompt hints).
    """
    logger = RunLogger(log_dir) if log_dir else None
    t0 = time.perf_counter()
    step = 0

    # Track worker usage separately since worker calls happen inside REPL
    worker_usage = Usage()

    def _worker_fn(prompt: str) -> str:
        nonlocal worker_usage
        msgs = [{"role": "user", "content": prompt}]
        resp = worker.chat(msgs)
        worker_usage += resp.usage
        if logger:
            logger.log_worker(step, prompt, resp)
        return resp.text

    def _worker_batch_fn(prompts: list[str]) -> list[str]:
        nonlocal worker_usage
        batches = [[{"role": "user", "content": p}] for p in prompts]
        responses = worker.chat_batch(batches)
        for prompt, resp in zip(prompts, responses):
            worker_usage += resp.usage
            if logger:
                logger.log_worker(step, prompt, resp)
        return [r.text for r in responses]

    repl = REPL(context, query, _worker_fn, _worker_batch_fn, output_limit=output_limit)
    messages = build_system_prompt(context, query, worker_ctx_k=worker_ctx_k, output_limit=output_limit)

    try:
        for step in range(max_iterations):
            # Call supervisor
            sup_resp = supervisor.chat(messages)
            if logger:
                logger.log_supervisor(step, messages, sup_resp)

            # Extract code blocks
            code_blocks = _CODE_BLOCK_RE.findall(sup_resp.text)

            if not code_blocks:
                # No code blocks — supervisor responded with text only.
                # Append as assistant turn and nudge to write code.
                messages.append({"role": "assistant", "content": sup_resp.text})
                messages.append({"role": "user", "content": "Please write a ```repl``` code block to proceed."})
                continue

            # Execute each code block
            all_output_parts = []
            for code in code_blocks:
                result = repl.execute(code)
                if logger:
                    logger.log_repl(step, code, result)

                truncated = repl.truncate_output(result)
                all_output_parts.append(f"Code:\n```python\n{code}```\nOutput:\n{truncated}")

                if repl.final_answer is not None:
                    return RunResult(
                        answer=repl.final_answer,
                        iterations=step + 1,
                        supervisor_usage=supervisor.total_usage if hasattr(supervisor, "total_usage") else Usage(),
                        worker_usage=worker_usage,
                        elapsed=time.perf_counter() - t0,
                    )

            # Append code + output to message history
            messages.append({"role": "assistant", "content": sup_resp.text})
            execution_summary = "\n\n".join(all_output_parts)
            nudge = build_user_prompt(step, max_iterations)
            messages.append({"role": "user", "content": f"{execution_summary}\n\n{nudge}"})

        # Max iterations reached — force final answer
        messages.append({
            "role": "user",
            "content": "Maximum iterations reached. Call FINAL(answer) with your best answer now.",
        })
        final_resp = supervisor.chat(messages)
        if logger:
            logger.log_supervisor(step + 1, messages, final_resp)

        # Try to execute any final code
        code_blocks = _CODE_BLOCK_RE.findall(final_resp.text)
        for code in code_blocks:
            result = repl.execute(code)
            if logger:
                logger.log_repl(step + 1, code, result)
            if repl.final_answer is not None:
                break

        return RunResult(
            answer=repl.final_answer or final_resp.text,
            iterations=step + 1,
            supervisor_usage=supervisor.total_usage if hasattr(supervisor, "total_usage") else Usage(),
            worker_usage=worker_usage,
            elapsed=time.perf_counter() - t0,
        )
    finally:
        if logger:
            logger.close()
