"""Supervisor system prompt and message builders."""

from __future__ import annotations

from typing import Any

SYSTEM_PROMPT = """\
You are a research supervisor. You answer questions about long documents by writing \
Python code that executes in a REPL. Your code has access to these primitives:

- `context` — the full document (str). Too long for you to read directly.
- `query` — the user's question (str).
- `worker(prompt: str) -> str` — send a prompt to a small, fast language model. \
The worker has a ~{worker_ctx}k context window, so keep prompts under {worker_chunk} chars.
- `worker_batch(prompts: list[str]) -> list[str]` — call the worker on multiple prompts in parallel.
- `FINAL(answer)` — call this with your final answer to end the task.

Wrap your code in ```repl``` blocks. Your namespace persists between blocks.

Strategy:
1. Inspect `context` (length, structure) with a short code block.
2. Chunk the context and use `worker_batch` to extract relevant information.
3. Aggregate worker responses, reason over them, and call `FINAL(answer)`.

Keep print output concise — only the first {output_limit} chars of stdout/stderr are shown to you \
(full output is logged separately). Prefer storing results in variables over printing large texts.
"""


def build_system_prompt(
    context: Any,
    query: str,
    *,
    worker_ctx_k: int = 8,
    output_limit: int = 2000,
) -> list[dict]:
    """Build the initial message list with system prompt and context metadata."""
    worker_chunk = worker_ctx_k * 1024  # rough char estimate
    ctx_len = len(context) if isinstance(context, str) else len(str(context))
    system_text = SYSTEM_PROMPT.format(
        worker_ctx=worker_ctx_k,
        worker_chunk=worker_chunk,
        output_limit=output_limit,
    )
    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": f"Context length: {ctx_len:,} characters.\n\nQuestion: {query}"},
    ]


def build_user_prompt(iteration: int, max_iterations: int) -> str:
    """Build iteration-specific nudge for the supervisor."""
    remaining = max_iterations - iteration
    if remaining <= 2:
        return (
            f"Iteration {iteration + 1}/{max_iterations}. You are almost out of steps. "
            "Aggregate what you have and call FINAL(answer) now."
        )
    return f"Iteration {iteration + 1}/{max_iterations}. Continue your analysis."
