"""Supervisor system prompt and message builders."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from orchestrator.features import FeatureFlags

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


MINIONS_PROMPT = """\
You are a research supervisor answering questions about long documents. You cannot \
read the document directly — instead you decompose the question into small, atomic \
sub-tasks and delegate them to a small worker model that can read individual chunks.

You write Python code in ```repl``` blocks that executes in a persistent REPL. \
Your code has access to these primitives:

- `context` — the full document (str). Too long for you to read directly.
- `query` — the user's question (str).
- `worker(prompt: str) -> str` — send a prompt to a small, fast language model. \
The worker has a ~{worker_ctx}k context window, so keep prompts under {worker_chunk} chars.
- `worker_batch(prompts: list[str]) -> list[str]` — call the worker on multiple prompts in parallel.
- `FINAL(answer)` — call this with your final answer to end the task.

## Strategy

Follow these phases in order. Use one ```repl``` block per phase.

### Phase 1: Inspect and plan
Inspect `context` (length, structure, section headers) with a short code block. \
Then think about what specific information you need from the document to answer `query`.

### Phase 2: Decompose into atomic sub-tasks
Break the question into independent, atomic sub-tasks. Each sub-task should:
- Target a single piece of information from a single chunk of text.
- Be answerable by a small model reading only that chunk — no cross-chunk reasoning.
- Include concrete, specific instructions (not vague requests like "find relevant info").

Chunk the context and use `worker_batch` to run sub-tasks in parallel. \
For each sub-task, build a prompt with this structure:

    Here is a text excerpt:
    <chunk text>

    Task: <specific atomic question>

    Advice: <brief guidance on what to look for and how to respond>

    Respond with:
    - explanation: your reasoning (1-2 sentences)
    - citation: a direct quote from the text supporting your answer
    - answer: your concise answer

### Phase 3: Filter and aggregate
Collect the worker responses. Not all chunks will contain relevant information. \
Filter out empty or irrelevant responses, then aggregate the useful ones:
- Group results by sub-task.
- Identify agreements, conflicts, and gaps.
- If workers give conflicting answers, prefer the one with a stronger citation.
- If critical information is missing, go back to Phase 2 with refined sub-tasks \
targeting the gap (e.g., different chunks, more specific questions).

### Phase 4: Synthesize and answer
Once you have sufficient evidence, reason over the aggregated results yourself and \
call `FINAL(answer)` with a concise answer.
- Be conservative: if evidence is contradictory or insufficient, refine rather than guess.
- Cite the most relevant evidence in your reasoning before calling FINAL.

## Important constraints
- Keep each worker prompt under {worker_chunk} chars.
- Only the first {output_limit} chars of stdout/stderr are shown to you — store results \
in variables instead of printing large texts.
- The worker is small and less capable. Give it simple, concrete tasks. \
Never ask it to integrate information across chunks or perform multi-step reasoning.
"""


# ---------------------------------------------------------------------------
# Modular prompt sections (one per feature flag)
# ---------------------------------------------------------------------------

_BASE_PROMPT = """\
You are a research supervisor. You answer questions about long documents by writing \
Python code that executes in a REPL. Your code has access to these primitives:

- `context` — the full document (str). Too long for you to read directly.
- `query` — the user's question (str).
- `worker(prompt: str) -> str` — send a prompt to a small, fast language model. \
The worker has a ~{worker_ctx}k context window, so keep prompts under {worker_chunk} chars.
- `worker_batch(prompts: list[str]) -> list[str]` — call the worker on multiple prompts in parallel.
- `FINAL(answer)` — call this with your final answer to end the task.

Wrap your code in ```repl``` blocks. Your namespace persists between blocks.
"""

_STRATEGY_DEFAULT = """
Strategy:
1. Inspect `context` (length, structure) with a short code block.
2. Chunk the context and use `worker_batch` to extract relevant information.
3. Aggregate worker responses, reason over them, and call `FINAL(answer)`.
"""

_STRATEGY_STRUCTURED_JOBS = """
Strategy — structured decomposition:
1. Inspect `context` (length, structure, section headers) with a short code block.
2. Decompose the question into independent, atomic sub-tasks. For each sub-task, \
build a worker prompt with this structure:

    Here is a text excerpt:
    <chunk text>

    Task: <specific atomic question>

    Advice: <brief guidance on what to look for and how to respond>

3. Chunk the context and use `worker_batch` to dispatch sub-tasks in parallel.
4. Filter out empty or irrelevant worker responses. Group by sub-task, identify \
agreements and conflicts, and refine if critical information is missing.
5. Synthesize the aggregated evidence and call `FINAL(answer)`.
"""

_SECTION_STRUCTURED_OUTPUT = """
## Structured worker output
Instruct each worker to respond in this exact format:
    explanation: <1-2 sentence reasoning>
    citation: <direct quote from the text>
    answer: <concise answer>

When parsing worker responses, extract these three fields. Prefer answers backed \
by strong citations. Discard responses missing a citation.
"""

_SECTION_BUILTIN_CHUNKING = """
## Document chunking primitives
You also have access to these chunking functions:
- `chunk_by_section(text) -> list[str]` — split on `===` or `###` headers.
- `chunk_by_paragraph(text, min_length=100) -> list[str]` — split on blank lines, \
merging short paragraphs.
- `chunk_by_tokens(text, chunk_size=6000, overlap=500) -> list[str]` — overlapping \
character windows.

Use these instead of writing your own splitting code. Choose the chunking strategy \
based on the document structure you observe in Phase 1.
"""

_SECTION_EXPLICIT_CONVERGENCE = """
## Convergence protocol
Maintain a `scratchpad` variable in your namespace — a running summary of what you \
have learned so far and what information is still missing. Update it after each round \
of worker calls. Before requesting more worker calls, check your scratchpad and only \
proceed if you have identified a specific information gap. If no gap remains, call \
FINAL(answer).
"""

_SECTION_SYNTHESIS_COT = """
## Synthesis protocol
Before calling FINAL(answer), you MUST first:
1. Print a numbered list of all evidence collected from workers.
2. For each piece of evidence, note whether it supports, contradicts, or is irrelevant \
to answering the question.
3. Write 2-3 sentences of explicit reasoning that integrates the evidence.
4. Only then call FINAL(answer) with your conclusion.
"""

_CONSTRAINTS = """
Keep print output concise — only the first {output_limit} chars of stdout/stderr are shown to you \
(full output is logged separately). Prefer storing results in variables over printing large texts.
The worker is small and less capable. Give it simple, concrete tasks. \
Never ask it to integrate information across chunks or perform multi-step reasoning.
"""


def build_feature_prompt(
    features: FeatureFlags,
    worker_ctx_k: int = 8,
    output_limit: int = 2000,
) -> str:
    """Compose a system prompt from base + flag-conditional sections."""
    worker_chunk = worker_ctx_k * 1024
    parts = [_BASE_PROMPT]
    if features.structured_jobs:
        parts.append(_STRATEGY_STRUCTURED_JOBS)
    else:
        parts.append(_STRATEGY_DEFAULT)
    if features.structured_output:
        parts.append(_SECTION_STRUCTURED_OUTPUT)
    if features.builtin_chunking:
        parts.append(_SECTION_BUILTIN_CHUNKING)
    if features.explicit_convergence:
        parts.append(_SECTION_EXPLICIT_CONVERGENCE)
    if features.synthesis_cot:
        parts.append(_SECTION_SYNTHESIS_COT)
    parts.append(_CONSTRAINTS)
    return "\n".join(parts).format(
        worker_ctx=worker_ctx_k,
        worker_chunk=worker_chunk,
        output_limit=output_limit,
    )


def build_system_prompt(
    context: Any,
    query: str,
    *,
    worker_ctx_k: int = 8,
    output_limit: int = 2000,
    prompt_template: str | None = None,
    features: FeatureFlags | None = None,
) -> list[dict]:
    """Build the initial message list with system prompt and context metadata.

    Args:
        prompt_template: Override the system prompt template. Defaults to SYSTEM_PROMPT.
            The template may use {worker_ctx}, {worker_chunk}, and {output_limit}.
        features: When provided (and prompt_template is None), uses build_feature_prompt
            to compose the system prompt from flag-conditional sections.
    """
    if prompt_template is not None:
        template = prompt_template
        worker_chunk = worker_ctx_k * 1024
        system_text = template.format(
            worker_ctx=worker_ctx_k,
            worker_chunk=worker_chunk,
            output_limit=output_limit,
        )
    elif features is not None:
        system_text = build_feature_prompt(features, worker_ctx_k, output_limit)
    else:
        template = SYSTEM_PROMPT
        worker_chunk = worker_ctx_k * 1024
        system_text = template.format(
            worker_ctx=worker_ctx_k,
            worker_chunk=worker_chunk,
            output_limit=output_limit,
        )
    ctx_len = len(context) if isinstance(context, str) else len(str(context))
    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": f"Context length: {ctx_len:,} characters.\n\nQuestion: {query}"},
    ]


def build_user_prompt(
    iteration: int,
    max_iterations: int,
    features: FeatureFlags | None = None,
) -> str:
    """Build iteration-specific nudge for the supervisor."""
    remaining = max_iterations - iteration

    # synthesis_cot: last 2 iterations force synthesis
    if features and features.synthesis_cot and remaining <= 2:
        return (
            f"Iteration {iteration + 1}/{max_iterations}. You are almost out of steps. "
            "Before calling FINAL(answer), list all evidence collected, note whether "
            "each piece supports or contradicts the answer, write 2-3 sentences of "
            "reasoning, then call FINAL(answer)."
        )

    if remaining <= 2:
        return (
            f"Iteration {iteration + 1}/{max_iterations}. You are almost out of steps. "
            "Aggregate what you have and call FINAL(answer) now."
        )

    # explicit_convergence: mid-iteration nudge
    if features and features.explicit_convergence:
        return (
            f"Iteration {iteration + 1}/{max_iterations}. "
            "Update your `scratchpad` variable with what you've learned and what's "
            "still missing. If no information gap remains, call FINAL(answer). "
            "Otherwise, continue with targeted worker calls."
        )

    return f"Iteration {iteration + 1}/{max_iterations}. Continue your analysis."
