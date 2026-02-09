# Supervisor Distillation

A minimal supervisor-worker deep research system. A large "supervisor" model writes Python programs that define how a small "worker" model processes long contexts — the supervisor's code *is* the research strategy.

## How It Works

1. The **supervisor** (large model) receives a question and metadata about a long document
2. It writes Python code in ` ```repl``` ` blocks that gets executed in a sandboxed REPL
3. The code can call **worker** (small model) functions to read and summarize chunks of the document
4. The supervisor iterates — inspecting results, refining its approach — until it calls `FINAL(answer)`

```
supervisor writes code → REPL executes it → worker processes chunks → supervisor aggregates → FINAL
```

## File Structure

```
supervisor_distillation/
├── pyproject.toml              # Project metadata and dependencies
├── distill/
│   ├── __init__.py             # Exports: run, ModelHandler, OpenAIHandler
│   ├── models.py               # ModelHandler ABC + OpenAIHandler (any OpenAI-compatible API)
│   ├── repl.py                 # Sandboxed REPL with worker primitives
│   ├── prompts.py              # Supervisor system prompt and message builders
│   ├── log.py                  # Streaming JSONL logger (3 files)
│   └── orchestrator.py         # Main loop: prompt → code → execute → repeat
├── examples/
│   └── basic_qa.py             # Minimal usage example
├── idea.md                     # Research concept and reading list
└── plan.md                     # Implementation plan
```

### Module Overview

**`distill/models.py`** — Model handler abstraction. `ModelHandler` is an ABC with two methods: `chat(messages)` and `chat_batch(message_batches)`. `OpenAIHandler` implements this using the `openai` SDK with a configurable `base_url`, so it works with Together AI, OpenAI, or any compatible endpoint.

**`distill/repl.py`** — Sandboxed Python REPL that the supervisor's code runs in. The namespace exposes five primitives: `context` (the document), `query` (the question), `worker(prompt)` (call the worker model), `worker_batch(prompts)` (parallel worker calls), and `FINAL(answer)` (signal completion). Dangerous builtins (`eval`, `exec`, `compile`, `input`) are blocked. The namespace persists across code blocks within a run.

**`distill/prompts.py`** — Builds the supervisor's system prompt, which teaches it the available primitives and a chunk-query-aggregate strategy. Also provides iteration-specific nudges that become more urgent as the step limit approaches.

**`distill/log.py`** — `RunLogger` writes events to three JSONL files as they happen (append + flush). Every event has `step` and `timestamp`. Model events additionally include `model`, `usage`, and `elapsed`.

**`distill/orchestrator.py`** — The `run()` function ties everything together. It creates the REPL and logger, then loops: call supervisor → extract ` ```repl``` ` blocks → execute in REPL → log everything → append truncated output to message history → repeat until `FINAL()` or max iterations.

## Setup

```bash
pip install -e .
```

Dependencies: `openai>=1.0`, `python-dotenv`.

Set your API key as an environment variable or in a `.env` file:

```bash
# For Together AI (default endpoint)
export TOGETHER_API_KEY=your-key-here

# Or for OpenAI
export OPENAI_API_KEY=your-key-here
```

## Usage

### Quick Start

```python
from distill import OpenAIHandler, run

supervisor = OpenAIHandler(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
    base_url="https://api.together.xyz/v1",
    temperature=0.7,
    max_tokens=2048,
)
worker = OpenAIHandler(
    model="meta-llama/Llama-3.1-8B-Instruct-Turbo",
    base_url="https://api.together.xyz/v1",
    temperature=0.2,
    max_tokens=512,
)

result = run(
    query="What are the key findings in this paper?",
    context=long_document_text,
    supervisor=supervisor,
    worker=worker,
    log_dir="logs/my_run",
)

print(result.answer)
```

### Running the Example

```bash
python examples/basic_qa.py
```

Configure via environment variables:

| Variable | Default | Description |
|---|---|---|
| `TOGETHER_API_KEY` | — | API key (also checks `OPENAI_API_KEY`) |
| `SUPERVISOR_MODEL` | `meta-llama/Llama-3.3-70B-Instruct-Turbo` | Large model |
| `WORKER_MODEL` | `meta-llama/Llama-3.1-8B-Instruct-Turbo` | Small model |
| `BASE_URL` | `https://api.together.xyz/v1` | API endpoint |
| `LOG_DIR` | `logs/basic_qa` | Log output directory |

### `run()` Parameters

```python
run(
    query="...",               # The question to answer
    context="...",             # Long document (str, list, or dict)
    supervisor=sup_handler,    # ModelHandler for the large model
    worker=wrk_handler,        # ModelHandler for the small model
    max_iterations=15,         # Max supervisor turns before forcing FINAL
    log_dir="logs/run_001",    # JSONL log directory (None to disable)
    output_limit=2000,         # Max chars of REPL output shown to supervisor
    worker_ctx_k=8,            # Worker context window hint (thousands of tokens)
)
```

Returns a `RunResult` with: `answer`, `iterations`, `supervisor_usage`, `worker_usage`, `elapsed`.

### Logs

When `log_dir` is set, three JSONL files are written in real time:

```
logs/my_run/
├── supervisor.jsonl   # Supervisor prompts and responses
├── worker.jsonl       # Worker prompts and responses
└── repl.jsonl         # Code executed and full stdout/stderr
```

REPL output in `repl.jsonl` is the full, untruncated output — the supervisor only sees the first `output_limit` characters, but the logs capture everything.

## Extending the System

### Adding a New Model Backend

Subclass `ModelHandler` and implement `chat` and `chat_batch`:

```python
from distill.models import ModelHandler, LMResponse, Usage

class VLLMHandler(ModelHandler):
    """Offline inference via vLLM."""

    def __init__(self, model_path: str, **kwargs):
        from vllm import LLM
        self.llm = LLM(model=model_path, **kwargs)
        self.total_usage = Usage()

    def chat(self, messages, **kwargs):
        # Convert messages to a single prompt, call self.llm.generate(),
        # return LMResponse(text=..., usage=..., model=..., elapsed=...)
        ...

    def chat_batch(self, message_batches, **kwargs):
        # vLLM natively supports batching — pass all prompts at once
        ...
```

Then use it like any other handler:

```python
supervisor = VLLMHandler("/path/to/llama-70b")
worker = VLLMHandler("/path/to/llama-8b")
result = run(query=..., context=..., supervisor=supervisor, worker=worker)
```

### Customizing the Supervisor Prompt

Edit `SYSTEM_PROMPT` in `distill/prompts.py`, or override the message construction:

```python
from distill.orchestrator import run
from distill.prompts import build_system_prompt

# Option 1: Modify the template in prompts.py directly

# Option 2: Build custom messages and pass to the orchestrator internals
# The orchestrator uses build_system_prompt() to create the initial messages —
# you can modify that function to inject domain-specific instructions,
# few-shot examples, or different chunking strategies.
```

The supervisor prompt uses three format variables: `{worker_ctx}` (worker context window in K), `{worker_chunk}` (max chars per worker prompt), and `{output_limit}` (REPL output truncation limit). These are filled in by `build_system_prompt()`.

### Adding REPL Primitives

To give the supervisor access to new tools (e.g., web search, retrieval), add them to the REPL namespace in `distill/repl.py`:

```python
# In REPL.__init__, add to self._namespace:
self._namespace = {
    ...
    "search": search_fn,           # web search primitive
    "retrieve": retrieval_fn,      # vector DB retrieval
    "supervisor": escalate_fn,     # escalate to supervisor (RLM-style)
}
```

Then update the system prompt in `distill/prompts.py` to document the new primitives so the supervisor knows how to use them.

### Using the Logs for Distillation

The JSONL logs capture complete supervisor-worker interaction traces — these are the training data for distillation experiments:

- **Few-shot demonstrations**: Sample (context_chunk, worker_response) pairs from `worker.jsonl` to construct few-shot prompts
- **SFT / imitation learning**: Use supervisor code traces from `supervisor.jsonl` as targets for fine-tuning a smaller model to mimic the supervisor's chunking and aggregation strategy
- **On-policy distillation**: Run the system, collect logs, fine-tune the worker, then re-run with the updated worker to generate new on-policy data
- **RL with compression objective**: Use `worker.jsonl` entries as episodes — the worker's response is the "compression" of the chunk, and downstream answer quality (from `supervisor.jsonl`) provides the reward signal
