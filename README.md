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
├── orchestrator/
│   ├── __init__.py             # Exports: run, ModelHandler, OpenAIHandler
│   ├── models.py               # ModelHandler ABC + OpenAIHandler (any OpenAI-compatible API)
│   ├── repl.py                 # Sandboxed REPL with worker primitives
│   ├── prompts.py              # Supervisor system prompt and message builders
│   ├── log.py                  # Streaming JSONL logger (3 files)
│   └── orchestrator.py         # Main loop: prompt → code → execute → repeat
├── examples/
│   ├── basic_qa.py             # Minimal usage example
│   └── run_experiment.py       # Benchmark runner (QASPER, etc.)
├── experiments/
│   └── ib_reproduce.sh         # Grid over supervisor/worker pairs
├── utils/
│   └── analysis.py             # Aggregate logs into a results table
├── tasks/
│   └── qasper/                 # QASPER task definition and scoring
├── idea.md                     # Research concept and reading list
└── plan.md                     # Implementation plan
```

### Module Overview

**`orchestrator/models.py`** — Model handler abstraction. `ModelHandler` is an ABC with two methods: `chat(messages)` and `chat_batch(message_batches)`. `OpenAIHandler` implements this using the `openai` SDK with a configurable `base_url`, so it works with Together AI, OpenAI, or any compatible endpoint. `VLLMHandler` provides offline inference via the vLLM engine.

**`orchestrator/repl.py`** — Sandboxed Python REPL that the supervisor's code runs in. The namespace exposes five primitives: `context` (the document), `query` (the question), `worker(prompt)` (call the worker model), `worker_batch(prompts)` (parallel worker calls), and `FINAL(answer)` (signal completion). Dangerous builtins (`eval`, `exec`, `compile`, `input`) are blocked. The namespace persists across code blocks within a run.

**`orchestrator/prompts.py`** — Builds the supervisor's system prompt, which teaches it the available primitives and a chunk-query-aggregate strategy. Also provides iteration-specific nudges that become more urgent as the step limit approaches.

**`orchestrator/log.py`** — `RunLogger` writes events to three JSONL files as they happen (append + flush). Every event has `step` and `timestamp`. Model events additionally include `model`, `usage`, and `elapsed`.

**`orchestrator/orchestrator.py`** — The `run()` function ties everything together. It creates the REPL and logger, then loops: call supervisor → extract ` ```repl``` ` blocks → execute in REPL → log everything → append truncated output to message history → repeat until `FINAL()` or max iterations.

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
from orchestrator import OpenAIHandler, run

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

## Experiments

### QASPER Benchmark

The QASPER task evaluates question-answering over NLP papers. `examples/run_experiment.py` runs the full supervisor-worker pipeline on a set of examples and saves per-question results.

```bash
# Single run with specific models
python examples/run_experiment.py \
  --supervisor-model "Qwen/Qwen3-235B-A22B-Instruct-2507-tput" \
  --worker-model "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo" \
  --seed 42 -n 20
```

| Flag | Default | Description |
|---|---|---|
| `--task` | `qasper` | Task name |
| `--supervisor-model` | `$SUPERVISOR_MODEL` or Llama-3.3-70B | Supervisor model |
| `--worker-model` | `$WORKER_MODEL` or Llama-3.1-8B | Worker model |
| `--judge-model` | Qwen3-235B | LLM judge for scoring |
| `-n` | 20 | Number of examples (`-1` for all) |
| `--seed` | 42 | Random seed for sampling |
| `--max-iterations` | 15 | Max supervisor turns per example |

Results are saved to `logs/<task>/<timestamp>/results.csv` alongside per-example log subdirectories.

### Running a Grid Experiment

`experiments/ib_reproduce.sh` runs all combinations of two supervisor and two worker models:

```bash
bash experiments/ib_reproduce.sh
```

This produces one timestamped run directory per combination under `logs/qasper/`.

### Analyzing Results

`utils/analysis.py` aggregates logs from one or more runs into a summary table with accuracy, step counts, and token usage.

```bash
# Analyze all runs under a parent directory
python utils/analysis.py logs/qasper

# Analyze specific run directories
python utils/analysis.py logs/qasper/20260211-163519 logs/qasper/20260211-170952
```

Example output:

```
Supervisor                         | Worker                           |   Accuracy | Avg Steps | Total Steps |  Sup In | Sup Out |  Wrk In | Wrk Out
-----------------------------------+----------------------------------+------------+-----------+-------------+---------+---------+---------+--------
Qwen3-235B-A22B-Instruct-2507-tput | Meta-Llama-3.1-8B-Instruct-Turbo | 40% (8/20) |       6.0 |         121 | 243,669 |  19,505 | 210,288 |  18,265
Llama-3.3-70B-Instruct-Turbo       | Qwen2.5-7B-Instruct-Turbo        | 20% (4/20) |       3.5 |          71 |  58,555 |  10,732 | 130,933 | 190,699
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for notes on adding new model backends, customizing prompts, adding REPL primitives, and using the logs for distillation.
