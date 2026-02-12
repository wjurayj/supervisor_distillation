# Contributing

## Extending the System

### Adding a New Model Backend

Subclass `ModelHandler` and implement `chat` and `chat_batch`:

```python
from distill.models import ModelHandler, LMResponse, Usage

class MyHandler(ModelHandler):
    def __init__(self, model: str, **kwargs):
        self.model = model
        self.total_usage = Usage()

    def chat(self, messages, **kwargs) -> LMResponse:
        # Call your backend, return LMResponse(text=..., usage=..., model=..., elapsed=...)
        ...

    def chat_batch(self, message_batches, **kwargs) -> list[LMResponse]:
        ...
```

A `VLLMHandler` for offline vLLM inference is already provided in `distill/models.py`. See that implementation for a complete example.

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
