# Minions vs Distill: Key Differences

Comparison of the [minions](https://github.com/stanford-futuredata/minions) project's supervisor/worker protocol with this repo's `distill` orchestrator.

---

## 1. Supervisor control mechanism

| | Minions | Distill |
|---|---|---|
| **How supervisor acts** | Writes Python code that constructs `JobManifest` objects (structured Pydantic models with `chunk`, `task`, `advice` fields) | Writes Python code executed in a sandboxed REPL that directly calls `worker()` / `worker_batch()` |
| **Decomposition output** | Two explicit functions: `prepare_jobs()` and `transform_outputs()` | Free-form code — supervisor decides how to chunk, call workers, and aggregate |
| **Aggregation** | Supervisor writes a `transform_outputs()` function that filters/merges job results before seeing them | Supervisor writes inline aggregation code; sees raw worker responses (truncated to 2000 chars) |

**Takeaway:** Minions enforces a structured job manifest abstraction. Distill gives the supervisor full programmatic freedom — more flexible, but less constrained.

---

## 2. Worker interface

| | Minions | Distill |
|---|---|---|
| **Worker input** | Structured `JobManifest`: `chunk` + `task` + `advice` | Free-form string prompt (supervisor constructs it in code) |
| **Worker output** | Structured JSON: `explanation`, `citation`, `answer` (required fields) | Free-form text string |
| **Enforcement** | Worker prompt template enforces output schema | No output schema — supervisor parses responses however it wants |

**Takeaway:** Minions workers always produce structured, citation-backed outputs. Distill workers return raw text that the supervisor must interpret.

---

## 3. Chunking strategy

| | Minions | Distill |
|---|---|---|
| **Built-in chunkers** | 5 strategies: `chunk_by_section`, `chunk_by_page`, `chunk_by_paragraph`, `chunk_by_code`, `chunk_by_function_and_class` | None built-in |
| **Who chunks** | Supervisor selects a chunking function in `prepare_jobs()` | Supervisor writes arbitrary Python slicing/splitting code |
| **Retrieval** | Optional BM25 and embedding retrieval to pre-filter chunks before creating jobs | No retrieval — supervisor must search context manually via worker calls |

**Takeaway:** Minions provides chunking and retrieval as first-class primitives. Distill leaves all chunking to the supervisor's generated code.

---

## 4. Iteration and convergence

| | Minions | Distill |
|---|---|---|
| **Loop structure** | Round-based: decompose &rarr; execute jobs &rarr; aggregate &rarr; synthesize &rarr; decide | Step-based: supervisor writes code &rarr; REPL executes &rarr; output appended &rarr; loop |
| **Stopping signal** | Supervisor JSON decision: `"provide_final_answer"` vs `"request_additional_info"` | `FINAL(answer)` call inside REPL code |
| **Feedback between rounds** | Explicit `feedback` (what's missing) + `scratchpad` (running summary) fields | Implicit: REPL stdout/stderr + iteration nudge appended to message history |
| **Default max rounds** | 3 | 15 |
| **Final round behavior** | Special prompt forces conclusion, no option to request more info | Nudge message: "You are almost out of steps. Call FINAL(answer) now." |

**Takeaway:** Minions has a more rigid round structure with explicit feedback/scratchpad state. Distill allows more iterations with a looser loop — the supervisor can call workers multiple times within a single iteration.

---

## 5. Synthesis / decision protocol

| | Minions | Distill |
|---|---|---|
| **Synthesis step** | Two-phase: chain-of-thought prompt, then structured JSON decision | No separate synthesis — supervisor reasons in code comments or print statements |
| **Decision format** | JSON with `decision`, `explanation`, `answer`, `feedback`, `scratchpad` | Implicit via `FINAL(answer)` or continuing to write more code |
| **Cross-round memory** | Explicit `scratchpad` field persists supervisor's running summary | REPL namespace persists variables; message history accumulates |

**Takeaway:** Minions forces the supervisor to explicitly articulate its reasoning and what information is still needed. Distill's supervisor reasoning is embedded in code.

---

## 6. Parallelism

| | Minions | Distill |
|---|---|---|
| **Job execution** | All jobs in a round run in parallel (up to 2048 per round) | `worker_batch()` runs prompts in parallel via `asyncio.gather` |
| **Multiple samples** | `num_samples_per_task` parameter for redundant worker calls | No built-in redundancy — supervisor would need to duplicate prompts manually |

---

## 7. Features in Minions not present in Distill

- **Privacy shield**: Optional PII extraction and filtering between supervisor and worker
- **MCP tool integration**: Supervisor can invoke external tools; results prepended to worker context
- **Structured retrieval**: BM25 and embedding-based chunk retrieval as built-in functions
- **Confidence via redundancy**: Multiple samples per task for agreement-based confidence
- **Explicit advice generation**: Dedicated initial round where supervisor generates high-level extraction advice before decomposing

## 8. Features in Distill not present in Minions

- **Full REPL sandbox**: Supervisor can run arbitrary Python (loops, string ops, regex, etc.), not just job construction
- **Multi-code-block execution**: Supervisor can emit multiple ```` ```repl` ```` blocks per turn, all executed sequentially
- **Flexible worker prompting**: No schema imposed on worker input/output — supervisor can use workers for any text-in/text-out task
- **Variable persistence**: REPL namespace carries state across all iterations (not just structured scratchpad)
- **Output truncation**: Supervisor sees truncated output (2000 chars) while full output is logged — prevents context blowup

---

## Summary

Minions is a **structured job-dispatch system**: the supervisor constructs typed job manifests, workers produce structured outputs, and aggregation follows a defined pipeline. It trades flexibility for reliability and auditability.

Distill is a **code-generation orchestrator**: the supervisor writes arbitrary Python that calls workers as functions. It trades structure for flexibility — the supervisor can implement any strategy, but nothing enforces quality patterns like citations or structured filtering.

The core research question for this project is whether Distill's flexible approach can match or exceed Minions' structured approach, and whether the supervisor's code-generation behavior can be distilled into smaller models.
