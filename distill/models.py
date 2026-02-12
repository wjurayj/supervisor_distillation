"""Model handlers exposing OpenAI-compatible chat completions interface."""

from __future__ import annotations

import asyncio
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from openai import AsyncOpenAI


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    def __iadd__(self, other: Usage) -> Usage:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        return self


@dataclass
class LMResponse:
    text: str
    usage: Usage
    model: str
    elapsed: float


class ModelHandler(ABC):
    """Abstract interface exposing OpenAI-compatible chat completions."""

    @abstractmethod
    def chat(self, messages: list[dict], **kwargs) -> LMResponse: ...

    @abstractmethod
    def chat_batch(self, message_batches: list[list[dict]], **kwargs) -> list[LMResponse]: ...


class OpenAIHandler(ModelHandler):
    """Uses openai SDK with configurable base_url (Together AI, OpenAI, any compatible endpoint)."""

    def __init__(
        self,
        model: str,
        base_url: str = "https://api.together.xyz/v1",
        api_key: str | None = None,
        **kwargs,
    ):
        self.model = model
        self.default_kwargs = kwargs
        self._client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("TOGETHER_API_KEY"),
        )
        self.total_usage = Usage()

    async def _achat(self, messages: list[dict], **kwargs) -> LMResponse:
        merged = {**self.default_kwargs, **kwargs}
        t0 = time.perf_counter()
        resp = await self._client.chat.completions.create(
            model=self.model, messages=messages, **merged
        )
        elapsed = time.perf_counter() - t0
        choice = resp.choices[0]
        usage = Usage(
            input_tokens=resp.usage.prompt_tokens if resp.usage else 0,
            output_tokens=resp.usage.completion_tokens if resp.usage else 0,
        )
        self.total_usage += usage
        return LMResponse(text=choice.message.content or "", usage=usage, model=self.model, elapsed=elapsed)

    def chat(self, messages: list[dict], **kwargs) -> LMResponse:
        return asyncio.run(self._achat(messages, **kwargs))

    def chat_batch(self, message_batches: list[list[dict]], **kwargs) -> list[LMResponse]:
        async def _gather():
            return await asyncio.gather(
                *(self._achat(msgs, **kwargs) for msgs in message_batches)
            )
        return asyncio.run(_gather())


class VLLMHandler(ModelHandler):
    """Offline inference using the vLLM engine.

    Args:
        model: Logical model name used in LMResponse and logs.
        model_path: Path or HF repo that vLLM loads weights from.
            Defaults to *model* when not provided.
        **kwargs: Forwarded to ``SamplingParams`` as defaults
            (e.g. temperature, max_tokens).  Any additional keyword
            arguments whose names start with ``engine_`` are stripped
            of that prefix and forwarded to the ``LLM`` constructor
            instead (e.g. ``engine_tensor_parallel_size=2``).
    """

    def __init__(
        self,
        model: str,
        model_path: str | None = None,
        **kwargs: Any,
    ):
        from vllm import LLM

        self.model = model
        self.model_path = model_path or model

        # Split kwargs: engine_* go to LLM(), rest go to SamplingParams.
        engine_kwargs: dict[str, Any] = {}
        sampling_kwargs: dict[str, Any] = {}
        for k, v in kwargs.items():
            if k.startswith("engine_"):
                engine_kwargs[k.removeprefix("engine_")] = v
            else:
                sampling_kwargs[k] = v

        self.default_kwargs = sampling_kwargs
        self._llm = LLM(model=self.model_path, **engine_kwargs)
        self.total_usage = Usage()

    def _build_sampling_params(self, **overrides: Any):
        from vllm import SamplingParams

        merged = {**self.default_kwargs, **overrides}
        return SamplingParams(**merged)

    def chat(self, messages: list[dict], **kwargs) -> LMResponse:
        params = self._build_sampling_params(**kwargs)
        t0 = time.perf_counter()
        outputs = self._llm.chat(messages=messages, sampling_params=params)
        elapsed = time.perf_counter() - t0

        req = outputs[0]
        comp = req.outputs[0]
        usage = Usage(
            input_tokens=len(req.prompt_token_ids),
            output_tokens=len(comp.token_ids),
        )
        self.total_usage += usage
        return LMResponse(text=comp.text, usage=usage, model=self.model, elapsed=elapsed)

    def chat_batch(self, message_batches: list[list[dict]], **kwargs) -> list[LMResponse]:
        params = self._build_sampling_params(**kwargs)
        t0 = time.perf_counter()
        outputs = self._llm.chat(messages=message_batches, sampling_params=params)
        elapsed = time.perf_counter() - t0

        results: list[LMResponse] = []
        for req in outputs:
            comp = req.outputs[0]
            usage = Usage(
                input_tokens=len(req.prompt_token_ids),
                output_tokens=len(comp.token_ids),
            )
            self.total_usage += usage
            results.append(LMResponse(text=comp.text, usage=usage, model=self.model, elapsed=elapsed))
        return results
