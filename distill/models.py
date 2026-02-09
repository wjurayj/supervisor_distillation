"""Model handlers exposing OpenAI-compatible chat completions interface."""

from __future__ import annotations

import asyncio
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

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
    """Abstract interface exposing OpenAI-compatible chat completions.

    Future: implement VLLMHandler for offline vLLM inference.
    """

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
