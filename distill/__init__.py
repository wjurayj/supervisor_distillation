"""Supervisor-worker deep research via programmatic distillation."""

from distill.models import ModelHandler, OpenAIHandler, VLLMHandler
from distill.orchestrator import run

__all__ = ["run", "ModelHandler", "OpenAIHandler", "VLLMHandler"]
