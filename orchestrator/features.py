"""Feature flags for ablation experiments."""

from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass(frozen=True)
class FeatureFlags:
    """Boolean flags for Minions-style features. All default to False."""

    structured_jobs: bool = False
    structured_output: bool = False
    builtin_chunking: bool = False
    explicit_convergence: bool = False
    synthesis_cot: bool = False

    def label(self) -> str:
        """Short label for directory naming, e.g. 'sj-so-bc' or 'none'."""
        abbrevs = {
            "structured_jobs": "sj",
            "structured_output": "so",
            "builtin_chunking": "bc",
            "explicit_convergence": "ec",
            "synthesis_cot": "sc",
        }
        parts = [abbrevs[f.name] for f in fields(self) if getattr(self, f.name)]
        return "-".join(parts) if parts else "none"

    @classmethod
    def all_on(cls) -> FeatureFlags:
        return cls(**{f.name: True for f in fields(cls)})

    @classmethod
    def all_but(cls, name: str) -> FeatureFlags:
        return cls(**{f.name: f.name != name for f in fields(cls)})

    @classmethod
    def only(cls, name: str) -> FeatureFlags:
        return cls(**{f.name: f.name == name for f in fields(cls)})
