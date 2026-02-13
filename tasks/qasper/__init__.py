"""QASPER task: long-context QA over scientific papers."""

from __future__ import annotations

import json
import os
import random
import re
from dataclasses import dataclass, field

from openai import OpenAI

from tasks.qasper.dataset import load_qasper_dataset


@dataclass
class QasperExample:
    id: str
    query: str
    context: str
    target: str
    title: str
    metadata: dict = field(default_factory=dict)


JUDGE_PROMPT = """\
You are an expert evaluator. Determine if a predicted answer matches the correct answer.

Question: {query}

Predicted Answer: {pred}

Correct Answer: {target}

Is the predicted answer correct? Answers may be phrased differently but convey the same information.

Respond with ONLY a JSON object:
{{"explanation": "brief explanation", "is_correct": true or false}}"""


class QasperTask:
    def __init__(
        self,
        split: str = "train",
        seed: int = 42,
        limit: int | None = None,
        num_distractors: int = 0,
        judge_model: str = "Qwen/Qwen3-235B-A22B-Instruct-2507-tput",
        base_url: str = "https://api.together.xyz/v1",
        api_key: str | None = None,
    ):
        self.split = split
        self.seed = seed
        self.limit = limit
        self.num_distractors = num_distractors
        self.judge_model = judge_model
        self._base_url = base_url
        self._api_key = api_key
        self._client = None

    def _get_client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(
                base_url=self._base_url,
                api_key=self._api_key or os.environ.get("TOGETHER_API_KEY") or os.environ.get("OPENAI_API_KEY"),
            )
        return self._client

    @staticmethod
    def _format_paper(title: str, content: str) -> str:
        return f"=== Paper: {title} ===\n\n{content}"

    def build_dataset(self) -> list[QasperExample]:
        raw = load_qasper_dataset(split=self.split)

        # Build paper pool for distractor sampling
        paper_pool: dict[str, dict] = {}
        if self.num_distractors > 0:
            for entry in raw:
                if entry["doc_id"] not in paper_pool:
                    paper_pool[entry["doc_id"]] = {
                        "title": entry["title"],
                        "context": entry["context"],
                    }

        rng = random.Random(self.seed)

        examples = []
        for entry in raw:
            if self.num_distractors > 0:
                gold = self._format_paper(entry["title"], entry["context"])
                distractors = [
                    self._format_paper(p["title"], p["context"])
                    for did, p in paper_pool.items()
                    if did != entry["doc_id"]
                ]
                k = min(self.num_distractors, len(distractors))
                selected = rng.sample(distractors, k)
                docs = [gold] + selected
                rng.shuffle(docs)
                context = "\n\n".join(docs)
            else:
                context = entry["context"]

            examples.append(QasperExample(
                id=entry["qa_id"],
                query=entry["question"],
                context=context,
                target=entry["answer"],
                title=entry["title"],
                metadata={
                    "doc_id": entry["doc_id"],
                    "evidence": entry.get("evidence"),
                    "highlighted_evidence": entry.get("highlighted_evidence"),
                },
            ))

        rng.shuffle(examples)

        if self.limit is not None and self.limit > 0:
            examples = examples[:self.limit]

        return examples

    def score(self, pred: str, example: QasperExample) -> int:
        """Score prediction against ground truth using LLM judge. Returns 0 or 1."""
        prompt = JUDGE_PROMPT.format(
            query=example.query,
            pred=pred,
            target=example.target,
        )
        resp = self._get_client().chat.completions.create(
            model=self.judge_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=256,
        )
        text = resp.choices[0].message.content
        try:
            result = json.loads(text)
            return 1 if result.get("is_correct", False) else 0
        except json.JSONDecodeError:
            match = re.search(r'"is_correct"\s*:\s*(true|false)', text, re.IGNORECASE)
            if match:
                return 1 if match.group(1).lower() == "true" else 0
            return 0
