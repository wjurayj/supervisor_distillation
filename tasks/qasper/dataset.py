"""Load and process the QASPER dataset from the original S3-hosted JSON files."""

import json
import os
import tarfile
import urllib.request
from pathlib import Path

_CACHE_DIR = Path.home() / ".cache" / "qasper"

_URL_TRAIN_DEV = "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz"
_URL_TEST = "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-test-and-evaluator-v0.3.tgz"

_SPLIT_FILES = {
    "train": ("qasper-train-v0.3.json", _URL_TRAIN_DEV),
    "validation": ("qasper-dev-v0.3.json", _URL_TRAIN_DEV),
    "test": ("qasper-test-v0.3.json", _URL_TEST),
}


def _ensure_downloaded(url: str) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    marker = _CACHE_DIR / (Path(url).stem + ".done")
    if marker.exists():
        return
    tgz_path = _CACHE_DIR / Path(url).name
    if not tgz_path.exists():
        print(f"Downloading {url} ...")
        urllib.request.urlretrieve(url, tgz_path)
    with tarfile.open(tgz_path, "r:gz") as tar:
        tar.extractall(_CACHE_DIR)
    marker.touch()


def _load_raw(split: str) -> dict:
    filename, url = _SPLIT_FILES[split]
    _ensure_downloaded(url)
    with open(_CACHE_DIR / filename) as f:
        return json.load(f)


def _get_context(paper: dict) -> str:
    """Build a single string from a paper's full_text and figures/tables."""
    sections = []
    for section in paper.get("full_text", []):
        name = section.get("section_name", "")
        paras = section.get("paragraphs", [])
        text = " ".join(paras)
        sections.append(f"### {name}\n\n{text}")

    for fig in paper.get("figures_and_tables", []):
        caption = fig.get("caption", "")
        if caption:
            sections.append(caption)

    return " ".join(sections)


def load_qasper_dataset(split: str = "train") -> list[dict]:
    """Load QASPER, returning a flat list of {doc_id, title, context, qa_id, question, answer, ...}."""
    raw = _load_raw(split)

    out = []
    for doc_id, paper in raw.items():
        context = _get_context(paper)
        title = paper.get("title", "")

        for qa in paper.get("qas", []):
            question = qa.get("question", "")
            qa_id = qa.get("question_id", "")

            answer = evidence = highlighted_evidence = None
            for a in qa.get("answers", []):
                ann = a.get("answer", {})
                if ann.get("unanswerable"):
                    continue
                spans = ann.get("extractive_spans", [])
                if spans:
                    answer = ", ".join(spans)
                    evidence = ", ".join(ann.get("evidence", []))
                    highlighted_evidence = ", ".join(ann.get("highlighted_evidence", []))
                    break

            if answer:
                out.append({
                    "doc_id": doc_id,
                    "title": title,
                    "context": context,
                    "qa_id": qa_id,
                    "question": question,
                    "answer": answer,
                    "evidence": evidence,
                    "highlighted_evidence": highlighted_evidence,
                })
    return out
