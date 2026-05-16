"""Generate instruction-tuning Q&A pairs from document chunks using an Ollama model."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable, Iterator

from . import ollama_client

QA_SYSTEM_PROMPT = (
    "You generate training data for fine-tuning a small language model. "
    "Given a passage, write self-contained question/answer pairs that test factual "
    "knowledge of the passage. Questions must be answerable from the passage alone. "
    "Answers must be concise but complete (1-4 sentences). "
    "Return ONLY a JSON array of objects with keys \"question\" and \"answer\". "
    "No prose, no markdown fences."
)


def _qa_user_prompt(chunk: str, n: int) -> str:
    return (
        f"Passage:\n\"\"\"\n{chunk}\n\"\"\"\n\n"
        f"Produce {n} Q&A pairs grounded in this passage. "
        f"Return a JSON array like "
        f'[{{"question":"...","answer":"..."}}, ...].'
    )


_JSON_ARRAY_RE = re.compile(r"\[\s*\{.*?\}\s*\]", re.DOTALL)


def _extract_pairs(raw: str) -> list[dict]:
    raw = raw.strip()
    # Strip markdown fences if the model added them.
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
        raw = re.sub(r"\n?```\s*$", "", raw)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        m = _JSON_ARRAY_RE.search(raw)
        if not m:
            return []
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return []
    if not isinstance(data, list):
        return []
    out = []
    for item in data:
        if not isinstance(item, dict):
            continue
        q = str(item.get("question", "")).strip()
        a = str(item.get("answer", "")).strip()
        if q and a:
            out.append({"question": q, "answer": a})
    return out


def build_dataset(
    chunks: list[dict],
    *,
    ollama_model: str,
    pairs_per_chunk: int,
    temperature: float,
    out_path: Path,
    progress_cb: Callable[[int, int, str], None] | None = None,
) -> tuple[Path, int]:
    """Generate Q&A pairs for each chunk and write a JSONL file in chat format.

    JSONL row: {"messages": [{"role":"user","content":...},{"role":"assistant","content":...}]}
    Returns (path, num_pairs).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = len(chunks)
    written = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for i, row in enumerate(chunks, start=1):
            chunk = row["text"]
            if not chunk or chunk.startswith("[ERROR"):
                if progress_cb:
                    progress_cb(i, total, f"skip {row.get('source')}#{row.get('chunk_index')}")
                continue
            try:
                raw = ollama_client.generate(
                    model=ollama_model,
                    prompt=_qa_user_prompt(chunk, pairs_per_chunk),
                    system=QA_SYSTEM_PROMPT,
                    temperature=temperature,
                    num_predict=1024,
                )
            except Exception as exc:
                if progress_cb:
                    progress_cb(i, total, f"error: {exc}")
                continue
            pairs = _extract_pairs(raw)
            for pair in pairs:
                fh.write(json.dumps({
                    "messages": [
                        {"role": "user", "content": pair["question"]},
                        {"role": "assistant", "content": pair["answer"]},
                    ],
                    "source": row.get("source"),
                    "chunk_index": row.get("chunk_index"),
                }, ensure_ascii=False) + "\n")
                written += 1
            if progress_cb:
                progress_cb(i, total, f"{row.get('source')}#{row.get('chunk_index')} → {len(pairs)} pairs")
    return out_path, written


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as fh:
        return sum(1 for _ in fh)


def preview(path: Path, n: int = 5) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if len(rows) >= n:
                break
    return rows
