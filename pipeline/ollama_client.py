"""Thin wrappers around the Ollama HTTP client + CLI."""
from __future__ import annotations

import json
import shutil
import subprocess
from typing import Iterator

import ollama


def list_models() -> list[dict]:
    try:
        resp = ollama.list()
    except Exception as exc:
        raise RuntimeError(
            "Could not reach Ollama. Is `ollama serve` running?"
        ) from exc
    # ollama-python returns either {"models": [...]} or a ListResponse object
    models = getattr(resp, "models", None) or resp.get("models", [])
    out = []
    for m in models:
        name = getattr(m, "model", None) or m.get("model") or m.get("name")
        size = getattr(m, "size", None) or m.get("size", 0)
        if name:
            out.append({"name": name, "size": size})
    return out


def generate(model: str, prompt: str, system: str | None = None,
             temperature: float = 0.4, num_predict: int = 512) -> str:
    """Single-shot text generation."""
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})
    resp = ollama.chat(
        model=model,
        messages=msgs,
        options={"temperature": temperature, "num_predict": num_predict},
    )
    return resp["message"]["content"]


def chat_stream(model: str, messages: list[dict],
                temperature: float = 0.7, top_p: float = 0.9,
                num_predict: int = 512) -> Iterator[str]:
    """Streaming chat — yields token chunks."""
    stream = ollama.chat(
        model=model,
        messages=messages,
        stream=True,
        options={
            "temperature": temperature,
            "top_p": top_p,
            "num_predict": num_predict,
        },
    )
    for part in stream:
        chunk = part.get("message", {}).get("content", "")
        if chunk:
            yield chunk


def create_model(name: str, modelfile_path: str) -> Iterator[str]:
    """Run `ollama create <name> -f <modelfile>` and stream progress lines."""
    if not shutil.which("ollama"):
        raise RuntimeError("ollama CLI not found on PATH")
    # Ollama emits UTF-8 spinner glyphs; Windows defaults to cp1252 which
    # crashes on them. Force UTF-8 decoding with replacement.
    proc = subprocess.Popen(
        ["ollama", "create", name, "-f", modelfile_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        yield line.rstrip()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ollama create failed with code {proc.returncode}")


def delete_model(name: str) -> None:
    """Delete a model from Ollama. Raises on failure so the UI can show it."""
    ollama.delete(name)
