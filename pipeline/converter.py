"""Merge LoRA adapter → convert to GGUF → register with Ollama."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable

from config import GGUF_DIR, LLAMA_CPP_DIR, MERGED_DIR
from . import ollama_client


def merge_lora(adapter_dir: Path, base_model_repo: str, out_dir: Path,
               log_cb: Callable[[str], None] | None = None) -> Path:
    """Load base + adapter, merge weights, save to `out_dir` in HF format."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    def log(m):
        if log_cb:
            log_cb(m)

    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"Loading base model {base_model_repo} for merge...")
    base = AutoModelForCausalLM.from_pretrained(
        base_model_repo, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    log(f"Attaching adapter from {adapter_dir}...")
    merged = PeftModel.from_pretrained(base, str(adapter_dir))
    log("Merging adapter weights into base...")
    merged = merged.merge_and_unload()
    log(f"Saving merged model to {out_dir}...")
    merged.save_pretrained(str(out_dir), safe_serialization=True)
    tok = AutoTokenizer.from_pretrained(str(adapter_dir))
    tok.save_pretrained(str(out_dir))
    log("Merge complete.")
    return out_dir


def ensure_llama_cpp(log_cb: Callable[[str], None] | None = None) -> Path:
    """Clone llama.cpp if missing — we only need its python convert script."""
    def log(m):
        if log_cb:
            log_cb(m)

    script = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"
    if script.exists():
        return script
    if not shutil.which("git"):
        raise RuntimeError(
            "git not found on PATH. Install Git for Windows, or manually place "
            f"llama.cpp at {LLAMA_CPP_DIR}."
        )
    log(f"Cloning llama.cpp into {LLAMA_CPP_DIR} (shallow)...")
    subprocess.run(
        ["git", "clone", "--depth", "1",
         "https://github.com/ggerganov/llama.cpp", str(LLAMA_CPP_DIR)],
        check=True,
    )
    if not script.exists():
        raise RuntimeError("convert_hf_to_gguf.py missing after clone")
    return script


def convert_to_gguf(merged_dir: Path, out_path: Path, quant: str,
                    log_cb: Callable[[str], None] | None = None) -> Path:
    """Convert merged HF model to GGUF.

    The converter's --outtype supports f16/f32/bf16/q8_0 natively. K-quants
    (q4_k_m / q5_k_m / etc.) require llama.cpp's `llama-quantize` binary,
    which is not pre-built. If we don't find it, we silently downgrade to
    q8_0 (still ~half the size of f16) and rename the output truthfully so
    no one is misled by the filename.

    Returns the ACTUAL path written (may differ from the requested path
    when we fall back).
    """
    def log(m):
        if log_cb:
            log_cb(m)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    script = ensure_llama_cpp(log_cb)

    direct = {"f16", "f32", "bf16", "q8_0"}
    quant = quant.lower()
    if quant in direct:
        log(f"Converting {merged_dir} → {out_path} (outtype={quant})...")
        subprocess.run(
            [sys.executable, str(script), str(merged_dir),
             "--outfile", str(out_path), "--outtype", quant],
            check=True,
        )
        return out_path

    # K-quant requested — need llama-quantize binary.
    qbin = _find_quantize_binary()
    if not qbin:
        fallback_quant = "q8_0"
        fallback_path = out_path.with_name(
            out_path.name.replace(f".{quant}.", f".{fallback_quant}.")
        )
        if fallback_path == out_path:  # name didn't contain the quant token
            fallback_path = out_path.with_suffix(f".{fallback_quant}.gguf")
        log(
            f"llama-quantize binary not found — `{quant}` requires it. "
            f"Falling back to {fallback_quant} (~half the size of f16, no build needed). "
            "To produce a true k-quant, build llama.cpp: "
            "`cd llama.cpp && cmake -B build && cmake --build build --config Release`."
        )
        log(f"Converting {merged_dir} → {fallback_path} (outtype={fallback_quant})...")
        subprocess.run(
            [sys.executable, str(script), str(merged_dir),
             "--outfile", str(fallback_path), "--outtype", fallback_quant],
            check=True,
        )
        return fallback_path

    # Binary present: produce f16 first, then quantize.
    f16_path = out_path.with_suffix(".f16.gguf")
    log(f"Converting {merged_dir} → {f16_path} (f16)...")
    subprocess.run(
        [sys.executable, str(script), str(merged_dir),
         "--outfile", str(f16_path), "--outtype", "f16"],
        check=True,
    )
    log(f"Quantizing {f16_path} → {out_path} ({quant})...")
    subprocess.run([qbin, str(f16_path), str(out_path), quant], check=True)
    try:
        f16_path.unlink()
    except OSError:
        pass
    return out_path


def _find_quantize_binary() -> str | None:
    candidates = [
        LLAMA_CPP_DIR / "build" / "bin" / "Release" / "llama-quantize.exe",
        LLAMA_CPP_DIR / "build" / "bin" / "llama-quantize.exe",
        LLAMA_CPP_DIR / "build" / "bin" / "llama-quantize",
        LLAMA_CPP_DIR / "llama-quantize.exe",
        LLAMA_CPP_DIR / "llama-quantize",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    in_path = shutil.which("llama-quantize")
    return in_path


def write_modelfile(gguf_path: Path, system_prompt: str,
                    temperature: float, top_p: float) -> Path:
    mf = gguf_path.with_suffix(".Modelfile")
    content = (
        f'FROM "{gguf_path.as_posix()}"\n'
        f'PARAMETER temperature {temperature}\n'
        f'PARAMETER top_p {top_p}\n'
        f'SYSTEM """{system_prompt}"""\n'
    )
    mf.write_text(content, encoding="utf-8")
    return mf


def register_with_ollama(name: str, modelfile: Path,
                         log_cb: Callable[[str], None] | None = None) -> str:
    def log(m):
        if log_cb:
            log_cb(m)
    log(f"Registering model `{name}` with Ollama...")
    for line in ollama_client.create_model(name, str(modelfile)):
        log(line)
    return name
