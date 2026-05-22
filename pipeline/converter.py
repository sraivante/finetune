"""Merge LoRA adapter → convert to GGUF → register with Ollama."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, Literal

from config import GGUF_DIR, LLAMA_CPP_DIR, MERGED_DIR
from . import ollama_client, templates as chat_templates

FolderKind = Literal["adapter", "merged", "unknown"]


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
                    temperature: float, top_p: float,
                    *,
                    template: str | None = None,
                    stop: list[str] | None = None) -> Path:
    """Write an Ollama Modelfile next to the GGUF.

    `template` + `stop` are optional but strongly recommended: Ollama does NOT
    reliably apply a model's embedded jinja chat_template, so without an
    explicit TEMPLATE the prompt can reach the model unwrapped.
    """
    mf = gguf_path.with_suffix(".Modelfile")
    lines: list[str] = [f'FROM "{gguf_path.as_posix()}"']
    if template:
        lines.append(f'TEMPLATE """{template}"""')
    for s in (stop or []):
        lines.append(f'PARAMETER stop "{s}"')
    lines.append(f'PARAMETER temperature {temperature}')
    lines.append(f'PARAMETER top_p {top_p}')
    if system_prompt:
        lines.append(f'SYSTEM """{system_prompt}"""')
    mf.write_text("\n".join(lines) + "\n", encoding="utf-8")
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


# ---------------------------------------------------------------------------
# Bring-your-own-adapter: register an externally fine-tuned folder with Ollama
# ---------------------------------------------------------------------------
def _has_model_weights(folder: Path) -> bool:
    """A merged HF model has either model.safetensors or *.bin shards."""
    if (folder / "model.safetensors").exists():
        return True
    if (folder / "model.safetensors.index.json").exists():
        return True
    if any(folder.glob("pytorch_model*.bin")):
        return True
    if any(folder.glob("model-*-of-*.safetensors")):
        return True
    return False


def inspect_external_folder(path: Path) -> dict:
    """Identify what's in a user-supplied fine-tune folder.

    Returns: {
        "kind": "adapter" | "merged" | "unknown",
        "base_model": str | None,
        "has_tokenizer": bool,
        "notes": list[str],
    }
    """
    notes: list[str] = []
    info = {"kind": "unknown", "base_model": None,
            "has_tokenizer": False, "notes": notes}

    if not path.exists() or not path.is_dir():
        notes.append(f"Path does not exist or is not a directory: {path}")
        return info

    adapter_cfg = path / "adapter_config.json"
    model_cfg = path / "config.json"

    if adapter_cfg.exists():
        info["kind"] = "adapter"
        try:
            data = json.loads(adapter_cfg.read_text(encoding="utf-8"))
            info["base_model"] = data.get("base_model_name_or_path")
        except Exception as exc:
            notes.append(f"Could not parse adapter_config.json: {exc}")
        if not (path / "adapter_model.safetensors").exists() and \
                not (path / "adapter_model.bin").exists():
            notes.append("No adapter_model.safetensors / .bin found — "
                         "merge will fail.")
    elif model_cfg.exists() and _has_model_weights(path):
        info["kind"] = "merged"
        try:
            data = json.loads(model_cfg.read_text(encoding="utf-8"))
            info["base_model"] = data.get("_name_or_path") or \
                                 data.get("name_or_path")
        except Exception as exc:
            notes.append(f"Could not parse config.json: {exc}")
    else:
        notes.append(
            "Folder contains neither adapter_config.json nor a full HF model "
            "(config.json + weights). If this is a Trainer output, try a "
            "subdirectory like checkpoint-XXXX/."
        )

    if (path / "tokenizer.json").exists() or \
            (path / "tokenizer_config.json").exists() or \
            (path / "tokenizer.model").exists():
        info["has_tokenizer"] = True
    elif info["kind"] != "unknown":
        notes.append("No tokenizer files in folder — GGUF conversion may fail "
                     "unless the base model's tokenizer is fetched from HF.")

    return info


def _slugify_for_ollama(name: str) -> str:
    """Ollama tags accept letters/digits/_/-/. — turn anything else into '-'."""
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9._-]+", "-", name)
    name = re.sub(r"-+", "-", name).strip("-.")
    return name or "imported"


def suggest_ollama_name(adapter_path: Path, base_model: str | None) -> str:
    """Build a reasonable default Ollama tag from the folder + base model."""
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    if base_model:
        base = base_model.split("/")[-1]
    else:
        base = adapter_path.name
    return f"{_slugify_for_ollama(base)}-imported-{ts}"


def import_external_to_ollama(
    folder: Path,
    *,
    base_model_repo: str | None,
    ollama_name: str,
    quant: str = "q4_k_m",
    system_prompt: str = "",
    temperature: float = 0.3,
    top_p: float = 0.9,
    mode: FolderKind = "unknown",
    chat_format_key: str | None = None,
    log_cb: Callable[[str], None] | None = None,
) -> dict:
    """Take a user-supplied folder and register it with Ollama.

    `mode` is either "adapter" (always merge), "merged" (skip merge), or
    "unknown" (auto-detect via inspect_external_folder).

    Returns the final paths and the Ollama tag.
    """
    def log(m: str) -> None:
        if log_cb:
            log_cb(m)

    folder = Path(folder)
    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"Folder not found: {folder}")

    if mode == "unknown":
        info = inspect_external_folder(folder)
        detected = info["kind"]
        if detected == "unknown":
            raise ValueError(
                "Could not detect folder kind. " +
                (" ".join(info["notes"]) or "")
            )
        mode = detected  # type: ignore[assignment]
        log(f"Auto-detected folder kind: {mode}")
        if info["base_model"] and not base_model_repo:
            base_model_repo = info["base_model"]
            log(f"Auto-detected base model: {base_model_repo}")

    run_id = _slugify_for_ollama(folder.name) + "-" + \
             datetime.now().strftime("%Y%m%d-%H%M%S")

    if mode == "adapter":
        if not base_model_repo:
            raise ValueError(
                "Base HF model repo is required to merge a LoRA adapter "
                "(e.g. 'TinyLlama/TinyLlama-1.1B-Chat-v1.0')."
            )
        merged_dir = MERGED_DIR / run_id
        merge_lora(folder, base_model_repo, merged_dir, log_cb=log_cb)
    elif mode == "merged":
        merged_dir = folder
        log(f"Skipping merge — using merged folder directly: {merged_dir}")
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    requested_gguf = GGUF_DIR / f"{run_id}.{quant}.gguf"
    gguf_path = convert_to_gguf(merged_dir, requested_gguf,
                                quant=quant, log_cb=log_cb)

    # Pick the chat template: explicit user override first, else auto-detect
    # from the merged folder's jinja chat_template / config.json.
    if chat_format_key and chat_format_key in chat_templates.CHAT_FORMATS:
        fmt = chat_templates.CHAT_FORMATS[chat_format_key]
        chat_template, chat_stop = fmt["template"], fmt["stop"]
        log(f"Using user-selected chat format: {chat_format_key}")
    else:
        det = chat_templates.detect_chat_format(merged_dir)
        chat_template, chat_stop = det["template"], det["stop"]
        log(f"Auto-detected chat format: {det['key']} (via {det['source']})")

    modelfile = write_modelfile(
        gguf_path,
        system_prompt=system_prompt,
        temperature=temperature,
        top_p=top_p,
        template=chat_template,
        stop=chat_stop,
    )
    ollama_tag = _slugify_for_ollama(ollama_name)
    register_with_ollama(ollama_tag, modelfile, log_cb=log_cb)

    return {
        "ollama_name": ollama_tag,
        "merged_dir": str(merged_dir),
        "gguf_path": str(gguf_path),
        "modelfile": str(modelfile),
    }
