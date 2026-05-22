"""In-process accuracy check for a fine-tuned adapter or merged HF model.

Run BEFORE GGUF conversion: loads the adapter (or merged folder) into
transformers, runs deterministic generation on N rows from a validation
JSONL, and reports exact-match accuracy. The Import-adapter tab uses this
to decide whether the fine-tune is worth converting / shipping.

Val JSONL row shape (same format as the trainer ingests):
    {"messages": [
        {"role": "system",    "content": "..."},
        {"role": "user",      "content": "..."},
        {"role": "assistant", "content": "..."}
    ]}
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Callable


# Filename patterns that strongly suggest a held-out split.
_VAL_RX = re.compile(r"(?:^|[_\-.])(val|eval|test|dev|hold[_\-]?out)(?:$|[_\-.])",
                     re.IGNORECASE)
# Family tokens we look for in filenames vs the base-model repo string.
_FAMILY_TOKENS = ("qwen", "tinyllama", "llama", "mistral", "gemma", "phi", "yi")


def _family_hints(base_model_repo: str | None) -> list[str]:
    """Extract lowercase tokens from the base-model repo string that we can
    cross-check against candidate filenames (e.g. 'qwen', '0.5b').
    """
    if not base_model_repo:
        return []
    name = base_model_repo.lower()
    hints = [k for k in _FAMILY_TOKENS if k in name]
    m = re.search(r"(\d+\.?\d*)b", name)
    if m:
        hints.append(m.group(0))         # e.g. '0.5b', '1.1b', '7b'
    return hints


def _filename_matches_hints(path: Path, hints: list[str]) -> bool:
    if not hints:
        return False
    low = path.name.lower()
    return any(h in low for h in hints)


def find_val_candidates(
    folder: Path,
    base_model_repo: str | None,
    project_root: Path | None = None,
    extra_dirs: list[Path] | None = None,
) -> list[dict]:
    """Return a ranked list of val-JSONL candidates for `folder`.

    Each candidate dict has:
        path           — absolute path string
        priority       — lower is better (used to sort)
        source         — short label: 'in-folder', 'sibling', 'project-eval',
                         'project-datasets', 'project-data'
        looks_like_val — filename matches val/eval/test/dev/holdout
        model_match    — filename contains a family token from the base repo
        reason         — one-line human-readable rationale

    Sorted best-first. If empty, the caller should warn the user that no
    likely val file was found.
    """
    folder = Path(folder)
    parent = folder.parent if folder.exists() else None
    hints = _family_hints(base_model_repo)
    seen: set[Path] = set()
    out: list[dict] = []

    def add(path: Path, *, priority: int, source: str, reason: str) -> None:
        rp = path.resolve()
        if rp in seen or not path.is_file():
            return
        seen.add(rp)
        out.append({
            "path": str(rp),
            "priority": priority,
            "source": source,
            "looks_like_val": bool(_VAL_RX.search(path.name)),
            "model_match": _filename_matches_hints(path, hints),
            "reason": reason,
        })

    # Tier 1: held-out splits *inside* the adapter folder (zero-config win).
    if folder.exists() and folder.is_dir():
        for p in sorted(folder.glob("*.jsonl")):
            if _VAL_RX.search(p.name):
                add(p, priority=1, source="in-folder",
                    reason="held-out split saved alongside the adapter")

    # Tier 2: held-out splits in the parent directory (Colab pattern:
    # train.jsonl + val.jsonl + adapter/ all unpacked together).
    if parent and parent.exists():
        for p in sorted(parent.glob("*.jsonl")):
            if _VAL_RX.search(p.name):
                add(p, priority=2, source="sibling",
                    reason="held-out split next to the adapter folder")

    # Tier 3: any JSONL inside the adapter folder (likely the training set,
    # useful as a memorisation smoke test even if it's not held-out).
    if folder.exists() and folder.is_dir():
        for p in sorted(folder.glob("*.jsonl")):
            add(p, priority=3, source="in-folder",
                reason="JSONL inside the adapter folder (likely the train set)")

    # Tier 4: any JSONL in the adapter's parent dir.
    if parent and parent.exists():
        for p in sorted(parent.glob("*.jsonl")):
            add(p, priority=4, source="sibling",
                reason="JSONL next to the adapter folder")

    # Tier 5/6: project data dirs. We split eval/ from datasets/data because
    # eval/ files are usually held-out by convention.
    search_dirs: list[tuple[Path, str, int]] = []
    if project_root:
        search_dirs += [
            (project_root / "data" / "eval", "project-eval", 5),
            (project_root / "data" / "datasets", "project-datasets", 6),
            (project_root / "data", "project-data", 6),
        ]
    for d in (extra_dirs or []):
        search_dirs.append((Path(d), "extra", 6))

    for d, src, base_pri in search_dirs:
        if not d.exists():
            continue
        for p in sorted(d.glob("*.jsonl")):
            # Bonus rank: model-name match jumps this up by 1 tier.
            pri = base_pri - 1 if _filename_matches_hints(p, hints) else base_pri
            reason_bits = [f"from {d.relative_to(project_root) if project_root and d.is_relative_to(project_root) else d}"]
            if _VAL_RX.search(p.name):
                reason_bits.append("name looks like a val/test split")
            if _filename_matches_hints(p, hints):
                reason_bits.append("filename matches the base-model family")
            add(p, priority=pri, source=src, reason="; ".join(reason_bits))

    out.sort(key=lambda c: (c["priority"], c["path"]))
    return out


def evaluate_adapter_accuracy(
    folder: Path,
    *,
    base_model_repo: str | None,
    val_jsonl: Path,
    n_samples: int = 30,
    mode: str = "auto",
    max_new_tokens: int = 16,
    log_cb: Callable[[str], None] | None = None,
) -> dict:
    """Return {'correct', 'total', 'accuracy', 'examples'} for the folder.

    `mode`:
        'auto'    — detect adapter vs merged from folder contents
        'adapter' — folder is a LoRA adapter; needs `base_model_repo`
        'merged'  — folder is a full HF model
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    def log(m: str) -> None:
        if log_cb:
            log_cb(m)

    folder = Path(folder)
    val_jsonl = Path(val_jsonl)
    if not folder.exists():
        raise FileNotFoundError(f"Model folder not found: {folder}")
    if not val_jsonl.exists():
        raise FileNotFoundError(f"Validation file not found: {val_jsonl}")

    if mode == "auto":
        mode = "adapter" if (folder / "adapter_config.json").exists() else "merged"

    if mode == "adapter":
        if not base_model_repo:
            raise ValueError(
                "Base HF model repo is required for an adapter accuracy check."
            )
        from peft import PeftModel
        log(f"Loading base model {base_model_repo}...")
        base = AutoModelForCausalLM.from_pretrained(
            base_model_repo, torch_dtype=torch.float16, low_cpu_mem_usage=True
        )
        log(f"Attaching adapter from {folder}...")
        model = PeftModel.from_pretrained(base, str(folder))
        tokenizer = AutoTokenizer.from_pretrained(str(folder))
    elif mode == "merged":
        log(f"Loading merged model from {folder}...")
        model = AutoModelForCausalLM.from_pretrained(
            str(folder), torch_dtype=torch.float16, low_cpu_mem_usage=True
        )
        tokenizer = AutoTokenizer.from_pretrained(str(folder))
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    model.eval()
    model.config.use_cache = True
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    rows: list[dict] = []
    with val_jsonl.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                log(f"skipping {val_jsonl.name}:{line_no}: {exc}")
    if not rows:
        raise ValueError(f"No usable rows in {val_jsonl}")

    random.seed(0)
    random.shuffle(rows)
    rows = rows[: n_samples]
    log(f"Evaluating on {len(rows)} examples...")

    examples: list[dict] = []
    correct = 0
    for i, r in enumerate(rows, start=1):
        try:
            msgs = r["messages"]
            sys_m = next(
                (m["content"] for m in msgs if m["role"] == "system"), ""
            )
            usr_m = next(m["content"] for m in msgs if m["role"] == "user")
            gold = next(m["content"] for m in msgs if m["role"] == "assistant")
        except (KeyError, StopIteration) as exc:
            log(f"skipping row {i}: missing field ({exc})")
            continue

        chat = [
            {"role": "system", "content": sys_m},
            {"role": "user", "content": usr_m},
        ]
        prompt = tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        pred = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()
        ok = pred == gold
        correct += int(ok)
        examples.append(
            {"user": usr_m, "gold": gold, "pred": pred, "correct": ok}
        )
        log(
            f"[{i}/{len(rows)}] {'OK' if ok else 'XX'}  "
            f"gold={gold!r}  pred={pred!r}"
        )

    total = len(examples)
    accuracy = correct / total if total else 0.0
    log(f"\nexact-match on {total}: {correct}/{total} = {accuracy:.0%}")
    return {
        "correct": correct,
        "total": total,
        "accuracy": accuracy,
        "examples": examples,
    }
