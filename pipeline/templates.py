"""Identify the correct chat template + stop tokens for a fine-tuned model,
so the generated Ollama Modelfile matches what the model was trained on.

Why: Ollama does NOT reliably apply a model's embedded chat template (Qwen's
complex tool-aware Jinja, for example). If the Modelfile has no explicit
TEMPLATE, the prompt can reach the model unwrapped and it falls back to base
completion behaviour. So we write an explicit TEMPLATE + stop, chosen per family.

Detection strategy (most reliable first):
  1. Sniff the model's OWN chat template (chat_template.jinja, or the
     `chat_template` field in tokenizer_config.json) for delimiter tokens.
     This keys on what the model actually expects, regardless of its name.
  2. Fall back to config.json `model_type`.
  3. Safe default: ChatML.
"""
from __future__ import annotations

import json
from pathlib import Path

# --- Ollama Go-template strings (note the real newlines: they matter) ---------

CHATML = """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ .Response }}<|im_end|>
"""

LLAMA3 = """{{ if .System }}<|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|>{{ end }}<|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{{ .Response }}<|eot_id|>"""

MISTRAL = """{{ if .System }}[INST] {{ .System }}

{{ .Prompt }}[/INST]{{ else }}[INST] {{ .Prompt }}[/INST]{{ end }} {{ .Response }}</s>"""

GEMMA = """<start_of_turn>user
{{ if .System }}{{ .System }}

{{ end }}{{ .Prompt }}<end_of_turn>
<start_of_turn>model
{{ .Response }}<end_of_turn>
"""

ZEPHYR = """{{ if .System }}<|system|>
{{ .System }}</s>
{{ end }}<|user|>
{{ .Prompt }}</s>
<|assistant|>
{{ .Response }}</s>
"""

PHI3 = """{{ if .System }}<|system|>
{{ .System }}<|end|>
{{ end }}<|user|>
{{ .Prompt }}<|end|>
<|assistant|>
{{ .Response }}<|end|>
"""

# format key -> spec. `markers` are substrings to look for in the model's own
# chat template. `model_types` are config.json model_type fallbacks.
CHAT_FORMATS = {
    "chatml":  {"template": CHATML,  "stop": ["<|im_end|>"],
                "markers": ["<|im_start|>"],
                "model_types": ["qwen2", "qwen3", "yi"],
                "examples": "Qwen2.5, Qwen3, Yi-1.5"},
    "llama3":  {"template": LLAMA3,  "stop": ["<|eot_id|>"],
                "markers": ["<|start_header_id|>"],
                "model_types": ["llama"],   # llama3.x (llama2 differs)
                "examples": "Llama-3.1 / 3.2 / 3.3 Instruct"},
    "gemma":   {"template": GEMMA,   "stop": ["<end_of_turn>"],
                "markers": ["<start_of_turn>"],
                "model_types": ["gemma", "gemma2"],
                "examples": "Gemma-2-2B / 9B-it"},
    "mistral": {"template": MISTRAL, "stop": ["</s>"],
                "markers": ["[INST]"],
                "model_types": ["mistral", "mixtral"],
                "examples": "Mistral-7B-Instruct v0.2/0.3, Mixtral"},
    "phi3":    {"template": PHI3,    "stop": ["<|end|>"],
                "markers": ["<|end|>"],
                "model_types": ["phi3"],
                "examples": "Phi-3-mini / medium-instruct"},
    "zephyr":  {"template": ZEPHYR,  "stop": ["</s>"],
                "markers": ["<|user|>"],
                "model_types": [],          # caught by marker sniff (TinyLlama)
                "examples": "TinyLlama-1.1B-Chat, Zephyr-7B"},
}

# Check specific markers before generic ones (phi3 <|end|> before zephyr <|user|>).
_MARKER_ORDER = ["chatml", "llama3", "gemma", "mistral", "phi3", "zephyr"]


def _read_chat_template_text(model_dir: Path) -> str:
    jinja = model_dir / "chat_template.jinja"
    if jinja.exists():
        return jinja.read_text(encoding="utf-8", errors="ignore")
    tc = model_dir / "tokenizer_config.json"
    if tc.exists():
        try:
            data = json.loads(tc.read_text(encoding="utf-8"))
        except Exception:
            return ""
        ct = data.get("chat_template")
        if isinstance(ct, list):                 # some repos store a list
            ct = (ct[0] or {}).get("template", "") if ct else ""
        return ct or ""
    return ""


def _model_type(model_dir: Path) -> str:
    cfg = model_dir / "config.json"
    if cfg.exists():
        try:
            return (json.loads(cfg.read_text(encoding="utf-8"))
                    .get("model_type") or "").lower()
        except Exception:
            pass
    return ""


def detect_chat_format(model_dir) -> dict:
    """Return {'key', 'template', 'stop', 'source'} for the model in model_dir.

    `model_dir` should be the merged HF model (or adapter) folder — anything
    containing chat_template.jinja / tokenizer_config.json / config.json.
    """
    model_dir = Path(model_dir)
    tmpl_text = _read_chat_template_text(model_dir)

    # 1) sniff the model's own template
    for key in _MARKER_ORDER:
        if any(m in tmpl_text for m in CHAT_FORMATS[key]["markers"]):
            fmt = CHAT_FORMATS[key]
            return {"key": key, "template": fmt["template"],
                    "stop": fmt["stop"], "source": "chat_template markers"}

    # 2) fall back to config.json model_type
    mt = _model_type(model_dir)
    for key, fmt in CHAT_FORMATS.items():
        if mt and mt in fmt["model_types"]:
            return {"key": key, "template": fmt["template"],
                    "stop": fmt["stop"], "source": f"model_type={mt}"}

    # 3) safe default
    fmt = CHAT_FORMATS["chatml"]
    return {"key": "chatml", "template": fmt["template"],
            "stop": fmt["stop"], "source": "default(chatml)"}


def detect_with_candidates(model_dir) -> dict:
    """Like `detect_chat_format`, but also reports per-format applicability.

    Useful for UI: shows the user which format is the recommended pick based
    on the model's own jinja chat_template, which are merely compatible, and
    which don't fit this model family at all.

    Returns:
        {
            "recommended": str,            # the auto-detected key
            "source": str,                 # how it was detected
            "template": str,               # template string of the recommended
            "stop": list[str],             # stop tokens of the recommended
            "chat_template_found": bool,   # did the folder expose a jinja?
            "model_type": str,             # config.json model_type (lower)
            "candidates": {
                <key>: {
                    "status": "recommended" | "compatible" | "not_applicable",
                    "template": str,
                    "stop": list[str],
                    "examples": str,
                    "reason": str,
                },
                ...
            },
        }

    Status meaning:
      - "recommended": the format detect_chat_format() chose for this model.
      - "compatible":  not the top pick, but its markers were also present in
                       the chat_template OR its model_types list includes
                       config.json's model_type. Usable, but not the best fit.
      - "not_applicable": no marker hit and no model_type hit — almost
                       certainly wrong for this model.
    """
    model_dir = Path(model_dir)
    tmpl_text = _read_chat_template_text(model_dir)
    mt = _model_type(model_dir)

    primary = detect_chat_format(model_dir)
    recommended = primary["key"]

    candidates: dict = {}
    for key, fmt in CHAT_FORMATS.items():
        marker_hit = bool(tmpl_text) and any(
            m in tmpl_text for m in fmt["markers"]
        )
        type_hit = bool(mt) and mt in fmt["model_types"]

        if key == recommended:
            status = "recommended"
            if marker_hit:
                reason = (
                    f"marker {fmt['markers'][0]!r} found in the model's "
                    f"own chat_template"
                )
            elif type_hit:
                reason = f"config.json model_type={mt} matches this family"
            else:
                reason = (
                    "safe default — no chat_template or model_type was "
                    "available to detect from"
                )
        elif marker_hit or type_hit:
            status = "compatible"
            bits = []
            if marker_hit:
                bits.append("marker present in chat_template")
            if type_hit:
                bits.append(f"model_type={mt}")
            reason = "Also matches: " + ", ".join(bits)
        else:
            status = "not_applicable"
            reason = "No marker match and model_type doesn't fit this family"

        candidates[key] = {
            "status": status,
            "template": fmt["template"],
            "stop": fmt["stop"],
            "examples": fmt["examples"],
            "reason": reason,
        }

    return {
        "recommended": recommended,
        "source": primary["source"],
        "template": primary["template"],
        "stop": primary["stop"],
        "chat_template_found": bool(tmpl_text),
        "model_type": mt,
        "candidates": candidates,
    }


if __name__ == "__main__":
    import sys
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    r = detect_chat_format(d)
    print(f"detected: {r['key']}  (via {r['source']})")
    print(f"stop: {r['stop']}")
    print("template:\n" + r["template"])
