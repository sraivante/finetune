"""Default + recommended config and Ollama→HuggingFace base-model mapping."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
DATASETS_DIR = DATA_DIR / "datasets"
CHECKPOINTS_DIR = DATA_DIR / "checkpoints"
MERGED_DIR = DATA_DIR / "merged"
GGUF_DIR = DATA_DIR / "gguf"
LLAMA_CPP_DIR = ROOT / "llama.cpp"

for _d in (UPLOADS_DIR, DATASETS_DIR, CHECKPOINTS_DIR, MERGED_DIR, GGUF_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Ollama model tag → HuggingFace repo id used as the LoRA base.
# Fine-tuning needs the original HF weights (Ollama only ships GGUF).
OLLAMA_TO_HF: dict[str, str] = {
    "qwen2.5:0.5b-instruct": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen2.5:3b-instruct": "Qwen/Qwen2.5-3B-Instruct",
    "qwen2.5-coder:14b": "Qwen/Qwen2.5-Coder-14B-Instruct",
    "qwen3:14b": "Qwen/Qwen3-14B",
    "tinyllama:latest": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "llama3.2:1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama3.2:latest": "meta-llama/Llama-3.2-3B-Instruct",
    "llama3.1:latest": "meta-llama/Llama-3.1-8B-Instruct",
    "mistral:7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "mistral-nemo:12b": "mistralai/Mistral-Nemo-Instruct-2407",
    "mistral-small:24b": "mistralai/Mistral-Small-Instruct-2409",
    "deepseek-llm:7b": "deepseek-ai/deepseek-llm-7b-chat",
    "deepseek-r1:32b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
}

# Models that are realistic to fine-tune on CPU (≤ ~1.5B params).
CPU_FRIENDLY = {
    "qwen2.5:0.5b-instruct",
    "tinyllama:latest",
    "llama3.2:1b",
}

DEFAULT_CONFIG: dict = {
    # --- Source model & dataset generation ---
    "ollama_model": "qwen2.5:0.5b-instruct",  # recommended CPU default
    # The model that *generates* training Q&A pairs is independent of the
    # one being *fine-tuned*. Pick a stronger model here for cleaner JSON
    # output; small models struggle with structured generation.
    "generation_model": "qwen2.5:0.5b-instruct",
    # Defaults tuned for SMALL policy/spec docs (~2-20 KB). The goal is to
    # produce enough paraphrased Q&A pairs that the LoRA actually generalises
    # instead of just memorising. With these settings a 2 KB doc gives ~90
    # training pairs, which is the regime where LoRA on a 0.5B model can move.
    "qa_pairs_per_chunk": 15,
    "max_chunks": 30,
    "chunk_size": 300,          # characters — smaller chunks → more chunks
    "chunk_overlap": 50,
    "generation_temperature": 0.4,
    # --- LoRA hyperparameters ---
    "lora_r": 16,               # more capacity to adapt to a new domain
    "lora_alpha": 32,           # rule of thumb: alpha = 2 * r
    "lora_dropout": 0.05,
    "lora_target_modules": "auto",
    # --- Training hyperparameters ---
    "epochs": 8,                # tiny dataset → more passes (watch overfit)
    "learning_rate": 3e-4,
    "per_device_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "max_seq_length": 384,      # Q&A pairs are short; saves CPU/RAM
    "warmup_ratio": 0.03,
    "weight_decay": 0.0,
    "lr_scheduler_type": "cosine",
    "optimizer": "adamw_torch",
    "seed": 42,
    "use_4bit": False,          # bitsandbytes — GPU only
    "fp16": False,              # GPU only
    "bf16": False,              # newer GPU only
    # --- Conversion / Ollama registration ---
    "gguf_quant": "q4_k_m",     # q4_k_m | q5_k_m | q8_0 | f16
    "new_model_suffix": "ft",   # final name = <base>-<suffix>-<timestamp>
    # --- Inference defaults (used by chat tab) ---
    "chat_temperature": 0.3,    # lower → less drift to base-model priors
    "chat_top_p": 0.9,
    "chat_num_predict": 512,
    "system_prompt": (
        "You are a domain-specific assistant fine-tuned on the user's documents. "
        "Answer strictly from that knowledge. If a question is outside the "
        "documents, say so briefly. Do not generate code or unrelated examples."
    ),
}


def get_default_config() -> dict:
    """Return a deep copy so the caller can mutate freely."""
    return deepcopy(DEFAULT_CONFIG)


def hf_repo_for(ollama_tag: str) -> str | None:
    """Map an Ollama tag to its HF base repo, or None if unknown."""
    return OLLAMA_TO_HF.get(ollama_tag)
