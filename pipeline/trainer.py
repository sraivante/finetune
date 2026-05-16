"""LoRA fine-tuning using transformers + PEFT."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass
class TrainResult:
    adapter_dir: Path
    base_model_repo: str
    num_examples: int
    final_loss: float | None


def _format_example(messages: list[dict], tokenizer, max_seq_length: int) -> dict:
    """Use the tokenizer's chat template to flatten messages → input_ids/labels."""
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    enc = tokenizer(
        text,
        truncation=True,
        max_length=max_seq_length,
        padding=False,
        return_tensors=None,
    )
    enc["labels"] = enc["input_ids"].copy()
    return enc


def train_lora(
    *,
    dataset_path: Path,
    base_model_repo: str,
    output_dir: Path,
    cfg: dict,
    log_cb: Callable[[str], None] | None = None,
    metrics_cb: Callable[[dict], None] | None = None,
) -> TrainResult:
    """Run a single LoRA fine-tune. Returns adapter path."""
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )

    def log(msg: str) -> None:
        if log_cb:
            log_cb(msg)

    output_dir.mkdir(parents=True, exist_ok=True)
    log(f"Loading tokenizer: {base_model_repo}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_repo, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- dtype / quantisation strategy ---
    use_4bit = bool(cfg.get("use_4bit")) and torch.cuda.is_available()
    bnb_config = None
    if use_4bit:
        try:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            log("Using 4-bit (bitsandbytes) quantization.")
        except Exception as exc:
            log(f"bitsandbytes unavailable ({exc}); falling back to fp32.")
            use_4bit = False

    torch_dtype = torch.float32
    if torch.cuda.is_available():
        if cfg.get("bf16"):
            torch_dtype = torch.bfloat16
        elif cfg.get("fp16"):
            torch_dtype = torch.float16

    log(f"Loading model (dtype={torch_dtype}, 4bit={use_4bit}) — this may take a while...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_repo,
        torch_dtype=torch_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        quantization_config=bnb_config,
        low_cpu_mem_usage=True,
    )
    if not torch.cuda.is_available():
        model.to("cpu")
    model.config.use_cache = False

    # Gradient checkpointing on CPU saves RAM but requires the input embeddings
    # to produce tensors with requires_grad=True, otherwise the backward pass
    # raises "element 0 of tensors does not require grad". Enable that here,
    # BEFORE wrapping in PEFT — get_peft_model freezes everything else.
    use_gradient_checkpointing = not torch.cuda.is_available()
    if use_gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def _make_inputs_require_grad(_module, _inp, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(_make_inputs_require_grad)

    # --- LoRA setup ---
    target_modules_cfg = (cfg.get("lora_target_modules") or "auto").strip()
    if target_modules_cfg == "auto" or not target_modules_cfg:
        target_modules = None  # PEFT will auto-detect for known archs
    else:
        target_modules = [m.strip() for m in target_modules_cfg.split(",") if m.strip()]

    lora = LoraConfig(
        r=int(cfg["lora_r"]),
        lora_alpha=int(cfg["lora_alpha"]),
        lora_dropout=float(cfg["lora_dropout"]),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # --- dataset ---
    log(f"Loading dataset: {dataset_path}")
    ds = load_dataset("json", data_files=str(dataset_path), split="train")

    def _map_fn(ex):
        return _format_example(ex["messages"], tokenizer, int(cfg["max_seq_length"]))

    ds = ds.map(_map_fn, remove_columns=ds.column_names, desc="tokenizing")
    log(f"Dataset size after tokenization: {len(ds)}")

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # --- training args ---
    args = TrainingArguments(
        output_dir=str(output_dir / "trainer"),
        num_train_epochs=float(cfg["epochs"]),
        learning_rate=float(cfg["learning_rate"]),
        per_device_train_batch_size=int(cfg["per_device_batch_size"]),
        gradient_accumulation_steps=int(cfg["gradient_accumulation_steps"]),
        warmup_ratio=float(cfg["warmup_ratio"]),
        weight_decay=float(cfg["weight_decay"]),
        lr_scheduler_type=str(cfg["lr_scheduler_type"]),
        optim=str(cfg["optimizer"]),
        logging_steps=1,  # log every optimizer step so the loss curve is dense
        save_strategy="no",
        report_to=[],
        fp16=bool(cfg.get("fp16")) and torch.cuda.is_available(),
        bf16=bool(cfg.get("bf16")) and torch.cuda.is_available(),
        seed=int(cfg["seed"]),
        gradient_checkpointing=use_gradient_checkpointing,  # save RAM on CPU
        gradient_checkpointing_kwargs={"use_reentrant": False} if use_gradient_checkpointing else None,
        remove_unused_columns=False,
    )

    class _LogCallback(TrainerCallback):
        def on_log(self, _args, state, _control, logs=None, **_):
            if not logs:
                return
            if log_cb:
                parts = [f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                         for k, v in logs.items()]
                log_cb("  " + " ".join(parts))
            if metrics_cb and ("loss" in logs or "learning_rate" in logs):
                metric = {"step": int(getattr(state, "global_step", 0) or 0)}
                for key in ("loss", "learning_rate", "epoch", "grad_norm"):
                    if key in logs:
                        try:
                            metric[key] = float(logs[key])
                        except (TypeError, ValueError):
                            pass
                metrics_cb(metric)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=collator,
        callbacks=[_LogCallback()],
    )

    log("Starting training...")
    train_out = trainer.train()
    final_loss = float(train_out.training_loss) if train_out.training_loss else None
    log(f"Training done. Final loss: {final_loss}")

    adapter_dir = output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    log(f"Adapter saved to {adapter_dir}")

    return TrainResult(
        adapter_dir=adapter_dir,
        base_model_repo=base_model_repo,
        num_examples=len(ds),
        final_loss=final_loss,
    )
