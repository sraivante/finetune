# Ollama Fine-Tune Studio

A Streamlit app that fine-tunes any Ollama model on your own documents using
LoRA, converts the result to GGUF, registers it back with Ollama, and lets you
chat with the new model to validate it.

```
Upload docs → Generate Q&A → LoRA train → Merge → GGUF → ollama create → Chat
```

---

## 1. Prerequisites

| Tool | Why | Check |
|------|-----|-------|
| Python 3.10+ | Runs the app and training | `python --version` |
| Ollama (running) | Source models + final inference | `ollama list` |
| Git | One-shot clone of llama.cpp for GGUF conversion | `git --version` |
| ~10 GB free disk | Training caches, merged model, GGUF | — |
| (Optional) NVIDIA GPU + CUDA | Lets you fine-tune 3B–7B models in minutes instead of hours | `nvidia-smi` |

> **CPU-only reality check.** Without a GPU, only models up to ~1.1B
> parameters are realistic to fine-tune. The app's default
> (`qwen2.5:0.5b-instruct`) is tuned for this case. `tinyllama` and
> `llama3.2:1b` are the other CPU-friendly picks.

Make sure Ollama is running in another terminal *before* you start the app:

```bat
ollama serve
```

(Skip this if Ollama is installed as a Windows service — it's already running.)

---

## 2. Install

From the project folder:

```bat
cd E:\A_FP_AI\LLM_FINE_TUNE\finetune
setup.bat
```

`setup.bat` will:

1. Create a `.venv` virtual environment
2. `pip install -r requirements.txt` (Streamlit, transformers, peft, datasets, torch, pypdf, python-docx, ollama-python, …)
3. Shallow-clone `llama.cpp` next to the project so the GGUF converter is available
4. Install llama.cpp's converter requirements

If the source Ollama model you want isn't already pulled, do that first:

```bat
ollama pull qwen2.5:0.5b-instruct
```

---

## 3. Start the app

```bat
run.bat
```

Streamlit will open at <http://localhost:8501>.

If you prefer to do it manually:

```bat
call .venv\Scripts\activate.bat
streamlit run app.py
```

---

## 4. End-to-end walkthrough

### Sidebar
- **Ollama model to fine-tune** — populated live from `ollama list`. The
  default `qwen2.5:0.5b-instruct` is the recommended CPU pick.
- **HF base** — the matching HuggingFace repo we'll load weights from. (LoRA
  needs the original weights — Ollama only ships GGUF.) If you see *No HF
  mapping*, add an entry to `OLLAMA_TO_HF` in `config.py`.
- **Reset all to defaults** — restores every option to the recommended preset.

### Tab 1 — Configure
1. **Upload documents.** PDF / DOCX / TXT / MD / CSV / JSON / PY. Files are
   saved under `data/uploads/`.
2. **Review options.** Every hyperparameter is pre-filled with a recommended
   default, grouped into expanders:
   - **Dataset generation** — Q&A pairs per chunk, max chunks, temperature, chunk size/overlap
   - **LoRA adapter** — rank (r), alpha, dropout, target modules
   - **Training loop** — epochs, LR, batch size, grad-accumulation, max-seq-length, scheduler, optimizer, fp16/bf16/4-bit
   - **Conversion & registration** — GGUF quantization, new model name suffix
   - **Inference defaults** — chat temperature, top-p, max new tokens, system prompt

> Tip: leave everything alone for your first run and use the **Reset all to
> defaults** button if you want to start over.

### Tab 2 — Fine-tune
Click **Start fine-tuning**. The 5-step pipeline runs with live logs and a
progress bar:

| Step | What happens | Output |
|------|--------------|--------|
| 1 | Documents loaded and chunked | in-memory |
| 2 | Source Ollama model generates Q&A pairs from each chunk | `data/datasets/<run>.jsonl` |
| 3 | LoRA fine-tune via `transformers` + `peft` | `data/checkpoints/<run>/adapter/` |
| 4 | Merge adapter into base weights + convert to GGUF | `data/merged/<run>/`, `data/gguf/<run>.<quant>.gguf` |
| 5 | Generate a `Modelfile` and run `ollama create` | new tag in `ollama list` |

The final tag looks like:

```
qwen2.5-0.5b-instruct-ft-20260516-103212
```

### Tab 3 — Chat & validate
Pick the new fine-tuned model (it's pre-selected for you) and ask it questions
about the documents you uploaded. Adjust temperature / top-p / max tokens in
the *Inference defaults* expander on the Configure tab.

---

## 5. Recommended starter run

If you just want to confirm the pipeline works end-to-end on CPU:

1. Pull the tiny default: `ollama pull qwen2.5:0.5b-instruct`
2. Upload **one small PDF or txt file** (a few pages)
3. On Configure → Dataset generation, set **Max chunks = 5** for the first run
4. Leave everything else at defaults
5. Hit *Start fine-tuning*

End-to-end this takes roughly 5–20 minutes on a modern CPU (most of it is
downloading the HuggingFace base model the first time and the LoRA step).
Once the base model is cached, subsequent runs are much faster.

---

## 6. Where files live

```
data/
  uploads/        ← your documents (gitignored)
  datasets/       ← generated training JSONL
  checkpoints/    ← LoRA adapter weights
  merged/         ← merged HF model (adapter folded into base)
  gguf/           ← final GGUF + Modelfile
llama.cpp/        ← cloned by setup.bat (gitignored)
```

You can safely delete anything in `data/` to reclaim disk space; just keep
the source documents in `data/uploads/` if you want to retrain.

---

## 7. Troubleshooting

**"Ollama unreachable"** — start `ollama serve` in another terminal, or check
that the Ollama service is running.

**"No HF mapping for `<model>`"** — open `config.py` and add the Ollama tag →
HuggingFace repo to the `OLLAMA_TO_HF` dict.

**Out of memory during training** — drop `per_device_batch_size` to 1,
increase `gradient_accumulation_steps`, lower `max_seq_length` (e.g. 256), and
pick a smaller base model.

**Out of memory during merge** — close other apps; the merge step loads the
full base model in fp16. For 7B+ models on CPU you'll likely need a machine
with 32 GB+ RAM.

**`llama-quantize` not found** — the converter will warn and fall back to
producing an `f16` (or `q8_0`) GGUF, which Ollama can still load. To get
smaller quantizations like `q4_k_m`/`q5_k_m`, build llama.cpp:

```bat
cd llama.cpp
cmake -B build
cmake --build build --config Release
```

**`bitsandbytes` errors** — bitsandbytes only works with CUDA on Windows.
Leave the *Use 4-bit (QLoRA, GPU only)* checkbox **off** if you don't have a
GPU.

**Gated HuggingFace models (e.g. Llama 3, Mistral)** — accept the license on
huggingface.co, then run `huggingface-cli login` and paste your token before
training.

---

## 8. Extending the app

- **Add more Ollama→HF mappings.** Edit `OLLAMA_TO_HF` in `config.py`.
- **Change recommended defaults.** Edit `DEFAULT_CONFIG` in `config.py`. The
  sidebar's *Reset to defaults* button always uses whatever is in that dict.
- **Add a new file format.** Add a reader to `_READERS` in
  `pipeline/document_loader.py`.
- **Tweak the Q&A generator prompt.** See `QA_SYSTEM_PROMPT` in
  `pipeline/dataset_builder.py`.
