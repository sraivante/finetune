# Ollama Fine-Tune Studio — User Guide

A Streamlit application that lets you fine-tune any Ollama model on your own documents using LoRA, convert the result to GGUF, register it back with Ollama, and chat with it to validate the result — all from a browser UI.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Installation](#2-installation)
3. [Application stages](#3-application-stages)
4. [Tab 1 — Configure](#4-tab-1--configure)
5. [Tab 2 — Dataset](#5-tab-2--dataset)
6. [Tab 3 — Fine-tune](#6-tab-3--fine-tune)
7. [Tab 4 — Chat & validate](#7-tab-4--chat--validate)
8. [Parameter reference](#8-parameter-reference)
9. [External APIs and integrations](#9-external-apis-and-integrations)
10. [File system layout](#10-file-system-layout)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. Prerequisites

| Requirement | Why it's needed | How to verify |
|---|---|---|
| **Python 3.10+** | Runs the Streamlit app and HuggingFace training stack | `python --version` |
| **Ollama 0.3+** running locally | Source models, generation model, final inference | `ollama list` |
| **Git** | Cloning `llama.cpp` for GGUF conversion | `git --version` |
| **~15 GB free disk** | Model downloads, training checkpoints, merged weights, GGUF | — |
| **8 GB RAM minimum** | Loading and training even a 0.5B model | — |
| **Internet access** (first run only) | HuggingFace base-model download, `llama.cpp` clone | — |
| **NVIDIA GPU + CUDA** *(optional)* | Drastically faster training; required for 3B+ models | `nvidia-smi` |

**Reality check on CPU-only machines:** without a GPU, only models up to ~1.1 B parameters are realistic to fine-tune (TinyLlama, Llama 3.2 1B, Qwen 2.5 0.5B). Anything larger will be very slow or run out of memory.

Make sure Ollama is running in another terminal *before* you start the app:

```bash
ollama serve
```

(Skip this if Ollama is installed as a Windows service — it's already running.)

---

## 2. Installation

### Quick start (Windows)

```bat
cd path\to\finetune
setup.bat
run.bat
```

`setup.bat` creates a `.venv`, installs `requirements.txt`, and shallow-clones `llama.cpp` next to the project. `run.bat` activates the venv and starts Streamlit on port 8501.

### Manual install (any OS)

```bash
python -m venv .venv
.venv\Scripts\activate           # or: source .venv/bin/activate
pip install -r requirements.txt
git clone --depth 1 https://github.com/ggerganov/llama.cpp llama.cpp
pip install -r llama.cpp/requirements.txt
streamlit run app.py
```

Open <http://localhost:8501>.

### Optional: build llama.cpp quantization binary

By default the GGUF converter writes `q8_0` files (~500 MB for a 0.5B model). To get smaller `q4_k_m` / `q5_k_m` files (~400 MB), build llama.cpp's quantize binary once:

```bash
cd llama.cpp
cmake -B build
cmake --build build --config Release
```

After that the converter auto-detects the binary and uses it.

### Pull at least one Ollama model

If you don't already have any:

```bash
ollama pull qwen2.5:0.5b-instruct       # smallest, CPU-friendly default
ollama pull qwen3:14b                   # strong generator for Q&A pairs
```

---

## 3. Application stages

The app is organized as four sequential tabs, designed to be used left-to-right. State (selected files, approved dataset, training log) is preserved across tabs.

```
┌───────────┐   ┌───────────┐   ┌───────────┐   ┌───────────────────┐
│ Configure │ → │  Dataset  │ → │ Fine-tune │ → │  Chat & validate  │
└───────────┘   └───────────┘   └───────────┘   └───────────────────┘
   upload         build,            train             query the new
   docs +         review,           LoRA →            model, manage
   set hyper-     approve           merge →           Ollama models,
   parameters    JSONL              GGUF →            delete unwanted
                                    register          ones
```

A **persistent sidebar** is visible on every tab. It contains:

- **Ollama model to fine-tune** — the *target* model that will be LoRA-tuned.
- Status badges (HF base mapping, CPU-friendliness warning).
- **Reset all to defaults** button — restores every form field to the recommended preset.

---

## 4. Tab 1 — Configure

This tab has two purposes: collect training documents and let the user override training hyperparameters.

### 4.1 Upload training documents

Drag-and-drop area accepts:

| Extension | Parser |
|---|---|
| `.pdf` | pypdf — per-page text extraction |
| `.docx` | python-docx — paragraph text |
| `.txt`, `.md`, `.markdown`, `.rst` | UTF-8 read with replace-on-error |
| `.csv`, `.json`, `.py` | Treated as plain text |

Files are saved to `data/uploads/`. The "Documents in uploads" expander lists them and lets you delete individual files.

### 4.2 Tuning options

All options are grouped into expanders, each pre-filled with a **recommended default**. The expanders are:

| Expander | What it controls |
|---|---|
| Dataset generation | How documents are chunked and how many Q&A pairs are produced per chunk |
| LoRA adapter | Rank, alpha, dropout, target modules |
| Training loop | Epochs, learning rate, batch size, scheduler, optimizer, fp16/bf16/4-bit |
| Conversion & registration | GGUF quantization, suffix for the new model name |
| Inference defaults | Chat temperature, top-p, max new tokens, system prompt |

See [Parameter reference](#8-parameter-reference) for what every individual field means and how to tune it.

> **Tip:** Use the sidebar's **Reset all to defaults** button to reload the recommended preset at any time.

---

## 5. Tab 2 — Dataset

The training set must be reviewed before any LoRA pass — bad pairs lead to model drift. This tab supports **combining multiple sources** (any mix of generated + uploaded + existing JSONL files) into a single approved dataset.

### 5.1 Approval banner

At the top, a status banner tells you whether a dataset is approved:

- ✅ **green** → approved file name + pair count; Fine-tune tab is unlocked.
- ℹ️ **blue** → no approval yet; Fine-tune is blocked.

### 5.2 Section 1 — Add new dataset files (optional)

Two collapsible sub-sections, both available at all times:

#### A. Generate from documents

A dedicated **Generation model** dropdown lets you pick *which* Ollama model produces the Q&A pairs (independent of the model being fine-tuned). Use a **strong** model here — tiny models often emit broken JSON and produce 0 pairs per chunk.

Workflow:
1. Pick generation model (e.g. `qwen3:14b` for cleanest JSON).
2. Click **Generate now**.
3. The app loads uploaded documents, chunks them, asks the generation model for `pairs/chunk` Q&A pairs per chunk, parses the JSON, and writes a JSONL file to `data/datasets/`.
4. The new file is automatically ticked in the file pool below.

A live progress bar and log show each chunk's pair count.

#### B. Upload my own JSONL

Drop in a pre-built JSONL file with the schema:

```json
{"messages": [
  {"role": "user", "content": "What is X?"},
  {"role": "assistant", "content": "X is ..."}
]}
```

One JSON object per line. Optional `"source"` field is supported. The app validates every line and shows a metrics summary (valid / total) plus a per-line errors expander.

Click **Save & add to pool** to copy it to `data/datasets/` and auto-add it to the file pool.

### 5.3 Section 2 — Choose JSONL files to combine

A multi-select widget lists every JSONL in `data/datasets/`, newest first, with each file's pair count. Tick as many files as you want; total pair count updates live.

Click **Merge & review →** when the selection looks right.

A **Download a selected file** popover lets you grab any selected JSONL to inspect outside the app.

### 5.4 Section 3 — Review merged dataset

The merged content from all selected files is loaded into an editable table:

| keep | question | answer | source |
|---|---|---|---|
| ☑ | (editable) | (editable) | (read-only — original filename) |

Actions:

- **Edit** any cell inline (click in).
- **Uncheck `keep`** to drop a row.
- **Add a row** at the bottom (the `+` icon).
- **Delete a row** by selecting it and pressing Del.

Live metrics: total rows, marked keep, dropped.

Two final buttons:

- **Save merged as new JSONL** → writes `merged-<timestamp>.jsonl` to `data/datasets/` with only your kept + edited rows. Adds it to the pool.
- **Approve & use for training** → saves AND locks this file as the approved training dataset. Banner turns green; Fine-tune unlocks.

---

## 6. Tab 3 — Fine-tune

Once a dataset is approved, this tab runs the LoRA training pipeline and registers the result with Ollama.

### 6.1 Header summary

Four metrics across the top show what's about to be trained:

| Metric | Source |
|---|---|
| Source model | Sidebar selection (the target Ollama tag) |
| HF base | The HuggingFace repo mapped to that Ollama tag |
| Approved dataset | Filename from Dataset tab approval |
| Training pairs | Pair count in the approved JSONL |

If any is missing, a warning explains what to do.

### 6.2 The pipeline (3 internal steps)

Clicking **Start fine-tuning** runs sequentially:

| Step | What happens | Output location |
|---|---|---|
| **1 — LoRA train** | Loads HF base in fp32 (CPU) / fp16 (GPU), wraps it with PEFT LoRA, trains on the approved JSONL using `transformers.Trainer`. Saves the adapter only. | `data/checkpoints/<run-id>/adapter/` |
| **2 — Merge + GGUF** | Loads base again, merges LoRA weights via `peft.PeftModel.merge_and_unload()`, saves the merged HF model, then runs llama.cpp's `convert_hf_to_gguf.py`. Falls back to `q8_0` if `llama-quantize` is missing. | `data/merged/<run-id>/`, `data/gguf/<run-id>.<quant>.gguf` |
| **3 — Register** | Writes a Modelfile with `FROM <gguf>`, default temperature, top_p, and system prompt. Runs `ollama create <new-name> -f Modelfile`. | New tag in `ollama list` (e.g. `qwen2.5-0.5b-instruct-ft-20260516-070745:latest`) |

A **live progress bar** advances from 0 → 100 across the three steps.

### 6.3 Training loss curve

While training runs, a real-time chart plots loss per optimizer step. Curve interpretation:

| Shape | Meaning |
|---|---|
| Smooth monotonic descent below ~1.5 | Healthy — model is learning |
| Plateau above ~2.5 | Underfit — raise epochs, LoRA rank, or pairs/chunk |
| U-shape (goes down then back up) | Overfit — lower epochs or raise LoRA dropout |
| Spiky / oscillating | Learning rate too high |

The curve persists across reruns of the app via `st.session_state`, so you can compare runs.

### 6.4 Live log

Every `transformers` log line, model load message, GGUF conversion line, and Ollama subprocess line streams into the log box. The last 200 lines are kept; older lines scroll off.

---

## 7. Tab 4 — Chat & validate

### 7.1 Manage models expander

A collapsible panel at the top lists every Ollama model on this machine with: name, size, **delete** button. Clicking delete opens a confirmation dialog — only after explicit confirmation is `ollama delete` called.

### 7.2 Chat with a model

- **Model to chat with** dropdown — lists every model live (refreshed each render). Defaults to the most-recently-fine-tuned model if one exists.
- **Clear chat** button — wipes the conversation buffer.
- **Delete model** button — opens the confirmation dialog for the currently-selected model.
- **Chat input** — standard Streamlit `st.chat_input`. The conversation streams from Ollama with the system prompt set in Configure → Inference defaults.

The conversation uses the inference defaults you configured (temperature, top-p, max new tokens).

---

## 8. Parameter reference

Every form field, in tab order.

### 8.1 Dataset generation (Configure)

| Field | Default | Typical range | What it does |
|---|---|---|---|
| **Q&A pairs per chunk** | 15 | 5–20 | How many Q&A pairs to request per chunk. Higher → larger dataset, more redundancy. Small models can't reliably exceed ~10. |
| **Max chunks (cap)** | 30 | 5–200 | Hard ceiling on chunks. Keeps CPU runs finite. |
| **Chunk size (characters)** | 300 | 200–1000 | Smaller chunks → more chunks → more pairs, but each pair sees less context. Larger chunks → richer content per pair but fewer total chunks. |
| **Chunk overlap** | 50 | 0–200 | Characters shared between consecutive chunks. Prevents Q&A topics from being cut off at chunk boundaries. |
| **Generation temperature** | 0.4 | 0.0–1.0 | Controls Q&A diversity. Too high → off-topic. Too low → repetitive paraphrases. |

### 8.2 LoRA adapter (Configure)

| Field | Default | Typical range | What it does |
|---|---|---|---|
| **LoRA rank (r)** | 16 | 4–64 | Capacity of the adapter. Higher r → more parameters → better fit, but slower and uses more RAM. |
| **LoRA alpha** | 32 | r to 4×r | Effective learning rate scaler. Rule of thumb: `alpha = 2 * r`. |
| **LoRA dropout** | 0.05 | 0.0–0.2 | Regularization. Raise to combat overfit on tiny datasets. |
| **Target modules** | `auto` | comma list or `auto` | Which transformer modules to LoRA-tune. `auto` lets PEFT pick (works for most architectures). For Llama/Qwen: `q_proj,k_proj,v_proj,o_proj`. |

### 8.3 Training loop (Configure)

| Field | Default | Typical range | What it does |
|---|---|---|---|
| **Epochs** | 8 | 1–30 | Passes over the dataset. Tiny datasets (< 100 pairs) need more (10-20); large datasets need fewer (2-3). |
| **Learning rate** | 3e-4 | 1e-5 to 1e-3 | LoRA loves 1e-4 to 5e-4. Too high → spiky loss curve. Too low → no learning. |
| **Per-device batch size** | 1 | 1–8 (CPU), 1–64 (GPU) | How many examples per forward pass. CPU users almost always need 1. |
| **Grad accumulation steps** | 4 | 1–32 | Effective batch = `batch_size × grad_accum`. Use this to simulate larger batches without the memory cost. |
| **Max sequence length** | 384 | 128–2048 | Truncates examples to this many tokens. Lower → less RAM, faster. Q&A pairs are short — 384 is plenty. |
| **Random seed** | 42 | any int | Makes runs reproducible. |
| **Warmup ratio** | 0.03 | 0.0–0.1 | Fraction of training spent ramping LR from 0 to peak. |
| **Weight decay** | 0.0 | 0.0–0.1 | L2 regularization. LoRA usually doesn't need it. |
| **LR scheduler** | `cosine` | cosine / linear / constant / constant_with_warmup | How LR changes over training. |
| **Optimizer** | `adamw_torch` | `adamw_torch` / `adamw_torch_fused` / `adafactor` / `sgd` | AdamW is the default everywhere. Adafactor uses less RAM. |
| **Use 4-bit (QLoRA)** | off | GPU only | Loads base model in 4-bit via bitsandbytes. Required for ≥7B models on consumer GPUs. |
| **fp16** / **bf16** | off | GPU only | Half-precision training. bf16 is preferred on Ampere+ GPUs. |

### 8.4 Conversion & registration (Configure)

| Field | Default | Choices | What it does |
|---|---|---|---|
| **GGUF quantization** | `q4_k_m` | `q4_k_m` / `q5_k_m` / `q8_0` / `f16` | Output GGUF size vs quality. `q4_k_m` requires `llama-quantize` binary (else falls back to `q8_0`). |
| **New model name suffix** | `ft` | any string | Final tag = `<original-name>-<suffix>-<timestamp>`. |

### 8.5 Inference defaults (Configure → used by Chat tab AND embedded in the new model's Modelfile)

| Field | Default | Range | What it does |
|---|---|---|---|
| **Chat temperature** | 0.3 | 0.0–1.5 | Lower → more deterministic, less drift to base-model priors. Higher → more creative. |
| **Chat top-p** | 0.9 | 0.0–1.0 | Nucleus sampling. 0.9 is a sane default. |
| **Max new tokens** | 512 | 32–8192 | Maximum length of each reply. |
| **System prompt** | (grounding prompt) | any text | Injected before every chat turn. Use this to constrain the model's behavior (e.g. "Do not generate code"). |

### 8.6 Dataset tab — additional fields

| Field | Default | Notes |
|---|---|---|
| **Generation model** | same as sidebar's target | Pick a STRONG model here (qwen3:14b, mistral-nemo:12b). Has no effect on what gets fine-tuned. |

---

## 9. External APIs and integrations

The application does NOT call any cloud LLM API. All inference happens via local Ollama. The only external network calls are:

| Endpoint | Purpose | When |
|---|---|---|
| `http://localhost:11434/api/tags` | List Ollama models | Every page render (`pipeline/ollama_client.list_models`) |
| `http://localhost:11434/api/chat` | Chat with a model | Streaming chat (Chat tab) AND Q&A generation (Dataset tab) |
| `http://localhost:11434/api/delete` | Delete a model | When user confirms deletion in Manage models dialog |
| `ollama create -f Modelfile` (CLI subprocess) | Register the fine-tuned GGUF as a new model tag | Final step of every Fine-tune run |
| `https://huggingface.co/<repo>` | Download base model weights via `transformers.from_pretrained` | First fine-tune of a previously-unused base model |
| `https://github.com/ggerganov/llama.cpp` | Shallow clone for GGUF conversion script | First Fine-tune ever (or first run after deleting `llama.cpp/`) |

If you want to use **gated HuggingFace models** (Llama 3, Mistral, etc.) you must accept their license on huggingface.co AND set `HF_TOKEN` in the environment or run `huggingface-cli login`.

### Wire format used internally

The training JSONL uses OpenAI-style chat messages:

```json
{
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "source": "tuning.txt"
}
```

The trainer formats each example through the tokenizer's chat template, computes loss on the full sequence (including the user turn — standard SFT setup).

---

## 10. File system layout

```
finetune/
├── app.py                  # Streamlit entry — all UI lives here
├── config.py               # DEFAULT_CONFIG + Ollama → HF mapping
├── requirements.txt
├── setup.bat / run.bat
├── pipeline/
│   ├── __init__.py
│   ├── ollama_client.py    # list / generate / chat-stream / create / delete wrappers
│   ├── document_loader.py  # PDF/DOCX/TXT/MD/CSV/JSON/PY parsing + chunking
│   ├── dataset_builder.py  # JSONL Q&A generation using an Ollama model
│   ├── trainer.py          # PEFT LoRA training, gradient checkpointing, metric callback
│   └── converter.py        # LoRA merge → GGUF via llama.cpp → Ollama register
├── data/                   # All gitignored
│   ├── uploads/            # Your source documents
│   ├── datasets/           # Generated/uploaded/edited JSONL files
│   ├── checkpoints/        # LoRA adapter weights, one folder per run
│   ├── merged/             # Merged HF models (adapter folded into base)
│   └── gguf/               # GGUF + Modelfile pairs
└── llama.cpp/              # Cloned automatically on first Fine-tune (gitignored)
```

You can freely delete anything in `data/` to reclaim disk; just keep `data/uploads/` if you want to regenerate datasets.

---

## 11. Troubleshooting

### "Ollama unreachable"

`ollama serve` isn't running, or the daemon crashed.

```bash
ollama serve            # starts the daemon in the foreground
```

Or check the Windows service: `Get-Service Ollama` in PowerShell.

### "No HF mapping for `<model>`"

The selected Ollama tag isn't in `OLLAMA_TO_HF` inside `config.py`. Add an entry:

```python
"my-model:tag": "huggingface-user/MyModel-Instruct",
```

LoRA fine-tuning needs the original HF weights — Ollama only ships GGUF.

### "Got 0 pairs from N chunks"

The generation model is producing broken JSON. Fix by:

1. Dataset tab → Generate from documents → set **Generation model** to a stronger model like `qwen3:14b` or `qwen2.5-coder:14b`. The tiny 0.5B model is unreliable at structured output.
2. Lower **Q&A pairs per chunk** to 5–8.
3. Increase **Chunk size** to 400–500 so each chunk has enough content to support many distinct questions.

### Training loss won't drop below ~2.5

Underfit. Try (in order):

1. More training data — generate again with `pairs/chunk = 15` and `chunk_size = 300` to get more diverse pairs from the same documents.
2. More epochs — bump from 8 to 15-20.
3. Larger LoRA rank — bump from 16 to 32.
4. Larger base model — `qwen2.5:3b-instruct` will fit much more capacity (slower on CPU).

### "element 0 of tensors does not require grad"

Already fixed in `pipeline/trainer.py`: gradient checkpointing + PEFT needs `model.enable_input_require_grads()` and `use_reentrant=False`. If you somehow regress, this is the cause.

### Out of memory during training

Lower in this order:

1. **`per_device_batch_size`** to 1
2. **`max_seq_length`** to 256 or 128
3. Increase **`gradient_accumulation_steps`** to compensate
4. Pick a smaller base model

### Out of memory during merge step

The merge loads the full base model in fp16. For 7B+ models you need ~16 GB RAM. Close other apps, or skip merge by manually serving the LoRA adapter via vLLM/transformers — but Ollama needs a GGUF.

### "llama-quantize not found" warning

The converter automatically falls back to `q8_0` (~500 MB for 0.5B). To get true `q4_k_m` (~400 MB), build llama.cpp:

```bash
cd llama.cpp
cmake -B build
cmake --build build --config Release
```

### UnicodeDecodeError 'charmap' on Windows

Already fixed: `ollama create` output is UTF-8 with spinner glyphs; the subprocess is now opened with `encoding="utf-8", errors="replace"`. If you regress, this is the cause.

### The fine-tuned model hallucinates / answers off-topic

Two causes:

1. Dataset too small → model didn't learn the domain → falls back to base-model priors. See "Training loss won't drop below 2.5" above.
2. System prompt isn't constraining behavior. Edit Configure → Inference defaults → System prompt to explicitly fence off failure modes, e.g.:

```text
You are HealthPlan Assist, a virtual assistant for insurance advisors.
You only answer using the company policy documents you were trained on.
If a question is outside that scope, say so briefly.
Do not generate code or unrelated examples.
```

The system prompt is embedded in the GGUF Modelfile at registration time, so every subsequent chat session uses it automatically.

### Streamlit session widget error: "cannot be modified after the widget X is instantiated"

This is a Streamlit rule: any widget with a `key=` parameter "owns" that key. You can't assign to `st.session_state[key]` after the widget renders. In this codebase we use a separate non-widget-bound key (e.g. `dataset_pool`) plus a versioned widget key (e.g. `ms_files_v0`, `ms_files_v1`) to programmatically refresh the multi-select.

---

*Last updated: 2026-05.*
