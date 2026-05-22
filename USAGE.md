# Ollama Fine-Tune Studio — User Guide

A Streamlit application that lets you fine-tune any Ollama model on your own documents using LoRA, convert the result to GGUF, register it back with Ollama, and chat with it to validate the result — all from a browser UI.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Installation](#2-installation)
3. [How to run the app (quick start)](#3-how-to-run-the-app-quick-start)
4. [Application stages](#4-application-stages)
5. [Tab 1 — Configure](#5-tab-1--configure)
6. [Tab 2 — Dataset](#6-tab-2--dataset)
7. [Tab 3 — Fine-tune](#7-tab-3--fine-tune)
8. [Tab 4 — Import adapter](#8-tab-4--import-adapter)
9. [Tab 5 — Evaluate](#9-tab-5--evaluate)
10. [Tab 6 — Chat & validate](#10-tab-6--chat--validate)
11. [Command-line evaluator (`scripts/eval_finetuned.py`)](#11-command-line-evaluator)
12. [Parameter reference](#12-parameter-reference)
13. [External APIs and integrations](#13-external-apis-and-integrations)
14. [File system layout](#14-file-system-layout)
15. [Troubleshooting](#15-troubleshooting)

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

## 3. How to run the app (quick start)

After [Installation](#2-installation) is done, **launching the app is two commands**:

```bat
:: 1. Make sure Ollama is running (skip if installed as a Windows service)
ollama serve

:: 2. Start the Streamlit app — opens http://localhost:8501 in your browser
run.bat
```

That's it. The sequence inside `run.bat` is just:

```bat
call .venv\Scripts\activate.bat
streamlit run app.py
```

So on macOS / Linux:

```bash
source .venv/bin/activate
streamlit run app.py
```

**Stopping the app:** click the terminal window that's running Streamlit and press <kbd>Ctrl</kbd>+<kbd>C</kbd>. If you lost track of the window, run `taskkill /F /IM streamlit.exe` on Windows, or `pkill -f streamlit` on macOS / Linux.

**Recommended first-run path:**

1. **Sidebar** — pick an Ollama model to fine-tune (default `qwen2.5:0.5b-instruct` is CPU-friendly).
2. **Configure** tab — upload one small document, leave hyperparameters at defaults.
3. **Dataset** tab — *Generate from documents* → pick a strong generation model → *Generate now* → *Approve & use for training*.
4. **Fine-tune** tab — *Start fine-tuning*. Wait for the loss curve and live log.
5. **Chat & validate** tab — chat with the new model.

If you already have a fine-tuned LoRA folder (e.g. from Colab), skip steps 2–4 and use the **Import adapter** tab instead.

---

## 4. Application stages

The app is organized as six tabs, designed to be used left-to-right. State (selected files, approved dataset, training log, eval results) is preserved across tabs.

```
┌──────────┐  ┌─────────┐  ┌──────────┐  ┌────────────────┐  ┌──────────┐  ┌──────────────┐
│ Configure│→│ Dataset │→│ Fine-tune│→│ Import adapter │→│ Evaluate │→│ Chat & valid.│
└──────────┘  └─────────┘  └──────────┘  └────────────────┘  └──────────┘  └──────────────┘
   upload      build,        train         bring your own      run a         query the
   docs +      review,       LoRA →        fine-tuned folder   JSONL test    new model,
   set hyper-  approve       merge →       (LoRA or merged)    set →         manage,
   parameters  JSONL         GGUF →        → GGUF → Ollama     HTML report   delete
                             register
```

The middle row of tabs (Fine-tune *and* Import adapter) both end at the same place: a new tag in `ollama list`. Use **Fine-tune** when you want the full pipeline from documents → dataset → training → Ollama. Use **Import adapter** when training already happened elsewhere (Colab, another machine) and you just want the merge + GGUF + Ollama-register steps.

A **persistent sidebar** is visible on every tab. It contains:

- **Ollama model to fine-tune** — the *target* model that will be LoRA-tuned.
- Status badges (HF base mapping, CPU-friendliness warning).
- **Reset all to defaults** button — restores every form field to the recommended preset.

---

## 5. Tab 1 — Configure

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

## 6. Tab 2 — Dataset

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

## 7. Tab 3 — Fine-tune

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

## 8. Tab 4 — Import adapter

Bring-your-own fine-tune entry point. Use this when training already happened *outside* this app — for example, on Google Colab, on a colleague's GPU box, or via the included `colab_finetune.ipynb` notebook. The tab accepts both **LoRA adapter folders** (PEFT output: `adapter_config.json` + `adapter_model.safetensors`) and **merged HF model folders** (`config.json` + `model.safetensors` / shards).

### 8.1 Inputs

| Field | What it is |
|---|---|
| **Path to fine-tuned folder** | Absolute path on this machine to the folder you want to import. The folder must contain either `adapter_config.json` (LoRA) **or** `config.json` + model weights (merged). |
| **Inspect path** button | Reads the folder, detects whether it's an adapter or a merged model, extracts the base model from `adapter_config.json` / `config.json`, and reports any issues (missing tokenizer, missing weights). |
| **Treat folder as** dropdown | *Auto-detect (recommended)* / *LoRA adapter* / *Merged HF model*. Overrides auto-detection if needed. |
| **Base HF model repo** | Required for LoRA mode — pre-filled from `adapter_config.json`. Ignored for merged mode (the folder already has full weights). |
| **New Ollama model name** | Final tag in `ollama list`. Allowed chars: `a-z 0-9 . _ -`. Auto-suggested as `<base>-imported-<timestamp>`. |
| **GGUF quantization** | Same as Fine-tune tab — `q4_k_m` / `q5_k_m` / `q8_0` / `f16`. |
| **Chat temperature / top-p / System prompt** | Embedded in the generated Modelfile, becomes the default for chat. |

### 8.2 What happens on Import

| Step | LoRA adapter folder | Merged HF model folder |
|---|---|---|
| 1 | Load base model from HuggingFace, attach the adapter via PEFT, run `merge_and_unload()`, save merged model to `data/merged/<run-id>/` | (skipped — folder is used directly) |
| 2 | Run `llama.cpp/convert_hf_to_gguf.py` on the merged folder | Same |
| 3 | Write a Modelfile and run `ollama create <new-tag> -f Modelfile` | Same |

Live log streams from each subprocess. On success, the new tag is also pre-selected in the Chat tab via `last_finetuned_model`.

### 8.3 Required disk space

LoRA-adapter mode loads the full HF base model in fp16 during the merge step. Plan for:

| Base model size | Disk needed during merge | Final GGUF (q8_0) |
|---|---|---|
| 0.5B | ~1 GB | ~600 MB |
| 1B (e.g. TinyLlama) | ~2 GB | ~1.1 GB |
| 3B | ~6 GB | ~3.2 GB |
| 7B | ~14 GB | ~7.3 GB |

Merged-mode imports skip the download — only GGUF conversion needs room.

---

## 9. Tab 5 — Evaluate

Runs a JSONL test set against any Ollama model (typically the one you just fine-tuned or imported), tallies pass / fail per row, and produces a downloadable HTML report.

### 9.1 Inputs

| Field | What it is |
|---|---|
| **Ollama model to test** | Dropdown of every model in `ollama list`. Pre-selects `last_finetuned_model` if available. |
| **Test JSONL** | Dropdown listing every `.jsonl` in `data/` and `data/eval/` (newest first). Pick *"(upload new file)"* to drop in a fresh file — uploads are persisted to `data/eval/uploaded-<timestamp>.jsonl`. |
| **Drop system prompt** | Toggle. If checked, sends only the user message — useful for *demonstrating* how much the model depends on the training-time system prompt. |
| **Temperature** | Defaults to **0.0** for classification reproducibility. |
| **Max new tokens** | Defaults to 64. Increase if expected answers are long. |

### 9.2 Expected JSONL format

One JSON object per line. `user` and `expected` are required; `system` and `task` are optional.

```jsonl
{"task": "investment_profile", "system": "You are a strict classifier. Map to Conservative -> 4.5, Moderate -> 6.5, Aggressive -> 8.5. Output ONLY the number.", "user": "agresive", "expected": "8.5"}
{"task": "yes_no_intent", "system": "Return only yes, no, or RETRY.", "user": "abort it", "expected": "no"}
{"task": "free_form", "user": "what is 2 + 2?", "expected": "4"}
```

A **sample file download** is exposed inside the "Expected JSONL format" expander on the tab.

Comparison rule: case-insensitive, trims whitespace and a trailing period. So `4.5`, `4.5.`, and `4.5` all match.

### 9.3 During the run

- Live counters: **Total / Passed / Failed / Done** update after every row.
- Progress bar `done / total`.
- **Abort** button — sets a flag the loop checks between rows. Worst-case lag is one Ollama call (typically 1–3 seconds).

To make Abort actually work, the tab uses a **one-row-per-rerun** pattern: each Streamlit cycle processes a single test row, appends the result, and calls `st.rerun()`. This keeps the UI responsive to button clicks during long evals.

### 9.4 After the run (or after Abort)

- **Per-task breakdown** table with a coloured accuracy progress column.
- **Download HTML report** button — produces a self-contained `.html` file (no external assets, no JS required for the core UI; uses one inline script only for the *Filter passed/failed* buttons). The same report is auto-saved to `data/eval/reports/eval-<model>-<timestamp>.html`.
- **In-app preview** expander listing the first 50 failed rows with their `user`, `expected`, and `got` values.

### 9.5 The HTML report

Open the downloaded file directly in any browser — works offline.

| Section | What it shows |
|---|---|
| Summary cards | Total / Passed / Failed / Accuracy %, plus the model name, timestamp, and whether the system prompt was sent or dropped. |
| Per-task breakdown | Each task with passed / failed / total / accuracy. Accuracy is colour-coded (green ≥ 90%, amber 70–90%, red < 70%). |
| Per-prompt results | One collapsible row per test case. Closed view: PASS / FAIL badge, task, user input, expected vs got. Click to expand — shows the full `system`, `user`, `expected`, `got`, and `error` text in `<pre>` blocks. |
| Filter buttons | *All / Failed only / Passed only* — toggles row visibility via a single inline script. |

---

## 10. Tab 6 — Chat & validate

### 10.1 Manage models expander

A collapsible panel at the top lists every Ollama model on this machine with: name, size, **delete** button. Clicking delete opens a confirmation dialog — only after explicit confirmation is `ollama delete` called.

### 10.2 Chat with a model

- **Model to chat with** dropdown — lists every model live (refreshed each render). Defaults to the most-recently-fine-tuned model if one exists.
- **Clear chat** button — wipes the conversation buffer.
- **Delete model** button — opens the confirmation dialog for the currently-selected model.
- **Chat input** — standard Streamlit `st.chat_input`. The conversation streams from Ollama with the system prompt set in Configure → Inference defaults.

The conversation uses the inference defaults you configured (temperature, top-p, max new tokens).

---

## 11. Command-line evaluator

If you'd rather run the eval from a shell (for CI, scripts, or just a tighter feedback loop), `scripts/eval_finetuned.py` does the same thing as the Evaluate tab without the UI.

```bat
:: From the project root, with .venv activated
python scripts/eval_finetuned.py --model <ollama-tag>

:: Drop the system prompt to see how much the model depends on it
python scripts/eval_finetuned.py --model <ollama-tag> --no-system

:: Other knobs
python scripts/eval_finetuned.py --model <tag> --test data/test_prompts.jsonl ^
                                 --temperature 0 --num-predict 32 --show-misses 50
```

Output: running accuracy every 10 rows, then a per-task breakdown table and the first N wrong answers. Exits with code 0 if every test passed, 1 if any failed.

The bundled `data/test_prompts.jsonl` is a 114-row sample (3 per task) drawn from a multi-classifier training set — replace it with your own to evaluate a different fine-tune.

---

## 12. Parameter reference

Every form field, in tab order.

### 12.1 Dataset generation (Configure)

| Field | Default | Typical range | What it does |
|---|---|---|---|
| **Q&A pairs per chunk** | 15 | 5–20 | How many Q&A pairs to request per chunk. Higher → larger dataset, more redundancy. Small models can't reliably exceed ~10. |
| **Max chunks (cap)** | 30 | 5–200 | Hard ceiling on chunks. Keeps CPU runs finite. |
| **Chunk size (characters)** | 300 | 200–1000 | Smaller chunks → more chunks → more pairs, but each pair sees less context. Larger chunks → richer content per pair but fewer total chunks. |
| **Chunk overlap** | 50 | 0–200 | Characters shared between consecutive chunks. Prevents Q&A topics from being cut off at chunk boundaries. |
| **Generation temperature** | 0.4 | 0.0–1.0 | Controls Q&A diversity. Too high → off-topic. Too low → repetitive paraphrases. |

### 12.2 LoRA adapter (Configure)

| Field | Default | Typical range | What it does |
|---|---|---|---|
| **LoRA rank (r)** | 16 | 4–64 | Capacity of the adapter. Higher r → more parameters → better fit, but slower and uses more RAM. |
| **LoRA alpha** | 32 | r to 4×r | Effective learning rate scaler. Rule of thumb: `alpha = 2 * r`. |
| **LoRA dropout** | 0.05 | 0.0–0.2 | Regularization. Raise to combat overfit on tiny datasets. |
| **Target modules** | `auto` | comma list or `auto` | Which transformer modules to LoRA-tune. `auto` lets PEFT pick (works for most architectures). For Llama/Qwen: `q_proj,k_proj,v_proj,o_proj`. |

### 12.3 Training loop (Configure)

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

### 12.4 Conversion & registration (Configure)

| Field | Default | Choices | What it does |
|---|---|---|---|
| **GGUF quantization** | `q4_k_m` | `q4_k_m` / `q5_k_m` / `q8_0` / `f16` | Output GGUF size vs quality. `q4_k_m` requires `llama-quantize` binary (else falls back to `q8_0`). |
| **New model name suffix** | `ft` | any string | Final tag = `<original-name>-<suffix>-<timestamp>`. |

### 12.5 Inference defaults (Configure → used by Chat tab AND embedded in the new model's Modelfile)

| Field | Default | Range | What it does |
|---|---|---|---|
| **Chat temperature** | 0.3 | 0.0–1.5 | Lower → more deterministic, less drift to base-model priors. Higher → more creative. |
| **Chat top-p** | 0.9 | 0.0–1.0 | Nucleus sampling. 0.9 is a sane default. |
| **Max new tokens** | 512 | 32–8192 | Maximum length of each reply. |
| **System prompt** | (grounding prompt) | any text | Injected before every chat turn. Use this to constrain the model's behavior (e.g. "Do not generate code"). |

### 12.6 Dataset tab — additional fields

| Field | Default | Notes |
|---|---|---|
| **Generation model** | same as sidebar's target | Pick a STRONG model here (qwen3:14b, mistral-nemo:12b). Has no effect on what gets fine-tuned. |

### 12.7 Import adapter tab

| Field | Default | Notes |
|---|---|---|
| **Path to fine-tuned folder** | — | Absolute path on the local machine. |
| **Treat folder as** | Auto-detect | Override only if auto-detection picked the wrong kind. |
| **Base HF model repo** | from `adapter_config.json` | Required for LoRA mode. Must be the *exact* repo the adapter was trained against. |
| **New Ollama model name** | `<base>-imported-<ts>` | Final tag in `ollama list`. |
| **GGUF quantization** | `q4_k_m` | Same fallback rules as the Fine-tune tab. |
| **Chat temperature / top-p / System prompt** | from Configure tab | Baked into the Modelfile. |

### 12.8 Evaluate tab

| Field | Default | Notes |
|---|---|---|
| **Ollama model to test** | `last_finetuned_model` | Any model in `ollama list`. |
| **Test JSONL** | newest file in `data/eval/` | Or *"(upload new file)"* to drop one in. |
| **Drop system prompt** | off | Useful diagnostic — see how reliant the model is on its training-time system prompt. |
| **Temperature** | 0.0 | Use 0 for classification reproducibility. |
| **Max new tokens** | 64 | Bump for longer expected answers. |

---

## 13. External APIs and integrations

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

## 14. File system layout

```
finetune/
├── app.py                  # Streamlit entry — all UI lives here
├── config.py               # DEFAULT_CONFIG + Ollama → HF mapping
├── requirements.txt
├── setup.bat / run.bat
├── colab_finetune.ipynb    # Optional: train a LoRA adapter on Colab,
│                           # then drop the folder into the Import adapter tab
├── pipeline/
│   ├── __init__.py
│   ├── ollama_client.py    # list / generate / chat-stream / create / delete wrappers
│   ├── document_loader.py  # PDF/DOCX/TXT/MD/CSV/JSON/PY parsing + chunking
│   ├── dataset_builder.py  # JSONL Q&A generation using an Ollama model
│   ├── trainer.py          # PEFT LoRA training, gradient checkpointing, metric callback
│   ├── converter.py        # LoRA merge → GGUF via llama.cpp → Ollama register;
│   │                       # also `inspect_external_folder` + `import_external_to_ollama`
│   │                       # for the Import adapter tab
│   └── eval_runner.py      # Load test JSONL, run rows against Ollama,
│                           # summarize, build self-contained HTML report
├── scripts/
│   ├── build_usage_html.py # Renders USAGE.md → usage.html
│   └── eval_finetuned.py   # Command-line counterpart of the Evaluate tab
├── data/                   # All gitignored
│   ├── uploads/            # Your source documents
│   ├── datasets/           # Generated/uploaded/edited JSONL files
│   ├── checkpoints/        # LoRA adapter weights, one folder per run
│   ├── merged/             # Merged HF models (adapter folded into base)
│   ├── gguf/               # GGUF + Modelfile pairs
│   ├── test_prompts.jsonl  # Bundled sample test set (3 rows × 38 tasks)
│   └── eval/
│       ├── *.jsonl         # Uploaded / saved test files
│       └── reports/        # Self-contained HTML eval reports
└── llama.cpp/              # Cloned automatically on first Fine-tune (gitignored)
```

You can freely delete anything in `data/` to reclaim disk; just keep `data/uploads/` if you want to regenerate datasets and `data/eval/` if you want to keep evaluation history.

---

## 15. Troubleshooting

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

### Import adapter: "Could not detect folder kind"

The folder you pointed at has neither `adapter_config.json` (LoRA) nor `config.json` + model weights (merged HF). Two common causes:

- You pointed at the *parent* directory of a Trainer output. Try the `checkpoint-XXXX/` subdirectory inside, or the named adapter folder produced by `trainer.save_model(...)`.
- The save was interrupted mid-write — only one or two files are present. Re-run the training and let it finish.

### Import adapter: "Base HF model repo is required to merge a LoRA adapter"

The detected `adapter_config.json` had no `base_model_name_or_path`, or the field was blank. Enter the HuggingFace repo manually (e.g. `TinyLlama/TinyLlama-1.1B-Chat-v1.0`). The base must be the *exact* repo the adapter was trained against — using a different one will silently produce garbage outputs.

### Evaluate: "Test file has no valid rows"

Every row must have at least `user` and `expected` fields. The loader raises on the first bad line with `line N: missing required 'user' and/or 'expected' field`. Open the file and check the offending line.

### Evaluate: Abort button doesn't react instantly

By design. The tab processes one test row per Streamlit rerun, so Abort takes effect after the *current* Ollama call returns (typically 1–3 seconds). If a single call is hanging for much longer, the bottleneck is Ollama, not the abort path.

### Evaluate: accuracy is far below 100% even on examples that were in the training data

Common — see the eval-failure analysis we keep in mind when reviewing reports:

1. **LoRA capacity too small.** If `target_modules` was left at the PEFT default (`q_proj`, `v_proj` only), the adapter has ~0.2% of params trainable. Re-train with all linear layers: `["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]`.
2. **Too few epochs with `constant` LR.** Try 8–12 epochs with `cosine` scheduler so the LR decays toward the end.
3. **Class imbalance.** If one label dominates a task (e.g. 30 `yes` / 23 `no` / 7 `RETRY`), the model learns the prior. Downsample majority or oversample minority.
4. **High-cardinality tasks under-sampled.** A task with 31 distinct outputs in 44 rows can't be learned reliably. Generate more examples per task, or split into smaller single-output classifiers.
5. **Base model too small.** TinyLlama 1.1B struggles with numeric-threshold reasoning and 37 simultaneous classifier heads. Qwen2.5-1.5B-Instruct or 3B-Instruct typically lifts accuracy by 5–10 points on the same data.

---

*Last updated: 2026-05-21.*
