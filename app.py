"""Streamlit app: fine-tune an Ollama model on uploaded documents."""
from __future__ import annotations

import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from config import (
    CHECKPOINTS_DIR,
    CPU_FRIENDLY,
    DATASETS_DIR,
    GGUF_DIR,
    MERGED_DIR,
    UPLOADS_DIR,
    get_default_config,
    hf_repo_for,
    OLLAMA_TO_HF,
)
from pipeline import (
    converter,
    dataset_builder,
    document_loader,
    eval_adapter,
    eval_runner,
    ollama_client,
    templates as chat_templates,
    trainer,
)

# ---------------------------------------------------------------------------
# Streamlit boilerplate
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Ollama Fine-Tune Studio",
    page_icon=":sparkles:",
    layout="wide",
)

CONFIG_KEYS = list(get_default_config().keys())


def _seed_state() -> None:
    # One-time: seed widget keys from defaults. Runs only on a brand-new session.
    if "initialized" not in st.session_state:
        for k, v in get_default_config().items():
            st.session_state[k] = v
        st.session_state["initialized"] = True
    # Idempotent: ensure auxiliary state keys exist every script run, so
    # adding a new key in code doesn't require killing existing sessions.
    st.session_state.setdefault("training_log", [])
    st.session_state.setdefault("training_metrics", [])
    st.session_state.setdefault("last_finetuned_model", None)
    st.session_state.setdefault("chat_history", [])
    # `dataset_pool` is our source of truth (free variable, no widget owns it).
    # The multi-select uses a versioned key so we can force it to re-read the
    # default when we programmatically add a file (Streamlit doesn't let us
    # write directly to a widget-bound key).
    st.session_state.setdefault("dataset_pool", [])
    st.session_state.setdefault("ms_version", 0)
    st.session_state.setdefault("merge_armed", False)    # show review section?
    st.session_state.setdefault("approved_dataset", None)
    st.session_state.setdefault("dataset_log", [])
    # generation_model may be missing in older sessions started before it was added.
    st.session_state.setdefault("generation_model",
                                st.session_state.get("ollama_model")
                                or get_default_config()["ollama_model"])


# ---------------------------------------------------------------------------
# OS-native folder / file picker (local Streamlit only)
# ---------------------------------------------------------------------------
def _pick_path_via_dialog(kind: str = "folder",
                          title: str = "",
                          initial_dir: str = "") -> str:
    """Open an OS-native folder/file dialog and return the chosen path.

    Runs the tk dialog in a subprocess so each pick is a fresh interpreter —
    Streamlit's script reruns interact badly with a long-lived tk root.
    Empty string on cancel or any failure (e.g. no display in headless env).
    """
    import subprocess
    import textwrap

    code = textwrap.dedent(f"""
        import sys
        try:
            import tkinter as tk
            from tkinter import filedialog
        except Exception:
            sys.exit(0)
        root = tk.Tk()
        root.withdraw()
        try:
            root.wm_attributes("-topmost", 1)
        except Exception:
            pass
        opts = {{"title": {title!r} or "Select"}}
        if {initial_dir!r}:
            opts["initialdir"] = {initial_dir!r}
        if {kind!r} == "folder":
            p = filedialog.askdirectory(**opts)
        else:
            p = filedialog.askopenfilename(**opts)
        if p:
            sys.stdout.write(p)
        root.destroy()
    """)
    try:
        out = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=300,
        )
        return (out.stdout or "").strip()
    except Exception:
        return ""


def _browse_button(label: str, *, state_key: str, kind: str,
                   dialog_title: str, button_key: str) -> None:
    """Render a "Browse..." button that updates `state_key` and reruns.

    Place this in a narrow column NEXT TO the matching st.text_input that
    binds the same `state_key`. On click, opens an OS-native picker; if the
    user chooses something we write to session_state and rerun so the
    text_input picks up the new value on the next script run.
    """
    if st.button(label, key=button_key, use_container_width=True,
                 help="Opens an OS file/folder dialog."):
        initial = st.session_state.get(state_key, "") or ""
        if initial:
            # If the current value is an existing file, start the dialog in
            # its parent dir; if it's an existing dir, start there directly.
            try:
                ip = Path(initial)
                initial = str(ip if ip.is_dir() else (ip.parent if ip.exists() else ""))
            except Exception:
                initial = ""
        chosen = _pick_path_via_dialog(kind=kind, title=dialog_title,
                                        initial_dir=initial)
        if chosen:
            st.session_state[state_key] = chosen
            st.rerun()


# ---------------------------------------------------------------------------
# JSONL helpers (Dataset tab)
# ---------------------------------------------------------------------------
def jsonl_to_rows(path: Path) -> list[dict]:
    """Read a JSONL training file into a list of editable row dicts."""
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            msgs = rec.get("messages") or []
            if len(msgs) < 2:
                continue
            rows.append({
                "keep": True,
                "question": str(msgs[0].get("content", "")),
                "answer": str(msgs[1].get("content", "")),
                "source": str(rec.get("source") or ""),
            })
    return rows


def rows_to_jsonl(rows, out_path: Path) -> int:
    """Write rows (iterable of dicts with question/answer/keep) to a JSONL file.
    Returns the number of pairs actually written.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for r in rows:
            if not r.get("keep", True):
                continue
            q = (r.get("question") or "").strip()
            a = (r.get("answer") or "").strip()
            if not q or not a:
                continue
            fh.write(json.dumps({
                "messages": [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": a},
                ],
                "source": r.get("source") or "manual",
            }, ensure_ascii=False) + "\n")
            written += 1
    return written


def validate_uploaded_jsonl(text: str) -> tuple[int, int, list[str]]:
    """Quick sanity check on uploaded JSONL. Returns (valid_lines, total_lines, errors)."""
    errors: list[str] = []
    valid = 0
    total = 0
    for i, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        total += 1
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"line {i}: not valid JSON ({exc.msg})")
            continue
        msgs = rec.get("messages")
        if not isinstance(msgs, list) or len(msgs) < 2:
            errors.append(f"line {i}: missing 'messages' array with ≥ 2 entries")
            continue
        if not (msgs[0].get("content") and msgs[1].get("content")):
            errors.append(f"line {i}: empty user or assistant content")
            continue
        valid += 1
    return valid, total, errors


def _reset_defaults() -> None:
    for k, v in get_default_config().items():
        st.session_state[k] = v
    st.toast("Reset all options to recommended defaults.", icon=":material/refresh:")


_seed_state()


# ---------------------------------------------------------------------------
# Sidebar — source model + system status
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title(":sparkles: Fine-Tune Studio")
    st.caption("Fine-tune any Ollama model on your documents (LoRA → GGUF → Ollama).")

    try:
        models = ollama_client.list_models()
        model_names = [m["name"] for m in models]
    except Exception as exc:
        models = []
        model_names = []
        st.error(f"Ollama unreachable: {exc}")

    st.subheader("Source model")
    if model_names:
        if st.session_state["ollama_model"] not in model_names:
            st.session_state["ollama_model"] = model_names[0]
        st.selectbox(
            "Ollama model to fine-tune",
            options=model_names,
            key="ollama_model",
            help="Pulled via `ollama pull`. We map it to a HuggingFace base for LoRA training.",
        )
    else:
        st.warning("No Ollama models found. Run `ollama pull <model>` first.")

    selected = st.session_state["ollama_model"]
    hf = hf_repo_for(selected)
    if hf:
        st.success(f"HF base: `{hf}`")
    else:
        st.error(
            f"No HF mapping for `{selected}`. Add an entry to OLLAMA_TO_HF in config.py."
        )
    if selected not in CPU_FRIENDLY:
        st.warning(
            "This model is **not CPU-friendly**. Without a CUDA GPU, training will be "
            "very slow or run out of memory. Recommended on CPU: "
            f"{', '.join(sorted(CPU_FRIENDLY))}."
        )

    st.divider()
    st.button(":material/refresh: Reset all to defaults", on_click=_reset_defaults,
              use_container_width=True)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_cfg, tab_data, tab_run, tab_import, tab_eval, tab_chat = st.tabs([
    ":material/tune: Configure",
    ":material/dataset: Dataset",
    ":material/play_arrow: Fine-tune",
    ":material/folder_open: Import adapter",
    ":material/grading: Evaluate",
    ":material/chat: Chat & validate",
])

# =========================================================================
# Tab 1 — Configure
# =========================================================================
with tab_cfg:
    st.header("1. Upload training documents")
    uploaded = st.file_uploader(
        "Drop PDF, DOCX, TXT, MD, CSV, JSON, PY files",
        type=["pdf", "docx", "txt", "md", "markdown", "rst", "csv", "json", "py"],
        accept_multiple_files=True,
    )
    if uploaded:
        for f in uploaded:
            dest = UPLOADS_DIR / f.name
            dest.write_bytes(f.getbuffer())
        st.success(f"Saved {len(uploaded)} file(s) to {UPLOADS_DIR}")

    existing = sorted(UPLOADS_DIR.glob("*"))
    if existing:
        with st.expander(f"Documents in uploads ({len(existing)})", expanded=False):
            for p in existing:
                cols = st.columns([0.8, 0.2])
                cols[0].write(f"`{p.name}` — {p.stat().st_size:,} bytes")
                if cols[1].button("Delete", key=f"del_{p.name}"):
                    p.unlink(missing_ok=True)
                    st.rerun()

    st.header("2. Tuning options")
    st.caption("All fields are pre-filled with recommended defaults. Use the sidebar to reset.")

    with st.expander(":material/dataset: Dataset generation", expanded=True):
        c1, c2, c3 = st.columns(3)
        c1.number_input("Q&A pairs per chunk", min_value=1, max_value=20,
                        step=1, key="qa_pairs_per_chunk")
        c2.number_input("Max chunks (cap for CPU runs)", min_value=1, max_value=2000,
                        step=1, key="max_chunks")
        c3.slider("Generation temperature", 0.0, 1.5, step=0.05,
                  key="generation_temperature")
        c4, c5 = st.columns(2)
        c4.number_input("Chunk size (characters)", min_value=200, max_value=4000,
                        step=50, key="chunk_size")
        c5.number_input("Chunk overlap", min_value=0, max_value=1000,
                        step=10, key="chunk_overlap")

    with st.expander(":material/extension: LoRA adapter", expanded=True):
        c1, c2, c3 = st.columns(3)
        c1.number_input("LoRA rank (r)", min_value=1, max_value=256, step=1, key="lora_r")
        c2.number_input("LoRA alpha", min_value=1, max_value=512, step=1, key="lora_alpha")
        c3.slider("LoRA dropout", 0.0, 0.5, step=0.01, key="lora_dropout")
        st.text_input(
            "Target modules (comma-separated, or 'auto')",
            key="lora_target_modules",
            help="e.g. q_proj,k_proj,v_proj,o_proj — leave 'auto' to let PEFT pick.",
        )

    with st.expander(":material/school: Training loop", expanded=True):
        c1, c2, c3 = st.columns(3)
        c1.number_input("Epochs", min_value=1, max_value=50, step=1, key="epochs")
        c2.number_input("Learning rate", min_value=1e-6, max_value=1e-2,
                        step=1e-5, format="%.6f", key="learning_rate")
        c3.number_input("Per-device batch size", min_value=1, max_value=64,
                        step=1, key="per_device_batch_size")
        c4, c5, c6 = st.columns(3)
        c4.number_input("Grad accumulation steps", min_value=1, max_value=64,
                        step=1, key="gradient_accumulation_steps")
        c5.number_input("Max sequence length", min_value=64, max_value=8192,
                        step=64, key="max_seq_length")
        c6.number_input("Random seed", min_value=0, max_value=2**31 - 1,
                        step=1, key="seed")
        c7, c8, c9 = st.columns(3)
        c7.slider("Warmup ratio", 0.0, 0.5, step=0.01, key="warmup_ratio")
        c8.slider("Weight decay", 0.0, 0.3, step=0.01, key="weight_decay")
        c9.selectbox("LR scheduler",
                     ["cosine", "linear", "constant", "constant_with_warmup"],
                     key="lr_scheduler_type")
        c10, c11, c12 = st.columns(3)
        c10.selectbox("Optimizer",
                      ["adamw_torch", "adamw_torch_fused", "adafactor", "sgd"],
                      key="optimizer")
        c11.checkbox("Use 4-bit (QLoRA, GPU only)", key="use_4bit")
        c12.checkbox("fp16 (GPU)", key="fp16")
        st.checkbox("bf16 (newer GPU)", key="bf16")

    with st.expander(":material/swap_horiz: Conversion & registration", expanded=True):
        c1, c2 = st.columns(2)
        c1.selectbox("GGUF quantization",
                     ["q4_k_m", "q5_k_m", "q8_0", "f16"],
                     key="gguf_quant",
                     help="q4_k_m gives smallest files; q8_0/f16 keep more quality.")
        c2.text_input("New model name suffix", key="new_model_suffix")

    with st.expander(":material/smart_toy: Inference defaults", expanded=False):
        c1, c2, c3 = st.columns(3)
        c1.slider("Chat temperature", 0.0, 1.5, step=0.05, key="chat_temperature")
        c2.slider("Chat top-p", 0.0, 1.0, step=0.01, key="chat_top_p")
        c3.number_input("Max new tokens", min_value=32, max_value=8192, step=32,
                        key="chat_num_predict")
        st.text_area("System prompt", key="system_prompt", height=100)


# =========================================================================
# Tab 2 — Dataset (generate / upload / review / approve)
# =========================================================================
with tab_data:
    st.header("Build & review the training dataset")
    st.caption(
        "Combine any mix of generated, uploaded, and existing JSONL files into a "
        "single training dataset. Review every pair, then approve it for fine-tuning."
    )

    # ---- Approval banner ----------------------------------------------------
    approved = st.session_state.get("approved_dataset")
    if approved and Path(approved).exists():
        approved_path = Path(approved)
        with approved_path.open("r", encoding="utf-8") as fh:
            approved_count = sum(1 for line in fh if line.strip())
        st.success(
            f"**Approved for training:** `{approved_path.name}` ({approved_count} pairs). "
            f"Go to the **Fine-tune** tab to start."
        )
    else:
        st.info("No dataset approved yet. Add sources, merge, review, then approve.")

    st.divider()

    # ============ 1. Add sources ============================================
    st.subheader("1. Add new dataset files (optional)")
    st.caption(
        "Both sections are independent — generate or upload as many times as you "
        "want; every saved file becomes available below."
    )

    add_gen, add_up = st.columns(2)

    # --- Generate ---
    with add_gen:
        with st.expander(":material/auto_awesome: Generate from documents",
                         expanded=False):
            docs_now = sorted(UPLOADS_DIR.glob("*"))

            # The model that GENERATES the Q&A pairs can be different from the
            # one we're fine-tuning. Small models produce broken JSON; use a
            # stronger one here for clean output.
            try:
                live_model_names = [m["name"] for m in ollama_client.list_models()]
            except Exception:
                live_model_names = []

            if live_model_names:
                if st.session_state.get("generation_model") not in live_model_names:
                    st.session_state["generation_model"] = (
                        st.session_state.get("ollama_model")
                        if st.session_state.get("ollama_model") in live_model_names
                        else live_model_names[0]
                    )
                st.selectbox(
                    "Generation model",
                    options=live_model_names,
                    key="generation_model",
                    help=(
                        "Used only for producing Q&A pairs from your documents. "
                        "Pick a STRONG model (e.g. qwen3:14b, mistral-nemo:12b) — "
                        "tiny models often emit broken JSON and yield 0 pairs. "
                        "The fine-tune target is the smaller model in the sidebar."
                    ),
                )
            else:
                st.error("No Ollama models found.")

            m_model = st.session_state.get("generation_model") or \
                      st.session_state["ollama_model"]
            target_model = st.session_state["ollama_model"]
            if m_model == target_model and target_model in CPU_FRIENDLY:
                st.warning(
                    f"Generation model is the same tiny model you're fine-tuning "
                    f"(`{target_model}`). Expect frequent 0-pair chunks. "
                    "Pick a stronger model above."
                )

            mc = st.columns(2)
            mc[0].metric("Docs", len(docs_now))
            mc[1].metric("Pairs/chunk", st.session_state["qa_pairs_per_chunk"])
            if not docs_now:
                st.warning("Upload documents on the Configure tab first.")
            gen_clicked = st.button(
                ":material/auto_awesome: Generate now",
                disabled=not docs_now, type="primary",
                key="gen_btn", use_container_width=True,
            )

            gen_log_box = st.empty()
            gen_prog = st.progress(0)

            def _push_dataset_log(msg: str) -> None:
                st.session_state["dataset_log"].append(msg)
                gen_log_box.code(
                    "\n".join(st.session_state["dataset_log"][-200:]),
                    language="text",
                )

            if st.session_state.get("dataset_log"):
                gen_log_box.code(
                    "\n".join(st.session_state["dataset_log"][-200:]),
                    language="text",
                )

            if gen_clicked:
                st.session_state["dataset_log"] = []
                try:
                    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
                    gen_id = f"{m_model.replace(':', '_').replace('/', '_')}-{ts}"
                    _push_dataset_log("Loading & chunking documents...")
                    chunks = document_loader.load_and_chunk(
                        docs_now,
                        int(st.session_state["chunk_size"]),
                        int(st.session_state["chunk_overlap"]),
                    )
                    chunks = [c for c in chunks if c["chunk_index"] >= 0][
                        : int(st.session_state["max_chunks"])
                    ]
                    _push_dataset_log(
                        f"Got {len(chunks)} chunks from {len(docs_now)} doc(s).")
                    gen_prog.progress(0.15)
                    if not chunks:
                        st.error("No usable text extracted from documents.")
                        st.stop()
                    ds_path = DATASETS_DIR / f"{gen_id}.jsonl"

                    def _ds_cb(i: int, total: int, msg: str) -> None:
                        gen_prog.progress(0.15 + 0.80 * (i / total))
                        if i % 2 == 0 or i == total:
                            _push_dataset_log(f"  [{i}/{total}] {msg}")

                    ds_path, n_pairs = dataset_builder.build_dataset(
                        chunks,
                        ollama_model=m_model,
                        pairs_per_chunk=int(st.session_state["qa_pairs_per_chunk"]),
                        temperature=float(st.session_state["generation_temperature"]),
                        out_path=ds_path,
                        progress_cb=_ds_cb,
                    )
                    gen_prog.progress(1.0)
                    _push_dataset_log(f"Wrote {n_pairs} pairs → {ds_path.name}")
                    pool = list(st.session_state.get("dataset_pool") or [])
                    if str(ds_path) not in pool:
                        pool.append(str(ds_path))
                        st.session_state["dataset_pool"] = pool
                        st.session_state["ms_version"] += 1
                    st.toast(f"Added `{ds_path.name}` ({n_pairs} pairs)",
                             icon=":material/check_circle:")
                except Exception as exc:
                    _push_dataset_log(f"ERROR: {exc}\n{traceback.format_exc()}")
                    st.error(f"Generation failed: {exc}")

    # --- Upload ---
    with add_up:
        with st.expander(":material/upload_file: Upload my own JSONL",
                         expanded=False):
            st.caption(
                'One JSON object per line with a `messages` array, e.g. '
                '`{"messages":[{"role":"user","content":"..."},'
                '{"role":"assistant","content":"..."}]}`'
            )
            uploaded_ds = st.file_uploader(
                "Upload .jsonl",
                type=["jsonl", "json"],
                accept_multiple_files=False,
                key="jsonl_uploader",
            )
            if uploaded_ds is not None:
                raw = uploaded_ds.getvalue().decode("utf-8", errors="replace")
                valid, total, errs = validate_uploaded_jsonl(raw)
                c1, c2 = st.columns(2)
                c1.metric("Valid pairs", valid)
                c2.metric("Total lines", total)
                if errs:
                    with st.expander(f":material/warning: {len(errs)} issues",
                                     expanded=False):
                        for e in errs[:50]:
                            st.caption(e)
                        if len(errs) > 50:
                            st.caption(f"... and {len(errs)-50} more.")
                if valid > 0 and st.button(
                    ":material/save: Save & add to pool",
                    type="primary", key="save_uploaded_jsonl",
                    use_container_width=True,
                ):
                    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
                    stem = Path(uploaded_ds.name).stem
                    save_path = DATASETS_DIR / f"{stem}-uploaded-{ts}.jsonl"
                    save_path.write_text(raw, encoding="utf-8")
                    pool = list(st.session_state.get("dataset_pool") or [])
                    if str(save_path) not in pool:
                        pool.append(str(save_path))
                        st.session_state["dataset_pool"] = pool
                        st.session_state["ms_version"] += 1
                    st.toast(f"Saved & added `{save_path.name}`",
                             icon=":material/check_circle:")
                    st.rerun()

    st.divider()

    # ============ 2. Pick files to combine ==================================
    st.subheader("2. Choose JSONL files to combine")

    all_jsonl = sorted(
        DATASETS_DIR.glob("*.jsonl"),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    if not all_jsonl:
        st.info("No JSONL files yet. Generate or upload one above.")
    else:
        counts: dict[str, int] = {}
        for p in all_jsonl:
            try:
                with p.open("r", encoding="utf-8") as fh:
                    counts[str(p)] = sum(1 for line in fh if line.strip())
            except OSError:
                counts[str(p)] = 0

        def _fmt(path_str: str) -> str:
            return f"{Path(path_str).name}  ·  {counts.get(path_str, 0)} pairs"

        options = [str(p) for p in all_jsonl]
        # Pool is our source of truth; drop entries whose files no longer exist.
        pool = [s for s in (st.session_state.get("dataset_pool") or [])
                if s in options]
        st.session_state["dataset_pool"] = pool

        picked = st.multiselect(
            "Select one or more files to merge into the training pool",
            options=options,
            default=pool,
            format_func=_fmt,
            key=f"ms_files_v{st.session_state['ms_version']}",
        )
        # Sync user's manual changes back into the pool (no widget owns this key).
        if list(picked) != pool:
            st.session_state["dataset_pool"] = list(picked)
        total_pairs = sum(counts.get(p, 0) for p in picked)
        st.caption(
            f"Selected: **{len(picked)} file(s)** → **{total_pairs} pairs** before edits."
        )

        bc1, bc2 = st.columns([0.4, 0.6])
        merge_clicked = bc1.button(
            ":material/merge: Merge & review →",
            disabled=not picked,
            type="primary",
            use_container_width=True,
        )
        # Quick download for any selected file
        if picked:
            with bc2.popover(":material/download: Download a selected file",
                             use_container_width=True):
                for p_str in picked:
                    p = Path(p_str)
                    try:
                        st.download_button(
                            p.name, data=p.read_bytes(), file_name=p.name,
                            mime="application/jsonl",
                            key=f"dl_pool_{p.name}",
                            use_container_width=True,
                        )
                    except OSError:
                        st.caption(f"(missing) {p.name}")

        if merge_clicked:
            st.session_state["merge_armed"] = True
            for k in list(st.session_state.keys()):
                if isinstance(k, str) and k.startswith("editor_merged::"):
                    del st.session_state[k]
            st.rerun()

    # ============ 3. Review merged dataset ==================================
    if st.session_state.get("merge_armed"):
        picked = list(st.session_state.get("dataset_pool") or [])
        paths = [Path(s) for s in picked if Path(s).exists()]
        if not paths:
            st.session_state["merge_armed"] = False
        else:
            merged_rows: list[dict] = []
            for p in paths:
                for r in jsonl_to_rows(p):
                    r = dict(r)
                    r["source"] = r.get("source") or p.name
                    merged_rows.append(r)

            st.divider()
            st.subheader(
                f"3. Review merged dataset — "
                f"{len(merged_rows)} pairs from {len(paths)} file(s)"
            )
            if not merged_rows:
                st.warning("Selected files have no valid pairs.")
            else:
                df = pd.DataFrame(merged_rows)
                editor_key = "editor_merged::" + "|".join(sorted(picked))
                edited = st.data_editor(
                    df,
                    key=editor_key,
                    column_config={
                        "keep": st.column_config.CheckboxColumn(
                            "keep", default=True, width="small",
                            help="Uncheck to drop this row.",
                        ),
                        "question": st.column_config.TextColumn(
                            "question", width="large"),
                        "answer": st.column_config.TextColumn(
                            "answer", width="large"),
                        "source": st.column_config.TextColumn(
                            "source", width="medium",
                            help="Original file this row came from."),
                    },
                    num_rows="dynamic",
                    hide_index=True,
                    height=460,
                    use_container_width=True,
                )

                kept_now = int(edited["keep"].sum()) if "keep" in edited else len(edited)
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("Total rows", len(edited))
                mc2.metric("Marked keep", kept_now)
                mc3.metric("Dropped", len(edited) - kept_now)

                b1, b2 = st.columns(2)
                save_clicked = b1.button(
                    ":material/save: Save merged as new JSONL",
                    use_container_width=True,
                )
                approve_clicked = b2.button(
                    ":material/verified: Approve & use for training",
                    type="primary", use_container_width=True,
                    disabled=kept_now == 0,
                )

                if save_clicked or approve_clicked:
                    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
                    merged_path = DATASETS_DIR / f"merged-{ts}.jsonl"
                    n = rows_to_jsonl(edited.to_dict("records"), merged_path)
                    if approve_clicked:
                        st.session_state["approved_dataset"] = str(merged_path)
                        st.toast(
                            f"Approved `{merged_path.name}` ({n} pairs). "
                            "Switch to Fine-tune.",
                            icon=":material/verified:",
                        )
                    else:
                        st.toast(f"Saved {n} pairs → {merged_path.name}",
                                 icon=":material/check_circle:")
                    pool = list(st.session_state.get("dataset_pool") or [])
                    if str(merged_path) not in pool:
                        pool.append(str(merged_path))
                        st.session_state["dataset_pool"] = pool
                        st.session_state["ms_version"] += 1
                    st.session_state["merge_armed"] = False
                    st.rerun()


# =========================================================================
# Tab 3 — Fine-tune
# =========================================================================
with tab_run:
    st.header("Fine-tune pipeline")
    st.caption(
        "Train a LoRA adapter on the approved dataset, merge it into the base model, "
        "convert to GGUF, and register the result with Ollama."
    )

    selected_model = st.session_state["ollama_model"]
    hf = hf_repo_for(selected_model)
    approved = st.session_state.get("approved_dataset")
    approved_path = Path(approved) if approved else None
    approved_count = 0
    if approved_path and approved_path.exists():
        try:
            with approved_path.open("r", encoding="utf-8") as fh:
                approved_count = sum(1 for line in fh if line.strip())
        except OSError:
            approved_count = 0

    # --------- Header summary ---------
    cols = st.columns(4)
    cols[0].metric("Source model", selected_model)
    cols[1].metric("HF base", (hf or "—").split("/")[-1])
    cols[2].metric("Approved dataset",
                   approved_path.name if approved_path else "—")
    cols[3].metric("Training pairs", approved_count or "—")

    if not approved_path or not approved_path.exists():
        st.warning(
            "No approved dataset. Go to the **Dataset** tab to generate, upload, "
            "review and approve a JSONL file before training."
        )
    if selected_model and not hf:
        st.error("No HF mapping for the selected Ollama model. Pick another or extend config.py.")

    ready = bool(approved_path and approved_path.exists()) and bool(hf) and bool(selected_model)
    run = st.button(":material/rocket_launch: Start fine-tuning",
                    disabled=not ready, type="primary")

    st.markdown("**Training loss curve**")
    chart_box = st.empty()
    log_box = st.empty()
    progress = st.progress(0)
    status_line = st.empty()

    def _render_chart(metrics: list[dict]) -> None:
        if not metrics:
            chart_box.info("Loss curve will appear here once training starts.")
            return
        try:
            df = pd.DataFrame(metrics)
            if "step" not in df.columns or "loss" not in df.columns:
                return
            df = df.set_index("step")
            chart_box.line_chart(df[["loss"]], height=240)
        except Exception:
            pass

    _render_chart(st.session_state.get("training_metrics", []))
    if st.session_state.get("training_log"):
        log_box.code(
            "\n".join(st.session_state["training_log"][-200:]),
            language="text",
        )

    def _push_log(msg: str) -> None:
        st.session_state["training_log"].append(msg)
        log_box.code("\n".join(st.session_state["training_log"][-200:]), language="text")

    def _push_metric(m: dict) -> None:
        st.session_state["training_metrics"].append(m)
        _render_chart(st.session_state["training_metrics"])

    if run:
        st.session_state["training_log"] = []
        st.session_state["training_metrics"] = []
        _render_chart([])
        try:
            cfg = {k: st.session_state[k] for k in CONFIG_KEYS}
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            run_id = f"{selected_model.replace(':', '_').replace('/', '_')}-{ts}"
            ds_path = approved_path
            _push_log(f"Using approved dataset: {ds_path.name} ({approved_count} pairs)")

            # --- 1. LoRA fine-tune
            status_line.info("Step 1/3 — LoRA fine-tuning (slowest step on CPU)...")
            adapter_root = CHECKPOINTS_DIR / run_id
            result = trainer.train_lora(
                dataset_path=ds_path,
                base_model_repo=hf,
                output_dir=adapter_root,
                cfg=cfg,
                log_cb=_push_log,
                metrics_cb=_push_metric,
            )
            progress.progress(0.55)
            _push_log(f"Adapter: {result.adapter_dir} (loss={result.final_loss})")

            # --- 2. Merge & convert to GGUF
            status_line.info("Step 2/3 — Merging adapter and converting to GGUF...")
            merged_dir = MERGED_DIR / run_id
            converter.merge_lora(result.adapter_dir, hf, merged_dir, log_cb=_push_log)
            progress.progress(0.80)

            requested_gguf = GGUF_DIR / f"{run_id}.{cfg['gguf_quant']}.gguf"
            gguf_path = converter.convert_to_gguf(
                merged_dir, requested_gguf,
                quant=str(cfg["gguf_quant"]),
                log_cb=_push_log,
            )
            if gguf_path != requested_gguf:
                _push_log(f"Note: GGUF written to {gguf_path.name} (quant fallback).")
            progress.progress(0.92)

            # --- 3. Register with Ollama
            status_line.info("Step 3/3 — Registering with Ollama...")
            new_name = (
                f"{selected_model.replace(':', '-')}-{cfg['new_model_suffix']}-{ts}"
            )
            modelfile = converter.write_modelfile(
                gguf_path,
                system_prompt=str(cfg["system_prompt"]),
                temperature=float(cfg["chat_temperature"]),
                top_p=float(cfg["chat_top_p"]),
            )
            converter.register_with_ollama(new_name, modelfile, log_cb=_push_log)
            progress.progress(1.0)

            st.session_state["last_finetuned_model"] = new_name
            status_line.success(f"Done. New model: `{new_name}`. Switch to the Chat tab.")
            st.balloons()

        except Exception as exc:
            tb = traceback.format_exc()
            _push_log(f"\nERROR: {exc}\n{tb}")
            status_line.error(f"Pipeline failed: {exc}")


# =========================================================================
# Tab 4 — Import adapter (bring your own fine-tuned folder)
# =========================================================================
with tab_import:
    st.header("Import an externally fine-tuned model into Ollama")
    st.caption(
        "Point at a LoRA adapter folder (e.g. from Colab) or an already-merged "
        "HF model folder. The app will merge (if needed), convert to GGUF, and "
        "register the result with Ollama."
    )

    st.session_state.setdefault("import_path", "")
    st.session_state.setdefault("import_inspection", None)
    st.session_state.setdefault("import_template_detection", None)
    st.session_state.setdefault("import_eval_result", None)
    st.session_state.setdefault("import_eval_log", [])
    st.session_state.setdefault("import_log", [])

    fld_cols = st.columns([0.78, 0.22])
    with fld_cols[0]:
        st.text_input(
            "Path to fine-tuned folder",
            key="import_path",
            placeholder=r"C:\Users\you\Downloads\my_fine_tuned_model",
            help="Folder containing adapter_config.json (LoRA) or config.json + "
                 "model weights (merged HF model). Use Browse to pick instead.",
        )
    with fld_cols[1]:
        # Spacer aligns the button with the input (the input's label adds height).
        st.write(" ")
        _browse_button(
            ":material/folder_open: Browse folder…",
            state_key="import_path",
            kind="folder",
            dialog_title="Select the fine-tuned model folder",
            button_key="browse_import_folder",
        )

    insp_cols = st.columns([0.25, 0.75])
    if insp_cols[0].button(":material/search: Inspect path",
                           use_container_width=True):
        path_str = (st.session_state.get("import_path") or "").strip().strip('"')
        if not path_str:
            st.warning("Enter a folder path first.")
        else:
            info = converter.inspect_external_folder(Path(path_str))
            st.session_state["import_inspection"] = info
            # Sniff the model's jinja chat_template for the recommended format.
            try:
                st.session_state["import_template_detection"] = (
                    chat_templates.detect_with_candidates(Path(path_str))
                )
            except Exception as exc:
                st.session_state["import_template_detection"] = None
                st.warning(f"Could not detect chat template: {exc}")
            # User picks again from scratch next time inspect is clicked.
            st.session_state.pop("import_chat_format_key", None)
            # A new folder = stale eval result; clear it.
            st.session_state["import_eval_result"] = None
            st.session_state["import_eval_log"] = []
            st.session_state["import_log"] = []
            st.rerun()

    info = st.session_state.get("import_inspection")
    if info:
        ic1, ic2, ic3 = st.columns(3)
        ic1.metric("Detected kind", info["kind"])
        ic2.metric("Has tokenizer", "yes" if info["has_tokenizer"] else "no")
        ic3.metric("Base model",
                   (info["base_model"] or "—").split("/")[-1])
        if info.get("notes"):
            with st.expander(":material/warning: Notes", expanded=False):
                for n in info["notes"]:
                    st.caption(n)

        st.subheader("Import options")

        mode_label_to_value = {
            "Auto-detect (recommended)": "unknown",
            "LoRA adapter (will merge with base model)": "adapter",
            "Merged HF model (skip merge — already has config.json + weights)":
                "merged",
        }
        mode_label = st.selectbox(
            "Treat folder as",
            options=list(mode_label_to_value.keys()),
            index=0,
            help=("Auto-detect picks 'adapter' if adapter_config.json is "
                  "present, otherwise 'merged'."),
            key="import_mode_label",
        )
        chosen_mode = mode_label_to_value[mode_label]

        # The base model is required for adapter mode; pre-fill with detected.
        effective_mode = chosen_mode
        if effective_mode == "unknown":
            effective_mode = info["kind"] if info["kind"] != "unknown" else "adapter"
        st.text_input(
            "Base HF model repo (LoRA adapter only)",
            value=info.get("base_model") or "",
            key="import_base_model",
            disabled=(effective_mode == "merged"),
            help="HuggingFace repo to load the original weights from for the "
                 "merge step. Ignored when the folder is already merged.",
        )

        st.text_input(
            "New Ollama model name",
            value=converter.suggest_ollama_name(
                Path(st.session_state["import_path"].strip().strip('"')),
                info.get("base_model"),
            ),
            key="import_ollama_name",
            help="Final tag in `ollama list`. Allowed chars: a-z, 0-9, . _ -",
        )

        oc1, oc2 = st.columns(2)
        oc1.selectbox(
            "GGUF quantization",
            ["q4_k_m", "q5_k_m", "q8_0", "f16"],
            key="import_quant",
            help="q4_k_m gives the smallest file. Falls back to q8_0 if "
                 "llama-quantize isn't built.",
        )
        oc2.slider("Chat temperature", 0.0, 1.5, value=0.3, step=0.05,
                   key="import_temperature")
        st.slider("Chat top-p", 0.0, 1.0, value=0.9, step=0.01,
                  key="import_top_p")
        st.text_area(
            "System prompt",
            value=st.session_state.get(
                "system_prompt",
                get_default_config()["system_prompt"],
            ),
            key="import_system_prompt",
            height=100,
        )

        # -------------------------------------------------------------------
        # Chat template picker — auto-detected from the model's own jinja,
        # with a dropdown override. Color code:
        #   green  = recommended for this model
        #   orange = user has picked a different one
        #   red    = not applicable to this model family
        # -------------------------------------------------------------------
        detection = st.session_state.get("import_template_detection")
        if detection and detection.get("candidates"):
            st.subheader("Chat template")

            rec_key = detection["recommended"]
            sel_key = st.session_state.get("import_chat_format_key", rec_key)

            chip_html_parts: list[str] = []
            for k, c in detection["candidates"].items():
                status = c["status"]
                if k == sel_key and sel_key != rec_key:
                    bg, label = "#ea580c", f"{k} (your pick)"          # orange
                elif status == "recommended":
                    bg, label = "#16a34a", f"{k} (recommended)"         # green
                elif status == "not_applicable":
                    bg, label = "#dc2626", f"{k} (n/a)"                 # red
                else:
                    bg, label = "#6b7280", k                             # grey
                chip_html_parts.append(
                    '<span style="background:{bg};color:white;padding:3px 10px;'
                    'border-radius:12px;margin:0 6px 6px 0;font-size:0.85em;'
                    'display:inline-block">{lbl}</span>'.format(bg=bg, lbl=label)
                )
            st.markdown("".join(chip_html_parts), unsafe_allow_html=True)

            if detection.get("chat_template_found"):
                st.caption(
                    f"Detected `{rec_key}` via {detection['source']}. "
                    "Green = recommended, orange = your override, red = "
                    "doesn't fit this model family."
                )
            else:
                st.caption(
                    "No jinja chat_template found in folder — falling back to "
                    f"`{rec_key}`. Override below if you know the right format."
                )

            format_keys = list(detection["candidates"].keys())

            def _fmt_option(k: str) -> str:
                c = detection["candidates"][k]
                icon = {
                    "recommended": "🟢",
                    "compatible": "⚪",
                    "not_applicable": "🔴",
                }[c["status"]]
                return f"{icon} {k} — {c['examples']}"

            st.selectbox(
                "Chat template to write into the Modelfile",
                options=format_keys,
                index=format_keys.index(rec_key),
                format_func=_fmt_option,
                key="import_chat_format_key",
                help=(
                    "Green ● = best match for this model (based on its own "
                    "jinja chat_template). Red ● = not applicable to this "
                    "model family — pick only if you really know what you're "
                    "doing."
                ),
            )

            sel_key = st.session_state["import_chat_format_key"]
            sel_info = detection["candidates"][sel_key]
            with st.expander(
                f"Preview: {sel_key} template + stop tokens", expanded=False
            ):
                st.caption(sel_info["reason"])
                st.caption(f"Stop tokens: {sel_info['stop']}")
                st.code(sel_info["template"], language="jinja2")
            if sel_key != rec_key:
                st.warning(
                    f"You've overridden the recommended template "
                    f"(`{rec_key}`). The Modelfile will be written with "
                    f"`{sel_key}` instead."
                )

        # -------------------------------------------------------------------
        # Optional accuracy gate: load adapter (or merged) + run deterministic
        # generation on N val rows, report exact-match. The verdict card has a
        # CSS hover tooltip with a colored breakdown so you can decide whether
        # to proceed with GGUF conversion or go back to fine-tuning.
        # -------------------------------------------------------------------
        with st.expander(
            ":material/science: Check accuracy before converting "
            "(optional, recommended)",
            expanded=False,
        ):
            st.caption(
                "Loads the adapter (or merged model) into transformers, runs "
                "deterministic greedy generation on N validation rows, and "
                "scores exact-match against the gold assistant message. Use "
                "this to gate GGUF conversion — if accuracy is low, train "
                "more instead of converting a bad model."
            )

            # Rank candidate val JSONLs. Different models (Qwen vs TinyLlama)
            # need different val files — picking alphabetically would be wrong.
            APP_DIR = Path(__file__).parent
            _adapter_path = Path(
                st.session_state["import_path"].strip().strip('"')
            )
            _candidates = eval_adapter.find_val_candidates(
                _adapter_path,
                base_model_repo=(
                    st.session_state.get("import_base_model")
                    or info.get("base_model")
                ),
                project_root=APP_DIR,
            )

            if _candidates:
                st.caption(
                    f"**Base model:** `"
                    f"{(info.get('base_model') or 'unknown')}` &nbsp;·&nbsp; "
                    f"showing {len(_candidates)} val-file candidate(s), "
                    "best match first."
                )

                def _fmt_candidate(idx: int) -> str:
                    c = _candidates[idx]
                    badges = []
                    if c["looks_like_val"]:
                        badges.append("🟢 val/test")
                    if c["model_match"]:
                        badges.append("🎯 matches model")
                    if c["source"] == "in-folder":
                        badges.append("📦 in adapter folder")
                    elif c["source"] == "sibling":
                        badges.append("📂 next to adapter")
                    elif c["source"] == "project-eval":
                        badges.append("🧪 project eval set")
                    name = Path(c["path"]).name
                    return f"{' '.join(badges)}  {name}" if badges else name

                idx_options = list(range(len(_candidates))) + [-1]

                def _fmt_all(i: int) -> str:
                    return "✏️  paste a custom path…" if i == -1 else _fmt_candidate(i)

                pick = st.selectbox(
                    "Validation JSONL",
                    options=idx_options,
                    format_func=_fmt_all,
                    index=0,
                    key="import_val_pick",
                    help="Ranked best-first. Files inside the adapter folder "
                         "or named like a val/test split rank highest. "
                         "Filenames matching the base model family (qwen / "
                         "tinyllama / llama…) get a 🎯 boost. Pick "
                         '"paste a custom path" if your file is somewhere else.',
                )

                if pick == -1:
                    val_cols = st.columns([0.75, 0.25])
                    with val_cols[0]:
                        st.text_input(
                            "Validation JSONL path",
                            value=st.session_state.get("import_val_jsonl", ""),
                            key="import_val_jsonl",
                            help='Row format: {"messages": [{"role":"system",...},'
                                 ' {"role":"user",...}, {"role":"assistant",...}]}. '
                                 "Use Browse to pick from disk.",
                        )
                    with val_cols[1]:
                        st.write(" ")
                        _browse_button(
                            ":material/upload_file: Browse file…",
                            state_key="import_val_jsonl",
                            kind="file",
                            dialog_title="Select validation JSONL",
                            button_key="browse_import_val_custom",
                        )
                    # Free-text path — we can't infer if it's train or val.
                    # Ask the user to confirm before running.
                    st.radio(
                        "Is this file a held-out validation split, or the "
                        "training set the model already saw?",
                        options=["held-out validation",
                                 "training set (memorisation check)"],
                        index=0,
                        key="import_val_kind",
                        horizontal=True,
                        help="Picking the training set is fine for a quick "
                             "smoke test — the score tells you if the model "
                             "memorised anything — but it OVERSTATES real "
                             "accuracy. Use a held-out split for an honest "
                             "verdict.",
                    )
                else:
                    chosen = _candidates[pick]
                    # Push the resolved path into session_state so the runner
                    # below sees it without needing another widget.
                    st.session_state["import_val_jsonl"] = chosen["path"]
                    st.caption(f"_Why this file:_ {chosen['reason']}")
                    st.code(chosen["path"], language="text")

                    # Surface UPFRONT what kind of file this is, so the user
                    # interprets the resulting accuracy correctly.
                    if chosen["looks_like_val"]:
                        st.success(
                            ":material/verified: **Held-out validation file** "
                            "— the accuracy score will be a meaningful "
                            "real-world estimate."
                        )
                        st.session_state["import_val_kind"] = (
                            "held-out validation"
                        )
                    elif chosen["source"] in ("in-folder", "sibling"):
                        st.warning(
                            ":material/warning: **This looks like the "
                            "TRAINING file** for this adapter (it lives "
                            "inside / next to the adapter folder and its "
                            "name doesn't match a val/test pattern). "
                            "Accuracy will be inflated by memorisation — "
                            "useful as a smoke test (\"did the model learn "
                            "*anything*?\") but **not** an honest score. "
                            "For a real verdict, create a held-out split "
                            "(e.g. last 10–20% of rows saved as "
                            "`fp_val.jsonl`) and re-run."
                        )
                        st.session_state["import_val_kind"] = (
                            "training set (memorisation check)"
                        )
                    elif chosen["source"] == "project-eval":
                        st.info(
                            ":material/info: **Project eval-set file** "
                            "(under `data/eval/`). Treated as held-out by "
                            "convention, but make sure it wasn't included "
                            "in the training data for this adapter."
                        )
                        st.session_state["import_val_kind"] = (
                            "held-out validation"
                        )
                    else:
                        st.info(
                            ":material/info: This is a **project dataset "
                            "file** — possibly the training set, possibly "
                            "not. Confirm below:"
                        )
                        st.radio(
                            "Is this file a held-out validation split, or "
                            "the training set the model already saw?",
                            options=["held-out validation",
                                     "training set (memorisation check)"],
                            index=1,
                            key="import_val_kind",
                            horizontal=True,
                            help="Default: training set (the safer "
                                 "interpretation). Change if you're sure "
                                 "the model never saw these rows.",
                        )
            else:
                st.warning(
                    "No JSONL candidates found in the adapter folder, its "
                    "parent dir, or the project's `data/` tree. Use Browse "
                    "or paste an absolute path below."
                )
                val_cols2 = st.columns([0.75, 0.25])
                with val_cols2[0]:
                    st.text_input(
                        "Validation JSONL path",
                        value=st.session_state.get("import_val_jsonl", ""),
                        key="import_val_jsonl",
                        help='Row format: {"messages": [{"role":"system",...}, '
                             '{"role":"user",...}, {"role":"assistant",...}]}.',
                    )
                with val_cols2[1]:
                    st.write(" ")
                    _browse_button(
                        ":material/upload_file: Browse file…",
                        state_key="import_val_jsonl",
                        kind="file",
                        dialog_title="Select validation JSONL",
                        button_key="browse_import_val_fallback",
                    )

            cc1, cc2 = st.columns(2)
            cc1.number_input(
                "Samples to test", min_value=5, max_value=200,
                value=30, step=5, key="import_val_samples",
            )
            cc2.number_input(
                "Max new tokens per row", min_value=4, max_value=128,
                value=16, step=4, key="import_val_max_tokens",
                help="Short answers (yes/no, labels) → 16 is plenty. "
                     "Sentence answers → 64+.",
            )

            run_eval = st.button(
                ":material/play_arrow: Run accuracy check",
                use_container_width=True,
                key="run_import_eval",
            )

            eval_log_box = st.empty()
            if st.session_state["import_eval_log"]:
                eval_log_box.code(
                    "\n".join(st.session_state["import_eval_log"][-200:]),
                    language="text",
                )

            def _push_eval_log(msg: str) -> None:
                st.session_state["import_eval_log"].append(msg)
                eval_log_box.code(
                    "\n".join(st.session_state["import_eval_log"][-200:]),
                    language="text",
                )

            if run_eval:
                st.session_state["import_eval_log"] = []
                st.session_state["import_eval_result"] = None
                _eval_mode = (
                    chosen_mode if chosen_mode in ("adapter", "merged") else "auto"
                )
                try:
                    res = eval_adapter.evaluate_adapter_accuracy(
                        Path(
                            st.session_state["import_path"].strip().strip('"')
                        ),
                        base_model_repo=(
                            st.session_state.get("import_base_model") or None
                        ),
                        val_jsonl=Path(st.session_state["import_val_jsonl"]),
                        n_samples=int(st.session_state["import_val_samples"]),
                        max_new_tokens=int(
                            st.session_state["import_val_max_tokens"]
                        ),
                        mode=_eval_mode,
                        log_cb=_push_eval_log,
                    )
                    st.session_state["import_eval_result"] = res
                except Exception as exc:
                    _push_eval_log(
                        f"\nERROR: {exc}\n{traceback.format_exc()}"
                    )
                    st.error(f"Accuracy check failed: {exc}")

            res = st.session_state.get("import_eval_result")
            if res and res["total"] == 0:
                st.error(
                    ":material/error: Accuracy check produced 0 usable rows — "
                    "every line in the validation JSONL was skipped. "
                    "Each line must be: "
                    '`{"messages": [{"role":"system",...}, '
                    '{"role":"user",...}, {"role":"assistant",...}]}` '
                    "(system is optional; user + assistant are required). "
                    "Check the log above for the per-row reason."
                )
            elif res:
                acc = res["accuracy"]
                is_train = (
                    st.session_state.get("import_val_kind", "").startswith(
                        "training"
                    )
                )
                # Stricter thresholds on the training set: the model is
                # supposed to memorise it, so even 80% is unremarkable.
                if is_train:
                    if acc >= 0.95:
                        bg, fg, bd = "#dcfce7", "#166534", "#16a34a"
                        label = ":material/check_circle: Memorised — adapter learned the training set"
                        advice = ("≥95% on the training set means the LoRA "
                                  "capacity was sufficient. Real-world "
                                  "accuracy is unknown until you score "
                                  "against a held-out split.")
                    elif acc >= 0.7:
                        bg, fg, bd = "#fef3c7", "#92400e", "#d97706"
                        label = ":material/warning: Partial memorisation"
                        advice = ("70–94% on the *training* set is weak — "
                                  "the adapter only partly learned what it "
                                  "saw. Increase epochs or LoRA rank.")
                    else:
                        bg, fg, bd = "#fee2e2", "#991b1b", "#dc2626"
                        label = ":material/cancel: Adapter failed to learn"
                        advice = ("<70% on the *training* set means "
                                  "training didn't take. Check learning "
                                  "rate, target_modules, and that the "
                                  "right base model was attached.")
                else:
                    if acc >= 0.8:
                        bg, fg, bd = "#dcfce7", "#166534", "#16a34a"
                        label = ":material/check_circle: Good — safe to proceed"
                        advice = ("Accuracy is healthy. Click "
                                  "**Import & register with Ollama** below.")
                    elif acc >= 0.6:
                        bg, fg, bd = "#fef3c7", "#92400e", "#d97706"
                        label = ":material/warning: Borderline"
                        advice = ("60–79% is OK for hard or open-ended "
                                  "tasks. For classification, train more "
                                  "epochs / increase LoRA rank before "
                                  "converting.")
                    else:
                        bg, fg, bd = "#fee2e2", "#991b1b", "#dc2626"
                        label = ":material/cancel: Poor — go back and fine-tune more"
                        advice = ("The adapter didn't learn enough. "
                                  "Increase epochs, raise LoRA rank, or "
                                  "add more training data, then re-import.")

                wrong = res["total"] - res["correct"]
                # Hover tooltip: colored chips for correct / wrong / N, the
                # three threshold bands, then the advice.
                tooltip_html = (
                    '<div class="acc-tip">'
                    '<div class="acc-tip__row">'
                    '<span style="color:#86efac">'
                    ':material/check: Correct</span>'
                    f'<b>{res["correct"]}</b></div>'
                    '<div class="acc-tip__row">'
                    '<span style="color:#fca5a5">'
                    ':material/close: Wrong</span>'
                    f'<b>{wrong}</b></div>'
                    '<div class="acc-tip__row">'
                    '<span style="color:#93c5fd">N</span>'
                    f'<b>{res["total"]}</b></div>'
                    '<hr>'
                    '<div style="color:#86efac">≥ 80% &nbsp;→ ✅ Ship it</div>'
                    '<div style="color:#fde68a">60–79% → ⚠️ Borderline</div>'
                    '<div style="color:#fca5a5">&lt; 60%&nbsp; → ❌ Train more</div>'
                    '<hr>'
                    f'<div class="acc-tip__advice">{advice}</div>'
                    '</div>'
                )

                # Render: visible colored card + hover tooltip (pure CSS).
                # Material icon syntax (:material/...) isn't parsed inside raw
                # HTML — strip it down to text for the inline strings.
                _strip_icons = (
                    lambda s: s.replace(":material/check_circle: ", "✅ ")
                               .replace(":material/warning: ", "⚠️ ")
                               .replace(":material/cancel: ", "❌ ")
                               .replace(":material/check: ", "✓ ")
                               .replace(":material/close: ", "✗ ")
                )
                label_html = _strip_icons(label)
                tooltip_html_plain = _strip_icons(tooltip_html)

                st.markdown(
                    f"""
<style>
.acc-card{{
  position:relative;display:flex;align-items:center;gap:18px;
  background:{bg};color:{fg};border:2px solid {bd};
  border-radius:12px;padding:18px 20px;margin:14px 0 4px;cursor:help;
}}
.acc-card .acc-pct{{font-size:2.4rem;font-weight:800;line-height:1}}
.acc-card .acc-msg{{font-size:1.05rem;font-weight:700}}
.acc-card .acc-sub{{font-size:.86rem;opacity:.85;margin-top:3px}}
.acc-card .acc-tip{{
  display:none;position:absolute;top:calc(100% + 8px);left:0;z-index:30;
  background:#1e293b;color:#fff;padding:12px 14px;border-radius:10px;
  min-width:260px;font-size:.86rem;
  box-shadow:0 12px 24px rgba(15,23,42,.30);
}}
.acc-card .acc-tip__row{{display:flex;justify-content:space-between;gap:14px;padding:2px 0}}
.acc-card .acc-tip hr{{border:none;border-top:1px solid #475569;margin:8px 0}}
.acc-card .acc-tip__advice{{color:#e2e8f0;line-height:1.45}}
.acc-card:hover .acc-tip{{display:block}}
.acc-card:focus-within .acc-tip{{display:block}}
</style>
<div class="acc-card" tabindex="0">
  <div class="acc-pct">{acc:.0%}</div>
  <div>
    <div class="acc-msg">{label_html}</div>
    <div class="acc-sub">
      {res['correct']} / {res['total']} exact-match
      &nbsp;·&nbsp; <em>{st.session_state.get('import_val_kind', 'unknown set')}</em>
      &nbsp;·&nbsp; hover for the colored breakdown
    </div>
  </div>
  {tooltip_html_plain}
</div>
                    """,
                    unsafe_allow_html=True,
                )

                with st.expander(
                    f"Show all {res['total']} examples", expanded=False
                ):
                    if not res["examples"]:
                        st.warning(
                            "No examples were scored — every row in the "
                            "validation JSONL got skipped. Expected row "
                            'format: `{"messages": [{"role":"system",...}, '
                            '{"role":"user",...}, {"role":"assistant",...}]}`. '
                            "Check the log above for the per-row reason "
                            "(usually a missing field)."
                        )
                    else:
                        df = pd.DataFrame(res["examples"])
                        df["status"] = df["correct"].map(
                            {True: "✓", False: "✗"}
                        )
                        st.dataframe(
                            df[["status", "user", "gold", "pred"]].rename(
                                columns={
                                    "user": "Prompt",
                                    "gold": "Expected",
                                    "pred": "Predicted",
                                }
                            ),
                            use_container_width=True,
                            hide_index=True,
                        )

        st.divider()
        run_import = st.button(
            ":material/rocket_launch: Import & register with Ollama",
            type="primary",
            use_container_width=True,
        )

        import_log_box = st.empty()
        if st.session_state["import_log"]:
            import_log_box.code(
                "\n".join(st.session_state["import_log"][-300:]),
                language="text",
            )

        def _push_import_log(msg: str) -> None:
            st.session_state["import_log"].append(msg)
            import_log_box.code(
                "\n".join(st.session_state["import_log"][-300:]),
                language="text",
            )

        if run_import:
            st.session_state["import_log"] = []
            try:
                result = converter.import_external_to_ollama(
                    Path(st.session_state["import_path"].strip().strip('"')),
                    base_model_repo=(
                        st.session_state.get("import_base_model") or None
                    ),
                    ollama_name=st.session_state["import_ollama_name"],
                    quant=st.session_state["import_quant"],
                    system_prompt=st.session_state["import_system_prompt"],
                    temperature=float(st.session_state["import_temperature"]),
                    top_p=float(st.session_state["import_top_p"]),
                    mode=chosen_mode,
                    chat_format_key=st.session_state.get(
                        "import_chat_format_key"
                    ),
                    log_cb=_push_import_log,
                )
                st.session_state["last_finetuned_model"] = result["ollama_name"]
                st.success(
                    f"Imported `{result['ollama_name']}` into Ollama. "
                    "Switch to the Chat tab to try it."
                )
                st.balloons()
            except Exception as exc:
                _push_import_log(
                    f"\nERROR: {exc}\n{traceback.format_exc()}"
                )
                st.error(f"Import failed: {exc}")
    else:
        st.info("Enter a path and click **Inspect path** to begin.")


# =========================================================================
# Tab 5 — Evaluate (run a JSONL test set against an Ollama model)
# =========================================================================
EVAL_DIR = Path(__file__).parent / "data" / "eval"
REPORTS_DIR = EVAL_DIR / "reports"
EVAL_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

with tab_eval:
    st.header("Evaluate a model against a JSONL test set")
    st.caption(
        "Pick an Ollama model and a test JSONL — the app sends each row's "
        "system + user prompt to Ollama, compares the reply to `expected`, "
        "and tallies pass/fail."
    )

    # ---- Session state -------------------------------------------------------
    ss = st.session_state
    ss.setdefault("eval_status", "idle")    # idle | running | done | aborted | error
    ss.setdefault("eval_rows", [])          # loaded test rows
    ss.setdefault("eval_results", [])       # per-row results
    ss.setdefault("eval_idx", 0)            # next row to process
    ss.setdefault("eval_abort", False)
    ss.setdefault("eval_model", None)
    ss.setdefault("eval_drop_system", False)
    ss.setdefault("eval_temperature", 0.0)
    ss.setdefault("eval_num_predict", 64)
    ss.setdefault("eval_test_label", "")
    ss.setdefault("eval_report_html", None)
    ss.setdefault("eval_report_name", None)

    # ---- Sample format expander ---------------------------------------------
    with st.expander(":material/info: Expected JSONL format (click for sample)",
                     expanded=False):
        st.markdown(
            "One JSON object per line. **`user`** and **`expected`** are "
            "required; **`system`** and **`task`** are optional."
        )
        st.code(eval_runner.SAMPLE_JSONL, language="json")
        st.caption(
            "**Notes:** `system` is sent as the system prompt (omit it for a "
            "'direct question' test). `expected` is compared after lowercasing "
            "and trimming a trailing period — so `4.5`, `4.5.`, and `4.5` are "
            "all treated as equal."
        )
        st.download_button(
            ":material/download: Download sample.jsonl",
            data=eval_runner.SAMPLE_JSONL,
            file_name="sample_test.jsonl",
            mime="application/jsonl",
            use_container_width=False,
        )

    running = ss["eval_status"] == "running"

    # ---- Model + test file pickers ------------------------------------------
    col_a, col_b = st.columns(2)
    with col_a:
        try:
            eval_models = [m["name"] for m in ollama_client.list_models()]
        except Exception as exc:
            eval_models = []
            st.error(f"Cannot list Ollama models: {exc}")
        default_idx = 0
        if ss.get("last_finetuned_model") in eval_models:
            default_idx = eval_models.index(ss["last_finetuned_model"])
        ss["eval_model"] = st.selectbox(
            "Ollama model to test",
            options=eval_models or ["—"],
            index=default_idx if eval_models else 0,
            disabled=running or not eval_models,
            key="eval_model_pick",
        )

    with col_b:
        existing = sorted(
            list(EVAL_DIR.glob("*.jsonl"))
            + [p for p in (Path(__file__).parent / "data").glob("*.jsonl")],
            key=lambda p: p.stat().st_mtime, reverse=True,
        )
        # Dedupe by absolute path (data/ may overlap data/eval/)
        seen: set[str] = set()
        existing = [p for p in existing
                    if not (str(p.resolve()) in seen or seen.add(str(p.resolve())))]
        labels = ["(upload new file)"] + [str(p) for p in existing]
        picked_label = st.selectbox(
            "Test JSONL",
            options=labels,
            index=1 if len(labels) > 1 else 0,
            disabled=running,
            key="eval_file_pick",
        )

    uploaded_test = None
    if picked_label == "(upload new file)":
        uploaded_test = st.file_uploader(
            "Upload a test .jsonl",
            type=["jsonl", "json"],
            accept_multiple_files=False,
            disabled=running,
            key="eval_test_uploader",
        )

    # ---- Options ------------------------------------------------------------
    oc1, oc2, oc3 = st.columns(3)
    oc1.checkbox(
        "Drop system prompt",
        key="eval_drop_system",
        disabled=running,
        help="Sends only the user message — exposes how much the model "
             "depends on the training system prompt.",
    )
    oc2.slider(
        "Temperature", 0.0, 1.5, step=0.05, key="eval_temperature",
        disabled=running,
        help="0.0 = deterministic. Use 0 for classification tests.",
    )
    oc3.number_input(
        "Max new tokens", min_value=8, max_value=2048, step=8,
        key="eval_num_predict", disabled=running,
    )

    # ---- Start / Abort buttons ----------------------------------------------
    bc1, bc2 = st.columns([0.5, 0.5])
    start_clicked = bc1.button(
        ":material/play_arrow: Start evaluation",
        type="primary", use_container_width=True,
        disabled=running or not eval_models,
    )
    abort_clicked = bc2.button(
        ":material/stop_circle: Abort",
        use_container_width=True,
        disabled=not running,
    )

    if abort_clicked and running:
        ss["eval_abort"] = True
        st.toast("Abort requested — stopping after the current row.",
                 icon=":material/stop_circle:")

    # ---- Kick off a new run -------------------------------------------------
    if start_clicked:
        try:
            if uploaded_test is not None:
                raw = uploaded_test.getvalue().decode("utf-8", errors="replace")
                save_name = f"uploaded-{datetime.now().strftime('%Y%m%d-%H%M%S')}.jsonl"
                save_path = EVAL_DIR / save_name
                save_path.write_text(raw, encoding="utf-8")
                test_path = save_path
                ss["eval_test_label"] = save_path.name
            elif picked_label and picked_label != "(upload new file)":
                test_path = Path(picked_label)
                ss["eval_test_label"] = test_path.name
            else:
                st.error("Pick or upload a test JSONL first.")
                st.stop()

            rows = eval_runner.load_test_jsonl(test_path)
            if not rows:
                st.error("Test file has no valid rows.")
                st.stop()

            ss["eval_rows"] = rows
            ss["eval_results"] = []
            ss["eval_idx"] = 0
            ss["eval_abort"] = False
            ss["eval_status"] = "running"
            ss["eval_report_html"] = None
            ss["eval_report_name"] = None
            st.rerun()
        except Exception as exc:
            st.error(f"Could not start: {exc}")

    # ---- Counters + progress ------------------------------------------------
    rows = ss["eval_rows"]
    results = ss["eval_results"]
    total = len(rows)
    done = len(results)
    passed = sum(1 for r in results if r["passed"])
    failed = done - passed

    pc1, pc2, pc3, pc4 = st.columns(4)
    pc1.metric("Total", total or "—")
    pc2.metric("Passed", passed if total else "—",
               delta=f"{(passed/done):.0%}" if done else None,
               delta_color="off")
    pc3.metric("Failed", failed if total else "—",
               delta=f"{(failed/done):.0%}" if done else None,
               delta_color="off")
    pc4.metric("Done", f"{done}/{total}" if total else "—")

    if total:
        st.progress(min(done / total, 1.0),
                    text=f"{done} / {total} processed")

    status_box = st.empty()
    if ss["eval_status"] == "running":
        status_box.info(
            f"Running… ({done}/{total}). Click **Abort** to stop after the "
            "current row."
        )
    elif ss["eval_status"] == "aborted":
        status_box.warning(f"Aborted at {done}/{total}. Partial report below.")
    elif ss["eval_status"] == "done" and total:
        acc = passed / total if total else 0
        status_box.success(
            f"Done. {passed}/{total} passed ({acc:.1%})."
        )

    # ---- One step per rerun (gives Abort a chance to fire) ------------------
    if ss["eval_status"] == "running":
        if ss["eval_abort"]:
            ss["eval_status"] = "aborted"
        elif ss["eval_idx"] >= len(rows):
            ss["eval_status"] = "done"
        else:
            row = rows[ss["eval_idx"]]
            try:
                r = eval_runner.run_one(
                    ss["eval_model"], row,
                    drop_system=ss["eval_drop_system"],
                    temperature=float(ss["eval_temperature"]),
                    num_predict=int(ss["eval_num_predict"]),
                )
                ss["eval_results"].append(r)
                ss["eval_idx"] += 1
            except Exception as exc:
                ss["eval_status"] = "error"
                status_box.error(f"Error at row {ss['eval_idx']+1}: {exc}")
            else:
                st.rerun()

    # ---- Final report (built when run finishes or is aborted) ---------------
    if ss["eval_status"] in ("done", "aborted") and ss["eval_results"]:
        summary = eval_runner.summarize(ss["eval_results"])

        st.subheader("Per-task breakdown")
        st.dataframe(
            pd.DataFrame(summary["per_task"]),
            use_container_width=True,
            hide_index=True,
            column_config={
                "task": "task",
                "total": "total",
                "passed": "passed",
                "failed": "failed",
                "accuracy": st.column_config.ProgressColumn(
                    "accuracy", format="%.1f%%", min_value=0.0, max_value=1.0
                ),
            },
        )

        # Build (or reuse) the HTML report
        if ss["eval_report_html"] is None:
            ss["eval_report_html"] = eval_runner.build_html_report(
                model=ss["eval_model"],
                results=ss["eval_results"],
                summary=summary,
                drop_system=ss["eval_drop_system"],
            )
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            safe_model = ss["eval_model"].replace(":", "-").replace("/", "-")
            ss["eval_report_name"] = f"eval-{safe_model}-{ts}.html"
            # Persist alongside the test file for later inspection.
            (REPORTS_DIR / ss["eval_report_name"]).write_text(
                ss["eval_report_html"], encoding="utf-8"
            )

        st.download_button(
            ":material/download: Download HTML report",
            data=ss["eval_report_html"],
            file_name=ss["eval_report_name"],
            mime="text/html",
            type="primary",
            use_container_width=True,
        )
        st.caption(
            f"Report also saved to `data/eval/reports/{ss['eval_report_name']}`."
        )

        with st.expander(":material/visibility: Preview failed rows in app",
                         expanded=False):
            fails = [r for r in ss["eval_results"] if not r["passed"]]
            if not fails:
                st.success("No failures.")
            else:
                for i, r in enumerate(fails[:50], 1):
                    st.markdown(
                        f"**{i}. [{r['task']}]** "
                        f"user: `{r['user']}` · expected: `{r['expected']}` · "
                        f"got: `{(r['got'] or '').strip()[:120]}`"
                    )
                if len(fails) > 50:
                    st.caption(f"… and {len(fails)-50} more — see HTML report.")


# =========================================================================
# Tab 6 — Chat & validate
# =========================================================================
@st.dialog("Confirm model deletion")
def _confirm_delete_dialog(name: str) -> None:
    st.warning(
        f"Delete **`{name}`** from Ollama? This permanently removes the model "
        "from disk and cannot be undone."
    )
    c1, c2 = st.columns(2)
    if c1.button("Cancel", use_container_width=True, key="dlg_cancel"):
        st.rerun()
    if c2.button(":material/delete: Delete", type="primary",
                 use_container_width=True, key="dlg_confirm"):
        try:
            ollama_client.delete_model(name)
        except Exception as exc:
            st.error(f"Delete failed: {exc}")
            return
        if st.session_state.get("last_finetuned_model") == name:
            st.session_state["last_finetuned_model"] = None
        if st.session_state.get("chat_model_pick") == name:
            st.session_state.pop("chat_model_pick", None)
            st.session_state["chat_history"] = []
        st.toast(f"Deleted `{name}`", icon=":material/check_circle:")
        st.rerun()


with tab_chat:
    st.header("Chat with a fine-tuned (or original) model")

    try:
        live_models_info = ollama_client.list_models()
        live_models = [m["name"] for m in live_models_info]
    except Exception as exc:
        live_models_info = []
        live_models = []
        st.error(f"Cannot list Ollama models: {exc}")

    # Trigger the dialog if a delete was queued from any row button.
    pending = st.session_state.pop("_pending_delete", None)
    if pending:
        _confirm_delete_dialog(pending)

    with st.expander(
        f":material/inventory_2: Manage models ({len(live_models_info)})",
        expanded=False,
    ):
        st.caption(
            "All Ollama models on this machine. Click :material/delete: to remove a model."
        )
        if not live_models_info:
            st.info("No Ollama models found.")
        for m in sorted(live_models_info, key=lambda x: x["name"].lower()):
            cols = st.columns([0.55, 0.30, 0.15])
            cols[0].markdown(f"`{m['name']}`")
            size_mb = (m.get("size") or 0) / (1024 * 1024)
            cols[1].caption(f"{size_mb:,.1f} MB")
            if cols[2].button(
                ":material/delete:",
                key=f"del_model_{m['name']}",
                help=f"Delete {m['name']} from Ollama",
                use_container_width=True,
            ):
                st.session_state["_pending_delete"] = m["name"]
                st.rerun()

    last = st.session_state.get("last_finetuned_model")
    default_idx = 0
    if last and last in live_models:
        default_idx = live_models.index(last)

    pick = st.selectbox("Model to chat with", options=live_models or ["—"],
                        index=default_idx if live_models else 0,
                        key="chat_model_pick")
    c1, c2, c3 = st.columns([0.70, 0.15, 0.15])
    if c2.button(":material/delete_sweep: Clear chat", use_container_width=True):
        st.session_state["chat_history"] = []
        st.rerun()
    if c3.button(":material/delete: Delete model", use_container_width=True,
                 disabled=not live_models, help="Delete the currently-selected model"):
        st.session_state["_pending_delete"] = pick
        st.rerun()

    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Ask a question about your fine-tuning documents...")
    if prompt and live_models:
        st.session_state["chat_history"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        msgs = [{"role": "system", "content": st.session_state["system_prompt"]}]
        msgs.extend(st.session_state["chat_history"])

        with st.chat_message("assistant"):
            placeholder = st.empty()
            buf = ""
            try:
                for chunk in ollama_client.chat_stream(
                    pick, msgs,
                    temperature=float(st.session_state["chat_temperature"]),
                    top_p=float(st.session_state["chat_top_p"]),
                    num_predict=int(st.session_state["chat_num_predict"]),
                ):
                    buf += chunk
                    placeholder.markdown(buf + "▌")
                placeholder.markdown(buf)
                st.session_state["chat_history"].append(
                    {"role": "assistant", "content": buf}
                )
            except Exception as exc:
                placeholder.error(f"Chat failed: {exc}")
