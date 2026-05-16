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
    ollama_client,
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
tab_cfg, tab_data, tab_run, tab_chat = st.tabs([
    ":material/tune: Configure",
    ":material/dataset: Dataset",
    ":material/play_arrow: Fine-tune",
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
# Tab 3 — Chat & validate
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
