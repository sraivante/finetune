"""Evaluate an Ollama model against a JSONL test file and build an HTML report.

Test JSONL row schema (one JSON object per line):
    {
        "task":     "investment_profile",        # optional, free-form label
        "system":   "You are a strict ...",      # optional — sent as system
        "user":     "agresive",                  # required — sent as user
        "expected": "8.5"                        # required — compared to output
    }
"""
from __future__ import annotations

import collections
import html
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import ollama

SAMPLE_JSONL = (
    '{"task": "investment_profile", '
    '"system": "You are a strict classifier. Map the user input to '
    'EXACTLY one of: Conservative -> 4.5, Moderate -> 6.5, Aggressive -> 8.5. '
    'Output ONLY the number.", '
    '"user": "agresive", "expected": "8.5"}\n'
    '{"task": "yes_no_intent", '
    '"system": "Return only yes, no, or RETRY.", '
    '"user": "abort it", "expected": "no"}\n'
    '{"task": "free_form", '
    '"user": "what is 2 + 2?", "expected": "4"}\n'
)


def load_test_jsonl(path: Path) -> list[dict]:
    """Read a test JSONL file. Skips blank lines; rejects malformed rows."""
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {i}: invalid JSON ({exc.msg})") from exc
            if "user" not in rec or "expected" not in rec:
                raise ValueError(
                    f"line {i}: missing required 'user' and/or 'expected' field"
                )
            rows.append({
                "task": str(rec.get("task") or "unspecified"),
                "system": rec.get("system") or "",
                "user": str(rec["user"]),
                "expected": str(rec["expected"]),
            })
    return rows


def normalize(s: str) -> str:
    """Lowercase, strip, drop a trailing period — for tolerant equality."""
    return s.strip().strip(".").lower()


def run_one(model: str, row: dict, *,
            drop_system: bool = False,
            temperature: float = 0.0,
            num_predict: int = 64) -> dict:
    """Send one test row to Ollama and return a result dict."""
    msgs: list[dict] = []
    if not drop_system and row.get("system"):
        msgs.append({"role": "system", "content": row["system"]})
    msgs.append({"role": "user", "content": row["user"]})

    error = None
    got = ""
    try:
        resp = ollama.chat(
            model=model,
            messages=msgs,
            options={"temperature": temperature, "num_predict": num_predict},
        )
        got = resp["message"]["content"]
    except Exception as exc:
        error = str(exc)

    passed = (error is None) and (normalize(got) == normalize(row["expected"]))
    return {
        "task": row["task"],
        "system": row["system"],
        "user": row["user"],
        "expected": row["expected"],
        "got": got,
        "passed": passed,
        "error": error,
    }


def summarize(results: list[dict]) -> dict:
    """Roll results up into overall + per-task counts."""
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    failed = total - passed
    per_task_total: dict[str, int] = collections.Counter()
    per_task_pass: dict[str, int] = collections.Counter()
    for r in results:
        per_task_total[r["task"]] += 1
        if r["passed"]:
            per_task_pass[r["task"]] += 1
    per_task = []
    for t in sorted(per_task_total):
        n = per_task_total[t]
        p = per_task_pass[t]
        per_task.append({
            "task": t, "total": n, "passed": p, "failed": n - p,
            "accuracy": p / n if n else 0.0,
        })
    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "accuracy": (passed / total) if total else 0.0,
        "per_task": per_task,
    }


# ---------------------------------------------------------------------------
# HTML report — self-contained, no external assets, no JS required
# ---------------------------------------------------------------------------
_REPORT_CSS = """
body { font-family: -apple-system, Segoe UI, Roboto, sans-serif;
       margin: 24px; color: #222; background: #fafafa; }
h1 { margin: 0 0 4px 0; }
.subtle { color: #666; font-size: 0.9rem; }
.summary { display: flex; gap: 16px; margin: 20px 0; flex-wrap: wrap; }
.card { background: white; border: 1px solid #ddd; border-radius: 8px;
        padding: 12px 18px; min-width: 110px; }
.card .v { font-size: 1.8rem; font-weight: 600; }
.card .l { font-size: 0.8rem; text-transform: uppercase; color: #777;
           letter-spacing: 0.5px; }
.card.pass .v { color: #1f883d; }
.card.fail .v { color: #cf222e; }
table { border-collapse: collapse; width: 100%; background: white;
        border: 1px solid #ddd; border-radius: 8px; overflow: hidden; }
th, td { text-align: left; padding: 8px 12px;
         border-bottom: 1px solid #eee; font-size: 0.95rem; }
th { background: #f5f5f5; font-weight: 600; }
tr:last-child td { border-bottom: none; }
.task-acc.good { color: #1f883d; font-weight: 600; }
.task-acc.bad  { color: #cf222e; font-weight: 600; }
.task-acc.mid  { color: #9a6700; font-weight: 600; }
details { background: white; border: 1px solid #ddd; border-radius: 8px;
          margin: 8px 0; padding: 0; }
details summary { padding: 10px 14px; cursor: pointer; user-select: none;
                  display: flex; gap: 12px; align-items: center; }
details summary::-webkit-details-marker { display: none; }
details summary::before { content: "▶"; font-size: 0.8rem; color: #999;
                          transition: transform 0.15s; }
details[open] summary::before { transform: rotate(90deg); }
.badge { padding: 2px 8px; border-radius: 12px; font-size: 0.75rem;
         font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
.badge.pass { background: #dcfce7; color: #166534; }
.badge.fail { background: #fee2e2; color: #991b1b; }
.badge.task { background: #e0e7ff; color: #3730a3; }
.summary-text { color: #444; flex: 1; }
.summary-text .u { font-family: ui-monospace, Menlo, Consolas, monospace; }
.detail-grid { padding: 14px 18px 18px 18px; display: grid;
               grid-template-columns: 100px 1fr; gap: 8px 16px;
               border-top: 1px solid #eee; }
.detail-grid .label { font-weight: 600; color: #555; font-size: 0.85rem;
                      text-transform: uppercase; letter-spacing: 0.5px; }
.detail-grid pre { margin: 0; padding: 8px 12px; background: #f8f8f8;
                   border: 1px solid #e5e5e5; border-radius: 4px;
                   white-space: pre-wrap; word-wrap: break-word;
                   font-family: ui-monospace, Menlo, Consolas, monospace;
                   font-size: 0.85rem; max-height: 300px; overflow: auto; }
.controls { margin: 16px 0 8px 0; display: flex; gap: 8px;
            align-items: center; flex-wrap: wrap; }
.controls .filter { padding: 6px 12px; border: 1px solid #ccc;
                    background: white; border-radius: 6px; cursor: pointer;
                    font-size: 0.9rem; }
.controls .filter.active { background: #222; color: white;
                           border-color: #222; }
.hide-pass details.row.pass { display: none; }
.hide-fail details.row.fail { display: none; }
"""

_FILTER_JS = """
function setFilter(mode) {
  var body = document.body;
  body.classList.remove('hide-pass', 'hide-fail');
  if (mode === 'fail') body.classList.add('hide-pass');
  if (mode === 'pass') body.classList.add('hide-fail');
  document.querySelectorAll('.filter').forEach(function(b) {
    b.classList.toggle('active', b.dataset.mode === mode);
  });
}
"""


def _task_class(acc: float) -> str:
    if acc >= 0.9:
        return "good"
    if acc >= 0.7:
        return "mid"
    return "bad"


def build_html_report(*, model: str, results: list[dict],
                      summary: dict, drop_system: bool) -> str:
    """Build a single self-contained HTML string for the report."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total = summary["total"]
    passed = summary["passed"]
    failed = summary["failed"]
    acc = summary["accuracy"]

    parts: list[str] = []
    parts.append("<!doctype html><html><head><meta charset='utf-8'>")
    parts.append(f"<title>Eval report — {html.escape(model)}</title>")
    parts.append(f"<style>{_REPORT_CSS}</style></head><body>")

    parts.append(f"<h1>Evaluation report</h1>")
    parts.append(
        f"<div class='subtle'>Model: <code>{html.escape(model)}</code> · "
        f"Run at {ts} · System prompt "
        f"{'<b>dropped</b>' if drop_system else 'sent'}</div>"
    )

    parts.append("<div class='summary'>")
    parts.append(f"<div class='card'><div class='v'>{total}</div>"
                 "<div class='l'>Total</div></div>")
    parts.append(f"<div class='card pass'><div class='v'>{passed}</div>"
                 "<div class='l'>Passed</div></div>")
    parts.append(f"<div class='card fail'><div class='v'>{failed}</div>"
                 "<div class='l'>Failed</div></div>")
    parts.append(f"<div class='card'><div class='v'>{acc:.1%}</div>"
                 "<div class='l'>Accuracy</div></div>")
    parts.append("</div>")

    parts.append("<h2>Per-task breakdown</h2>")
    parts.append("<table><thead><tr>"
                 "<th>Task</th><th>Passed</th><th>Failed</th>"
                 "<th>Total</th><th>Accuracy</th></tr></thead><tbody>")
    for row in summary["per_task"]:
        cls = _task_class(row["accuracy"])
        parts.append(
            f"<tr><td>{html.escape(row['task'])}</td>"
            f"<td>{row['passed']}</td><td>{row['failed']}</td>"
            f"<td>{row['total']}</td>"
            f"<td class='task-acc {cls}'>{row['accuracy']:.1%}</td></tr>"
        )
    parts.append("</tbody></table>")

    parts.append("<h2>Per-prompt results</h2>")
    parts.append(
        "<div class='controls'>"
        "<span>Filter:</span>"
        "<button class='filter active' data-mode='all' "
        "onclick=\"setFilter('all')\">All</button>"
        "<button class='filter' data-mode='fail' "
        "onclick=\"setFilter('fail')\">Failed only</button>"
        "<button class='filter' data-mode='pass' "
        "onclick=\"setFilter('pass')\">Passed only</button>"
        "<span class='subtle'>Click a row to see the full prompt and output.</span>"
        "</div>"
    )

    for i, r in enumerate(results, 1):
        status_cls = "pass" if r["passed"] else "fail"
        badge = "PASS" if r["passed"] else "FAIL"
        parts.append(
            f"<details class='row {status_cls}'><summary>"
            f"<span class='badge {status_cls}'>{badge}</span>"
            f"<span class='badge task'>{html.escape(r['task'])}</span>"
            f"<span class='summary-text'>"
            f"#{i} · <span class='u'>{html.escape(r['user'][:80])}</span> "
            f"→ expected <code>{html.escape(r['expected'])}</code>, "
            f"got <code>{html.escape((r['got'] or '').strip()[:80])}</code>"
            f"</span></summary>"
            f"<div class='detail-grid'>"
            f"<div class='label'>Task</div>"
            f"<div>{html.escape(r['task'])}</div>"
            f"<div class='label'>System</div>"
            f"<pre>{html.escape(r['system']) if r['system'] else '(none sent)'}</pre>"
            f"<div class='label'>User</div>"
            f"<pre>{html.escape(r['user'])}</pre>"
            f"<div class='label'>Expected</div>"
            f"<pre>{html.escape(r['expected'])}</pre>"
            f"<div class='label'>Got</div>"
            f"<pre>{html.escape(r['got'] or '')}</pre>"
        )
        if r.get("error"):
            parts.append(
                f"<div class='label'>Error</div>"
                f"<pre>{html.escape(r['error'])}</pre>"
            )
        parts.append("</div></details>")

    parts.append(f"<script>{_FILTER_JS}</script></body></html>")
    return "".join(parts)
