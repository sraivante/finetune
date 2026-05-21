"""Evaluate a fine-tuned Ollama model against a JSONL test set.

Usage (from the project root):
    python scripts/eval_finetuned.py --model <ollama_tag>
    python scripts/eval_finetuned.py --model <ollama_tag> --no-system

`--no-system` drops the training system prompt so you can see *for yourself*
that the model relies on it (it'll score much lower without it).

Test file format (one JSON object per line):
    {"task": "...", "system": "...", "user": "...", "expected": "..."}
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import ollama


def normalize(s: str) -> str:
    """Strip whitespace, lowercase, and trim a trailing period for fair compare."""
    return s.strip().strip(".").lower()


def run(model: str, test_path: Path, drop_system: bool,
        temperature: float, num_predict: int, show_misses: int) -> int:
    rows = []
    with test_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    per_task_correct: dict[str, int] = collections.Counter()
    per_task_total: dict[str, int] = collections.Counter()
    misses: list[dict] = []

    for i, row in enumerate(rows, 1):
        msgs = []
        if not drop_system and row.get("system"):
            msgs.append({"role": "system", "content": row["system"]})
        msgs.append({"role": "user", "content": row["user"]})

        try:
            resp = ollama.chat(
                model=model,
                messages=msgs,
                options={"temperature": temperature, "num_predict": num_predict},
            )
            got = resp["message"]["content"]
        except Exception as exc:
            print(f"  [{i}/{len(rows)}] ERROR: {exc}")
            got = ""

        exp = row["expected"]
        ok = normalize(got) == normalize(exp)
        task = row.get("task", "unknown")
        per_task_total[task] += 1
        if ok:
            per_task_correct[task] += 1
        else:
            misses.append({"task": task, "user": row["user"],
                           "expected": exp, "got": got.strip()})

        if i % 10 == 0 or i == len(rows):
            done = sum(per_task_total.values())
            correct = sum(per_task_correct.values())
            print(f"  [{i}/{len(rows)}] running accuracy: "
                  f"{correct}/{done} = {correct/done:.1%}")

    print()
    print(f"=== Results for `{model}` "
          f"(system_prompt={'dropped' if drop_system else 'kept'}) ===")
    total_correct = sum(per_task_correct.values())
    total_seen = sum(per_task_total.values())
    print(f"Overall: {total_correct}/{total_seen} = "
          f"{total_correct/total_seen:.1%}\n")

    print(f"{'task':<35} {'acc':>10}   n")
    print("-" * 55)
    for task in sorted(per_task_total):
        n = per_task_total[task]
        c = per_task_correct[task]
        print(f"{task:<35} {c/n:>9.1%}   {n}")

    if misses and show_misses > 0:
        print(f"\n=== First {min(show_misses, len(misses))} misses ===")
        for m in misses[:show_misses]:
            print(f"  [{m['task']}] user={m['user']!r}  "
                  f"expected={m['expected']!r}  got={m['got']!r}")

    return 0 if total_correct == total_seen else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True,
                        help="Ollama model tag, e.g. tinyllama-imported-...")
    parser.add_argument("--test", default="data/test_prompts.jsonl",
                        help="Path to test JSONL (default: data/test_prompts.jsonl)")
    parser.add_argument("--no-system", action="store_true",
                        help="Drop the training-time system prompt — exposes "
                             "the model's reliance on it.")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Generation temperature (default 0.0 for classification).")
    parser.add_argument("--num-predict", type=int, default=32,
                        help="Max tokens to generate (default 32).")
    parser.add_argument("--show-misses", type=int, default=20,
                        help="Print first N wrong answers (default 20, 0 to hide).")
    args = parser.parse_args()

    test_path = Path(args.test)
    if not test_path.exists():
        print(f"Test file not found: {test_path}", file=sys.stderr)
        return 2

    return run(
        model=args.model,
        test_path=test_path,
        drop_system=args.no_system,
        temperature=args.temperature,
        num_predict=args.num_predict,
        show_misses=args.show_misses,
    )


if __name__ == "__main__":
    sys.exit(main())
