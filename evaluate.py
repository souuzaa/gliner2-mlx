#!/usr/bin/env python3
"""Quality check of converted MLX variants on the fastino/fast-decisions dev split.

For each variant: head-level exact-match accuracy (averaged over the 17 domains),
agreement of the top answer with the fp32 reference, mean / max |Δp| vs fp32,
median latency (batch size 1) and peak memory. The fp32 reference is the original
checkpoint run by gliner2_mlx in float32 (matches PyTorch gliner2 to ~1e-5, see
verify_parity.py).

    python evaluate.py mlx_models/GLiNER2.5-Decide-4bit mlx_models/GLiNER2.5-Decide-8bit
    python evaluate.py mlx_models/* --limit 20 --markdown quality.md
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

import gliner2_mlx

DATASET = "fastino/fast-decisions"


def load_rows(limit: int | None):
    from huggingface_hub import snapshot_download

    root = Path(snapshot_download(DATASET, repo_type="dataset", allow_patterns=["*.jsonl"]))
    rows = []
    for f in sorted(root.glob("*.jsonl")):
        lines = f.read_text().splitlines()[:limit]
        rows.extend((f.stem, json.loads(line)) for line in lines if line.strip())
    return rows


def tasks_of(row):
    return {
        c["task"]: {"labels": c["labels"], "multi_label": c.get("multi_label", False)}
        for c in row["output"]["classifications"]
    }


def run(model, rows):
    preds, latencies = [], []
    for _, row in rows:
        t = time.perf_counter()
        preds.append(model.predict(row["input"], tasks_of(row)))
        latencies.append(time.perf_counter() - t)
    return preds, latencies


def decide(probs: dict, multi: bool, threshold: float = 0.5):
    if multi:
        chosen = {l for l, p in probs.items() if p >= threshold}
        return chosen or {max(probs, key=probs.get)}
    return max(probs, key=probs.get)


def score(rows, preds, ref=None):
    per_domain, agree, deltas = {}, [], []
    for i, ((domain, row), pred) in enumerate(zip(rows, preds)):
        for c in row["output"]["classifications"]:
            multi = c.get("multi_label", False)
            answer = decide(pred[c["task"]], multi)
            gold = set(c["true_label"]) if multi else c["true_label"][0]
            per_domain.setdefault(domain, []).append(answer == gold)
            if ref is not None:
                r = ref[i][c["task"]]
                agree.append(answer == decide(r, multi))
                deltas.append(max(abs(pred[c["task"]][l] - r[l]) for l in r))
    accuracy = 100 * statistics.mean(statistics.mean(v) for v in per_domain.values())
    out = {"accuracy": accuracy, "heads": sum(len(v) for v in per_domain.values())}
    if ref is not None:
        out.update(agree=100 * statistics.mean(agree), mean_dp=statistics.mean(deltas), max_dp=max(deltas))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("variants", nargs="+", type=Path, help="converted MLX model dirs")
    p.add_argument("--reference", default="fastino/GLiNER2.5-Decide", help="original checkpoint (run in fp32)")
    p.add_argument("--limit", type=int, help="rows per domain (default: all 100)")
    p.add_argument("--markdown", type=Path, help="write the results table here (for convert.py --quality)")
    args = p.parse_args()

    rows = load_rows(args.limit)
    print(f"[INFO] {len(rows)} rows from {DATASET} (dev split)")

    results = []
    mx.reset_peak_memory()
    ref_model = gliner2_mlx.load(args.reference, dtype=mx.float32)
    ref, ref_lat = run(ref_model, rows)
    results.append(("fp32 reference", None, score(rows, ref), ref_lat, mx.get_peak_memory()))
    del ref_model

    for path in args.variants:
        mx.clear_cache()
        mx.reset_peak_memory()
        model = gliner2_mlx.load(path)
        preds, lat = run(model, rows)
        size = (path / "model.safetensors").stat().st_size / 1e6
        results.append((path.name, size, score(rows, preds, ref), lat, mx.get_peak_memory()))
        del model

    head = ("| Variant | Size | Accuracy (dev) | Same answer as fp32 | Mean / max abs Δp | Median latency | Peak memory |\n"
            "|---|---|---|---|---|---|---|")
    lines = [head]
    for name, size, s, lat, peak in results:
        lines.append(
            f"| {name} | {f'{size:.0f} MB' if size else '1.9 GB'} | {s['accuracy']:.1f}% | "
            f"{f'{s['agree']:.1f}%' if 'agree' in s else '—'} | "
            f"{f'{s['mean_dp']:.4f} / {s['max_dp']:.3f}' if 'agree' in s else '—'} | "
            f"{1000 * statistics.median(lat):.0f} ms | {peak / 1e9:.2f} GB |"
        )
    table = "\n".join(lines)
    n_heads = results[0][2]["heads"]
    note = (
        f"\n\nMeasured on {len(rows)} rows / {n_heads} heads of the `{DATASET}` **dev** split "
        "(the published benchmark uses a held-out test split, so these accuracies are not comparable to it). "
        "Accuracy is head-level exact match averaged over the 17 domains; multi-label heads use threshold 0.5. "
        "Latency is per `predict()` call at batch size 1."
    )
    print(table + note)
    if args.markdown:
        args.markdown.write_text("## Quality check: fast-decisions (dev split)\n\n" + table + note + "\n")


if __name__ == "__main__":
    main()
