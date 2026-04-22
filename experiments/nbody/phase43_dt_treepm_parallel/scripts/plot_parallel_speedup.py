#!/usr/bin/env python3
"""Genera `speedup_vs_threads.png` y CSV a partir de `test5_parallel_speedup.json`."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test5", required=True, type=Path)
    ap.add_argument("--outdir", required=True, type=Path)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    with args.test5.open() as f:
        d = json.load(f)
    rows = d["rows"]

    ts = [r["threads"] for r in rows]
    walls = [r["wall_s"] for r in rows]
    speedups = [r.get("speedup_vs_1", None) for r in rows]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ts, speedups, "o-", label="speedup medido")
    ax.plot(ts, ts, "k--", alpha=0.5, label="ideal (lineal)")
    ax.set_xlabel("hilos Rayon")
    ax.set_ylabel("speedup vs 1 hilo")
    ax.set_title(f"Phase 43 — speedup TreePM (N={d.get('n','?')}³)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.outdir / "speedup_vs_threads.png", dpi=150)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.plot(ts, walls, "o-")
    ax2.set_xlabel("hilos Rayon")
    ax2.set_ylabel("wall-clock de un step TreePM [s]")
    ax2.set_yscale("log")
    ax2.set_title(f"Phase 43 — wall-time vs hilos (N={d.get('n','?')}³)")
    ax2.grid(True, which="both", alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(args.outdir / "walltime_vs_threads.png", dpi=150)
    plt.close(fig2)

    with (args.outdir / "phase43_parallel_speedup.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["threads", "wall_s", "speedup_vs_1"])
        for r in rows:
            w.writerow([r["threads"], r["wall_s"], r.get("speedup_vs_1")])

    print(f"[phase43] paralelo → {args.outdir}")
    print(f"  decision: {d.get('decision')}")
    print(f"  speedup_at_highest: {d.get('speedup_at_highest')}")


if __name__ == "__main__":
    main()
