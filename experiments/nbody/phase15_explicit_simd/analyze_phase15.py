#!/usr/bin/env python3
"""
Phase 15 — Análisis de resultados Explicit AVX2 SIMD.

Lee los timings.json generados por run_phase15.sh y produce:
  1. Tabla de speedup P15 vs P14 por configuración
  2. Breakdown de tiempos por fase (let_tree_walk, apply_leaf, etc.)
  3. Figuras PNG comparativas
"""

import json
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path("results")
PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)


def load_timings(result_dir: Path) -> dict | None:
    tj = result_dir / "timings.json"
    if not tj.exists():
        return None
    with open(tj) as f:
        return json.load(f)


def parse_run_dir(name: str) -> dict:
    m = re.match(r"^(.+)_N(\d+)_P(\d+)_(p14_fused|p15_explicit)$", name)
    if not m:
        return {}
    return {
        "group": m.group(1),
        "n": int(m.group(2)),
        "p": int(m.group(3)),
        "variant": m.group(4),
    }


def collect_results() -> list[dict]:
    rows = []
    if not RESULTS_DIR.exists():
        print(f"ERROR: {RESULTS_DIR} no existe.", file=sys.stderr)
        return rows

    for d in sorted(RESULTS_DIR.iterdir()):
        if not d.is_dir():
            continue
        meta = parse_run_dir(d.name)
        if not meta:
            continue
        data = load_timings(d)
        if data is None:
            print(f"  WARNING: sin timings.json en {d.name}", file=sys.stderr)
            continue

        hpc = data.get("hpc", {})
        row = {**meta}
        row["mean_wall_s"] = data.get("mean_step_wall_s", None)
        for key in [
            "mean_let_tree_walk_s", "mean_apply_leaf_s",
            "mean_walk_local_s", "mean_let_alltoallv_s", "mean_rmn_soa_pack_s",
            "mean_accel_from_let_soa_s", "soa_simd_active",
        ]:
            row[key] = hpc.get(key, None)

        rows.append(row)
    return rows


def fmt(v, unit="ms", decimals=2):
    if v is None:
        return "—"
    if unit == "ms":
        return f"{v*1000:.{decimals}f}"
    if unit == "x":
        return f"{v:.{decimals}f}×"
    return f"{v:.{decimals}f}"


def main():
    rows = collect_results()
    if not rows:
        print("Sin resultados. Ejecuta run_phase15.sh primero.")
        return

    # Agrupar por (group, N, P)
    from collections import defaultdict
    by_key = defaultdict(dict)
    for r in rows:
        k = (r["group"], r["n"], r["p"])
        by_key[k][r["variant"]] = r

    # ── Tabla de speedup ──────────────────────────────────────────────────────
    print("\n" + "="*90)
    print("TABLA: Speedup P15 vs P14")
    print("="*90)
    hdr = f"{'Config':<35} {'P14 wall(ms)':>14} {'P15 wall(ms)':>14} {'Speedup':>10} {'P14 LT(ms)':>12} {'P15 LT(ms)':>12} {'LT speedup':>12}"
    print(hdr)
    print("-"*90)

    speedups = []
    lt_speedups = []
    configs_sorted = sorted(by_key.keys(), key=lambda x: (x[0], x[1], x[2]))

    for k in configs_sorted:
        group, n, p = k
        variants = by_key[k]
        p14 = variants.get("p14_fused")
        p15 = variants.get("p15_explicit")

        if p14 is None or p15 is None:
            print(f"  {group}_N{n}_P{p}: faltan variantes ({list(variants.keys())})")
            continue

        wall14 = p14.get("mean_wall_s")
        wall15 = p15.get("mean_wall_s")
        lt14 = p14.get("mean_let_tree_walk_s")
        lt15 = p15.get("mean_let_tree_walk_s")

        speedup = (wall14 / wall15) if (wall14 and wall15) else None
        lt_sp = (lt14 / lt15) if (lt14 and lt15) else None

        if speedup:
            speedups.append(speedup)
        if lt_sp:
            lt_speedups.append(lt_sp)

        label = f"{group}_N{n}_P{p}"
        print(f"  {label:<33} {fmt(wall14):>14} {fmt(wall15):>14} {fmt(speedup,'x'):>10} {fmt(lt14):>12} {fmt(lt15):>12} {fmt(lt_sp,'x'):>12}")

    print("-"*90)
    if speedups:
        print(f"  {'Media':>33} {'':>14} {'':>14} {np.mean(speedups):.2f}× {''*12} {''*12} {np.mean(lt_speedups):.2f}×" if lt_speedups else "")

    # ── Figura: wall time por N ───────────────────────────────────────────────
    for p_filter in [2, 4]:
        scale_keys = [(g, n, p) for (g, n, p) in configs_sorted
                      if g == "scaling_mpi" and p == p_filter]
        if not scale_keys:
            continue

        ns = [k[1] for k in scale_keys]
        wall14s = []
        wall15s = []
        lt14s = []
        lt15s = []
        for k in scale_keys:
            variants = by_key[k]
            p14 = variants.get("p14_fused")
            p15 = variants.get("p15_explicit")
            wall14s.append((p14.get("mean_wall_s", 0) or 0) * 1000)
            wall15s.append((p15.get("mean_wall_s", 0) or 0) * 1000 if p15 else 0)
            lt14s.append((p14.get("mean_let_tree_walk_s", 0) or 0) * 1000)
            lt15s.append((p15.get("mean_let_tree_walk_s", 0) or 0) * 1000 if p15 else 0)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        x = np.arange(len(ns))
        w = 0.35
        ax1.bar(x - w/2, wall14s, w, label="P14 fused", color="steelblue")
        ax1.bar(x + w/2, wall15s, w, label="P15 explicit AVX2", color="darkorange")
        ax1.set_xlabel("N")
        ax1.set_ylabel("Wall time por paso (ms)")
        ax1.set_title(f"Wall time total — P={p_filter}")
        ax1.set_xticks(x)
        ax1.set_xticklabels([f"N={n}" for n in ns])
        ax1.legend()
        ax1.grid(axis="y", alpha=0.3)

        ax2.bar(x - w/2, lt14s, w, label="P14 fused", color="steelblue")
        ax2.bar(x + w/2, lt15s, w, label="P15 explicit AVX2", color="darkorange")
        ax2.set_xlabel("N")
        ax2.set_ylabel("LetTree walk (ms)")
        ax2.set_title(f"LetTree walk — P={p_filter}")
        ax2.set_xticks(x)
        ax2.set_xticklabels([f"N={n}" for n in ns])
        ax2.legend()
        ax2.grid(axis="y", alpha=0.3)

        fig.tight_layout()
        out = PLOTS_DIR / f"wall_time_P{p_filter}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  Saved: {out}")

    # ── Figura: speedup por N ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    for p_filter in [2, 4]:
        scale_keys = [(g, n, p) for (g, n, p) in configs_sorted
                      if g == "scaling_mpi" and p == p_filter]
        if not scale_keys:
            continue
        ns_plot = [k[1] for k in scale_keys]
        sps = []
        for k in scale_keys:
            variants = by_key[k]
            p14 = variants.get("p14_fused")
            p15 = variants.get("p15_explicit")
            if p14 and p15 and p14.get("mean_wall_s") and p15.get("mean_wall_s"):
                sps.append(p14["mean_wall_s"] / p15["mean_wall_s"])
            else:
                sps.append(0)
        ax.plot(ns_plot, sps, marker="o", label=f"P={p_filter}")

    ax.axhline(1.0, color="gray", linestyle="--", label="sin mejora")
    ax.set_xlabel("N")
    ax.set_ylabel("Speedup (P14 / P15)")
    ax.set_title("Speedup P15 explicit AVX2 vs P14 fused")
    ax.legend()
    ax.grid(alpha=0.3)
    out = PLOTS_DIR / "speedup.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")

    print("\n=== Análisis completado ===")
    print(f"Figuras en: {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
