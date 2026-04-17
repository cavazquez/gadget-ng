#!/usr/bin/env python3
"""
Phase 15b — Análisis del sweep de leaf_max.

Produce:
  1. Tabla: batch_size_avg vs leaf_max (por N, P)
  2. Tabla: wall time y speedup P15/P14 vs leaf_max
  3. Tabla: LetTree walk speedup vs leaf_max
  4. Tabla: volumen LET (bytes, nodos) vs leaf_max
  5. Figura 1: batch size promedio vs leaf_max
  6. Figura 2: wall time vs leaf_max (barras P14 vs P15)
  7. Figura 3: speedup P15/P14 vs leaf_max (con línea 1.0)
  8. Figura 4: LET nodes importados vs leaf_max
"""

import json
import re
import sys
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path("results")
PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)

LEAF_MAX_VALUES = [8, 16, 32, 64]


def load_run(result_dir: Path) -> dict | None:
    tj = result_dir / "timings.json"
    if not tj.exists():
        return None
    with open(tj) as f:
        data = json.load(f)
    hpc = data.get("hpc", {})
    calls = hpc.get("mean_apply_leaf_calls", 0) or 0
    rmns  = hpc.get("mean_apply_leaf_rmn_count", 0) or 0
    return {
        "wall_s":         data.get("mean_step_wall_s"),
        "lt_walk_s":      hpc.get("mean_let_tree_walk_s"),
        "apply_leaf_s":   hpc.get("mean_apply_leaf_s"),
        "batch_size_avg": (rmns / calls) if calls > 0 else None,
        "let_imported":   hpc.get("mean_let_nodes_imported"),
        "let_exported":   hpc.get("mean_let_nodes_exported"),
        "bytes_sent":     hpc.get("mean_bytes_sent"),
        "bytes_recv":     hpc.get("mean_bytes_recv"),
        "soa_active":     hpc.get("soa_simd_active"),
    }


def parse_name(name: str) -> dict | None:
    m = re.match(r"^lm(\d+)_N(\d+)_P(\d+)_(p14_fused|p15_explicit)$", name)
    if not m:
        return None
    return {
        "leaf_max": int(m.group(1)),
        "n":        int(m.group(2)),
        "p":        int(m.group(3)),
        "variant":  m.group(4),
    }


def collect() -> dict:
    """Retorna {(leaf_max, n, p): {"p14_fused": row, "p15_explicit": row}}"""
    by_key = defaultdict(dict)
    if not RESULTS_DIR.exists():
        return by_key
    for d in RESULTS_DIR.iterdir():
        if not d.is_dir():
            continue
        meta = parse_name(d.name)
        if not meta:
            continue
        row = load_run(d)
        if row is None:
            print(f"  WARNING: sin timings.json en {d.name}", file=sys.stderr)
            continue
        k = (meta["leaf_max"], meta["n"], meta["p"])
        by_key[k][meta["variant"]] = row
    return by_key


def fmt(v, scale=1, decimals=2, unit=""):
    if v is None:
        return "—"
    return f"{v*scale:.{decimals}f}{unit}"


def speedup(a, b):
    if a and b and b > 0:
        return a / b
    return None


def main():
    data = collect()
    if not data:
        print("Sin resultados. Ejecuta run_phase15b.sh primero.")
        return

    # ── Tabla 1: batch size y rendimiento ─────────────────────────────────────
    print("\n" + "="*110)
    print("TABLA: Batch size y rendimiento por leaf_max")
    print("="*110)
    hdr = (f"{'Config':<26} {'leaf_max':>8} {'batch_avg':>10} "
           f"{'P14 wall(ms)':>14} {'P15 wall(ms)':>14} {'Wall sp':>8} "
           f"{'P14 LT(ms)':>12} {'P15 LT(ms)':>12} {'LT sp':>8}")
    print(hdr)
    print("-"*110)

    # Agrupar resultados para figuras
    fig_data = defaultdict(lambda: defaultdict(list))  # fig_data[(n,p)][leaf_max] = metrics

    keys_sorted = sorted(data.keys())
    for k in keys_sorted:
        leaf_max, n, p = k
        variants = data[k]
        p14 = variants.get("p14_fused")
        p15 = variants.get("p15_explicit")

        if p14 is None:
            continue

        batch = p14.get("batch_size_avg")
        wall14 = p14.get("wall_s")
        wall15 = p15.get("wall_s") if p15 else None
        lt14 = p14.get("lt_walk_s")
        lt15 = p15.get("lt_walk_s") if p15 else None
        wsp = speedup(wall14, wall15)
        lsp = speedup(lt14, lt15)

        label = f"N={n} P={p}"
        print(f"  {label:<24} {leaf_max:>8} {fmt(batch, decimals=1):>10} "
              f"{fmt(wall14, 1000):>14} {fmt(wall15, 1000):>14} {fmt(wsp, unit='×'):>8} "
              f"{fmt(lt14, 1000):>12} {fmt(lt15, 1000):>12} {fmt(lsp, unit='×'):>8}")

        fig_data[(n, p)][leaf_max] = {
            "batch_avg": batch, "wall14": wall14, "wall15": wall15,
            "lt14": lt14, "lt15": lt15, "wsp": wsp, "lsp": lsp,
            "let_imported": p14.get("let_imported"),
            "bytes_sent": p14.get("bytes_sent"),
        }

    print("-"*110)

    # ── Tabla 2: LET volume ────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("TABLA: Volumen LET vs leaf_max (P14, N=8000 P=2)")
    print("="*80)
    print(f"{'leaf_max':>10} {'LET imported':>14} {'bytes_sent':>14}")
    print("-"*40)
    for lm in LEAF_MAX_VALUES:
        k = (lm, 8000, 2)
        d = data.get(k, {}).get("p14_fused")
        if d:
            print(f"  {lm:>8} {fmt(d.get('let_imported'), decimals=0):>14} "
                  f"{fmt(d.get('bytes_sent'), 1/1024, decimals=0, unit='KB'):>14}")

    # ── Figuras ────────────────────────────────────────────────────────────────
    NP_COMBOS = [(8000, 2), (16000, 2), (8000, 4), (16000, 4)]
    COLORS_P14 = ["steelblue", "royalblue", "navy", "darkblue"]
    COLORS_P15 = ["darkorange", "orangered", "tomato", "firebrick"]

    # Fig 1: Batch size vs leaf_max
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, p_filter in zip(axes, [2, 4]):
        for n_filter in [8000, 16000]:
            key = (n_filter, p_filter)
            lm_vals = sorted(fig_data[key].keys())
            batches = [fig_data[key][lm].get("batch_avg") for lm in lm_vals]
            if any(b is not None for b in batches):
                ax.plot(lm_vals, [b or 0 for b in batches], marker="o",
                        label=f"N={n_filter}")

        ax.axhline(4, color="gray", linestyle="--", alpha=0.6, label="SIMD width (4)")
        ax.axhline(8, color="lightgray", linestyle=":", alpha=0.6, label="leaf_max default (8)")
        ax.set_xlabel("leaf_max")
        ax.set_ylabel("Batch size promedio (RMNs/llamada)")
        ax.set_title(f"Batch size vs leaf_max — P={p_filter}")
        ax.set_xticks(LEAF_MAX_VALUES)
        ax.legend()
        ax.grid(alpha=0.3)

    fig.tight_layout()
    out = PLOTS_DIR / "batch_size_vs_leaf_max.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\n  Saved: {out}")

    # Fig 2: Wall time vs leaf_max (barras P14 vs P15)
    for p_filter in [2, 4]:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, n_filter in zip(axes, [8000, 16000]):
            key = (n_filter, p_filter)
            lm_vals = sorted(fig_data[key].keys())
            wall14s = [(fig_data[key][lm].get("wall14") or 0)*1000 for lm in lm_vals]
            wall15s = [(fig_data[key][lm].get("wall15") or 0)*1000 for lm in lm_vals]

            x = np.arange(len(lm_vals))
            w = 0.35
            ax.bar(x - w/2, wall14s, w, label="P14 fused", color="steelblue")
            ax.bar(x + w/2, wall15s, w, label="P15 explicit AVX2", color="darkorange")
            ax.set_xlabel("leaf_max")
            ax.set_ylabel("Wall time/paso (ms)")
            ax.set_title(f"Wall time vs leaf_max — N={n_filter}, P={p_filter}")
            ax.set_xticks(x)
            ax.set_xticklabels([str(lm) for lm in lm_vals])
            ax.legend()
            ax.grid(axis="y", alpha=0.3)

        fig.tight_layout()
        out = PLOTS_DIR / f"wall_time_vs_leaf_max_P{p_filter}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  Saved: {out}")

    # Fig 3: Speedup P15/P14 vs leaf_max
    fig, ax = plt.subplots(figsize=(9, 5))
    for (n_filter, p_filter) in NP_COMBOS:
        key = (n_filter, p_filter)
        lm_vals = sorted(fig_data[key].keys())
        wsps = [fig_data[key][lm].get("wsp") for lm in lm_vals]
        if any(s is not None for s in wsps):
            ax.plot(lm_vals, [s or 0 for s in wsps], marker="o",
                    label=f"N={n_filter} P={p_filter}")

    ax.axhline(1.0, color="gray", linestyle="--", label="sin mejora (1.0×)")
    ax.set_xlabel("leaf_max")
    ax.set_ylabel("Speedup P15/P14 (> 1.0 = P15 gana)")
    ax.set_title("Speedup P15 vs P14 por leaf_max")
    ax.set_xticks(LEAF_MAX_VALUES)
    ax.legend()
    ax.grid(alpha=0.3)
    out = PLOTS_DIR / "speedup_p15_vs_p14.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")

    # Fig 4: LET nodes importados vs leaf_max
    fig, ax = plt.subplots(figsize=(8, 5))
    for (n_filter, p_filter) in [(8000, 2), (16000, 2)]:
        key = (n_filter, p_filter)
        lm_vals = sorted(fig_data[key].keys())
        lets = [fig_data[key][lm].get("let_imported") for lm in lm_vals]
        if any(l is not None for l in lets):
            ax.plot(lm_vals, [l or 0 for l in lets], marker="s",
                    label=f"N={n_filter} P={p_filter}")

    ax.set_xlabel("leaf_max")
    ax.set_ylabel("LET nodes importados (media)")
    ax.set_title("Volumen LET vs leaf_max")
    ax.set_xticks(LEAF_MAX_VALUES)
    ax.legend()
    ax.grid(alpha=0.3)
    out = PLOTS_DIR / "let_nodes_vs_leaf_max.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")

    print("\n=== Análisis completado ===")


if __name__ == "__main__":
    main()
