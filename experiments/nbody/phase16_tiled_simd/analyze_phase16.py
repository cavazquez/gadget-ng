#!/usr/bin/env python3
"""
Phase 16 — Análisis de benchmarks tileados 4×N_i vs P14 vs P15.

Lee timings.json de results/{p14,p15,p16}/bench_N*_P*/ y produce:
  1. Tabla comparativa de wall time y let_tree_walk_ns
  2. Tabla de utilización de tiles (tile_utilization_ratio, batch_size_avg)
  3. Speedups P15/P14 y P16/P14
  4. Figuras en plots/
"""

import json
import os
import sys
from pathlib import Path
import re

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

PHASE_DIR = Path(__file__).parent
RESULTS_DIR = PHASE_DIR / "results"
PLOTS_DIR = PHASE_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# ── Carga de datos ────────────────────────────────────────────────────────────

def load_timings(variant: str, cfg_base: str) -> dict | None:
    p = RESULTS_DIR / variant / cfg_base / "timings.json"
    if not p.exists():
        return None
    with open(p) as f:
        data = json.load(f)
    # Normalizar: puede estar en top-level o bajo "hpc"
    if "hpc" in data:
        data = {**data, **data["hpc"]}
    return data

def extract_n_p(cfg_base: str):
    m = re.search(r'N(\d+)_P(\d+)', cfg_base)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None

# ── Métricas ──────────────────────────────────────────────────────────────────

def get_wall_s(d: dict) -> float:
    return d.get("total_wall_s", d.get("wall_s", float("nan")))

def get_ltw_s(d: dict) -> float:
    return d.get("mean_let_tree_walk_s", float("nan"))

def get_tile_util(d: dict) -> float:
    return d.get("tile_utilization_ratio", float("nan"))

def get_tile_calls(d: dict) -> float:
    return d.get("mean_apply_leaf_tile_calls", float("nan"))

def get_tile_i(d: dict) -> float:
    return d.get("mean_apply_leaf_tile_i_count", float("nan"))

def get_leaf_rmn(d: dict) -> float:
    return d.get("mean_apply_leaf_rmn_count", float("nan"))

def get_leaf_calls(d: dict) -> float:
    return d.get("mean_apply_leaf_calls", float("nan"))

def batch_size_avg(d: dict) -> float:
    calls = get_leaf_calls(d)
    rmn = get_leaf_rmn(d)
    if calls and calls > 0:
        return rmn / calls
    # P16: usar tile counters
    t_calls = get_tile_calls(d)
    t_i = get_tile_i(d)
    if t_calls and t_calls > 0:
        return t_i / t_calls  # media de RMNs por leaf (aproximado con tile_i)
    return float("nan")

# ── Carga de todos los resultados ─────────────────────────────────────────────

configs = []
for f in sorted((RESULTS_DIR / "p16").iterdir()) if (RESULTS_DIR / "p16").exists() else []:
    if f.is_dir():
        configs.append(f.name)

if not configs:
    # Intentar descubrir desde p14
    base_dir = RESULTS_DIR / "p14"
    if base_dir.exists():
        configs = [f.name for f in sorted(base_dir.iterdir()) if f.is_dir()]

if not configs:
    print("ERROR: No results found. Run run_phase16.sh first.")
    sys.exit(1)

rows = []
for cfg in configs:
    n, p_mpi = extract_n_p(cfg)
    d14 = load_timings("p14", cfg)
    d15 = load_timings("p15", cfg)
    d16 = load_timings("p16", cfg)

    if d14 is None and d15 is None and d16 is None:
        continue

    wall14 = get_wall_s(d14) if d14 else float("nan")
    wall15 = get_wall_s(d15) if d15 else float("nan")
    wall16 = get_wall_s(d16) if d16 else float("nan")
    ltw14  = get_ltw_s(d14) if d14 else float("nan")
    ltw15  = get_ltw_s(d15) if d15 else float("nan")
    ltw16  = get_ltw_s(d16) if d16 else float("nan")

    sp15_wall = wall14 / wall15 if wall15 > 0 else float("nan")
    sp16_wall = wall14 / wall16 if wall16 > 0 else float("nan")
    sp15_ltw  = ltw14  / ltw15  if ltw15  > 0 else float("nan")
    sp16_ltw  = ltw14  / ltw16  if ltw16  > 0 else float("nan")

    tile_util = get_tile_util(d16) if d16 else float("nan")
    tile_calls_p16 = get_tile_calls(d16) if d16 else float("nan")
    tile_i_p16 = get_tile_i(d16) if d16 else float("nan")

    rows.append({
        "cfg": cfg, "N": n, "P": p_mpi,
        "wall_p14": wall14, "wall_p15": wall15, "wall_p16": wall16,
        "ltw_p14":  ltw14,  "ltw_p15":  ltw15,  "ltw_p16":  ltw16,
        "sp15_wall": sp15_wall, "sp16_wall": sp16_wall,
        "sp15_ltw":  sp15_ltw,  "sp16_ltw":  sp16_ltw,
        "tile_util": tile_util,
        "tile_calls_p16": tile_calls_p16,
        "tile_i_p16": tile_i_p16,
    })

if not rows:
    print("ERROR: No timing data loaded.")
    sys.exit(1)

# ── Tabla 1: Wall time y speedups ─────────────────────────────────────────────

print("\n" + "="*85)
print("TABLA 1: Wall time total y speedups vs P14 (fused kernel)")
print("="*85)
print(f"{'Config':<30} {'N':>6} {'P':>2} | {'wall_P14':>9} {'wall_P15':>9} {'wall_P16':>9} | {'sp P15':>7} {'sp P16':>7}")
print("-"*85)
for r in rows:
    print(f"{r['cfg']:<30} {r['N']:>6} {r['P']:>2} | "
          f"{r['wall_p14']:9.3f} {r['wall_p15']:9.3f} {r['wall_p16']:9.3f} | "
          f"{r['sp15_wall']:7.3f}x {r['sp16_wall']:7.3f}x")

# ── Tabla 2: LetTree walk time ─────────────────────────────────────────────────

print("\n" + "="*85)
print("TABLA 2: LetTree walk time y speedups (let_tree_walk_ns)")
print("="*85)
print(f"{'Config':<30} | {'ltw_P14 (s)':>12} {'ltw_P15 (s)':>12} {'ltw_P16 (s)':>12} | {'sp P15':>7} {'sp P16':>7}")
print("-"*85)
for r in rows:
    print(f"{r['cfg']:<30} | "
          f"{r['ltw_p14']:12.4f} {r['ltw_p15']:12.4f} {r['ltw_p16']:12.4f} | "
          f"{r['sp15_ltw']:7.3f}x {r['sp16_ltw']:7.3f}x")

# ── Tabla 3: Utilización de tiles ─────────────────────────────────────────────

print("\n" + "="*75)
print("TABLA 3: Métricas de tiles P16 (tile_utilization_ratio)")
print("="*75)
print(f"{'Config':<30} | {'tile_calls/step':>14} {'tile_i/step':>12} {'util_ratio':>10}")
print("-"*75)
for r in rows:
    print(f"{r['cfg']:<30} | "
          f"{r['tile_calls_p16']:14.1f} {r['tile_i_p16']:12.1f} {r['tile_util']:10.4f}")

# ── Figuras ───────────────────────────────────────────────────────────────────

if HAS_MPL and rows:
    labels = [f"N={r['N']}\nP={r['P']}" for r in rows]
    x = range(len(rows))
    w = 0.28

    # Fig 1: Wall time comparativo
    fig, ax = plt.subplots(figsize=(10, 5))
    bars14 = ax.bar([i - w for i in x], [r["wall_p14"] for r in rows], w, label="P14 fused", color="#4878cf")
    bars15 = ax.bar([i       for i in x], [r["wall_p15"] for r in rows], w, label="P15 explicit (1xi)", color="#6acc65")
    bars16 = ax.bar([i + w for i in x], [r["wall_p16"] for r in rows], w, label="P16 tiled (4xi)", color="#d65f5f")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Wall time (s)")
    ax.set_title("Phase 16 — Wall time: P14 vs P15 vs P16")
    ax.legend()
    ax.grid(axis="y", alpha=0.4)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "p16_wall_time.png", dpi=150)
    plt.close()

    # Fig 2: LetTree walk time
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([i - w for i in x], [r["ltw_p14"] for r in rows], w, label="P14 fused", color="#4878cf")
    ax.bar([i       for i in x], [r["ltw_p15"] for r in rows], w, label="P15 explicit (1xi)", color="#6acc65")
    ax.bar([i + w for i in x], [r["ltw_p16"] for r in rows], w, label="P16 tiled (4xi)", color="#d65f5f")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("LetTree walk time (s/paso)")
    ax.set_title("Phase 16 — LetTree walk: P14 vs P15 vs P16")
    ax.legend()
    ax.grid(axis="y", alpha=0.4)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "p16_let_tree_walk.png", dpi=150)
    plt.close()

    # Fig 3: Speedup vs P14
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(labels, [r["sp15_wall"] for r in rows], "o-", label="P15/P14 wall", color="#6acc65")
    ax.plot(labels, [r["sp16_wall"] for r in rows], "s-", label="P16/P14 wall", color="#d65f5f")
    ax.plot(labels, [r["sp15_ltw"] for r in rows], "o--", label="P15/P14 LTW", color="#6acc65", alpha=0.6)
    ax.plot(labels, [r["sp16_ltw"] for r in rows], "s--", label="P16/P14 LTW", color="#d65f5f", alpha=0.6)
    ax.axhline(1.0, color="black", linestyle=":", linewidth=0.8)
    ax.set_ylabel("Speedup vs P14")
    ax.set_title("Phase 16 — Speedup vs P14 baseline")
    ax.legend()
    ax.grid(alpha=0.4)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "p16_speedup.png", dpi=150)
    plt.close()

    # Fig 4: Tile utilization
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, [r["tile_util"] for r in rows], color="#d65f5f", alpha=0.85)
    ax.axhline(1.0, color="green", linestyle="--", linewidth=1.2, label="ideal (1.0)")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Tile utilization ratio (tile_i / (calls×4))")
    ax.set_title("Phase 16 — Tile utilization ratio (P16)")
    ax.legend()
    ax.grid(axis="y", alpha=0.4)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "p16_tile_utilization.png", dpi=150)
    plt.close()

    print(f"\nFiguras guardadas en: {PLOTS_DIR}")
else:
    if not HAS_MPL:
        print("\nNOTA: matplotlib no disponible, figuras omitidas.")

print("\n=== Análisis completo ===")
