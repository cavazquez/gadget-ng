#!/usr/bin/env python3
"""
Phase 14 — Análisis de resultados SoA + SIMD.

Lee los timings.json generados por run_phase14.sh y produce:
  1. Tabla de top hotspots (breakdown por fase)
  2. Tabla de speedup por configuración (baseline vs soa_simd)
  3. Figuras PNG (wall time, speedup, breakdown)
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

# ── Carga de datos ─────────────────────────────────────────────────────────────

def load_timings(result_dir: Path) -> dict | None:
    tj = result_dir / "timings.json"
    if not tj.exists():
        return None
    with open(tj) as f:
        return json.load(f)

def parse_run_dir(name: str) -> dict:
    """Extrae group, N, P, variant del nombre del directorio."""
    m = re.match(r"^(.+)_N(\d+)_P(\d+)_(baseline|soa_simd)$", name)
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
        print(f"ERROR: {RESULTS_DIR} no existe. Ejecuta run_phase14.sh primero.", file=sys.stderr)
        return rows

    for d in sorted(RESULTS_DIR.iterdir()):
        if not d.is_dir():
            continue
        meta = parse_run_dir(d.name)
        if not meta:
            continue
        data = load_timings(d)
        if data is None:
            print(f"  WARN: no timings.json en {d.name}")
            continue

        hpc = data.get("hpc", {})
        total_s = data.get("total_wall_s", 0.0)
        steps   = data.get("num_steps", 1)
        wall_per_step = total_s / max(steps, 1)

        row = {
            **meta,
            "wall_per_step_s": wall_per_step,
            "mean_let_tree_walk_s":  hpc.get("mean_let_tree_walk_s", 0.0),
            "mean_walk_local_s":     hpc.get("mean_walk_local_s", 0.0),
            "mean_apply_let_s":      hpc.get("mean_apply_let_s", 0.0),
            "mean_let_tree_build_s": hpc.get("mean_let_tree_build_s", 0.0),
            "mean_let_alltoallv_s":  hpc.get("mean_let_alltoallv_s", 0.0),
            "mean_tree_build_s":     hpc.get("mean_tree_build_s", 0.0),
            "mean_apply_leaf_s":     hpc.get("mean_apply_leaf_s", 0.0),
            "mean_apply_leaf_rmn":   hpc.get("mean_apply_leaf_rmn_count", 0.0),
            "mean_apply_leaf_calls": hpc.get("mean_apply_leaf_calls", 0.0),
            "mean_rmn_soa_pack_s":   hpc.get("mean_rmn_soa_pack_s", 0.0),
            "mean_accel_from_let_soa_s": hpc.get("mean_accel_from_let_soa_s", 0.0),
            "soa_simd_active":       hpc.get("soa_simd_active", False),
            "let_tree_parallel":     hpc.get("let_tree_parallel", False),
        }
        rows.append(row)
    return rows

# ── Tabla top hotspots ─────────────────────────────────────────────────────────

def print_hotspots(rows: list[dict]):
    print("\n" + "="*80)
    print("TOP HOTSPOTS — Breakdown por fase (P=1, profiling)")
    print("="*80)
    print(f"{'Config':<35} {'Variant':<12} {'WallStep':>9} {'WalkLocal':>10} {'LTWalk':>8} {'LTBuild':>8} {'LeafCalls':>10} {'RMN/leaf':>9}")
    print("-"*100)

    p1_rows = sorted(
        [r for r in rows if r["p"] == 1 and r["group"] == "profiling_p1"],
        key=lambda r: (r["n"], r["variant"])
    )
    for r in p1_rows:
        cfg_name = f"N={r['n']}"
        calls = r["mean_apply_leaf_calls"]
        rmn_count = r["mean_apply_leaf_rmn"]
        avg_rmn = rmn_count / calls if calls > 0 else 0
        print(f"{cfg_name:<35} {r['variant']:<12} "
              f"{r['wall_per_step_s']*1000:>8.1f}ms "
              f"{r['mean_walk_local_s']*1000:>9.1f}ms "
              f"{r['mean_let_tree_walk_s']*1000:>7.1f}ms "
              f"{r['mean_let_tree_build_s']*1000:>7.1f}ms "
              f"{calls:>10.0f} "
              f"{avg_rmn:>9.1f}")

# ── Tabla speedup ─────────────────────────────────────────────────────────────

def build_speedup_table(rows: list[dict]) -> list[dict]:
    by_key = {}
    for r in rows:
        key = (r["group"], r["n"], r["p"])
        by_key.setdefault(key, {})[r["variant"]] = r

    speedups = []
    for key, variants in sorted(by_key.items()):
        if "baseline" not in variants or "soa_simd" not in variants:
            continue
        bl  = variants["baseline"]
        soa = variants["soa_simd"]
        w_bl  = bl["wall_per_step_s"]
        w_soa = soa["wall_per_step_s"]
        ltw_speedup = (bl["mean_let_tree_walk_s"] / soa["mean_let_tree_walk_s"]
                       if soa["mean_let_tree_walk_s"] > 0 else 0.0)
        speedups.append({
            "group": key[0], "n": key[1], "p": key[2],
            "wall_baseline_ms": w_bl * 1000,
            "wall_soa_ms":      w_soa * 1000,
            "speedup_total":    w_bl / w_soa if w_soa > 0 else 0.0,
            "speedup_ltw":      ltw_speedup,
            "ltw_baseline_ms":  bl["mean_let_tree_walk_s"] * 1000,
            "ltw_soa_ms":       soa["mean_let_tree_walk_s"] * 1000,
        })
    return speedups

def print_speedup_table(rows: list[dict]):
    speedups = build_speedup_table(rows)
    if not speedups:
        print("No hay suficientes datos para tabla de speedup.")
        return

    print("\n" + "="*80)
    print("SPEEDUP: baseline vs SoA+SIMD")
    print("="*80)
    print(f"{'Group':<20} {'N':>6} {'P':>3} {'Baseline':>11} {'SoA+SIMD':>10} {'Speedup':>8} {'LTW-BL':>9} {'LTW-SOA':>9} {'SpeedupLTW':>11}")
    print("-"*90)
    for s in speedups:
        print(f"{s['group']:<20} {s['n']:>6} {s['p']:>3} "
              f"{s['wall_baseline_ms']:>10.1f}ms {s['wall_soa_ms']:>9.1f}ms "
              f"{s['speedup_total']:>7.2f}x "
              f"{s['ltw_baseline_ms']:>8.1f}ms "
              f"{s['ltw_soa_ms']:>8.1f}ms "
              f"{s['speedup_ltw']:>10.2f}x")

# ── Figuras ───────────────────────────────────────────────────────────────────

def plot_speedup(rows: list[dict]):
    speedups = build_speedup_table(rows)
    if not speedups:
        return

    # Wall time vs N para P=1 (profiling group)
    p1 = [s for s in speedups if s["p"] == 1 and s["group"] == "profiling_p1"]
    if p1:
        ns = [s["n"] for s in p1]
        bl = [s["wall_baseline_ms"] for s in p1]
        so = [s["wall_soa_ms"] for s in p1]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        ax = axes[0]
        ax.plot(ns, bl, "o--", label="Baseline (AoS)", color="tab:blue")
        ax.plot(ns, so, "s-",  label="SoA+SIMD",       color="tab:orange")
        ax.set_xlabel("N (partículas)")
        ax.set_ylabel("Wall time / paso (ms)")
        ax.set_title("Wall time vs N (P=1)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        sp = [s["speedup_total"] for s in p1]
        ax.bar(range(len(ns)), sp, color="tab:green")
        ax.set_xticks(range(len(ns)))
        ax.set_xticklabels([str(n) for n in ns])
        ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("N (partículas)")
        ax.set_ylabel("Speedup (baseline / SoA+SIMD)")
        ax.set_title("Speedup total (P=1)")
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "fig1_wall_speedup_p1.png", dpi=150)
        plt.close(fig)
        print(f"  Figura guardada: {PLOTS_DIR}/fig1_wall_speedup_p1.png")

    # Speedup vs P (MPI scaling)
    mpi = [s for s in speedups if s["group"] == "scaling_mpi"]
    if mpi:
        ns_unique = sorted(set(s["n"] for s in mpi))
        fig, ax = plt.subplots(figsize=(8, 5))
        for n in ns_unique:
            sub = sorted([s for s in mpi if s["n"] == n], key=lambda s: s["p"])
            ps = [s["p"] for s in sub]
            sp = [s["speedup_total"] for s in sub]
            ax.plot(ps, sp, "o-", label=f"N={n}")
        ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("P (ranks MPI)")
        ax.set_ylabel("Speedup SoA+SIMD / Baseline")
        ax.set_title("Speedup SoA+SIMD vs P (MPI)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "fig2_speedup_vs_p_mpi.png", dpi=150)
        plt.close(fig)
        print(f"  Figura guardada: {PLOTS_DIR}/fig2_speedup_vs_p_mpi.png")

    # Breakdown por fase (P=1, N=8000)
    p1_8k = [r for r in rows if r["p"] == 1 and r["n"] == 8000 and r["group"] == "profiling_p1"]
    if len(p1_8k) >= 2:
        variants_map = {r["variant"]: r for r in p1_8k}
        if "baseline" in variants_map and "soa_simd" in variants_map:
            bl  = variants_map["baseline"]
            soa = variants_map["soa_simd"]

            phases_bl  = [
                bl["mean_tree_build_s"] * 1000,
                bl["mean_walk_local_s"] * 1000,
                bl["mean_let_tree_build_s"] * 1000,
                bl["mean_let_tree_walk_s"] * 1000,
            ]
            phases_soa = [
                soa["mean_tree_build_s"] * 1000,
                soa["mean_walk_local_s"] * 1000,
                soa["mean_let_tree_build_s"] * 1000,
                soa["mean_let_tree_walk_s"] * 1000,
            ]
            labels = ["Tree\nBuild", "Walk\nLocal", "LT\nBuild", "LT\nWalk"]

            x = np.arange(len(labels))
            w = 0.35
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.bar(x - w/2, phases_bl,  w, label="Baseline (AoS)", color="tab:blue")
            ax.bar(x + w/2, phases_soa, w, label="SoA+SIMD",       color="tab:orange")
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.set_ylabel("Tiempo por paso (ms)")
            ax.set_title("Breakdown de fases — N=8000, P=1")
            ax.legend()
            ax.grid(True, axis="y", alpha=0.3)
            fig.tight_layout()
            fig.savefig(PLOTS_DIR / "fig3_phase_breakdown_N8000_P1.png", dpi=150)
            plt.close(fig)
            print(f"  Figura guardada: {PLOTS_DIR}/fig3_phase_breakdown_N8000_P1.png")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    rows = collect_results()
    if not rows:
        print("No hay resultados. Ejecuta run_phase14.sh primero.")
        return

    print(f"Loaded {len(rows)} run(s) from {RESULTS_DIR}/")

    print_hotspots(rows)
    print_speedup_table(rows)

    print("\n--- Generando figuras ---")
    plot_speedup(rows)

    # Resumen en JSON para el reporte
    speedups = build_speedup_table(rows)
    summary = {
        "n_runs": len(rows),
        "speedups": speedups,
    }
    with open(RESULTS_DIR / "summary_phase14.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResumen guardado en: {RESULTS_DIR}/summary_phase14.json")

if __name__ == "__main__":
    main()
