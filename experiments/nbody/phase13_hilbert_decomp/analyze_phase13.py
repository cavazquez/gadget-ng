#!/usr/bin/env python3
"""
Phase 13 — Análisis Morton vs Hilbert
Parsea timings.json de cada run y genera:
  - phase13_summary.csv
  - Plots: wall_time, bytes, imbalance, let_nodes (Morton vs Hilbert)
"""

import os
import json
import csv
import sys
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False
    print("WARNING: matplotlib no disponible, no se generarán plots")

SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results"

# ── Parsear resultados ────────────────────────────────────────────────────────

rows = []
for run_dir in sorted(RESULTS_DIR.iterdir()):
    if not run_dir.is_dir():
        continue
    timings_path = run_dir / "timings.json"
    if not timings_path.exists():
        continue

    with open(timings_path) as f:
        data = json.load(f)

    name = run_dir.name

    # Parsear nombre: {group}_N{n}_P{p}_{sfc_kind}
    parts = name.split("_")
    group = parts[0]
    n = int(next(p[1:] for p in parts if p.startswith("N")))
    p = int(next(p[1:] for p in parts if p.startswith("P")))
    kind = parts[-1]

    hpc = data.get("hpc", {})

    row = {
        "name":             name,
        "group":            group,
        "n":                n,
        "p":                p,
        "sfc_kind":         kind,
        "steps":            data.get("steps", 0),
        "total_wall_s":     data.get("total_wall_s", 0),
        "mean_step_wall_s": data.get("mean_step_wall_s", 0),
        "comm_fraction":    data.get("comm_fraction", 0),
        # HPC metrics
        "mean_bytes_sent":          hpc.get("mean_bytes_sent", 0),
        "mean_bytes_recv":          hpc.get("mean_bytes_recv", 0),
        "mean_let_nodes_exported":  hpc.get("mean_let_nodes_exported", 0),
        "mean_let_nodes_imported":  hpc.get("mean_let_nodes_imported", 0),
        "mean_max_let_nodes_per_rank": hpc.get("mean_max_let_nodes_per_rank", 0),
        "mean_local_tree_nodes":    hpc.get("mean_local_tree_nodes", 0),
        "mean_export_prune_ratio":  hpc.get("mean_export_prune_ratio", 0),
        "mean_local_particle_count": hpc.get("mean_local_particle_count", 0),
        "particle_imbalance_ratio": hpc.get("particle_imbalance_ratio", 1),
        "mean_domain_rebalance_s":  hpc.get("mean_domain_rebalance_s", 0),
        "mean_domain_migration_s":  hpc.get("mean_domain_migration_s", 0),
        "mean_let_tree_build_s":    hpc.get("mean_let_tree_build_s", 0),
        "mean_let_tree_walk_s":     hpc.get("mean_let_tree_walk_s", 0),
        "wait_fraction":            hpc.get("wait_fraction", 0),
    }
    rows.append(row)

print(f"Encontrados {len(rows)} resultados")

if not rows:
    print("No hay resultados para analizar")
    sys.exit(0)

# ── CSV ───────────────────────────────────────────────────────────────────────

csv_path = SCRIPT_DIR / "phase13_summary.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
print(f"Guardado: {csv_path}")

# ── Análisis comparativo Morton vs Hilbert ────────────────────────────────────

print("\n=== RESUMEN COMPARATIVO Morton vs Hilbert (grupo: scaling) ===")
print(f"{'N':>6} {'P':>3} {'Curva':>8} {'Wall(s)':>9} {'Bytes/rank':>12} "
      f"{'LET_exp':>10} {'Imbalance':>10} {'PruneRatio':>11}")
print("-" * 75)

for r in sorted(rows, key=lambda x: (x["group"], x["n"], x["p"], x["sfc_kind"])):
    if r["group"] != "scaling":
        continue
    print(f"{r['n']:>6} {r['p']:>3} {r['sfc_kind']:>8} "
          f"{r['mean_step_wall_s']:>9.4f} "
          f"{r['mean_bytes_sent']:>12.0f} "
          f"{r['mean_let_nodes_exported']:>10.0f} "
          f"{r['particle_imbalance_ratio']:>10.3f} "
          f"{r['mean_export_prune_ratio']:>11.4f}")

# ── Plots ─────────────────────────────────────────────────────────────────────

if not HAVE_MPL:
    sys.exit(0)

def plot_comparison(rows, group, metric, ylabel, filename, log=False):
    """Genera un plot Morton vs Hilbert de una métrica."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    ps = [2, 4, 8]

    for ax, p_val in zip(axes, ps):
        data_m = sorted(
            [(r["n"], r[metric]) for r in rows if r["group"] == group
             and r["p"] == p_val and r["sfc_kind"] == "morton"],
            key=lambda x: x[0]
        )
        data_h = sorted(
            [(r["n"], r[metric]) for r in rows if r["group"] == group
             and r["p"] == p_val and r["sfc_kind"] == "hilbert"],
            key=lambda x: x[0]
        )
        if data_m:
            ns, vals = zip(*data_m)
            ax.plot(ns, vals, "o-", label="Morton", color="steelblue")
        if data_h:
            ns, vals = zip(*data_h)
            ax.plot(ns, vals, "s--", label="Hilbert", color="darkorange")
        ax.set_title(f"P = {p_val}")
        ax.set_xlabel("N partículas")
        ax.set_ylabel(ylabel)
        if log:
            ax.set_yscale("log")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Morton vs Hilbert — {ylabel}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = SCRIPT_DIR / filename
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Plot: {path}")

plot_comparison(rows, "scaling", "mean_step_wall_s",
    "Wall time/step (s)", "plot_wall_morton_vs_hilbert.svg")
plot_comparison(rows, "scaling", "mean_bytes_sent",
    "Bytes enviados/paso", "plot_bytes_morton_vs_hilbert.svg", log=True)
plot_comparison(rows, "scaling", "particle_imbalance_ratio",
    "Particle imbalance ratio", "plot_imbalance_morton_vs_hilbert.svg")
plot_comparison(rows, "scaling", "mean_let_nodes_exported",
    "LET nodes exportados/paso", "plot_let_nodes_morton_vs_hilbert.svg", log=True)

# Plot de imbalance vs P para N=16000 (sensitivity)
fig, ax = plt.subplots(figsize=(8, 5))
for kind, color, marker in [("morton", "steelblue", "o"), ("hilbert", "darkorange", "s")]:
    data = sorted(
        [(r["p"], r["particle_imbalance_ratio"]) for r in rows
         if r["group"] == "sensitivity_p" and r["sfc_kind"] == kind],
        key=lambda x: x[0]
    )
    if data:
        ps, vals = zip(*data)
        ax.plot(ps, vals, f"{marker}-", label=kind.capitalize(), color=color)
ax.set_xlabel("P (ranks)")
ax.set_ylabel("Particle imbalance ratio")
ax.set_title("Imbalance de partículas vs P (N=16000)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
path = SCRIPT_DIR / "plot_imbalance_vs_p_sensitivity.svg"
plt.savefig(path, dpi=120)
plt.close()
print(f"Plot: {path}")

print("\n=== Análisis Phase 13 completo ===")
