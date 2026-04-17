#!/usr/bin/env python3
"""analyze_weak_scaling.py — Analiza weak scaling y genera plots.

Lee weak_timing_raw.csv y genera:
    results/weak_scaling.csv
    plots/weak_efficiency.png
    plots/weak_time_vs_n.png
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR  = Path(__file__).parent
EXP_DIR     = SCRIPT_DIR.parent
RESULTS_DIR = EXP_DIR / "results"
PLOTS_DIR   = EXP_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

RAW_CSV = RESULTS_DIR / "weak_timing_raw.csv"

if not RAW_CSV.exists():
    print(f"ERROR: {RAW_CSV} no encontrado. Ejecuta run_weak_scaling.sh primero.")
    sys.exit(1)

df = pd.read_csv(RAW_CSV).sort_values("N").reset_index(drop=True)
print(f"Datos de weak scaling:\n{df.to_string()}")

df.to_csv(RESULTS_DIR / "weak_scaling.csv", index=False)

# ── Plot 1: Weak efficiency vs ranks ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
x_col = "rank" if "rank" in df.columns else "N"
ax.plot(df[x_col], df["weak_efficiency"], "o-", color="#7C3AED", linewidth=2.5, markersize=8,
        label="gadget-ng (medido)")
ax.axhline(100, linestyle="--", color="black", alpha=0.5, linewidth=1.5, label="Ideal (100%)")
ax.axhline(75,  linestyle=":",  color="#DC2626", alpha=0.5, linewidth=1.5, label="75% umbral")
ax.set_xlabel("Ranks MPI" if x_col == "rank" else "N total", fontsize=12)
ax.set_ylabel("Weak Efficiency (%)", fontsize=12)
ax.set_ylim(0, 110)
ax.set_title(f"Weak Scaling Efficiency\n(N ∝ ranks, {df['N'].iloc[0]//df['rank'].iloc[0] if 'rank' in df.columns else ''} partículas/rank)", fontsize=12)
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(PLOTS_DIR / "weak_efficiency.png", dpi=150)
plt.close()
print("  → weak_efficiency.png")

# ── Plot 2: Wall time vs N ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
ax.semilogx(df["N"], df["total_wall_s"], "s-", color="#059669", linewidth=2.5, markersize=8,
            label="gadget-ng (medido)")
t1 = df["total_wall_s"].iloc[0]
ax.axhline(t1, linestyle="--", color="black", alpha=0.5, linewidth=1.5,
           label=f"Ideal (constante = {t1:.2f}s)")
ax.set_xlabel("N total", fontsize=12)
ax.set_ylabel("Wall time total (s)", fontsize=12)
ax.set_title("Weak Scaling: Wall time vs N\n(ideal: constante)", fontsize=12)
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(PLOTS_DIR / "weak_time_vs_n.png", dpi=150)
plt.close()
print("  → weak_time_vs_n.png")

# ── Plot 3: Desglose comm vs N ────────────────────────────────────────────────
if "comm_frac" in df.columns:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogx(df["N"], df["comm_frac"]*100, "^--", color="#DC2626", linewidth=2, markersize=8)
    ax.set_xlabel("N total", fontsize=12)
    ax.set_ylabel("Fracción comunicación MPI (%)", fontsize=12)
    ax.set_title("Overhead de comunicación vs N\n(crece con Allgatherv O(N))", fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "weak_comm_frac.png", dpi=150)
    plt.close()
    print("  → weak_comm_frac.png")

print(f"\nPlots guardados en {PLOTS_DIR}/")
print("\n=== Resumen ===")
print(df[["rank" if "rank" in df.columns else "N", "N", "total_wall_s", "weak_efficiency", "comm_frac"]].to_string(index=False))
