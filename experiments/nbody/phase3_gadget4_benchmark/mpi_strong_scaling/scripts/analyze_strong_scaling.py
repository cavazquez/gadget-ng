#!/usr/bin/env python3
"""analyze_strong_scaling.py — Calcula speedup y efficiency del strong scaling.

Lee strong_timing_raw.csv (generado por run_strong_scaling.sh) y produce:
    results/strong_scaling.csv  — tabla con speedup, efficiency, overhead
    plots/strong_speedup.png    — speedup vs ranks
    plots/strong_efficiency.png — efficiency vs ranks
    plots/strong_breakdown.png  — desglose comm/gravity/integration vs ranks
"""

import sys
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR  = Path(__file__).parent
EXP_DIR     = SCRIPT_DIR.parent
RESULTS_DIR = EXP_DIR / "results"
PLOTS_DIR   = EXP_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

RAW_CSV = RESULTS_DIR / "strong_timing_raw.csv"

if not RAW_CSV.exists():
    print(f"ERROR: {RAW_CSV} no encontrado. Ejecuta run_strong_scaling.sh primero.")
    sys.exit(1)

df = pd.read_csv(RAW_CSV)
df = df.sort_values("rank").reset_index(drop=True)
print(f"Datos cargados:\n{df.to_string()}")

# Calcular speedup y efficiency usando T(1) como referencia.
t1 = df[df["rank"] == 1]["total_wall_s"].values
if len(t1) == 0:
    print("ERROR: No hay datos para 1 rank. No se puede calcular speedup.")
    sys.exit(1)

T1 = t1[0]
df["speedup"]    = T1 / df["total_wall_s"]
df["efficiency"] = df["speedup"] / df["rank"] * 100   # %
df["serial_frac"] = df.apply(  # Amdahl: f = (1/speedup - 1/ranks) / (1 - 1/ranks)
    lambda r: (1/r["speedup"] - 1/r["rank"]) / (1 - 1/r["rank"]) if r["rank"] > 1 else 0.0,
    axis=1
)

df.to_csv(RESULTS_DIR / "strong_scaling.csv", index=False)
print(f"\nCSV guardado en {RESULTS_DIR}/strong_scaling.csv")
print("\n=== Tabla ===")
print(df[["rank", "total_wall_s", "mean_step_ms", "speedup", "efficiency", "comm_frac"]].to_string(index=False))

# ── Plot 1: Speedup ───────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(df["rank"], df["speedup"], "o-", color="#2563EB", linewidth=2.5, markersize=8,
        label="gadget-ng (medido)")
ranks_ideal = np.array(sorted(df["rank"].unique()))
ax.plot(ranks_ideal, ranks_ideal, "--", color="black", alpha=0.5, linewidth=1.5, label="Ideal (linear)")

# Amdahl con fracción serial observada (media sobre todos los multi-rank).
serial_fracs = df[df["rank"] > 1]["serial_frac"].dropna()
if len(serial_fracs) > 0:
    f_serial = serial_fracs.mean()
    r_amdahl = np.linspace(1, df["rank"].max(), 100)
    speedup_amdahl = 1.0 / (f_serial + (1 - f_serial) / r_amdahl)
    ax.plot(r_amdahl, speedup_amdahl, ":", color="#DC2626", linewidth=1.5,
            label=f"Amdahl (f_serial={f_serial:.2f})")

ax.set_xlabel("Número de ranks MPI", fontsize=12)
ax.set_ylabel("Speedup", fontsize=12)
ax.set_title(f"Strong Scaling: N=1000, BH θ=0.5\n(gadget-ng Allgatherv)", fontsize=12)
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(PLOTS_DIR / "strong_speedup.png", dpi=150)
plt.close()
print("  → strong_speedup.png")

# ── Plot 2: Efficiency ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(df["rank"], df["efficiency"], "s-", color="#059669", linewidth=2.5, markersize=8)
ax.axhline(100, linestyle="--", color="black", alpha=0.5, linewidth=1.5, label="Ideal (100%)")
ax.axhline(75,  linestyle=":",  color="#DC2626", alpha=0.5, linewidth=1.5, label="75% umbral")
ax.set_xlabel("Número de ranks MPI", fontsize=12)
ax.set_ylabel("Efficiency (%)", fontsize=12)
ax.set_ylim(0, 110)
ax.set_title(f"Strong Scaling Efficiency: N=1000\n(efficiency = speedup / ranks)", fontsize=12)
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(PLOTS_DIR / "strong_efficiency.png", dpi=150)
plt.close()
print("  → strong_efficiency.png")

# ── Plot 3: Desglose comm/gravity ─────────────────────────────────────────────
if "comm_frac" in df.columns and "gravity_frac" in df.columns:
    fig, ax = plt.subplots(figsize=(7, 5))
    width = 0.35
    x = np.arange(len(df))
    bars1 = ax.bar(x - width/2, df["gravity_frac"]*100, width, label="Fuerzas", color="#2563EB")
    bars2 = ax.bar(x + width/2, df["comm_frac"]*100, width, label="Comunicación MPI", color="#DC2626")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r} rank{'s' if r>1 else ''}" for r in df["rank"]])
    ax.set_ylabel("Fracción del tiempo (%)", fontsize=12)
    ax.set_title("Desglose temporal: Fuerzas vs Comunicación", fontsize=12)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "strong_breakdown.png", dpi=150)
    plt.close()
    print("  → strong_breakdown.png")

print(f"\nPlots guardados en {PLOTS_DIR}/")
