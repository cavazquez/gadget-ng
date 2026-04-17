#!/usr/bin/env python3
"""plot_scaling_n.py — Genera plots de tiempo vs N para Direct y Barnes-Hut.

Genera:
    plots/scaling_n_wall_time.png    — wall time por paso vs N (log-log)
    plots/scaling_n_complexity.png   — ajuste de complejidad O(N²) vs O(N log N)
    plots/scaling_n_speedup.png      — speedup BH/Direct vs N
    plots/scaling_n_gravity_frac.png — fracción del tiempo en fuerzas vs N
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit

SCRIPT_DIR  = Path(__file__).parent
EXP_DIR     = SCRIPT_DIR.parent
CSV_PATH    = EXP_DIR / "results" / "scaling_n.csv"
PLOTS_DIR   = EXP_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

if not CSV_PATH.exists():
    print(f"ERROR: CSV no encontrado en {CSV_PATH}")
    print("Ejecuta primero: python3 scripts/run_scaling_n.py")
    sys.exit(1)

df = pd.read_csv(CSV_PATH)
print(f"Datos cargados: {len(df)} filas")
print(df[["solver", "N", "mean_step_wall_s"]].to_string())

direct = df[df["solver"] == "direct"].sort_values("N")
bh     = df[df["solver"] == "barnes_hut"].sort_values("N")

# ── Plot 1: wall time por paso vs N (log-log) ─────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
if not direct.empty:
    ax.loglog(direct["N"], direct["mean_step_wall_s"]*1e3, "o-",
              color="#2563EB", label="DirectGravity O(N²)", linewidth=2, markersize=7)
if not bh.empty:
    ax.loglog(bh["N"], bh["mean_step_wall_s"]*1e3, "s-",
              color="#DC2626", label="Barnes-Hut θ=0.5", linewidth=2, markersize=7)

# Líneas de referencia teóricas.
if not direct.empty:
    N_ref = np.logspace(np.log10(direct["N"].min()), np.log10(direct["N"].max()), 100)
    N0    = direct["N"].iloc[0]
    t0    = direct["mean_step_wall_s"].iloc[0] * 1e3
    ax.loglog(N_ref, t0 * (N_ref/N0)**2, "--", color="#2563EB", alpha=0.4, label="~N²")
if not bh.empty:
    N_ref = np.logspace(np.log10(bh["N"].min()), np.log10(bh["N"].max()), 100)
    N0    = bh["N"].iloc[0]
    t0    = bh["mean_step_wall_s"].iloc[0] * 1e3
    ax.loglog(N_ref, t0 * (N_ref/N0) * np.log(N_ref/N0 + 1), "--",
              color="#DC2626", alpha=0.4, label="~N log N")

ax.set_xlabel("Número de partículas N", fontsize=12)
ax.set_ylabel("Tiempo por paso (ms)", fontsize=12)
ax.set_title("Costo computacional vs N\n(gadget-ng, 1 rank, Plummer IC)", fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, which="both", alpha=0.3)
plt.tight_layout()
fig.savefig(PLOTS_DIR / "scaling_n_wall_time.png", dpi=150)
plt.close()
print("  → scaling_n_wall_time.png")

# ── Plot 2: speedup BH/Direct ─────────────────────────────────────────────────
if not direct.empty and not bh.empty:
    common_N = sorted(set(direct["N"]) & set(bh["N"]))
    if common_N:
        fig, ax = plt.subplots(figsize=(7, 5))
        spd = []
        for n in common_N:
            t_d = direct[direct["N"] == n]["mean_step_wall_s"].values[0]
            t_b = bh[bh["N"] == n]["mean_step_wall_s"].values[0]
            spd.append(t_d / max(t_b, 1e-10))
        ax.semilogx(common_N, spd, "o-", color="#059669", linewidth=2, markersize=8)
        ax.axhline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.6,
                   label="Speedup = 1 (paridad)")
        ax.set_xlabel("N", fontsize=12)
        ax.set_ylabel("Speedup Direct/BH", fontsize=12)
        ax.set_title("Speedup Barnes-Hut vs DirectGravity\n(> 1 significa BH es más rápido)", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(PLOTS_DIR / "scaling_n_speedup.png", dpi=150)
        plt.close()
        print("  → scaling_n_speedup.png")

# ── Plot 3: fracción de tiempo en fuerzas ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
for solver_name, sub, color in [("DirectGravity", direct, "#2563EB"), ("Barnes-Hut", bh, "#DC2626")]:
    if not sub.empty and "gravity_fraction" in sub.columns:
        ax.semilogx(sub["N"], sub["gravity_fraction"]*100, "o-", color=color,
                    label=solver_name, linewidth=2, markersize=7)
ax.set_xlabel("N", fontsize=12)
ax.set_ylabel("Fracción del tiempo en fuerzas (%)", fontsize=12)
ax.set_title("Fracción de tiempo en cálculo gravitatorio vs N", fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 105)
plt.tight_layout()
fig.savefig(PLOTS_DIR / "scaling_n_gravity_frac.png", dpi=150)
plt.close()
print("  → scaling_n_gravity_frac.png")

# ── Plot 4: comparación N cruzado Direct vs BH (panel) ───────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Panel izquierdo: tiempo absoluto
if not direct.empty:
    ax1.loglog(direct["N"], direct["mean_step_wall_s"]*1e3, "o-",
               color="#2563EB", label="Direct", linewidth=2, markersize=7)
if not bh.empty:
    ax1.loglog(bh["N"], bh["mean_step_wall_s"]*1e3, "s-",
               color="#DC2626", label="BH θ=0.5", linewidth=2, markersize=7)
ax1.set_xlabel("N", fontsize=12); ax1.set_ylabel("ms/paso", fontsize=12)
ax1.set_title("Tiempo por paso", fontsize=12); ax1.legend(); ax1.grid(True, which="both", alpha=0.3)

# Panel derecho: speedup
if not direct.empty and not bh.empty and common_N:
    ax2.semilogx(common_N, spd, "o-", color="#059669", linewidth=2, markersize=8)
    ax2.axhline(1.0, linestyle="--", color="black", alpha=0.5)
    ax2.set_xlabel("N", fontsize=12); ax2.set_ylabel("Speedup Direct/BH", fontsize=12)
    ax2.set_title("Speedup BH vs Direct", fontsize=12); ax2.grid(True, alpha=0.3)

fig.suptitle("Escalado de costo computacional: DirectGravity vs Barnes-Hut", fontsize=13)
plt.tight_layout()
fig.savefig(PLOTS_DIR / "scaling_n_panel.png", dpi=150)
plt.close()
print("  → scaling_n_panel.png")

print(f"\nPlots guardados en {PLOTS_DIR}/")
