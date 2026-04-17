#!/usr/bin/env python3
"""plot_block_compare.py — Plots de comparación global dt vs block timesteps."""

import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

SCRIPT_DIR  = Path(__file__).parent
EXP_DIR     = SCRIPT_DIR.parent
RESULTS_DIR = EXP_DIR / "results"
PLOTS_DIR   = EXP_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

CSV_PATH     = RESULTS_DIR / "block_compare.csv"
SUMMARY_PATH = RESULTS_DIR / "block_summary.csv"

if not CSV_PATH.exists():
    print(f"ERROR: {CSV_PATH} no encontrado. Ejecuta analyze_block_compare.py primero.")
    sys.exit(1)

df = pd.read_csv(CSV_PATH)
summary = pd.read_csv(SUMMARY_PATH) if SUMMARY_PATH.exists() else None

COLORS = {"global_dt": "#2563EB", "hierarchical": "#DC2626"}
LABELS = {"global_dt": "Timestep global fijo", "hierarchical": "Block timesteps (Aarseth)"}

# ── Plot 1: Energía cinética vs paso ─────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

time_col = "time" if "time" in df.columns else "step"
ke_col   = "KE" if "KE" in df.columns else "kinetic_energy"
de_col   = "dE_rel" if "dE_rel" in df.columns else "ke_drift"

for mode in df["mode"].unique():
    sub = df[df["mode"] == mode].sort_values(time_col)
    color = COLORS.get(mode, "gray")
    label = LABELS.get(mode, mode)

    ax1.plot(sub[time_col], sub[ke_col], color=color, label=label, linewidth=1.8)
    ax2.semilogy(sub[time_col], sub[de_col].clip(lower=1e-10),
                 color=color, label=label, linewidth=1.8)

ax1.set_ylabel("Energía cinética (u.i.)", fontsize=11)
ax1.set_title("Colapso frío: Timestep global vs Block timesteps", fontsize=12)
ax1.legend(fontsize=10); ax1.grid(True, alpha=0.3)

ax2.set_xlabel("Tiempo (u.i.)", fontsize=11)
ax2.set_ylabel("|ΔE/E₀|", fontsize=11)
ax2.set_title("Drift relativo de energía total", fontsize=12)
ax2.legend(fontsize=10); ax2.grid(True, which="both", alpha=0.3)

plt.tight_layout()
fig.savefig(PLOTS_DIR / "block_energy_comparison.png", dpi=150)
plt.close()
print("  → block_energy_comparison.png")

# ── Plot 2: |ΔE| acumulado (solo drift) ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
for mode in df["mode"].unique():
    sub = df[df["mode"] == mode].sort_values(time_col)
    ax.semilogy(sub[time_col], sub[de_col].clip(lower=1e-10),
                color=COLORS.get(mode, "gray"),
                label=LABELS.get(mode, mode), linewidth=2)

ax.set_xlabel("Tiempo (u.i.)", fontsize=12)
ax.set_ylabel("|ΔE/E₀|", fontsize=12)
ax.set_title("Drift relativo de energía: global vs block timesteps", fontsize=12)
ax.legend(fontsize=11); ax.grid(True, which="both", alpha=0.3)
plt.tight_layout()
fig.savefig(PLOTS_DIR / "block_ke_drift.png", dpi=150)
plt.close()
print("  → block_ke_drift.png")

# ── Plot 3: Tabla resumen (si disponible) ─────────────────────────────────────
if summary is not None and not summary.empty:
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis("off")
    cols = ["label", "total_wall_s", "mean_step_ms", "ke_drift_max", "ke_drift_final"]
    available = [c for c in cols if c in summary.columns]
    data = summary[available].copy()
    # Formatear columnas numéricas.
    for col in ["total_wall_s", "mean_step_ms"]:
        if col in data.columns:
            data[col] = data[col].map(lambda x: f"{x:.3f}")
    for col in ["ke_drift_max", "ke_drift_final"]:
        if col in data.columns:
            data[col] = data[col].map(lambda x: f"{x:.4f}")
    table = ax.table(
        cellText=data.values,
        colLabels=available,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    ax.set_title("Resumen: Global dt vs Block Timesteps", fontsize=12, pad=15)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "block_summary_table.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  → block_summary_table.png")

print(f"\nPlots guardados en {PLOTS_DIR}/")
