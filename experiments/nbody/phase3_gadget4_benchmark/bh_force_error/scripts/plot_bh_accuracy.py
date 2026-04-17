#!/usr/bin/env python3
"""plot_bh_accuracy.py — Genera plots de precisión BH vs Direct.

Genera:
    plots/bh_error_vs_theta.png     — error relativo medio vs θ (log-log)
    plots/bh_speedup_vs_theta.png   — speedup BH/Direct vs θ
    plots/bh_max_error_vs_theta.png — error máximo vs θ
"""

import os
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.dirname(SCRIPT_DIR)
CSV_PATH = os.path.join(EXP_DIR, "results", "bh_accuracy.csv")
PLOTS_DIR = os.path.join(EXP_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

if not os.path.exists(CSV_PATH):
    print(f"ERROR: CSV no encontrado en {CSV_PATH}")
    print("Ejecuta primero: bash scripts/run_bh_accuracy.sh")
    sys.exit(1)

df = pd.read_csv(CSV_PATH)
print(f"Datos cargados: {len(df)} filas")
print(df.to_string())

distributions = df["distribution"].unique()
markers = {"uniform_sphere": "o", "plummer": "s"}
colors  = {"uniform_sphere": "#2563EB", "plummer": "#DC2626"}
labels  = {"uniform_sphere": "Esfera uniforme", "plummer": "Plummer (a=0.1)"}

# ── Plot 1: error medio vs θ ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
thetas_ref = np.linspace(0.1, 1.1, 200)

for dist in distributions:
    sub = df[df["distribution"] == dist].sort_values("theta")
    ax.semilogy(
        sub["theta"],
        sub["mean_err"] * 100,
        marker=markers.get(dist, "o"),
        color=colors.get(dist, "gray"),
        label=labels.get(dist, dist),
        linewidth=2,
        markersize=7,
    )

# Líneas de referencia: umbrales de aceptación
ax.axhline(1.0, color="green", linestyle="--", linewidth=1, alpha=0.7, label="1% umbral")
ax.axhline(5.0, color="orange", linestyle="--", linewidth=1, alpha=0.7, label="5% umbral")
ax.axhline(15.0, color="red", linestyle="--", linewidth=1, alpha=0.7, label="15% umbral")

ax.set_xlabel("Parámetro de apertura θ", fontsize=12)
ax.set_ylabel("Error relativo medio (%)", fontsize=12)
ax.set_title("Precisión de fuerza: Barnes-Hut vs DirectGravity\n(monopolo, N=500)", fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, which="both", alpha=0.3)
ax.set_xlim(0.1, 1.05)
plt.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "bh_error_vs_theta.png"), dpi=150)
plt.close()
print("  → bh_error_vs_theta.png")

# ── Plot 2: speedup BH/Direct vs θ ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
for dist in distributions:
    sub = df[df["distribution"] == dist].sort_values("theta")
    speedup = sub["time_direct_ms"] / sub["time_bh_ms"].clip(lower=0.001)
    ax.plot(
        sub["theta"],
        speedup,
        marker=markers.get(dist, "o"),
        color=colors.get(dist, "gray"),
        label=labels.get(dist, dist),
        linewidth=2,
        markersize=7,
    )

ax.axhline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.6, label="Speedup = 1 (paridad)")
ax.set_xlabel("Parámetro de apertura θ", fontsize=12)
ax.set_ylabel("Speedup BH / Direct", fontsize=12)
ax.set_title("Speedup Barnes-Hut vs Direct (N=500)\n(nota: BH justifica su costo para N ≫ 500)", fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(0.1, 1.05)
plt.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "bh_speedup_vs_theta.png"), dpi=150)
plt.close()
print("  → bh_speedup_vs_theta.png")

# ── Plot 3: error máximo vs θ (alerta de worst-case) ─────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
for dist in distributions:
    sub = df[df["distribution"] == dist].sort_values("theta")
    ax.semilogy(
        sub["theta"],
        sub["max_err"] * 100,
        marker=markers.get(dist, "o"),
        color=colors.get(dist, "gray"),
        label=labels.get(dist, dist),
        linewidth=2,
        markersize=7,
        linestyle="--",
    )
    ax.semilogy(
        sub["theta"],
        sub["mean_err"] * 100,
        marker=markers.get(dist, "o"),
        color=colors.get(dist, "gray"),
        label=f"{labels.get(dist, dist)} (media)",
        linewidth=2,
        markersize=5,
        linestyle="-",
        alpha=0.7,
    )

ax.set_xlabel("Parámetro de apertura θ", fontsize=12)
ax.set_ylabel("Error relativo de fuerza (%)", fontsize=12)
ax.set_title("Error medio y máximo de fuerza BH\n(línea continua = media, línea punteada = máximo)", fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, which="both", alpha=0.3)
ax.set_xlim(0.1, 1.05)
plt.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "bh_max_error_vs_theta.png"), dpi=150)
plt.close()
print("  → bh_max_error_vs_theta.png")

# ── Plot 4: comparación Esfera vs Plummer (panel doble) ───────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for dist, ax in zip(["uniform_sphere", "plummer"], [ax1, ax2]):
    sub = df[df["distribution"] == dist].sort_values("theta")
    ax.semilogy(sub["theta"], sub["mean_err"]*100, "o-", color="#2563EB",
                label="Media", linewidth=2, markersize=7)
    ax.semilogy(sub["theta"], sub["rms_err"]*100, "s--", color="#7C3AED",
                label="RMS", linewidth=2, markersize=6)
    ax.semilogy(sub["theta"], sub["max_err"]*100, "^:", color="#DC2626",
                label="Máximo", linewidth=2, markersize=6)
    ax.set_xlabel("θ", fontsize=12)
    ax.set_ylabel("Error de fuerza (%)", fontsize=12)
    ax.set_title(labels.get(dist, dist), fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlim(0.1, 1.05)

fig.suptitle("Distribución de errores: BH vs Direct (N=500)", fontsize=13)
plt.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "bh_error_panel.png"), dpi=150)
plt.close()
print("  → bh_error_panel.png")

print(f"\nPlots guardados en {PLOTS_DIR}/")
