#!/usr/bin/env python3
"""plot_collapse.py — Gráficos del colapso gravitacional frío.

Genera:
  - plots/collapse_overview.png: r_hm(t), Q(t), E(t) y |ΔE/E₀|(t) en 4 paneles.
  - plots/rHm_vs_Tff.png: r_hm(t) con referencia al T_ff analítico.

Uso:
    cd experiments/nbody/cold_collapse
    python scripts/plot_collapse.py
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = EXPERIMENT_DIR / "results"
PLOTS_DIR = EXPERIMENT_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

G = 1.0
R = 1.0
M_TOT = 1.0
EPS = 0.05
T_FF = math.pi * math.sqrt(R**3 / (2.0 * G * M_TOT))
R_HM_THEORY = R * (0.5 ** (1.0 / 3.0))


def load_data() -> pd.DataFrame:
    path = RESULTS_DIR / "collapse_timeseries.csv"
    if not path.exists():
        print(
            "ERROR: results/collapse_timeseries.csv no encontrado.\n"
            "Ejecuta primero: python scripts/analyze_collapse.py"
        )
        sys.exit(1)
    return pd.read_csv(path)


def plot_overview(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    x = df["t_over_tff"]
    color = "#1f77b4"

    # Panel 1: r_hm(t)
    ax = axes[0, 0]
    ax.plot(x, df["r_hm"], color=color, linewidth=2)
    ax.axhline(R_HM_THEORY, color="gray", linewidth=1, linestyle=":",
               label=f"Plummer r_hm teórico ≈ {R_HM_THEORY:.3f}")
    ax.axvline(1.0, color="red", linewidth=1, linestyle="--", alpha=0.6, label=r"$t = T_{\rm ff}$")
    ax.set_ylabel(r"$r_{\rm hm}$ [u.l.]", fontsize=11)
    ax.set_title("Radio de media masa", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Panel 2: Q(t) = -T/W
    ax = axes[0, 1]
    ax.plot(x, df["Q"], color="#ff7f0e", linewidth=2)
    ax.axhline(0.5, color="k", linewidth=1.2, linestyle="--", label="Equilibrio Q=0.5")
    ax.axvline(1.0, color="red", linewidth=1, linestyle="--", alpha=0.6)
    ax.fill_between(x, 0.3, 0.7, alpha=0.1, color="green")
    ax.set_ylabel(r"$Q = -T/W$", fontsize=11)
    ax.set_title("Ratio virial (virialización → Q=0.5)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_ylim(bottom=0)

    # Panel 3: E(t)
    ax = axes[1, 0]
    ax.plot(x, df["E"], color=color, linewidth=2, label=r"$E_{\rm total}$")
    ax.plot(x, df["KE"], color="#2ca02c", linewidth=1.5, linestyle="--", label=r"$T$ (cinética)")
    ax.plot(x, df["PE"], color="#d62728", linewidth=1.5, linestyle=":", label=r"$W$ (potencial)")
    ax.axvline(1.0, color="red", linewidth=1, linestyle="--", alpha=0.6)
    ax.set_ylabel("Energía", fontsize=11)
    ax.set_xlabel(r"$t / T_{\rm ff}$", fontsize=11)
    ax.set_title("Evolución energética", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Panel 4: |ΔE/E₀|(t)
    ax = axes[1, 1]
    ax.semilogy(x, df["dE_rel"].clip(lower=1e-16), color="#9467bd", linewidth=2)
    ax.axvline(1.0, color="red", linewidth=1, linestyle="--", alpha=0.6, label=r"$t = T_{\rm ff}$")
    ax.set_ylabel(r"$|\Delta E / E_0|$", fontsize=11)
    ax.set_xlabel(r"$t / T_{\rm ff}$", fontsize=11)
    ax.set_title("Error acumulado de energía", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    fig.suptitle(
        "Colapso Gravitacional Frío — Esfera Uniforme (N=200, G=1, R=1)\n"
        fr"gadget-ng leapfrog KDK, $\varepsilon={EPS}$, $\theta=0.5$, "
        fr"$T_{{ff}} = \pi/\sqrt{{2}} \approx {T_FF:.3f}$",
        fontsize=12,
    )
    plt.tight_layout()
    out = PLOTS_DIR / "collapse_overview.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Guardado: {out}")


def plot_rHm_vs_Tff(df: pd.DataFrame):
    """Gráfico dedicado de r_hm(t) con comparación analítica."""
    fig, ax = plt.subplots(figsize=(8, 4))
    x = df["t_over_tff"]

    ax.plot(x, df["r_hm"] / R, color="#1f77b4", linewidth=2.5, label=r"$r_{\rm hm}(t)$ (simulación)")

    # Caída libre analítica (esfera uniforme): r(t) ≈ R·cos²(πt/(2T_ff)) para t < T_ff
    t_arr = np.linspace(0, min(x.max(), 1.0), 200)
    r_analytic = np.cos(0.5 * np.pi * t_arr) ** (4.0 / 3.0)  # aproximación analítica
    ax.plot(t_arr, r_analytic * R_HM_THEORY / R, color="gray", linewidth=1.5,
            linestyle="--", alpha=0.7, label=r"Aprox. analítica $\propto \cos^{4/3}(\pi t/2T_{\rm ff})$")

    ax.axvline(1.0, color="red", linewidth=1.5, linestyle=":", label=r"$T_{\rm ff}$")
    ax.axhline(R_HM_THEORY / R, color="green", linewidth=1, linestyle=":",
               alpha=0.7, label=fr"$r_{{hm,0}}/R \approx {R_HM_THEORY/R:.3f}$")

    ax.set_xlabel(r"$t / T_{\rm ff}$", fontsize=12)
    ax.set_ylabel(r"$r_{\rm hm} / R$", fontsize=12)
    ax.set_title(
        fr"Radio de media masa — Colapso frío (gadget-ng vs. analítica, $T_{{ff}} \approx {T_FF:.3f}$)",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    out = PLOTS_DIR / "rHm_vs_Tff.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Guardado: {out}")


def main():
    print("=== Generando gráficos de colapso frío ===\n")
    df = load_data()
    print(f"Frames disponibles: {len(df)}")
    print(f"T_ff analítico: {T_FF:.4f}")
    print(f"t_total = {df['t'].max():.3f} = {df['t_over_tff'].max():.2f}·T_ff\n")

    plot_overview(df)
    plot_rHm_vs_Tff(df)

    print("\nListo.")


if __name__ == "__main__":
    main()
