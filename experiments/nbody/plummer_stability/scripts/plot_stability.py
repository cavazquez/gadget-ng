#!/usr/bin/env python3
"""plot_stability.py — Gráficos de estabilidad de esfera de Plummer.

Genera:
  - plots/energy_evolution.png: E(t), KE(t), PE(t)
  - plots/virial_ratio.png: Q(t) = -T/W
  - plots/half_mass_radius.png: r_hm(t)
  - plots/serial_mpi_parity.png: comparación serial vs MPI (si hay datos)

Uso:
    cd experiments/nbody/plummer_stability
    python scripts/plot_stability.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = EXPERIMENT_DIR / "results"
PLOTS_DIR = EXPERIMENT_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# Colores por ejecución
COLORS = {"serial": "#1f77b4", "mpi_2rank": "#ff7f0e", "mpi_4rank": "#2ca02c"}
LABELS = {"serial": "Serial", "mpi_2rank": "MPI 2 rangos", "mpi_4rank": "MPI 4 rangos"}


def load_timeseries() -> dict:
    """Carga todas las series temporales disponibles."""
    data = {}
    for run in ["serial", "mpi_2rank", "mpi_4rank"]:
        path = RESULTS_DIR / f"{run}_timeseries.csv"
        if path.exists():
            data[run] = pd.read_csv(path)
    return data


def plot_energy(data: dict):
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)

    for run, df in data.items():
        color = COLORS.get(run, "gray")
        label = LABELS.get(run, run)
        x = df["t_over_tcross"] if "t_over_tcross" in df else df["t"]
        xlabel = r"$t / t_{\rm cross}$"

        axes[0].plot(x, df["E"], color=color, label=label, linewidth=1.5)
        axes[1].plot(x, df["KE"], color=color, linewidth=1.5, linestyle="-")
        axes[1].plot(x, -df["PE"], color=color, linewidth=1.5, linestyle="--", alpha=0.6)
        axes[2].semilogy(x, df["dE_rel"].clip(lower=1e-16), color=color,
                         label=label, linewidth=1.5)

    axes[0].set_ylabel(r"$E_{\rm total}$", fontsize=11)
    axes[0].set_title("Energía total", fontsize=10)
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    axes[1].set_ylabel(r"Energía", fontsize=11)
    axes[1].set_title(r"Cinética ($T$, línea sólida) y $|$Potencial$|$ ($|W|$, discontinua)", fontsize=10)
    axes[1].grid(alpha=0.3)

    axes[2].set_ylabel(r"$|\Delta E / E_0|$", fontsize=11)
    axes[2].set_title("Error relativo de energía", fontsize=10)
    axes[2].legend(fontsize=9)
    axes[2].grid(alpha=0.3, which="both")
    axes[2].set_xlabel(xlabel, fontsize=11)

    fig.suptitle(
        "Evolución energética — Esfera de Plummer (N=200, θ=0.5)\ngadget-ng leapfrog KDK",
        fontsize=12,
    )
    plt.tight_layout()
    out = PLOTS_DIR / "energy_evolution.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Guardado: {out}")


def plot_virial(data: dict):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axhline(0.5, color="k", linewidth=1.5, linestyle="--", label="Equilibrio Q=0.5")

    for run, df in data.items():
        x = df["t_over_tcross"] if "t_over_tcross" in df else df["t"]
        ax.plot(x, df["Q"], color=COLORS.get(run, "gray"),
                label=LABELS.get(run, run), linewidth=1.5)

    ax.fill_between([0, ax.get_xlim()[1] if data else 12], 0.35, 0.65,
                    alpha=0.1, color="green", label="Rango virial [0.35, 0.65]")

    ax.set_xlabel(r"$t / t_{\rm cross}$", fontsize=11)
    ax.set_ylabel(r"$Q = -T/W$", fontsize=11)
    ax.set_title(
        "Ratio virial Q(t) — Esfera de Plummer en equilibrio\n"
        r"Teorema del virial: $2T + W = 0 \Rightarrow Q = 0.5$",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.5)

    plt.tight_layout()
    out = PLOTS_DIR / "virial_ratio.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Guardado: {out}")


def plot_half_mass_radius(data: dict):
    fig, ax = plt.subplots(figsize=(8, 4))

    for run, df in data.items():
        x = df["t_over_tcross"] if "t_over_tcross" in df else df["t"]
        ax.plot(x, df["r_hm"], color=COLORS.get(run, "gray"),
                label=LABELS.get(run, run), linewidth=1.5)

    # Radio de media masa teórico de Plummer: r_hm = a·(2^(2/3)-1)^(1/2) ≈ 1.305·a
    r_hm_theory = 1.0 * (2.0 ** (2.0 / 3.0) - 1.0) ** 0.5
    ax.axhline(r_hm_theory, color="gray", linewidth=1, linestyle=":",
               label=f"Teórico Plummer r_hm ≈ {r_hm_theory:.3f}")

    ax.set_xlabel(r"$t / t_{\rm cross}$", fontsize=11)
    ax.set_ylabel(r"$r_{\rm hm}$", fontsize=11)
    ax.set_title(
        "Radio de media masa — Esfera de Plummer\n"
        r"(estable en equilibrio: $r_{\rm hm} \approx $ cte.)",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out = PLOTS_DIR / "half_mass_radius.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Guardado: {out}")


def plot_serial_mpi_parity():
    """Gráfico de dispersión: error serial vs MPI por partícula."""
    for mpi_label in ["mpi_2rank", "mpi_4rank"]:
        cmp_path = RESULTS_DIR / f"serial_{mpi_label}_comparison.csv"
        if not cmp_path.exists():
            continue

        df = pd.read_csv(cmp_path)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        axes[0].semilogy(df["global_id"], df["dr"].clip(lower=1e-16),
                         "o", markersize=3, color="#1f77b4", alpha=0.6)
        axes[0].set_xlabel("global_id", fontsize=10)
        axes[0].set_ylabel(r"$|\Delta r|$ [u.l.]", fontsize=10)
        axes[0].set_title(f"Error posicional serial vs {mpi_label}", fontsize=10)
        axes[0].grid(alpha=0.3, which="both")

        axes[1].semilogy(df["global_id"], df["dv"].clip(lower=1e-16),
                         "o", markersize=3, color="#ff7f0e", alpha=0.6)
        axes[1].set_xlabel("global_id", fontsize=10)
        axes[1].set_ylabel(r"$|\Delta v|$ [u.v.]", fontsize=10)
        axes[1].set_title(f"Error de velocidad serial vs {mpi_label}", fontsize=10)
        axes[1].grid(alpha=0.3, which="both")

        fig.suptitle(
            f"Paridad numérica serial vs {mpi_label} — final de simulación\n"
            f"max|Δr|={df['dr'].max():.2e}  max|Δv|={df['dv'].max():.2e}",
            fontsize=11,
        )
        plt.tight_layout()
        out = PLOTS_DIR / f"serial_{mpi_label}_parity.png"
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"Guardado: {out}")


def main():
    print("=== Generando gráficos de estabilidad Plummer ===\n")
    data = load_timeseries()

    if not data:
        print(
            "ERROR: No se encontraron series temporales en results/.\n"
            "Ejecuta primero: python scripts/analyze_stability.py"
        )
        sys.exit(1)

    print(f"Ejecuciones disponibles: {list(data.keys())}\n")

    plot_energy(data)
    plot_virial(data)
    plot_half_mass_radius(data)
    plot_serial_mpi_parity()

    print("\nListo.")


if __name__ == "__main__":
    main()
