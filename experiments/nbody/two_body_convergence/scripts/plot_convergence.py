#!/usr/bin/env python3
"""plot_convergence.py — Gráficos de convergencia Kepler.

Genera:
  - plots/convergence_loglog.png: |ΔE/E₀| y |ΔL/L₀| vs dt en escala log-log.
  - plots/energy_timeseries.png: E(t) para cada valor de dt.

Uso:
    cd experiments/nbody/two_body_convergence
    python scripts/plot_convergence.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # sin display
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = EXPERIMENT_DIR / "results"
PLOTS_DIR = EXPERIMENT_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

T_ORBIT = 2.0 * 3.14159265358979  # período orbital


def check_data():
    conv_path = RESULTS_DIR / "convergence.csv"
    ts_path = RESULTS_DIR / "energy_timeseries.csv"
    if not conv_path.exists():
        print(
            "ERROR: No se encontró results/convergence.csv\n"
            "Ejecuta primero: python scripts/analyze_convergence.py"
        )
        sys.exit(1)
    return conv_path, ts_path


def plot_convergence_loglog(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 5))

    colors = {"dE_rel": "#1f77b4", "dL_rel": "#ff7f0e"}

    for metric, label, color in [
        ("dE_rel", r"$|\Delta E / E_0|$", colors["dE_rel"]),
        ("dL_rel", r"$|\Delta L_z / L_{z,0}|$", colors["dL_rel"]),
    ]:
        if metric not in df.columns:
            continue
        mask = df[metric] > 0
        if mask.sum() < 2:
            continue
        ax.loglog(df.loc[mask, "dt"], df.loc[mask, metric],
                  "o-", color=color, label=label, linewidth=2, markersize=7)

    # Línea de referencia O(dt²)
    dt_ref = np.array([df["dt"].min(), df["dt"].max()])
    # Normalizar por el valor en dt_max
    e_ref = df["dE_rel"].values[-1] if "dE_rel" in df else 1.0
    dt_max = df["dt"].values[-1]
    ax.loglog(dt_ref, e_ref * (dt_ref / dt_max) ** 2,
              "k--", linewidth=1.5, alpha=0.6, label=r"$\propto \Delta t^2$ (referencia)")

    # Anotar los labels T/N
    if "label" in df.columns:
        for _, row in df.iterrows():
            if row.get("dE_rel", 0) > 0:
                ax.annotate(
                    row["label"],
                    xy=(row["dt"], row["dE_rel"]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                )

    ax.set_xlabel(r"Paso de tiempo $\Delta t$ [unidades de simulación]", fontsize=12)
    ax.set_ylabel("Error relativo tras 1 período", fontsize=12)
    ax.set_title(
        "Convergencia leapfrog KDK — Problema de Kepler (dos cuerpos)\n"
        r"$G=1,\ M=1,\ r=1,\ T_{\rm orbit}=2\pi$",
        fontsize=11,
    )
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    ax.xaxis.set_major_formatter(ticker.LogFormatterMathtext())
    ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext())

    out = PLOTS_DIR / "convergence_loglog.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Guardado: {out}")


def plot_energy_timeseries(df_ts: pd.DataFrame):
    if df_ts.empty:
        print("Sin datos de series temporales.")
        return

    runs = df_ts["run"].unique()
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=False)

    cmap = plt.cm.viridis
    colors = {r: cmap(i / max(len(runs) - 1, 1)) for i, r in enumerate(runs)}
    labels_map = {
        "dt020": "T/20", "dt050": "T/50", "dt100": "T/100",
        "dt200": "T/200", "dt500": "T/500",
    }

    # Panel 1: E(t) / E₀  (oscilación del hamiltoniano sombra)
    ax = axes[0]
    for run in runs:
        sub = df_ts[df_ts["run"] == run].sort_values("t")
        if sub.empty:
            continue
        e0 = sub["E"].iloc[0]
        ax.plot(
            sub["t"] / T_ORBIT,
            (sub["E"] - e0) / abs(e0),
            color=colors[run],
            label=labels_map.get(run, run),
            linewidth=1.5,
        )
    ax.axhline(0, color="k", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_ylabel(r"$(E(t) - E_0) / |E_0|$", fontsize=11)
    ax.set_title("Conservación de energía — leapfrog KDK", fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3)
    ax.set_xlabel(r"Tiempo $t / T_{\rm orbit}$", fontsize=11)

    # Panel 2: |ΔE/E₀| acumulado
    ax2 = axes[1]
    for run in runs:
        sub = df_ts[df_ts["run"] == run].sort_values("t")
        if sub.empty:
            continue
        ax2.semilogy(
            sub["t"] / T_ORBIT,
            sub["dE_rel"].clip(lower=1e-16),
            color=colors[run],
            label=labels_map.get(run, run),
            linewidth=1.5,
        )
    ax2.set_xlabel(r"Tiempo $t / T_{\rm orbit}$", fontsize=11)
    ax2.set_ylabel(r"$|\Delta E / E_0|$", fontsize=11)
    ax2.set_title("Error acumulado de energía", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3, which="both")

    fig.suptitle(
        "Experimento: Convergencia Kepler — gadget-ng vs. solución analítica",
        fontsize=12, y=1.01,
    )
    plt.tight_layout()
    out = PLOTS_DIR / "energy_timeseries.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Guardado: {out}")


def main():
    print("=== Generando gráficos de convergencia Kepler ===\n")
    conv_path, ts_path = check_data()

    df_conv = pd.read_csv(conv_path)
    print(f"Tabla de convergencia: {len(df_conv)} filas")
    plot_convergence_loglog(df_conv)

    if ts_path.exists():
        df_ts = pd.read_csv(ts_path)
        print(f"Series temporales: {len(df_ts)} filas")
        plot_energy_timeseries(df_ts)
    else:
        print("Sin datos de series temporales (results/energy_timeseries.csv no encontrado)")

    print("\nListo.")


if __name__ == "__main__":
    main()
