#!/usr/bin/env python3
"""
plot_ensemble_pk.py — Fase 31: Figuras del espectro de potencia del ensemble.

Genera tres figuras:
  1. P_mean(k) ± stderr vs referencia EH (log-log) para cada resolución.
  2. R(k) = P_mean / P_EH con bandas de error (1σ).
  3. Comparación N=32³ vs N=64³ en el ratio R(k) (si ambos disponibles).

Uso:
  python plot_ensemble_pk.py \\
      --stats-n32 stats_N32_a002_2lpt_pm.json \\
      [--stats-n64 stats_N64_a002_2lpt_pm.json] \\
      --output-dir figures/
"""

import argparse
import json
import math
import os
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def load_stats(path):
    with open(path) as f:
        return json.load(f)


def plot_pk_vs_eh(stats, ax, label, color="steelblue"):
    """Grafica P_mean(k) ± stderr y P_EH(k) en log-log."""
    bins = [b for b in stats["bins"] if b["p_mean_hmpc"] > 0.0 and math.isfinite(b["k_hmpc"])]
    k = [b["k_hmpc"] for b in bins]
    p_mean = [b["p_mean_hmpc"] for b in bins]
    p_stderr = [b["p_stderr_hmpc"] for b in bins]
    p_eh = [b["pk_eh"] for b in bins]

    k = np.array(k)
    p_mean = np.array(p_mean)
    p_stderr = np.array(p_stderr)
    p_eh = np.array(p_eh)

    n_seeds = stats["n_seeds"]
    ax.errorbar(k, p_mean, yerr=p_stderr, fmt="o-", color=color,
                label=f"{label} (N_seeds={n_seeds})", capsize=3, ms=5, lw=1.5)
    ax.plot(k, p_eh, "k--", lw=1.5, alpha=0.7, label="P_EH (referencia)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$k$ [h/Mpc]")
    ax.set_ylabel(r"$P(k)$ [(Mpc/h)³]")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)


def plot_r_k(stats_list, ax, colors=None):
    """Grafica R(k) = P_mean/P_EH con bandas de error para múltiples variantes."""
    if colors is None:
        colors = ["steelblue", "darkorange", "forestgreen", "crimson"]

    ax.axhline(1.0, color="gray", ls="--", lw=1.0, label="R=1 (match perfecto)")

    for stats, color in zip(stats_list, colors):
        label = stats["label"]
        n_seeds = stats["n_seeds"]
        bins = [b for b in stats["bins"]
                if math.isfinite(b.get("r_k", float("nan"))) and b["r_k"] > 0.0]
        if not bins:
            continue

        k = np.array([b["k_hmpc"] for b in bins])
        r = np.array([b["r_k"] for b in bins])
        # Error en R ≈ stderr(P_mean) / P_EH
        r_err = np.array([b["p_stderr_hmpc"] / b["pk_eh"] for b in bins
                         if b["pk_eh"] > 0])
        if len(r_err) != len(r):
            r_err = np.zeros_like(r)

        ax.errorbar(k, r, yerr=r_err, fmt="o-", color=color,
                    label=f"{label} (N_seeds={n_seeds})", capsize=3, ms=5, lw=1.5)

        # Banda ±1σ del R (variación entre seeds)
        r_std = np.array([b["p_std_hmpc"] / b["pk_eh"] for b in bins
                         if b["pk_eh"] > 0])
        if len(r_std) == len(k):
            ax.fill_between(k, r - r_std, r + r_std, color=color, alpha=0.1)

    ax.set_xscale("log")
    ax.set_xlabel(r"$k$ [h/Mpc]")
    ax.set_ylabel(r"$R(k) = P_\mathrm{mean}/P_\mathrm{EH}$")
    ax.set_title("Ratio espectral medido / referencia EH")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())


def plot_resolution_comparison(stats_n32, stats_n64, ax):
    """Compara R(k) para N=32³ y N=64³."""
    for stats, marker, color, label_suffix in [
        (stats_n32, "o", "steelblue", " N=32³"),
        (stats_n64, "s", "darkorange", " N=64³"),
    ]:
        if stats is None:
            continue
        bins = [b for b in stats["bins"]
                if math.isfinite(b.get("r_k", float("nan"))) and b["r_k"] > 0.0]
        if not bins:
            continue
        k = np.array([b["k_hmpc"] for b in bins])
        r = np.array([b["r_k"] for b in bins])
        r_err = np.array([b["p_stderr_hmpc"] / b["pk_eh"] for b in bins])
        label = stats["label"].split("_")[0] + label_suffix + f" (Nseeds={stats['n_seeds']})"
        ax.errorbar(k, r, yerr=r_err, fmt=f"{marker}-", color=color,
                    label=label, capsize=3, ms=6, lw=1.5)

    ax.axhline(1.0, color="gray", ls="--", lw=1.0)
    ax.set_xscale("log")
    ax.set_xlabel(r"$k$ [h/Mpc]")
    ax.set_ylabel(r"$R(k) = P_\mathrm{mean}/P_\mathrm{EH}$")
    ax.set_title("Comparación N=32³ vs N=64³: ratio espectral R(k)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats-n32", required=True,
                        help="JSON de stats del ensemble N=32³")
    parser.add_argument("--stats-n64", default=None,
                        help="JSON de stats del ensemble N=64³ (opcional)")
    parser.add_argument("--output-dir", default="figures",
                        help="Directorio de salida para las figuras")
    args = parser.parse_args()

    if not HAS_MATPLOTLIB:
        print("matplotlib no disponible. Generando solo tabla de texto.")
        stats = load_stats(args.stats_n32)
        print(f"\nLabel: {stats['label']}  N_seeds: {stats['n_seeds']}")
        print(f"{'k [h/Mpc]':>12}  {'P_mean':>14}  {'stderr':>12}  {'R(k)':>10}  CV")
        print("-" * 60)
        for b in stats["bins"]:
            print(
                f"  {b['k_hmpc']:10.4f}  {b['p_mean_hmpc']:14.4e}  "
                f"{b['p_stderr_hmpc']:12.4e}  {b['r_k']:10.4e}  {b['cv']:.4f}"
            )
        return

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats_n32 = load_stats(args.stats_n32)
    stats_n64 = load_stats(args.stats_n64) if args.stats_n64 else None

    # ── Figura 1: P_mean(k) vs P_EH ──────────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    plot_pk_vs_eh(stats_n32, ax1, label=stats_n32["label"], color="steelblue")
    ax1.set_title(
        f"P_mean(k) ± stderr  vs  P_EH — {stats_n32['label']}\n"
        f"N_seeds={stats_n32['n_seeds']},  "
        f"box={stats_n32.get('box_mpc_h', '?')} Mpc/h"
    )
    fig1.tight_layout()
    p1 = out_dir / f"pk_mean_vs_eh_{stats_n32['label']}.png"
    fig1.savefig(p1, dpi=150)
    print(f"Figura 1 guardada: {p1}")
    plt.close(fig1)

    if stats_n64 is not None:
        fig1b, ax1b = plt.subplots(figsize=(8, 5))
        plot_pk_vs_eh(stats_n64, ax1b, label=stats_n64["label"], color="darkorange")
        ax1b.set_title(
            f"P_mean(k) ± stderr  vs  P_EH — {stats_n64['label']}\n"
            f"N_seeds={stats_n64['n_seeds']},  "
            f"box={stats_n64.get('box_mpc_h', '?')} Mpc/h"
        )
        fig1b.tight_layout()
        p1b = out_dir / f"pk_mean_vs_eh_{stats_n64['label']}.png"
        fig1b.savefig(p1b, dpi=150)
        print(f"Figura 1b guardada: {p1b}")
        plt.close(fig1b)

    # ── Figura 2: R(k) con bandas de error ────────────────────────────────────
    all_stats = [stats_n32] + ([stats_n64] if stats_n64 else [])
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    plot_r_k(all_stats, ax2)
    fig2.tight_layout()
    p2 = out_dir / f"r_k_ratio_{stats_n32['label']}.png"
    fig2.savefig(p2, dpi=150)
    print(f"Figura 2 guardada: {p2}")
    plt.close(fig2)

    # ── Figura 3: comparación N=32³ vs N=64³ (si disponible) ─────────────────
    if stats_n64 is not None:
        fig3, ax3 = plt.subplots(figsize=(8, 5))
        plot_resolution_comparison(stats_n32, stats_n64, ax3)
        fig3.tight_layout()
        p3 = out_dir / "r_k_N32_vs_N64.png"
        fig3.savefig(p3, dpi=150)
        print(f"Figura 3 guardada: {p3}")
        plt.close(fig3)

    # Imprimir tabla de texto
    print(f"\n=== Tabla de estadísticas: {stats_n32['label']} ===")
    print(f"{'k [h/Mpc]':>12}  {'P_mean [(Mpc/h)³]':>18}  {'stderr':>12}  {'CV':>8}  {'R(k)':>10}  n_modes")
    print("-" * 80)
    for b in stats_n32["bins"]:
        print(
            f"  {b['k_hmpc']:10.4f}  {b['p_mean_hmpc']:18.4e}  "
            f"{b['p_stderr_hmpc']:12.4e}  {b['cv']:8.4f}  "
            f"{b.get('r_k', float('nan')):10.4e}  {b['n_modes']}"
        )


if __name__ == "__main__":
    main()
