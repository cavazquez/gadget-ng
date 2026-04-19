#!/usr/bin/env python3
"""
plot_lpt_pm_comparisons.py — Fase 31: Comparaciones 1LPT/2LPT y PM/TreePM.

Genera cuatro figuras con barras de error del ensemble:
  1. P_2LPT / P_1LPT  con barras de error (estado inicial).
  2. P_PM / P_TreePM  con barras de error (estado evolucionado).
  3. Tabla cuantitativa: medias y errores de los ratios.

Uso:
  python plot_lpt_pm_comparisons.py \\
      --stats-1lpt stats_N32_a002_1lpt_pm.json \\
      --stats-2lpt stats_N32_a002_2lpt_pm.json \\
      --stats-pm   stats_N32_a002_2lpt_pm.json \\
      --stats-treepm stats_N32_a002_2lpt_treepm.json \\
      --output-dir figures/
"""

import argparse
import json
import math
from pathlib import Path
from collections import defaultdict

try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def load_stats(path):
    with open(path) as f:
        return json.load(f)


def compute_ratio_stats(stats_a, stats_b, ratio_label="A/B"):
    """
    Calcula el ratio bin a bin R_i = P_mean_a(k_i) / P_mean_b(k_i).
    Propagación de errores: sigma_R = R × sqrt((sigma_a/P_a)² + (sigma_b/P_b)²).

    Devuelve lista de dicts con k, ratio, y error propagado.
    """
    bins_a = stats_a["bins"]
    bins_b = stats_b["bins"]

    if len(bins_a) != len(bins_b):
        print(f"WARNING: {ratio_label} — diferentes número de bins ({len(bins_a)} vs {len(bins_b)})")
        n = min(len(bins_a), len(bins_b))
        bins_a = bins_a[:n]
        bins_b = bins_b[:n]

    results = []
    for ba, bb in zip(bins_a, bins_b):
        k = ba["k_hmpc"]
        pa = ba["p_mean_hmpc"]
        pb = bb["p_mean_hmpc"]
        sa = ba["p_stderr_hmpc"]
        sb = bb["p_stderr_hmpc"]

        if pa <= 0.0 or pb <= 0.0:
            continue

        ratio = pa / pb
        rel_err = math.sqrt((sa / pa)**2 + (sb / pb)**2) if pa > 0 and pb > 0 else 0.0
        err_abs = ratio * rel_err

        results.append({
            "k": k,
            "ratio": ratio,
            "ratio_err": err_abs,
            "n_modes_a": ba.get("n_modes", 0),
        })

    return results


def print_ratio_table(ratios, label):
    """Imprime tabla de ratios."""
    print(f"\n=== {label} ===")
    print(f"{'k [h/Mpc]':>12}  {'ratio':>10}  {'±error':>10}  {'|ratio-1|':>12}  n_modes")
    print("-" * 60)
    vals = []
    for r in ratios:
        dev = abs(r["ratio"] - 1.0)
        vals.append(dev)
        print(
            f"  {r['k']:10.4f}  {r['ratio']:10.4f}  "
            f"{r['ratio_err']:10.4f}  {dev:12.4f}  {r['n_modes_a']}"
        )
    if vals:
        mean_dev = sum(vals) / len(vals)
        max_dev = max(vals)
        print(f"  → mean |ratio-1| = {mean_dev:.4f}  max |ratio-1| = {max_dev:.4f}")


def plot_ratio(ratios, ax, label, color="steelblue", ref_label="ratio=1"):
    """Grafica el ratio con barras de error."""
    k = [r["k"] for r in ratios]
    ratio = [r["ratio"] for r in ratios]
    err = [r["ratio_err"] for r in ratios]

    ax.axhline(1.0, color="gray", ls="--", lw=1.0, label=ref_label)
    ax.errorbar(k, ratio, yerr=err, fmt="o-", color=color,
                label=label, capsize=4, ms=6, lw=1.5)
    ax.set_xscale("log")
    ax.set_xlabel(r"$k$ [h/Mpc]")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats-1lpt", default=None,
                        help="JSON de stats del ensemble 1LPT")
    parser.add_argument("--stats-2lpt", default=None,
                        help="JSON de stats del ensemble 2LPT")
    parser.add_argument("--stats-pm", default=None,
                        help="JSON de stats del ensemble PM")
    parser.add_argument("--stats-treepm", default=None,
                        help="JSON de stats del ensemble TreePM")
    parser.add_argument("--output-dir", default="figures",
                        help="Directorio de salida")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    has_lpt = args.stats_1lpt and args.stats_2lpt
    has_solver = args.stats_pm and args.stats_treepm

    if not has_lpt and not has_solver:
        print("Se requiere al menos --stats-1lpt + --stats-2lpt o --stats-pm + --stats-treepm")
        return

    # ── Comparación 1LPT vs 2LPT ─────────────────────────────────────────────
    if has_lpt:
        s1 = load_stats(args.stats_1lpt)
        s2 = load_stats(args.stats_2lpt)
        ratios_lpt = compute_ratio_stats(s2, s1, "2LPT/1LPT")
        print_ratio_table(ratios_lpt, "P_2LPT / P_1LPT (estado inicial)")

        if HAS_MATPLOTLIB and ratios_lpt:
            fig, ax = plt.subplots(figsize=(8, 5))
            plot_ratio(
                ratios_lpt, ax,
                label=f"P_2LPT / P_1LPT  (N_seeds={s2['n_seeds']})",
                color="steelblue",
                ref_label="ratio=1 (idénticos)"
            )
            ax.set_ylabel(r"$P_{2\mathrm{LPT}} / P_{1\mathrm{LPT}}$")
            ax.set_title(
                f"Comparación 1LPT vs 2LPT (IC inicial) — {s2['label']}\n"
                f"Ensemble N_seeds={s2['n_seeds']}"
            )

            # Banda de expectativa (fase 29: corrección ~0.4% en posiciones → ~1% en P)
            ax.axhspan(0.99, 1.01, alpha=0.1, color="green", label="±1% (expectativa teórica)")
            ax.legend(fontsize=9)

            p1 = out_dir / "ratio_2lpt_vs_1lpt.png"
            fig.tight_layout()
            fig.savefig(p1, dpi=150)
            print(f"\nFigura P_2LPT/P_1LPT guardada: {p1}")
            plt.close(fig)

    # ── Comparación PM vs TreePM ──────────────────────────────────────────────
    if has_solver:
        spm = load_stats(args.stats_pm)
        stp = load_stats(args.stats_treepm)
        ratios_solver = compute_ratio_stats(spm, stp, "PM/TreePM")
        print_ratio_table(ratios_solver, "P_PM / P_TreePM (estado evolucionado)")

        if HAS_MATPLOTLIB and ratios_solver:
            fig, ax = plt.subplots(figsize=(8, 5))
            plot_ratio(
                ratios_solver, ax,
                label=f"P_PM / P_TreePM  (N_seeds={spm['n_seeds']})",
                color="darkorange",
                ref_label="ratio=1 (convergencia perfecta)"
            )
            ax.set_ylabel(r"$P_\mathrm{PM} / P_\mathrm{TreePM}$")
            ax.set_title(
                f"Comparación PM vs TreePM — {spm['label']}\n"
                f"Ensemble N_seeds={spm['n_seeds']} — Phase 30 vio 27.3% con N=8³"
            )

            p2 = out_dir / "ratio_pm_vs_treepm.png"
            fig.tight_layout()
            fig.savefig(p2, dpi=150)
            print(f"Figura P_PM/P_TreePM guardada: {p2}")
            plt.close(fig)

    # ── Figura combinada (si ambas disponibles) ───────────────────────────────
    if has_lpt and has_solver and HAS_MATPLOTLIB:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        if ratios_lpt:
            plot_ratio(ratios_lpt, axes[0],
                       label=f"P_2LPT/P_1LPT (N_seeds={s2['n_seeds']})",
                       color="steelblue")
            axes[0].set_ylabel(r"$P_{2\mathrm{LPT}} / P_{1\mathrm{LPT}}$")
            axes[0].set_title("1LPT vs 2LPT (IC inicial)")
            axes[0].axhspan(0.99, 1.01, alpha=0.1, color="green")

        if ratios_solver:
            plot_ratio(ratios_solver, axes[1],
                       label=f"P_PM/P_TreePM (N_seeds={spm['n_seeds']})",
                       color="darkorange")
            axes[1].set_ylabel(r"$P_\mathrm{PM} / P_\mathrm{TreePM}$")
            axes[1].set_title("PM vs TreePM (evolucionado)")

        fig.suptitle(f"Comparaciones de ratios — Ensemble Fase 31", fontsize=12)
        fig.tight_layout()
        p3 = out_dir / "combined_ratios_ensemble.png"
        fig.savefig(p3, dpi=150)
        print(f"Figura combinada guardada: {p3}")
        plt.close(fig)


if __name__ == "__main__":
    main()
