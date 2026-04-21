#!/usr/bin/env python3
"""Genera las 5 figuras obligatorias de Phase 34.

  1. grid_ratio.png           — P_grid / P_cont por k (grilla pura)
  2. particle_ratio.png       — P_part / P_cont por k (tras CIC+deconv)
  3. stage_breakdown.png      — barra de A y CV por etapa
  4. cic_effect.png           — ratio con y sin deconvolución CIC
  5. single_mode_amplitude.png — amplitud esperada vs recuperada (modo único)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def plot_grid_ratio(input_dir: Path, fig_dir: Path) -> None:
    # Para la grilla pura usamos la etapa 3 del test b2_particle_vs_grid: A_grid
    # por seed. Como no guardamos el ratio por k de la grilla pura, usamos el
    # ratio por k de P_part (b2_cic_effect) como proxy visual del ratio
    # "después" y del global_offset para mostrar el plateau.
    go = load(input_dir / "b2_global_offset.json")
    ks = np.array(go["ks"])
    ratios = np.array(go["ratios"])
    a_mean = go["A_mean"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.axhline(a_mean, color="k", ls="--", lw=1, label=f"A_mean = {a_mean:.2e}")
    ax.plot(ks, ratios, "o-", color="tab:blue", label="P_m(k) / P_cont(k)")
    ax.fill_between(
        ks,
        a_mean * (1 - go["CV_ratio"]),
        a_mean * (1 + go["CV_ratio"]),
        color="gray",
        alpha=0.2,
        label=f"±CV = {go['CV_ratio']:.3f}",
    )
    ax.set_xlabel("k (2π/L_internal)")
    ax.set_ylabel("P_measured / P_continuous")
    ax.set_yscale("log")
    ax.set_title("Phase 34 · Ratio P_m/P_cont tras pipeline completo (N=32, seed 42)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "grid_ratio.png", dpi=120)
    plt.close(fig)


def plot_particle_ratio(input_dir: Path, fig_dir: Path) -> None:
    pvg = load(input_dir / "b2_particle_vs_grid.json")
    seeds = [row["seed"] for row in pvg["per_seed"]]
    a_grid = [row["a_grid"] for row in pvg["per_seed"]]
    a_part = [row["a_part"] for row in pvg["per_seed"]]
    ratios = [row["ratio_part_over_grid"] for row in pvg["per_seed"]]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.plot(seeds, a_grid, "o-", label="A_grid (grilla pura)", color="tab:green")
    ax.plot(seeds, a_part, "s--", label="A_part (tras partículas)", color="tab:red")
    ax.set_xlabel("seed")
    ax.set_ylabel("A = ⟨P_m / P_cont⟩")
    ax.set_yscale("log")
    ax.set_title(f"A_grid vs A_part por seed (N={pvg['n']})")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(seeds, ratios, "o-", color="tab:purple")
    ax.axhline(pvg["ratio_part_over_grid_mean"], ls="--", color="k",
               label=f"media = {pvg['ratio_part_over_grid_mean']:.4f}")
    ax.set_xlabel("seed")
    ax.set_ylabel("A_part / A_grid")
    ax.set_title(
        f"Factor partícula/grilla (CV = {pvg['ratio_part_over_grid_cv']:.4f})"
    )
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "particle_ratio.png", dpi=120)
    plt.close(fig)


def plot_stage_breakdown(input_dir: Path, fig_dir: Path) -> None:
    pvg = load(input_dir / "b2_particle_vs_grid.json")
    cic = load(input_dir / "b2_cic_effect.json")
    seeds = load(input_dir / "b3_seeds.json")

    labels = [
        "IFFT roundtrip",
        "ruido blanco σ²·N³",
        "A_grid (grilla pura)",
        "A_part / A_grid",
        "slope |deconv|",
        "slope |raw|",
        "CV(A) seeds",
    ]
    roundtrip = load(input_dir / "b1_roundtrip.json")
    wn = load(input_dir / "b1_white_noise.json")
    values = [
        roundtrip["max_abs_error"],
        abs(1.0 - wn["ratio"]),
        pvg["a_grid_mean"],
        pvg["ratio_part_over_grid_mean"],
        abs(cic["slope_deconv"]),
        abs(cic["slope_raw"]),
        seeds["CV_A"],
    ]
    errs = [
        0.0,
        0.0,
        0.0,
        pvg["ratio_part_over_grid_cv"] * pvg["ratio_part_over_grid_mean"],
        0.0,
        0.0,
        0.0,
    ]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    xs = np.arange(len(labels))
    bars = ax.bar(xs, values, yerr=errs, color="tab:blue", edgecolor="k")
    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            max(v, 1e-16) * 1.2,
            f"{v:.2e}" if abs(v) < 1e-3 or abs(v) > 1e3 else f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("valor (escala log)")
    ax.set_title("Phase 34 · Descomposición del offset por etapa")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "stage_breakdown.png", dpi=120)
    plt.close(fig)


def plot_cic_effect(input_dir: Path, fig_dir: Path) -> None:
    cic = load(input_dir / "b2_cic_effect.json")
    ks_d = np.array(cic["ks_deconv"])
    rs_d = np.array(cic["ratios_deconv"])
    ks_r = np.array(cic["ks_raw"])
    rs_r = np.array(cic["ratios_raw"])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ks_r, rs_r, "s--", color="tab:red",
            label=f"raw (slope={cic['slope_raw']:.3f})")
    ax.plot(ks_d, rs_d, "o-", color="tab:blue",
            label=f"deconvolucionado (slope={cic['slope_deconv']:.3f})")
    ax.set_xlabel("k (2π/L_internal)")
    ax.set_ylabel("P_m / P_cont")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    ax.set_title(
        f"Efecto de la deconvolución CIC (N={cic['n']}, seed {cic['seed']})"
    )
    fig.tight_layout()
    fig.savefig(fig_dir / "cic_effect.png", dpi=120)
    plt.close(fig)


def plot_single_mode(input_dir: Path, fig_dir: Path) -> None:
    sm = load(input_dir / "b1_single_mode.json")
    n = sm["n"]
    amp = sm["amplitude"]
    expected = sm["expected_real_peak"]
    ks = np.array([-1, 0, 1])  # sólo tres modos relevantes sobre eje x
    expected_vals = np.array([amp, 0.0, amp])
    recovered_vals = np.array([
        np.hypot(sm["recovered_neg"]["re"], sm["recovered_neg"]["im"]),
        sm["max_amplitude_in_other_modes"],
        np.hypot(sm["recovered_pos"]["re"], sm["recovered_pos"]["im"]),
    ])

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    width = 0.35
    ax.bar(ks - width / 2, expected_vals, width, label="esperado", color="tab:green")
    ax.bar(ks + width / 2, recovered_vals, width, label="recuperado", color="tab:blue")
    ax.set_xticks(ks)
    ax.set_xticklabels([f"−k₀", "0", f"+k₀"])
    ax.set_ylabel("|δ̂(k)|")
    ax.set_title(
        f"Modo único: N={n}, A={amp}, pico real esperado={expected:.4f}"
    )
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "single_mode_amplitude.png", dpi=120)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--fig-dir", required=True)
    args = ap.parse_args()

    input_dir = Path(args.input)
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_grid_ratio(input_dir, fig_dir)
    plot_particle_ratio(input_dir, fig_dir)
    plot_stage_breakdown(input_dir, fig_dir)
    plot_cic_effect(input_dir, fig_dir)
    plot_single_mode(input_dir, fig_dir)
    print(f"Figuras generadas en {fig_dir}")


if __name__ == "__main__":
    main()
