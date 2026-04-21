#!/usr/bin/env python3
"""Genera las 5 figuras obligatorias de Phase 35.

Lee los JSONs de `target/phase35/` y el modelo de `output/rn_model.json`.
Escribe PNGs a `figures/`.

Uso:
    python plot_r_n.py --target-dir ../../../target/phase35 \
                       --model output/rn_model.json \
                       --output-dir ../figures
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def plot_rn_vs_n(target_dir: Path, model: dict, out: Path) -> None:
    rows = load_json(target_dir / "rn_by_seed.json")["per_n"]
    ns = np.array([row["n"] for row in rows], dtype=float)
    r_means = np.array([row["r_mean"] for row in rows], dtype=float)
    r_stds = np.array(
        [np.std(row["r_list"], ddof=0) if len(row["r_list"]) > 1 else 0.0
         for row in rows]
    )

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.errorbar(ns, r_means, yerr=r_stds, fmt="o", color="C0", label="datos (4 seeds)")
    nfine = np.logspace(np.log10(ns.min() * 0.9), np.log10(ns.max() * 1.1), 200)
    mA = model["model_a"]
    ax.plot(
        nfine,
        mA["c"] * nfine ** (-mA["alpha"]),
        "-",
        color="C1",
        label=f"Modelo A: C·N^(-α), α={mA['alpha']:.3f}",
    )
    mB = model.get("model_b")
    if mB and math.isfinite(mB.get("aic", float("nan"))):
        ax.plot(
            nfine,
            mB["c"] * nfine ** (-mB["alpha"]) + mB["r_inf"],
            "--",
            color="C2",
            label=f"Modelo B: +R∞, R∞={mB['r_inf']:.2e}",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N (celdas por eje)")
    ax.set_ylabel(r"$R(N) = P_\mathrm{m}/(A_\mathrm{grid}\,P_\mathrm{cont})$")
    winner = model["selection"]["winner"]
    ax.set_title(f"Phase 35 — R(N) vs N (ganador: Modelo {winner})")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[plot] {out}")


def plot_rn_of_k(target_dir: Path, out: Path) -> None:
    data = load_json(target_dir / "rn_of_k.json")
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    colors = plt.get_cmap("viridis")(np.linspace(0.1, 0.9, len(data["per_n"])))
    for color, entry in zip(colors, data["per_n"]):
        ks = np.asarray(entry["ks"], dtype=float)
        rs = np.asarray(entry["ratios_avg_over_seeds"], dtype=float)
        if ks.size == 0:
            continue
        ax.plot(ks, rs, marker="o", ms=3, color=color, label=f"N={entry['n']}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("k [2π / box]")
    ax.set_ylabel("R(N, k)  (promediado en seeds)")
    ax.set_title("Phase 35 — R(N, k) vs k (k ≤ k_Nyq/2)")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[plot] {out}")


def plot_fit_residuals(model: dict, out: Path) -> None:
    rows = model["table"]
    ns = np.array([r["n"] for r in rows], dtype=float)
    r_obs = np.array([r["r_mean"] for r in rows], dtype=float)
    mA = model["model_a"]
    mB = model["model_b"]
    pred_a = mA["c"] * ns ** (-mA["alpha"])
    pred_b = mB["c"] * ns ** (-mB["alpha"]) + mB["r_inf"]
    res_a = np.log10(r_obs / pred_a)
    res_b = np.log10(r_obs / pred_b)

    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.axhline(0.0, color="k", lw=0.7)
    ax.plot(ns, res_a, "o-", color="C1", label=f"Modelo A (RMS={mA['rms_log10_residual']:.3f})")
    ax.plot(ns, res_b, "s--", color="C2", label=f"Modelo B (RMS={mB['rms_log10_residual']:.3f})")
    ax.set_xscale("log")
    ax.set_xlabel("N")
    ax.set_ylabel(r"$\log_{10}(R_\mathrm{obs}/R_\mathrm{model})$")
    ax.set_title("Phase 35 — Residuos del fit")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[plot] {out}")


def plot_p_corrected_vs_theory(target_dir: Path, out: Path) -> None:
    data = load_json(target_dir / "rn_correction_error.json")
    per_n = data["per_n"]
    ns = np.array([row["n"] for row in per_n], dtype=float)
    raw = np.array([row["median_abs_log_err_raw"] for row in per_n])
    cor = np.array([row["median_abs_log_err_corrected"] for row in per_n])

    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    width = 0.35
    idx = np.arange(len(ns))
    ax.bar(idx - width / 2, raw, width, label="sin corrección", color="C3")
    ax.bar(idx + width / 2, cor, width, label="con modelo A + A_grid", color="C0")
    ax.set_yscale("log")
    ax.set_xticks(idx)
    ax.set_xticklabels([f"N={int(n)}" for n in ns])
    ax.set_ylabel(r"Mediana $|\log_{10}(P/P_\mathrm{cont})|$")
    ax.set_title("Phase 35 — Error de amplitud antes/después de la corrección")
    ax.grid(True, axis="y", which="both", ls=":", alpha=0.5)
    ax.legend()
    for x, yr, yc in zip(idx, raw, cor):
        ax.text(x - width / 2, yr * 1.1, f"{yr:.2g}", ha="center", fontsize=8)
        ax.text(x + width / 2, yc * 1.1, f"{yc:.2g}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[plot] {out}")


def plot_tsc_vs_cic(target_dir: Path, out: Path) -> None:
    data = load_json(target_dir / "tsc_vs_cic.json")
    fig, ax = plt.subplots(figsize=(5, 4.2))
    labels = ["CIC", "TSC"]
    values = [data["r_cic"], data["r_tsc"]]
    colors = ["C0", "C2"]
    ax.bar(labels, values, color=colors)
    for i, v in enumerate(values):
        ax.text(i, v * 1.02, f"{v:.4g}", ha="center", fontsize=10)
    ax.set_ylabel("R_mean(N=32, seed=42)")
    ax.set_title(f"Phase 35 — CIC vs TSC (ratio = {data['ratio_max_over_min']:.3f})")
    ax.grid(True, axis="y", ls=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[plot] {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    tgt = Path(args.target_dir)
    model = load_json(Path(args.model))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    plot_rn_vs_n(tgt, model, out / "rn_vs_N.png")
    plot_rn_of_k(tgt, out / "rn_of_k.png")
    plot_fit_residuals(model, out / "fit_residuals.png")
    plot_p_corrected_vs_theory(tgt, out / "p_corrected_vs_theory.png")
    plot_tsc_vs_cic(tgt, out / "tsc_vs_cic.png")


if __name__ == "__main__":
    main()
