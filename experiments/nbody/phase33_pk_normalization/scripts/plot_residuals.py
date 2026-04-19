#!/usr/bin/env python3
"""
plot_residuals.py — Fase 33.

Genera las figuras del análisis de normalización:

  1. `pk_measured_vs_theory.png` — log-log, P_mean(k) del ensemble vs P_EH(k).
  2. `r_of_k.png` — R(k) = P_mean / P_EH sin corregir, con barras de error.
  3. `r_of_k_corrected.png` — R(k) / A_pred, debe fluctuar cerca de 1.
  4. `cic_effect.png` — pendiente de R(k) con W²(k) reintroducida vs deconvolucionada.

Todas las figuras incluyen: N, n_seeds, A_pred y log_ratio.

Uso:
    python plot_residuals.py \
        --stats-file ../output/stats_N32.json \
        --grid-size 32 \
        --box-internal 1.0 \
        --out-dir ../figures
"""

import argparse
import json
import math
import os
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("ERROR: matplotlib no está instalado. Instala con: pip install matplotlib")
    raise


def a_pred_minimal(n, box_internal=1.0):
    return (box_internal ** 6) / (n ** 9)


def load_stats(path):
    with open(path) as f:
        return json.load(f)


def sinc_numpy(x):
    """sinc(πx) / πx sin numpy: puro Python."""
    if abs(x) < 1e-12:
        return 1.0
    return math.sin(math.pi * x) / (math.pi * x)


def cic_window_cubed_sq(k_internal, n, box_internal=1.0):
    """
    W²(k) ≈ sinc⁶(k·Δx / (2π)) con Δx = box_internal/N.
    Aproximado usando el módulo de k; el código real aplica la ventana por
    componente, lo cual es más preciso.
    """
    n_abs = k_internal * box_internal / (2.0 * math.pi)
    w1 = sinc_numpy(n_abs / n) if n_abs > 0 else 1.0
    return (w1 ** 3) ** 2


def plot_measured_vs_theory(stats, out_path, n_grid, n_seeds, a_pred):
    bins = stats["bins"]
    k = [b["k_hmpc"] for b in bins if b.get("p_mean", 0) > 0]
    p_m = [b["p_mean_hmpc"] for b in bins if b.get("p_mean", 0) > 0]
    p_err = [b.get("p_stderr_hmpc", 0.0) for b in bins if b.get("p_mean", 0) > 0]
    p_eh = [b["pk_eh"] for b in bins if b.get("p_mean", 0) > 0]

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.errorbar(k, p_m, yerr=p_err, fmt="o-", label="P_measured (ensemble)", capsize=3)
    ax.plot(k, p_eh, "--", label="P_EH (teoría)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("k  [h/Mpc]")
    ax.set_ylabel("P(k)  [(Mpc/h)³]")
    ax.set_title(
        f"P_measured vs P_EH — N={n_grid}³, {n_seeds} seeds\n"
        f"A_pred = {a_pred:.3e}"
    )
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  {out_path}")


def plot_r_of_k(stats, out_path, n_grid, n_seeds, corrected=False, a_pred=None):
    bins = stats["bins"]
    k = []
    r = []
    r_err = []
    for b in bins:
        if b.get("p_mean_hmpc", 0) > 0 and b.get("pk_eh", 0) > 0:
            r_val = b["p_mean_hmpc"] / b["pk_eh"]
            r_err_val = b.get("p_stderr_hmpc", 0.0) / b["pk_eh"]
            if corrected and a_pred:
                r_val /= a_pred
                r_err_val /= a_pred
            k.append(b["k_hmpc"])
            r.append(r_val)
            r_err.append(r_err_val)

    if not k:
        print(f"  SKIP {out_path}: no hay bins válidos")
        return

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.errorbar(k, r, yerr=r_err, fmt="o-", capsize=3,
                label="R(k) / A_pred" if corrected else "R(k) = P_m/P_EH")
    if corrected:
        ax.axhline(1.0, color="red", linestyle="--", label="y = 1")
    else:
        mean_r = sum(r) / len(r)
        ax.axhline(mean_r, color="red", linestyle="--",
                   label=f"⟨R(k)⟩ = {mean_r:.2e}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("k  [h/Mpc]")
    ax.set_ylabel("R(k) / A_pred" if corrected else "R(k) = P_m / P_EH")
    title = (
        f"R(k) corregido por A_pred = {a_pred:.3e} — N={n_grid}³, {n_seeds} seeds"
        if corrected
        else f"R(k) = P_m / P_EH (sin corregir) — N={n_grid}³, {n_seeds} seeds"
    )
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  {out_path}")


def plot_cic_effect(stats, out_path, n_grid, n_seeds, box_internal):
    """
    R(k) deconvolucionado (como entrega el estimador actual) vs
    R(k) reintroducida W²(k) (caso "sin deconvolución").
    """
    bins = stats["bins"]
    k_int = []
    r_dec = []
    r_raw = []
    for b in bins:
        if b.get("p_mean_hmpc", 0) > 0 and b.get("pk_eh", 0) > 0:
            r_val = b["p_mean_hmpc"] / b["pk_eh"]
            w_sq = cic_window_cubed_sq(b["k"], n_grid, box_internal)
            k_int.append(b["k"])
            r_dec.append(r_val)
            r_raw.append(r_val * w_sq)

    if not k_int:
        print(f"  SKIP {out_path}: no hay bins válidos")
        return

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.plot(k_int, r_dec, "o-", label="R(k) deconvolucionado (estimador actual)")
    ax.plot(k_int, r_raw, "s--", label="R(k) × W²(k)  (reintroducido)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("k_internal  [2π / BOX]")
    ax.set_ylabel("R(k) = P_m / P_EH")
    ax.set_title(
        f"Efecto de la deconvolución CIC — N={n_grid}³, {n_seeds} seeds"
    )
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats-file", required=True,
                        help="Archivo JSON de compute_ensemble_stats.py")
    parser.add_argument("--grid-size", type=int, required=True,
                        help="Tamaño de grid N (para A_pred y titles)")
    parser.add_argument("--box-internal", type=float, default=1.0)
    parser.add_argument("--out-dir", default="../figures",
                        help="Directorio de salida de las figuras")
    args = parser.parse_args()

    stats = load_stats(args.stats_file)
    n_seeds = stats.get("n_seeds", 0)
    a_pred = a_pred_minimal(args.grid_size, args.box_internal)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fase 33 — figuras para N={args.grid_size}³ ({n_seeds} seeds)")
    print(f"  A_pred (V²/N⁹) = {a_pred:.4e}")
    print()

    plot_measured_vs_theory(
        stats,
        out_dir / f"pk_measured_vs_theory_N{args.grid_size}.png",
        args.grid_size, n_seeds, a_pred,
    )
    plot_r_of_k(
        stats,
        out_dir / f"r_of_k_N{args.grid_size}.png",
        args.grid_size, n_seeds, corrected=False, a_pred=a_pred,
    )
    plot_r_of_k(
        stats,
        out_dir / f"r_of_k_corrected_N{args.grid_size}.png",
        args.grid_size, n_seeds, corrected=True, a_pred=a_pred,
    )
    plot_cic_effect(
        stats,
        out_dir / f"cic_effect_N{args.grid_size}.png",
        args.grid_size, n_seeds, args.box_internal,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
