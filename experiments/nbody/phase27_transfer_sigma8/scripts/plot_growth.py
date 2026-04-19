#!/usr/bin/env python3
"""
plot_growth.py — Crecimiento lineal de estructura con ICs Eisenstein–Hu.

Compara el crecimiento relativo de delta_rms(a) vs D(a)/D(a_init) esperado
por teoría lineal para ΛCDM.

Uso:
  python plot_growth.py --diag diagnostics.json \\
                         --omega_m 0.315          \\
                         --omega_lambda 0.685      \\
                         --a_init 0.02             \\
                         --out plot_growth.png

Formato diagnostics.json: lista de {"a": ..., "delta_rms": ..., "v_rms": ...}
"""

import argparse
import json
import math
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ── Factor de crecimiento lineal para ΛCDM ────────────────────────────────────

def hubble_normalized(a, omega_m=0.315, omega_lambda=0.685):
    """H(a)/H₀ = sqrt(Ω_m/a³ + Ω_Λ)."""
    return math.sqrt(omega_m / a**3 + omega_lambda)


def growth_factor_D(a, omega_m=0.315, omega_lambda=0.685, n_steps=1000):
    """
    Factor de crecimiento D(a) normalizado a D(1) = 1 para ΛCDM.

    Integral: D(a) ∝ H(a) ∫_0^a da'/(a' H(a'))³
    (Carroll, Press & Turner 1992, eq. A12 simplificada).

    Nota: esta aproximación es válida para ΛCDM plana (Ω_m + Ω_Λ = 1).
    """
    def integrand(ap):
        Hp = hubble_normalized(ap, omega_m, omega_lambda)
        return 1.0 / (ap * Hp)**3

    # Integral numérica desde a_min hasta a
    a_min = 1e-4
    a_arr = np.linspace(a_min, a, n_steps)
    da = a_arr[1] - a_arr[0]
    integral = sum(integrand(ap) * da for ap in a_arr)
    H_a = hubble_normalized(a, omega_m, omega_lambda)
    return H_a * integral


def growth_factor_normalized(a_arr, a_init, omega_m=0.315, omega_lambda=0.685):
    """
    Retorna D(a)/D(a_init) para un array de a.
    """
    D_init = growth_factor_D(a_init, omega_m, omega_lambda)
    return np.array([growth_factor_D(a, omega_m, omega_lambda) / D_init
                     for a in a_arr])


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diag",      required=True,   help="JSON de diagnósticos")
    parser.add_argument("--diag_pl",   default=None,    help="JSON de diagnósticos power-law (comparación)")
    parser.add_argument("--omega_m",   type=float, default=0.315)
    parser.add_argument("--omega_l",   type=float, default=0.685)
    parser.add_argument("--a_init",    type=float, default=0.02)
    parser.add_argument("--out",       default="plot_growth.png")
    args = parser.parse_args()

    with open(args.diag) as f:
        diag = json.load(f)

    a_sim       = np.array([d["a"]         for d in diag])
    delta_rms   = np.array([d["delta_rms"] for d in diag])

    has_pl = False
    if args.diag_pl:
        try:
            with open(args.diag_pl) as f:
                diag_pl = json.load(f)
            a_pl       = np.array([d["a"]         for d in diag_pl])
            delta_pl   = np.array([d["delta_rms"] for d in diag_pl])
            has_pl = True
        except Exception as e:
            print(f"Advertencia: no se pudo cargar {args.diag_pl}: {e}")

    # Crecimiento lineal teórico
    a_theory = np.linspace(a_sim.min(), a_sim.max() * 1.1, 200)
    D_theory = growth_factor_normalized(a_theory, args.a_init,
                                        args.omega_m, args.omega_l)

    # Normalizar delta_rms medido a 1 en a_init (primera entrada)
    delta0 = delta_rms[0] if delta_rms[0] > 0 else 1.0
    delta_normalized = delta_rms / delta0

    print("Crecimiento relativo δ_rms:")
    for a, d in zip(a_sim, delta_normalized):
        print(f"  a={a:.4f}  δ_rms/δ₀={d:.4f}")

    # Comparar con teoría
    D_at_sim = growth_factor_normalized(a_sim, args.a_init, args.omega_m, args.omega_l)
    ratio = delta_normalized / D_at_sim
    print(f"\nRatio δ_medido/D_teoría: {ratio.mean():.3f} ± {ratio.std():.3f}")

    if not HAS_MPL:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: crecimiento relativo
    ax = axes[0]
    ax.plot(a_sim, delta_normalized, "o-", color="steelblue", ms=5,
            label="δ_rms medido / δ₀ [EH]")
    ax.plot(a_theory, D_theory, "--", color="gray", lw=2,
            label="D(a)/D(a_init) [ΛCDM lineal]")
    if has_pl:
        delta0_pl = delta_pl[0] if delta_pl[0] > 0 else 1.0
        ax.plot(a_pl, delta_pl / delta0_pl, "s-", color="coral", ms=5,
                label="δ_rms medido / δ₀ [power-law]")

    ax.set_xlabel("a (factor de escala)")
    ax.set_ylabel("δ_rms / δ_rms(a_init)")
    ax.set_title("Crecimiento lineal de δ_rms")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: error relativo respecto a teoría
    ax = axes[1]
    err_pct = (delta_normalized / D_at_sim - 1) * 100
    ax.plot(a_sim, err_pct, "o-", color="steelblue", ms=5, label="EH vs ΛCDM lineal")
    ax.axhline(0, color="gray", ls="--", lw=1)
    ax.axhline(+10, color="orange", ls=":", lw=0.8, alpha=0.7)
    ax.axhline(-10, color="orange", ls=":", lw=0.8, alpha=0.7)
    ax.set_xlabel("a (factor de escala)")
    ax.set_ylabel("Error relativo (%)")
    ax.set_title("δ_rms / D_teoría − 1")
    ax.set_ylim(-30, 30)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle("Fase 27: Crecimiento lineal con ICs Eisenstein–Hu\n"
                 f"ΛCDM: Ω_m={args.omega_m}, Ω_Λ={args.omega_l}, a_init={args.a_init}",
                 fontsize=9)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"\nFigura guardada: {args.out}")


if __name__ == "__main__":
    main()
