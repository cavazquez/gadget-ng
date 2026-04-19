#!/usr/bin/env python3
"""
validate_sigma8.py — Validación de la normalización σ₈ del campo generado.

Calcula σ₈ desde el P(k) medido del campo de partículas inicial y compara
con el target σ₈ = 0.8 especificado en la configuración.

Uso:
  python validate_sigma8.py --pk pk_initial.json \\
                             --box 100.0          \\
                             --sigma8 0.8         \\
                             --out validate_sigma8.png

Formato JSON: lista de {"k": ..., "pk": ..., "n_modes": ...}
donde k está en unidades del grid (k_fund = 1, k_Nyq = N/2).
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


def tophat_window(x):
    """W(x) = 3[sin(x) - x cos(x)] / x³, con W(0) = 1."""
    if abs(x) < 1e-4:
        return 1.0 - x**2 / 10.0
    return 3 * (math.sin(x) - x * math.cos(x)) / x**3


def sigma_from_bins(k_hmpc, pk_mpc_h3, R_mpc_h=8.0):
    """
    Calcula σ(R) desde bins de P(k).

    σ²(R) = (1/2π²) ∫ k² P(k) W²(kR) dk
           ≈ (1/2π²) Σ k² P(k) W²(kR) Δk

    k_hmpc : array de k en h/Mpc
    pk_mpc_h3 : array de P(k) en (Mpc/h)³
    R_mpc_h : radio del filtro top-hat en Mpc/h
    """
    sigma_sq = 0.0
    n = len(k_hmpc)
    for i in range(n):
        k  = k_hmpc[i]
        pk = pk_mpc_h3[i]
        if k <= 0 or pk <= 0:
            continue
        x  = k * R_mpc_h
        W  = tophat_window(x)
        # Δk desde diferencias finitas
        if n > 1:
            if i == 0:
                dk = k_hmpc[1] - k_hmpc[0]
            elif i == n - 1:
                dk = k_hmpc[i] - k_hmpc[i - 1]
            else:
                dk = 0.5 * (k_hmpc[i + 1] - k_hmpc[i - 1])
        else:
            dk = k * 0.1
        sigma_sq += k**2 * pk * W**2 * dk
    return math.sqrt(max(0.0, sigma_sq / (2 * math.pi**2)))


def sigma_curve(R_array_mpc_h, k_hmpc, pk_mpc_h3):
    """Calcula σ(R) para un rango de radios."""
    return [sigma_from_bins(k_hmpc, pk_mpc_h3, R) for R in R_array_mpc_h]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pk",     required=True,  help="JSON de P(k) inicial")
    parser.add_argument("--box",    type=float, default=100.0, help="caja en Mpc/h")
    parser.add_argument("--sigma8", type=float, default=0.8,   help="σ₈ objetivo")
    parser.add_argument("--h",      type=float, default=0.674, help="parámetro h de Hubble")
    parser.add_argument("--out",    default="validate_sigma8.png")
    args = parser.parse_args()

    with open(args.pk) as f:
        data = json.load(f)

    # Filtrar bins con señal
    bins = [(b["k"], b["pk"]) for b in data if b.get("pk", 0) > 0 and b.get("k", 0) > 0]
    if not bins:
        print("ERROR: No hay bins con señal en el P(k)")
        return

    k_grid  = np.array([b[0] for b in bins])
    pk_grid = np.array([b[1] for b in bins])

    # Convertir unidades: k_grid [1/box_internal] → k_hmpc = k_grid × 2π/box_mpc_h
    # P_grid [box_internal³] → P_mpc_h3 = P_grid × (box_mpc_h / 2π)³
    # Nota: gadget-ng mide k en múltiplos de k_fund = 1 (unidades del grid).
    # La conversión depende de si k está en unidades de 2π/L o 1/L.
    # Asumimos k en 2π/L → k_hmpc = k_grid × h / box_mpc_h × (2π × box_mpc_h / (2π)) = k_grid × h / box_mpc_h
    # ¡REVISAR según la salida del estimador gadget-ng!
    k_hmpc  = k_grid * args.h / args.box      # h/Mpc
    pk_phys = pk_grid * (args.box / args.h)**3  # (Mpc/h)³

    # Calcular σ₈ medido
    sigma8_measured = sigma_from_bins(k_hmpc, pk_phys, R_mpc_h=8.0)
    rel_err = (sigma8_measured - args.sigma8) / args.sigma8

    print(f"σ₈ target  = {args.sigma8:.4f}")
    print(f"σ₈ medido  = {sigma8_measured:.4f}")
    print(f"Error relativo = {rel_err:+.2%}")
    print()
    print(f"Rango k: [{k_hmpc.min():.4f}, {k_hmpc.max():.4f}] h/Mpc")
    print(f"Bins con señal: {len(bins)}")

    if abs(rel_err) < 0.05:
        print("✓ σ₈ dentro del 5% del objetivo")
    elif abs(rel_err) < 0.20:
        print("⚠ σ₈ entre 5% y 20% del objetivo (grid pequeño, resolución limitada)")
    else:
        print("✗ σ₈ fuera del 20% del objetivo — revisar normalización")

    if not HAS_MPL:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: P(k) medido vs objetivo
    ax = axes[0]
    ax.loglog(k_hmpc, pk_phys, "o-", color="steelblue", ms=5, label="P(k) medido")

    # Curva analítica EH (requiere scipy)
    try:
        from compare_spectra import power_spectrum_target
        k_th = np.geomspace(k_hmpc.min() / 2, k_hmpc.max() * 2, 200)
        pk_th = power_spectrum_target(k_th, sigma8=args.sigma8)
        ax.loglog(k_th, pk_th, "--", color="steelblue", alpha=0.6,
                  label=r"$P(k)=A^2 k^{n_s} T^2_{EH}$ (objetivo)")
    except Exception:
        pass

    ax.set_xlabel("k [h/Mpc]")
    ax.set_ylabel(r"$P(k)$ [(Mpc/h)$^3$]")
    ax.set_title("P(k) medido vs objetivo EH")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.text(0.05, 0.05, f"σ₈ medido = {sigma8_measured:.3f}\nσ₈ target = {args.sigma8:.3f}\n"
            f"err = {rel_err:+.1%}",
            transform=ax.transAxes, fontsize=9, va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow"))

    # Panel 2: curva σ(R)
    ax = axes[1]
    R_arr = np.geomspace(0.5, 50.0, 80)
    sigma_arr = sigma_curve(R_arr, k_hmpc, pk_phys)

    ax.loglog(R_arr, sigma_arr, "-", color="steelblue", lw=2, label="σ(R) medido")
    ax.axvline(8.0, color="red", ls="--", lw=1.2, label="R=8 Mpc/h")
    ax.axhline(args.sigma8, color="green", ls=":", lw=1.2, label=f"σ₈ target={args.sigma8}")
    ax.axhline(sigma8_measured, color="coral", ls="-.", lw=1.2,
               label=f"σ₈ medido={sigma8_measured:.3f}")
    ax.set_xlabel("R [Mpc/h]")
    ax.set_ylabel("σ(R)")
    ax.set_title("Curva σ(R) con filtro top-hat")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.suptitle("Fase 27: Validación σ₈\n"
                 f"Planck18, caja={args.box} Mpc/h, N=32³",
                 fontsize=9)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"\nFigura guardada: {args.out}")


if __name__ == "__main__":
    main()
