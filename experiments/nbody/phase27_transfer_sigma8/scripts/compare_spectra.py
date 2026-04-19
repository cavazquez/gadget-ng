#!/usr/bin/env python3
"""
compare_spectra.py — Comparación de P(k) EH vs power-law puro.

Figura 1: P(k) objetivo (EH + n_s=0.965) vs P(k) ley de potencia pura.
Figura 2: P(k) medido desde snapshot inicial (EH) vs objetivo.
Figura 3: Pendiente efectiva del espectro medido.

Uso:
  python compare_spectra.py --eh    pk_eh_initial.json    \\
                             --pl    pk_pl_initial.json    \\
                             --out   compare_spectra.png

Formato de los JSON: lista de {"k": ..., "pk": ..., "n_modes": ...}
(salida del estimador gadget-ng-analysis::power_spectrum en formato gadget-ng).
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


# ── Eisenstein–Hu no-wiggle (Python reference implementation) ─────────────────

def eh_nowiggle_T(k_hmpc, omega_m=0.315, omega_b=0.049, h=0.674):
    """
    Función de transferencia EH no-wiggle (EH98, eq. 29-31).

    k_hmpc: número de onda en h/Mpc
    Retorna T(k) ∈ (0, 1]
    """
    if k_hmpc <= 0:
        return 1.0

    omh2 = omega_m * h**2
    obh2 = omega_b * h**2
    fb   = omega_b / omega_m

    s      = 44.5 * math.log(9.83 / omh2) / math.sqrt(1 + 10 * obh2**0.75)  # Mpc
    alphaG = 1 - 0.328 * math.log(431 * omh2) * fb + 0.38 * math.log(22.3 * omh2) * fb**2

    k_mpc = k_hmpc * h   # Mpc^{-1}
    ks    = 0.43 * k_mpc * s
    Geff  = omega_m * h * (alphaG + (1 - alphaG) / (1 + ks**4))

    q  = k_hmpc / max(Geff, 1e-30)
    L0 = math.log(math.e * 2 + 1.8 * q)
    C0 = 14.2 + 731 / (1 + 62.5 * q)
    return max(0.0, min(1.0, L0 / (L0 + C0 * q**2)))


def power_spectrum_target(k_hmpc_array, n_s=0.965, sigma8=0.8,
                          omega_m=0.315, omega_b=0.049, h=0.674):
    """
    Calcula P(k) = A² · k^n_s · T²(k) normalizado a σ₈.

    Retorna P(k) en (Mpc/h)³.
    """
    from scipy import integrate

    def tophat(x):
        if abs(x) < 1e-4:
            return 1.0 - x**2 / 10.0
        return 3 * (math.sin(x) - x * math.cos(x)) / x**3

    def integrand_sigma(lnk):
        k = math.exp(lnk)
        T = eh_nowiggle_T(k, omega_m, omega_b, h)
        W = tophat(k * 8.0)
        return k**(n_s + 3) * T**2 * W**2

    # Integral de σ²(8, A=1)
    result, _ = integrate.quad(integrand_sigma, math.log(1e-5), math.log(5e2))
    sigma_sq_unit = result / (2 * math.pi**2)
    A = sigma8 / math.sqrt(sigma_sq_unit)

    Pk = []
    for k in k_hmpc_array:
        if k <= 0:
            Pk.append(0.0)
        else:
            T = eh_nowiggle_T(k, omega_m, omega_b, h)
            Pk.append(A**2 * k**n_s * T**2)
    return np.array(Pk)


def power_spectrum_powerlaw(k_hmpc_array, n_s=0.965, amplitude=1e-3, box_mpc_h=100.0):
    """
    P(k) ∝ amplitude² · k^n_s (sin T(k)).

    Normalización: en unidades del grid con box = box_mpc_h Mpc/h.
    """
    Pk = []
    for k in k_hmpc_array:
        if k <= 0:
            Pk.append(0.0)
        else:
            Pk.append(amplitude**2 * k**n_s)
    return np.array(Pk)


# ── Carga de datos ────────────────────────────────────────────────────────────

def load_pk(path):
    """Carga bins de P(k) desde JSON."""
    with open(path) as f:
        data = json.load(f)
    k   = np.array([b["k"]  for b in data if b.get("pk", 0) > 0])
    pk  = np.array([b["pk"] for b in data if b.get("pk", 0) > 0])
    return k, pk


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eh",  required=True,  help="JSON P(k) EH inicial")
    parser.add_argument("--pl",  required=True,  help="JSON P(k) power-law inicial")
    parser.add_argument("--box", type=float, default=100.0, help="caja en Mpc/h")
    parser.add_argument("--out", default="compare_spectra.png")
    args = parser.parse_args()

    k_eh, pk_eh = load_pk(args.eh)
    k_pl, pk_pl = load_pk(args.pl)

    # Escala k del grid (adimensional) → h/Mpc
    # k_grid [1/box_internal] → k_hmpc = k_grid / box_mpc_h * 2π
    # Asumimos que el JSON ya trae k en unidades de 2π/L (k fundamental = 1).
    k_eh_phys = k_eh * 2 * math.pi / args.box  # h/Mpc
    k_pl_phys = k_pl * 2 * math.pi / args.box

    k_theory = np.geomspace(1e-2, 2.0, 200)
    pk_eh_theory = power_spectrum_target(k_theory)
    pk_pl_theory = power_spectrum_powerlaw(k_theory, amplitude=1e-3, box_mpc_h=args.box)

    if not HAS_MPL:
        print("matplotlib no disponible — solo diagnósticos de texto")
        print(f"Bins EH:  {len(k_eh)}, k=[{k_eh_phys.min():.3f}, {k_eh_phys.max():.3f}] h/Mpc")
        print(f"Bins PL:  {len(k_pl)}, k=[{k_pl_phys.min():.3f}, {k_pl_phys.max():.3f}] h/Mpc")

        # Diferencia relativa promedio entre espectros medidos
        if len(k_eh) == len(k_pl):
            rel_diff = np.abs(pk_eh - pk_pl) / (0.5 * (pk_eh + pk_pl) + 1e-30)
            print(f"Diferencia relativa promedio P(k) EH vs PL: {rel_diff.mean():.2%}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: P(k) medido vs objetivo
    ax = axes[0]
    ax.loglog(k_eh_phys, pk_eh, "o-", color="steelblue", ms=4, label="P(k) medido [EH]")
    ax.loglog(k_pl_phys, pk_pl, "s-", color="coral",    ms=4, label="P(k) medido [PL]")
    ax.loglog(k_theory, pk_eh_theory, "--", color="steelblue", alpha=0.6,
              label=r"$P(k)=A^2 k^{n_s}T^2_{EH}$ (objetivo)")
    ax.loglog(k_theory, pk_pl_theory, "--", color="coral", alpha=0.6,
              label=r"$P(k)\propto k^{n_s}$ (objetivo)")
    ax.set_xlabel("k [h/Mpc]")
    ax.set_ylabel(r"$P(k)$ [(Mpc/h)$^3$]")
    ax.set_title("P(k): EH vs power-law")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel 2: ratio EH / PL (solo si k coinciden)
    ax = axes[1]
    k_common = np.geomspace(1e-2, 1.5, 100)
    pk_eh_c  = power_spectrum_target(k_common)
    pk_pl_c  = power_spectrum_powerlaw(k_common, amplitude=1e-3, box_mpc_h=args.box)
    ratio = pk_eh_c / (pk_pl_c + 1e-30)

    ax.semilogx(k_common, ratio, "-", color="darkgreen", lw=2)
    ax.axhline(1.0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("k [h/Mpc]")
    ax.set_ylabel("P(k)[EH] / P(k)[PL]  (amplitud relativa)")
    ax.set_title(r"Ratio espectros (teoría): $T^2_{EH}(k)$")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.2)
    ax.text(0.02, 0.95, r"$T^2(k) \to 1$ para $k\to 0$",
            transform=ax.transAxes, fontsize=9, va="top")

    plt.suptitle("Fase 27: Función de transferencia Eisenstein–Hu\n"
                 r"Planck18: $\Omega_m=0.315$, $\Omega_b=0.049$, $h=0.674$, "
                 r"$n_s=0.965$, $\sigma_8=0.8$, caja=100 Mpc/h",
                 fontsize=9)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Figura guardada: {args.out}")


if __name__ == "__main__":
    main()
