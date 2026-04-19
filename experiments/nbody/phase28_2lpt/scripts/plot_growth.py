#!/usr/bin/env python3
"""
plot_growth.py — Crecimiento del contraste de densidad δ_rms vs D(a).

Compara la evolución temporal de 1LPT vs 2LPT para verificar que ambas
siguen el factor de crecimiento lineal D(a) al nivel esperado.

En el régimen lineal: δ_rms(a) ∝ D₁(a). Las ICs 2LPT corrigen la
distribución inicial pero no cambian el crecimiento lineal posterior.

Uso:
  python plot_growth.py \\
      --diag1lpt   diagnostics_1lpt.json \\
      --diag2lpt   diagnostics_2lpt.json \\
      --a_init     0.02 \\
      --omega_m    0.315 \\
      --omega_l    0.685 \\
      --out        plot_growth.png

Formato de diagnostics JSON:
  Lista de {"step": int, "a": float, "delta_rms": float, "vrms": float}
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


# ── Factor de crecimiento numérico ΛCDM ──────────────────────────────────────

def growth_integrand(a, omega_m, omega_l):
    """Integrando de D₁(a) = H(a) ∫_0^a da'/(a'·H(a'))³."""
    ha2 = omega_m / a**3 + omega_l
    return 1.0 / (a * math.sqrt(ha2))**3 if ha2 > 0 else 0.0


def growth_factor_D1(a_arr, omega_m=0.315, omega_l=0.685, n_steps=1000):
    """
    Calcula D₁(a) normalizada a D₁(1) = 1 para una lista de valores de a.

    Usa integración de Simpson sobre log(a).
    """
    from scipy import integrate

    def D_unnorm(a):
        if a <= 0:
            return 0.0
        ha = math.sqrt(omega_m / a**3 + omega_l)
        result, _ = integrate.quad(
            growth_integrand, 1e-5, a,
            args=(omega_m, omega_l),
            limit=200
        )
        return ha * result

    D1 = D_unnorm(1.0)
    return np.array([D_unnorm(a) / D1 for a in a_arr])


# ── Carga de datos ────────────────────────────────────────────────────────────

def load_diagnostics(path):
    """Carga diagnósticos desde JSON → dict of arrays."""
    with open(path) as f:
        data = json.load(f)
    return {
        "step":      np.array([d["step"]      for d in data]),
        "a":         np.array([d["a"]         for d in data]),
        "delta_rms": np.array([d["delta_rms"] for d in data]),
        "vrms":      np.array([d.get("vrms", 0.0) for d in data]),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diag1lpt", required=True, help="JSON diagnósticos 1LPT")
    parser.add_argument("--diag2lpt", required=True, help="JSON diagnósticos 2LPT")
    parser.add_argument("--a_init",   type=float, default=0.02)
    parser.add_argument("--omega_m",  type=float, default=0.315)
    parser.add_argument("--omega_l",  type=float, default=0.685)
    parser.add_argument("--out",      default="plot_growth.png")
    args = parser.parse_args()

    d1 = load_diagnostics(args.diag1lpt)
    d2 = load_diagnostics(args.diag2lpt)

    # Factor de crecimiento teórico D₁(a) normalizado a D₁(a_init)
    a_all = np.union1d(d1["a"], d2["a"])
    a_all = np.sort(a_all[a_all > 0])

    try:
        D1_arr = growth_factor_D1(a_all, args.omega_m, args.omega_l)
        D1_init = float(growth_factor_D1([args.a_init], args.omega_m, args.omega_l)[0])
    except Exception:
        print("scipy no disponible o error de integración — se omite D₁(a) teórico")
        D1_arr = None

    # Normalizar δ_rms(a_init) a 1 para comparación de escala
    delta_rms_1lpt_0 = d1["delta_rms"][0] if len(d1["delta_rms"]) > 0 else 1.0
    delta_rms_2lpt_0 = d2["delta_rms"][0] if len(d2["delta_rms"]) > 0 else 1.0

    # ── Diagnósticos de texto ─────────────────────────────────────────────────
    print("=" * 60)
    print("CRECIMIENTO DE DENSIDAD: 1LPT vs 2LPT — Fase 28")
    print("=" * 60)
    if len(d1["a"]) > 0 and len(d2["a"]) > 0:
        print(f"  δ_rms inicial 1LPT : {d1['delta_rms'][0]:.4e}")
        print(f"  δ_rms inicial 2LPT : {d2['delta_rms'][0]:.4e}")
        ratio_init = d2["delta_rms"][0] / (d1["delta_rms"][0] + 1e-30)
        print(f"  Ratio 2LPT/1LPT (a={d1['a'][0]:.3f}): {ratio_init:.4f}")
        if len(d1["a"]) > 1 and len(d2["a"]) > 1:
            print(f"  δ_rms final 1LPT : {d1['delta_rms'][-1]:.4e}  (a={d1['a'][-1]:.3f})")
            print(f"  δ_rms final 2LPT : {d2['delta_rms'][-1]:.4e}  (a={d2['a'][-1]:.3f})")
    print("=" * 60)

    if not HAS_MPL:
        print("matplotlib no disponible — solo diagnósticos de texto")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: δ_rms vs a (escala log-log)
    ax = axes[0]
    ax.loglog(d1["a"], d1["delta_rms"], "o-", ms=4, color="steelblue",
              label="1LPT (Zel'dovich)")
    ax.loglog(d2["a"], d2["delta_rms"], "s-", ms=4, color="coral",
              label="2LPT")
    if D1_arr is not None:
        # Normalizar al valor inicial 1LPT
        D1_at_ainit = float(growth_factor_D1([args.a_init], args.omega_m, args.omega_l)[0])
        scale = delta_rms_1lpt_0 / (D1_at_ainit + 1e-30)
        ax.loglog(a_all, D1_arr * scale, "--", color="gray", lw=2,
                  label=r"$D_1(a)$ teórico (ΛCDM)", alpha=0.7)
    ax.set_xlabel("Factor de escala $a$")
    ax.set_ylabel(r"$\delta_{\rm rms}$")
    ax.set_title(r"Crecimiento $\delta_{\rm rms}(a)$")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: ratio δ_rms(2LPT) / δ_rms(1LPT) vs a
    ax = axes[1]
    if len(d1["a"]) == len(d2["a"]):
        ratio = d2["delta_rms"] / (d1["delta_rms"] + 1e-30)
        ax.semilogx(d1["a"], ratio, "o-", ms=4, color="darkgreen")
        ax.axhline(1.0, color="gray", ls="--", lw=1)
        ax.fill_between(d1["a"], 0.95, 1.05, alpha=0.15, color="gray",
                        label="±5%")
    ax.set_xlabel("Factor de escala $a$")
    ax.set_ylabel(r"$\delta_{\rm rms}$(2LPT) / $\delta_{\rm rms}$(1LPT)")
    ax.set_title("Ratio de crecimiento 2LPT/1LPT")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"Fase 28: Crecimiento lineal 1LPT vs 2LPT\n"
        r"$\Omega_m=0.315$, $\Omega_\Lambda=0.685$, $\sigma_8=0.8$, EH no-wiggle, "
        f"$a_{{init}}={args.a_init}$",
        fontsize=9
    )
    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Figura guardada: {args.out}")


if __name__ == "__main__":
    main()
