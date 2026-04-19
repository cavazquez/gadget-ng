#!/usr/bin/env python3
"""
plot_growth.py — Figura 4 y 5: crecimiento de estructura δ_rms(a) y v_rms(a).

Compara la evolución temporal de:
  - δ_rms(a): contraste de densidad RMS (medido con CIC en una malla)
  - v_rms(a): velocidad peculiar RMS
  entre ICs 1LPT (Zel'dovich) y 2LPT, para múltiples a_init.

También superpone la curva analítica D₁(a) (aproximación EdS y ΛCDM).

Figuras generadas:
  1. δ_rms(a) / δ_rms(a_init) normalizado, 1LPT vs 2LPT vs D₁(a)
  2. v_rms(a), 1LPT vs 2LPT
  3. Diferencia relativa |δ_rms_2LPT − δ_rms_1LPT| / δ_rms_1LPT

Uso:
  python plot_growth.py \\
      --diag1lpt diag_1lpt_a002.json diag_1lpt_a005.json \\
      --diag2lpt diag_2lpt_a002.json diag_2lpt_a005.json \\
      --labels "a_init=0.02" "a_init=0.05" \\
      --omega-m 0.315 --omega-l 0.685 \\
      --out fig_growth.png

  # Solo un par:
  python plot_growth.py \\
      --diag1lpt diag_1lpt_a002.json \\
      --diag2lpt diag_2lpt_a002.json \\
      --out fig_growth_a002.png

Formato JSON de diagnósticos (salida de gadget-ng):
  {
    "steps": [
      {"a": 0.02, "t": 0.0, "delta_rms": 0.012, "v_rms": 0.034, "step": 0},
      ...
    ]
  }
  o lista directa: [{"a": ..., "delta_rms": ..., "v_rms": ...}, ...]
"""

import argparse
import json
import math
import sys
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ── Factor de crecimiento lineal (aproximación numérica) ─────────────────────

def growth_rate_f(a, omega_m=0.315, omega_l=0.685):
    """f(a) = d ln D / d ln a ≈ Ω_m(a)^0.55 (Linder 2005)."""
    if a <= 0:
        return 1.0
    h_sq = omega_m / a**3 + omega_l
    h    = math.sqrt(max(h_sq, 1e-30))
    h0   = math.sqrt(omega_m + omega_l)  # H(a=1)/H₀ = 1 en unidades con H₀=1
    omega_m_a = (omega_m / a**3) / h_sq
    return max(omega_m_a, 0.0) ** 0.55


def growth_factor_D1_normalized(a_values, a_init, omega_m=0.315, omega_l=0.685):
    """
    D₁(a) / D₁(a_init) normalizado a 1 en a=a_init.

    Integra usando la aproximación:
    D₁(a) ∝ H(a) ∫_{0}^{a} da' / (H(a') a')³   (integral de Carroll, Press & Turner 1992)

    Para simplificar, usamos D₁(a) ≈ a en EdS como aproximación rápida,
    con corrección ΛCDM: D₁(a) ≈ a · g(a) donde g(a) ≈ (Ω_m/a³ / h²)^0.45.
    """
    # Normalización relativa sencilla: D₁(a)/D₁(a_init) ≈ (a/a_init) × g(a)/g(a_init)
    # donde g(a) ≈ Ω_m(a)^0.45 para ΛCDM plano
    def g_lcdm(a):
        h_sq = omega_m / a**3 + omega_l
        om_a = (omega_m / a**3) / h_sq
        return om_a ** 0.45

    g_init = g_lcdm(a_init)
    D = []
    for a in a_values:
        if a <= 0:
            D.append(0.0)
        else:
            D.append((a / a_init) * g_lcdm(a) / g_init)
    return np.array(D)


# ── Carga de diagnósticos ──────────────────────────────────────────────────────

def load_diagnostics(path):
    """
    Carga serie temporal de diagnósticos desde JSON.

    Acepta:
      {"steps": [{"a": ..., "delta_rms": ..., "v_rms": ...}, ...]}
      o lista directa: [{"a": ..., "delta_rms": ..., "v_rms": ...}, ...]
    """
    with open(path) as f:
        data = json.load(f)

    if isinstance(data, dict):
        steps = data.get("steps", data.get("diagnostics", []))
    elif isinstance(data, list):
        steps = data
    else:
        steps = []

    a_vals     = np.array([s.get("a",         s.get("scale_factor", 0)) for s in steps])
    delta_vals = np.array([s.get("delta_rms", s.get("delta",         0)) for s in steps])
    v_vals     = np.array([s.get("v_rms",     s.get("vrms",          0)) for s in steps])

    # Filtrar ceros/negativos
    mask = (a_vals > 0) & (delta_vals >= 0)
    return a_vals[mask], delta_vals[mask], v_vals[mask]


# ── Diagnósticos de texto ─────────────────────────────────────────────────────

def print_growth_summary(a_1, d_1, a_2, d_2, label=""):
    print(f"\n{'─'*65}")
    print(f"Crecimiento δ_rms(a) — {label}")
    print(f"{'─'*65}")
    if len(a_1) == 0 or len(a_2) == 0:
        print("  Sin datos.")
        return

    a_init = a_1[0]
    d0_1   = d_1[0] if d_1[0] > 0 else 1e-15
    d0_2   = d_2[0] if d_2[0] > 0 else 1e-15

    print(f"  {'a':>8} {'δ_1LPT':>12} {'δ_2LPT':>12} "
          f"{'D₁_1LPT':>10} {'D₁_2LPT':>10} {'Δrel%':>8}")
    print(f"{'─'*65}")

    n = min(len(a_1), len(a_2))
    for i in range(n):
        a_i  = a_1[i]
        d1_i = d_1[i]
        d2_i = d_2[i] if i < len(d_2) else float("nan")
        growth_1 = d1_i / d0_1
        growth_2 = d2_i / d0_2 if not math.isnan(d2_i) else float("nan")
        rel = (d2_i - d1_i) / d1_i * 100 if d1_i > 0 and not math.isnan(d2_i) else float("nan")
        print(f"  {a_i:8.4f} {d1_i:12.4e} {d2_i:12.4e} "
              f"{growth_1:10.3f} {growth_2:10.3f} {rel:7.2f}%")

    # Resumen final
    rel_final = (d_2[-1] - d_1[-1]) / d_1[-1] * 100 if d_1[-1] > 0 else float("nan")
    growth_1 = d_1[-1] / d0_1
    growth_2 = d_2[-1] / d0_2
    print(f"\n  Crecimiento total:")
    print(f"    1LPT: δ(a_f)/δ(a_i) = {growth_1:.3f}")
    print(f"    2LPT: δ(a_f)/δ(a_i) = {growth_2:.3f}")
    print(f"    Diferencia relativa final: {rel_final:.2f}%")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--diag1lpt", nargs="+", required=True,
                        help="JSON diagnósticos 1LPT (uno por caso)")
    parser.add_argument("--diag2lpt", nargs="+", required=True,
                        help="JSON diagnósticos 2LPT (uno por caso)")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Etiquetas para cada par")
    parser.add_argument("--omega-m", type=float, default=0.315)
    parser.add_argument("--omega-l", type=float, default=0.685)
    parser.add_argument("--out", default="fig_growth.png")
    args = parser.parse_args()

    if len(args.diag1lpt) != len(args.diag2lpt):
        print("ERROR: --diag1lpt y --diag2lpt deben tener el mismo número de archivos")
        sys.exit(1)

    n_cases = len(args.diag1lpt)
    labels  = args.labels or [f"Caso {i+1}" for i in range(n_cases)]
    colors_1lpt = ["steelblue", "darkorange",  "purple"]
    colors_2lpt = ["coral",     "gold",         "orchid"]

    loaded = []
    for i, (f1, f2) in enumerate(zip(args.diag1lpt, args.diag2lpt)):
        try:
            a1, d1, v1 = load_diagnostics(f1)
            a2, d2, v2 = load_diagnostics(f2)
            loaded.append((a1, d1, v1, a2, d2, v2))
            print_growth_summary(a1, d1, a2, d2, label=labels[i])
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"  ERROR {labels[i]}: {e}")
            loaded.append(None)

    if not any(x is not None for x in loaded):
        print("\nNo se pudieron cargar datos — sin figura.")
        return

    if not HAS_MPL:
        print("\nmatplotlib no disponible — solo diagnósticos de texto (ver arriba).")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # ── Panel 1: δ_rms(a) normalizado ─────────────────────────────────────────
    ax = axes[0]
    for i, entry in enumerate(loaded):
        if entry is None:
            continue
        a1, d1, _, a2, d2, _ = entry
        if len(a1) == 0:
            continue
        d0_1 = d1[0] if d1[0] > 0 else 1e-15
        d0_2 = d2[0] if len(d2) > 0 and d2[0] > 0 else 1e-15
        a_init = a1[0]

        lbl_1 = f"1LPT — {labels[i]}"
        lbl_2 = f"2LPT — {labels[i]}"
        ax.plot(a1, d1 / d0_1, "-",  color=colors_1lpt[i % 3], lw=2, label=lbl_1)
        ax.plot(a2, d2 / d0_2, "--", color=colors_2lpt[i % 3], lw=2, label=lbl_2)

        # Curva analítica D₁(a)
        a_th = np.linspace(a_init, a1[-1], 100)
        D1_th = growth_factor_D1_normalized(a_th, a_init, args.omega_m, args.omega_l)
        ax.plot(a_th, D1_th, ":", color="gray", lw=1, alpha=0.7,
                label=r"$D_1(a)/D_1(a_\mathrm{init})$ (teórico)" if i == 0 else None)

    ax.set_xlabel("a")
    ax.set_ylabel(r"$\delta_\mathrm{rms}(a) / \delta_\mathrm{rms}(a_\mathrm{init})$")
    ax.set_title("Crecimiento normalizado de δ")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # ── Panel 2: v_rms(a) ─────────────────────────────────────────────────────
    ax = axes[1]
    for i, entry in enumerate(loaded):
        if entry is None:
            continue
        a1, _, v1, a2, _, v2 = entry
        if len(a1) == 0 or np.max(v1) == 0:
            continue
        ax.plot(a1, v1, "-",  color=colors_1lpt[i % 3], lw=2, label=f"1LPT — {labels[i]}")
        ax.plot(a2, v2, "--", color=colors_2lpt[i % 3], lw=2, label=f"2LPT — {labels[i]}")

    ax.set_xlabel("a")
    ax.set_ylabel(r"$v_\mathrm{rms}(a)$ [unidades internas]")
    ax.set_title(r"Velocidad peculiar RMS")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # ── Panel 3: diferencia relativa |δ_2LPT − δ_1LPT| / δ_1LPT ─────────────
    ax = axes[2]
    for i, entry in enumerate(loaded):
        if entry is None:
            continue
        a1, d1, _, a2, d2, _ = entry
        n = min(len(a1), len(a2))
        if n == 0:
            continue
        a_common = a1[:n]
        rel_diff = np.abs(d2[:n] - d1[:n]) / np.maximum(d1[:n], 1e-30) * 100
        ax.plot(a_common, rel_diff, "-", color=f"C{i}", lw=2, label=labels[i])

    ax.axhline(5.0,  color="gray",   ls="--", lw=0.8, alpha=0.7, label="5%")
    ax.axhline(10.0, color="orange", ls="--", lw=0.8, alpha=0.7, label="10%")
    ax.set_xlabel("a")
    ax.set_ylabel(r"$|\delta_{2\mathrm{LPT}} - \delta_{1\mathrm{LPT}}| / \delta_{1\mathrm{LPT}}$ [%]")
    ax.set_title("Diferencia relativa δ_rms(a)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        "Fase 29: Crecimiento de estructura 1LPT vs 2LPT\n"
        r"ΛCDM Planck18: $\Omega_m=0.315$, $\sigma_8=0.8$, EH no-wiggle, $N=32^3$, caja=100 Mpc/h",
        fontsize=9
    )
    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"\nFigura guardada: {args.out}")


if __name__ == "__main__":
    main()
