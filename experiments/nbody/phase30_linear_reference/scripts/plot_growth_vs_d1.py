#!/usr/bin/env python3
"""
plot_growth_vs_d1.py — Fase 30: Crecimiento temporal vs D1(a)
=============================================================

Compara el crecimiento de P(k) medido en gadget-ng con el crecimiento
esperado por la teoría lineal D1(a).

## Observable central

    ratio(a) = P(k, a) / P(k, a_init)

En el régimen lineal esto debe seguir:

    ratio(a) ≈ [D1(a) / D1(a_init)]²

donde D1(a) se calcula por integración numérica de la ecuación del
factor de crecimiento en ΛCDM plano.

## Unidades

El ratio P(k,a)/P(k,a_init) es dimensionless e independiente de la
normalización absoluta — NO está afectado por el offset R = P_measured/P_EH.
Esta validación es correcta incluso con el offset de normalización conocido.

## Uso

    python plot_growth_vs_d1.py \\
        --snapshots snap_a*.json \\
        --a-init 0.02 \\
        [--omega-m 0.315 --omega-l 0.685] \\
        [--out-prefix phase30]
"""

import argparse
import json
import math
import sys

try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[AVISO] matplotlib/numpy no disponible — solo texto")


# ── Factor de crecimiento D1(a) por integración numérica ─────────────────────

def growth_integrand(a, omega_m, omega_l):
    """Integrando de la integral de Peebles para D1(a)."""
    h_sq = omega_m / a**3 + omega_l
    if h_sq <= 0.0:
        return 0.0
    return 1.0 / (a * math.sqrt(h_sq))**3


def growth_factor_d1(a, omega_m, omega_l, n_steps=1000):
    """
    Factor de crecimiento D1(a) por integración de la fórmula de Peebles:

        D1(a) ∝ H(a) ∫_0^a da' / [a' H(a')]³

    Normalizado a D1(1) = 1 (convencionalmente).
    """
    h_a = math.sqrt(omega_m / a**3 + omega_l)

    # Integral de 0 a a (con límite inferior ε para evitar singularidad)
    a_lo = 1e-4
    a_hi = a
    if a_hi <= a_lo:
        return 0.0

    da = (a_hi - a_lo) / n_steps
    total = 0.0
    f_prev = growth_integrand(a_lo, omega_m, omega_l)
    for i in range(1, n_steps + 1):
        ai = a_lo + i * da
        f  = growth_integrand(ai, omega_m, omega_l)
        total += 0.5 * (f_prev + f) * da
        f_prev = f

    return h_a * total


def growth_ratio(a, a_init, omega_m, omega_l):
    """
    Retorna [D1(a)/D1(a_init)]².
    """
    d1_a    = growth_factor_d1(a, omega_m, omega_l)
    d1_init = growth_factor_d1(a_init, omega_m, omega_l)
    if d1_init <= 0.0:
        return float("nan")
    return (d1_a / d1_init)**2


# ── EdS approximation D1 ∝ a ─────────────────────────────────────────────────

def eds_growth_ratio(a, a_init):
    """Aproximación Einstein-de Sitter: D1 ∝ a."""
    return (a / a_init)**2


# ── Cargar snapshots ──────────────────────────────────────────────────────────

def load_snapshot(path):
    """
    Carga un snapshot JSON con formato:
        {"a": ..., "pk_bins": [{k, pk, n_modes}, ...]}
    o alternativamente:
        {"a": ..., "bins": [{k, pk}, ...]}
    """
    with open(path) as f:
        data = json.load(f)
    a = data.get("a") or data.get("scale_factor")
    bins = data.get("pk_bins") or data.get("bins") or []
    return float(a), bins


def mean_pk(bins, n_low=None):
    """Media de P(k) sobre los bins de menor k (más lineales)."""
    valid = [(b["k"], b["pk"]) for b in bins if b.get("pk", 0.0) > 0.0]
    if not valid:
        return 0.0
    if n_low is not None:
        valid = valid[:n_low]
    return sum(pk for _, pk in valid) / len(valid)


# ── Diagnósticos de texto ─────────────────────────────────────────────────────

def print_growth_diagnostics(snapshots_data, a_init, omega_m, omega_l):
    """
    Imprime tabla de crecimiento:
        a | ratio_medido | ratio_D1 | ratio_EdS | error_vs_D1
    """
    # P(k) inicial
    a0, bins0 = snapshots_data[0]
    pk0 = mean_pk(bins0, n_low=len(bins0) // 2 or 1)
    if pk0 <= 0.0:
        print("  ERROR: P(k) inicial nulo")
        return

    print(f"\n{'a':>8} {'ratio_meas':>12} {'ratio_D1^2':>12} "
          f"{'ratio_EdS':>10} {'err_D1(%)':>10}")
    print("-" * 58)

    for a, bins in snapshots_data:
        pk_a = mean_pk(bins, n_low=len(bins) // 2 or 1)
        if pk_a <= 0.0:
            continue
        r_meas = pk_a / pk0
        r_d1   = growth_ratio(a, a_init, omega_m, omega_l)
        r_eds  = eds_growth_ratio(a, a_init)
        err_d1 = abs(r_meas / r_d1 - 1.0) * 100.0 if r_d1 > 0 else float("nan")

        print(f"{a:>8.4f} {r_meas:>12.4f} {r_d1:>12.4f} "
              f"{r_eds:>10.4f} {err_d1:>9.1f}%")


# ── Figuras ───────────────────────────────────────────────────────────────────

def make_growth_figure(snapshots_1lpt, snapshots_2lpt,
                       a_init, omega_m, omega_l, out_prefix):
    if not HAS_MPL:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Fase 30: Crecimiento temporal vs D1(a)", fontsize=12)

    # Curva D1²(a)/D1²(a_init) numérica
    a_arr = [a_init * (1.0 / a_init) ** (i / 99) for i in range(100)]
    a_arr = [a_init + i * (0.3 - a_init) / 99 for i in range(100)]
    d1_arr = [growth_ratio(a, a_init, omega_m, omega_l) for a in a_arr]
    eds_arr = [eds_growth_ratio(a, a_init) for a in a_arr]

    a0_1, bins0_1 = snapshots_1lpt[0]
    pk0_1 = mean_pk(bins0_1)
    a0_2, bins0_2 = snapshots_2lpt[0] if snapshots_2lpt else (a0_1, bins0_1)
    pk0_2 = mean_pk(bins0_2) if snapshots_2lpt else pk0_1

    # ── Panel 1: ratio_medido vs D1 ───────────────────────────────────────
    ax = axes[0]
    ax.plot(a_arr, d1_arr,  "k-",  lw=2,   label="D1(a)² / D1(a_init)²  (numérico ΛCDM)")
    ax.plot(a_arr, eds_arr, "k--", lw=1.5, label="EdS: (a/a_init)²", alpha=0.7)

    if snapshots_1lpt:
        as1 = [s[0] for s in snapshots_1lpt]
        rs1 = [mean_pk(s[1]) / pk0_1 if pk0_1 > 0 else 0 for s in snapshots_1lpt]
        ax.plot(as1, rs1, "bs-", lw=1.5, label="gadget-ng 1LPT", markersize=5)

    if snapshots_2lpt:
        as2 = [s[0] for s in snapshots_2lpt]
        rs2 = [mean_pk(s[1]) / pk0_2 if pk0_2 > 0 else 0 for s in snapshots_2lpt]
        ax.plot(as2, rs2, "ro-", lw=1.5, label="gadget-ng 2LPT", markersize=5)

    ax.set_xlabel("a (factor de escala)")
    ax.set_ylabel("P(k, a) / P(k, a_init)")
    ax.set_title("Crecimiento de P(k) (modos lineales)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel 2: error relativo vs D1 ─────────────────────────────────────
    ax = axes[1]
    ax.axhline(0, color="k", lw=1)
    ax.axhspan(-20, 20, color="green", alpha=0.1, label="±20%")
    ax.axhspan(-50, -20, color="yellow", alpha=0.1)
    ax.axhspan(20, 50, color="yellow", alpha=0.1)

    if snapshots_1lpt:
        as1 = [s[0] for s in snapshots_1lpt]
        errs1 = []
        for a, bins in snapshots_1lpt:
            pk_a = mean_pk(bins)
            r_meas = pk_a / pk0_1 if pk0_1 > 0 else 0
            r_d1   = growth_ratio(a, a_init, omega_m, omega_l)
            errs1.append((r_meas / r_d1 - 1.0) * 100.0 if r_d1 > 0 else 0)
        ax.plot(as1, errs1, "bs-", label="1LPT", markersize=5)

    if snapshots_2lpt:
        as2 = [s[0] for s in snapshots_2lpt]
        errs2 = []
        for a, bins in snapshots_2lpt:
            pk_a = mean_pk(bins)
            r_meas = pk_a / pk0_2 if pk0_2 > 0 else 0
            r_d1   = growth_ratio(a, a_init, omega_m, omega_l)
            errs2.append((r_meas / r_d1 - 1.0) * 100.0 if r_d1 > 0 else 0)
        ax.plot(as2, errs2, "ro-", label="2LPT", markersize=5)

    ax.set_xlabel("a (factor de escala)")
    ax.set_ylabel("Error relativo vs D1(a) [%]")
    ax.set_title("Error relativo del crecimiento")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-100, 100)

    plt.tight_layout()
    out_path = f"{out_prefix}_growth_vs_d1.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Figura guardada: {out_path}")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compara crecimiento P(k,a)/P(k,a_init) vs D1(a)²"
    )
    parser.add_argument("--snapshots",      type=str, nargs="+", required=True,
                        help="Archivos JSON de snapshots 1LPT (ordenados por a)")
    parser.add_argument("--snapshots-2lpt", type=str, nargs="*", default=None,
                        help="Archivos JSON snapshots 2LPT (opcional)")
    parser.add_argument("--a-init",   type=float, default=0.02)
    parser.add_argument("--omega-m",  type=float, default=0.315)
    parser.add_argument("--omega-l",  type=float, default=0.685)
    parser.add_argument("--out-prefix", type=str, default="phase30")
    args = parser.parse_args()

    snaps_1lpt = sorted(
        [load_snapshot(p) for p in args.snapshots],
        key=lambda x: x[0]
    )

    snaps_2lpt = []
    if args.snapshots_2lpt:
        snaps_2lpt = sorted(
            [load_snapshot(p) for p in args.snapshots_2lpt],
            key=lambda x: x[0]
        )

    print(f"\n[plot_growth_vs_d1] a_init={args.a_init}, "
          f"omega_m={args.omega_m}, omega_l={args.omega_l}")
    print(f"  Snapshots 1LPT: {len(snaps_1lpt)} a ∈ "
          f"[{snaps_1lpt[0][0]:.4f}, {snaps_1lpt[-1][0]:.4f}]")
    if snaps_2lpt:
        print(f"  Snapshots 2LPT: {len(snaps_2lpt)}")

    print("\n  === 1LPT ===")
    print_growth_diagnostics(snaps_1lpt, args.a_init, args.omega_m, args.omega_l)
    if snaps_2lpt:
        print("\n  === 2LPT ===")
        print_growth_diagnostics(snaps_2lpt, args.a_init, args.omega_m, args.omega_l)

    make_growth_figure(snaps_1lpt, snaps_2lpt,
                       args.a_init, args.omega_m, args.omega_l, args.out_prefix)
    return 0


if __name__ == "__main__":
    sys.exit(main())
