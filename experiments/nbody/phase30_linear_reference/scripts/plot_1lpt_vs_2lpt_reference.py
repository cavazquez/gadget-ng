#!/usr/bin/env python3
"""
plot_1lpt_vs_2lpt_reference.py — Fase 30: 1LPT vs 2LPT contra referencia
=========================================================================

Compara las ICs y la evolución de 1LPT y 2LPT contra la referencia EH,
cuantificando cuál de las dos variantes está más cerca de la referencia.

## Observables

1. **P_2lpt(k) / P_1lpt(k)** al tiempo inicial: debe ser ≈ 1 ± 2%
   (la corrección 2LPT en posiciones es ~0.4% → efecto en P(k) ~ 1%)

2. **R_1lpt(k) = P_1lpt / P_EH** y **R_2lpt(k) = P_2lpt / P_EH**:
   ¿cuál es más constante? ¿Cambia el offset global?

3. **Crecimiento relativo** después de N pasos:
   ¿mantiene 2LPT mejor el crecimiento esperado?

## Nota sobre la corrección 2LPT

La corrección 2LPT afecta principalmente las VELOCIDADES (a través de f₂),
no la normalización del espectro inicial (que está fijada por σ₈). Por tanto:
- P(k) inicial: 1LPT ≈ 2LPT dentro del ruido estadístico
- P(k) tras evolución: 2LPT puede divergir menos del crecimiento lineal
  (menos transitorios) pero el efecto es pequeño con N=32³

## Uso

    python plot_1lpt_vs_2lpt_reference.py \\
        --pk-1lpt pk_init_1lpt.json \\
        --pk-2lpt pk_init_2lpt.json \\
        --pk-ref reference_pk.json \\
        --box 100.0 --h 0.674 \\
        [--snaps-1lpt snap_1lpt_a*.json] \\
        [--snaps-2lpt snap_2lpt_a*.json] \\
        [--a-init 0.02] \\
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


# ── Utilidades ────────────────────────────────────────────────────────────────

def load_pk(path):
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return data.get("bins") or data.get("pk_bins") or []


def load_ref(path):
    with open(path) as f:
        data = json.load(f)
    return data.get("bins") or [], data.get("source", "?")


def load_snapshot(path):
    with open(path) as f:
        data = json.load(f)
    a = float(data.get("a") or data.get("scale_factor", 0.0))
    bins = data.get("pk_bins") or data.get("bins") or []
    return a, bins


def interp_ref(k, bins_ref):
    ks  = [b["k"]  for b in bins_ref]
    pks = [b["pk"] for b in bins_ref]
    if k <= ks[0]:  return pks[0]
    if k >= ks[-1]: return pks[-1]
    for i in range(len(ks) - 1):
        if ks[i] <= k <= ks[i + 1]:
            t = math.log(k / ks[i]) / math.log(ks[i + 1] / ks[i])
            return math.exp(
                math.log(pks[i]) + t * (math.log(pks[i + 1]) - math.log(pks[i]))
            )
    return pks[-1]


def growth_factor_d1(a, omega_m=0.315, omega_l=0.685, n_steps=500):
    """D1(a) por integración numérica de Peebles."""
    h_a = math.sqrt(omega_m / a**3 + omega_l)
    a_lo = 1e-4
    da   = (a - a_lo) / n_steps
    total = 0.0
    a_prev = a_lo
    h_prev = math.sqrt(omega_m / a_lo**3 + omega_l)
    f_prev = 1.0 / (a_lo * h_prev)**3
    for i in range(1, n_steps + 1):
        ai = a_lo + i * da
        hi = math.sqrt(omega_m / ai**3 + omega_l)
        fi = 1.0 / (ai * hi)**3
        total += 0.5 * (f_prev + fi) * da
        f_prev = fi
    return h_a * total


# ── Diagnósticos ──────────────────────────────────────────────────────────────

def compare_vs_ref(label, bins_g, bins_ref, box_mpc_h, h_dimless):
    """Imprime R(k) y CV para una variante vs referencia."""
    print(f"\n  [{label}]")
    r_vals = []
    for b in bins_g:
        if b["pk"] <= 0.0:
            continue
        k_hmpc  = b["k"] * h_dimless / box_mpc_h
        pk_hmpc = b["pk"] * box_mpc_h**3
        p_ref   = interp_ref(k_hmpc, bins_ref)
        if p_ref <= 0.0:
            continue
        r = pk_hmpc / p_ref
        r_vals.append(r)
        print(f"    k={k_hmpc:.4f} h/Mpc: R = {r:.4e}")

    if r_vals:
        mean_r = sum(r_vals) / len(r_vals)
        var    = sum((r - mean_r)**2 for r in r_vals) / len(r_vals)
        cv     = math.sqrt(var) / mean_r if mean_r > 0 else float("inf")
        print(f"  mean R = {mean_r:.4e},  CV = {cv:.3f}")
        return mean_r, cv
    return None, None


def compare_ratio_1_vs_2(bins_1, bins_2, box_mpc_h, h_dimless):
    """Ratio P_2lpt/P_1lpt por bin."""
    print("\n  [P_2lpt/P_1lpt por bin]")
    max_diff = 0.0
    for b1, b2 in zip(bins_1, bins_2):
        if b1["pk"] > 0 and b2["pk"] > 0:
            r = b2["pk"] / b1["pk"]
            k_hmpc = b1["k"] * h_dimless / box_mpc_h
            diff = abs(r - 1.0) * 100.0
            if diff > max_diff:
                max_diff = diff
            print(f"    k={k_hmpc:.4f} h/Mpc: P2/P1 = {r:.6f}  (diff={diff:.3f}%)")
    print(f"  max |P2/P1 - 1| = {max_diff:.3f}%")
    return max_diff


# ── Figuras ───────────────────────────────────────────────────────────────────

def make_figure(bins_1, bins_2, bins_ref, ref_src, box_mpc_h, h_dimless,
                snaps_1, snaps_2, a_init, omega_m, omega_l, out_prefix):
    if not HAS_MPL:
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Fase 30: 1LPT vs 2LPT respecto a referencia EH", fontsize=12)

    # ── Panel 1: P(k) inicial + referencia ────────────────────────────────
    ax = axes[0]
    k_ref  = [b["k"]  for b in bins_ref]
    pk_ref = [b["pk"] for b in bins_ref]
    ax.loglog(k_ref, pk_ref, "k-", lw=2, label=f"Ref ({ref_src})")

    def gk(bins): return [b["k"] * h_dimless / box_mpc_h for b in bins if b["pk"] > 0]
    def gp(bins): return [b["pk"] * box_mpc_h**3          for b in bins if b["pk"] > 0]

    ax.loglog(gk(bins_1), gp(bins_1), "bs--", lw=1.5, label="1LPT IC", markersize=6)
    if bins_2:
        ax.loglog(gk(bins_2), gp(bins_2), "ro-.", lw=1.5, label="2LPT IC", markersize=6)

    ax.set_xlabel("k [h/Mpc]")
    ax.set_ylabel("P(k) [(Mpc/h)³]")
    ax.set_title("P(k) inicial")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.02,
            "Offset vertical por conv. unidades\n(forma válida, amplitud no)",
            transform=ax.transAxes, fontsize=6, color="gray")

    # ── Panel 2: P_2lpt/P_1lpt ────────────────────────────────────────────
    ax = axes[1]
    ax.axhline(1.0, color="k", lw=1, linestyle="--")
    ax.axhspan(0.95, 1.05, color="green", alpha=0.15, label="±5%")
    ax.axhspan(0.85, 0.95, color="yellow", alpha=0.1)
    ax.axhspan(1.05, 1.15, color="yellow", alpha=0.1)

    if bins_2:
        ratio_k  = [b["k"] * h_dimless / box_mpc_h for b1, b in zip(bins_1, bins_2)
                    if b1["pk"] > 0 and b["pk"] > 0]
        ratio_v  = [b["pk"] / b1["pk"] for b1, b in zip(bins_1, bins_2)
                    if b1["pk"] > 0 and b["pk"] > 0]
        ax.semilogx(ratio_k, ratio_v, "mo-", lw=1.5, markersize=6, label="P_2lpt/P_1lpt")

    ax.set_xlabel("k [h/Mpc]")
    ax.set_ylabel("P_2lpt(k) / P_1lpt(k)")
    ax.set_title("Corrección 2LPT en P(k) inicial")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.8, 1.2)

    # ── Panel 3: Crecimiento 1LPT vs 2LPT ────────────────────────────────
    ax = axes[2]
    ax.set_title("Crecimiento P(k,a)/P(k,a_init)")
    ax.set_xlabel("a")
    ax.set_ylabel("ratio")

    # Curva D1²(a)/D1²(a_init)
    a_arr = [a_init + i * (0.3 - a_init) / 49 for i in range(50)]
    d1_0  = growth_factor_d1(a_init, omega_m, omega_l)
    d1_arr = [(growth_factor_d1(a, omega_m, omega_l) / d1_0)**2 for a in a_arr]
    ax.plot(a_arr, d1_arr, "k-", lw=2, label="D1(a)²/D1(a_init)²")

    def mean_pk(bins):
        valid = [b["pk"] for b in bins if b.get("pk", 0) > 0]
        return sum(valid) / len(valid) if valid else 0.0

    if snaps_1 and mean_pk(snaps_1[0][1]) > 0:
        pk0_1 = mean_pk(snaps_1[0][1])
        ax.plot([s[0] for s in snaps_1],
                [mean_pk(s[1]) / pk0_1 for s in snaps_1],
                "bs-", lw=1.5, label="1LPT", markersize=5)

    if snaps_2 and mean_pk(snaps_2[0][1]) > 0:
        pk0_2 = mean_pk(snaps_2[0][1])
        ax.plot([s[0] for s in snaps_2],
                [mean_pk(s[1]) / pk0_2 for s in snaps_2],
                "ro-", lw=1.5, label="2LPT", markersize=5)

    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = f"{out_prefix}_1lpt_vs_2lpt_reference.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Figura guardada: {out_path}")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compara 1LPT vs 2LPT respecto a referencia EH/CAMB"
    )
    parser.add_argument("--pk-1lpt",   type=str, required=True)
    parser.add_argument("--pk-2lpt",   type=str, default=None)
    parser.add_argument("--pk-ref",    type=str, required=True)
    parser.add_argument("--box",       type=float, default=100.0)
    parser.add_argument("--h",         type=float, default=0.674)
    parser.add_argument("--snaps-1lpt", type=str, nargs="*", default=[])
    parser.add_argument("--snaps-2lpt", type=str, nargs="*", default=[])
    parser.add_argument("--a-init",    type=float, default=0.02)
    parser.add_argument("--omega-m",   type=float, default=0.315)
    parser.add_argument("--omega-l",   type=float, default=0.685)
    parser.add_argument("--out-prefix", type=str, default="phase30")
    args = parser.parse_args()

    bins_1  = load_pk(args.pk_1lpt)
    bins_ref, ref_src = load_ref(args.pk_ref)
    bins_2  = load_pk(args.pk_2lpt) if args.pk_2lpt else None

    snaps_1 = sorted([load_snapshot(p) for p in args.snaps_1lpt], key=lambda x: x[0])
    snaps_2 = sorted([load_snapshot(p) for p in args.snaps_2lpt], key=lambda x: x[0])

    print(f"\n[plot_1lpt_vs_2lpt_reference] box={args.box} Mpc/h, h={args.h}")
    print(f"  Referencia: {ref_src}")
    print(f"\n{'='*60}")
    print("  Comparación vs referencia EH")
    print(f"{'='*60}")

    mean_r1, cv1 = compare_vs_ref("1LPT", bins_1, bins_ref, args.box, args.h)
    if bins_2:
        mean_r2, cv2 = compare_vs_ref("2LPT", bins_2, bins_ref, args.box, args.h)
        compare_ratio_1_vs_2(bins_1, bins_2, args.box, args.h)

        if cv1 is not None and cv2 is not None:
            print(f"\n  Resumen: CV_1lpt={cv1:.3f}  CV_2lpt={cv2:.3f}")
            winner = "1LPT" if cv1 < cv2 else "2LPT" if cv2 < cv1 else "empate"
            print(f"  → {winner} tiene forma espectral ligeramente más constante")
            if abs(cv1 - cv2) < 0.05:
                print("  → Diferencia < 0.05: ambas variantes son estadísticamente equivalentes")

    make_figure(bins_1, bins_2, bins_ref, ref_src, args.box, args.h,
                snaps_1, snaps_2, args.a_init, args.omega_m, args.omega_l,
                args.out_prefix)
    return 0


if __name__ == "__main__":
    sys.exit(main())
