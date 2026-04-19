#!/usr/bin/env python3
"""
plot_pk.py — Figura 2 y 3: espectro de potencia P(k) comparativo 1LPT vs 2LPT.

Genera figuras con:
  - Panel izq: P(k) inicial de 1LPT y 2LPT superpuestos (log-log)
  - Panel der: ratio P_2LPT(k) / P_1LPT(k) por bin (escala lineal)
  - Tabla de errores relativos por bin

Opcionalmente, si se proveen datos finales (--pk1lpt-final, --pk2lpt-final),
genera una figura adicional comparando la evolución.

Uso:
  python plot_pk.py \\
      --pk1lpt   pk_1lpt_a002.json \\
      --pk2lpt   pk_2lpt_a002.json \\
      --box      100.0 \\
      --out      fig_pk_comparison.png

  # Con evolución:
  python plot_pk.py \\
      --pk1lpt        pk_1lpt_a002.json \\
      --pk2lpt        pk_2lpt_a002.json \\
      --pk1lpt-final  pk_1lpt_a002_final.json \\
      --pk2lpt-final  pk_2lpt_a002_final.json \\
      --box 100.0 --out fig_pk_evolution.png

Formato JSON: lista de {"k": ..., "pk": ..., "n_modes": ...}
(salida de gadget-ng-analysis::power_spectrum)
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


# ── Función de transferencia EH no-wiggle (referencia Python) ─────────────────

def eh_nowiggle_T(k_hmpc, omega_m=0.315, omega_b=0.049, h=0.674):
    if k_hmpc <= 0:
        return 1.0
    omh2 = omega_m * h**2
    obh2 = omega_b * h**2
    fb   = omega_b / omega_m
    s    = 44.5 * math.log(9.83 / omh2) / math.sqrt(1 + 10 * obh2**0.75)
    aG   = 1 - 0.328 * math.log(431 * omh2) * fb + 0.38 * math.log(22.3 * omh2) * fb**2
    k_m  = k_hmpc * h
    ks   = 0.43 * k_m * s
    G    = omega_m * h * (aG + (1 - aG) / (1 + ks**4))
    q    = k_hmpc / max(G, 1e-30)
    L0   = math.log(math.e * 2 + 1.8 * q)
    C0   = 14.2 + 731 / (1 + 62.5 * q)
    return max(0.0, min(1.0, L0 / (L0 + C0 * q**2)))


def pk_target_shape(k_hmpc_array, n_s=0.965, omega_m=0.315, omega_b=0.049, h=0.674):
    """P(k) ∝ k^n_s · T²(k) sin normalizar (para mostrar la forma espectral)."""
    pk = []
    for k in k_hmpc_array:
        if k <= 0:
            pk.append(0.0)
        else:
            T = eh_nowiggle_T(k, omega_m, omega_b, h)
            pk.append(k**n_s * T**2)
    return np.array(pk)


# ── Carga de datos ─────────────────────────────────────────────────────────────

def load_pk(path):
    """Carga bins de P(k) desde JSON. Retorna (k_array, pk_array, nmodes_array)."""
    with open(path) as f:
        data = json.load(f)
    bins = [b for b in data if b.get("pk", 0) > 0 and b.get("n_modes", 0) > 0]
    k      = np.array([b["k"]       for b in bins])
    pk     = np.array([b["pk"]      for b in bins])
    nmodes = np.array([b["n_modes"] for b in bins], dtype=float)
    return k, pk, nmodes


# ── Diagnósticos de texto ──────────────────────────────────────────────────────

def print_pk_summary(k_1, pk_1, k_2, pk_2, box_mpc_h, label="inicial"):
    """Imprime tabla de comparación bin a bin."""
    # Convertir k de unidades de grid (2π/L_box) a h/Mpc
    k_fund_hmpc = 2 * math.pi / box_mpc_h  # k fundamental en h/Mpc
    print(f"\n{'─'*65}")
    print(f"P(k) comparativo 1LPT vs 2LPT — {label}")
    print(f"{'─'*65}")
    print(f"{'k [h/Mpc]':>12} {'P_1LPT':>14} {'P_2LPT':>14} {'ratio-1 [%]':>12}")
    print(f"{'─'*65}")

    min_n = min(len(k_1), len(k_2))
    for i in range(min_n):
        k_phys = k_1[i] * k_fund_hmpc  # solo si k está en unidades de 2π/L
        ratio_1 = (pk_2[i] / pk_1[i] - 1) * 100 if pk_1[i] > 0 else float("nan")
        flag = "  ← > 10%" if abs(ratio_1) > 10 else ""
        print(f"  {k_phys:10.4f} {pk_1[i]:14.4e} {pk_2[i]:14.4e} {ratio_1:11.2f}%{flag}")

    print(f"{'─'*65}")

    if min_n > 0:
        ratios = pk_2[:min_n] / np.maximum(pk_1[:min_n], 1e-30)
        print(f"\nEstadísticas del ratio P_2LPT / P_1LPT:")
        print(f"  Media   : {ratios.mean():.4f}  ({(ratios.mean()-1)*100:.2f}%)")
        print(f"  Mediana : {np.median(ratios):.4f}  ({(np.median(ratios)-1)*100:.2f}%)")
        print(f"  Max |desviación| : {np.max(np.abs(ratios - 1))*100:.2f}%")
        print(f"  Bins > 5%  : {np.sum(np.abs(ratios - 1) > 0.05)}/{min_n}")
        print(f"  Bins > 10% : {np.sum(np.abs(ratios - 1) > 0.10)}/{min_n}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pk1lpt",       required=True, help="JSON P(k) 1LPT inicial")
    parser.add_argument("--pk2lpt",       required=True, help="JSON P(k) 2LPT inicial")
    parser.add_argument("--pk1lpt-final", default=None,  help="JSON P(k) 1LPT tras evolución")
    parser.add_argument("--pk2lpt-final", default=None,  help="JSON P(k) 2LPT tras evolución")
    parser.add_argument("--box",    type=float, default=100.0, help="Caja en Mpc/h")
    parser.add_argument("--a-init", type=float, default=0.02,  help="a inicial (para título)")
    parser.add_argument("--out",    default="fig_pk_comparison.png")
    args = parser.parse_args()

    try:
        k_1, pk_1, nm_1 = load_pk(args.pk1lpt)
        k_2, pk_2, nm_2 = load_pk(args.pk2lpt)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"ERROR cargando P(k): {e}")
        return

    k_fund_hmpc = 2 * math.pi / args.box

    print_pk_summary(k_1, pk_1, k_2, pk_2, args.box, label=f"t=0 (a_init={args.a_init})")

    has_final = False
    if args.pk1lpt_final and args.pk2lpt_final:
        try:
            k_1f, pk_1f, _ = load_pk(args.pk1lpt_final)
            k_2f, pk_2f, _ = load_pk(args.pk2lpt_final)
            has_final = True
            print_pk_summary(k_1f, pk_1f, k_2f, pk_2f, args.box, label="final")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"  WARN: no se pudo cargar P(k) final: {e}")

    if not HAS_MPL:
        print("\nmatplotlib no disponible — solo diagnósticos de texto (ver arriba).")
        return

    n_panels = 3 if has_final else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))

    k_1_phys = k_1 * k_fund_hmpc
    k_2_phys = k_2 * k_fund_hmpc

    # ── Panel 1: P(k) inicial 1LPT vs 2LPT ───────────────────────────────────
    ax = axes[0]
    ax.loglog(k_1_phys, pk_1, "o-", color="steelblue", ms=5, lw=1.5,
              label="1LPT (Zel'dovich)")
    ax.loglog(k_2_phys, pk_2, "s--", color="coral", ms=5, lw=1.5,
              label="2LPT")
    # Curva objetivo (forma espectral renormalizada)
    k_th  = np.geomspace(k_1_phys.min() * 0.5, k_1_phys.max() * 2, 200)
    pk_th = pk_target_shape(k_th)
    # Renormalizar al valor medio de pk_1
    scale = np.median(pk_1) / np.median(pk_target_shape(k_1_phys))
    ax.loglog(k_th, pk_th * scale, "-", color="gray", lw=1, alpha=0.6, label="forma EH (referencia)")
    ax.set_xlabel("k [h/Mpc]")
    ax.set_ylabel(r"$P(k)$ [arb.]")
    ax.set_title(f"P(k) inicial — a_init={args.a_init}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")

    # ── Panel 2: ratio P_2LPT / P_1LPT ──────────────────────────────────────
    ax = axes[1]
    min_n = min(len(k_1_phys), len(k_2_phys))
    ratio_init = pk_2[:min_n] / np.maximum(pk_1[:min_n], 1e-30)
    ax.semilogx(k_1_phys[:min_n], ratio_init, "o-", color="darkgreen", lw=2, ms=6,
                label="inicial")

    if has_final:
        ratio_final = pk_2f[:min(len(k_1f), len(k_2f))] / \
                      np.maximum(pk_1f[:min(len(k_1f), len(k_2f))], 1e-30)
        k_f_phys = k_1f[:len(ratio_final)] * k_fund_hmpc
        ax.semilogx(k_f_phys, ratio_final, "s--", color="purple", lw=2, ms=6,
                    label="final (tras evolución)")

    ax.axhline(1.0,  color="black", ls="-",  lw=0.8, alpha=0.5)
    ax.axhline(1.05, color="gray",  ls="--", lw=0.8, alpha=0.6, label="±5%")
    ax.axhline(0.95, color="gray",  ls="--", lw=0.8, alpha=0.6)
    ax.axhline(1.10, color="orange", ls=":", lw=0.8, alpha=0.6, label="±10%")
    ax.axhline(0.90, color="orange", ls=":", lw=0.8, alpha=0.6)
    ax.set_xlabel("k [h/Mpc]")
    ax.set_ylabel(r"$P_{2\mathrm{LPT}}(k) / P_{1\mathrm{LPT}}(k)$")
    ax.set_title("Ratio espectral 2LPT/1LPT")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")
    ax.set_ylim(0.7, 1.3)

    # ── Panel 3 (opcional): P(k) final comparado ─────────────────────────────
    if has_final:
        ax = axes[2]
        k_1f_phys = k_1f * k_fund_hmpc
        k_2f_phys = k_2f * k_fund_hmpc
        ax.loglog(k_1_phys, pk_1, "o-", color="steelblue", ms=4, lw=1, alpha=0.5,
                  label="1LPT inicial")
        ax.loglog(k_2_phys, pk_2, "s-", color="coral", ms=4, lw=1, alpha=0.5,
                  label="2LPT inicial")
        ax.loglog(k_1f_phys, pk_1f, "o-", color="steelblue", ms=5, lw=2,
                  label="1LPT final")
        ax.loglog(k_2f_phys, pk_2f, "s--", color="coral", ms=5, lw=2,
                  label="2LPT final")
        ax.set_xlabel("k [h/Mpc]")
        ax.set_ylabel(r"$P(k)$ [arb.]")
        ax.set_title("Evolución P(k): inicial vs final")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, which="both")

    plt.suptitle(
        "Fase 29: Comparación espectral 1LPT vs 2LPT\n"
        fr"ΛCDM Planck18: $a_\mathrm{{init}}={args.a_init}$, $\sigma_8=0.8$, "
        r"$N=32^3$, caja=100 Mpc/h",
        fontsize=9
    )
    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"\nFigura guardada: {args.out}")


if __name__ == "__main__":
    main()
