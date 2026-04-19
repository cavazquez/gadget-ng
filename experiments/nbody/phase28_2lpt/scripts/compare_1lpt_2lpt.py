#!/usr/bin/env python3
"""
compare_1lpt_2lpt.py — Comparación cuantitativa de ICs 1LPT vs 2LPT.

Produce tres figuras:
  Fig 1: Histograma del desplazamiento |Ψ¹| vs |D₂/D₁² · Ψ²|
           (magnitud de las correcciones 1LPT y 2LPT por partícula)
  Fig 2: P(k) de las posiciones iniciales 1LPT vs 2LPT
  Fig 3: Diferencia relativa de posiciones |x_2LPT - x_1LPT| / d_grid

Uso:
  python compare_1lpt_2lpt.py \\
      --pos1lpt  positions_1lpt.json \\
      --pos2lpt  positions_2lpt.json \\
      --pk1lpt   pk_1lpt.json \\
      --pk2lpt   pk_2lpt.json \\
      --box      100.0 \\
      --n        32 \\
      --out      compare_1lpt_2lpt.png

Formato de posiciones JSON:
  Lista de [x, y, z] (en unidades internas de caja, box=1.0).

Formato de P(k) JSON:
  Lista de {"k": float, "pk": float, "n_modes": int}
  (salida de gadget-ng-analysis::power_spectrum).
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


# ── Carga de datos ────────────────────────────────────────────────────────────

def load_positions(path):
    """Carga posiciones desde JSON → array (N, 3)."""
    with open(path) as f:
        data = json.load(f)
    return np.array(data)


def load_pk(path):
    """Carga bins de P(k) desde JSON."""
    with open(path) as f:
        data = json.load(f)
    k   = np.array([b["k"]  for b in data if b.get("pk", 0) > 0])
    pk  = np.array([b["pk"] for b in data if b.get("pk", 0) > 0])
    return k, pk


# ── Análisis cuantitativo ─────────────────────────────────────────────────────

def compute_displacement_stats(pos1, pos2, n_grid, box_internal=1.0):
    """
    Calcula estadísticas del desplazamiento de cada partícula desde la retícula.

    pos1, pos2: arrays (N, 3) con posiciones 1LPT y 2LPT en unidades internas.
    n_grid: número de celdas por eje.
    box_internal: tamaño de la caja en unidades internas (default 1.0).

    Devuelve dict con RMS de |Ψ¹|, |Ψ²_correction|, y fracción que difiere.
    """
    n = len(pos1)
    d_grid = box_internal / n_grid

    # Coordenadas de la retícula (mismo orden que gadget-ng: gid = ix*n*n + iy*n + iz)
    indices = np.arange(n)
    ix = indices // (n_grid * n_grid)
    iy = (indices % (n_grid * n_grid)) // n_grid
    iz = indices % n_grid
    q = np.stack([
        (ix + 0.5) * d_grid,
        (iy + 0.5) * d_grid,
        (iz + 0.5) * d_grid,
    ], axis=1)

    # Desplazamiento 1LPT desde la retícula (con imagen mínima periódica)
    dp1 = pos1 - q
    dp1 = dp1 - box_internal * np.round(dp1 / box_internal)
    psi1_mag = np.linalg.norm(dp1, axis=1)

    # Corrección 2LPT: diferencia entre 2LPT y 1LPT (imagen mínima)
    dp2 = pos2 - pos1
    dp2 = dp2 - box_internal * np.round(dp2 / box_internal)
    psi2_mag = np.linalg.norm(dp2, axis=1)

    return {
        "psi1_rms":   float(np.sqrt(np.mean(psi1_mag**2))),
        "psi1_mean":  float(np.mean(psi1_mag)),
        "psi1_max":   float(np.max(psi1_mag)),
        "psi2_rms":   float(np.sqrt(np.mean(psi2_mag**2))),
        "psi2_mean":  float(np.mean(psi2_mag)),
        "psi2_max":   float(np.max(psi2_mag)),
        "ratio_rms":  float(np.sqrt(np.mean(psi2_mag**2)) / np.sqrt(np.mean(psi1_mag**2) + 1e-30)),
        "d_grid":     float(d_grid),
        "psi2_over_dgrid_rms": float(np.sqrt(np.mean(psi2_mag**2)) / d_grid),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pos1lpt",  required=True, help="JSON posiciones 1LPT")
    parser.add_argument("--pos2lpt",  required=True, help="JSON posiciones 2LPT")
    parser.add_argument("--pk1lpt",   required=True, help="JSON P(k) 1LPT")
    parser.add_argument("--pk2lpt",   required=True, help="JSON P(k) 2LPT")
    parser.add_argument("--box",      type=float, default=100.0,
                        help="Tamaño de caja en Mpc/h (default 100)")
    parser.add_argument("--n",        type=int, default=32,
                        help="Número de celdas de retícula por eje (default 32)")
    parser.add_argument("--out",      default="compare_1lpt_2lpt.png")
    args = parser.parse_args()

    pos1 = load_positions(args.pos1lpt)
    pos2 = load_positions(args.pos2lpt)
    k_1lpt, pk_1lpt = load_pk(args.pk1lpt)
    k_2lpt, pk_2lpt = load_pk(args.pk2lpt)

    # k del grid (unidades internas) → h/Mpc
    k_1lpt_phys = k_1lpt * 2 * math.pi / args.box
    k_2lpt_phys = k_2lpt * 2 * math.pi / args.box

    stats = compute_displacement_stats(pos1, pos2, args.n)

    # ── Diagnósticos de texto ─────────────────────────────────────────────────
    print("=" * 60)
    print("COMPARACIÓN 1LPT vs 2LPT — Fase 28")
    print("=" * 60)
    print(f"  N partículas    : {len(pos1)}")
    print(f"  d_grid          : {stats['d_grid']:.4f} (unidades internas)")
    print()
    print("  |Ψ¹|_rms        :", f"{stats['psi1_rms']:.4e}  ({stats['psi1_rms']/stats['d_grid']*100:.2f}% de d_grid)")
    print("  |D₂/D₁²·Ψ²|_rms:", f"{stats['psi2_rms']:.4e}  ({stats['psi2_over_dgrid_rms']*100:.3f}% de d_grid)")
    print("  |Ψ²|/|Ψ¹| (rms) :", f"{stats['ratio_rms']:.4f}")
    print()
    print("  |Ψ¹|_max        :", f"{stats['psi1_max']:.4e}")
    print("  |D₂/D₁²·Ψ²|_max:", f"{stats['psi2_max']:.4e}")
    print()

    # P(k) a k=k_fund
    if len(k_1lpt) > 0 and len(k_2lpt) > 0:
        ratio_pk = pk_2lpt / (pk_1lpt + 1e-30)
        print(f"  P(k)_2LPT / P(k)_1LPT:")
        print(f"    Mínimo: {ratio_pk.min():.4f}")
        print(f"    Máximo: {ratio_pk.max():.4f}")
        print(f"    Media : {ratio_pk.mean():.4f}")
    print("=" * 60)

    if not HAS_MPL:
        print("matplotlib no disponible — solo diagnósticos de texto")
        return

    # ── Figuras ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Fig 1: Histograma de magnitudes de desplazamiento
    ax = axes[0]
    n_grid = args.n
    d_grid = 1.0 / n_grid
    dp1 = pos1 - np.array([
        [(gid // (n_grid**2) + 0.5) * d_grid for gid in range(len(pos1))],
        [((gid % n_grid**2) // n_grid + 0.5) * d_grid for gid in range(len(pos1))],
        [(gid % n_grid + 0.5) * d_grid for gid in range(len(pos1))],
    ]).T
    dp1 -= np.round(dp1)
    psi1_m = np.linalg.norm(dp1, axis=1)

    dp2 = pos2 - pos1
    dp2 -= np.round(dp2)
    psi2_m = np.linalg.norm(dp2, axis=1)

    bins = np.linspace(0, max(psi1_m.max(), psi2_m.max()) * 1.05, 40)
    ax.hist(psi1_m, bins=bins, alpha=0.6, color="steelblue", label=r"$|\Psi^1|$ (1LPT)")
    ax.hist(psi2_m, bins=bins, alpha=0.6, color="coral", label=r"$|D_2/D_1^2 \cdot \Psi^2|$")
    ax.axvline(d_grid, color="gray", ls="--", lw=1, label=f"d_grid={d_grid:.3f}")
    ax.set_xlabel("Magnitud de desplazamiento [unidades internas]")
    ax.set_ylabel("Número de partículas")
    ax.set_title(r"Distribución de $|\Psi|$")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Fig 2: P(k) 1LPT vs 2LPT
    ax = axes[1]
    ax.loglog(k_1lpt_phys, pk_1lpt, "o-", ms=4, color="steelblue",
              label="P(k) 1LPT (Zel'dovich)")
    ax.loglog(k_2lpt_phys, pk_2lpt, "s-", ms=4, color="coral",
              label="P(k) 2LPT")
    ax.set_xlabel("k [h/Mpc]")
    ax.set_ylabel(r"P(k) [(Mpc/h)$^3$]")
    ax.set_title("Espectro de potencia inicial")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Fig 3: Ratio P(k) 2LPT/1LPT
    ax = axes[2]
    if len(k_1lpt) == len(k_2lpt):
        ratio = pk_2lpt / (pk_1lpt + 1e-30)
        ax.semilogx(k_1lpt_phys, ratio, "o-", ms=4, color="darkgreen")
        ax.axhline(1.0, color="gray", ls="--", lw=1)
        ax.fill_between(k_1lpt_phys, 0.95, 1.05, alpha=0.15, color="gray",
                        label="±5%")
    ax.set_xlabel("k [h/Mpc]")
    ax.set_ylabel("P(k)[2LPT] / P(k)[1LPT]")
    ax.set_title(r"Ratio P(k): corrección 2LPT")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"Fase 28: Comparación 1LPT vs 2LPT\n"
        r"$\Omega_m=0.315$, $\sigma_8=0.8$, EH no-wiggle, "
        f"N={args.n}³, caja={args.box} Mpc/h\n"
        rf"$|\Psi^2|_{{rms}}/|\Psi^1|_{{rms}} = {stats['ratio_rms']:.4f}$",
        fontsize=9
    )
    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Figura guardada: {args.out}")


if __name__ == "__main__":
    main()
