#!/usr/bin/env python3
"""
plot_displacements.py — Figura 1: desplazamientos 1LPT vs 2LPT.

Calcula y visualiza:
  - |Ψ¹|_rms: RMS del desplazamiento desde la retícula Lagrangiana (1LPT)
  - |Ψ²|_rms: RMS de la corrección de segundo orden (2LPT − 1LPT)
  - ratio |Ψ²| / |Ψ¹|: fracción de la corrección vs el desplazamiento total

Genera una figura de 2 paneles:
  - Izq: |Ψ¹|_rms y |Ψ²|_rms en función del parámetro de variación
  - Der: ratio |Ψ²|/|Ψ¹| con indicación del régimen lineal

Uso:
  # Comparación por σ₈ (carga 3 pares de archivos de posiciones):
  python plot_displacements.py \\
      --pos1lpt pos_1lpt_s04.json pos_1lpt_s08.json pos_1lpt_s16.json \\
      --pos2lpt pos_2lpt_s04.json pos_2lpt_s08.json pos_2lpt_s16.json \\
      --labels "σ₈=0.4" "σ₈=0.8" "σ₈=1.6" \\
      --grid 32 --box 1.0 \\
      --out fig_displacements.png

  # Comparación por a_init (los mismos archivos base, distintas velocidades):
  python plot_displacements.py \\
      --pos1lpt pos_1lpt_a002.json pos_1lpt_a005.json pos_1lpt_a010.json \\
      --pos2lpt pos_2lpt_a002.json pos_2lpt_a005.json pos_2lpt_a010.json \\
      --labels "a=0.02 (z≈49)" "a=0.05 (z≈19)" "a=0.10 (z≈9)" \\
      --grid 32 --box 1.0 \\
      --out fig_displacements_ainit.png

Formato JSON de posiciones (salida de gadget-ng):
  Lista de objetos: [{"x": ..., "y": ..., "z": ...}, ...]
  o lista de arrays: [[x, y, z], ...]
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


# ── Carga de datos ─────────────────────────────────────────────────────────────

def load_positions(path):
    """Carga posiciones desde JSON. Acepta [{"x":..,"y":..,"z":..}] o [[x,y,z]]."""
    with open(path) as f:
        data = json.load(f)

    if not data:
        return np.empty((0, 3))

    if isinstance(data[0], dict):
        pos = np.array([[p.get("x", p.get("position", {}).get("x", 0)),
                         p.get("y", p.get("position", {}).get("y", 0)),
                         p.get("z", p.get("position", {}).get("z", 0))]
                        for p in data])
    else:
        pos = np.array(data)

    return pos.reshape(-1, 3)


# ── Cálculo de desplazamientos ────────────────────────────────────────────────

def lagrangian_grid(n_part, box_size):
    """Genera posiciones de la retícula Lagrangiana (centros de celda)."""
    n = round(n_part ** (1.0 / 3.0))
    assert n ** 3 == n_part, f"n_part={n_part} no es cubo perfecto"
    d = box_size / n
    coords = []
    for ix in range(n):
        for iy in range(n):
            for iz in range(n):
                coords.append([(ix + 0.5) * d, (iy + 0.5) * d, (iz + 0.5) * d])
    return np.array(coords)


def rms_displacement(pos, q, box_size):
    """
    RMS del desplazamiento pos − q con condiciones periódicas.

    Aplica la imagen mínima: Δ = (pos − q + L/2) mod L − L/2.
    """
    delta = pos - q
    delta = (delta + box_size / 2) % box_size - box_size / 2
    return math.sqrt(np.mean(np.sum(delta ** 2, axis=1)))


def psi_rms_pair(pos_1lpt, pos_2lpt, q, box_size):
    """Calcula |Ψ¹|_rms y |Ψ²|_rms (= |pos_2lpt − pos_1lpt|_rms)."""
    psi1 = rms_displacement(pos_1lpt, q, box_size)
    # Ψ² = (pos_2lpt − pos_1lpt)
    delta2 = pos_2lpt - pos_1lpt
    delta2 = (delta2 + box_size / 2) % box_size - box_size / 2
    psi2 = math.sqrt(np.mean(np.sum(delta2 ** 2, axis=1)))
    return psi1, psi2


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pos1lpt", nargs="+", required=True,
                        help="Archivos JSON de posiciones 1LPT (uno por caso)")
    parser.add_argument("--pos2lpt", nargs="+", required=True,
                        help="Archivos JSON de posiciones 2LPT (uno por caso)")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Etiquetas para cada caso (default: 'Caso N')")
    parser.add_argument("--grid",  type=int,   default=32,  help="N del grid (N³ partículas)")
    parser.add_argument("--box",   type=float, default=1.0, help="Tamaño de caja (unidades internas)")
    parser.add_argument("--out",   default="fig_displacements.png")
    args = parser.parse_args()

    if len(args.pos1lpt) != len(args.pos2lpt):
        print("ERROR: --pos1lpt y --pos2lpt deben tener el mismo número de archivos")
        sys.exit(1)

    n_cases = len(args.pos1lpt)
    labels = args.labels or [f"Caso {i+1}" for i in range(n_cases)]

    n_part = args.grid ** 3
    q = lagrangian_grid(n_part, args.box)

    psi1_vals = []
    psi2_vals = []
    ratios    = []

    print(f"\n{'─'*60}")
    print(f"{'Caso':<20} {'|Ψ¹|_rms':>12} {'|Ψ²|_rms':>12} {'ratio':>10}")
    print(f"{'─'*60}")

    for i, (f1, f2) in enumerate(zip(args.pos1lpt, args.pos2lpt)):
        try:
            pos1 = load_positions(f1)
            pos2 = load_positions(f2)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"  ERROR {labels[i]}: {e}")
            continue

        if pos1.shape[0] != n_part or pos2.shape[0] != n_part:
            print(f"  WARN {labels[i]}: n_part={pos1.shape[0]} ≠ {n_part}")
            q_local = lagrangian_grid(pos1.shape[0], args.box)
        else:
            q_local = q

        p1, p2 = psi_rms_pair(pos1, pos2, q_local, args.box)
        ratio = p2 / p1 if p1 > 0 else float("nan")

        psi1_vals.append(p1)
        psi2_vals.append(p2)
        ratios.append(ratio)

        print(f"  {labels[i]:<18} {p1:12.4e} {p2:12.4e} {ratio:10.4e}  ({ratio*100:.2f}%)")

    print(f"{'─'*60}\n")

    if not psi1_vals:
        print("No se pudo cargar ningún archivo — sin figura.")
        return

    if not HAS_MPL:
        print("matplotlib no disponible — solo diagnósticos de texto (ver arriba).")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(psi1_vals))

    # Panel izquierdo: |Ψ¹| y |Ψ²| en escala log
    ax = axes[0]
    ax.bar(x - 0.2, psi1_vals, 0.4, label=r"$|\Psi^1|_\mathrm{rms}$",
           color="steelblue", alpha=0.85)
    ax.bar(x + 0.2, psi2_vals, 0.4, label=r"$|\Psi^2|_\mathrm{rms}$",
           color="coral",     alpha=0.85)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels[:len(psi1_vals)], fontsize=9)
    ax.set_ylabel(r"RMS desplazamiento [box]")
    ax.set_title(r"$|\Psi^1|$ y $|\Psi^2|$ por caso")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    # Panel derecho: ratio |Ψ²|/|Ψ¹|
    ax = axes[1]
    valid_x = [xi for xi, r in zip(x, ratios) if not math.isnan(r)]
    valid_r = [r for r in ratios if not math.isnan(r)]
    valid_l = [labels[i] for i, r in enumerate(ratios) if not math.isnan(r)]

    ax.plot(valid_x, [r * 100 for r in valid_r], "o-", color="darkgreen", lw=2, ms=8)
    ax.axhline(5.0,  color="gray",  ls="--", lw=0.8, alpha=0.7, label="5% (umbral visible)")
    ax.axhline(10.0, color="orange", ls="--", lw=0.8, alpha=0.7, label="10% (no lineal)")
    ax.set_xticks(valid_x)
    ax.set_xticklabels(valid_l, fontsize=9)
    ax.set_ylabel(r"$|\Psi^2| / |\Psi^1|$ [%]")
    ax.set_title(r"Ratio corrección 2LPT / desplazamiento 1LPT")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # Anotar valores
    for xi, r in zip(valid_x, valid_r):
        ax.annotate(f"{r*100:.2f}%", (xi, r * 100),
                    xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=8)

    plt.suptitle(
        "Fase 29: Desplazamientos 1LPT vs 2LPT\n"
        r"ΛCDM Planck18: $\Omega_m=0.315$, $\sigma_8=0.8$, EH no-wiggle, $N=32^3$, caja=100 Mpc/h",
        fontsize=9
    )
    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Figura guardada: {args.out}")


if __name__ == "__main__":
    main()
