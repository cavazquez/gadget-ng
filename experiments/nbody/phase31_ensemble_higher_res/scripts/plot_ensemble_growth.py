#!/usr/bin/env python3
"""
plot_ensemble_growth.py — Fase 31: Crecimiento de P(k) vs teoría lineal.

Dados snapshots en distintos tiempos para múltiples seeds, calcula P(k, a)
y el crecimiento relativo. Compara con D₁(a) ≈ a (aproximación EdS).

Genera dos figuras:
  1. P_mean(k, a) / P_mean(k, a_init) vs D₁²(a)/D₁²(a_init) para bins de bajo k.
  2. Ratio observado/esperado con barras de error entre seeds.

Formato de entrada: directorio con snapshots nombrados como:
  snapshot_<step>_seed<NNN>.json  (cada uno contiene bins de P(k) y el valor de a)

Uso:
  python plot_ensemble_growth.py \\
      --snapshots-dir out/N32_a002_2lpt_pm/ \\
      --seeds 42 137 271 314 \\
      --a-init 0.02 \\
      --label "N32_2LPT_PM" \\
      --output-dir figures/
"""

import argparse
import json
import math
import os
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def load_snapshot(path):
    """Carga un snapshot de P(k). Espera dict con 'a' y 'bins'."""
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict) and "bins" in data:
        a = data.get("a", None)
        bins = data["bins"]
        return a, bins
    # Fallback: lista de bins sin a
    if isinstance(data, list):
        return None, data
    raise ValueError(f"Formato no reconocido en {path}")


def growth_factor_eds(a, a_init=0.02):
    """D₁(a)/D₁(a_init) ≈ a/a_init para el universo de materia dominante."""
    return a / a_init


def pk_low_k_mean(bins, n_bins_low=2):
    """Media de P(k) en los primeros n_bins_low bins de k."""
    valid = [b["pk"] for b in bins[:n_bins_low] if b.get("pk", 0) > 0.0]
    if not valid:
        return 0.0
    return sum(valid) / len(valid)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots-dir", required=True,
                        help="Directorio con snapshots JSON de P(k) en distintos tiempos")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 137, 271, 314],
                        help="Lista de seeds del ensemble")
    parser.add_argument("--a-init", type=float, default=0.02,
                        help="Factor de escala inicial")
    parser.add_argument("--label", default="ensemble",
                        help="Etiqueta para las figuras")
    parser.add_argument("--output-dir", default="figures",
                        help="Directorio de salida para las figuras")
    args = parser.parse_args()

    snap_dir = Path(args.snapshots_dir)
    if not snap_dir.exists():
        print(f"ERROR: directorio de snapshots no encontrado: {snap_dir}")
        return

    # Descubrir todos los pasos de tiempo (snapshots)
    snap_files = sorted(snap_dir.glob("snapshot_*.json"))
    if not snap_files:
        print(f"No se encontraron snapshots en {snap_dir}")
        return

    print(f"Encontrados {len(snap_files)} snapshots en {snap_dir}")

    # Organizar por seed y paso
    # Formato esperado: snapshot_<step>_seed<NNN>.json
    from collections import defaultdict
    seed_data = defaultdict(list)  # seed -> [(a, pk_low_mean)]

    for sf in snap_files:
        name = sf.stem  # snapshot_010_seed042
        parts = name.split("_")
        if len(parts) < 3:
            continue
        try:
            seed_str = parts[-1].replace("seed", "")
            seed = int(seed_str)
        except ValueError:
            continue
        if seed not in args.seeds:
            continue

        a, bins = load_snapshot(sf)
        if a is None or not bins:
            continue

        pk_lk = pk_low_k_mean(bins, n_bins_low=2)
        if pk_lk > 0.0:
            seed_data[seed].append((a, pk_lk))

    if not seed_data:
        print("No se encontraron datos de crecimiento. Verificar formato de snapshots.")
        print("Formato esperado: snapshot_<step>_seed<NNN>.json con {'a': ..., 'bins': [...]}")
        # Mostrar archivos encontrados
        for sf in snap_files[:5]:
            print(f"  {sf.name}")
        return

    # Normalizar por el valor en a_init
    for seed in seed_data:
        seed_data[seed].sort(key=lambda x: x[0])

    # Para cada seed: ratio P(k,a)/P(k,a_init)
    all_a_vals = sorted({round(a, 6) for items in seed_data.values() for a, _ in items})

    print(f"\nTiempos encontrados: {all_a_vals[:10]}{'...' if len(all_a_vals) > 10 else ''}")

    # Calcular ratio observado vs esperado (D1^2 EdS)
    growth_data = {}  # seed -> [(a, ratio_obs/ratio_expected)]
    for seed, items in seed_data.items():
        a_init_val = items[0][0]  # usar el primer snapshot como referencia
        pk_init_val = items[0][1]
        if pk_init_val <= 0.0:
            continue
        pts = []
        for a, pk_lk in items:
            ratio_obs = pk_lk / pk_init_val
            d1_ratio_sq = growth_factor_eds(a, a_init_val)**2
            ratio_over_expected = ratio_obs / d1_ratio_sq if d1_ratio_sq > 0 else float("nan")
            pts.append((a, ratio_obs, d1_ratio_sq, ratio_over_expected))
        growth_data[seed] = pts

    # Imprimir tabla de texto
    print(f"\n{'Seed':>6}  {'a':>8}  {'P_ratio':>10}  {'D1²_ratio':>12}  {'obs/exp':>10}")
    print("-" * 55)
    for seed, pts in growth_data.items():
        for a, r_obs, d1sq, r_over_exp in pts:
            print(f"  {seed:4d}  {a:8.4f}  {r_obs:10.3f}  {d1sq:12.3f}  {r_over_exp:10.3f}")

    if not HAS_MATPLOTLIB:
        print("\nmatplotlib no disponible — solo se imprimió la tabla.")
        return

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    colors = ["steelblue", "darkorange", "forestgreen", "crimson",
              "purple", "brown", "pink", "gray"]

    # ── Figura 1: P(k,a)/P(k,a_init) vs D1²(a)/D1²(a_init) ──────────────────
    fig1, ax1 = plt.subplots(figsize=(8, 5))

    a_range = np.linspace(args.a_init, max(all_a_vals) * 1.1, 100)
    d1sq_range = [(a / args.a_init)**2 for a in a_range]
    ax1.plot(a_range, d1sq_range, "k--", lw=2, label=r"$(a/a_\mathrm{init})^2$ (EdS)")

    for (seed, pts), color in zip(growth_data.items(), colors):
        a_vals = [p[0] for p in pts]
        r_obs = [p[1] for p in pts]
        ax1.plot(a_vals, r_obs, "o-", color=color, ms=5, lw=1.5, label=f"seed={seed}")

    ax1.set_xlabel(r"$a$")
    ax1.set_ylabel(r"$P_\mathrm{low-k}(a) / P_\mathrm{low-k}(a_\mathrm{init})$")
    ax1.set_title(f"Crecimiento P(k) bajo-k vs EdS — {args.label}\n(barras = dispersión entre seeds)")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    p1 = out_dir / f"growth_pk_{args.label}.png"
    fig1.savefig(p1, dpi=150)
    print(f"\nFigura 1 guardada: {p1}")
    plt.close(fig1)

    # ── Figura 2: ratio observado/esperado con dispersión entre seeds ─────────
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.axhline(1.0, color="gray", ls="--", lw=1.5, label="ratio=1 (match EdS)")

    # Calcular media y std del ratio obs/exp sobre seeds en cada tiempo
    by_time = defaultdict(list)
    for seed, pts in growth_data.items():
        for a, r_obs, d1sq, r_over_exp in pts:
            if math.isfinite(r_over_exp):
                by_time[round(a, 6)].append(r_over_exp)

    a_sorted = sorted(by_time.keys())
    means = [sum(by_time[a]) / len(by_time[a]) for a in a_sorted]
    stds = [
        math.sqrt(sum((v - m)**2 for v in by_time[a]) / max(len(by_time[a]), 1))
        for a, m in zip(a_sorted, means)
    ]

    ax2.errorbar(a_sorted, means, yerr=stds, fmt="ko-", ms=6, lw=1.5,
                 capsize=4, label=f"media ± std (N_seeds={len(growth_data)})")
    ax2.fill_between(a_sorted,
                     [m - s for m, s in zip(means, stds)],
                     [m + s for m, s in zip(means, stds)],
                     alpha=0.15, color="gray")

    ax2.set_xlabel(r"$a$")
    ax2.set_ylabel(r"$(P_\mathrm{obs}/P_\mathrm{init}) \;/\; (a/a_\mathrm{init})^2$")
    ax2.set_title(f"Ratio observado/esperado (EdS) — {args.label}")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    p2 = out_dir / f"growth_ratio_{args.label}.png"
    fig2.savefig(p2, dpi=150)
    print(f"Figura 2 guardada: {p2}")
    plt.close(fig2)


if __name__ == "__main__":
    main()
