#!/usr/bin/env python3
"""analyze_local_global.py — Relaciona error local de fuerza con drift global de E.

Lee:
    results/bh_mac_softening.csv        (error local por variante, generado por el test Rust)
    results/phase5_summary.csv          (drift dinámico acumulado por run)

Las claves de emparejamiento son (distribution, N, variant). Algunos IDs
difieren ligeramente entre el test Rust y los runs (p.ej. `uniform_sphere` vs
`uniform`); se normaliza aquí.

Produce:
    results/local_vs_global.csv        (38-40 filas, 1 por run con error local y drift)

y reporta:
    - coeficiente de correlación de Pearson entre mean_err y |ΔE/E|_final
    - coeficiente de correlación entre max_err y |ΔE/E|_max
    - estimación del 'piso del integrador': el dE_rel_final mínimo observado,
      que no depende ya del error local
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = EXP_DIR / "results"

LOCAL_CSV = RESULTS_DIR / "bh_mac_softening.csv"
GLOBAL_CSV = RESULTS_DIR / "phase5_summary.csv"
OUT_CSV = RESULTS_DIR / "local_vs_global.csv"


def normalize_dist(d: str) -> str:
    # El test local usa 'uniform_sphere'; los runs dinámicos usan 'uniform'.
    if d == "uniform_sphere":
        return "uniform"
    return d


def main() -> int:
    local = pd.read_csv(LOCAL_CSV)
    global_ = pd.read_csv(GLOBAL_CSV)

    local["distribution"] = local["distribution"].apply(normalize_dist)

    merged = pd.merge(
        local[["distribution", "N", "variant", "mean_err", "max_err", "p95_err",
               "rms_err", "mean_angular_err", "max_angular_err",
               "time_bh_ms", "opened_nodes", "mean_depth"]],
        global_[["distribution", "N", "variant",
                 "dE_rel_final", "dE_rel_max", "dE_rel_mean",
                 "dp_rel_max", "dL_rel_max",
                 "total_wall_s", "mean_step_wall_s", "Q_virial_final",
                 "r_hm_final", "r_hm_init"]],
        on=["distribution", "N", "variant"],
        how="inner",
    )
    merged.to_csv(OUT_CSV, index=False)
    print(f"Matriz conjunta → {OUT_CSV}  ({len(merged)} filas)")

    # ── Correlaciones globales ───────────────────────────────────────────────
    def corr(xcol: str, ycol: str) -> float:
        x = np.log10(np.clip(merged[xcol].values, 1e-12, None))
        y = np.log10(np.clip(merged[ycol].values, 1e-12, None))
        return float(np.corrcoef(x, y)[0, 1])

    r_mean_final = corr("mean_err", "dE_rel_final")
    r_mean_max = corr("mean_err", "dE_rel_max")
    r_max_final = corr("max_err", "dE_rel_final")
    r_max_max = corr("max_err", "dE_rel_max")

    print("\n== Correlaciones log-log (global) ==")
    print(f"  corr(log mean_err, log dE_rel_final) = {r_mean_final:+.3f}")
    print(f"  corr(log mean_err, log dE_rel_max)   = {r_mean_max:+.3f}")
    print(f"  corr(log max_err , log dE_rel_final) = {r_max_final:+.3f}")
    print(f"  corr(log max_err , log dE_rel_max)   = {r_max_max:+.3f}")

    # ── Piso del integrador ──────────────────────────────────────────────────
    # Tomamos el min de dE_rel_final agrupado por distribución: si muchas
    # variantes con error local muy distinto comparten dE_rel ~ constante,
    # ese valor es el piso del integrador para esa distribución.
    floor = merged.groupby(["distribution", "N"]).agg(
        dE_floor=("dE_rel_final", "min"),
        dE_spread=("dE_rel_final", lambda x: x.max() / max(x.min(), 1e-15)),
        local_spread=("mean_err", lambda x: x.max() / max(x.min(), 1e-15)),
    ).reset_index()
    print("\n== Piso del integrador por distribución ==")
    print(floor.to_string(index=False))

    # ── Correlación por distribución ─────────────────────────────────────────
    print("\n== Correlaciones por (distribución, N) ==")
    by_group = []
    for (dist, n), sub in merged.groupby(["distribution", "N"]):
        if len(sub) < 3:
            continue
        x = np.log10(np.clip(sub["mean_err"].values, 1e-12, None))
        y = np.log10(np.clip(sub["dE_rel_final"].values, 1e-12, None))
        r = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else float("nan")
        by_group.append({"distribution": dist, "N": n, "r_local_vs_global": r,
                         "local_min": sub["mean_err"].min(),
                         "local_max": sub["mean_err"].max(),
                         "global_min": sub["dE_rel_final"].min(),
                         "global_max": sub["dE_rel_final"].max()})
    by_group_df = pd.DataFrame(by_group)
    print(by_group_df.to_string(index=False))
    by_group_df.to_csv(RESULTS_DIR / "local_vs_global_by_group.csv", index=False)

    # Resumen final
    print("\n== Resumen ==")
    print(f"Corr log-log global mean_err ↔ dE_rel_final: {r_mean_final:+.3f}")
    if r_mean_final > 0.5:
        print("  → correlación fuerte: reducir error local reduce drift")
    elif r_mean_final > 0.2:
        print("  → correlación débil: presencia del piso del integrador")
    else:
        print("  → sin correlación: drift dominado por integrador/simpléctica")

    return 0


if __name__ == "__main__":
    sys.exit(main())
