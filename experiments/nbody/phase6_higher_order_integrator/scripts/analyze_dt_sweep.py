#!/usr/bin/env python3
"""analyze_dt_sweep.py — Unifica los barridos dt producidos por los tests Rust.

Los tests unitarios de `gadget-ng-integrators` vuelcan CSVs en
`results/harmonic_convergence.csv` y `results/kepler_convergence.csv`. Este
script los une en un único `results/yoshida_convergence.csv` con columnas:

    system, integrator, dt, err_metric, err_value, fitted_order

Para el oscilador armónico `err_metric = dE_rel_max`. Para Kepler se generan
3 métricas: `dE_rel_final`, `dL_rel_final`, `closure`.

Los ajustes log-log se calculan por mínimos cuadrados en log(err) vs log(dt).

Uso:
    cargo test --release -p gadget-ng-integrators --test yoshida_harmonic_convergence
    cargo test --release -p gadget-ng-integrators --test yoshida_kepler_orbit
    python3 analyze_dt_sweep.py
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

EXP_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = EXP_DIR / "results"


def fit_slope(x: np.ndarray, y: np.ndarray) -> float:
    lx = np.log(x)
    ly = np.log(np.clip(y, 1e-300, None))
    mx = lx.mean()
    my = ly.mean()
    num = ((lx - mx) * (ly - my)).sum()
    den = ((lx - mx) ** 2).sum()
    if den == 0:
        return float("nan")
    return float(num / den)


def main() -> int:
    harmonic = RESULTS_DIR / "harmonic_convergence.csv"
    kepler = RESULTS_DIR / "kepler_convergence.csv"

    rows = []
    if harmonic.exists():
        h = pd.read_csv(harmonic)
        for integ in h["integrator"].unique():
            sub = h[h["integrator"] == integ].sort_values("dt")
            slope = fit_slope(sub["dt"].values, sub["err_rel_max"].values)
            for _, r in sub.iterrows():
                rows.append({
                    "system": "harmonic",
                    "integrator": integ,
                    "dt": float(r["dt"]),
                    "err_metric": "dE_rel_max",
                    "err_value": float(r["err_rel_max"]),
                    "fitted_order": slope,
                })
    else:
        print(f"(aviso) no existe {harmonic}; corre `cargo test` antes.",
              file=sys.stderr)

    if kepler.exists():
        k = pd.read_csv(kepler)
        metric_map = {
            "dE_rel_final": "dE_rel_final",
            "dL_rel_final": "dL_rel_final",
            "closure": "closure",
        }
        for integ in k["integrator"].unique():
            sub = k[k["integrator"] == integ].sort_values("dt")
            for metric_col, metric_out in metric_map.items():
                slope = fit_slope(sub["dt"].values, sub[metric_col].values)
                for _, r in sub.iterrows():
                    rows.append({
                        "system": "kepler",
                        "integrator": integ,
                        "dt": float(r["dt"]),
                        "err_metric": metric_out,
                        "err_value": float(r[metric_col]),
                        "fitted_order": slope,
                    })
    else:
        print(f"(aviso) no existe {kepler}; corre `cargo test` antes.",
              file=sys.stderr)

    if not rows:
        print("Sin datos — nada que escribir.", file=sys.stderr)
        return 1

    out = RESULTS_DIR / "yoshida_convergence.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Escrito → {out}")

    summary = (
        pd.DataFrame(rows)
        .groupby(["system", "integrator", "err_metric"])["fitted_order"]
        .first()
        .reset_index()
    )
    print("\nPendientes ajustadas (log-log):")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
