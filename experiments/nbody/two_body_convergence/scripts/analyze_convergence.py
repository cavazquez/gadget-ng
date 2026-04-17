#!/usr/bin/env python3
"""analyze_convergence.py — Análisis de convergencia Kepler.

Lee los snapshots de cada ejecución en runs/ y calcula:
  - Error relativo de energía |ΔE/E₀| al final de 1 período
  - Error relativo de momento angular |ΔL/L₀| al final de 1 período
  - Serie temporal E(t) para cada dt

Guarda resultados en results/convergence.csv y results/energy_timeseries.csv.

Uso:
    cd experiments/nbody/two_body_convergence
    python scripts/analyze_convergence.py
"""

import math
import os
import sys
from pathlib import Path

# Añadir el directorio de análisis al path.
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "analysis"))

import pandas as pd
from snapshot_metrics import (
    load_snapshot_dir,
    iter_snapshot_dirs,
    kinetic_energy,
    angular_momentum_z,
    potential_energy,
)

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
RUNS_DIR = EXPERIMENT_DIR / "runs"
RESULTS_DIR = EXPERIMENT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Parámetros físicos del problema
G = 1.0
SOFTENING = 1e-6
T_ORBIT = 2.0 * math.pi  # período Kepler con G=1, M=1, r=1

# Etiquetas y dt nominales
RUNS = {
    "dt020": {"dt": T_ORBIT / 20, "label": "T/20"},
    "dt050": {"dt": T_ORBIT / 50, "label": "T/50"},
    "dt100": {"dt": T_ORBIT / 100, "label": "T/100"},
    "dt200": {"dt": T_ORBIT / 200, "label": "T/200"},
    "dt500": {"dt": T_ORBIT / 500, "label": "T/500"},
}


def analyze_run(run_label: str, run_info: dict) -> dict:
    """Analiza una ejecución y devuelve métricas de convergencia."""
    run_dir = RUNS_DIR / run_label
    if not run_dir.exists():
        print(f"  ADVERTENCIA: directorio {run_dir} no existe (ejecuta run_convergence.sh)")
        return None

    snap_dirs = list(iter_snapshot_dirs(run_dir))
    if not snap_dirs:
        print(f"  ADVERTENCIA: sin snapshots en {run_dir}")
        return None

    # Primer snapshot: estado inicial.
    p0, t0 = load_snapshot_dir(snap_dirs[0])
    ke0 = kinetic_energy(p0)
    pe0 = potential_energy(p0, softening=SOFTENING, G=G)
    e0 = ke0 + pe0
    l0 = angular_momentum_z(p0)

    # Último snapshot: estado final.
    p_f, t_f = load_snapshot_dir(snap_dirs[-1])
    ke_f = kinetic_energy(p_f)
    pe_f = potential_energy(p_f, softening=SOFTENING, G=G)
    e_f = ke_f + pe_f
    l_f = angular_momentum_z(p_f)

    de_rel = abs(e_f - e0) / abs(e0) if abs(e0) > 1e-30 else 0.0
    dl_rel = abs(l_f - l0) / abs(l0) if abs(l0) > 1e-30 else 0.0

    return {
        "run": run_label,
        "label": run_info["label"],
        "dt": run_info["dt"],
        "n_snaps": len(snap_dirs),
        "t_final": t_f,
        "E0": e0,
        "E_final": e_f,
        "dE_rel": de_rel,
        "L0": l0,
        "L_final": l_f,
        "dL_rel": dl_rel,
    }


def build_timeseries(run_label: str) -> pd.DataFrame:
    """Construye la serie temporal E(t) para una ejecución."""
    run_dir = RUNS_DIR / run_label
    snap_dirs = list(iter_snapshot_dirs(run_dir))
    if not snap_dirs:
        return pd.DataFrame()

    p0, _ = load_snapshot_dir(snap_dirs[0])
    e0 = kinetic_energy(p0) + potential_energy(p0, softening=SOFTENING, G=G)

    rows = []
    for snap_dir in snap_dirs:
        p, t = load_snapshot_dir(snap_dir)
        ke = kinetic_energy(p)
        pe = potential_energy(p, softening=SOFTENING, G=G)
        e = ke + pe
        rows.append({
            "run": run_label,
            "t": t,
            "KE": ke,
            "PE": pe,
            "E": e,
            "dE_rel": abs(e - e0) / abs(e0) if abs(e0) > 1e-30 else 0.0,
        })
    return pd.DataFrame(rows)


def main():
    print("=== Análisis de Convergencia Kepler ===\n")

    # Comprobar si existen datos
    if not RUNS_DIR.exists() or not any(
        (RUNS_DIR / r).exists() for r in RUNS
    ):
        print(
            "ERROR: No se encontraron datos de ejecución.\n"
            "Ejecuta primero: bash scripts/run_convergence.sh\n"
        )
        sys.exit(1)

    # Métricas de convergencia
    rows = []
    for run_label, run_info in RUNS.items():
        print(f"Analizando {run_label} ({run_info['label']})...")
        result = analyze_run(run_label, run_info)
        if result is not None:
            rows.append(result)
            print(
                f"  dt={result['dt']:.5f}  "
                f"|ΔE/E₀|={result['dE_rel']:.3e}  "
                f"|ΔL/L₀|={result['dL_rel']:.3e}"
            )

    if not rows:
        print("Sin resultados válidos.")
        sys.exit(1)

    df_conv = pd.DataFrame(rows)
    out_conv = RESULTS_DIR / "convergence.csv"
    df_conv.to_csv(out_conv, index=False)
    print(f"\nTabla de convergencia guardada en {out_conv}")

    # Verificar orden de convergencia (pendiente en log-log)
    if len(df_conv) >= 2:
        import numpy as np
        log_dt = np.log10(df_conv["dt"].values)
        log_de = np.log10(df_conv["dE_rel"].values + 1e-16)
        slope, _ = np.polyfit(log_dt, log_de, 1)
        print(f"\n  Orden de convergencia estimado: {slope:.2f}")
        print(f"  (Esperado ≈ 2.0 para leapfrog KDK de segundo orden)")
        df_conv["convergence_order_estimate"] = slope

    # Series temporales
    print("\nConstruyendo series temporales E(t)...")
    ts_frames = []
    for run_label in RUNS:
        df_ts = build_timeseries(run_label)
        if not df_ts.empty:
            ts_frames.append(df_ts)

    if ts_frames:
        df_ts_all = pd.concat(ts_frames, ignore_index=True)
        out_ts = RESULTS_DIR / "energy_timeseries.csv"
        df_ts_all.to_csv(out_ts, index=False)
        print(f"Series temporales guardadas en {out_ts}")

    # Resumen
    print("\n=== Tabla de Convergencia ===")
    print(df_conv[["label", "dt", "dE_rel", "dL_rel"]].to_string(index=False))
    print("\nListo. Siguiente: python scripts/plot_convergence.py")


if __name__ == "__main__":
    main()
