#!/usr/bin/env python3
"""analyze_stability.py — Análisis de estabilidad de esfera de Plummer.

Calcula la serie temporal de métricas físicas para la evolución de la
esfera de Plummer y compara las ejecuciones serial vs MPI.

Genera:
  - results/serial_timeseries.csv
  - results/mpi_2rank_timeseries.csv    (si existe)
  - results/mpi_4rank_timeseries.csv    (si existe)
  - results/serial_mpi_comparison.csv   (si existe runs/mpi_*)

Uso:
    cd experiments/nbody/plummer_stability
    python scripts/analyze_stability.py
"""

import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "analysis"))

import pandas as pd
from snapshot_metrics import (
    load_timeseries,
    compare_serial_mpi,
)

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
RUNS_DIR = EXPERIMENT_DIR / "runs"
RESULTS_DIR = EXPERIMENT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Parámetros físicos
G = 1.0
A = 1.0       # radio de escala Plummer
M_TOT = 1.0   # masa total
EPS = 0.05    # suavizado
T_CROSS = math.sqrt(6.0 * A**3 / (G * M_TOT))  # ≈ 2.449


def analyze_run(run_label: str, out_csv: Path) -> pd.DataFrame | None:
    """Carga la serie temporal de una ejecución y la guarda en CSV."""
    run_dir = RUNS_DIR / run_label
    if not run_dir.exists():
        print(f"  AVISO: {run_dir} no existe (omitido)")
        return None

    print(f"  Cargando {run_label}... ", end="", flush=True)
    try:
        df = load_timeseries(run_dir, softening=EPS, G=G)
    except FileNotFoundError as e:
        print(f"SIN DATOS: {e}")
        return None

    # Añadir columnas derivadas
    if not df.empty:
        df["t_over_tcross"] = df["t"] / T_CROSS
        df["run"] = run_label

    df.to_csv(out_csv, index=False)
    print(f"OK ({len(df)} frames)")
    return df


def print_summary(df: pd.DataFrame, label: str):
    """Imprime resumen de métricas."""
    if df is None or df.empty:
        return
    last = df.iloc[-1]
    first = df.iloc[0]
    print(f"\n  [{label}] Resumen final (t = {last['t']:.3f} = {last['t_over_tcross']:.1f}·t_cross):")
    print(f"    |ΔE/E₀|  = {last['dE_rel']:.4e}")
    print(f"    Q virial = {last['Q']:.4f}  (equilibrio → 0.5)")
    print(f"    r_hm     = {last['r_hm']:.4f}  (inicial = {first['r_hm']:.4f})")
    print(f"    |p_total| = {last['p_norm']:.4e}")


def main():
    print("=== Análisis de Estabilidad Plummer ===\n")
    print(f"t_cross = {T_CROSS:.4f} [unidades internas]")
    print(f"t_total ≈ {1000 * 0.025:.1f} = {1000 * 0.025 / T_CROSS:.1f}·t_cross\n")

    if not RUNS_DIR.exists():
        print(
            "ERROR: directorio runs/ no encontrado.\n"
            "Ejecuta primero: bash scripts/run_stability.sh"
        )
        sys.exit(1)

    # Analizar todas las ejecuciones disponibles
    runs_found = {}
    for run_label in ["serial", "mpi_2rank", "mpi_4rank"]:
        out_csv = RESULTS_DIR / f"{run_label}_timeseries.csv"
        df = analyze_run(run_label, out_csv)
        if df is not None:
            runs_found[run_label] = df
            print_summary(df, run_label)

    if not runs_found:
        print("\nSin datos disponibles.")
        sys.exit(1)

    # Comparación serial vs MPI (último snapshot)
    if "serial" in runs_found:
        for mpi_label in ["mpi_2rank", "mpi_4rank"]:
            if (RUNS_DIR / mpi_label).exists():
                print(f"\n--- Comparando serial vs {mpi_label} ---")
                try:
                    df_cmp = compare_serial_mpi(
                        RUNS_DIR / "serial",
                        RUNS_DIR / mpi_label,
                    )
                    max_dr = df_cmp["dr"].max()
                    max_dv = df_cmp["dv"].max()
                    mean_dr = df_cmp["dr"].mean()
                    print(f"  max|Δr| = {max_dr:.4e}")
                    print(f"  mean|Δr| = {mean_dr:.4e}")
                    print(f"  max|Δv| = {max_dv:.4e}")

                    parity_ok = max_dr < 1e-10
                    print(f"  Paridad numérica serial/MPI: {'OK ✓' if parity_ok else 'FALLO ✗'}")
                    print(f"  (tolerancia: max|Δr| < 1e-10)")

                    out = RESULTS_DIR / f"serial_{mpi_label}_comparison.csv"
                    df_cmp.to_csv(out, index=False)
                    print(f"  Guardado: {out}")
                except Exception as e:
                    print(f"  Error en comparación: {e}")

    print("\n=== Listo ===")
    print("Siguiente: python scripts/plot_stability.py")


if __name__ == "__main__":
    main()
