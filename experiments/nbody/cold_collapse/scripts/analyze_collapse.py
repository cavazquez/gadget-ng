#!/usr/bin/env python3
"""analyze_collapse.py — Análisis del colapso gravitacional frío.

Calcula la serie temporal de métricas físicas y compara con predicciones
analíticas (tiempo de caída libre, virialización).

Genera:
  - results/collapse_timeseries.csv
  - Imprime en consola tabla de métricas clave

Uso:
    cd experiments/nbody/cold_collapse
    python scripts/analyze_collapse.py
"""

import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "analysis"))

import pandas as pd
from snapshot_metrics import load_timeseries

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
RUNS_DIR = EXPERIMENT_DIR / "runs"
RESULTS_DIR = EXPERIMENT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Parámetros físicos
G = 1.0
R = 1.0
M_TOT = 1.0
EPS = 0.05

# Tiempo de caída libre analítico: T_ff = π·√(R³/(2·G·M))
T_FF = math.pi * math.sqrt(R**3 / (2.0 * G * M_TOT))
# Radio de media masa de esfera uniforme: r_hm = R·(1/2)^(1/3) ≈ 0.794·R
R_HM_THEORY = R * (0.5 ** (1.0 / 3.0))


def main():
    print("=== Análisis de Colapso Gravitacional Frío ===\n")
    print(f"Parámetros: G={G}, M={M_TOT}, R={R}, ε={EPS}")
    print(f"T_ff = π·√(R³/2GM) = {T_FF:.4f} [unidades internas]")
    print(f"r_hm inicial teórico = {R_HM_THEORY:.4f} = {R_HM_THEORY/R:.3f}·R\n")

    run_dir = RUNS_DIR / "collapse"
    if not run_dir.exists():
        print(
            "ERROR: No se encontraron datos.\n"
            "Ejecuta primero: bash scripts/run_collapse.sh"
        )
        sys.exit(1)

    print("Cargando snapshots... ", end="", flush=True)
    try:
        df = load_timeseries(run_dir, softening=EPS, G=G, verbose=False)
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)

    if df.empty:
        print("Sin datos válidos.")
        sys.exit(1)

    print(f"OK ({len(df)} frames)\n")

    # Añadir tiempo normalizado por T_ff
    df["t_over_tff"] = df["t"] / T_FF

    # Guardar serie temporal
    out = RESULTS_DIR / "collapse_timeseries.csv"
    df.to_csv(out, index=False)
    print(f"Serie temporal guardada en {out}\n")

    # Hitos del colapso
    r_hm_0 = df["r_hm"].iloc[0]
    print(f"r_hm inicial (simulación): {r_hm_0:.4f}  (teórico: {R_HM_THEORY:.4f})")

    # ¿Cuándo cae r_hm al 50% del inicial?
    collapsed = df[df["r_hm"] < 0.5 * r_hm_0]
    if not collapsed.empty:
        t_collapse = collapsed.iloc[0]["t"]
        print(f"r_hm = 0.5·r_hm₀ alcanzado en t = {t_collapse:.3f} = {t_collapse/T_FF:.2f}·T_ff")
    else:
        print("AVISO: r_hm nunca cae al 50% del inicial en la simulación actual")

    # ¿Cuándo cae r_hm al 20%?
    deep_collapse = df[df["r_hm"] < 0.2 * r_hm_0]
    if not deep_collapse.empty:
        t_dc = deep_collapse.iloc[0]["t"]
        print(f"r_hm = 0.2·r_hm₀ alcanzado en t = {t_dc:.3f} = {t_dc/T_FF:.2f}·T_ff")

    # Estado final
    last = df.iloc[-1]
    print(f"\nEstado final (t = {last['t']:.3f} = {last['t_over_tff']:.2f}·T_ff):")
    print(f"  r_hm     = {last['r_hm']:.4f}")
    print(f"  Q virial = {last['Q']:.4f}  (equilibrio → 0.5)")
    print(f"  |ΔE/E₀|  = {last['dE_rel']:.4e}")
    print(f"  KE       = {last['KE']:.6f}")
    print(f"  PE       = {last['PE']:.6f}")
    print(f"  E_total  = {last['E']:.6f}")

    # Tabla resumen en puntos t_ff × 0.5, 1.0, 1.5, 2.0, 3.0, 5.0
    print("\n--- Evolución en hitos de T_ff ---")
    print(f"{'t/T_ff':>8}  {'r_hm':>8}  {'Q':>8}  {'|ΔE/E₀|':>12}  {'KE':>10}  {'PE':>10}")
    for target in [0.3, 0.6, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]:
        mask = df["t_over_tff"] >= target
        if mask.any():
            row = df[mask].iloc[0]
            print(
                f"{row['t_over_tff']:>8.2f}  "
                f"{row['r_hm']:>8.4f}  "
                f"{row['Q']:>8.4f}  "
                f"{row['dE_rel']:>12.4e}  "
                f"{row['KE']:>10.6f}  "
                f"{row['PE']:>10.6f}"
            )

    print("\n=== Listo ===")
    print("Siguiente: python scripts/plot_collapse.py")


if __name__ == "__main__":
    main()
