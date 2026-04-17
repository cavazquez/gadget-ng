#!/usr/bin/env python3
"""
Phase 20: Análisis de resultados del PM slab distribuido.

Compara:
- Phase 20 (pm_slab) vs Phase 19 (pm_distributed) para equivalencia física.
- Escalado P=1,2,4 en wall time y v_rms/delta_rms.
- Tabla de bytes/rank para Phase 19 (allreduce nm³) vs Phase 20 (alltoall nm³/P).
"""

import json
import os
import sys
from pathlib import Path


def load_diag(path: Path):
    """Carga diagnostics.jsonl → lista de dicts."""
    lines = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        lines.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    except FileNotFoundError:
        pass
    return lines


def get_final(diag):
    """Retorna el último diagnóstico cosmológico."""
    cosmo = [d for d in diag if "cosmo" in d]
    if cosmo:
        return cosmo[-1]["cosmo"]
    return None


def comm_bytes_phase19(nm: int, n_ranks: int) -> int:
    """Comunicación allreduce: nm³ * 8 bytes por rank."""
    return nm**3 * 8


def comm_bytes_phase20(nm: int, n_ranks: int) -> int:
    """Comunicación alltoall: 2 * (nm³/P) * 8 bytes (2 transposes × nm³/P)
       + halo 4*nm²*8 bytes (density + force borders)."""
    if n_ranks == 0:
        return 0
    transpose = 2 * (nm**3 // n_ranks) * 8
    halos = 4 * nm * nm * 8
    return transpose + halos


def print_comm_table():
    """Tabla comparativa de bytes/rank."""
    print("\n── Comunicación por rank y paso (bytes) ────────────────────────────────────")
    print(f"{'Grid':>8} {'P':>3} {'Phase19 allreduce':>20} {'Phase20 alltoall':>20} {'Mejora':>8}")
    print("-" * 65)
    for nm in [16, 32, 64]:
        for p in [1, 2, 4, 8]:
            b19 = comm_bytes_phase19(nm, p)
            b20 = comm_bytes_phase20(nm, p)
            mejora = b19 / b20 if b20 > 0 else float("inf")
            print(
                f"{nm:>5}³ {p:>3} "
                f"{b19/1024:>17.1f} KB "
                f"{b20/1024:>17.1f} KB "
                f"{mejora:>7.1f}×"
            )
    print()


def print_equivalence(results_dir: Path):
    """Compara v_rms y delta_rms entre Phase 20 y Phase 19."""
    print("── Equivalencia física Phase 20 vs Phase 19 ────────────────────────────────")
    cases = ["eds_N512"]
    for case in cases:
        for p in [1, 2, 4]:
            p19_path = results_dir / f"{case}_phase19_P{p}" / "diagnostics.jsonl"
            p20_path = results_dir / f"{case}_slab_P{p}" / "diagnostics.jsonl"
            d19 = load_diag(p19_path)
            d20 = load_diag(p20_path)
            if not d19 or not d20:
                continue
            f19 = get_final(d19)
            f20 = get_final(d20)
            if not f19 or not f20:
                continue
            vrms_err = abs(f19["v_rms"] - f20["v_rms"]) / (abs(f19["v_rms"]) + 1e-30)
            drms_err = abs(f19["delta_rms"] - f20["delta_rms"]) / (abs(f19["delta_rms"]) + 1e-30)
            print(
                f"  {case} P={p}: "
                f"Δv_rms={vrms_err:.2e}  Δδ_rms={drms_err:.2e}"
            )
    print()


def print_scaling(results_dir: Path):
    """Tabla de wall time para P=1,2,4."""
    print("── Escalado wall time Phase 20 ──────────────────────────────────────────────")
    print(f"{'Caso':>25} {'P':>3} {'wall_time_s':>14}")
    print("-" * 45)
    cases = ["eds_N512_slab", "lcdm_N2000_slab", "eds_N4000_slab"]
    for case in cases:
        for p in [1, 2, 4]:
            path = results_dir / f"{case}_P{p}" / "diagnostics.jsonl"
            diag = load_diag(path)
            if not diag:
                print(f"  {case:>25} P={p}: (sin datos)")
                continue
            # Buscar HpcStepStats o step_wall_ns
            wt = None
            for d in diag:
                if "hpc" in d and "step_wall_ns" in d.get("hpc", {}):
                    wt = (wt or 0) + d["hpc"]["step_wall_ns"]
            if wt is not None:
                print(f"  {case:>25} P={p}: {wt/1e9:>12.3f} s")
            else:
                print(f"  {case:>25} P={p}: (sin hpc stats)")
    print()


def main():
    script_dir = Path(__file__).parent
    results_dir = script_dir / "results"

    if not results_dir.exists():
        print(f"[analyze_phase20] No hay resultados en {results_dir}")
        print("  Ejecuta primero: bash run_phase20.sh")
        sys.exit(0)

    print("=" * 70)
    print(" Phase 20: Análisis PM Slab Distribuido")
    print("=" * 70)

    print_comm_table()
    print_equivalence(results_dir)
    print_scaling(results_dir)

    print("── Estabilidad ──────────────────────────────────────────────────────────────")
    for case in ["eds_N512_slab", "lcdm_N2000_slab"]:
        for p in [1, 2, 4]:
            path = results_dir / f"{case}_P{p}" / "diagnostics.jsonl"
            diag = load_diag(path)
            if not diag:
                continue
            has_nan = any(
                d.get("cosmo", {}).get("v_rms") != d.get("cosmo", {}).get("v_rms")
                for d in diag
                if "cosmo" in d
            )
            status = "NaN detectado ⚠" if has_nan else "estable ✓"
            print(f"  {case} P={p}: {status}")
    print()


if __name__ == "__main__":
    main()
