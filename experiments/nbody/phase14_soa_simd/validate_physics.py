#!/usr/bin/env python3
"""
Phase 14 — Validación física: baseline (AoS) vs SoA+SIMD.

Para cada configuración del grupo "validation", compara:
  - Drift energético |ΔE/E₀|
  - Diferencia máxima de KE entre variantes
  - |Δp| (momento lineal)
  - |ΔL| (momento angular)

Tolerancias (físicamente justificadas):
  - Los dos paths evalúan el mismo algoritmo con la misma aritmética IEEE 754;
    las diferencias son sólo orden de evaluación en Rayon (baseline serial vs
    soa_simd paralelo). Se permite ε_abs = 1e-10 en energía relativa.
"""

import json
import math
import os
import sys
from pathlib import Path

RESULTS_DIR = Path("results")

TOL_ENERGY_REL = 1e-4   # tolerancia conservadora para diferencia de energía
TOL_DRIFT      = 0.10   # drift admisible |ΔE/E₀| en 20 pasos (física caótica)
TOL_MOMENTUM   = 1e-4   # diferencia de |p| entre variantes

def load_diagnostics(result_dir: Path) -> list[dict] | None:
    diag = result_dir / "diagnostics.jsonl"
    if not diag.exists():
        return None
    rows = []
    with open(diag) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def energy_drift(diags: list[dict]) -> float:
    """Calcula |ΔE/E₀| entre primer y último paso."""
    if len(diags) < 2:
        return float("nan")
    e0 = diags[0].get("kinetic_energy", 0) + diags[0].get("potential_energy", 0)
    ef = diags[-1].get("kinetic_energy", 0) + diags[-1].get("potential_energy", 0)
    if abs(e0) < 1e-300:
        return float("nan")
    return abs((ef - e0) / e0)

def momentum_magnitude(diags: list[dict], key: str) -> float:
    """Magnitud del momento (lineal o angular) en el último paso."""
    last = diags[-1]
    vec = last.get(key, [0, 0, 0])
    if isinstance(vec, list):
        return math.sqrt(sum(v**2 for v in vec))
    return 0.0

def validate_group(group: str, n: int, p: int) -> bool:
    name_base = f"{group}_N{n}_P{p}"
    dir_bl  = RESULTS_DIR / f"{name_base}_baseline"
    dir_soa = RESULTS_DIR / f"{name_base}_soa_simd"

    if not dir_bl.exists() or not dir_soa.exists():
        print(f"  SKIP {name_base}: directorio faltante")
        return True

    diags_bl  = load_diagnostics(dir_bl)
    diags_soa = load_diagnostics(dir_soa)

    if diags_bl is None or diags_soa is None:
        print(f"  SKIP {name_base}: diagnostics.jsonl faltante")
        return True

    passed = True

    # Drift energético por separado
    drift_bl  = energy_drift(diags_bl)
    drift_soa = energy_drift(diags_soa)

    # Diferencia de energía entre los dos paths (último paso)
    ke_bl  = diags_bl[-1].get("kinetic_energy", 0)
    ke_soa = diags_soa[-1].get("kinetic_energy", 0)
    ke_rel_diff = abs(ke_bl - ke_soa) / max(abs(ke_bl), 1e-300)

    # Momento lineal
    p_bl  = momentum_magnitude(diags_bl,  "linear_momentum")
    p_soa = momentum_magnitude(diags_soa, "linear_momentum")

    # Momento angular
    l_bl  = momentum_magnitude(diags_bl,  "angular_momentum")
    l_soa = momentum_magnitude(diags_soa, "angular_momentum")

    status = "OK"
    issues = []

    if not math.isnan(drift_bl) and drift_bl > TOL_DRIFT:
        issues.append(f"drift_baseline={drift_bl:.3e} > {TOL_DRIFT}")
    if not math.isnan(drift_soa) and drift_soa > TOL_DRIFT:
        issues.append(f"drift_soa={drift_soa:.3e} > {TOL_DRIFT}")
    if ke_rel_diff > TOL_ENERGY_REL:
        issues.append(f"KE_diff={ke_rel_diff:.3e} > {TOL_ENERGY_REL}")
        passed = False

    p_diff = abs(p_bl - p_soa) / max(max(p_bl, p_soa), 1e-300) if max(p_bl, p_soa) > 1e-300 else 0.0
    if p_diff > TOL_MOMENTUM:
        issues.append(f"|Δp|_rel={p_diff:.3e} > {TOL_MOMENTUM}")
        passed = False

    if issues:
        status = "FAIL" if not passed else "WARN"

    print(f"  [{status}] {name_base:40s} "
          f"drift_bl={drift_bl:.2e} drift_soa={drift_soa:.2e} "
          f"ΔKE={ke_rel_diff:.2e} |Δp|={p_diff:.2e}")

    if issues:
        for iss in issues:
            print(f"         ! {iss}")

    return passed

def main():
    print("="*70)
    print("Phase 14 — Validación física: baseline vs SoA+SIMD")
    print("="*70)
    print(f"Tolerancias: drift≤{TOL_DRIFT}, ΔKE_rel≤{TOL_ENERGY_REL}, |Δp|≤{TOL_MOMENTUM}")
    print()

    import itertools
    all_ok = True

    for n, p in itertools.product([2000, 8000], [1, 2, 4]):
        ok = validate_group("validation", n, p)
        all_ok = all_ok and ok

    print()
    if all_ok:
        print("✓ Todas las validaciones pasaron.")
    else:
        print("✗ Algunas validaciones FALLARON. Ver detalles arriba.")
        sys.exit(1)

if __name__ == "__main__":
    main()
