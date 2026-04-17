#!/usr/bin/env python3
"""
Phase 13 — Validación física Morton vs Hilbert
Compara resultados de los 4 casos de validación y verifica tolerancias explícitas.

Casos: N ∈ {2000, 8000} × P ∈ {2, 4}
Tolerancias (justificación en comentarios):
  - |ΔE/E₀|_diff < 0.05      : diferencia relativa de drift entre Morton y Hilbert < 5%
  - |Δ|Δp||     < 1e-6       : momento lineal idéntico (invariante global)
  - |Δ|ΔL||     < 1e-4       : momento angular muy similar
  - |ΔKE_max|   < 0.10       : diferencia máxima de KE < 10%
"""

import os
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results"

TOL_DRIFT_DIFF  = 0.05   # diferencia relativa en |ΔE/E₀| entre Morton y Hilbert
TOL_MOMENTUM    = 1e-6   # diferencia en |Δp| final
TOL_ANG_MOM     = 1e-4   # diferencia en |ΔL| final
TOL_KE_MAX_DIFF = 0.10   # diferencia max de KE relativa

VALID_CASES = [
    {"n": 2000, "p": 2},
    {"n": 2000, "p": 4},
    {"n": 8000, "p": 2},
    {"n": 8000, "p": 4},
]

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"

def load_timings(n, p, kind):
    name = f"valid_N{n}_P{p}_{kind}"
    path = RESULTS_DIR / name / "timings.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)

def extract_physics(data):
    if data is None:
        return None
    return {
        "drift":      data.get("energy_drift_rel", None),
        "momentum":   data.get("momentum_drift", None),
        "ang_mom":    data.get("ang_mom_drift", None),
        "ke_final":   data.get("ke_final", None),
        "ke_initial": data.get("ke_initial", None),
    }

print("=" * 70)
print("Phase 13 — Validación física Morton vs Hilbert")
print("=" * 70)
print()

all_pass = True
results = []

for case in VALID_CASES:
    n, p = case["n"], case["p"]

    data_m = load_timings(n, p, "morton")
    data_h = load_timings(n, p, "hilbert")

    if data_m is None and data_h is None:
        print(f"[{SKIP}] N={n}, P={p}: sin resultados (ejecutar run_phase13.sh primero)")
        results.append({"n": n, "p": p, "status": SKIP})
        continue
    if data_m is None:
        print(f"[{SKIP}] N={n}, P={p}: falta Morton")
        results.append({"n": n, "p": p, "status": SKIP})
        continue
    if data_h is None:
        print(f"[{SKIP}] N={n}, P={p}: falta Hilbert")
        results.append({"n": n, "p": p, "status": SKIP})
        continue

    phys_m = extract_physics(data_m)
    phys_h = extract_physics(data_h)

    print(f"N={n:>5}, P={p}")

    case_pass = True
    checks = []

    # 1. Drift energético
    drift_m = phys_m.get("drift")
    drift_h = phys_h.get("drift")
    if drift_m is not None and drift_h is not None:
        diff = abs(drift_m - drift_h)
        ok = diff < TOL_DRIFT_DIFF
        status = PASS if ok else FAIL
        if not ok:
            case_pass = False
        checks.append(f"  drift: Morton={drift_m:.4e}, Hilbert={drift_h:.4e}, "
                       f"|diff|={diff:.4e} < {TOL_DRIFT_DIFF:.2e} → {status}")
    else:
        checks.append(f"  drift: no disponible")

    # 2. Momento lineal
    mom_m = phys_m.get("momentum")
    mom_h = phys_h.get("momentum")
    if mom_m is not None and mom_h is not None:
        diff = abs(mom_m - mom_h)
        ok = diff < TOL_MOMENTUM
        status = PASS if ok else FAIL
        if not ok:
            case_pass = False
        checks.append(f"  |Δp|:  Morton={mom_m:.4e}, Hilbert={mom_h:.4e}, "
                       f"|diff|={diff:.4e} < {TOL_MOMENTUM:.2e} → {status}")
    else:
        checks.append(f"  |Δp|:  no disponible")

    # 3. Momento angular
    ang_m = phys_m.get("ang_mom")
    ang_h = phys_h.get("ang_mom")
    if ang_m is not None and ang_h is not None:
        diff = abs(ang_m - ang_h)
        ok = diff < TOL_ANG_MOM
        status = PASS if ok else FAIL
        if not ok:
            case_pass = False
        checks.append(f"  |ΔL|:  Morton={ang_m:.4e}, Hilbert={ang_h:.4e}, "
                       f"|diff|={diff:.4e} < {TOL_ANG_MOM:.2e} → {status}")
    else:
        checks.append(f"  |ΔL|:  no disponible")

    # 4. KE final relativo
    ke_m = phys_m.get("ke_final")
    ke_h = phys_h.get("ke_final")
    ke0  = phys_m.get("ke_initial")
    if ke_m is not None and ke_h is not None and ke0 is not None and ke0 != 0:
        diff_rel = abs(ke_m - ke_h) / abs(ke0)
        ok = diff_rel < TOL_KE_MAX_DIFF
        status = PASS if ok else FAIL
        if not ok:
            case_pass = False
        checks.append(f"  ΔKE:   Morton={ke_m:.4e}, Hilbert={ke_h:.4e}, "
                       f"|Δ|/KE₀={diff_rel:.4e} < {TOL_KE_MAX_DIFF:.2e} → {status}")
    else:
        checks.append(f"  ΔKE:   no disponible")

    for c in checks:
        print(c)

    case_status = PASS if case_pass else FAIL
    print(f"  → CASO: {case_status}")
    print()

    if not case_pass:
        all_pass = False

    results.append({"n": n, "p": p, "status": case_status})

# Resumen
print("=" * 70)
passed  = sum(1 for r in results if r["status"] == PASS)
failed  = sum(1 for r in results if r["status"] == FAIL)
skipped = sum(1 for r in results if r["status"] == SKIP)

print(f"Resultado: {passed} PASS / {failed} FAIL / {skipped} SKIP de {len(results)} casos")
if failed == 0 and skipped == 0:
    print("✓ VALIDACIÓN FÍSICA COMPLETA: Morton y Hilbert son físicamente equivalentes")
elif skipped == len(results):
    print("⚠ Sin datos: ejecutar run_phase13.sh --filter valid primero")
else:
    print("✗ FALLOS DE VALIDACIÓN: revisar tolerancias o implementación Hilbert")

sys.exit(0 if (failed == 0) else 1)
