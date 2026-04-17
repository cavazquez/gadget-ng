#!/usr/bin/env python3
"""
Validación física MPI end-to-end: baseline export vs export reducido (Fase 12).

Compara pares (factor=0.0, factor=1.4) con idénticos parámetros físicos y
verifica que reducir el volumen LET no degrada la física más allá de las
tolerancias definidas.

Metodología:
  - Mismo seed, mismo N, misma distribución (Plummer a/ε=2), mismo dt
  - 20 pasos, P=2 y P=4
  - Métricas: KE, |p|, |L| de diagnostics.jsonl

Tolerancias (justificadas abajo):
  1. drift_KE_baseline < 5%  (20 pasos Plummer denso → caótico)
  2. drift_KE_reduced  < 5%  (idem)
  3. max_KE_diff(baseline_vs_reduced) < 5%
     (factor=1.4 → theta_export=0.7; error O(0.7^4)≈0.24 vs O(0.5^4)≈0.06
      pero la diferencia en KE integrada a 20 pasos es mucho menor que el
      error de fuerza instantáneo; Fase 11 mostró <1% con LetTree; aquí
      aceptamos hasta 5% para acomodar mayor truncación del export)
  4. max_|p|_baseline < 0.05  (momento lineal conservado)
  5. max_|p|_reduced  < 0.05
  6. drift_|L|_baseline < 0.10
  7. drift_|L|_reduced  < 0.10

Uso:
    python3 validate_physics.py [--build-only] [--skip-run]
"""

import os
import sys
import json
import math
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.abspath(os.path.join(SCRIPT_DIR, "../../../.."))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")
CONFIGS_DIR = os.path.join(SCRIPT_DIR, "..", "configs")

BUILD_ONLY = "--build-only" in sys.argv
SKIP_RUN   = "--skip-run"   in sys.argv

# ── Compilación ───────────────────────────────────────────────────────────────
if not SKIP_RUN:
    print("=== Compilando gadget-ng (features: simd,mpi) ===")
    result = subprocess.run(
        ["cargo", "build", "--release", "--features", "simd,mpi", "-p", "gadget-ng-cli"],
        cwd=REPO_ROOT,
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print("ERROR al compilar:")
        print(result.stderr[-2000:])
        sys.exit(1)
    print("Compilación OK")

GADGET_BIN = os.path.join(REPO_ROOT, "target", "release", "gadget-ng")

if BUILD_ONLY:
    print("--build-only: saliendo tras compilación")
    sys.exit(0)

# ── Casos de validación ───────────────────────────────────────────────────────
# Cuatro pares: N ∈ {2000, 8000} × P ∈ {2, 4}
CASES = [
    {"n": 2000, "p": 2},
    {"n": 2000, "p": 4},
    {"n": 8000, "p": 2},
    {"n": 8000, "p": 4},
]

# Las dos variantes a comparar (nombres de config: valid_nN_pP_f{factor_str}.toml)
FACTOR_BASELINE = "0p0"
FACTOR_REDUCED  = "1p4"


# ── Ejecución ─────────────────────────────────────────────────────────────────
def run_case(n, p, factor_str):
    cfg_name = f"valid_n{n}_p{p}_f{factor_str}"
    toml = os.path.join(CONFIGS_DIR, f"{cfg_name}.toml")
    if not os.path.exists(toml):
        print(f"  Config no encontrada: {toml}")
        return False
    out_dir = os.path.join(RESULTS_DIR, f"{cfg_name}_p{p}")
    os.makedirs(out_dir, exist_ok=True)

    cmd = [GADGET_BIN, "stepping", "--config", toml, "--out", out_dir]
    if p > 1:
        cmd = ["mpirun", "--oversubscribe", "-n", str(p)] + cmd

    print(f"  Ejecutando {cfg_name} P={p} ... ", end="", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"FAILED (rc={proc.returncode})")
        print(proc.stderr[-800:])
        return False
    print("ok")
    return True


if not SKIP_RUN:
    print("\n=== Ejecutando runs de validación ===")
    for case in CASES:
        run_case(case["n"], case["p"], FACTOR_BASELINE)
        run_case(case["n"], case["p"], FACTOR_REDUCED)

# ── Carga de diagnósticos ─────────────────────────────────────────────────────
def load_diag(n, p, factor_str):
    out_dir = os.path.join(RESULTS_DIR, f"valid_n{n}_p{p}_f{factor_str}_p{p}")
    diag_path = os.path.join(out_dir, "diagnostics.jsonl")
    if not os.path.exists(diag_path):
        return None
    rows = []
    with open(diag_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    return sorted(rows, key=lambda x: x["step"]) if rows else None


def vec_norm(v):
    return math.sqrt(sum(x * x for x in v))


# ── Análisis de un par ────────────────────────────────────────────────────────
def analyze_pair(n, p):
    base = load_diag(n, p, FACTOR_BASELINE)
    redu = load_diag(n, p, FACTOR_REDUCED)

    if base is None or redu is None:
        missing = []
        if base is None: missing.append(f"baseline(f={FACTOR_BASELINE})")
        if redu is None: missing.append(f"reduced(f={FACTOR_REDUCED})")
        return None, f"  N={n},P={p}: MISSING diagnostics — {', '.join(missing)}"

    if len(base) < 2 or len(redu) < 2:
        return None, f"  N={n},P={p}: menos de 2 pasos en diagnostics"

    ke_base = [r["kinetic_energy"] for r in base]
    ke_redu = [r["kinetic_energy"] for r in redu]
    p_base  = [vec_norm(r["momentum"]) for r in base]
    p_redu  = [vec_norm(r["momentum"]) for r in redu]
    l_base  = [vec_norm(r["angular_momentum"]) for r in base]
    l_redu  = [vec_norm(r["angular_momentum"]) for r in redu]

    ke0_base = max(abs(ke_base[0]), 1e-30)
    ke0_redu = max(abs(ke_redu[0]), 1e-30)

    # 1–2. Drift KE dentro de cada path
    drift_base = abs(ke_base[-1] - ke_base[0]) / ke0_base
    drift_redu = abs(ke_redu[-1] - ke_redu[0]) / ke0_redu

    # 3. Diferencia KE entre paths por paso
    n_steps = min(len(ke_base), len(ke_redu))
    ke_diff_rel = [abs(ke_redu[i] - ke_base[i]) / max(abs(ke_base[i]), 1e-30)
                   for i in range(n_steps)]
    max_ke_diff = max(ke_diff_rel)

    # 4–5. Momento lineal (conservado → debe ser pequeño)
    max_p_base = max(p_base)
    max_p_redu = max(p_redu)

    # 6–7. Momento angular: drift relativo
    l0_base = max(l_base[0], 1e-30)
    l0_redu = max(l_redu[0], 1e-30)
    drift_l_base = max(abs(l - l_base[0]) / l0_base for l in l_base)
    drift_l_redu = max(abs(l - l_redu[0]) / l0_redu for l in l_redu)

    # Tolerancias — justificadas en el docstring del módulo
    TOL_DRIFT   = 0.05   # 5% drift KE (20 pasos, Plummer caótico)
    TOL_KE_DIFF = 0.05   # 5% diferencia KE entre baseline y reduced
    TOL_P       = 0.05   # |p| < 5% (escala del sistema)
    TOL_DL      = 0.10   # drift |L| < 10%

    checks = {
        "drift_KE_baseline":               (drift_base,    TOL_DRIFT),
        "drift_KE_reduced":                (drift_redu,    TOL_DRIFT),
        "max_KE_diff(baseline_vs_reduced)": (max_ke_diff,  TOL_KE_DIFF),
        "max_|p|_baseline":                (max_p_base,    TOL_P),
        "max_|p|_reduced":                 (max_p_redu,    TOL_P),
        "drift_|L|_baseline":              (drift_l_base,  TOL_DL),
        "drift_|L|_reduced":               (drift_l_redu,  TOL_DL),
    }

    passed = all(v <= tol for v, tol in checks.values())
    status = "PASS" if passed else "FAIL"

    lines = [f"  N={n}, P={p}: [{status}]"]
    for name, (val, tol) in checks.items():
        ok = "OK" if val <= tol else "FAIL"
        lines.append(f"    {name:<43} {val:.3e}  (tol={tol:.2e})  [{ok}]")

    return passed, "\n".join(lines)


# ── Ejecución del análisis ────────────────────────────────────────────────────
print("\n=== Análisis de validación física (Phase 12) ===")
print(f"    Comparando baseline (f={FACTOR_BASELINE}) vs reduced (f={FACTOR_REDUCED})")
print()

all_passed = True
results = []
for case in CASES:
    passed, report = analyze_pair(case["n"], case["p"])
    print(report)
    print()
    if passed is not None:
        results.append((case, passed))
        if not passed:
            all_passed = False
    else:
        all_passed = False

if not results:
    print("No hay resultados para analizar.")
    sys.exit(1)

n_pass  = sum(1 for _, p in results if p)
n_total = len(results)
print(f"=== Resultado: {n_pass}/{n_total} casos PASS ===")
print()
print("Justificación de tolerancias:")
print("  - drift_KE < 5%: 20 pasos en Plummer denso caótico; drift esperado ~1-5%.")
print("  - KE_diff < 5%: factor=1.4 → theta_export=0.7 introduce mayor truncación")
print("    que LetTree puro (Fase 11 mostraba <1% con theta=0.5).")
print("    La diferencia en KE integrada es mucho menor que el error de fuerza.")
print("    5% es conservador y permite comparar comportamiento sin ser restrictivo.")
print("  - |p| < 5%: momento lineal debe mantenerse pequeño (conservación numérica).")
print("  - drift_|L| < 10%: momento angular con mayor truncación puede acumularse.")
print()

if all_passed:
    print("[VALIDACIÓN FÍSICA FASE 12: PASADA]")
    sys.exit(0)
else:
    print("[VALIDACIÓN FÍSICA FASE 12: FALLIDA]")
    sys.exit(1)
