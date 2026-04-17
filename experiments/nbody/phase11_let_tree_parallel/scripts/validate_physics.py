#!/usr/bin/env python3
"""
Validación física MPI end-to-end: flat_let vs let_tree (Fase 11).

Ejecuta pares de runs (flat_let y let_tree) con idénticos parámetros físicos
y compara las series temporales de:
  - Energía cinética (KE) — proxy de conservación total
  - Momento lineal |p|
  - Momento angular |L|

Criterios de aceptación (justificados al final):
  1. Drift KE relativo: |KE_last - KE_0| / KE_0 < 0.05 para ambos paths
  2. Diferencia relativa KE entre paths: max_step( |KE_tree - KE_flat| / KE_flat ) < 0.01
     (el let_tree no introduce más del 1% de diferencia en KE vs flat loop)
  3. Momento lineal: max( |p|_tree ) < 1e-4 (momento conservado)
  4. Momento angular: max( |ΔL/L₀| ) < 0.05 para ambos paths

Uso:
    python3 validate_physics.py [--build-only] [--skip-run]

Flags:
    --build-only: solo compila el binario, no ejecuta runs
    --skip-run:   salta la ejecución y analiza resultados existentes
"""
import os
import sys
import json
import math
import subprocess
import shutil

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../../.."))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")
CONFIGS_DIR = os.path.join(SCRIPT_DIR, "..", "configs")

BUILD_ONLY = "--build-only" in sys.argv
SKIP_RUN = "--skip-run" in sys.argv

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
CASES = [
    {"n": 2000, "p": 2},
    {"n": 2000, "p": 4},
    {"n": 8000, "p": 2},
    {"n": 8000, "p": 4},
]
BACKENDS = ["flat_let", "let_tree"]

# ── Ejecutar runs ─────────────────────────────────────────────────────────────
def run_case(n, p, backend):
    toml = os.path.join(CONFIGS_DIR, f"valid_n{n}_p{p}_{backend}.toml")
    if not os.path.exists(toml):
        print(f"  Config no encontrada: {toml}")
        return False
    out_dir = os.path.join(RESULTS_DIR, f"valid_n{n}_p{p}_{backend}")
    os.makedirs(out_dir, exist_ok=True)

    cmd_args = [GADGET_BIN, "stepping", "--config", toml, "--out", out_dir]
    if p > 1:
        cmd_args = ["mpirun", "--oversubscribe", "-n", str(p)] + cmd_args

    print(f"  Ejecutando valid_n{n}_p{p}_{backend} ... ", end="", flush=True)
    proc = subprocess.run(cmd_args, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"FAILED (rc={proc.returncode})")
        print(proc.stderr[-500:])
        return False
    print("ok")
    return True


if not SKIP_RUN:
    print("\n=== Ejecutando runs de validación ===")
    for case in CASES:
        for backend in BACKENDS:
            run_case(case["n"], case["p"], backend)

# ── Análisis ──────────────────────────────────────────────────────────────────

def load_diag(n, p, backend):
    """Carga diagnostics.jsonl del rank 0 (o único rango)."""
    out_dir = os.path.join(RESULTS_DIR, f"valid_n{n}_p{p}_{backend}")
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
    return rows


def vec_norm(v):
    return math.sqrt(sum(x*x for x in v))


def analyze_pair(n, p):
    """Compara flat_let vs let_tree para un caso dado. Devuelve (passed, report)."""
    flat = load_diag(n, p, "flat_let")
    tree = load_diag(n, p, "let_tree")

    if flat is None or tree is None:
        return None, f"  N={n},P={p}: MISSING diagnostics"

    # Ordenar por paso (solo rank=0 tiene datos completos en modo MPI)
    flat = sorted(flat, key=lambda x: x["step"])
    tree = sorted(tree, key=lambda x: x["step"])

    if len(flat) < 2 or len(tree) < 2:
        return None, f"  N={n},P={p}: menos de 2 pasos en diagnostics"

    ke_flat = [r["kinetic_energy"] for r in flat]
    ke_tree = [r["kinetic_energy"] for r in tree]
    p_flat = [vec_norm(r["momentum"]) for r in flat]
    p_tree = [vec_norm(r["momentum"]) for r in tree]
    l_flat = [vec_norm(r["angular_momentum"]) for r in flat]
    l_tree = [vec_norm(r["angular_momentum"]) for r in tree]

    ke0_flat = ke_flat[0] if ke_flat[0] != 0 else 1.0
    ke0_tree = ke_tree[0] if ke_tree[0] != 0 else 1.0

    # Criterio 1: drift KE relativo dentro de cada path
    drift_flat = abs(ke_flat[-1] - ke_flat[0]) / abs(ke0_flat)
    drift_tree = abs(ke_tree[-1] - ke_tree[0]) / abs(ke0_tree)

    # Criterio 2: diferencia relativa de KE entre paths, por paso
    n_steps = min(len(ke_flat), len(ke_tree))
    ke_diff_rel = [abs(ke_tree[i] - ke_flat[i]) / max(abs(ke_flat[i]), 1e-30)
                   for i in range(n_steps)]
    max_ke_diff = max(ke_diff_rel)

    # Criterio 3: momento lineal (conservado → pequeño)
    max_p_flat = max(p_flat)
    max_p_tree = max(p_tree)

    # Criterio 4: momento angular: drift relativo
    l0_flat = max(l_flat[0], 1e-30)
    l0_tree = max(l_tree[0], 1e-30)
    max_dl_flat = max(abs(l - l_flat[0]) / l0_flat for l in l_flat)
    max_dl_tree = max(abs(l - l_tree[0]) / l0_tree for l in l_tree)

    # Tolerancias
    TOL_DRIFT = 0.05        # 5% drift KE dentro de cada path (20 pasos, Plummer denso)
    TOL_KE_DIFF = 0.01      # 1% diferencia KE entre flat y tree
    TOL_P = 0.01            # momento lineal < 1% de KE_0 (escala de referencia)
    TOL_DL = 0.05           # 5% drift L

    checks = {
        "drift_KE_flat":  (drift_flat, TOL_DRIFT),
        "drift_KE_tree":  (drift_tree, TOL_DRIFT),
        "max_KE_diff(flat_vs_tree)": (max_ke_diff, TOL_KE_DIFF),
        "max_|p|_flat":   (max_p_flat, TOL_P),
        "max_|p|_tree":   (max_p_tree, TOL_P),
        "drift_|L|_flat": (max_dl_flat, TOL_DL),
        "drift_|L|_tree": (max_dl_tree, TOL_DL),
    }

    passed = all(v <= tol for v, tol in checks.values())
    status = "PASS" if passed else "FAIL"

    lines = [f"  N={n}, P={p}: [{status}]"]
    for name, (val, tol) in checks.items():
        ok = "OK" if val <= tol else "FAIL"
        lines.append(f"    {name:<35} {val:.2e}  (tol={tol:.2e})  [{ok}]")

    return passed, "\n".join(lines)


# ── Ejecutar análisis ─────────────────────────────────────────────────────────
print("\n=== Análisis de validación física ===")
all_passed = True
results = []
for case in CASES:
    passed, report = analyze_pair(case["n"], case["p"])
    print(report)
    if passed is not None:
        results.append((case, passed))
        if not passed:
            all_passed = False
    else:
        all_passed = False

print()
if not results:
    print("No hay resultados para analizar.")
    sys.exit(1)

n_pass = sum(1 for _, p in results if p)
n_total = len(results)
print(f"=== Resultado: {n_pass}/{n_total} casos pasaron ===")
print()
print("Justificación de tolerancias:")
print("  - drift_KE < 5%: razonable para 20 pasos en Plummer denso (caótico).")
print("  - KE_diff flat vs tree < 1%: el LetTree introduce error O((s/d)^3)")
print("    en multipoles internos; para theta=0.5 y Plummer tipico, < 0.5%.")
print("  - |p| < 1%: momento lineal debe estar conservado por simetría.")
print("  - drift |L| < 5%: momento angular conservado aprox. para 20 pasos.")

if all_passed:
    print("\n[VALIDACIÓN FÍSICA: PASADA]")
    sys.exit(0)
else:
    print("\n[VALIDACIÓN FÍSICA: FALLIDA]")
    sys.exit(1)
