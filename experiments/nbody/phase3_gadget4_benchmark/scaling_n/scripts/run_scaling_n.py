#!/usr/bin/env python3
"""run_scaling_n.py — Mide tiempo de cómputo vs N para DirectGravity y Barnes-Hut.

Para cada N en la lista, genera un config TOML mínimo (Plummer, 10 pasos),
ejecuta el CLI en modo release, lee timings.json y guarda resultados en CSV.

Uso:
    cd <repo_root>
    python3 experiments/nbody/phase3_gadget4_benchmark/scaling_n/scripts/run_scaling_n.py
    [--solver direct|bh|both]  [--N 100,500,1000,5000]  [--steps 10]

Genera:
    experiments/nbody/phase3_gadget4_benchmark/scaling_n/results/scaling_n.csv
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT  = SCRIPT_DIR.parent.parent.parent.parent.parent  # 5 levels up to repo root
EXP_DIR    = SCRIPT_DIR.parent
RESULTS_DIR = EXP_DIR / "results"
CONFIGS_DIR = EXP_DIR / "configs"
CLI_BIN     = REPO_ROOT / "target" / "release" / "gadget-ng"

RESULTS_DIR.mkdir(exist_ok=True)
CONFIGS_DIR.mkdir(exist_ok=True)

DEFAULT_N_LIST = [100, 500, 1000, 2000, 5000]
DEFAULT_STEPS  = 10
DEFAULT_THETA  = 0.5

# ── Generación de configs TOML ────────────────────────────────────────────────

TOML_TEMPLATE = """\
[simulation]
particle_count = {n}
dt             = 0.01
num_steps      = {steps}
softening      = 0.05
gravitational_constant = 1.0
box_size       = 20.0
seed           = 42

[initial_conditions]
kind = {{ plummer = {{ a = 1.0 }} }}

[gravity]
solver = "{solver}"
theta  = {theta}

[output]
snapshot_interval   = 0
checkpoint_interval = 0
"""

def make_config(n: int, steps: int, solver: str, theta: float, out_path: Path) -> None:
    content = TOML_TEMPLATE.format(n=n, steps=steps, solver=solver, theta=theta)
    out_path.write_text(content)


# ── Ejecución del CLI ─────────────────────────────────────────────────────────

def run_cli(config_path: Path, out_dir: Path) -> dict | None:
    """Ejecuta el CLI y devuelve el contenido de timings.json, o None si falla."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(CLI_BIN), "stepping",
        "--config", str(config_path),
        "--out", str(out_dir),
    ]
    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    wall_s = time.perf_counter() - t0

    if result.returncode != 0:
        print(f"  ERROR al ejecutar CLI:\n{result.stderr[:500]}", file=sys.stderr)
        return None

    timings_path = out_dir / "timings.json"
    if timings_path.exists():
        with open(timings_path) as f:
            data = json.load(f)
        data["wall_external_s"] = wall_s
        return data
    else:
        # Fallback: solo tiempo de pared externo.
        return {"total_wall_s": wall_s, "wall_external_s": wall_s,
                "mean_step_wall_s": wall_s / 10, "mean_gravity_s": 0.0, "mean_comm_s": 0.0}


# ── Loop principal ────────────────────────────────────────────────────────────

def run_all(n_list: list[int], steps: int, solvers: list[str]) -> list[dict]:
    # Verificar que el binario existe.
    if not CLI_BIN.exists():
        print(f"[scaling_n] Compilando gadget-ng-cli en release...")
        subprocess.run(
            ["cargo", "build", "-p", "gadget-ng-cli", "--release"],
            cwd=REPO_ROOT, check=True
        )

    results = []
    total = len(n_list) * len(solvers)
    done = 0

    for solver in solvers:
        for n in n_list:
            done += 1
            solver_label = "barnes_hut" if solver == "bh" else "direct"
            theta = DEFAULT_THETA if solver_label == "barnes_hut" else 0.5
            config_path = CONFIGS_DIR / f"{solver_label}_N{n}.toml"
            out_dir = EXP_DIR / "runs" / f"{solver_label}_N{n}"

            make_config(n, steps, solver_label, theta, config_path)

            print(f"[{done}/{total}] solver={solver_label} N={n} steps={steps} ...", end=" ", flush=True)
            t0 = time.perf_counter()
            data = run_cli(config_path, out_dir)
            elapsed = time.perf_counter() - t0

            if data is None:
                print("FAIL")
                continue

            row = {
                "solver": solver_label,
                "N": n,
                "steps": steps,
                "total_wall_s": data.get("total_wall_s", elapsed),
                "mean_step_wall_s": data.get("mean_step_wall_s", elapsed / steps),
                "mean_gravity_s": data.get("mean_gravity_s", 0.0),
                "mean_comm_s": data.get("mean_comm_s", 0.0),
                "gravity_fraction": data.get("gravity_fraction", 0.0),
                "comm_fraction": data.get("comm_fraction", 0.0),
            }
            results.append(row)
            print(f"wall={elapsed:.2f}s  mean_step={row['mean_step_wall_s']*1e3:.1f}ms")

    return results


def save_csv(results: list[dict], csv_path: Path) -> None:
    if not results:
        print("[scaling_n] No hay resultados para guardar.")
        return
    import csv
    fieldnames = list(results[0].keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)
    print(f"\n[scaling_n] CSV guardado en {csv_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", choices=["direct", "bh", "both"], default="both")
    parser.add_argument("--N", default=",".join(map(str, DEFAULT_N_LIST)),
                        help="Valores de N separados por coma")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    args = parser.parse_args()

    n_list = [int(x) for x in args.N.split(",")]
    solvers = ["direct", "bh"] if args.solver == "both" else [args.solver]

    print(f"[scaling_n] Barrido N={n_list}, solvers={solvers}, steps={args.steps}")
    results = run_all(n_list, args.steps, solvers)

    csv_path = RESULTS_DIR / "scaling_n.csv"
    save_csv(results, csv_path)

    print("\n=== Resumen ===")
    print(f"{'Solver':<12} {'N':>6} {'mean_step_ms':>14} {'gravity_frac':>14}")
    for r in results:
        print(f"{r['solver']:<12} {r['N']:>6} {r['mean_step_wall_s']*1e3:>14.2f} {r['gravity_fraction']:>14.3f}")

    print("\nPara generar plots:")
    print("  python3 experiments/nbody/phase3_gadget4_benchmark/scaling_n/scripts/plot_scaling_n.py")


if __name__ == "__main__":
    main()
