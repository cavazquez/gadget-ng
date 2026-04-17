#!/usr/bin/env python3
"""analyze_parity.py — Cuantifica la diferencia entre ejecuciones serial vs MPI.

Compara el snapshot final de cada run y calcula:
    max |Δr|  — desvío máximo en posición (cualquier partícula)
    max |Δv|  — desvío máximo en velocidad
    |ΔE/E|   — diferencia relativa de energía total
    |ΔL/L|   — diferencia relativa de momento angular

Si mpirun no está disponible, compara serial_run1 vs serial_run2
para verificar reproducibilidad determinística.

Genera:
    results/parity.csv   — desvíos por partícula (referencia = serial)
    results/parity_summary.csv — tabla resumen por modo
"""

import json
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

SCRIPT_DIR  = Path(__file__).parent
EXP_DIR     = SCRIPT_DIR.parent
RUNS_DIR    = EXP_DIR / "runs"
RESULTS_DIR = EXP_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

G    = 1.0
EPS  = 0.05
EPS2 = EPS * EPS


class ParticleState(NamedTuple):
    gid: int
    mass: float
    x: float; y: float; z: float
    vx: float; vy: float; vz: float


def load_snapshot_final(run_dir: Path) -> list[ParticleState]:
    """Carga el snapshot final de un run (snapshot_final/ o frames/snap_XXXXXX/)."""
    # Intentar snapshot_final primero.
    snap_dir = run_dir / "snapshot_final"
    if not snap_dir.exists():
        # Buscar el último frame.
        frames_dir = run_dir / "frames"
        if frames_dir.exists():
            sorted_frames = sorted(frames_dir.iterdir())
            if sorted_frames:
                snap_dir = sorted_frames[-1]

    if not snap_dir.exists():
        return []

    parts_path = snap_dir / "particles.jsonl"
    if not parts_path.exists():
        return []

    particles = []
    with open(parts_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            gid  = rec["global_id"]
            mass = float(rec["mass"])
            if "px" in rec:
                x, y, z = float(rec["px"]), float(rec["py"]), float(rec["pz"])
                vx, vy, vz = float(rec["vx"]), float(rec["vy"]), float(rec["vz"])
            else:
                pos = rec["position"]
                vel = rec["velocity"]
                x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
                vx, vy, vz = float(vel[0]), float(vel[1]), float(vel[2])
            particles.append(ParticleState(gid=gid, mass=mass,
                                           x=x, y=y, z=z, vx=vx, vy=vy, vz=vz))

    return sorted(particles, key=lambda p: p.gid)


def total_energy(particles: list[ParticleState]) -> float:
    ke = sum(0.5 * p.mass * (p.vx**2 + p.vy**2 + p.vz**2) for p in particles)
    pe = 0.0
    n = len(particles)
    for i in range(n):
        for j in range(i + 1, n):
            dx = particles[i].x - particles[j].x
            dy = particles[i].y - particles[j].y
            dz = particles[i].z - particles[j].z
            r2 = dx*dx + dy*dy + dz*dz + EPS2
            pe -= G * particles[i].mass * particles[j].mass / r2**0.5
    return ke + pe


def total_momentum(particles: list[ParticleState]):
    px = sum(p.mass * p.vx for p in particles)
    py = sum(p.mass * p.vy for p in particles)
    pz = sum(p.mass * p.vz for p in particles)
    return px, py, pz


def angular_momentum_z(particles: list[ParticleState]) -> float:
    return sum(p.mass * (p.x * p.vy - p.y * p.vx) for p in particles)


def compare_runs(ref: list[ParticleState], alt: list[ParticleState]) -> dict:
    if len(ref) != len(alt):
        return {"error": f"N mismatch: ref={len(ref)}, alt={len(alt)}"}

    dr = []
    dv = []
    for r, a in zip(ref, alt):
        dr_i = ((r.x-a.x)**2 + (r.y-a.y)**2 + (r.z-a.z)**2)**0.5
        dv_i = ((r.vx-a.vx)**2 + (r.vy-a.vy)**2 + (r.vz-a.vz)**2)**0.5
        dr.append(dr_i)
        dv.append(dv_i)

    E_ref = total_energy(ref)
    E_alt = total_energy(alt)
    dE_rel = abs(E_ref - E_alt) / abs(E_ref) if abs(E_ref) > 1e-15 else 0.0

    Lz_ref = angular_momentum_z(ref)
    Lz_alt = angular_momentum_z(alt)
    dL_rel = abs(Lz_ref - Lz_alt) / abs(Lz_ref) if abs(Lz_ref) > 1e-15 else 0.0

    px_r, py_r, pz_r = total_momentum(ref)
    px_a, py_a, pz_a = total_momentum(alt)
    dp_rel = ((px_r-px_a)**2+(py_r-py_a)**2+(pz_r-pz_a)**2)**0.5

    return {
        "N": len(ref),
        "max_dr": max(dr),
        "mean_dr": sum(dr)/len(dr),
        "max_dv": max(dv),
        "mean_dv": sum(dv)/len(dv),
        "E_ref": E_ref,
        "E_alt": E_alt,
        "dE_rel": dE_rel,
        "Lz_ref": Lz_ref,
        "Lz_alt": Lz_alt,
        "dLz_rel": dL_rel,
        "dp_abs": dp_rel,
        "per_particle_dr": dr,
        "per_particle_dv": dv,
    }


def main():
    # Determinar qué runs existen.
    available = {d.name: d for d in RUNS_DIR.iterdir() if d.is_dir()} if RUNS_DIR.exists() else {}
    if not available:
        print(f"ERROR: No hay runs en {RUNS_DIR}. Ejecuta run_parity.sh primero.")
        sys.exit(1)

    print(f"Runs encontrados: {sorted(available.keys())}")

    # Cargar referencia serial.
    ref_name = "serial"
    if ref_name not in available:
        print(f"ERROR: Run serial no encontrado en {RUNS_DIR}")
        sys.exit(1)

    ref_particles = load_snapshot_final(available[ref_name])
    if not ref_particles:
        print(f"ERROR: No se pudo cargar snapshot de {ref_name}")
        sys.exit(1)

    print(f"\nReferencia: {ref_name} (N={len(ref_particles)})")

    E_ref = total_energy(ref_particles)
    Lz_ref = angular_momentum_z(ref_particles)
    print(f"  E_total = {E_ref:.6f}")
    print(f"  Lz      = {Lz_ref:.6f}")

    # Comparar contra todos los demás runs.
    summary_rows = []
    per_particle_rows = []
    compare_targets = sorted(k for k in available if k != ref_name)

    for target_name in compare_targets:
        alt_particles = load_snapshot_final(available[target_name])
        if not alt_particles:
            print(f"\nWARN: Sin datos en {target_name}, omitiendo.")
            continue

        result = compare_runs(ref_particles, alt_particles)
        if "error" in result:
            print(f"\nWARN {target_name}: {result['error']}")
            continue

        print(f"\n[serial vs {target_name}]")
        print(f"  max |Δr|:    {result['max_dr']:.3e}")
        print(f"  mean |Δr|:   {result['mean_dr']:.3e}")
        print(f"  max |Δv|:    {result['max_dv']:.3e}")
        print(f"  |ΔE/E|:      {result['dE_rel']:.3e} ({result['dE_rel']*100:.4f}%)")
        print(f"  |ΔLz/Lz|:   {result['dLz_rel']:.3e}")

        summary_rows.append({
            "reference":    ref_name,
            "compared_to":  target_name,
            "N":            result["N"],
            "max_dr":       result["max_dr"],
            "mean_dr":      result["mean_dr"],
            "max_dv":       result["max_dv"],
            "mean_dv":      result["mean_dv"],
            "E_ref":        result["E_ref"],
            "E_alt":        result["E_alt"],
            "dE_rel":       result["dE_rel"],
            "dLz_rel":      result["dLz_rel"],
            "dp_abs":       result["dp_abs"],
        })

        for i, (dr, dv) in enumerate(zip(result["per_particle_dr"], result["per_particle_dv"])):
            per_particle_rows.append({
                "compared_to": target_name,
                "gid": ref_particles[i].gid,
                "dr": dr,
                "dv": dv,
            })

    if not summary_rows:
        print("No se generó ningún resultado comparativo.")
        sys.exit(0)

    pd.DataFrame(summary_rows).to_csv(RESULTS_DIR / "parity_summary.csv", index=False)
    print(f"\nResumen guardado en {RESULTS_DIR}/parity_summary.csv")

    pd.DataFrame(per_particle_rows).to_csv(RESULTS_DIR / "parity.csv", index=False)
    print(f"Datos por partícula guardados en {RESULTS_DIR}/parity.csv")

    print("\n=== Tabla resumen ===")
    print(pd.DataFrame(summary_rows)[
        ["compared_to", "max_dr", "max_dv", "dE_rel", "dLz_rel"]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
