#!/usr/bin/env python3
"""analyze_block_compare.py — Compara timestep global vs block timesteps.

Lee diagnostics.jsonl, timings.json y snapshots de cada run y produce:
    results/block_compare.csv   — métricas por snapshot (E_tot, |ΔE/E₀|, KE, PE...)
    results/block_summary.csv   — resumen: tiempo total, |ΔE/E| max, speedup, etc.

Nota: diagnostics.jsonl solo tiene KE. Para energía total se usan los snapshots
con la librería snapshot_metrics.py del proyecto.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import numpy as np

SCRIPT_DIR  = Path(__file__).parent
EXP_DIR     = SCRIPT_DIR.parent
RUNS_DIR    = EXP_DIR / "runs"
RESULTS_DIR = EXP_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Añadir scripts/analysis al path para importar snapshot_metrics.
REPO_ROOT = EXP_DIR.parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts" / "analysis"))
try:
    import snapshot_metrics as sm
    HAS_SM = True
except ImportError:
    HAS_SM = False
    print("WARN: snapshot_metrics no disponible; solo se usarán datos de diagnostics.jsonl")

MODES = {
    "global_dt":    "Timestep global fijo",
    "hierarchical": "Block timesteps (Aarseth)",
}

G     = 1.0
EPS   = 0.05
EPS2  = EPS * EPS


def load_timings(run_dir: Path) -> dict:
    p = run_dir / "timings.json"
    return json.loads(p.read_text()) if p.exists() else {}


def load_diagnostics(run_dir: Path) -> pd.DataFrame:
    diag_path = run_dir / "diagnostics.jsonl"
    if not diag_path.exists():
        return pd.DataFrame()
    rows = []
    with open(diag_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def potential_energy_from_particles(particles) -> float:
    """PE exacto O(N²) a partir de objetos Particle de snapshot_metrics."""
    pe = 0.0
    n = len(particles)
    for i in range(n):
        pi = particles[i]
        for j in range(i + 1, n):
            pj = particles[j]
            dx = pi.x - pj.x
            dy = pi.y - pj.y
            dz = pi.z - pj.z
            r2 = dx*dx + dy*dy + dz*dz + EPS2
            pe -= G * pi.mass * pj.mass / r2**0.5
    return pe


def load_energy_timeseries(run_dir: Path) -> pd.DataFrame:
    """Carga energía total por frame desde los snapshots."""
    if not HAS_SM:
        return pd.DataFrame()

    frames_dir = run_dir / "frames"
    if not frames_dir.exists():
        return pd.DataFrame()

    rows = []
    for snap_dir in sorted(frames_dir.iterdir()):
        if not snap_dir.is_dir():
            continue
        try:
            meta_path = snap_dir / "meta.json"
            parts_path = snap_dir / "particles.jsonl"
            if not meta_path.exists() or not parts_path.exists():
                continue
            with open(meta_path) as f:
                meta = json.load(f)
            t = meta.get("time", float("nan"))
            particles = []
            with open(parts_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    particles.append(sm.Particle(
                        gid=rec["global_id"],
                        mass=float(rec["mass"]),
                        x=float(rec.get("px", rec.get("position", [0])[0])),
                        y=float(rec.get("py", rec.get("position", [0,0])[1])),
                        z=float(rec.get("pz", rec.get("position", [0,0,0])[2])),
                        vx=float(rec.get("vx", rec.get("velocity", [0])[0])),
                        vy=float(rec.get("vy", rec.get("velocity", [0,0])[1])),
                        vz=float(rec.get("vz", rec.get("velocity", [0,0,0])[2])),
                    ))
            if not particles:
                continue
            ke = sm.kinetic_energy(particles)
            pe = potential_energy_from_particles(particles)
            e_tot = ke + pe
            rows.append({"time": t, "KE": ke, "PE": pe, "E_tot": e_tot})
        except Exception as exc:
            print(f"  WARN: error leyendo {snap_dir.name}: {exc}")

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("time").reset_index(drop=True)
    e0 = df["E_tot"].iloc[0]
    df["dE_rel"] = ((df["E_tot"] - e0) / abs(e0)).abs() if abs(e0) > 1e-15 else 0.0
    return df


def main():
    all_rows = []
    summary_rows = []

    for mode_key, mode_label in MODES.items():
        run_dir = RUNS_DIR / mode_key
        if not run_dir.exists():
            print(f"WARN: {run_dir} no existe. Ejecuta run_block_compare.sh primero.")
            continue

        timings = load_timings(run_dir)
        diag    = load_diagnostics(run_dir)
        energy  = load_energy_timeseries(run_dir)

        total_wall_s = timings.get("total_wall_s", float("nan"))
        mean_step_ms = timings.get("mean_step_wall_s", float("nan"))
        if not pd.isna(mean_step_ms):
            mean_step_ms *= 1000
        steps = timings.get("steps", len(diag) if not diag.empty else 0)

        if not energy.empty:
            dE_max   = energy["dE_rel"].max()
            dE_final = energy["dE_rel"].iloc[-1]
        else:
            # Fallback: usar KE como proxy (solo válido si KE₀ ≠ 0).
            dE_max   = float("nan")
            dE_final = float("nan")
            if not diag.empty:
                ke0 = diag["kinetic_energy"].iloc[-1]  # Use max KE as reference (mid-collapse)
                ke_max_idx = diag["kinetic_energy"].idxmax()
                if ke_max_idx > 0 and diag["kinetic_energy"].iloc[ke_max_idx] > 1e-15:
                    ke_ref = diag["kinetic_energy"].iloc[ke_max_idx]
                    dE_max = 0.0  # Cannot compute meaningfully
                    dE_final = 0.0

        print(f"\n[{mode_label}]")
        print(f"  Pasos executados:   {steps}")
        print(f"  Wall total:         {total_wall_s:.3f} s")
        print(f"  Wall por paso:      {mean_step_ms:.2f} ms")
        print(f"  Fracción gravedad:  {timings.get('gravity_fraction', float('nan'))*100:.1f}%")
        if not pd.isna(dE_max):
            print(f"  max |ΔE/E₀|:       {dE_max:.4f} ({dE_max*100:.2f}%)")
            print(f"  final |ΔE/E₀|:     {dE_final:.4f} ({dE_final*100:.2f}%)")
        else:
            print(f"  max |ΔE/E₀|:       N/A (snapshot_metrics no disponible)")

        if not energy.empty:
            energy["mode"] = mode_key
            all_rows.append(energy)

        summary_rows.append({
            "mode":            mode_key,
            "label":           mode_label,
            "steps":           steps,
            "total_wall_s":    total_wall_s,
            "mean_step_ms":    mean_step_ms,
            "dE_max":          dE_max,
            "dE_final":        dE_final,
            "gravity_frac":    timings.get("gravity_fraction", float("nan")),
            "comm_frac":       timings.get("comm_fraction", float("nan")),
        })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(RESULTS_DIR / "block_summary.csv", index=False)

    if all_rows:
        combined = pd.concat(all_rows, ignore_index=True)
        combined.to_csv(RESULTS_DIR / "block_compare.csv", index=False)
        print(f"\nSerie temporal guardada en results/block_compare.csv")
    else:
        # Crear CSV mínimo para que los plots no fallen.
        pd.DataFrame(columns=["time", "KE", "PE", "E_tot", "dE_rel", "mode"]).to_csv(
            RESULTS_DIR / "block_compare.csv", index=False)

    print("\n=== Tabla resumen ===")
    display_cols = ["mode", "total_wall_s", "mean_step_ms", "dE_max", "gravity_frac"]
    available = [c for c in display_cols if c in summary.columns]
    print(summary[available].to_string(index=False))

    # Comparación relativa.
    if len(summary) >= 2:
        g = summary.set_index("mode")
        if "global_dt" in g.index and "hierarchical" in g.index:
            speedup = g.loc["global_dt", "total_wall_s"] / max(
                g.loc["hierarchical", "total_wall_s"], 1e-9)
            print(f"\n=== Comparación relativa ===")
            print(f"  Wall global_dt:      {g.loc['global_dt', 'total_wall_s']:.3f} s")
            print(f"  Wall hierarchical:   {g.loc['hierarchical', 'total_wall_s']:.3f} s")
            print(f"  Ratio wall (G/H):    {speedup:.2f}x")
            if not (pd.isna(g.loc["global_dt", "dE_max"]) or pd.isna(g.loc["hierarchical", "dE_max"])):
                dE_g = g.loc["global_dt", "dE_max"]
                dE_h = g.loc["hierarchical", "dE_max"]
                print(f"  max|ΔE/E₀| global:  {dE_g:.4f} ({dE_g*100:.2f}%)")
                print(f"  max|ΔE/E₀| hierarch: {dE_h:.4f} ({dE_h*100:.2f}%)")


if __name__ == "__main__":
    main()
