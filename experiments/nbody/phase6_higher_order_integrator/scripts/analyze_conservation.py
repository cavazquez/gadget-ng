#!/usr/bin/env python3
"""analyze_conservation.py — Análisis de conservación multi-step de los 12 runs Fase 6.

Para cada run bajo `runs/<tag>/`:
    1) Lee `diagnostics.jsonl` → serie por paso (KE, p, L, COM).
    2) Lee los `frames/snap_XXXXXX/` → serie por snapshot con PE (via
       `snapshot_metrics.potential_energy`), virial, half-mass radius.
    3) Combina ambas para obtener series |ΔE/E|, |Δp|, |ΔL|.
    4) Lee `timings.json` → coste total y por paso.

Salidas:
    results/timeseries/<tag>.csv
    results/phase6_summary.csv

Uso:
    python3 analyze_conservation.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP_DIR = Path(__file__).resolve().parents[1]
RUNS_DIR = EXP_DIR / "runs"
RESULTS_DIR = EXP_DIR / "results"
TIMESERIES_DIR = RESULTS_DIR / "timeseries"

sys.path.insert(0, str(EXP_DIR.parents[2] / "scripts" / "analysis"))
from snapshot_metrics import (  # noqa: E402
    load_snapshot_dir,
    kinetic_energy,
    potential_energy,
    half_mass_radius,
    center_of_mass,
    virial_ratio,
)

SOFTENING = 0.05
G = 1.0


def parse_tag(tag: str) -> dict:
    """tag = '<dist>_N<N>_<integrator>' → dict con componentes."""
    for i, ch in enumerate(tag):
        if ch == "N" and i + 1 < len(tag) and tag[i - 1] == "_" and tag[i + 1].isdigit():
            break
    else:
        return {"dist": tag, "N": 0, "integrator": ""}
    dist = tag[: i - 1]
    rest = tag[i + 1:]
    n_end = 0
    while n_end < len(rest) and rest[n_end].isdigit():
        n_end += 1
    n_val = int(rest[:n_end])
    integrator = rest[n_end + 1:] if n_end < len(rest) else ""
    return {"dist": dist, "N": n_val, "integrator": integrator}


def load_diagnostics(path: Path) -> pd.DataFrame:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    for col, idx in [("px", ("momentum", 0)), ("py", ("momentum", 1)),
                     ("pz", ("momentum", 2)),
                     ("Lx", ("angular_momentum", 0)),
                     ("Ly", ("angular_momentum", 1)),
                     ("Lz", ("angular_momentum", 2)),
                     ("com_x", ("com", 0)), ("com_y", ("com", 1)),
                     ("com_z", ("com", 2))]:
        src, k = idx
        df[col] = df[src].apply(lambda v: v[k] if isinstance(v, list) else float("nan"))
    df["p_norm"] = np.sqrt(df["px"] ** 2 + df["py"] ** 2 + df["pz"] ** 2)
    df["L_norm"] = np.sqrt(df["Lx"] ** 2 + df["Ly"] ** 2 + df["Lz"] ** 2)
    return df.sort_values("step").reset_index(drop=True)


def analyze_run(run_dir: Path) -> tuple[pd.DataFrame, dict]:
    tag = run_dir.name
    meta = parse_tag(tag)

    diag = load_diagnostics(run_dir / "diagnostics.jsonl")
    step0 = diag.iloc[0]
    p0 = np.array([step0["px"], step0["py"], step0["pz"]])
    l0 = np.array([step0["Lx"], step0["Ly"], step0["Lz"]])
    dp_vec = diag[["px", "py", "pz"]].values - p0[None, :]
    dl_vec = diag[["Lx", "Ly", "Lz"]].values - l0[None, :]
    diag["dp_abs"] = np.linalg.norm(dp_vec, axis=1)
    diag["dL_abs"] = np.linalg.norm(dl_vec, axis=1)
    p_norms = np.maximum(np.linalg.norm(p0), diag["p_norm"].values)
    l_norms = np.maximum(np.linalg.norm(l0), diag["L_norm"].values)
    diag["dp_rel"] = np.where(p_norms > 1e-300, diag["dp_abs"] / p_norms, diag["dp_abs"])
    diag["dL_rel"] = np.where(l_norms > 1e-300, diag["dL_abs"] / l_norms, diag["dL_abs"])

    frames_dir = run_dir / "frames"
    frame_dirs = sorted([d for d in frames_dir.iterdir() if d.is_dir()])
    frame_rows = []
    for fd in frame_dirs:
        particles, t = load_snapshot_dir(fd)
        ke = kinetic_energy(particles)
        pe = potential_energy(particles, softening=SOFTENING, G=G)
        e = ke + pe
        r_hm = half_mass_radius(particles)
        com = center_of_mass(particles)
        q = virial_ratio(ke, pe)
        step = int(fd.name.split("_")[-1])
        frame_rows.append({
            "step": step, "t": t,
            "KE": ke, "PE": pe, "E": e,
            "r_hm": r_hm,
            "Q_virial": q,
            "com_x": com[0], "com_y": com[1], "com_z": com[2],
        })
    frames = pd.DataFrame(frame_rows).sort_values("step").reset_index(drop=True)
    e0 = float(frames["E"].iloc[0])
    frames["dE_rel"] = (frames["E"] - e0).abs() / (abs(e0) if abs(e0) > 1e-300 else 1.0)

    merged = pd.merge(
        diag[["step", "kinetic_energy", "p_norm", "L_norm",
              "dp_abs", "dL_abs", "dp_rel", "dL_rel"]],
        frames[["step", "t", "PE", "E", "dE_rel", "Q_virial", "r_hm"]],
        on="step", how="left",
    )

    timings = {}
    timings_path = run_dir / "timings.json"
    if timings_path.exists():
        with open(timings_path) as f:
            timings = json.load(f)

    summary = {
        "tag": tag,
        "distribution": meta["dist"],
        "N": meta["N"],
        "integrator": meta["integrator"],
        "total_wall_s": timings.get("total_wall_s", float("nan")),
        "total_gravity_s": timings.get("total_gravity_s", float("nan")),
        "mean_step_wall_s": timings.get("mean_step_wall_s", float("nan")),
        "E0": e0,
        "E_final": float(frames["E"].iloc[-1]),
        "dE_rel_final": float(frames["dE_rel"].iloc[-1]),
        "dE_rel_max": float(frames["dE_rel"].max()),
        "dE_rel_mean": float(frames["dE_rel"].mean()),
        "dp_rel_final": float(diag["dp_rel"].iloc[-1]),
        "dp_rel_max": float(diag["dp_rel"].max()),
        "dL_rel_final": float(diag["dL_rel"].iloc[-1]),
        "dL_rel_max": float(diag["dL_rel"].max()),
        "Q_virial_final": float(frames["Q_virial"].iloc[-1]),
        "r_hm_final": float(frames["r_hm"].iloc[-1]),
        "r_hm_init": float(frames["r_hm"].iloc[0]),
    }
    return merged, summary


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    TIMESERIES_DIR.mkdir(parents=True, exist_ok=True)
    run_dirs = sorted(
        d for d in RUNS_DIR.iterdir()
        if d.is_dir() and (d / "diagnostics.jsonl").exists()
           and (d / "frames").exists()
    )
    print(f"Analizando {len(run_dirs)} runs...")
    summaries = []
    for i, rd in enumerate(run_dirs, 1):
        print(f"  [{i:2}/{len(run_dirs)}] {rd.name}", end=" ... ", flush=True)
        try:
            ts, summary = analyze_run(rd)
            ts.to_csv(TIMESERIES_DIR / f"{rd.name}.csv", index=False)
            summaries.append(summary)
            print(f"OK  dE_rel_final={summary['dE_rel_final']:.3e}  "
                  f"wall={summary['total_wall_s']:.1f}s")
        except Exception as exc:
            print(f"FAIL: {exc}")
    df = pd.DataFrame(summaries).sort_values(["N", "distribution", "integrator"])
    out = RESULTS_DIR / "phase6_summary.csv"
    df.to_csv(out, index=False)
    print(f"\nResumen global → {out}")
    print(f"Series por run en {TIMESERIES_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
