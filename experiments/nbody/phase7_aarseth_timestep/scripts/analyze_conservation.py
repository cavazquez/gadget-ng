#!/usr/bin/env python3
"""analyze_conservation.py — Análisis de conservación de los 54 runs de Fase 7.

Para cada run bajo `runs/<tag>/`:
    1) Lee `diagnostics.jsonl` → serie por paso (KE, p, L, COM, + campos jerárquicos).
    2) Lee los `frames/snap_XXXXXX/` → serie por snapshot con PE, virial, r_hm.
    3) Combina ambas para obtener series |ΔE/E|, |Δp|, |ΔL|.
    4) Lee `timings.json` → coste total y por paso.

Salidas:
    results/timeseries/<tag>.csv   — serie completa por paso
    results/phase7_summary.csv     — una fila por run con todas las métricas clave

Uso:
    python3 analyze_conservation.py
    python3 analyze_conservation.py plummer_a1   # filtra por prefijo
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP_DIR = Path(__file__).resolve().parents[1]
RUNS_DIR = EXP_DIR / "runs"
RESULTS_DIR = EXP_DIR / "results"
TIMESERIES_DIR = RESULTS_DIR / "timeseries"

# Ruta al módulo de análisis de snapshots (reutilizado de fases anteriores)
_SCRIPT_ANALYSIS = EXP_DIR.parents[2] / "scripts" / "analysis"
if _SCRIPT_ANALYSIS.exists():
    sys.path.insert(0, str(_SCRIPT_ANALYSIS))
    from snapshot_metrics import (  # noqa: E402
        load_snapshot_dir,
        kinetic_energy,
        potential_energy,
        half_mass_radius,
        center_of_mass,
        virial_ratio,
    )
    _HAS_SNAPSHOT_METRICS = True
else:
    _HAS_SNAPSHOT_METRICS = False

SOFTENING = 0.05
G = 1.0

# ── Parseo de tag ─────────────────────────────────────────────────────────────

_TAG_RE = re.compile(
    r"^(?P<dist>[a-z0-9_]+?)_N(?P<n>\d+)_(?P<variant>.+)$"
)


def parse_tag(tag: str) -> dict:
    """Parsea tag Fase 7.

    Formato: <dist>_N<N>_<variant>
    Variante puede ser:
      fixed_dt025, fixed_dt0125, fixed_dt00625
      hier_acc_eta001, hier_acc_eta002, hier_acc_eta005
      hier_jerk_eta001, hier_jerk_eta002, hier_jerk_eta005
    """
    m = _TAG_RE.match(tag)
    if not m:
        return {"dist": tag, "N": 0, "variant": "", "adaptive": False,
                "criterion": "", "eta": float("nan"), "dt_fixed": float("nan")}

    dist = m.group("dist")
    n = int(m.group("n"))
    variant = m.group("variant")

    if variant.startswith("fixed_dt"):
        dt_str = variant[len("fixed_dt"):]
        # "025" → 0.025, "0125" → 0.0125, "00625" → 0.00625
        try:
            dt_val = float("0." + dt_str) if not dt_str.startswith("0.") else float(dt_str)
        except ValueError:
            dt_val = float("nan")
        return {
            "dist": dist, "N": n, "variant": variant,
            "adaptive": False, "criterion": "fixed", "eta": float("nan"),
            "dt_fixed": dt_val,
        }

    if variant.startswith("hier_"):
        parts = variant.split("_")
        # hier_acc_eta001 → parts = ["hier", "acc", "eta001"]
        # hier_jerk_eta001 → parts = ["hier", "jerk", "eta001"]
        criterion = parts[1] if len(parts) > 1 else ""
        eta_str = parts[2][3:] if len(parts) > 2 and parts[2].startswith("eta") else "0"
        # "001" → 0.01, "002" → 0.02, "005" → 0.05
        try:
            eta_val = int(eta_str) / 100.0
        except ValueError:
            eta_val = float("nan")
        return {
            "dist": dist, "N": n, "variant": variant,
            "adaptive": True, "criterion": criterion, "eta": eta_val,
            "dt_fixed": float("nan"),
        }

    return {"dist": dist, "N": n, "variant": variant,
            "adaptive": False, "criterion": "", "eta": float("nan"),
            "dt_fixed": float("nan")}


# ── Carga de diagnostics ──────────────────────────────────────────────────────

def load_diagnostics(path: Path) -> pd.DataFrame:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)

    # Columnas vectoriales estándar
    for col, (src, k) in [
        ("px", ("momentum", 0)), ("py", ("momentum", 1)), ("pz", ("momentum", 2)),
        ("Lx", ("angular_momentum", 0)), ("Ly", ("angular_momentum", 1)),
        ("Lz", ("angular_momentum", 2)),
        ("com_x", ("com", 0)), ("com_y", ("com", 1)), ("com_z", ("com", 2)),
    ]:
        if src in df.columns:
            df[col] = df[src].apply(lambda v, k=k: v[k] if isinstance(v, list) else float("nan"))

    df["p_norm"] = np.sqrt(df.get("px", 0)**2 + df.get("py", 0)**2 + df.get("pz", 0)**2)
    df["L_norm"] = np.sqrt(df.get("Lx", 0)**2 + df.get("Ly", 0)**2 + df.get("Lz", 0)**2)

    # Columnas jerárquicas opcionales (solo presentes en runs con hierarchical=true)
    for col in ["active_total", "force_evals", "dt_min_effective", "dt_max_effective"]:
        if col not in df.columns:
            df[col] = float("nan")

    if "level_histogram" not in df.columns:
        df["level_histogram"] = None

    return df.sort_values("step").reset_index(drop=True)


# ── Análisis por run ──────────────────────────────────────────────────────────

def analyze_run(run_dir: Path) -> tuple[pd.DataFrame, dict]:
    tag = run_dir.name
    meta = parse_tag(tag)

    diag = load_diagnostics(run_dir / "diagnostics.jsonl")
    if diag.empty:
        raise ValueError(f"diagnostics.jsonl vacío en {run_dir}")

    step0 = diag.iloc[0]
    p0 = np.array([step0.get("px", 0), step0.get("py", 0), step0.get("pz", 0)])
    l0 = np.array([step0.get("Lx", 0), step0.get("Ly", 0), step0.get("Lz", 0)])
    dp_vec = diag[["px", "py", "pz"]].fillna(0).values - p0[None, :]
    dl_vec = diag[["Lx", "Ly", "Lz"]].fillna(0).values - l0[None, :]
    diag["dp_abs"] = np.linalg.norm(dp_vec, axis=1)
    diag["dL_abs"] = np.linalg.norm(dl_vec, axis=1)
    p_norms = np.maximum(np.linalg.norm(p0), diag["p_norm"].values)
    l_norms = np.maximum(np.linalg.norm(l0), diag["L_norm"].values)
    diag["dp_rel"] = np.where(p_norms > 1e-300, diag["dp_abs"] / p_norms, diag["dp_abs"])
    diag["dL_rel"] = np.where(l_norms > 1e-300, diag["dL_abs"] / l_norms, diag["dL_abs"])

    # Métricas jerárquicas agregadas
    hier_summary: dict = {}
    if meta["adaptive"] and "active_total" in diag.columns:
        valid = diag["active_total"].dropna()
        if not valid.empty:
            hier_summary["mean_active_per_step"] = float(valid.mean())
        valid_dtmin = diag["dt_min_effective"].dropna()
        if not valid_dtmin.empty:
            hier_summary["dt_min_obs"] = float(valid_dtmin.min())
            hier_summary["dt_min_mean"] = float(valid_dtmin.mean())
        valid_dtmax = diag["dt_max_effective"].dropna()
        if not valid_dtmax.empty:
            hier_summary["dt_max_obs"] = float(valid_dtmax.max())
            hier_summary["dt_max_mean"] = float(valid_dtmax.mean())
        # Distribución de niveles (último paso con histograma)
        hists = diag["level_histogram"].dropna()
        if not hists.empty:
            last_hist = hists.iloc[-1]
            if isinstance(last_hist, list):
                for lvl, cnt in enumerate(last_hist):
                    hier_summary[f"level_{lvl}_count"] = int(cnt)

    # ── Snapshots / energía total ────────────────────────────────────────────
    frames_data: list[dict] = []
    if _HAS_SNAPSHOT_METRICS:
        frames_dir = run_dir / "frames"
        if frames_dir.exists():
            frame_dirs = sorted([d for d in frames_dir.iterdir() if d.is_dir()])
            for fd in frame_dirs:
                try:
                    particles, t = load_snapshot_dir(fd)
                    ke = kinetic_energy(particles)
                    pe = potential_energy(particles, softening=SOFTENING, G=G)
                    e = ke + pe
                    r_hm = half_mass_radius(particles)
                    q = virial_ratio(ke, pe)
                    step = int(fd.name.split("_")[-1])
                    frames_data.append({"step": step, "t": t,
                                        "KE": ke, "PE": pe, "E": e,
                                        "r_hm": r_hm, "Q_virial": q})
                except Exception:
                    pass

    e0 = float("nan")
    dE_rel_final = float("nan")
    dE_rel_max = float("nan")
    Q_virial_final = float("nan")
    r_hm_final = float("nan")
    r_hm_init = float("nan")

    if frames_data:
        frames = pd.DataFrame(frames_data).sort_values("step").reset_index(drop=True)
        e0 = float(frames["E"].iloc[0])
        denom = abs(e0) if abs(e0) > 1e-300 else 1.0
        frames["dE_rel"] = (frames["E"] - e0).abs() / denom
        dE_rel_final = float(frames["dE_rel"].iloc[-1])
        dE_rel_max = float(frames["dE_rel"].max())
        Q_virial_final = float(frames["Q_virial"].iloc[-1])
        r_hm_final = float(frames["r_hm"].iloc[-1])
        r_hm_init = float(frames["r_hm"].iloc[0])

        # Merge con diag
        cols_diag = ["step", "kinetic_energy", "p_norm", "L_norm",
                     "dp_abs", "dL_abs", "dp_rel", "dL_rel",
                     "active_total", "force_evals",
                     "dt_min_effective", "dt_max_effective"]
        cols_diag = [c for c in cols_diag if c in diag.columns]
        merged = pd.merge(
            diag[cols_diag],
            frames[["step", "t", "PE", "E", "dE_rel", "Q_virial", "r_hm"]],
            on="step", how="left",
        )
    else:
        # Sin snapshots: solo usamos diagnostics para KE-only
        merged = diag.copy()

    # ── Timings ───────────────────────────────────────────────────────────────
    timings: dict = {}
    timings_path = run_dir / "timings.json"
    if timings_path.exists():
        with open(timings_path) as f:
            timings = json.load(f)

    summary: dict = {
        "tag": tag,
        "distribution": meta["dist"],
        "N": meta["N"],
        "variant": meta["variant"],
        "adaptive": meta["adaptive"],
        "criterion": meta["criterion"],
        "eta": meta["eta"],
        "dt_fixed": meta["dt_fixed"],
        # Tiempos
        "total_wall_s": timings.get("total_wall_s", float("nan")),
        "total_gravity_s": timings.get("total_gravity_s", float("nan")),
        "mean_step_wall_s": timings.get("mean_step_wall_s", float("nan")),
        # Física
        "E0": e0,
        "dE_rel_final": dE_rel_final,
        "dE_rel_max": dE_rel_max,
        "dp_rel_final": float(diag["dp_rel"].iloc[-1]) if "dp_rel" in diag.columns else float("nan"),
        "dp_rel_max": float(diag["dp_rel"].max()) if "dp_rel" in diag.columns else float("nan"),
        "dL_rel_final": float(diag["dL_rel"].iloc[-1]) if "dL_rel" in diag.columns else float("nan"),
        "dL_rel_max": float(diag["dL_rel"].max()) if "dL_rel" in diag.columns else float("nan"),
        "Q_virial_final": Q_virial_final,
        "r_hm_final": r_hm_final,
        "r_hm_init": r_hm_init,
        **hier_summary,
    }
    return merged, summary


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    TIMESERIES_DIR.mkdir(parents=True, exist_ok=True)

    filter_prefix = sys.argv[1] if len(sys.argv) > 1 else ""

    run_dirs = sorted(
        d for d in RUNS_DIR.iterdir()
        if d.is_dir()
        and (d / "diagnostics.jsonl").exists()
        and (not filter_prefix or filter_prefix in d.name)
    )

    if not run_dirs:
        print(f"No se encontraron runs en {RUNS_DIR}", file=sys.stderr)
        return 1

    print(f"Analizando {len(run_dirs)} runs...")
    summaries = []

    for i, rd in enumerate(run_dirs, 1):
        print(f"  [{i:2}/{len(run_dirs)}] {rd.name}", end=" ... ", flush=True)
        try:
            ts, summary = analyze_run(rd)
            ts.to_csv(TIMESERIES_DIR / f"{rd.name}.csv", index=False)
            summaries.append(summary)
            dE = summary.get("dE_rel_final", float("nan"))
            wall = summary.get("total_wall_s", float("nan"))
            print(f"OK  dE_rel_final={dE:.3e}  wall={wall:.1f}s")
        except Exception as exc:
            print(f"FAIL: {exc}")

    if summaries:
        df = pd.DataFrame(summaries).sort_values(
            ["distribution", "N", "adaptive", "criterion", "eta", "dt_fixed"]
        )
        out = RESULTS_DIR / "phase7_summary.csv"
        df.to_csv(out, index=False)
        print(f"\nResumen global → {out}")
        print(f"Series por run en {TIMESERIES_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
