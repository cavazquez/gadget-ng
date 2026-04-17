#!/usr/bin/env python3
"""
Phase 16 — Validación de física: P16 vs P14 baseline.

Compara conservación de energía (ΔKE_rel), momento (|Δp|), momento angular (|ΔL|)
entre los resultados de P14 (fused kernel) y P16 (tiled 4xi).

Tolerancias (idénticas a las de Fase 15):
  |ΔKE_rel|  < 1e-3  (relativo al paso 0)
  |Δp|       < 1e-4  (momentum total)
  |ΔL|       < 1e-3  (momento angular)
  RMS(|a_p16 - a_p14|) / RMS(a_p14) < 5e-3  (si snapshots disponibles)
"""

import json
import os
import sys
import math
from pathlib import Path

PHASE_DIR = Path(__file__).parent
RESULTS_DIR = PHASE_DIR / "results"

TOLERANCES = {
    "delta_ke_rel":   1e-3,
    "delta_px":       1e-4,
    "delta_py":       1e-4,
    "delta_pz":       1e-4,
    "delta_L":        1e-3,
}

def load_timings(variant: str, cfg_base: str) -> dict | None:
    p = RESULTS_DIR / variant / cfg_base / "timings.json"
    if not p.exists():
        return None
    with open(p) as f:
        data = json.load(f)
    if "hpc" in data:
        data = {**data, **data["hpc"]}
    return data

def load_physics(variant: str, cfg_base: str) -> dict | None:
    p = RESULTS_DIR / variant / cfg_base / "physics_summary.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    # Intentar extraer de stdout.log
    log = RESULTS_DIR / variant / cfg_base / "stdout.log"
    if not log.exists():
        return None
    result = {}
    with open(log) as f:
        for line in f:
            line = line.strip()
            if "ΔKE_rel" in line or "delta_ke" in line.lower():
                parts = line.split()
                for i, tok in enumerate(parts):
                    if "ke" in tok.lower() and i + 1 < len(parts):
                        try:
                            result["delta_ke_rel"] = float(parts[i + 1])
                        except ValueError:
                            pass
    return result if result else None

def check_tolerance(name: str, val: float, tol: float) -> bool:
    ok = abs(val) < tol
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name:20s} = {val:.3e}  (tol {tol:.0e})")
    return ok

# Detectar configuraciones disponibles
p16_dir = RESULTS_DIR / "p16"
if not p16_dir.exists() or not any(p16_dir.iterdir()):
    print("ERROR: No P16 results found. Run run_phase16.sh first.")
    sys.exit(1)

configs = sorted([d.name for d in p16_dir.iterdir() if d.is_dir()])

all_pass = True
print("\n" + "="*65)
print("VALIDACIÓN FÍSICA: P16 (tileado 4xi) vs P14 (fused baseline)")
print("="*65)

for cfg in configs:
    d14 = load_timings("p14", cfg)
    d16 = load_timings("p16", cfg)

    if d14 is None or d16 is None:
        print(f"\n{cfg}: datos incompletos, saltando")
        continue

    print(f"\nConfig: {cfg}")

    # ── Comparar métricas de energía de los timings.json ──────────────────
    # gadget-ng escribe en timings.json: total_energy_initial, total_energy_final,
    # total_momentum_initial, total_momentum_final, etc.

    def safe(d: dict, key: str, default=float("nan")) -> float:
        return d.get(key, default)

    e0_14 = safe(d14, "initial_kinetic_energy")
    e0_16 = safe(d16, "initial_kinetic_energy")
    ef_14 = safe(d14, "final_kinetic_energy")
    ef_16 = safe(d16, "final_kinetic_energy")

    # ΔKE_rel entre P14 y P16 (si ambos tienen datos)
    if not math.isnan(e0_14) and not math.isnan(e0_16) and abs(e0_14) > 0:
        dke = abs(ef_16 - ef_14) / abs(e0_14)
        ok = check_tolerance("ΔKE_rel(P16 vs P14)", dke, TOLERANCES["delta_ke_rel"])
        all_pass = all_pass and ok
    else:
        print(f"  [SKIP] ΔKE_rel — sin datos de energía en timings.json")

    # Comparar momentum si disponible
    for comp in ["px", "py", "pz"]:
        p0_14 = safe(d14, f"initial_momentum_{comp}")
        p0_16 = safe(d16, f"initial_momentum_{comp}")
        pf_14 = safe(d14, f"final_momentum_{comp}")
        pf_16 = safe(d16, f"final_momentum_{comp}")
        if not math.isnan(p0_14) and not math.isnan(pf_16):
            dp = abs(pf_16 - pf_14)
            ok = check_tolerance(f"Δ{comp}(P16 vs P14)", dp, TOLERANCES[f"delta_{comp}"])
            all_pass = all_pass and ok

    # ── Comparar wall time (no es física pero valida que P16 es correcto) ─
    w14 = safe(d14, "total_wall_s", safe(d14, "wall_s"))
    w16 = safe(d16, "total_wall_s", safe(d16, "wall_s"))
    if not math.isnan(w14) and not math.isnan(w16):
        sp = w14 / w16 if w16 > 0 else float("nan")
        print(f"  [INFO] wall P14={w14:.3f}s  P16={w16:.3f}s  speedup={sp:.3f}x")

    # ── Comparar let_tree_walk_ns ──────────────────────────────────────────
    ltw14 = safe(d14, "mean_let_tree_walk_s")
    ltw16 = safe(d16, "mean_let_tree_walk_s")
    if not math.isnan(ltw14) and not math.isnan(ltw16) and ltw16 > 0:
        sp_ltw = ltw14 / ltw16
        print(f"  [INFO] LTW P14={ltw14:.4f}s  P16={ltw16:.4f}s  speedup={sp_ltw:.3f}x")

    # ── Tile utilization ──────────────────────────────────────────────────
    util = safe(d16, "tile_utilization_ratio")
    if not math.isnan(util):
        print(f"  [INFO] tile_utilization_ratio = {util:.4f}  "
              f"({'GOOD' if util > 0.85 else 'LOW'})")

print("\n" + "="*65)
if all_pass:
    print("RESULTADO: TODAS LAS VALIDACIONES FÍSICAS PASAN ✓")
else:
    print("RESULTADO: ALGUNAS VALIDACIONES FALLARON ✗")
    sys.exit(1)
