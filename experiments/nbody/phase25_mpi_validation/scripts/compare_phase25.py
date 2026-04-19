#!/usr/bin/env python3
"""
compare_phase25.py — Análisis comparativo de benchmarks MPI Phase 25
=====================================================================

Extrae métricas de timings.json y diagnostics.jsonl para cada run de la
matriz (variante × N × P) y produce:
  - Tablas comparativas en stdout
  - phase25_comparison.csv en results/
  - Respuestas a preguntas A–E del plan

Uso:
  python3 compare_phase25.py [--results-dir <dir>]
"""

import json
import csv
import sys
import os
from pathlib import Path


# ── Configuración ─────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent
BASE_DIR   = SCRIPT_DIR.parent
RESULTS_DIR = BASE_DIR / "results"

# Partícula completa en Fase 23 (clone): pos(24) + vel(24) + acc(24) + mass(8) + gid(8) = 88 bytes
PARTICLE_BYTES_FULL = 88

# Scatter (Fase 24): gid(8) + pos(24) + mass(8) = 40 bytes
SCATTER_BYTES_PER = 40
# Gather (Fase 24): gid(8) + acc(24) = 32 bytes
GATHER_BYTES_PER = 32
# Total Fase 24 por partícula: 40 + 32 = 72 bytes
TOTAL_BYTES_F24_PER = SCATTER_BYTES_PER + GATHER_BYTES_PER
# Fase 23 envía 2× (ida y vuelta): 88 * 2 = 176 bytes/partícula
TOTAL_BYTES_F23_PER = PARTICLE_BYTES_FULL * 2
# Reducción teórica: 176 / 72 ≈ 2.44×
THEORETICAL_REDUCTION = TOTAL_BYTES_F23_PER / TOTAL_BYTES_F24_PER


# ── Funciones de extracción ────────────────────────────────────────────────────

def load_timings(run_dir: Path) -> dict:
    """Carga timings.json; devuelve {} si no existe."""
    p = run_dir / "timings.json"
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


def load_last_diag(run_dir: Path) -> dict:
    """Carga la última línea de diagnostics.jsonl; devuelve {} si no existe."""
    p = run_dir / "diagnostics.jsonl"
    if not p.exists():
        return {}
    last = {}
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    last = json.loads(line)
                except json.JSONDecodeError:
                    pass
    return last


def parse_run_id(run_id: str):
    """Parsea 'fase23_N512_P2' → ('fase23', 512, 2)."""
    parts = run_id.split("_")
    variante = parts[0]
    N = int(parts[1][1:])
    P = int(parts[2][1:])
    return variante, N, P


def collect_all_runs(results_dir: Path) -> list[dict]:
    """Recolecta métricas de todos los runs disponibles."""
    rows = []
    for run_dir in sorted(results_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        run_id = run_dir.name
        try:
            variante, N, P = parse_run_id(run_id)
        except (ValueError, IndexError):
            continue

        t = load_timings(run_dir)
        d = load_last_diag(run_dir)
        if not t:
            continue

        hpc = t.get("treepm_hpc", {})
        treepm_diag = d.get("treepm", {})

        scatter_ns   = hpc.get("mean_scatter_s", 0.0) * 1e9
        gather_ns    = hpc.get("mean_gather_s",  0.0) * 1e9
        pm_solve_ns  = hpc.get("mean_pm_solve_s", 0.0) * 1e9
        sr_halo_ns   = hpc.get("mean_sr_halo_s",  0.0) * 1e9
        tree_sr_ns   = hpc.get("mean_tree_sr_s",  0.0) * 1e9
        scatter_parts  = hpc.get("mean_scatter_particles", 0.0)
        scatter_bytes  = hpc.get("mean_scatter_bytes", 0.0)
        gather_bytes   = hpc.get("mean_gather_bytes",  0.0)
        pm_sync_frac   = hpc.get("pm_sync_fraction", 0.0)
        total_treepm_s = hpc.get("mean_treepm_total_s", 0.0)
        path_active    = hpc.get("path_active", "unknown")

        # Bytes totales medidos por paso (scatter + gather)
        total_bytes_measured = scatter_bytes + gather_bytes

        # Para Fase 23: estimar bytes teóricos (clone+migrate no reporta bytes directamente)
        if variante == "fase23":
            bytes_theoretical_clone = scatter_parts * TOTAL_BYTES_F23_PER
        else:
            bytes_theoretical_clone = None

        row = {
            "run_id":           run_id,
            "variante":         variante,
            "N":                N,
            "P":                P,
            "path_active":      path_active,
            # Wall time
            "total_wall_s":     t.get("total_wall_s", 0.0),
            "mean_step_wall_s": t.get("mean_step_wall_s", 0.0),
            # Comm / gravity
            "total_comm_s":     t.get("total_comm_s", 0.0),
            "comm_fraction":    t.get("comm_fraction", 0.0),
            "gravity_fraction": t.get("gravity_fraction", 0.0),
            # TreePM desglose (ns por paso medio)
            "scatter_ns":      scatter_ns,
            "gather_ns":       gather_ns,
            "pm_solve_ns":     pm_solve_ns,
            "sr_halo_ns":      sr_halo_ns,
            "tree_sr_ns":      tree_sr_ns,
            # Bytes
            "scatter_particles": scatter_parts,
            "scatter_bytes":   scatter_bytes,
            "gather_bytes":    gather_bytes,
            "total_bytes":     total_bytes_measured,
            "bytes_theo_clone": bytes_theoretical_clone or 0.0,
            # Fracciones
            "pm_sync_fraction": pm_sync_frac,
            "mean_treepm_total_s": total_treepm_s,
            # Física (último paso)
            "v_rms":     d.get("v_rms", float("nan")),
            "delta_rms": d.get("delta_rms", float("nan")),
            "a":         d.get("a", float("nan")),
        }
        rows.append(row)
    return rows


# ── Tablas ────────────────────────────────────────────────────────────────────

def fmt(v, prec=4, unit=""):
    if v is None or (isinstance(v, float) and v != v):
        return "—"
    if isinstance(v, float):
        return f"{v:.{prec}f}{unit}"
    return str(v)


def print_section(title: str):
    print()
    print("═" * 70)
    print(f"  {title}")
    print("═" * 70)


def table_wall_time(rows: list[dict]):
    print_section("A. Tiempo de pared (total_wall_s) — Fase 23 vs Fase 24")
    header = f"{'N':>6} {'P':>3} {'Fase23_s':>10} {'Fase24_s':>10} {'Δ%':>8}"
    print(header)
    print("-" * len(header))

    by_NP = {}
    for r in rows:
        key = (r["N"], r["P"])
        by_NP.setdefault(key, {})[r["variante"]] = r["total_wall_s"]

    for (N, P) in sorted(by_NP.keys()):
        d = by_NP[(N, P)]
        f23 = d.get("fase23", float("nan"))
        f24 = d.get("fase24", float("nan"))
        if f23 and f24 and f23 > 0:
            delta_pct = (f24 - f23) / f23 * 100
            flag = " ←+" if delta_pct > 2 else (" ←−" if delta_pct < -2 else "")
            print(f"{N:>6} {P:>3} {f23:>10.4f} {f24:>10.4f} {delta_pct:>+7.1f}%{flag}")
        else:
            print(f"{N:>6} {P:>3} {'—':>10} {'—':>10} {'—':>8}")


def table_comm_fraction(rows: list[dict]):
    print_section("B. Fracción de comunicación (comm_fraction) — Fase 23 vs Fase 24")
    header = f"{'N':>6} {'P':>3} {'F23_comm%':>10} {'F24_comm%':>10} {'Reducción':>10}"
    print(header)
    print("-" * len(header))

    by_NP = {}
    for r in rows:
        key = (r["N"], r["P"])
        by_NP.setdefault(key, {})[r["variante"]] = r["comm_fraction"]

    for (N, P) in sorted(by_NP.keys()):
        d = by_NP[(N, P)]
        f23 = d.get("fase23")
        f24 = d.get("fase24")
        if f23 is not None and f24 is not None and f24 > 0:
            reduccion = f23 / f24
            print(f"{N:>6} {P:>3} {f23*100:>9.2f}% {f24*100:>9.2f}% {reduccion:>9.2f}×")
        else:
            print(f"{N:>6} {P:>3} {'—':>10} {'—':>10} {'—':>10}")


def table_bytes(rows: list[dict]):
    print_section("C. Bytes/rank por paso — Fase 23 (teórico) vs Fase 24 (medido)")
    header = f"{'N':>6} {'P':>3} {'F23_bytes_teo':>14} {'F24_bytes_med':>14} {'Reducción':>10}"
    print(header)
    print("-" * len(header))

    by_NP = {}
    for r in rows:
        key = (r["N"], r["P"])
        by_NP.setdefault(key, {})[r["variante"]] = r

    for (N, P) in sorted(by_NP.keys()):
        d = by_NP[(N, P)]
        r23 = d.get("fase23")
        r24 = d.get("fase24")
        if r23 and r24:
            b23 = r23["bytes_theo_clone"]
            b24 = r24["total_bytes"]
            if b24 > 0:
                reduccion = b23 / b24
                print(f"{N:>6} {P:>3} {b23:>14,.0f} {b24:>14,.0f} {reduccion:>9.2f}×")
            else:
                print(f"{N:>6} {P:>3} {b23:>14,.0f} {'—':>14} {'—':>10}")
        else:
            print(f"{N:>6} {P:>3} {'—':>14} {'—':>14} {'—':>10}")

    print()
    print(f"  Reducción teórica esperada: {THEORETICAL_REDUCTION:.2f}× "
          f"({TOTAL_BYTES_F23_PER} bytes/part Fase23 vs {TOTAL_BYTES_F24_PER} bytes/part Fase24)")


def table_sg_timing(rows: list[dict]):
    print_section("D. Tiempos scatter/gather vs comm total — Fase 24 únicamente")
    header = f"{'N':>6} {'P':>3} {'scatter_µs':>11} {'gather_µs':>10} {'pm_sync%':>9} {'total_ns':>10}"
    print(header)
    print("-" * len(header))

    for r in sorted(rows, key=lambda x: (x["N"], x["P"])):
        if r["variante"] != "fase24":
            continue
        sc_us = r["scatter_ns"] / 1000
        ga_us = r["gather_ns"] / 1000
        frac  = r["pm_sync_fraction"] * 100
        tot_s = r["mean_treepm_total_s"] * 1e6
        print(f"{r['N']:>6} {r['P']:>3} {sc_us:>10.1f}µ {ga_us:>9.1f}µ {frac:>8.3f}% {tot_s:>9.1f}µ")


def table_physics(rows: list[dict]):
    print_section("E. Equivalencia física — v_rms y delta_rms al paso final")
    header = f"{'N':>6} {'P':>3} {'variante':>8} {'a':>8} {'v_rms':>12} {'delta_rms':>10}"
    print(header)
    print("-" * len(header))

    for r in sorted(rows, key=lambda x: (x["N"], x["P"], x["variante"])):
        print(f"{r['N']:>6} {r['P']:>3} {r['variante']:>8} "
              f"{r['a']:>8.4f} {r['v_rms']:>12.4f} {r['delta_rms']:>10.6f}")


def table_physics_diff(rows: list[dict]):
    print_section("E2. Diferencia física Fase24 − Fase23 por (N, P)")
    header = f"{'N':>6} {'P':>3} {'Δv_rms':>14} {'Δdelta_rms':>14} {'Δv%':>8}"
    print(header)
    print("-" * len(header))

    by_NP = {}
    for r in rows:
        key = (r["N"], r["P"])
        by_NP.setdefault(key, {})[r["variante"]] = r

    for (N, P) in sorted(by_NP.keys()):
        d = by_NP[(N, P)]
        r23 = d.get("fase23")
        r24 = d.get("fase24")
        if r23 and r24:
            dv = r24["v_rms"] - r23["v_rms"]
            dd = r24["delta_rms"] - r23["delta_rms"]
            dv_pct = abs(dv / r23["v_rms"]) * 100 if r23["v_rms"] else float("nan")
            flag = " ← DIVERGE" if dv_pct > 1 else ""
            print(f"{N:>6} {P:>3} {dv:>+14.6f} {dd:>+14.8f} {dv_pct:>7.4f}%{flag}")
        else:
            print(f"{N:>6} {P:>3} {'—':>14} {'—':>14} {'—':>8}")


# ── CSV ───────────────────────────────────────────────────────────────────────

CSV_FIELDS = [
    "run_id", "variante", "N", "P", "path_active",
    "total_wall_s", "mean_step_wall_s",
    "total_comm_s", "comm_fraction", "gravity_fraction",
    "scatter_ns", "gather_ns", "pm_solve_ns", "sr_halo_ns", "tree_sr_ns",
    "scatter_particles", "scatter_bytes", "gather_bytes", "total_bytes",
    "bytes_theo_clone", "pm_sync_fraction", "mean_treepm_total_s",
    "v_rms", "delta_rms", "a",
]


def write_csv(rows: list[dict], out_path: Path):
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\n  → CSV guardado en: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    results_dir = RESULTS_DIR
    for arg in sys.argv[1:]:
        if arg.startswith("--results-dir="):
            results_dir = Path(arg.split("=", 1)[1])
        elif arg == "--results-dir":
            idx = sys.argv.index(arg)
            results_dir = Path(sys.argv[idx + 1])

    rows = collect_all_runs(results_dir)
    if not rows:
        print(f"ERROR: No se encontraron runs en {results_dir}")
        sys.exit(1)

    print(f"Phase 25: Análisis comparativo MPI — {len(rows)} runs procesados")
    print(f"Reducción teórica de bytes Fase23→Fase24: {THEORETICAL_REDUCTION:.2f}×")

    table_wall_time(rows)
    table_comm_fraction(rows)
    table_bytes(rows)
    table_sg_timing(rows)
    table_physics(rows)
    table_physics_diff(rows)

    # Resumen ejecutivo
    print_section("RESUMEN EJECUTIVO")

    by_NP = {}
    for r in rows:
        key = (r["N"], r["P"])
        by_NP.setdefault(key, {})[r["variante"]] = r

    print(f"\n{'':>40} Fase23  Fase24  Mejora")
    max_comm_reduction = 0.0
    max_wall_change    = 0.0
    for (N, P) in sorted(by_NP.keys()):
        d = by_NP[(N, P)]
        if "fase23" not in d or "fase24" not in d:
            continue
        r23, r24 = d["fase23"], d["fase24"]
        c23 = r23["comm_fraction"]
        c24 = r24["comm_fraction"]
        w23 = r23["total_wall_s"]
        w24 = r24["total_wall_s"]
        cr = c23 / c24 if c24 > 1e-9 else float("inf")
        wd = (w24 - w23) / w23 * 100
        max_comm_reduction = max(max_comm_reduction, cr)
        max_wall_change    = max(abs(wd), max_wall_change)
        print(f"  N={N:>4} P={P}: comm_frac {c23*100:.2f}% → {c24*100:.2f}% ({cr:.1f}× menor) | "
              f"wall {w23:.3f}s → {w24:.3f}s ({wd:+.1f}%)")

    print(f"\n  Máxima reducción de comm: {max_comm_reduction:.1f}×")
    print(f"  Máximo cambio de wall time: ±{max_wall_change:.1f}%")
    print()

    if max_comm_reduction > 5:
        print("  ✓ Fase 24 reduce significativamente la fracción de comunicación MPI.")
    if max_wall_change < 5:
        print("  ✓ Fase 24 es wall-time neutral (cambio < ±5%) — sin regresión.")
    if max_comm_reduction > 5 and max_wall_change < 5:
        print("  → Recomendación: treepm_pm_scatter_gather = true puede ser el default")
        print("    para P>1; el overhead en P=1 es despreciable.")

    write_csv(rows, results_dir / "phase25_comparison.csv")


if __name__ == "__main__":
    main()
