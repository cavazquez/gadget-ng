#!/usr/bin/env python3
"""
Fase 24: Análisis comparativo PM Scatter/Gather vs Fase 23 (clone+migrate).

Extrae métricas de timings.json y diagnostics.jsonl para comparar:
- Fase 23: clone → exchange_domain_by_z → PM → exchange_domain_sfc → HashMap
- Fase 24: scatter (gid+pos+mass) → PM slab → gather (gid+acc)

Uso:
    python3 compare_results.py <results_dir>
"""

import json
import os
import sys
from pathlib import Path


def load_timings(run_dir: Path) -> dict:
    """Carga timings.json de un directorio de resultados."""
    f = run_dir / "timings.json"
    if not f.exists():
        return {}
    with open(f) as fp:
        return json.load(fp)


def load_diagnostics(run_dir: Path) -> list[dict]:
    """Carga diagnostics.jsonl (una línea JSON por paso)."""
    f = run_dir / "diagnostics.jsonl"
    if not f.exists():
        return []
    lines = []
    with open(f) as fp:
        for line in fp:
            line = line.strip()
            if line:
                try:
                    lines.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return lines


def extract_physics(diags: list[dict]) -> dict:
    """Extrae métricas físicas del último paso."""
    if not diags:
        return {}
    last = diags[-1]
    return {
        "a":           last.get("a", None),
        "v_rms":       last.get("v_rms", None),
        "delta_rms":   last.get("delta_rms", None),
        "kinetic_e":   last.get("kinetic_e", None),
        "step":        last.get("step", None),
    }


def extract_timing(t: dict) -> dict:
    """Extrae métricas de timing principales."""
    return {
        "total_wall_s":    t.get("total_wall_s", None),
        "mean_step_wall_s": t.get("mean_step_wall_s", None),
        "comm_fraction":   t.get("comm_fraction", None),
        "gravity_fraction": t.get("gravity_fraction", None),
        "steps":           t.get("steps", None),
        "total_particles": t.get("total_particles", None),
    }


def print_table(rows: list[tuple], headers: list[str], title: str):
    """Imprime tabla formateada."""
    print(f"\n{'═' * 72}")
    print(f"  {title}")
    print(f"{'═' * 72}")
    col_w = [max(len(h), max((len(str(r[i])) for r in rows), default=0))
             for i, h in enumerate(headers)]
    fmt = "  " + "  ".join(f"{{:<{w}}}" for w in col_w)
    sep = "  " + "  ".join("─" * w for w in col_w)
    print(fmt.format(*headers))
    print(sep)
    for row in rows:
        print(fmt.format(*[str(x) for x in row]))


def main(results_dir: str):
    base = Path(results_dir)
    runs = sorted([d for d in base.iterdir() if d.is_dir()])

    if not runs:
        print(f"No se encontraron directorios de resultados en: {results_dir}")
        sys.exit(0)

    print(f"\n{'═' * 72}")
    print("  FASE 24: Análisis PM Scatter/Gather vs Fase 23 (clone+migrate)")
    print(f"{'═' * 72}")
    print(f"  Directorio: {results_dir}")
    print(f"  Runs encontrados: {len(runs)}")
    for r in runs:
        print(f"    • {r.name}")

    # ── Tabla de timings ─────────────────────────────────────────────────────
    timing_rows = []
    for run in runs:
        t = extract_timing(load_timings(run))
        timing_rows.append((
            run.name,
            f"{t.get('total_particles', '?')}",
            f"{t.get('steps', '?')}",
            f"{t.get('total_wall_s', 0):.3f}s" if t.get('total_wall_s') else "?",
            f"{t.get('mean_step_wall_s', 0)*1000:.1f}ms" if t.get('mean_step_wall_s') else "?",
            f"{t.get('comm_fraction', 0)*100:.1f}%" if t.get('comm_fraction') is not None else "?",
            f"{t.get('gravity_fraction', 0)*100:.1f}%" if t.get('gravity_fraction') is not None else "?",
        ))
    print_table(timing_rows,
                ["Run", "N", "Pasos", "Wall total", "Wall/paso", "f_comm", "f_grav"],
                "Timings")

    # ── Tabla de física ──────────────────────────────────────────────────────
    phys_rows = []
    for run in runs:
        diags = load_diagnostics(run)
        p = extract_physics(diags)
        phys_rows.append((
            run.name,
            f"{p.get('step', '?')}",
            f"{p.get('a', 0):.4f}" if p.get('a') else "?",
            f"{p.get('v_rms', 0):.4e}" if p.get('v_rms') is not None else "?",
            f"{p.get('delta_rms', 0):.4e}" if p.get('delta_rms') is not None else "?",
        ))
    print_table(phys_rows,
                ["Run", "Paso final", "a(t_fin)", "v_rms", "delta_rms"],
                "Física (último paso)")

    # ── Comparación Fase 23 vs Fase 24 ──────────────────────────────────────
    clone_run = next((r for r in runs if "clone" in r.name.lower()), None)
    sg_run    = next((r for r in runs if "sg_N512" in r.name and "p1" in r.name.lower()), None)

    if clone_run and sg_run:
        t23 = load_timings(clone_run)
        t24 = load_timings(sg_run)
        d23 = load_diagnostics(clone_run)
        d24 = load_diagnostics(sg_run)

        wall23 = t23.get("total_wall_s", 0)
        wall24 = t24.get("total_wall_s", 0)
        speedup = wall23 / wall24 if wall24 > 0 else float("nan")

        # Comparación física (último paso)
        p23 = extract_physics(d23)
        p24 = extract_physics(d24)

        print(f"\n{'═' * 72}")
        print("  Comparación directa: Fase 23 vs Fase 24 (N=512, P=1)")
        print(f"{'═' * 72}")
        print(f"  {'Métrica':<30} {'Fase 23 (clone)':>20} {'Fase 24 (scatter)':>20}")
        print(f"  {'─'*30} {'─'*20} {'─'*20}")
        print(f"  {'Wall total (s)':<30} {wall23:>20.3f} {wall24:>20.3f}")
        print(f"  {'Speedup Fase24/Fase23':<30} {'─':>20} {speedup:>20.3f}×")

        for key, label in [("v_rms", "v_rms"), ("delta_rms", "delta_rms"), ("a", "a(t_fin)")]:
            v23 = p23.get(key)
            v24 = p24.get(key)
            if v23 is not None and v24 is not None and v23 != 0:
                rel_diff = abs(v24 - v23) / abs(v23)
                print(f"  {label:<30} {v23:>20.6e} {v24:>20.6e}  (Δ_rel={rel_diff:.2e})")
            else:
                print(f"  {label:<30} {'N/A':>20} {'N/A':>20}")

    # ── Bytes scatter/gather ─────────────────────────────────────────────────
    print(f"\n{'═' * 72}")
    print("  Estimación teórica de bytes ahorrados por scatter/gather")
    print(f"{'═' * 72}")
    for run in runs:
        t = load_timings(run)
        n = t.get("total_particles", 0)
        if n:
            bytes_fase23 = n * 88 * 2  # Particle completo × 2 migraciones
            bytes_fase24_scatter = n * 5 * 8  # (gid,x,y,z,mass) = 5×f64
            bytes_fase24_gather  = n * 4 * 8  # (gid,ax,ay,az) = 4×f64
            bytes_fase24 = bytes_fase24_scatter + bytes_fase24_gather
            ratio = bytes_fase23 / bytes_fase24 if bytes_fase24 > 0 else 0
            print(f"  {run.name}: N={n}")
            print(f"    Fase 23: ~{bytes_fase23:,} bytes ({n}×88×2)")
            print(f"    Fase 24: ~{bytes_fase24:,} bytes ({n}×(40+32))")
            print(f"    Reducción: {ratio:.1f}× menos bytes de red")

    # ── Guardar CSV de comparación ────────────────────────────────────────────
    csv_path = base / "comparison_phase24.csv"
    with open(csv_path, "w") as f:
        f.write("run,n_particles,steps,total_wall_s,mean_step_ms,comm_fraction,gravity_fraction,a_fin,v_rms,delta_rms\n")
        for run in runs:
            t = extract_timing(load_timings(run))
            p = extract_physics(load_diagnostics(run))
            f.write(",".join([
                run.name,
                str(t.get("total_particles", "")),
                str(t.get("steps", "")),
                f"{t.get('total_wall_s', ''):.4f}" if t.get('total_wall_s') else "",
                f"{t.get('mean_step_wall_s', 0)*1000:.2f}" if t.get('mean_step_wall_s') else "",
                f"{t.get('comm_fraction', ''):.4f}" if t.get('comm_fraction') is not None else "",
                f"{t.get('gravity_fraction', ''):.4f}" if t.get('gravity_fraction') is not None else "",
                f"{p.get('a', ''):.6f}" if p.get('a') else "",
                f"{p.get('v_rms', ''):.6e}" if p.get('v_rms') is not None else "",
                f"{p.get('delta_rms', ''):.6e}" if p.get('delta_rms') is not None else "",
            ]) + "\n")
    print(f"\n  CSV guardado en: {csv_path}")
    print(f"\n{'═' * 72}\n")


if __name__ == "__main__":
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
    main(results_dir)
