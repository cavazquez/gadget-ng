#!/usr/bin/env python3
"""
Análisis de benchmarks Fase 19: PM Distribuido sin allgather de partículas.

Compara:
  1. PM distribuido (Fase 19) vs PM clásico (Fase 18) — equivalencia física
  2. Serial (P=1) vs MPI (P=2, P=4) para el path distribuido
  3. Estimación de bytes/rank: allgather O(N·P) vs allreduce O(nm³)
  4. Wall time por paso
"""

import json
import os
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"


def load_diagnostics(run_dir: Path) -> list[dict]:
    diag = run_dir / "diagnostics.jsonl"
    if not diag.exists():
        return []
    rows = []
    with open(diag) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def final_row(rows: list[dict]) -> dict | None:
    return rows[-1] if rows else None


def wall_time_per_step_ms(rows: list[dict]) -> float | None:
    times = [r.get("step_time_ns", None) for r in rows if r.get("step_time_ns")]
    if not times:
        return None
    return sum(times) / len(times) / 1e6  # ns → ms


def print_separator(title: str = "") -> None:
    w = 72
    if title:
        print(f"\n{'─'*4} {title} {'─'*(w - len(title) - 6)}")
    else:
        print("─" * w)


def bytes_allgather(n: int, p: int) -> int:
    """Bytes totales enviados en allgather de partículas: 5 f64 × N × P."""
    return 5 * 8 * n * p


def bytes_allreduce_grid(nm: int) -> int:
    """Bytes totales enviados en allreduce del grid: 8 × nm³."""
    return 8 * nm ** 3


def main() -> None:
    if not RESULTS_DIR.exists():
        print(f"[ERROR] No se encontró el directorio de resultados: {RESULTS_DIR}")
        print("Ejecuta primero: bash run_phase19.sh")
        sys.exit(1)

    # ── Sección 1: Comunicación teórica ───────────────────────────────────────
    print_separator("1. COMPARATIVA TEÓRICA DE COMUNICACIÓN")
    print(f"{'N':>8} {'P':>4} {'nm':>4} | {'allgather(KB)':>14} {'allreduce(KB)':>14} {'ratio':>8}")
    print_separator()
    for n in [512, 1000, 2000, 4000, 10_000, 100_000, 1_000_000]:
        for p in [1, 2, 4, 8]:
            for nm in [16, 32]:
                ag = bytes_allgather(n, p) / 1024
                ar = bytes_allreduce_grid(nm) / 1024
                ratio = ag / ar
                print(f"{n:>8} {p:>4} {nm:>4} | {ag:>13.1f}K {ar:>13.1f}K {ratio:>8.2f}x")
            if p == 1:
                break  # para P=1 solo una vez

    # ── Sección 2: Equivalencia PM distribuido vs PM clásico ─────────────────
    print_separator("2. EQUIVALENCIA PM DISTRIBUIDO vs PM CLÁSICO")

    classic_dir = RESULTS_DIR / "N512_classic_P1"
    dist_dir    = RESULTS_DIR / "N512_dist_P1"

    classic_rows = load_diagnostics(classic_dir)
    dist_rows    = load_diagnostics(dist_dir)

    if not classic_rows or not dist_rows:
        print("[SKIP] Faltan resultados de N512_classic_P1 o N512_dist_P1. Ejecuta run_phase19.sh.")
    else:
        print(f"{'paso':>6} {'a(classic)':>12} {'a(dist)':>12} {'Δa/a':>10} {'vrms_classic':>14} {'vrms_dist':>14} {'Δvrms':>10}")
        print_separator()
        for rc, rd in zip(classic_rows, dist_rows):
            step = rc.get("step", "?")
            ac = rc.get("cosmo", {}).get("a", None)
            ad = rd.get("cosmo", {}).get("a", None)
            vc = rc.get("cosmo", {}).get("v_rms", None)
            vd = rd.get("cosmo", {}).get("v_rms", None)
            if ac is None or ad is None:
                continue
            da = abs(ac - ad) / max(abs(ac), 1e-15)
            dv = abs(vc - vd) if (vc is not None and vd is not None) else float("nan")
            print(f"{step:>6} {ac:>12.6f} {ad:>12.6f} {da:>10.2e} {vc:>14.6e} {vd:>14.6e} {dv:>10.2e}")

    # ── Sección 3: Serial vs MPI (path distribuido) ───────────────────────────
    print_separator("3. SERIAL vs MPI — PATH DISTRIBUIDO N=512 EdS")

    for p in [1, 2, 4]:
        run_dir = RESULTS_DIR / f"N512_dist_P{p}"
        rows = load_diagnostics(run_dir)
        fr = final_row(rows)
        if not fr:
            print(f"  P={p}: [NO DATOS] — ejecuta run_phase19.sh con mpirun.")
            continue
        cosmo = fr.get("cosmo", {})
        a_fin  = cosmo.get("a", float("nan"))
        vrms   = cosmo.get("v_rms", float("nan"))
        drms   = cosmo.get("delta_rms", float("nan"))
        wtms   = wall_time_per_step_ms(rows)
        wt_str = f"{wtms:.2f} ms/paso" if wtms else "N/A"
        print(f"  P={p}: a_fin={a_fin:.6f}  v_rms={vrms:.4e}  delta_rms={drms:.4e}  {wt_str}")

    # ── Sección 4: Escalabilidad N=2000 ΛCDM ─────────────────────────────────
    print_separator("4. ESCALABILIDAD ΛCDM N=2000")

    for p in [1, 2, 4]:
        run_dir = RESULTS_DIR / f"N2000_lcdm_dist_P{p}"
        rows = load_diagnostics(run_dir)
        fr = final_row(rows)
        if not fr:
            print(f"  P={p}: [NO DATOS]")
            continue
        cosmo = fr.get("cosmo", {})
        a_fin = cosmo.get("a", float("nan"))
        vrms  = cosmo.get("v_rms", float("nan"))
        wtms  = wall_time_per_step_ms(rows)
        wt_str = f"{wtms:.2f} ms/paso" if wtms else "N/A"
        print(f"  P={p}: a_fin={a_fin:.6f}  v_rms={vrms:.4e}  {wt_str}")

    # ── Sección 5: N=4000 EdS ─────────────────────────────────────────────────
    print_separator("5. ESCALABILIDAD EdS N=4000")

    for p in [1, 2, 4]:
        run_dir = RESULTS_DIR / f"N4000_dist_P{p}"
        rows = load_diagnostics(run_dir)
        fr = final_row(rows)
        if not fr:
            print(f"  P={p}: [NO DATOS]")
            continue
        cosmo = fr.get("cosmo", {})
        a_fin = cosmo.get("a", float("nan"))
        vrms  = cosmo.get("v_rms", float("nan"))
        wtms  = wall_time_per_step_ms(rows)
        wt_str = f"{wtms:.2f} ms/paso" if wtms else "N/A"
        print(f"  P={p}: a_fin={a_fin:.6f}  v_rms={vrms:.4e}  {wt_str}")

    # ── Sección 6: Resumen ─────────────────────────────────────────────────────
    print_separator("6. RESUMEN ARQUITECTURAL")
    print("  Path Fase 18 (clásico): allgatherv_state → 5·N f64/rank → O(N·P) bytes")
    print("  Path Fase 19 (dist):    allreduce(nm³)  → nm³ f64 (fijo) → O(nm³) bytes")
    print()
    print("  Ventaja del path distribuido (asintótica, P=4):")
    for nm in [16, 32]:
        ar_kb = bytes_allreduce_grid(nm) / 1024
        for n in [512, 4000, 100_000]:
            ag_kb = bytes_allgather(n, 4) / 1024
            ratio = ag_kb / ar_kb
            print(f"    N={n:>8}, nm={nm}: allgather={ag_kb:.0f}KB vs allreduce={ar_kb:.0f}KB → {ratio:.1f}x menos")
    print()
    print("  Próximo cuello de botella: solve Poisson O(nm³·log nm) replicado en todos los ranks.")
    print("  Para nm>64 → FFT distribuida (FFTW-MPI / pencil decomposition).")


if __name__ == "__main__":
    main()
