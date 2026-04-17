#!/usr/bin/env python3
"""
Analiza los resultados de benchmarks de Fase 9.

Lee timings.json y diagnostics.jsonl de cada corrida y produce:
  - phase9_summary.csv
  - Figuras SVG:
      fig1_blocking_vs_overlap.svg
      fig2_allgather_vs_sfc_let.svg
      fig3_comm_breakdown.svg
      fig4_thread_imbalance.svg
"""
import os
import json
import csv
import sys
from pathlib import Path
from collections import defaultdict

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[warn] matplotlib no disponible; se omiten figuras.")

SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR.parent / "results"
PLOTS_DIR = SCRIPT_DIR.parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# ── Parseo de resultados ─────────────────────────────────────────────────────

def parse_run_dir(run_dir: Path) -> dict | None:
    """Lee timings.json y wall_ms.txt de un directorio de corrida."""
    timings_path = run_dir / "timings.json"
    wall_path = run_dir / "wall_ms.txt"
    if not timings_path.exists():
        return None

    with open(timings_path) as f:
        t = json.load(f)

    wall_ms = None
    if wall_path.exists():
        try:
            wall_ms = int(wall_path.read_text().strip())
        except ValueError:
            pass

    rec = {
        "run_dir": str(run_dir.name),
        "steps": t.get("steps", 0),
        "total_particles": t.get("total_particles", 0),
        "total_wall_s": t.get("total_wall_s", 0),
        "wall_ms": wall_ms,
        "comm_fraction": t.get("comm_fraction", 0),
        "gravity_fraction": t.get("gravity_fraction", 0),
        "mean_step_wall_s": t.get("mean_step_wall_s", 0),
    }

    hpc = t.get("hpc")
    if hpc:
        for k, v in hpc.items():
            rec[f"hpc_{k}"] = v

    # Parsear nombre del directorio para extraer metadatos
    name = run_dir.name
    # Formatos: strong_N2000_allgather_P1, weak500_N1000_blocking_P2
    parts = name.split("_")
    rec["scaling"] = "strong" if name.startswith("strong") else "weak"
    rec["n"] = 0
    rec["backend"] = "unknown"
    rec["p"] = 1
    for p in parts:
        if p.startswith("N") and p[1:].isdigit():
            rec["n"] = int(p[1:])
        elif p in ("allgather", "blocking", "overlap"):
            rec["backend"] = p
        elif p.startswith("P") and p[1:].isdigit():
            rec["p"] = int(p[1:])

    return rec


def load_all_results() -> list[dict]:
    rows = []
    if not RESULTS_DIR.exists():
        print(f"[warn] No existe {RESULTS_DIR}; no hay resultados.")
        return rows
    for d in sorted(RESULTS_DIR.iterdir()):
        if d.is_dir():
            rec = parse_run_dir(d)
            if rec:
                rows.append(rec)
    return rows


# ── CSV ──────────────────────────────────────────────────────────────────────

def write_csv(rows: list[dict]):
    if not rows:
        return
    out = SCRIPT_DIR.parent / "results" / "phase9_summary.csv"
    keys = sorted(set(k for r in rows for k in r.keys()))
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    print(f"[analyze] CSV: {out} ({len(rows)} filas)")


# ── Figuras ──────────────────────────────────────────────────────────────────

def fig1_blocking_vs_overlap(rows):
    """Figura 1: blocking vs overlap — wall time y wait_fraction vs P."""
    if not HAS_MPL:
        return
    strong = [r for r in rows if r["scaling"] == "strong" and r["backend"] in ("blocking", "overlap")]
    if not strong:
        print("[fig1] Sin datos strong para blocking/overlap.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for backend, color, ls in [("blocking", "C0", "-"), ("overlap", "C1", "--")]:
        for n, marker in [(2000, "o"), (4000, "s"), (8000, "^")]:
            subset = sorted(
                [r for r in strong if r["backend"] == backend and r["n"] == n],
                key=lambda r: r["p"],
            )
            if not subset:
                continue
            ps = [r["p"] for r in subset]
            walls = [r["mean_step_wall_s"] * 1000 for r in subset]  # ms/step
            axes[0].plot(ps, walls, marker=marker, linestyle=ls, color=color,
                         label=f"{backend} N={n}")

            wf = [r.get(f"hpc_wait_fraction", r.get("comm_fraction", 0)) for r in subset]
            axes[1].plot(ps, wf, marker=marker, linestyle=ls, color=color)

    axes[0].set_xlabel("Rangos MPI (P)")
    axes[0].set_ylabel("Tiempo medio por paso (ms)")
    axes[0].set_title("Fig 1a: Wall time — blocking vs overlap")
    axes[0].legend(fontsize=8, ncol=2)
    axes[0].set_xticks([1, 2, 4, 8])

    axes[1].set_xlabel("Rangos MPI (P)")
    axes[1].set_ylabel("Fracción de espera MPI")
    axes[1].set_title("Fig 1b: Wait fraction — blocking vs overlap")
    axes[1].set_xticks([1, 2, 4, 8])

    plt.tight_layout()
    out = PLOTS_DIR / "fig1_blocking_vs_overlap.svg"
    plt.savefig(out)
    plt.close()
    print(f"[fig1] {out}")


def fig2_allgather_vs_sfc_let(rows):
    """Figura 2: Allgather vs SFC+LET — strong/weak scaling efficiency."""
    if not HAS_MPL:
        return
    strong = [r for r in rows if r["scaling"] == "strong"]
    if not strong:
        print("[fig2] Sin datos strong.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    for backend, color in [("allgather", "C2"), ("blocking", "C0"), ("overlap", "C1")]:
        for n, ls, marker in [(2000, "-", "o"), (4000, "--", "s"), (8000, ":", "^")]:
            subset = sorted(
                [r for r in strong if r["backend"] == backend and r["n"] == n],
                key=lambda r: r["p"],
            )
            if not subset:
                continue
            # Eficiencia: speedup_ideal / speedup_real
            t1 = next((r["mean_step_wall_s"] for r in subset if r["p"] == 1), None)
            if t1 is None:
                continue
            ps = [r["p"] for r in subset]
            effs = [t1 / (r["mean_step_wall_s"] * r["p"]) if r["mean_step_wall_s"] > 0 else 0
                    for r in subset]
            ax.plot(ps, effs, marker=marker, linestyle=ls, color=color,
                    label=f"{backend} N={n}")

    ax.axhline(1.0, color="k", linestyle=":", linewidth=0.8, label="ideal")
    ax.set_xlabel("Rangos MPI (P)")
    ax.set_ylabel("Eficiencia de strong scaling")
    ax.set_title("Fig 2: Allgather vs SFC+LET — strong scaling efficiency")
    ax.legend(fontsize=7, ncol=3)
    ax.set_xticks([1, 2, 4, 8])
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    out = PLOTS_DIR / "fig2_allgather_vs_sfc_let.svg"
    plt.savefig(out)
    plt.close()
    print(f"[fig2] {out}")


def fig3_comm_breakdown(rows):
    """Figura 3: Desglose de tiempos (stacked bar) para overlap path."""
    if not HAS_MPL:
        return
    overlap = [r for r in rows if r["backend"] == "overlap" and r.get("n", 0) == 2000]
    if not overlap:
        print("[fig3] Sin datos overlap N=2000.")
        return

    overlap.sort(key=lambda r: r["p"])
    ps = [r["p"] for r in overlap]
    labels = ["tree_build", "let_export", "let_pack", "aabb_allgather", "let_alltoallv_wait", "walk_local", "apply_let"]
    colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#ff9da7", "#59a14f", "#b07aa1"]

    def ns_s(r, key):
        return r.get(f"hpc_mean_{key}_s", 0.0)

    data = {
        "tree_build": [ns_s(r, "tree_build") * 1000 for r in overlap],
        "let_export": [ns_s(r, "let_export") * 1000 for r in overlap],
        "let_pack": [ns_s(r, "let_pack") * 1000 for r in overlap],
        "aabb_allgather": [ns_s(r, "aabb_allgather") * 1000 for r in overlap],
        "let_alltoallv_wait": [max(0, ns_s(r, "let_alltoallv") - ns_s(r, "walk_local")) * 1000 for r in overlap],
        "walk_local": [ns_s(r, "walk_local") * 1000 for r in overlap],
        "apply_let": [ns_s(r, "apply_let") * 1000 for r in overlap],
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    bottom = [0.0] * len(ps)
    for label, color in zip(labels, colors):
        vals = data.get(label, [0.0] * len(ps))
        ax.bar([str(p) for p in ps], vals, bottom=bottom, label=label, color=color)
        bottom = [b + v for b, v in zip(bottom, vals)]

    ax.set_xlabel("Rangos MPI (P)")
    ax.set_ylabel("Tiempo medio por paso (ms)")
    ax.set_title("Fig 3: Desglose de tiempos SFC+LET overlap (N=2000)")
    ax.legend(fontsize=8, loc="upper right")
    plt.tight_layout()
    out = PLOTS_DIR / "fig3_comm_breakdown.svg"
    plt.savefig(out)
    plt.close()
    print(f"[fig3] {out}")


def fig4_thread_imbalance(rows):
    """Figura 4: Imbalance de hilos (wait_fraction) vs P."""
    if not HAS_MPL:
        return
    # wait_fraction proxy: comm_fraction o hpc_wait_fraction
    relevant = [r for r in rows if r["backend"] in ("blocking", "overlap") and r["scaling"] == "strong"]
    if not relevant:
        print("[fig4] Sin datos para imbalance plot.")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    for backend, color, ls in [("blocking", "C0", "-"), ("overlap", "C1", "--")]:
        for n, marker in [(2000, "o"), (4000, "s")]:
            subset = sorted(
                [r for r in relevant if r["backend"] == backend and r["n"] == n],
                key=lambda r: r["p"],
            )
            if not subset:
                continue
            ps = [r["p"] for r in subset]
            wf = [r.get("hpc_wait_fraction", r.get("comm_fraction", 0)) for r in subset]
            ax.plot(ps, wf, marker=marker, linestyle=ls, color=color,
                    label=f"{backend} N={n}")

    ax.set_xlabel("Rangos MPI (P)")
    ax.set_ylabel("Fracción de espera MPI")
    ax.set_title("Fig 4: Wait fraction (comm imbalance proxy) vs P")
    ax.legend(fontsize=9)
    ax.set_xticks([1, 2, 4, 8])
    plt.tight_layout()
    out = PLOTS_DIR / "fig4_wait_fraction.svg"
    plt.savefig(out)
    plt.close()
    print(f"[fig4] {out}")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rows = load_all_results()
    if not rows:
        print("[analyze] No hay resultados aún. Ejecuta primero run_phase9.sh.")
        sys.exit(0)

    write_csv(rows)
    print(f"[analyze] {len(rows)} corridas cargadas.")

    fig1_blocking_vs_overlap(rows)
    fig2_allgather_vs_sfc_let(rows)
    fig3_comm_breakdown(rows)
    fig4_thread_imbalance(rows)

    print("[analyze] Análisis completado.")
