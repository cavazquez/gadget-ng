#!/usr/bin/env python3
"""
Phase 12 — analyze_phase12.py

Lee los timings.json de todos los runs, construye phase12_summary.csv
y genera figuras SVG de rendimiento y comunicación LET.
"""

import json
import csv
import re
import os
import sys
import pathlib
import collections

SCRIPT_DIR = pathlib.Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR.parent / "results"
OUT_DIR = RESULTS_DIR

# ── Cargar resultados ─────────────────────────────────────────────────────────

Row = collections.namedtuple("Row", [
    "cfg_name", "group", "N", "P", "factor",
    "wall_s",
    "mean_bytes_sent", "mean_bytes_recv",
    "mean_let_nodes_exported", "mean_let_nodes_imported",
    "mean_max_let_nodes_per_rank", "mean_local_tree_nodes",
    "mean_export_prune_ratio",
    "mean_let_alltoallv_s", "mean_walk_local_s",
    "mean_let_tree_build_s", "mean_let_tree_walk_s",
    "mean_apply_let_s",
    "wait_fraction",
])


def parse_name(name: str):
    """Extrae group, N, P, factor del nombre del directorio."""
    m = re.match(r"(scale|sens|valid)_n(\d+)_p(\d+)_f(\d+p\d+)_p(\d+)$", name)
    if not m:
        return None
    group = m.group(1)
    N = int(m.group(2))
    # El P puede estar en el nombre del config (p3) o en el sufijo del directorio
    cfg_P = int(m.group(3))
    factor_str = m.group(4).replace("p", ".")
    factor = float(factor_str)
    dir_P = int(m.group(5))
    return group, N, dir_P, factor


def load_results():
    rows = []
    for run_dir in sorted(RESULTS_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        tf = run_dir / "timings.json"
        if not tf.exists():
            continue

        parsed = parse_name(run_dir.name)
        if parsed is None:
            continue
        group, N, P, factor = parsed

        try:
            t = json.loads(tf.read_text())
        except Exception as e:
            print(f"[WARN] No se pudo leer {tf}: {e}")
            continue

        hpc = t.get("hpc", {})
        wall_s = t.get("wall_time_s", 0.0)

        rows.append(Row(
            cfg_name=run_dir.name,
            group=group,
            N=N, P=P, factor=factor,
            wall_s=wall_s,
            mean_bytes_sent=hpc.get("mean_bytes_sent", 0.0),
            mean_bytes_recv=hpc.get("mean_bytes_recv", 0.0),
            mean_let_nodes_exported=hpc.get("mean_let_nodes_exported", 0.0),
            mean_let_nodes_imported=hpc.get("mean_let_nodes_imported", 0.0),
            mean_max_let_nodes_per_rank=hpc.get("mean_max_let_nodes_per_rank", 0.0),
            mean_local_tree_nodes=hpc.get("mean_local_tree_nodes", 0.0),
            mean_export_prune_ratio=hpc.get("mean_export_prune_ratio", 0.0),
            mean_let_alltoallv_s=hpc.get("mean_let_alltoallv_s", 0.0),
            mean_walk_local_s=hpc.get("mean_walk_local_s", 0.0),
            mean_let_tree_build_s=hpc.get("mean_let_tree_build_s", 0.0),
            mean_let_tree_walk_s=hpc.get("mean_let_tree_walk_s", 0.0),
            mean_apply_let_s=hpc.get("mean_apply_let_s", 0.0),
            wait_fraction=hpc.get("wait_fraction", 0.0),
        ))
    return rows


# ── Guardar CSV ───────────────────────────────────────────────────────────────

def save_csv(rows):
    out = OUT_DIR / "phase12_summary.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(Row._fields)
        for r in rows:
            w.writerow(r)
    print(f"CSV: {out}  ({len(rows)} filas)")
    return out


# ── Figuras SVG (matplotlib opcional) ────────────────────────────────────────

def try_plot(rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[INFO] matplotlib no disponible, saltando figuras.")
        return

    import numpy as np

    factors = sorted({r.factor for r in rows})
    FACTORS_LABELS = {f: (f"baseline (f=0)" if f == 0.0 else f"f={f:.1f}") for f in factors}
    colors = plt.cm.viridis(np.linspace(0.0, 0.9, len(factors)))
    fcolor = {f: c for f, c in zip(factors, colors)}

    # ── Figura 1: bytes/rank vs N (grupo scale, P=4) ─────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    for fac in sorted(factors):
        sub = sorted([r for r in rows if r.group == "scale" and r.P == 4 and r.factor == fac],
                     key=lambda r: r.N)
        if not sub:
            continue
        xs = [r.N for r in sub]
        ys = [(r.mean_bytes_sent + r.mean_bytes_recv) / 2 / 1024 for r in sub]
        ax.plot(xs, ys, marker="o", label=FACTORS_LABELS[fac], color=fcolor[fac])
    ax.set_xlabel("N partículas")
    ax.set_ylabel("bytes/rank (KB, media)")
    ax.set_title("Bytes LET por rank vs N  [P=4]")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "plot_bytes_vs_N.svg")
    plt.close(fig)

    # ── Figura 2: nodos exportados vs factor (sens, N=8000, P=4) ─────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    sub = sorted([r for r in rows if r.group == "sens"], key=lambda r: r.factor)
    if sub:
        xs = [r.factor for r in sub]
        ys = [r.mean_let_nodes_exported for r in sub]
        ax.plot(xs, ys, marker="o", color="steelblue")
    ax.set_xlabel("let_theta_export_factor")
    ax.set_ylabel("Nodos LET exportados (media)")
    ax.set_title("Nodos LET exportados vs factor  [N=8000, P=4]")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "plot_nodes_exported_vs_factor.svg")
    plt.close(fig)

    # ── Figura 3: wall time vs factor (sens, N=8000, P=4) ────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    if sub:
        xs = [r.factor for r in sub]
        ys = [r.wall_s for r in sub]
        ax.plot(xs, ys, marker="s", color="coral")
    ax.set_xlabel("let_theta_export_factor")
    ax.set_ylabel("Wall time (s)")
    ax.set_title("Wall time vs factor  [N=8000, P=4]")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "plot_wall_vs_factor.svg")
    plt.close(fig)

    # ── Figura 4: prune ratio vs factor ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    if sub:
        xs = [r.factor for r in sub]
        ys = [r.mean_export_prune_ratio for r in sub]
        ax.plot(xs, ys, marker="^", color="green")
    ax.set_xlabel("let_theta_export_factor")
    ax.set_ylabel("Export prune ratio")
    ax.set_title("Ratio de poda vs factor  [N=8000, P=4]")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "plot_prune_ratio_vs_factor.svg")
    plt.close(fig)

    # ── Figura 5: bytes/rank vs P (scale, N=8000) ────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    for fac in sorted(factors):
        sub2 = sorted([r for r in rows if r.group == "scale" and r.N == 8000 and r.factor == fac],
                      key=lambda r: r.P)
        if not sub2:
            continue
        xs = [r.P for r in sub2]
        ys = [(r.mean_bytes_sent + r.mean_bytes_recv) / 2 / 1024 for r in sub2]
        ax.plot(xs, ys, marker="o", label=FACTORS_LABELS[fac], color=fcolor[fac])
    ax.set_xlabel("P (ranks MPI)")
    ax.set_ylabel("bytes/rank (KB, media)")
    ax.set_title("Bytes LET por rank vs P  [N=8000]")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "plot_bytes_vs_P.svg")
    plt.close(fig)

    print("Figuras guardadas en", OUT_DIR)


# ── Tabla resumen en terminal ─────────────────────────────────────────────────

def print_summary(rows):
    print("\n=== Phase 12 — Resumen ===")
    print(f"{'Config':<40} {'N':>6} {'P':>3} {'factor':>6} "
          f"{'wall_s':>8} {'bytes_sent KB':>14} {'nodes_exp':>10} {'prune_r':>8}")
    print("-" * 100)
    for r in sorted(rows, key=lambda x: (x.group, x.N, x.P, x.factor)):
        print(f"{r.cfg_name:<40} {r.N:>6} {r.P:>3} {r.factor:>6.1f} "
              f"{r.wall_s:>8.2f} {r.mean_bytes_sent/1024:>14.1f} "
              f"{r.mean_let_nodes_exported:>10.0f} {r.mean_export_prune_ratio:>8.3f}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rows = load_results()
    if not rows:
        print("[INFO] No hay resultados todavía en", RESULTS_DIR)
        print("       Ejecuta run_phase12.sh primero.")
        sys.exit(0)

    save_csv(rows)
    print_summary(rows)
    try_plot(rows)
