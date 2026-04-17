#!/usr/bin/env python3
"""
Analiza los resultados de Fase 10 y genera tablas/figuras comparativas.

Produce:
  - phase10_summary.csv: tiempos por config
  - plot_apply_speedup.svg: speedup LetTree vs flat_let por N y P
  - plot_apply_breakdown.svg: desglose de tiempos (build+walk vs flat loop)
  - plot_wall_comparison.svg: wall time total flat vs tree
  - plot_let_nodes_vs_tree_nodes.svg: nodos LET importados vs nodos LetTree
"""

import json
import os
import glob
import csv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")
OUT_DIR = os.path.join(SCRIPT_DIR, "..", "plots")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Lectura de resultados ─────────────────────────────────────────────────────

def parse_run(run_dir):
    """Lee timings.json y wall_ms.txt de un directorio de resultados."""
    timings_path = os.path.join(run_dir, "timings.json")
    wall_path = os.path.join(run_dir, "wall_ms.txt")

    result = {
        "wall_ms": None,
        "mean_apply_let_s": None,
        "mean_let_tree_build_s": None,
        "mean_let_tree_walk_s": None,
        "mean_let_tree_nodes": None,
        "mean_let_nodes_imported": None,
        "mean_walk_local_s": None,
        "mean_tree_build_s": None,
    }

    if os.path.exists(wall_path):
        try:
            result["wall_ms"] = float(open(wall_path).read().strip())
        except Exception:
            pass

    if os.path.exists(timings_path):
        try:
            data = json.load(open(timings_path))
            hpc = data.get("hpc", {})
            for key in result:
                if key in hpc:
                    result[key] = hpc[key]
                elif key in data:
                    result[key] = data[key]
        except Exception as e:
            print(f"  WARNING: no se pudo parsear {timings_path}: {e}")

    return result


def parse_run_name(name):
    """Extrae (n, p, backend) del nombre 'n2000_p2_let_tree'."""
    parts = name.split("_")
    n = int(parts[0][1:])
    p = int(parts[1][1:])
    backend = "_".join(parts[2:])
    return n, p, backend


# ── Cargar todos los resultados ───────────────────────────────────────────────

rows = []
for run_dir in sorted(glob.glob(os.path.join(RESULTS_DIR, "n*_p*_*"))):
    name = os.path.basename(run_dir)
    try:
        n, p, backend = parse_run_name(name)
    except Exception:
        continue
    stats = parse_run(run_dir)
    rows.append({"name": name, "n": n, "p": p, "backend": backend, **stats})

if not rows:
    print("No se encontraron resultados en", RESULTS_DIR)
    print("Ejecuta primero: ./run_phase10.sh")
    exit(0)

# ── CSV ───────────────────────────────────────────────────────────────────────

csv_path = os.path.join(SCRIPT_DIR, "..", "phase10_summary.csv")
fieldnames = [
    "name", "n", "p", "backend",
    "wall_ms",
    "mean_apply_let_s",
    "mean_let_tree_build_s",
    "mean_let_tree_walk_s",
    "mean_let_tree_nodes",
    "mean_let_nodes_imported",
    "mean_walk_local_s",
    "mean_tree_build_s",
]
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
print(f"CSV escrito: {csv_path}")

# ── Figuras ───────────────────────────────────────────────────────────────────

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("matplotlib no disponible — saltando figuras")

if HAS_MPL:
    Ns = sorted(set(r["n"] for r in rows))
    Ps = sorted(set(r["p"] for r in rows))

    def get(n, p, backend, key):
        for r in rows:
            if r["n"] == n and r["p"] == p and r["backend"] == backend:
                return r.get(key)
        return None

    # ── Figura 1: speedup apply_let (flat vs tree) ──────────────────────────
    fig, axes = plt.subplots(1, len(Ps), figsize=(4 * len(Ps), 4), sharey=True)
    if len(Ps) == 1:
        axes = [axes]

    for ax, p in zip(axes, Ps):
        speedups = []
        ns_used = []
        for n in Ns:
            flat_t = get(n, p, "flat_let", "mean_apply_let_s")
            tree_t = get(n, p, "let_tree", "mean_apply_let_s")
            if flat_t and tree_t and flat_t > 0 and tree_t > 0:
                speedups.append(flat_t / tree_t)
                ns_used.append(n)
        if ns_used:
            ax.plot(ns_used, speedups, "o-", color="steelblue", linewidth=2, markersize=8)
            ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
            ax.set_title(f"P={p}")
            ax.set_xlabel("N")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_ylabel("Speedup apply_let (flat/tree)" if p == Ps[0] else "")
            ax.grid(True, alpha=0.3)

    fig.suptitle("Fase 10: Speedup LetTree vs flat LET loop", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "plot_apply_speedup.svg"))
    plt.close()
    print("Figura: plot_apply_speedup.svg")

    # ── Figura 2: desglose build+walk vs flat (P=1) ─────────────────────────
    p_ref = 1
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(Ns))
    width = 0.25
    flat_vals = [get(n, p_ref, "flat_let", "mean_apply_let_s") or 0 for n in Ns]
    build_vals = [get(n, p_ref, "let_tree", "mean_let_tree_build_s") or 0 for n in Ns]
    walk_vals = [get(n, p_ref, "let_tree", "mean_let_tree_walk_s") or 0 for n in Ns]
    ax.bar(x - width, flat_vals, width, label="flat loop (apply_let)", color="tomato")
    ax.bar(x, build_vals, width, label="LetTree build", color="steelblue")
    ax.bar(x, walk_vals, width, bottom=build_vals, label="LetTree walk", color="deepskyblue")
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in Ns])
    ax.set_xlabel("N")
    ax.set_ylabel("Tiempo medio por paso (s)")
    ax.set_title(f"Fase 10: Desglose apply_let, P={p_ref}")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "plot_apply_breakdown.svg"))
    plt.close()
    print("Figura: plot_apply_breakdown.svg")

    # ── Figura 3: wall time total flat vs tree ──────────────────────────────
    fig, axes = plt.subplots(1, len(Ps), figsize=(4 * len(Ps), 4), sharey=False)
    if len(Ps) == 1:
        axes = [axes]
    for ax, p in zip(axes, Ps):
        flat_ws = [get(n, p, "flat_let", "wall_ms") or 0 for n in Ns]
        tree_ws = [get(n, p, "let_tree", "wall_ms") or 0 for n in Ns]
        x = np.arange(len(Ns))
        w = 0.35
        ax.bar(x - w / 2, flat_ws, w, label="flat_let", color="tomato")
        ax.bar(x + w / 2, tree_ws, w, label="let_tree", color="steelblue")
        ax.set_xticks(x)
        ax.set_xticklabels([str(n) for n in Ns])
        ax.set_xlabel("N")
        ax.set_ylabel("Wall time (ms)" if p == Ps[0] else "")
        ax.set_title(f"P={p}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle("Fase 10: Wall time total", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "plot_wall_comparison.svg"))
    plt.close()
    print("Figura: plot_wall_comparison.svg")

    # ── Figura 4: nodos LET importados vs nodos LetTree ─────────────────────
    p_ref = 1
    fig, ax = plt.subplots(figsize=(6, 4))
    let_imported = [get(n, p_ref, "let_tree", "mean_let_nodes_imported") or 0 for n in Ns]
    let_tree_nodes = [get(n, p_ref, "let_tree", "mean_let_tree_nodes") or 0 for n in Ns]
    ax.plot(Ns, let_imported, "o-", label="nodos LET importados", color="tomato")
    ax.plot(Ns, let_tree_nodes, "s--", label="nodos LetTree", color="steelblue")
    ax.set_xlabel("N")
    ax.set_ylabel("Número de nodos (media por paso)")
    ax.set_title(f"Fase 10: LET importados vs LetTree, P={p_ref}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "plot_let_nodes_vs_tree_nodes.svg"))
    plt.close()
    print("Figura: plot_let_nodes_vs_tree_nodes.svg")

# ── Tabla de texto ────────────────────────────────────────────────────────────
print("\n=== Resumen ===")
print(f"{'Config':<25} {'wall(ms)':>10} {'apply_flat(ms)':>16} {'ltree_walk(ms)':>15} {'speedup':>8}")
print("-" * 78)
for r in sorted(rows, key=lambda x: (x["n"], x["p"], x["backend"])):
    wall = r["wall_ms"] or 0
    apply_flat = (r.get("mean_apply_let_s") or 0) * 1000
    ltw = (r.get("mean_let_tree_walk_s") or 0) * 1000
    ltb = (r.get("mean_let_tree_build_s") or 0) * 1000
    if r["backend"] == "flat_let":
        sp = "-"
    else:
        flat_ref = get(r["n"], r["p"], "flat_let", "mean_apply_let_s") or 0
        tree_t = (r.get("mean_let_tree_build_s") or 0) + (r.get("mean_let_tree_walk_s") or 0)
        sp = f"{flat_ref / tree_t:.2f}x" if tree_t > 0 and flat_ref > 0 else "N/A"
    print(f"{r['name']:<25} {wall:>10.0f} {apply_flat:>16.3f} {ltw:>15.3f} {sp:>8}")
