#!/usr/bin/env python3
"""
Analiza resultados de Fase 11 y genera tablas + figuras.

Produce:
  - phase11_summary.csv
  - plot_wall_vs_N.svg          wall time vs N (flat_let vs let_tree)
  - plot_speedup_vs_P.svg        speedup let_tree/flat_let vs P
  - plot_phase_breakdown.svg     fracciones de tiempo por fase (N=8000, P=2)
  - plot_sensitivity_threshold.svg  sensibilidad al umbral threshold
  - plot_sensitivity_leaf_max.svg   sensibilidad al leaf_max
"""
import json
import os
import glob
import csv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")
OUT_DIR = os.path.join(SCRIPT_DIR, "..", "plots")
os.makedirs(OUT_DIR, exist_ok=True)


def parse_run(run_dir):
    timings_path = os.path.join(run_dir, "timings.json")
    wall_path = os.path.join(run_dir, "wall_ms.txt")
    result = {k: None for k in [
        "wall_ms", "mean_apply_let_s", "mean_let_tree_build_s",
        "mean_let_tree_walk_s", "mean_let_tree_nodes", "mean_let_nodes_imported",
        "mean_walk_local_s", "mean_tree_build_s", "mean_bytes_sent",
        "mean_bytes_recv", "let_tree_parallel", "wait_fraction",
    ]}
    if os.path.exists(wall_path):
        try:
            result["wall_ms"] = float(open(wall_path).read().strip())
        except Exception:
            pass
    if os.path.exists(timings_path):
        try:
            data = json.load(open(timings_path))
            hpc = data.get("hpc", {})
            for k in result:
                if k in hpc:
                    result[k] = hpc[k]
                elif k in data:
                    result[k] = data[k]
        except Exception as e:
            print(f"  WARNING: {timings_path}: {e}")
    return result


def parse_name(name):
    """Devuelve (kind, n, p, backend, threshold, leaf_max) desde el nombre."""
    import re
    if name.startswith("bench_"):
        m = re.match(r"bench_n(\d+)_p(\d+)_(flat_let|let_tree)", name)
        if m:
            return "bench", int(m.group(1)), int(m.group(2)), m.group(3), 64, 8
    if name.startswith("valid_"):
        m = re.match(r"valid_n(\d+)_p(\d+)_(flat_let|let_tree)", name)
        if m:
            return "valid", int(m.group(1)), int(m.group(2)), m.group(3), 64, 8
    if name.startswith("sens_threshold_"):
        m = re.match(r"sens_threshold_(\d+)", name)
        if m:
            return "sensitivity", 8000, 2, "let_tree", int(m.group(1)), 8
    if name.startswith("sens_leafmax_"):
        m = re.match(r"sens_leafmax_(\d+)", name)
        if m:
            return "sensitivity", 8000, 2, "let_tree", 64, int(m.group(1))
    return None, None, None, name, 64, 8


# ── Cargar resultados ─────────────────────────────────────────────────────────
rows = []
for run_dir in sorted(glob.glob(os.path.join(RESULTS_DIR, "*"))):
    if not os.path.isdir(run_dir):
        continue
    name = os.path.basename(run_dir)
    kind, n, p, backend, threshold, leaf_max = parse_name(name)
    if kind is None:
        continue
    stats = parse_run(run_dir)
    rows.append({
        "name": name, "kind": kind, "n": n, "p": p,
        "backend": backend, "threshold": threshold, "leaf_max": leaf_max,
        **stats,
    })

if not rows:
    print(f"No se encontraron resultados en {RESULTS_DIR}")
    print("Ejecuta primero: ./run_phase11.sh")
    exit(0)

# ── CSV ───────────────────────────────────────────────────────────────────────
fields = [
    "name", "kind", "n", "p", "backend", "threshold", "leaf_max",
    "wall_ms", "mean_apply_let_s", "mean_let_tree_build_s",
    "mean_let_tree_walk_s", "mean_let_tree_nodes", "mean_let_nodes_imported",
    "mean_walk_local_s", "mean_bytes_sent", "mean_bytes_recv",
    "let_tree_parallel", "wait_fraction",
]
csv_path = os.path.join(SCRIPT_DIR, "..", "phase11_summary.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
print(f"CSV: {csv_path}")


def get(rows_, kind, n, p, backend, key, threshold=64, leaf_max=8):
    for r in rows_:
        if (r["kind"] == kind and r["n"] == n and r["p"] == p
                and r["backend"] == backend
                and r.get("threshold", 64) == threshold
                and r.get("leaf_max", 8) == leaf_max):
            return r.get(key)
    return None


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
    bench_rows = [r for r in rows if r["kind"] == "bench"]
    Ns = sorted(set(r["n"] for r in bench_rows))
    Ps = sorted(set(r["p"] for r in bench_rows))

    # ── Fig 1: wall time vs N ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, len(Ps), figsize=(3.5 * len(Ps), 4), sharey=False)
    if len(Ps) == 1:
        axes = [axes]
    for ax, p in zip(axes, Ps):
        flat_ws = [get(bench_rows, "bench", n, p, "flat_let", "wall_ms") or 0 for n in Ns]
        tree_ws = [get(bench_rows, "bench", n, p, "let_tree", "wall_ms") or 0 for n in Ns]
        x = np.arange(len(Ns))
        w = 0.35
        ax.bar(x - w/2, flat_ws, w, label="flat_let", color="tomato")
        ax.bar(x + w/2, tree_ws, w, label="let_tree", color="steelblue")
        ax.set_xticks(x)
        ax.set_xticklabels([str(n) for n in Ns], rotation=30, fontsize=8)
        ax.set_xlabel("N")
        ax.set_ylabel("Wall time (ms)" if p == Ps[0] else "")
        ax.set_title(f"P={p}")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle("Fase 11: Wall time total flat_let vs let_tree", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "plot_wall_vs_N.svg"))
    plt.close()
    print("Figura: plot_wall_vs_N.svg")

    # ── Fig 2: speedup vs P ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, len(Ns), figsize=(3.5 * len(Ns), 4), sharey=True)
    if len(Ns) == 1:
        axes = [axes]
    for ax, n in zip(axes, Ns):
        speedups = []
        ps_used = []
        for p in Ps:
            flat_t = get(bench_rows, "bench", n, p, "flat_let", "mean_apply_let_s")
            tree_t_b = get(bench_rows, "bench", n, p, "let_tree", "mean_let_tree_build_s") or 0
            tree_t_w = get(bench_rows, "bench", n, p, "let_tree", "mean_let_tree_walk_s") or 0
            tree_t = tree_t_b + tree_t_w
            if flat_t and tree_t > 0:
                speedups.append(flat_t / tree_t)
                ps_used.append(p)
        if ps_used:
            ax.plot(ps_used, speedups, "o-", color="steelblue", linewidth=2, markersize=8)
            ax.axhline(1.0, color="gray", linestyle="--")
            ax.set_title(f"N={n}")
            ax.set_xlabel("P")
            ax.set_ylabel("Speedup apply_let" if n == Ns[0] else "")
            ax.grid(True, alpha=0.3)
    fig.suptitle("Fase 11: Speedup apply_let (flat/tree) vs P", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "plot_speedup_vs_P.svg"))
    plt.close()
    print("Figura: plot_speedup_vs_P.svg")

    # ── Fig 3: desglose por fases (N=8000, P=2) ─────────────────────────────
    n_ref, p_ref = 8000, 2
    flat_r = next((r for r in bench_rows if r["n"]==n_ref and r["p"]==p_ref and r["backend"]=="flat_let"), {})
    tree_r = next((r for r in bench_rows if r["n"]==n_ref and r["p"]==p_ref and r["backend"]=="let_tree"), {})
    keys = ["mean_tree_build_s", "mean_walk_local_s", "mean_let_tree_build_s",
            "mean_let_tree_walk_s", "mean_apply_let_s", "wait_fraction"]
    labels = ["tree_build", "walk_local", "ltree_build", "ltree_walk", "apply_let_flat", "mpi_wait"]
    flat_vals = [(flat_r.get(k) or 0) * 1000 for k in keys]
    tree_vals = [(tree_r.get(k) or 0) * 1000 for k in keys]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 4))
    w = 0.35
    ax.bar(x - w/2, flat_vals, w, label="flat_let", color="tomato")
    ax.bar(x + w/2, tree_vals, w, label="let_tree", color="steelblue")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Tiempo medio por paso (ms)")
    ax.set_title(f"Desglose por fase — N={n_ref}, P={p_ref}")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "plot_phase_breakdown.svg"))
    plt.close()
    print("Figura: plot_phase_breakdown.svg")

    # ── Fig 4: sensibilidad threshold ────────────────────────────────────────
    sens_thr_rows = [r for r in rows if r["kind"] == "sensitivity" and r.get("leaf_max") == 8]
    if sens_thr_rows:
        thresholds = sorted(set(r["threshold"] for r in sens_thr_rows))
        walk_ns = [(next((r.get("mean_let_tree_walk_s") or 0
                          for r in sens_thr_rows if r["threshold"] == t), 0) * 1000)
                   for t in thresholds]
        build_ns = [(next((r.get("mean_let_tree_build_s") or 0
                           for r in sens_thr_rows if r["threshold"] == t), 0) * 1000)
                    for t in thresholds]
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(thresholds, walk_ns, "o-", label="walk (ms)", color="steelblue")
        ax.plot(thresholds, build_ns, "s--", label="build (ms)", color="tomato")
        ax.set_xlabel("let_tree_threshold")
        ax.set_ylabel("Tiempo medio por paso (ms)")
        ax.set_title("Sensibilidad al threshold (N=8000, P=2)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "plot_sensitivity_threshold.svg"))
        plt.close()
        print("Figura: plot_sensitivity_threshold.svg")

    # ── Fig 5: sensibilidad leaf_max ─────────────────────────────────────────
    sens_lm_rows = [r for r in rows if r["kind"] == "sensitivity" and r.get("threshold") == 64]
    if sens_lm_rows:
        lms = sorted(set(r["leaf_max"] for r in sens_lm_rows))
        walk_ns = [(next((r.get("mean_let_tree_walk_s") or 0
                          for r in sens_lm_rows if r["leaf_max"] == lm), 0) * 1000)
                   for lm in lms]
        build_ns = [(next((r.get("mean_let_tree_build_s") or 0
                           for r in sens_lm_rows if r["leaf_max"] == lm), 0) * 1000)
                    for lm in lms]
        tree_nodes = [(next((r.get("mean_let_tree_nodes") or 0
                             for r in sens_lm_rows if r["leaf_max"] == lm), 0))
                      for lm in lms]
        fig, ax1 = plt.subplots(figsize=(5, 4))
        ax1.plot(lms, walk_ns, "o-", label="walk (ms)", color="steelblue")
        ax1.plot(lms, build_ns, "s--", label="build (ms)", color="tomato")
        ax1.set_xlabel("let_tree_leaf_max")
        ax1.set_ylabel("Tiempo medio por paso (ms)")
        ax1.set_title("Sensibilidad al leaf_max (N=8000, P=2)")
        ax2 = ax1.twinx()
        ax2.plot(lms, tree_nodes, "^:", label="nodos", color="green", alpha=0.6)
        ax2.set_ylabel("Nodos LetTree")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
        ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "plot_sensitivity_leaf_max.svg"))
        plt.close()
        print("Figura: plot_sensitivity_leaf_max.svg")

# ── Resumen de texto ──────────────────────────────────────────────────────────
print("\n=== Resumen benchmarks principales ===")
print(f"{'Config':<30} {'wall(ms)':>10} {'apply(ms)':>10} {'ltree(ms)':>10} {'speedup':>8} {'parallel':>9}")
print("-" * 84)
bench_rows_sorted = sorted(
    [r for r in rows if r["kind"] == "bench"],
    key=lambda x: (x["n"], x["p"], x["backend"])
)
for r in bench_rows_sorted:
    wall = r["wall_ms"] or 0
    apply_ms = (r.get("mean_apply_let_s") or 0) * 1000
    ltb = (r.get("mean_let_tree_build_s") or 0) * 1000
    ltw = (r.get("mean_let_tree_walk_s") or 0) * 1000
    lt_total = ltb + ltw
    if r["backend"] == "flat_let":
        sp_str = "-"
    else:
        flat_ref = get(rows, "bench", r["n"], r["p"], "flat_let", "mean_apply_let_s") or 0
        sp_str = f"{flat_ref / (lt_total/1000):.2f}x" if lt_total > 0 and flat_ref > 0 else "N/A"
    par = str(r.get("let_tree_parallel") or "-")
    print(f"{r['name']:<30} {wall:>10.0f} {apply_ms:>10.2f} {lt_total:>10.2f} {sp_str:>8} {par:>9}")
