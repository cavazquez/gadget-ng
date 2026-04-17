#!/usr/bin/env python3
"""
Analiza resultados de benchmarks Fase 8 (strong/weak scaling).

Lee wall_seconds.txt y diagnostics.jsonl de cada run y produce:
  - phase8_summary.csv: tabla con N, P, backend, wall_s, comm_frac, energy_drift
  - plots/: gráficos de scaling

Uso:
  python3 analyze_scaling.py [--results-dir <dir>] [--plots-dir <dir>]
"""
import os, sys, json, csv, re, argparse
from pathlib import Path
from collections import defaultdict


def parse_tag(tag: str):
    """Extrae (tipo, N, backend, ranks) del nombre del directorio."""
    m = re.match(r"(strong|weak)_N(\d+)_(allgather|sfc_let)_R(\d+)", tag)
    if not m:
        return None
    return {
        "tipo": m.group(1),
        "N": int(m.group(2)),
        "backend": m.group(3),
        "ranks": int(m.group(4)),
    }


def load_diagnostics(diag_path: Path):
    """Carga el último paso del diagnostics.jsonl."""
    if not diag_path.exists():
        return {}
    rows = []
    with open(diag_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    if not rows:
        return {}
    last = rows[-1]
    first = rows[0]
    e0 = first.get("total_energy", None)
    ef = last.get("total_energy", None)
    drift = abs(ef - e0) / abs(e0) if (e0 and ef and abs(e0) > 1e-300) else None
    comm_frac = last.get("comm_fraction", None)
    return {"energy_drift": drift, "comm_frac": comm_frac}


def main():
    parser = argparse.ArgumentParser()
    base = Path(__file__).parent.parent
    parser.add_argument("--results-dir", default=str(base / "results"))
    parser.add_argument("--plots-dir", default=str(base / "plots"))
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for run_dir in sorted(results_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        meta = parse_tag(run_dir.name)
        if not meta:
            continue

        wall_file = run_dir / "wall_seconds.txt"
        wall_s = None
        if wall_file.exists():
            try:
                wall_s = float(wall_file.read_text().strip())
            except ValueError:
                pass

        diag = load_diagnostics(run_dir / "diagnostics.jsonl")
        timings_path = run_dir / "timings.json"
        timings = {}
        if timings_path.exists():
            try:
                timings = json.loads(timings_path.read_text())
            except Exception:
                pass

        row = {
            "tipo": meta["tipo"],
            "N": meta["N"],
            "backend": meta["backend"],
            "ranks": meta["ranks"],
            "wall_s": wall_s or timings.get("total_wall_s"),
            "comm_frac": diag.get("comm_frac") or timings.get("comm_fraction"),
            "energy_drift": diag.get("energy_drift"),
        }
        rows.append(row)
        print(f"  {run_dir.name}: wall={row['wall_s']:.1f}s, "
              f"drift={row['energy_drift']:.3e}" if row['energy_drift'] else
              f"  {run_dir.name}: wall={row['wall_s']}")

    if not rows:
        print("No se encontraron resultados en:", results_dir)
        return

    # Guardar CSV.
    csv_path = results_dir.parent / "phase8_summary.csv"
    fieldnames = ["tipo", "N", "backend", "ranks", "wall_s", "comm_frac", "energy_drift"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResumen guardado en: {csv_path}")

    # Calcular eficiencias de scaling.
    by_tipo = defaultdict(list)
    for r in rows:
        by_tipo[r["tipo"]].append(r)

    print("\n--- Strong Scaling ---")
    by_backend_strong = defaultdict(list)
    for r in by_tipo["strong"]:
        by_backend_strong[r["backend"]].append(r)
    for backend, runs in by_backend_strong.items():
        runs.sort(key=lambda x: x["ranks"])
        base_run = next((r for r in runs if r["ranks"] == 1), None)
        if base_run and base_run["wall_s"]:
            print(f"  {backend}:")
            for r in runs:
                if r["wall_s"]:
                    speedup = base_run["wall_s"] / r["wall_s"]
                    eff = speedup / r["ranks"] * 100
                    print(f"    P={r['ranks']:2d}: wall={r['wall_s']:.1f}s  speedup={speedup:.2f}×  eff={eff:.1f}%")

    print("\n--- Weak Scaling ---")
    by_backend_weak = defaultdict(list)
    for r in by_tipo["weak"]:
        by_backend_weak[r["backend"]].append(r)
    for backend, runs in by_backend_weak.items():
        runs.sort(key=lambda x: x["ranks"])
        base_run = next((r for r in runs if r["ranks"] == 1), None)
        if base_run and base_run["wall_s"]:
            print(f"  {backend}:")
            for r in runs:
                if r["wall_s"]:
                    eff = base_run["wall_s"] / r["wall_s"] * 100
                    print(f"    P={r['ranks']:2d} N={r['N']:5d}: wall={r['wall_s']:.1f}s  eff={eff:.1f}%")

    # Generar plots si matplotlib disponible.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        generate_plots(by_tipo, plots_dir)
        print(f"\nPlots guardados en: {plots_dir}")
    except ImportError:
        print("\nmatplotlib no disponible; omitiendo gráficos.")


def generate_plots(by_tipo, plots_dir: Path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = {"allgather": "tab:red", "sfc_let": "tab:blue"}

    # Strong scaling.
    ax = axes[0]
    by_backend = defaultdict(list)
    for r in by_tipo["strong"]:
        by_backend[r["backend"]].append(r)
    for backend, runs in by_backend.items():
        runs.sort(key=lambda x: x["ranks"])
        base = next((r for r in runs if r["ranks"] == 1), None)
        if base and base["wall_s"]:
            rs = [r["ranks"] for r in runs if r["wall_s"]]
            eff = [base["wall_s"] / r["wall_s"] / r["ranks"] * 100 for r in runs if r["wall_s"]]
            ax.plot(rs, eff, "o-", label=backend, color=colors.get(backend, "gray"))
    ideal_rs = [1, 2, 4, 8]
    ax.plot(ideal_rs, [100] * len(ideal_rs), "--k", label="ideal")
    ax.set_xlabel("Rangos MPI")
    ax.set_ylabel("Eficiencia de scaling (%)")
    ax.set_title("Strong Scaling (N=2000)")
    ax.legend()
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.3)

    # Weak scaling.
    ax = axes[1]
    by_backend = defaultdict(list)
    for r in by_tipo["weak"]:
        by_backend[r["backend"]].append(r)
    for backend, runs in by_backend.items():
        runs.sort(key=lambda x: x["ranks"])
        base = next((r for r in runs if r["ranks"] == 1), None)
        if base and base["wall_s"]:
            rs = [r["ranks"] for r in runs if r["wall_s"]]
            eff = [base["wall_s"] / r["wall_s"] * 100 for r in runs if r["wall_s"]]
            ax.plot(rs, eff, "o-", label=backend, color=colors.get(backend, "gray"))
    ax.plot(ideal_rs, [100] * len(ideal_rs), "--k", label="ideal")
    ax.set_xlabel("Rangos MPI")
    ax.set_ylabel("Eficiencia de weak scaling (%)")
    ax.set_title("Weak Scaling (N/P = 500)")
    ax.legend()
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plots_dir / "phase8_scaling.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
