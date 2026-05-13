#!/usr/bin/env python3
"""
Ejecuta el benchmark Criterion `direct_gravity` en tres configuraciones de features,
fusiona tiempos (cada corrida Criterion sobrescribe grupos) y escribe CSV + un PNG comparativo.

Uso desde la raíz del repo:
  python3 scripts/bench_direct_cpu_plot.py
  ./scripts/bench_direct_cpu_plot.sh
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def clear_direct_cpu_criterion(criterion_dir: Path) -> None:
    if not criterion_dir.is_dir():
        return
    for p in criterion_dir.iterdir():
        if p.is_dir() and p.name.startswith("direct_cpu_"):
            shutil.rmtree(p, ignore_errors=True)


def collect_mean_ms(criterion_dir: Path) -> dict[tuple[str, int], float]:
    """(grupo, n_partículas) -> tiempo medio en ms."""
    out: dict[tuple[str, int], float] = {}
    if not criterion_dir.is_dir():
        return out
    for est in criterion_dir.glob("direct_cpu_*/**/new/estimates.json"):
        rel = est.relative_to(criterion_dir).parts
        if len(rel) < 4 or rel[-2] != "new" or rel[-1] != "estimates.json":
            continue
        group, n_str = rel[0], rel[1]
        try:
            n = int(n_str)
        except ValueError:
            continue
        with est.open(encoding="utf-8") as f:
            data = json.load(f)
        mean_ns = float(data["mean"]["point_estimate"])
        out[(group, n)] = mean_ns / 1e6
    return out


def merge_unique(
    merged: dict[tuple[str, int], float], snap: dict[tuple[str, int], float]
) -> None:
    for k, v in snap.items():
        merged[k] = v


LABELS: dict[str, str] = {
    "direct_cpu_serial": "CPU serial (AoS)",
    "direct_cpu_rayon_scalar_inner": "Rayon (interno escalar)",
    "direct_cpu_simd_serial_runtime": "SIMD 1 hilo (runtime)",
    "direct_cpu_simd_serial_avx2": "SIMD 1 hilo (AVX2)",
    "direct_cpu_simd_serial_avx512": "SIMD 1 hilo (AVX-512)",
    "direct_cpu_rayon_simd_runtime": "Rayon + SIMD (runtime)",
    "direct_cpu_rayon_simd_avx2": "Rayon + SIMD (AVX2)",
    "direct_cpu_rayon_simd_avx512": "Rayon + SIMD (AVX-512)",
}


def pretty_label(group: str) -> str:
    return LABELS.get(group, group.removeprefix("direct_cpu_").replace("_", " "))


def run_cargo_bench(repo: Path, features: str | None) -> None:
    cmd = ["cargo", "bench", "-p", "gadget-ng-core", "--bench", "direct_gravity"]
    if features:
        cmd.extend(["--features", features])
    cmd.extend(["--", "--quick"])
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=repo, check=True)


def write_csv(path: Path, rows: list[tuple[str, int, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        f.write("group,n,mean_ms,label\n")
        for g, n, ms in rows:
            f.write(f"{g},{n},{ms:.6f},{pretty_label(g)}\n")


def plot_png(path: Path, merged: dict[tuple[str, int], float]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"AVISO: sin matplotlib ({exc}); sólo CSV.", file=sys.stderr)
        return

    ns = sorted({n for (_, n) in merged})
    groups = sorted({g for (g, _) in merged})

    fig, axes = plt.subplots(1, len(ns), figsize=(4.2 * len(ns), 6.0), squeeze=False)
    for ax, n in zip(axes[0], ns):
        chunk = [(g, merged[(g, n)]) for g in groups if (g, n) in merged]
        chunk.sort(key=lambda x: x[1])
        labels = [pretty_label(g) for g, _ in chunk]
        times = [t for _, t in chunk]
        ax.barh(labels, times, color="steelblue", alpha=0.85)
        ax.set_xlabel("Tiempo medio [ms]")
        ax.set_title(f"N = {n}")
        ax.grid(True, axis="x", alpha=0.3)
    fig.suptitle("Gravedad directa: comparación de implementaciones CPU (Criterion, --quick)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    repo = repo_root()
    out_dir = repo / "runs" / "benchmarks" / "direct-cpu"
    csv_path = out_dir / "direct_gravity_mean_times.csv"
    png_path = out_dir / "direct_gravity_mean_times.png"
    criterion = repo / "target" / "criterion"

    clear_direct_cpu_criterion(criterion)
    merged: dict[tuple[str, int], float] = {}

    run_cargo_bench(repo, None)
    merge_unique(merged, collect_mean_ms(criterion))

    run_cargo_bench(repo, "rayon")
    merge_unique(merged, collect_mean_ms(criterion))

    run_cargo_bench(repo, "simd,rayon")
    merge_unique(merged, collect_mean_ms(criterion))

    rows = sorted((g, n, merged[(g, n)]) for (g, n) in merged)
    write_csv(csv_path, rows)
    plot_png(png_path, merged)
    print(f"CSV: {csv_path}")
    print(f"PNG: {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
