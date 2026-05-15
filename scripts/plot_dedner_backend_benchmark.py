#!/usr/bin/env python3
"""Genera un gráfico de barras con los cuatro backends de Dedner (Criterion).

Uso (tras `cargo bench -p gadget-ng-mhd --bench dedner_backend_bench --features bench-all-dedner-paths`):

    python3 scripts/plot_dedner_backend_benchmark.py
    python3 scripts/plot_dedner_backend_benchmark.py --n 256 --out runs/benchmarks/dedner_backends_N256.png

Requiere matplotlib (`pip install matplotlib`).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CRIT = ROOT / "target" / "criterion" / "dedner_cleaning_backends"

BACKENDS = [
    ("cpu_sin_rayon_scalar", "CPU sin Rayon (escalar)"),
    ("cpu_con_rayon", "CPU con Rayon"),
    ("simd_sin_rayon_avx2", "SIMD sin Rayon AVX2"),
    ("simd_sin_rayon_avx512", "SIMD sin Rayon AVX-512"),
]


def load_mean_ns(crit_dir: Path, backend: str, n: int) -> float:
    sub = crit_dir / backend / str(n)
    for leaf in ("new", "base"):
        est = sub / leaf / "estimates.json"
        if est.is_file():
            data = json.loads(est.read_text())
            return float(data["mean"]["point_estimate"])
    raise FileNotFoundError(f"No se encontró estimates.json bajo {sub}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--criterion-dir",
        type=Path,
        default=DEFAULT_CRIT,
        help="Directorio criterion/dedner_cleaning_backends",
    )
    ap.add_argument("--n", type=int, default=1024, help="Tamaño N (256 o 1024 según el bench)")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="PNG de salida (por defecto runs/benchmarks/dedner_backends_N{n}.png)",
    )
    args = ap.parse_args()
    crit: Path = args.criterion_dir
    n: int = args.n
    out = args.out or (ROOT / "runs" / "benchmarks" / f"dedner_backends_N{n}.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "matplotlib no está instalado. Ej.: pip install matplotlib",
            file=sys.stderr,
        )
        return 1

    labels: list[str] = []
    means_us: list[float] = []
    for key, label in BACKENDS:
        ns = load_mean_ns(crit, key, n)
        labels.append(label)
        means_us.append(ns / 1000.0)

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759"]
    bars = ax.bar(labels, means_us, color=colors)
    ax.set_ylabel("Tiempo medio (µs)")
    ax.set_title(f"dedner_cleaning_step — N = {n} (Criterion mean)")
    ax.bar_label(bars, fmt="{:.1f}", padding=3)
    plt.xticks(rotation=15, ha="right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Escrito {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
