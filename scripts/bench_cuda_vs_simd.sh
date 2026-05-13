#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
OUT_DIR="$REPO_ROOT/runs/benchmarks/cuda-vs-simd"
CSV="$OUT_DIR/cuda_vs_simd_direct.csv"
PNG="$OUT_DIR/cuda_vs_simd_direct.png"

mkdir -p "$OUT_DIR"
cd "$REPO_ROOT"

if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: python3 no encontrado; se necesita para generar CSV/PNG." >&2
    exit 1
fi

echo "Running CUDA vs SIMD Criterion benchmark..."
cargo bench -p gadget-ng-cuda --features simd --bench cuda_vs_simd

python3 - "$REPO_ROOT" "$CSV" "$PNG" <<'PY'
import csv
import json
import sys
from pathlib import Path

repo = Path(sys.argv[1])
csv_path = Path(sys.argv[2])
png_path = Path(sys.argv[3])
criterion_root = repo / "target" / "criterion" / "cuda_vs_simd_direct"

rows = []
for estimates in criterion_root.glob("*/**/new/estimates.json"):
    rel = estimates.relative_to(criterion_root).parts
    if len(rel) < 3:
        continue
    solver = rel[0]
    n = int(rel[1])
    with estimates.open("r", encoding="utf-8") as f:
        data = json.load(f)
    mean_ns = float(data["mean"]["point_estimate"])
    rows.append((n, solver, mean_ns / 1_000_000.0))

rows.sort(key=lambda r: (r[0], r[1]))
csv_path.parent.mkdir(parents=True, exist_ok=True)
with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["n", "solver", "mean_ms"])
    writer.writerows(rows)

try:
    import matplotlib.pyplot as plt
except Exception as exc:
    print(f"WARN: matplotlib no disponible ({exc}); CSV generado en {csv_path}")
    sys.exit(0)

by_solver = {}
for n, solver, mean_ms in rows:
    by_solver.setdefault(solver, []).append((n, mean_ms))

plt.figure(figsize=(7.5, 4.8))
for solver, points in sorted(by_solver.items()):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    plt.plot(xs, ys, marker="o", label=solver)

plt.xlabel("N particles")
plt.ylabel("Mean time [ms]")
plt.title("Direct gravity: CUDA vs SIMD")
plt.xscale("log", base=2)
plt.yscale("log")
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(png_path, dpi=160)
print(f"CSV: {csv_path}")
print(f"PNG: {png_path}")
PY
