#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

CFG="${1:-$ROOT/experiments/nbody/mvp_smoke/config/parity.toml}"
OUT_BASE="${2:-$ROOT/experiments/nbody/mvp_smoke/runs/parity_cmp}"

cargo build
cargo build --features mpi

S_OUT="$OUT_BASE/serial"
M_OUT="$OUT_BASE/mpi"
rm -rf "$S_OUT" "$M_OUT"
mkdir -p "$S_OUT" "$M_OUT"

target/debug/gadget-ng stepping --config "$CFG" --out "$S_OUT" --snapshot
MPIRUN="${MPIRUN:-mpiexec}"
$MPIRUN -n 4 target/debug/gadget-ng stepping --config "$CFG" --out "$M_OUT" --snapshot

python3 - "$OUT_BASE" <<'PY'
import json, pathlib, sys

def load_particles(path):
    rows = []
    for line in pathlib.Path(path).read_text().splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    rows.sort(key=lambda r: r["global_id"])
    return rows

def max_diff(a, b, keys):
    m = 0.0
    for pa, pb in zip(a, b):
        assert pa["global_id"] == pb["global_id"]
        for k in keys:
            m = max(m, abs(float(pa[k]) - float(pb[k])))
    return m

root = pathlib.Path(sys.argv[1])
s = load_particles(root / "serial/snapshot_final/particles.jsonl")
m = load_particles(root / "mpi/snapshot_final/particles.jsonl")
keys = ["px", "py", "pz", "vx", "vy", "vz", "mass"]
d = max_diff(s, m, keys)
print("max_abs_diff", d)
tol = 1e-12
if d > tol:
    print("FAIL: exceeds tolerance", tol)
    sys.exit(1)
print("OK serial vs MPI within", tol)
PY

echo "compare_serial_mpi.sh: OK"
