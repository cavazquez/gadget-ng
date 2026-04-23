#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "== cargo fmt =="
cargo fmt --all -- --check

echo "== cargo clippy =="
cargo clippy --workspace -- -D warnings

echo "== cargo test (workspace) =="
cargo test --workspace

echo "== cargo build release (MPI) =="
cargo build --release -p gadget-ng-cli --features mpi

echo "== tests MPI (feature mpi, serial runtime) =="
cargo test -p gadget-ng-parallel --features mpi

echo "== smoke MPI multirank (2 ranks) =="
cargo test -p gadget-ng-parallel --features mpi --no-run 2>/dev/null
BIN=$(find target/debug/deps -name "sfc_hardening-*" -executable 2>/dev/null | head -1)
if [ -n "$BIN" ]; then
    mpirun --oversubscribe -n 2 "$BIN" --test-threads=1
fi
BIN=$(find target/debug/deps -name "let_validation-*" -executable 2>/dev/null | head -1)
if [ -n "$BIN" ]; then
    mpirun --oversubscribe -n 2 "$BIN" --test-threads=1
fi

echo "check.sh: OK"
