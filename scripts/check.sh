#!/usr/bin/env bash
set -euo pipefail
trap 'echo "ERROR at line $LINENO"' ERR
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Alinear con CI (.github/workflows/ci.yml): warnings de rustc/clippy = error.
export RUSTFLAGS="-D warnings"

echo "== cargo fmt =="
cargo fmt --all -- --check

echo "== MSRV check =="
rustup run 1.85 cargo check --workspace

echo "== cargo clippy =="
# Igual que job «Clippy» bloqueante en .github/workflows/ci.yml.
cargo clippy --workspace -- -D warnings

echo "== cargo test (workspace) =="
cargo test --workspace

echo "== cargo test (doc) =="
cargo test --workspace --doc

echo "== cargo doc (no-deps) =="
cargo doc --workspace --no-deps

echo "== cargo build release (MPI) =="
cargo build --release -p gadget-ng-cli --features mpi

echo "== validate TOML (examples + configs listadas) =="
GADGET_NG_BIN="$ROOT/target/release/gadget-ng" ./scripts/validate_example_configs.sh

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
