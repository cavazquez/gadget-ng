#!/usr/bin/env bash
set -euo pipefail
trap 'echo "ERROR at line $LINENO"' ERR
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Alinear con CI (.github/workflows/ci.yml): warnings de rustc/clippy = error.
export RUSTFLAGS="-D warnings"

# Tests MPI multirank (mismo conjunto que job «mpi-multirank» en ci.yml).
MPI_MULTIRANK_TESTS=(
    sfc_hardening
    let_validation
    treepm_distributed
    v2_hierarchical_cosmo
    phase43_dt_treepm_parallel
)

run_mpi_multirank() {
    local ranks=$1
    local test bin
    for test in "${MPI_MULTIRANK_TESTS[@]}"; do
        bin=$(find target/debug/deps -name "${test}-*" -executable 2>/dev/null | head -1 || true)
        if [[ -n "${bin:-}" ]]; then
            mpirun --oversubscribe -n "$ranks" "$bin" --test-threads=1
        fi
    done
}

echo "== cargo fmt =="
cargo fmt --all -- --check

echo "== MSRV check =="
# MSRV 1.95: rust-toolchain.toml pin 1.95.0; CI job msrv usa dtolnay/rust-toolchain@1.95.0.
cargo check --workspace

echo "== cargo clippy =="
cargo clippy --workspace -- -D warnings

echo "== cargo clippy (all-targets) =="
cargo clippy --workspace --all-targets -- -D warnings

echo "== cargo test (workspace) =="
cargo test --workspace

echo "== cargo test (doc) =="
cargo test --workspace --doc

echo "== cargo doc (no-deps) =="
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps

echo "== cargo build release (MPI) =="
cargo build --release -p gadget-ng-cli --features mpi

echo "== validate TOML (examples + configs listadas) =="
GADGET_NG_BIN="$ROOT/target/release/gadget-ng" ./scripts/validate_example_configs.sh

echo "== tests MPI (feature mpi, serial runtime) =="
cargo test -p gadget-ng-parallel --features mpi
cargo test -p gadget-ng-physics --features mpi

echo "== compilar tests MPI (sin ejecutar) =="
cargo test -p gadget-ng-parallel --features mpi --no-run
cargo test -p gadget-ng-physics --features mpi --no-run

echo "== smoke MPI multirank (2 ranks) =="
run_mpi_multirank 2

echo "== smoke MPI multirank (4 ranks) =="
run_mpi_multirank 4

echo "check.sh: OK"
