#!/usr/bin/env bash
# check_release.sh — verificaciones completas en modo release.
# Incluye Phase 37 (PM, sin TREEPM) y opcionalmente Phase 41 (sin N=256).
# Uso:
#   bash scripts/check_release.sh              # suite completa
#   SKIP_PHASE37=1 bash scripts/check_release.sh  # omitir Phase 37
#   RUN_PHASE41=1  bash scripts/check_release.sh  # incluir Phase 41 (~3-4 h)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "== cargo fmt =="
cargo fmt --all

echo "== cargo clippy =="
cargo clippy --workspace -- -D warnings

echo "== cargo test --release (workspace) =="
cargo test --workspace --release

echo "== Phase 37 PM completo (sin TREEPM, ~6 min) =="
if [ "${SKIP_PHASE37:-0}" != "1" ]; then
    cargo test -p gadget-ng-physics --test phase37_growth_rescaled_ics --release \
        -- --test-threads=1
else
    echo "   [omitido por SKIP_PHASE37=1]"
fi

echo "== Phase 41 alta resolución (sin N=256) =="
if [ "${RUN_PHASE41:-0}" = "1" ]; then
    cargo test -p gadget-ng-physics --test phase41_high_resolution_validation --release \
        -- --test-threads=1
else
    echo "   [omitido por defecto — activar con RUN_PHASE41=1, tarda ~3-4 h]"
fi

echo "== cargo build release (MPI) =="
cargo build --release -p gadget-ng-cli --features mpi

echo "== tests MPI (feature mpi, serial runtime, release) =="
cargo test -p gadget-ng-parallel --features mpi --release

echo "== smoke MPI multirank (2 ranks) =="
cargo test -p gadget-ng-parallel --features mpi --release --no-run 2>/dev/null
BIN=$(find target/release/deps -name "sfc_hardening-*" -executable 2>/dev/null | head -1)
if [ -n "$BIN" ]; then
    mpirun --oversubscribe -n 2 "$BIN" --test-threads=1
fi
BIN=$(find target/release/deps -name "let_validation-*" -executable 2>/dev/null | head -1)
if [ -n "$BIN" ]; then
    mpirun --oversubscribe -n 2 "$BIN" --test-threads=1
fi

echo "check_release.sh: OK"
