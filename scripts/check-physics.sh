#!/usr/bin/env bash
set -euo pipefail
trap 'echo "ERROR at line $LINENO"' ERR

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[physics-check] 1/3 Transfer tabulada (unit)"
cargo test -p gadget-ng-core ic_zeldovich::tests::tabulated_transfer_reconstructs_knots_and_midpoints -- --nocapture

echo "[physics-check] 2/3 Pancake Zel'dovich"
cargo test -p gadget-ng-physics --test zeldovich_pancake -- --nocapture

echo "[physics-check] 3/3 CLASS validation completa (release + ignored)"
cargo test -p gadget-ng-physics --release --test phase38_class_validation -- --include-ignored --nocapture

echo "[physics-check] OK: validaciones físicas completas."
