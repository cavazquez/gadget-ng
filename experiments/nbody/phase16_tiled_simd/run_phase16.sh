#!/usr/bin/env bash
# Phase 16 — Tiled 4×N_i SIMD benchmarks
# Compara Fase 14 (kernel fusionado, auto-vec) vs Fase 15 (intrinsics AVX2 1xi)
# vs Fase 16 (walk tileado 4xi, instrinsics AVX2).
#
# P16 estrategia:
#   - Loop Rayon con par_chunks_mut(4) → walk_accel_4xi
#   - Kernel: accel_range_4xi → mono_pass_avx2_4xi (broadcast+ymm) + quad_oct_pass_scalar_4xi
#   - Efecto: ~4 iteraciones SIMD completas por llamada apply_leaf (vs ~0.9 en P15)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PHASE_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$PHASE_DIR/results"
CONFIGS_DIR="$PHASE_DIR/configs"

echo "=== Phase 16: Tiled 4×N_i AVX2 benchmarks ==="
echo "Repo root: $REPO_ROOT"

# ── Build P16 ─────────────────────────────────────────────────────────────────
echo ""
echo "--- Building p16_tiled (--features mpi,simd + walk_accel_4xi) ---"
RUSTFLAGS="-C target-cpu=native -C target-feature=+avx2,+fma" cargo build --release \
    --manifest-path "$REPO_ROOT/Cargo.toml" \
    --package gadget-ng-cli \
    --features mpi,simd \
    2>&1 | tail -3
cp "$REPO_ROOT/target/release/gadget-ng" "$PHASE_DIR/gadget-ng-p16-tiled"

# ── Reusar binarios P14 y P15 ─────────────────────────────────────────────────
P14_BIN="$REPO_ROOT/experiments/nbody/phase14_soa_simd/gadget-ng-soa-simd"
P15_BIN="$REPO_ROOT/experiments/nbody/phase15_explicit_simd/gadget-ng-p15-explicit"

if [[ -f "$P14_BIN" ]]; then
    echo "--- Reusing Phase 14 binary: $P14_BIN ---"
    cp "$P14_BIN" "$PHASE_DIR/gadget-ng-p14-fused"
else
    echo "WARN: Phase 14 binary not found at $P14_BIN"
    cp "$PHASE_DIR/gadget-ng-p16-tiled" "$PHASE_DIR/gadget-ng-p14-fused"
fi

if [[ -f "$P15_BIN" ]]; then
    echo "--- Reusing Phase 15 binary: $P15_BIN ---"
    cp "$P15_BIN" "$PHASE_DIR/gadget-ng-p15-explicit"
else
    echo "WARN: Phase 15 binary not found at $P15_BIN"
    cp "$PHASE_DIR/gadget-ng-p16-tiled" "$PHASE_DIR/gadget-ng-p15-explicit"
fi

# ── Función de ejecución ──────────────────────────────────────────────────────
run_config() {
    local cfg="$1"
    local variant="$2"
    local binary="$3"
    local p="$4"

    local base
    base=$(basename "$cfg" .toml)
    local out_dir="$RESULTS_DIR/${variant}/${base}"
    mkdir -p "$out_dir"

    local result_json="$out_dir/timings.json"
    if [[ -f "$result_json" ]]; then
        echo "  SKIP (already exists): ${variant}/${base}"
        return
    fi

    echo "  RUN [P=$p] ${variant}/${base}"
    if [[ "$p" -gt 1 ]]; then
        mpirun -n "$p" "$binary" stepping \
            --config "$cfg" \
            --out "$out_dir" \
            > "$out_dir/stdout.log" 2> "$out_dir/stderr.log" || {
                echo "  ERROR: ${variant}/${base} (exit $?)" >&2
                return 1
            }
    else
        "$binary" stepping \
            --config "$cfg" \
            --out "$out_dir" \
            > "$out_dir/stdout.log" 2> "$out_dir/stderr.log" || {
                echo "  ERROR: ${variant}/${base} (exit $?)" >&2
                return 1
            }
    fi
}

echo ""
echo "--- Running benchmarks ---"
mkdir -p "$RESULTS_DIR/p14" "$RESULTS_DIR/p15" "$RESULTS_DIR/p16"

for cfg_path in "$CONFIGS_DIR"/*.toml; do
    p=$(basename "$cfg_path" .toml | grep -oP '_P\K[0-9]+' || echo "1")
    [[ -z "$p" ]] && p=1

    run_config "$cfg_path" "p14" "$PHASE_DIR/gadget-ng-p14-fused"   "$p"
    run_config "$cfg_path" "p15" "$PHASE_DIR/gadget-ng-p15-explicit" "$p"
    run_config "$cfg_path" "p16" "$PHASE_DIR/gadget-ng-p16-tiled"    "$p"
done

echo ""
echo "=== All runs complete ==="
echo "Results in: $RESULTS_DIR"
echo "Run: python3 analyze_phase16.py"
