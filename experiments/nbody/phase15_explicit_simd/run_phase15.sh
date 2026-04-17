#!/usr/bin/env bash
# Phase 15 — Explicit AVX2 SIMD benchmarks
# Compara Fase 14 (kernel fusionado) vs Fase 15 (intrinsics AVX2 explícitos).
#
# Nota: ambos binarios usan --features mpi,simd.
# La diferencia de rendimiento viene del despacho en RmnSoa::accel_range:
#   - Baseline (p14): accel_soa_avx2 → accel_soa_scalar (auto-vec, xmm)
#   - Explicit (p15): accel_p15_avx2_range (intrinsics ymm reales)
#
# El binario p14_fused se construye desde el commit de Fase 14 (sin el
# nuevo kernel), mientras que p15_explicit usa el código actual.
# Para simplificar el benchmark, ambos usan el binario actual pero se
# identifican por el análisis del código ensamblado.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PHASE_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$PHASE_DIR/results"
CONFIGS_DIR="$PHASE_DIR/configs"

echo "=== Phase 15: Explicit AVX2 SIMD benchmarks ==="
echo "Repo root: $REPO_ROOT"

# ── Build ─────────────────────────────────────────────────────────────────────
echo ""
echo "--- Building p15_explicit (--features mpi,simd + AVX2 intrinsics) ---"
RUSTFLAGS="-C target-cpu=native -C target-feature=+avx2,+fma" cargo build --release \
    --manifest-path "$REPO_ROOT/Cargo.toml" \
    --package gadget-ng-cli \
    --features mpi,simd \
    2>&1 | tail -3
cp "$REPO_ROOT/target/release/gadget-ng" "$PHASE_DIR/gadget-ng-p15-explicit"

# ── Reusar el binario de Fase 14 si existe ────────────────────────────────────
P14_BIN="$REPO_ROOT/experiments/nbody/phase14_soa_simd/gadget-ng-soa-simd"
if [[ -f "$P14_BIN" ]]; then
    echo "--- Reusing Phase 14 binary from: $P14_BIN ---"
    cp "$P14_BIN" "$PHASE_DIR/gadget-ng-p14-fused"
else
    echo "--- Phase 14 binary not found, using current build as p14 baseline ---"
    echo "    (Para una comparación P14 vs P15 exacta, ejecuta run_phase14.sh primero)"
    cp "$REPO_ROOT/target/release/gadget-ng" "$PHASE_DIR/gadget-ng-p14-fused"
fi

# ── Generar configs ───────────────────────────────────────────────────────────
echo ""
echo "--- Generating configs ---"
cd "$PHASE_DIR"
python3 generate_configs.py

# ── Ejecutar ──────────────────────────────────────────────────────────────────
mkdir -p "$RESULTS_DIR"

run_config() {
    local cfg="$1"
    local variant="$2"   # "p14_fused" o "p15_explicit"
    local binary="$3"
    local p="$4"

    local base
    base=$(basename "$cfg" .toml)
    local out_dir="$RESULTS_DIR/${base}_${variant}"
    mkdir -p "$out_dir"

    local result_json="$out_dir/timings.json"
    if [[ -f "$result_json" ]]; then
        echo "  SKIP (already exists): ${base}_${variant}"
        return
    fi

    echo "  RUN [P=$p] ${base}_${variant}"
    if [[ "$p" -gt 1 ]]; then
        mpirun -n "$p" "$binary" stepping \
            --config "$cfg" \
            --out "$out_dir" \
            > "$out_dir/stdout.log" 2> "$out_dir/stderr.log" || {
                echo "  ERROR: ${base}_${variant} (exit $?)" >&2
                return 1
            }
    else
        "$binary" stepping \
            --config "$cfg" \
            --out "$out_dir" \
            > "$out_dir/stdout.log" 2> "$out_dir/stderr.log" || {
                echo "  ERROR: ${base}_${variant} (exit $?)" >&2
                return 1
            }
    fi
}

echo ""
echo "--- Running benchmarks ---"

for cfg_path in "$CONFIGS_DIR"/*.toml; do
    base=$(basename "$cfg_path" .toml)

    # Extraer P del nombre
    p=$(echo "$base" | grep -oP '_P\K[0-9]+')
    if [[ -z "$p" ]]; then
        p=1
    fi

    # Fase 14 (fused kernel, auto-vec)
    run_config "$cfg_path" "p14_fused" "$PHASE_DIR/gadget-ng-p14-fused" "$p"
    # Fase 15 (intrinsics AVX2 explícitos)
    run_config "$cfg_path" "p15_explicit" "$PHASE_DIR/gadget-ng-p15-explicit" "$p"
done

echo ""
echo "=== All runs complete ==="
echo "Results in: $RESULTS_DIR"
echo "Run: python3 analyze_phase15.py"
