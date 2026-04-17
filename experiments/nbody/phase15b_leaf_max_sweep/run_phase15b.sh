#!/usr/bin/env bash
# Phase 15b — Sweep de leaf_max para aprovechamiento SIMD.
#
# Reutiliza los binarios de Phase 15 (sin recompilar):
#   - p14_fused:    kernel fusionado Phase 14 (accel_soa_avx2 → scalar, xmm)
#   - p15_explicit: intrinsics AVX2 Phase 15 (accel_p15_avx2_range, ymm)
#
# Solo varía let_tree_leaf_max en los TOML: {8, 16, 32, 64}.
set -euo pipefail

PHASE_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$PHASE_DIR/results"
CONFIGS_DIR="$PHASE_DIR/configs"

P14_BIN="$PHASE_DIR/../phase15_explicit_simd/gadget-ng-p14-fused"
P15_BIN="$PHASE_DIR/../phase15_explicit_simd/gadget-ng-p15-explicit"

echo "=== Phase 15b: leaf_max sweep ==="
echo "P14 binary: $P14_BIN"
echo "P15 binary: $P15_BIN"

if [[ ! -f "$P14_BIN" ]]; then
    echo "ERROR: binario P14 no encontrado en $P14_BIN"
    echo "       Ejecuta primero experiments/nbody/phase15_explicit_simd/run_phase15.sh"
    exit 1
fi

# Generar configs si no existen
cd "$PHASE_DIR"
python3 generate_configs.py

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

    if [[ -f "$out_dir/timings.json" ]]; then
        echo "  SKIP: ${base}_${variant}"
        return
    fi

    echo "  RUN [P=$p] ${base}_${variant}"
    if [[ "$p" -gt 1 ]]; then
        mpirun -n "$p" "$binary" stepping \
            --config "$cfg" \
            --out "$out_dir" \
            > "$out_dir/stdout.log" 2> "$out_dir/stderr.log" \
            && echo "    OK" || echo "    FAIL (exit $?)"
    else
        "$binary" stepping \
            --config "$cfg" \
            --out "$out_dir" \
            > "$out_dir/stdout.log" 2> "$out_dir/stderr.log" \
            && echo "    OK" || echo "    FAIL (exit $?)"
    fi
}

echo ""
echo "--- Running benchmarks ---"

for cfg_path in "$CONFIGS_DIR"/*.toml; do
    base=$(basename "$cfg_path" .toml)
    p=$(echo "$base" | grep -oP '_P\K[0-9]+')
    [[ -z "$p" ]] && p=1

    run_config "$cfg_path" "p14_fused"    "$P14_BIN" "$p"
    run_config "$cfg_path" "p15_explicit" "$P15_BIN" "$p"
done

echo ""
echo "=== All runs complete ==="
echo "Results in: $RESULTS_DIR"
echo "Run: python3 analyze_phase15b.py"
