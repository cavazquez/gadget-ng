#!/usr/bin/env bash
# Phase 14 — SoA + SIMD benchmarks
# Compila dos binarios (baseline y soa_simd) y corre todas las configs.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
PHASE_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$PHASE_DIR/results"
CONFIGS_DIR="$PHASE_DIR/configs"

echo "=== Phase 14: SoA + SIMD benchmarks ==="
echo "Repo root: $REPO_ROOT"

# ── Build ─────────────────────────────────────────────────────────────────────
echo ""
echo "--- Building baseline (--features mpi) ---"
RUSTFLAGS="-C target-cpu=native" cargo build --release \
    --manifest-path "$REPO_ROOT/Cargo.toml" \
    --package gadget-ng-cli \
    --features mpi \
    2>&1 | tail -3
cp "$REPO_ROOT/target/release/gadget-ng" "$PHASE_DIR/gadget-ng-baseline"

echo ""
echo "--- Building soa_simd (--features mpi,simd) ---"
RUSTFLAGS="-C target-cpu=native -C target-feature=+avx2,+fma" cargo build --release \
    --manifest-path "$REPO_ROOT/Cargo.toml" \
    --package gadget-ng-cli \
    --features mpi,simd \
    2>&1 | tail -3
cp "$REPO_ROOT/target/release/gadget-ng" "$PHASE_DIR/gadget-ng-soa-simd"

# ── Generar configs si no existen ─────────────────────────────────────────────
echo ""
echo "--- Generating configs ---"
cd "$PHASE_DIR"
python3 generate_configs.py

# ── Ejecutar ──────────────────────────────────────────────────────────────────
mkdir -p "$RESULTS_DIR"

run_config() {
    local cfg="$1"
    local variant="$2"   # "baseline" o "soa_simd"
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

    # Extraer P del nombre (P=1,2,4)
    p=$(echo "$base" | grep -oP '_P\K[0-9]+')
    if [[ -z "$p" ]]; then
        p=1
    fi

    # Baseline
    run_config "$cfg_path" "baseline" "$PHASE_DIR/gadget-ng-baseline" "$p"
    # SoA+SIMD
    run_config "$cfg_path" "soa_simd" "$PHASE_DIR/gadget-ng-soa-simd" "$p"
done

echo ""
echo "=== All runs complete ==="
echo "Results in: $RESULTS_DIR"
echo "Run python3 analyze_phase14.py to generate tables"
