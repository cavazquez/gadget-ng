#!/usr/bin/env bash
# Phase 13 — Ejecutar benchmarks Morton vs Hilbert
# Uso: ./run_phase13.sh [--dry-run] [--filter PATTERN]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
BINARY="$REPO_ROOT/target/release/gadget-ng"
CONFIGS_DIR="$SCRIPT_DIR/configs"
RESULTS_DIR="$SCRIPT_DIR/results"
DRY_RUN=false
FILTER=""

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        --filter) shift; FILTER="$1" ;;
        *) ;;
    esac
done

# Compilar con MPI + SIMD
echo "=== Compilando gadget-ng (release, mpi) ==="
if ! $DRY_RUN; then
    cargo build --release --features mpi -p gadget-ng-cli \
        --manifest-path "$REPO_ROOT/Cargo.toml"
fi

# Generar configs si no existen
if [ ! -d "$CONFIGS_DIR" ]; then
    echo "=== Generando configs ==="
    cd "$SCRIPT_DIR"
    python3 generate_configs.py
fi

mkdir -p "$RESULTS_DIR"

# Función para ejecutar un benchmark
run_one() {
    local cfg="$1"
    local name
    name="$(basename "$cfg" .toml)"

    # Extraer P del nombre (Pn_)
    local p
    p=$(echo "$name" | grep -oP '(?<=_P)\d+' | head -1)

    local out_dir="$RESULTS_DIR/$name"
    mkdir -p "$out_dir"

    if [[ -n "$FILTER" && "$name" != *"$FILTER"* ]]; then
        return 0
    fi

    echo "--- Ejecutando: $name (P=$p) ---"
    if $DRY_RUN; then
        echo "[DRY] mpirun -n $p $BINARY stepping --config $cfg --out $out_dir"
        return 0
    fi

    local t0
    t0=$(date +%s%N)
    if mpirun -n "$p" "$BINARY" stepping --config "$cfg" --out "$out_dir" \
        2>"$out_dir/stderr.log"; then
        local t1
        t1=$(date +%s%N)
        local elapsed_ms=$(( (t1 - t0) / 1000000 ))
        echo "  OK ($elapsed_ms ms)"
    else
        echo "  FAILED — ver $out_dir/stderr.log"
    fi
}

# Ejecutar todos (o filtrados)
total=0
done_count=0
for cfg in "$CONFIGS_DIR"/*.toml; do
    total=$((total + 1))
done

echo "=== Ejecutando $total benchmarks ==="
for cfg in "$CONFIGS_DIR"/*.toml; do
    run_one "$cfg"
    done_count=$((done_count + 1))
    echo "  Progreso: $done_count/$total"
done

echo ""
echo "=== Todos los benchmarks completados ==="
echo "Resultados en: $RESULTS_DIR"
