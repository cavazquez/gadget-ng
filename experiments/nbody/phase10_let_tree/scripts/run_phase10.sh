#!/usr/bin/env bash
# Fase 10 — LET-tree benchmarks: flat_let vs let_tree.
#
# Uso: ./run_phase10.sh [--dry-run] [--release]
#
# Variables de entorno:
#   GADGET_BIN   ruta al binario gadget-ng (default: detecta en target/)
#   RESULTS_DIR  directorio de resultados (default: ../results)
#
# Requisitos: mpirun, cargo (si se compila aquí).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
CONFIGS_DIR="$SCRIPT_DIR/../configs"
RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/../results}"
DRY_RUN=0
PROFILE="--release"

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        --debug)   PROFILE="" ;;
        --release) PROFILE="--release" ;;
    esac
done

# ── Compilar ──────────────────────────────────────────────────────────────────
echo "=== Compilando gadget-ng (features: mpi) ==="
(cd "$REPO_ROOT" && cargo build $PROFILE --features mpi -p gadget-ng-cli 2>&1 | tail -5)

if [[ -n "$PROFILE" ]]; then
    GADGET_BIN="${GADGET_BIN:-$REPO_ROOT/target/release/gadget-ng}"
else
    GADGET_BIN="${GADGET_BIN:-$REPO_ROOT/target/debug/gadget-ng}"
fi
echo "Binario: $GADGET_BIN"

mkdir -p "$RESULTS_DIR"

# ── Ejecutar configs ─────────────────────────────────────────────────────────
run_config() {
    local toml="$1"
    local base; base="$(basename "$toml" .toml)"
    local p; p="$(echo "$base" | grep -oP '(?<=_p)\d+')"

    local out_dir="$RESULTS_DIR/$base"
    mkdir -p "$out_dir"

    # Actualizar output_dir dentro del toml con la ruta real.
    local tmp_toml; tmp_toml="$(mktemp /tmp/gadget_phase10_XXXXXX.toml)"
    sed "s|results/$base|$out_dir|g" "$toml" > "$tmp_toml"

    echo -n "  [$base] P=$p ... "
    local t_start; t_start=$(date +%s%3N)

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "(dry-run)"
        rm -f "$tmp_toml"
        return
    fi

    if [[ "$p" -gt 1 ]]; then
        mpirun --oversubscribe -n "$p" "$GADGET_BIN" "$tmp_toml" \
            > "$out_dir/stdout.log" 2> "$out_dir/stderr.log"
    else
        "$GADGET_BIN" "$tmp_toml" \
            > "$out_dir/stdout.log" 2> "$out_dir/stderr.log"
    fi

    local t_end; t_end=$(date +%s%3N)
    local wall_ms=$(( t_end - t_start ))
    echo "$wall_ms" > "$out_dir/wall_ms.txt"
    echo "done (${wall_ms}ms)"
    rm -f "$tmp_toml"
}

echo ""
echo "=== Ejecutando benchmarks ==="
total=0; ok=0
for toml in $(ls "$CONFIGS_DIR"/*.toml | sort); do
    (( total++ )) || true
    run_config "$toml" && (( ok++ )) || true
done

echo ""
echo "=== Completado: $ok/$total configs ==="
echo "Resultados en: $RESULTS_DIR"
