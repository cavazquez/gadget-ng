#!/usr/bin/env bash
# Fase 11 — benchmarks LetTree paralelo.
#
# Uso: ./run_phase11.sh [--dry-run] [--only bench|valid|sensitivity]
#
# Compila con features simd+mpi para activar Rayon en el walk del LetTree.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
CONFIGS_DIR="$SCRIPT_DIR/../configs"
RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR/../results}"
DRY_RUN=0
ONLY=""

for arg in "$@"; do
    case "$arg" in
        --dry-run)          DRY_RUN=1 ;;
        --only)             ;;
        bench|valid|sensitivity) ONLY="$arg" ;;
    esac
done

# ── Compilar con simd+mpi ─────────────────────────────────────────────────────
echo "=== Compilando gadget-ng (features: simd,mpi) ==="
(cd "$REPO_ROOT" && cargo build --release --features simd,mpi -p gadget-ng-cli 2>&1 | tail -5)
GADGET_BIN="${GADGET_BIN:-$REPO_ROOT/target/release/gadget-ng}"
echo "Binario: $GADGET_BIN"

mkdir -p "$RESULTS_DIR"

# ── Ejecutar una config ───────────────────────────────────────────────────────
run_config() {
    local toml="$1"
    local base; base="$(basename "$toml" .toml)"
    local p; p="$(echo "$base" | grep -oP '(?<=_p)\d+' | head -1)"
    [[ -z "$p" ]] && p=2  # default para configs de sensitivity (ejecutar con P=2)

    # Filtro por tipo si --only está activo
    if [[ -n "$ONLY" ]]; then
        case "$base" in
            bench_*)       [[ "$ONLY" == "bench" ]] || return 0 ;;
            valid_*)       [[ "$ONLY" == "valid" ]] || return 0 ;;
            sens_*)        [[ "$ONLY" == "sensitivity" ]] || return 0 ;;
        esac
    fi

    local out_dir="$RESULTS_DIR/$base"
    mkdir -p "$out_dir"

    local tmp_toml; tmp_toml="$(mktemp /tmp/gadget_p11_XXXXXX.toml)"
    cp "$toml" "$tmp_toml"

    echo -n "  [$base] P=$p ... "
    local t_start; t_start=$(date +%s%3N)

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "(dry-run)"
        rm -f "$tmp_toml"
        return
    fi

    local rc=0
    if [[ "$p" -gt 1 ]]; then
        mpirun --oversubscribe -n "$p" "$GADGET_BIN" stepping \
            --config "$tmp_toml" --out "$out_dir" \
            > "$out_dir/stdout.log" 2> "$out_dir/stderr.log" || rc=$?
    else
        "$GADGET_BIN" stepping \
            --config "$tmp_toml" --out "$out_dir" \
            > "$out_dir/stdout.log" 2> "$out_dir/stderr.log" || rc=$?
    fi

    local t_end; t_end=$(date +%s%3N)
    local wall_ms=$(( t_end - t_start ))
    echo "$wall_ms" > "$out_dir/wall_ms.txt"

    if [[ "$rc" -ne 0 ]]; then
        echo "FAILED (rc=$rc, ${wall_ms}ms)"
    else
        echo "ok (${wall_ms}ms)"
    fi
    rm -f "$tmp_toml"
}

echo ""
echo "=== Ejecutando benchmarks Fase 11 ==="
total=0; ok=0
for toml in $(ls "$CONFIGS_DIR"/*.toml | sort); do
    (( total++ )) || true
    run_config "$toml" && (( ok++ )) || true
done

echo ""
echo "=== Completado: $ok/$total configs ==="
echo "Resultados en: $RESULTS_DIR"
