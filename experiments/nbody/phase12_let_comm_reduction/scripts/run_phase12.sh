#!/usr/bin/env bash
# ============================================================
# Phase 12 — run_phase12.sh
# Ejecuta todos los benchmarks de reducción de comunicación LET.
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
CFGS="$SCRIPT_DIR/../configs"
RESULTS="$SCRIPT_DIR/../results"
BIN="$ROOT/target/release/gadget-ng"

# ── Argumentos opcionales ─────────────────────────────────────────────────────
GROUP="${1:-all}"   # all | scale | sens | valid
NPROC_MAX="${2:-8}" # máximo de ranks MPI a usar

mkdir -p "$RESULTS"

# ── Compilar en release con simd ──────────────────────────────────────────────
echo "==> Compilando gadget-ng (release + simd)..."
cargo build --release --features simd --manifest-path "$ROOT/Cargo.toml" 2>&1 | tail -3

# ── Helper: run one config ────────────────────────────────────────────────────
run_cfg() {
    local cfg_path="$1"
    local P="$2"
    local cfg_name
    cfg_name="$(basename "$cfg_path" .toml)"
    local out_dir="$RESULTS/${cfg_name}_p${P}"

    if [[ -f "$out_dir/timings.json" ]]; then
        echo "  [SKIP] $cfg_name P=$P (ya existe)"
        return
    fi

    mkdir -p "$out_dir"
    echo "  [RUN ] $cfg_name P=$P"
    mpirun --oversubscribe -n "$P" "$BIN" stepping \
        --config "$cfg_path" \
        --out    "$out_dir" \
        > "$out_dir/stdout.log" 2> "$out_dir/stderr.log" || {
        echo "  [FAIL] $cfg_name P=$P — ver $out_dir/stderr.log"
    }
}

# ── Extraer P del nombre del config ──────────────────────────────────────────
extract_p() {
    local name="$1"
    # Extraer el número después de _p (antes de _f o fin)
    echo "$name" | grep -oP '(?<=_p)\d+' | head -1
}

# ── Ejecutar por grupo ────────────────────────────────────────────────────────
run_group() {
    local pattern="$1"
    for cfg in "$CFGS"/${pattern}*.toml; do
        [[ -f "$cfg" ]] || continue
        local name
        name="$(basename "$cfg" .toml)"
        local P
        P="$(extract_p "$name")"
        if [[ -z "$P" ]]; then
            echo "  [WARN] No se pudo extraer P de $name, saltando"
            continue
        fi
        if [[ "$P" -gt "$NPROC_MAX" ]]; then
            echo "  [SKIP] $name P=$P (> NPROC_MAX=$NPROC_MAX)"
            continue
        fi
        run_cfg "$cfg" "$P"
    done
}

echo "==> Iniciando Phase 12 benchmarks (grupo=$GROUP, NPROC_MAX=$NPROC_MAX)"
echo "    BIN: $BIN"
echo "    CFGS: $CFGS"
echo "    RESULTS: $RESULTS"
echo ""

case "$GROUP" in
    scale)  run_group "scale_" ;;
    sens)   run_group "sens_"  ;;
    valid)  run_group "valid_" ;;
    all)
        run_group "scale_"
        run_group "sens_"
        run_group "valid_"
        ;;
    *)
        echo "Uso: $0 [all|scale|sens|valid] [NPROC_MAX]"
        exit 1
        ;;
esac

echo ""
echo "==> Completado. Resultados en $RESULTS"
