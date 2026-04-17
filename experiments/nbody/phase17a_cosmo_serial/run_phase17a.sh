#!/usr/bin/env bash
# ── run_phase17a.sh — Fase 17a: Modo Cosmológico Serial ───────────────────────
#
# Ejecuta dos simulaciones cosmológicas seriales:
#   1. EdS (Ω_m=1, Ω_Λ=0)  — N=512,  100 pasos
#   2. ΛCDM (Ω_m=0.3, Ω_Λ=0.7) — N=1000, 50 pasos
#
# Requisitos:
#   - cargo (Rust toolchain)
#   - mpirun (openmpi o mpich, solo 1 rango)
#   - python3 con numpy + matplotlib (para analyze_phase17a.py)
#
# Uso:
#   bash run_phase17a.sh [--skip-build] [--out-dir <dir>]
#
# Opciones:
#   --skip-build   No recompila el binario (usar si ya existe)
#   --out-dir DIR  Directorio base de salida (default: results/)
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
BINARY="$REPO_ROOT/target/release/gadget-ng"
CONFIGS_DIR="$SCRIPT_DIR/configs"

# Opciones por defecto.
SKIP_BUILD=false
OUT_DIR="$SCRIPT_DIR/results"

# Parseo de argumentos.
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-build) SKIP_BUILD=true; shift ;;
        --out-dir) OUT_DIR="$2"; shift 2 ;;
        *) echo "Opción desconocida: $1"; exit 1 ;;
    esac
done

# ── Compilar ──────────────────────────────────────────────────────────────────

if [[ "$SKIP_BUILD" == false ]]; then
    echo "=== Compilando gadget-ng (release) ==="
    cargo build --release --manifest-path "$REPO_ROOT/Cargo.toml" -p gadget-ng-cli
    echo "=== Build completado: $BINARY ==="
else
    echo "=== Skip build: usando $BINARY ==="
fi

if [[ ! -x "$BINARY" ]]; then
    echo "ERROR: binario no encontrado en $BINARY" >&2
    exit 1
fi

# ── Directorios de salida ─────────────────────────────────────────────────────

EDS_OUT="$OUT_DIR/eds_N512"
LCDM_OUT="$OUT_DIR/lcdm_N1000"
mkdir -p "$EDS_OUT" "$LCDM_OUT"

# ── Función de ejecución ──────────────────────────────────────────────────────

run_sim() {
    local name="$1"
    local config="$2"
    local out="$3"

    echo ""
    echo "=== Iniciando: $name ==="
    echo "    config : $config"
    echo "    output : $out"

    local t0
    t0=$(date +%s%3N)

    mpirun -np 1 "$BINARY" stepping \
        --config "$config" \
        --out "$out"

    local t1
    t1=$(date +%s%3N)
    local elapsed=$(( t1 - t0 ))
    echo "    Tiempo: ${elapsed} ms"
    echo "    Diagnósticos: $out/diagnostics.jsonl ($(wc -l < "$out/diagnostics.jsonl") líneas)"
}

# ── Corridas ──────────────────────────────────────────────────────────────────

run_sim "EdS N=512 (100 pasos)" \
    "$CONFIGS_DIR/eds_N512_serial.toml" \
    "$EDS_OUT"

run_sim "ΛCDM N=1000 (50 pasos)" \
    "$CONFIGS_DIR/lcdm_N1000_serial.toml" \
    "$LCDM_OUT"

# ── Resumen ───────────────────────────────────────────────────────────────────

echo ""
echo "=== Fase 17a completada ==="
echo "    EdS   → $EDS_OUT/diagnostics.jsonl"
echo "    ΛCDM  → $LCDM_OUT/diagnostics.jsonl"
echo ""
echo "Para analizar:"
echo "    python3 $SCRIPT_DIR/analyze_phase17a.py --eds $EDS_OUT --lcdm $LCDM_OUT"
