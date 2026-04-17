#!/usr/bin/env bash
# ── run_phase18.sh — Fase 18: Cosmología Periódica con PM ─────────────────────
#
# Ejecuta benchmarks de validación para el modo cosmológico periódico.
# Para cada configuración ejecuta P=1 (serial) y P=2,4 (MPI) para comparar
# equivalencia física.
#
# Uso:
#   bash run_phase18.sh [--skip-build] [--out-dir <dir>] [--p-values "1 2 4"]
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
BINARY="$REPO_ROOT/target/release/gadget-ng"
CONFIGS_DIR="$SCRIPT_DIR/configs"

SKIP_BUILD=false
OUT_DIR="$SCRIPT_DIR/results"
P_VALUES="1 2 4"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-build) SKIP_BUILD=true; shift ;;
        --out-dir)    OUT_DIR="$2"; shift 2 ;;
        --p-values)   P_VALUES="$2"; shift 2 ;;
        *) echo "Opción desconocida: $1"; exit 1 ;;
    esac
done

if [[ "$SKIP_BUILD" == false ]]; then
    echo "=== Compilando gadget-ng (release) ==="
    cargo build --release --manifest-path "$REPO_ROOT/Cargo.toml" -p gadget-ng-cli
fi

[[ -x "$BINARY" ]] || { echo "ERROR: $BINARY no encontrado"; exit 1; }

declare -A CONFIGS
CONFIGS["eds_N512_pm"]="$CONFIGS_DIR/eds_N512_pm.toml"
CONFIGS["lcdm_N1000_pm"]="$CONFIGS_DIR/lcdm_N1000_pm.toml"
CONFIGS["eds_N512_treepm"]="$CONFIGS_DIR/eds_N512_treepm.toml"
CONFIGS["eds_N512_pm_grid32"]="$CONFIGS_DIR/eds_N512_pm_grid32.toml"

run_sim() {
    local name="$1" config="$2" out="$3" np="$4"
    mkdir -p "$out"
    local t0 t1 elapsed
    t0=$(date +%s%3N)
    mpirun --oversubscribe -np "$np" "$BINARY" stepping \
        --config "$config" \
        --out "$out" 2>&1 | grep -v "^\[" || true
    t1=$(date +%s%3N)
    elapsed=$(( t1 - t0 ))
    local nlines="N/A"
    [[ -f "$out/diagnostics.jsonl" ]] && nlines=$(wc -l < "$out/diagnostics.jsonl")
    echo "    [P=$np, $name] ${elapsed} ms — ${nlines} líneas de diagnóstico"
}

echo ""
echo "=== Fase 18: Cosmología Periódica con PM ==="
echo "    P_VALUES = $P_VALUES"
echo ""

for name in "${!CONFIGS[@]}"; do
    config="${CONFIGS[$name]}"
    echo "--- Config: $name ---"
    for p in $P_VALUES; do
        out="$OUT_DIR/$name/P$p"
        run_sim "$name" "$config" "$out" "$p"
    done
    echo ""
done

echo "=== Fase 18 completada ==="
echo "Resultados en: $OUT_DIR"
echo ""
echo "Para analizar:"
echo "    python3 $SCRIPT_DIR/analyze_phase18.py --results-dir $OUT_DIR"
