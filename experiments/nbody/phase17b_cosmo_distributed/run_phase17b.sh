#!/usr/bin/env bash
# ── run_phase17b.sh — Fase 17b: Cosmología Distribuida ───────────────────────
#
# Ejecuta benchmarks de validación serial vs MPI para el modo cosmológico SFC+LET.
# Para cada configuración y P ∈ {1, 2, 4}, ejecuta la simulación y guarda:
#   results/<nombre>/P<N>/diagnostics.jsonl
#   results/<nombre>/P<N>/timings.json
#
# Uso:
#   bash run_phase17b.sh [--skip-build] [--out-dir <dir>] [--p-values "1 2 4"]
#
# Dependencias:
#   - cargo + mpirun
#   - python3 + numpy + matplotlib (para analyze_phase17b.py)
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

# ── Build ─────────────────────────────────────────────────────────────────────

if [[ "$SKIP_BUILD" == false ]]; then
    echo "=== Compilando gadget-ng (release) ==="
    cargo build --release --manifest-path "$REPO_ROOT/Cargo.toml" -p gadget-ng-cli
fi

[[ -x "$BINARY" ]] || { echo "ERROR: $BINARY no encontrado"; exit 1; }

# ── Configs a ejecutar ────────────────────────────────────────────────────────

declare -A CONFIGS
CONFIGS["eds_N512"]="$CONFIGS_DIR/eds_N512_mpi.toml"
CONFIGS["lcdm_N1000"]="$CONFIGS_DIR/lcdm_N1000_mpi.toml"
CONFIGS["eds_N2000"]="$CONFIGS_DIR/eds_N2000_mpi.toml"

# ── Función de ejecución ──────────────────────────────────────────────────────

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

# ── Loop principal ─────────────────────────────────────────────────────────────

echo ""
echo "=== Fase 17b: Validación de equivalencia serial vs MPI ==="
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

# ── Resumen ───────────────────────────────────────────────────────────────────

echo "=== Fase 17b completada ==="
echo ""
echo "Resultados en: $OUT_DIR"
echo ""
echo "Para analizar:"
echo "    python3 $SCRIPT_DIR/analyze_phase17b.py --results-dir $OUT_DIR"
