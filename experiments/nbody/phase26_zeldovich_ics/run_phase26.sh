#!/usr/bin/env bash
# ── run_phase26.sh — Fase 26: Condiciones Iniciales de Zel'dovich ─────────────
#
# Ejecuta los experimentos de validación física de las ICs de Zel'dovich.
# Corre PM y TreePM con distintos espectros y genera diagnósticos.
#
# Uso:
#   bash run_phase26.sh [--skip-build] [--out-dir <dir>] [--skip-plots]
#
# Después de correr, ejecuta los scripts de validación:
#   python3 scripts/validate_pk.py --results-dir results/
#   python3 scripts/plot_growth.py --results-dir results/
#   python3 scripts/compare_pm_treepm.py --results-dir results/
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
BINARY="$REPO_ROOT/target/release/gadget-ng"
CONFIGS_DIR="$SCRIPT_DIR/configs"

SKIP_BUILD=false
SKIP_PLOTS=false
OUT_DIR="$SCRIPT_DIR/results"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-build) SKIP_BUILD=true; shift ;;
        --skip-plots) SKIP_PLOTS=true; shift ;;
        --out-dir)    OUT_DIR="$2"; shift 2 ;;
        *) echo "Opción desconocida: $1"; exit 1 ;;
    esac
done

# ── Compilar ──────────────────────────────────────────────────────────────────

if [[ "$SKIP_BUILD" == false ]]; then
    echo "=== Compilando gadget-ng (release) ==="
    cargo build --release --manifest-path "$REPO_ROOT/Cargo.toml" -p gadget-ng-cli
fi

[[ -x "$BINARY" ]] || { echo "ERROR: $BINARY no encontrado"; exit 1; }

# ── Configuraciones a ejecutar ────────────────────────────────────────────────

declare -A CONFIGS
CONFIGS["eds_N32_ns-2_pm"]="$CONFIGS_DIR/eds_N32_ns-2_pm.toml"
CONFIGS["eds_N32_ns-2_treepm"]="$CONFIGS_DIR/eds_N32_ns-2_treepm.toml"
CONFIGS["eds_N32_ns-1_treepm"]="$CONFIGS_DIR/eds_N32_ns-1_treepm.toml"

# ── Función de ejecución ──────────────────────────────────────────────────────

run_sim() {
    local name="$1" config="$2" out="$3"
    mkdir -p "$out"
    echo "  Corriendo: $name → $out"
    local t0 t1 elapsed
    t0=$(date +%s%3N)
    "$BINARY" stepping \
        --config "$config" \
        --out "$out" 2>&1 || true
    t1=$(date +%s%3N)
    elapsed=$(( t1 - t0 ))
    local nlines="N/A"
    [[ -f "$out/diagnostics.jsonl" ]] && nlines=$(wc -l < "$out/diagnostics.jsonl")
    echo "    [OK] $name: ${elapsed} ms, ${nlines} pasos de diagnóstico"
}

# También generar snapshot inicial para medir P(k).
run_snapshot() {
    local name="$1" config="$2" out="$3"
    mkdir -p "$out"
    echo "  Snapshot inicial: $name"
    "$BINARY" snapshot \
        --config "$config" \
        --out "$out/snapshot_init.jsonl" 2>&1 || true
}

# ── Ejecutar ───────────────────────────────────────────────────────────────────

echo ""
echo "=== Fase 26: Condiciones Iniciales de Zel'dovich ==="
echo ""

for name in "${!CONFIGS[@]}"; do
    config="${CONFIGS[$name]}"
    out="$OUT_DIR/$name"
    echo "--- Config: $name ---"
    run_snapshot "$name" "$config" "$out"
    run_sim      "$name" "$config" "$out"
    echo ""
done

# ── Análisis de P(k) en snapshot inicial ─────────────────────────────────────

for name in "${!CONFIGS[@]}"; do
    out="$OUT_DIR/$name"
    snap="$out/snapshot_init.jsonl"
    pk_out="$out/pk_initial.jsonl"
    if [[ -f "$snap" ]]; then
        echo "  Midiendo P(k) inicial: $name"
        "$BINARY" analyse \
            --snapshot "$snap" \
            --box-size 1.0 \
            --pk-mesh 32 \
            --out "$out/" 2>&1 || true
        [[ -f "$out/power_spectrum.jsonl" ]] && \
            cp "$out/power_spectrum.jsonl" "$pk_out"
    fi
done

echo ""
echo "=== Fase 26 completada ==="
echo "Resultados en: $OUT_DIR"
echo ""

if [[ "$SKIP_PLOTS" == false ]]; then
    echo "=== Generando figuras de validación ==="
    SCRIPTS_DIR="$SCRIPT_DIR/scripts"

    if command -v python3 &>/dev/null; then
        echo "  validate_pk.py ..."
        python3 "$SCRIPTS_DIR/validate_pk.py" --results-dir "$OUT_DIR" || echo "  [WARN] validate_pk.py falló"

        echo "  plot_growth.py ..."
        python3 "$SCRIPTS_DIR/plot_growth.py" --results-dir "$OUT_DIR" || echo "  [WARN] plot_growth.py falló"

        echo "  compare_pm_treepm.py ..."
        python3 "$SCRIPTS_DIR/compare_pm_treepm.py" --results-dir "$OUT_DIR" || echo "  [WARN] compare_pm_treepm.py falló"

        echo "  plot_density_slice.py ..."
        python3 "$SCRIPTS_DIR/plot_density_slice.py" --results-dir "$OUT_DIR" || echo "  [WARN] plot_density_slice.py falló"
    else
        echo "  [SKIP] python3 no disponible; saltar figuras"
    fi
fi

echo ""
echo "Para ejecutar solo los tests automáticos:"
echo "    cargo test -p gadget-ng-physics --test zeldovich_ics"
