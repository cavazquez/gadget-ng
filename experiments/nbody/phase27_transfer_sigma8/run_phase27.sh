#!/usr/bin/env bash
# ── run_phase27.sh — Experimentos de validación Phase 27 ─────────────────────
#
# Ejecuta los tres casos de simulación:
#   1. ΛCDM + EH + σ₈=0.8, PM
#   2. ΛCDM + EH + σ₈=0.8, TreePM
#   3. ΛCDM + ley de potencia (legacy), PM (comparación)
#
# Para cada caso:
#   - Genera el snapshot inicial (a_init = 0.02)
#   - Ejecuta la simulación por 50 pasos
#   - Extrae P(k) y diagnósticos
#   - Corre los scripts de validación Python
#
# Uso: bash run_phase27.sh [--binary /path/to/gadget-ng] [--out-dir ./output]
#
# Prerrequisitos:
#   - gadget-ng compilado en release: cargo build --release -p gadget-ng
#   - Python 3 con numpy, matplotlib, scipy

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
BINARY="${GADGET_BINARY:-$REPO_ROOT/target/release/gadget-ng}"
OUT_DIR="${1:-$SCRIPT_DIR/output}"
PY_SCRIPTS="$SCRIPT_DIR/scripts"

# ── Verificaciones previas ────────────────────────────────────────────────────

if [ ! -f "$BINARY" ]; then
    echo "❌ Binario no encontrado: $BINARY"
    echo "   Compilar con: cd $REPO_ROOT && cargo build --release -p gadget-ng"
    exit 1
fi

mkdir -p "$OUT_DIR/eh_pm" "$OUT_DIR/eh_treepm" "$OUT_DIR/pl_pm"
echo "✓ Directorio de salida: $OUT_DIR"
echo "✓ Binario: $BINARY"
echo ""

# ── Función auxiliar ──────────────────────────────────────────────────────────

run_sim() {
    local label="$1"
    local config="$2"
    local out_dir="$3"

    echo "━━━ Corriendo: $label ━━━━━━━━━━━━━━━━━━━━━━━━"
    "$BINARY" --config "$config" --output-dir "$out_dir" 2>&1 | tee "$out_dir/run.log"
    echo "✓ $label completado"
    echo ""
}

extract_pk() {
    local snap_dir="$1"
    local out_file="$2"

    # El binario gadget-ng genera pk_*.json en el directorio de salida
    # (snapshot_every = 10, primera salida en step 0 = snapshot inicial)
    local first_snap
    first_snap=$(find "$snap_dir" -name "pk_*.json" | sort | head -n 1)
    if [ -n "$first_snap" ]; then
        cp "$first_snap" "$out_file"
        echo "  P(k) inicial: $first_snap → $out_file"
    else
        echo "  ⚠ No se encontró pk_*.json en $snap_dir"
    fi
}

extract_diag() {
    local snap_dir="$1"
    local out_file="$2"

    local diag
    diag=$(find "$snap_dir" -name "diagnostics.json" | head -n 1)
    if [ -n "$diag" ]; then
        cp "$diag" "$out_file"
        echo "  Diagnósticos: $diag → $out_file"
    else
        echo "  ⚠ No se encontró diagnostics.json en $snap_dir"
    fi
}

# ── Simulaciones ──────────────────────────────────────────────────────────────

echo "═══ Phase 27: Función de Transferencia Eisenstein–Hu + σ₈ ═══"
echo ""

run_sim "ΛCDM EH PM"     "$SCRIPT_DIR/configs/lcdm_N32_eh_pm.toml"     "$OUT_DIR/eh_pm"
run_sim "ΛCDM EH TreePM" "$SCRIPT_DIR/configs/lcdm_N32_eh_treepm.toml" "$OUT_DIR/eh_treepm"
run_sim "ΛCDM PL PM"     "$SCRIPT_DIR/configs/lcdm_N32_powerlaw_pm.toml" "$OUT_DIR/pl_pm"

# ── Extracción de datos ───────────────────────────────────────────────────────

echo "━━━ Extrayendo datos ━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

extract_pk   "$OUT_DIR/eh_pm"     "$OUT_DIR/pk_eh_initial.json"
extract_pk   "$OUT_DIR/pl_pm"     "$OUT_DIR/pk_pl_initial.json"
extract_diag "$OUT_DIR/eh_pm"     "$OUT_DIR/diag_eh_pm.json"
extract_diag "$OUT_DIR/eh_treepm" "$OUT_DIR/diag_eh_treepm.json"
extract_diag "$OUT_DIR/pl_pm"     "$OUT_DIR/diag_pl_pm.json"

echo ""

# ── Scripts de validación Python ─────────────────────────────────────────────

echo "━━━ Validación Python ━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if command -v python3 &>/dev/null; then
    # 1. Comparación de espectros
    if [ -f "$OUT_DIR/pk_eh_initial.json" ] && [ -f "$OUT_DIR/pk_pl_initial.json" ]; then
        python3 "$PY_SCRIPTS/compare_spectra.py" \
            --eh  "$OUT_DIR/pk_eh_initial.json" \
            --pl  "$OUT_DIR/pk_pl_initial.json" \
            --box 100.0 \
            --out "$OUT_DIR/fig_compare_spectra.png"
    else
        echo "  ⚠ Faltan archivos P(k) para compare_spectra.py"
    fi

    # 2. Validación σ₈
    if [ -f "$OUT_DIR/pk_eh_initial.json" ]; then
        python3 "$PY_SCRIPTS/validate_sigma8.py" \
            --pk     "$OUT_DIR/pk_eh_initial.json" \
            --box    100.0 \
            --sigma8 0.8 \
            --h      0.674 \
            --out    "$OUT_DIR/fig_validate_sigma8.png"
    fi

    # 3. Crecimiento lineal
    if [ -f "$OUT_DIR/diag_eh_pm.json" ]; then
        PL_ARG=""
        [ -f "$OUT_DIR/diag_pl_pm.json" ] && PL_ARG="--diag_pl $OUT_DIR/diag_pl_pm.json"
        python3 "$PY_SCRIPTS/plot_growth.py" \
            --diag   "$OUT_DIR/diag_eh_pm.json" \
            $PL_ARG \
            --omega_m 0.315 \
            --omega_l 0.685 \
            --a_init  0.02 \
            --out     "$OUT_DIR/fig_growth.png"
    fi
else
    echo "  ⚠ python3 no disponible — omitiendo validación Python"
fi

echo ""
echo "═══ Resumen Phase 27 ════════════════════════════════"
echo ""
echo "  Figuras generadas en: $OUT_DIR/"
ls -la "$OUT_DIR"/fig_*.png 2>/dev/null || echo "  (ninguna figura generada)"
echo ""
echo "  Logs de simulación:"
for log in "$OUT_DIR"/*/run.log; do
    if [ -f "$log" ]; then
        echo "    $log"
        grep -E "(ERROR|WARN|completed|steps)" "$log" | tail -3 || true
    fi
done
echo ""
echo "✓ Phase 27 completada"
