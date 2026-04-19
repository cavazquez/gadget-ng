#!/usr/bin/env bash
# ── run_phase28.sh — Experimentos de validación Phase 28: 2LPT ICs ───────────
#
# Ejecuta los tres casos de simulación:
#   1. ΛCDM + EH + σ₈=0.8, 1LPT baseline, PM
#   2. ΛCDM + EH + σ₈=0.8, 2LPT, PM
#   3. ΛCDM + EH + σ₈=0.8, 2LPT, TreePM
#
# Para cada caso:
#   - Genera ICs (a_init = 0.02, z ≈ 49)
#   - Ejecuta 50 pasos de simulación
#   - Extrae posiciones iniciales, P(k) y diagnósticos
#   - Corre los scripts de validación Python:
#       * compare_1lpt_2lpt.py — diferencias de posición y P(k)
#       * plot_growth.py        — δ_rms(a) vs D(a)
#
# Uso:
#   bash run_phase28.sh [--out-dir ./output]
#   GADGET_BINARY=/ruta/al/binario bash run_phase28.sh
#
# Prerrequisitos:
#   - gadget-ng compilado en release:
#       cargo build --release -p gadget-ng
#   - Python 3 con numpy, matplotlib, scipy

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
BINARY="${GADGET_BINARY:-$REPO_ROOT/target/release/gadget-ng}"
OUT_DIR="${1:-$SCRIPT_DIR/output}"
PY_SCRIPTS="$SCRIPT_DIR/scripts"
CONFIGS="$SCRIPT_DIR/configs"

# ── Verificaciones previas ────────────────────────────────────────────────────

if [ ! -f "$BINARY" ]; then
    echo "❌ Binario no encontrado: $BINARY"
    echo "   Compilar con: cd $REPO_ROOT && cargo build --release -p gadget-ng"
    exit 1
fi

mkdir -p "$OUT_DIR/1lpt_pm" "$OUT_DIR/2lpt_pm" "$OUT_DIR/2lpt_treepm"
echo "✓ Directorio de salida: $OUT_DIR"
echo "✓ Binario: $BINARY"
echo ""

# ── Funciones auxiliares ──────────────────────────────────────────────────────

run_sim() {
    local label="$1"
    local config="$2"
    local out_dir="$3"
    echo "━━━ Corriendo: $label ━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    "$BINARY" --config "$config" --output-dir "$out_dir" 2>&1 | tee "$out_dir/run.log"
    echo "✓ $label completado"
    echo ""
}

extract_pk() {
    local snap_dir="$1"
    local out_file="$2"
    local first_snap
    first_snap=$(find "$snap_dir" -name "pk_*.json" | sort | head -n 1)
    if [ -n "$first_snap" ]; then
        cp "$first_snap" "$out_file"
        echo "  P(k) inicial: $first_snap → $out_file"
    else
        echo "  ⚠ No se encontró pk_*.json en $snap_dir"
    fi
}

extract_positions() {
    local snap_dir="$1"
    local out_file="$2"
    local first_snap
    first_snap=$(find "$snap_dir" -name "positions_*.json" | sort | head -n 1)
    if [ -n "$first_snap" ]; then
        cp "$first_snap" "$out_file"
        echo "  Posiciones:  $first_snap → $out_file"
    else
        echo "  ⚠ No se encontró positions_*.json en $snap_dir"
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

echo "═══ Phase 28: ICs de Segundo Orden (2LPT) ════════════════════"
echo ""
echo "Configuraciones:"
echo "  1. 1LPT baseline (Zel'dovich), PM"
echo "  2. 2LPT (x = q + Ψ¹ + D₂Ψ²), PM"
echo "  3. 2LPT, TreePM"
echo ""

run_sim "1LPT baseline PM"   "$CONFIGS/lcdm_N32_1lpt_pm.toml"    "$OUT_DIR/1lpt_pm"
run_sim "2LPT PM"            "$CONFIGS/lcdm_N32_2lpt_pm.toml"    "$OUT_DIR/2lpt_pm"
run_sim "2LPT TreePM"        "$CONFIGS/lcdm_N32_2lpt_treepm.toml" "$OUT_DIR/2lpt_treepm"

# ── Extracción de datos ───────────────────────────────────────────────────────

echo "━━━ Extrayendo datos ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

extract_pk        "$OUT_DIR/1lpt_pm"    "$OUT_DIR/pk_1lpt.json"
extract_pk        "$OUT_DIR/2lpt_pm"    "$OUT_DIR/pk_2lpt.json"
extract_positions "$OUT_DIR/1lpt_pm"    "$OUT_DIR/pos_1lpt.json"
extract_positions "$OUT_DIR/2lpt_pm"    "$OUT_DIR/pos_2lpt.json"
extract_diag      "$OUT_DIR/1lpt_pm"    "$OUT_DIR/diag_1lpt.json"
extract_diag      "$OUT_DIR/2lpt_pm"    "$OUT_DIR/diag_2lpt.json"
extract_diag      "$OUT_DIR/2lpt_treepm" "$OUT_DIR/diag_2lpt_treepm.json"

echo ""

# ── Scripts de validación Python ─────────────────────────────────────────────

echo "━━━ Validación Python ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if command -v python3 &>/dev/null; then
    # 1. Comparación 1LPT vs 2LPT (posiciones y P(k))
    if [ -f "$OUT_DIR/pos_1lpt.json" ] && [ -f "$OUT_DIR/pos_2lpt.json" ] \
       && [ -f "$OUT_DIR/pk_1lpt.json" ] && [ -f "$OUT_DIR/pk_2lpt.json" ]; then
        echo "  Corriendo compare_1lpt_2lpt.py..."
        python3 "$PY_SCRIPTS/compare_1lpt_2lpt.py" \
            --pos1lpt "$OUT_DIR/pos_1lpt.json" \
            --pos2lpt "$OUT_DIR/pos_2lpt.json" \
            --pk1lpt  "$OUT_DIR/pk_1lpt.json" \
            --pk2lpt  "$OUT_DIR/pk_2lpt.json" \
            --box     100.0 \
            --n       32 \
            --out     "$OUT_DIR/fig_compare_1lpt_2lpt.png"
    else
        echo "  ⚠ Faltan archivos de posiciones o P(k) — omitiendo compare_1lpt_2lpt.py"
    fi

    # 2. Crecimiento lineal 1LPT vs 2LPT
    if [ -f "$OUT_DIR/diag_1lpt.json" ] && [ -f "$OUT_DIR/diag_2lpt.json" ]; then
        echo "  Corriendo plot_growth.py..."
        python3 "$PY_SCRIPTS/plot_growth.py" \
            --diag1lpt "$OUT_DIR/diag_1lpt.json" \
            --diag2lpt "$OUT_DIR/diag_2lpt.json" \
            --omega_m  0.315 \
            --omega_l  0.685 \
            --a_init   0.02 \
            --out      "$OUT_DIR/fig_growth_1lpt_2lpt.png"
    else
        echo "  ⚠ Faltan diagnósticos — omitiendo plot_growth.py"
    fi
else
    echo "  ⚠ python3 no disponible — omitiendo validación Python"
fi

echo ""
echo "═══ Resumen Phase 28 ══════════════════════════════════════════"
echo ""
echo "  Figuras generadas en: $OUT_DIR/"
ls -la "$OUT_DIR"/fig_*.png 2>/dev/null || echo "  (ninguna figura generada)"
echo ""
echo "  Logs de simulación:"
for log in "$OUT_DIR"/*/run.log; do
    if [ -f "$log" ]; then
        dir_name=$(basename "$(dirname "$log")")
        echo "  [$dir_name]"
        grep -E "(ERROR|WARN|completed|steps|2LPT)" "$log" | tail -3 || true
    fi
done
echo ""
echo "━━━ Métricas clave a verificar ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  1. |Ψ²|_rms / |Ψ¹|_rms ≈ 0.01–0.05 (corrección subleading)"
echo "  2. P(k)[2LPT] / P(k)[1LPT] ≈ 1 para k < k_Nyq"
echo "  3. δ_rms(a) / D₁(a) ≈ cte (crescimiento lineal respetado)"
echo "  4. Sin NaN/Inf en posición y velocidad"
echo ""
echo "✓ Phase 28 completada"
