#!/usr/bin/env bash
# ── run_phase29.sh — Experimentos de validación Fase 29: 1LPT vs 2LPT ────────
#
# Pregunta central:
#   ¿Cuánto reduce 2LPT los transientes y errores sistemáticos de 1LPT en
#   gadget-ng, y en qué régimen esa mejora se vuelve relevante?
#
# Diseño experimental:
#   Todos los casos usan:
#     N = 32³,  seed = 12345,  σ₈ = 0.8 (EH),  caja = 100 Mpc/h
#     dt = 0.0004,  num_steps = 50
#   Variaciones:
#     A. a_init = 0.02 (z≈49, control)   | 1LPT vs 2LPT | PM
#     B. a_init = 0.05 (z≈19, intermedio)| 1LPT vs 2LPT | PM
#     C. a_init = 0.10 (z≈9,  tardío)    | 1LPT vs 2LPT | PM
#     D. a_init = 0.02, 0.05              | 2LPT         | TreePM
#
# Nota sobre a_init en gadget-ng:
#   a_init solo afecta las VELOCIDADES (p = a²·f·H·Ψ). Las posiciones se
#   fijan por la normalización σ₈, independientemente de a_init.
#   El efecto de "inicio tardío" se manifiesta a través de la corrección de
#   velocidad 2LPT y la evolución dinámica resultante.
#
# Uso:
#   bash run_phase29.sh [--out-dir ./output]
#   GADGET_BINARY=/ruta/al/binario bash run_phase29.sh
#
# Prerrequisitos:
#   - gadget-ng compilado:  cargo build --release -p gadget-ng
#   - Python 3 con numpy, matplotlib (scipy opcional para tests σ₈)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
BINARY="${GADGET_BINARY:-$REPO_ROOT/target/release/gadget-ng}"
OUT_BASE="${1:-$SCRIPT_DIR/output}"
PY_SCRIPTS="$SCRIPT_DIR/scripts"
CONFIGS="$SCRIPT_DIR/configs"

# ── Verificaciones previas ────────────────────────────────────────────────────

if [ ! -f "$BINARY" ]; then
    echo "ERROR: Binario no encontrado: $BINARY"
    echo "       Compilar con: cd $REPO_ROOT && cargo build --release -p gadget-ng"
    exit 1
fi

# Crear directorios de salida para todos los casos
CASES=(
    "a002_1lpt_pm"
    "a002_2lpt_pm"
    "a005_1lpt_pm"
    "a005_2lpt_pm"
    "a010_1lpt_pm"
    "a010_2lpt_pm"
    "a002_2lpt_treepm"
    "a005_2lpt_treepm"
)
for case in "${CASES[@]}"; do
    mkdir -p "$OUT_BASE/$case"
done

echo "========================================================================"
echo "Fase 29: Validación Física Comparativa 1LPT vs 2LPT"
echo "========================================================================"
echo "Binario:  $BINARY"
echo "Salida:   $OUT_BASE"
echo "Casos:    ${#CASES[@]}"
echo ""

# ── Función auxiliar: correr simulación ──────────────────────────────────────

run_sim() {
    local label="$1"
    local config_name="$2"
    local out_dir="$3"
    local config_path="$CONFIGS/$config_name"

    if [ ! -f "$config_path" ]; then
        echo "  WARN: config no encontrado: $config_path — omitiendo $label"
        return 0
    fi

    echo "━━━ $label ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    "$BINARY" --config "$config_path" --output-dir "$out_dir" 2>&1 | tee "$out_dir/run.log"
    echo "    Completado: $label"
    echo ""
}

# ── Función auxiliar: extraer archivos de datos ───────────────────────────────

extract_first() {
    local snap_dir="$1"
    local pattern="$2"
    local out_file="$3"
    local first
    first=$(find "$snap_dir" -name "$pattern" | sort | head -n 1 2>/dev/null || true)
    if [ -n "$first" ]; then
        cp "$first" "$out_file"
        echo "  Extraído: $(basename "$first") → $(basename "$out_file")"
    fi
}

extract_last() {
    local snap_dir="$1"
    local pattern="$2"
    local out_file="$3"
    local last
    last=$(find "$snap_dir" -name "$pattern" | sort | tail -n 1 2>/dev/null || true)
    if [ -n "$last" ]; then
        cp "$last" "$out_file"
        echo "  Extraído: $(basename "$last") → $(basename "$out_file")"
    fi
}

extract_diag() {
    local snap_dir="$1"
    local out_file="$2"
    local diag
    diag=$(find "$snap_dir" -name "diagnostics*.json" | head -n 1 2>/dev/null || true)
    if [ -n "$diag" ]; then
        cp "$diag" "$out_file"
        echo "  Diagnósticos: $(basename "$diag") → $(basename "$out_file")"
    fi
}

# ════════════════════════════════════════════════════════════════════════════
# Bloque A: a_init = 0.02 (z≈49) — control
# ════════════════════════════════════════════════════════════════════════════

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Bloque A: a_init = 0.02 (z ≈ 49) — inicio temprano (control)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

run_sim "A1: 1LPT + PM  (a_init=0.02)" "lcdm_N32_a002_1lpt_pm.toml"    "$OUT_BASE/a002_1lpt_pm"
run_sim "A2: 2LPT + PM  (a_init=0.02)" "lcdm_N32_a002_2lpt_pm.toml"    "$OUT_BASE/a002_2lpt_pm"
run_sim "A3: 2LPT+TreePM(a_init=0.02)" "lcdm_N32_a002_2lpt_treepm.toml" "$OUT_BASE/a002_2lpt_treepm"

# ════════════════════════════════════════════════════════════════════════════
# Bloque B: a_init = 0.05 (z≈19) — intermedio
# ════════════════════════════════════════════════════════════════════════════

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Bloque B: a_init = 0.05 (z ≈ 19) — inicio intermedio"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

run_sim "B1: 1LPT + PM  (a_init=0.05)" "lcdm_N32_a005_1lpt_pm.toml"    "$OUT_BASE/a005_1lpt_pm"
run_sim "B2: 2LPT + PM  (a_init=0.05)" "lcdm_N32_a005_2lpt_pm.toml"    "$OUT_BASE/a005_2lpt_pm"
run_sim "B3: 2LPT+TreePM(a_init=0.05)" "lcdm_N32_a005_2lpt_treepm.toml" "$OUT_BASE/a005_2lpt_treepm"

# ════════════════════════════════════════════════════════════════════════════
# Bloque C: a_init = 0.10 (z≈9) — tardío
# ════════════════════════════════════════════════════════════════════════════

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Bloque C: a_init = 0.10 (z ≈ 9) — inicio tardío"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

run_sim "C1: 1LPT + PM  (a_init=0.10)" "lcdm_N32_a010_1lpt_pm.toml"    "$OUT_BASE/a010_1lpt_pm"
run_sim "C2: 2LPT + PM  (a_init=0.10)" "lcdm_N32_a010_2lpt_pm.toml"    "$OUT_BASE/a010_2lpt_pm"

# ════════════════════════════════════════════════════════════════════════════
# Extracción de datos
# ════════════════════════════════════════════════════════════════════════════

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Extrayendo datos de simulaciones..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

for ainit in a002 a005 a010; do
    for lpt in 1lpt 2lpt; do
        src_dir="$OUT_BASE/${ainit}_${lpt}_pm"
        if [ -d "$src_dir" ]; then
            extract_first "$src_dir" "positions_*.json" "$OUT_BASE/pos_${ainit}_${lpt}.json"
            extract_first "$src_dir" "pk_*.json"        "$OUT_BASE/pk_${ainit}_${lpt}_init.json"
            extract_last  "$src_dir" "pk_*.json"        "$OUT_BASE/pk_${ainit}_${lpt}_final.json"
            extract_diag  "$src_dir"                    "$OUT_BASE/diag_${ainit}_${lpt}.json"
        fi
    done
done

# TreePM
for ainit in a002 a005; do
    src_dir="$OUT_BASE/${ainit}_2lpt_treepm"
    if [ -d "$src_dir" ]; then
        extract_diag "$src_dir" "$OUT_BASE/diag_${ainit}_2lpt_treepm.json"
    fi
done

echo ""

# ════════════════════════════════════════════════════════════════════════════
# Scripts de validación Python
# ════════════════════════════════════════════════════════════════════════════

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Validación Python..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if ! command -v python3 &>/dev/null; then
    echo "  WARN: python3 no disponible — omitiendo validación Python"
else

  # ── Figura 1: Desplazamientos 1LPT vs 2LPT (por a_init) ──────────────────
  POS_FILES_1LPT=""
  POS_FILES_2LPT=""
  AINIT_LABELS=""
  for ainit in a002 a005 a010; do
      f1="$OUT_BASE/pos_${ainit}_1lpt.json"
      f2="$OUT_BASE/pos_${ainit}_2lpt.json"
      if [ -f "$f1" ] && [ -f "$f2" ]; then
          POS_FILES_1LPT="$POS_FILES_1LPT $f1"
          POS_FILES_2LPT="$POS_FILES_2LPT $f2"
          case "$ainit" in
              a002) AINIT_LABELS="$AINIT_LABELS a=0.02 (z≈49)" ;;
              a005) AINIT_LABELS="$AINIT_LABELS a=0.05 (z≈19)" ;;
              a010) AINIT_LABELS="$AINIT_LABELS a=0.10 (z≈9)"  ;;
          esac
      fi
  done

  if [ -n "$POS_FILES_1LPT" ]; then
      echo "  Corriendo plot_displacements.py..."
      python3 "$PY_SCRIPTS/plot_displacements.py" \
          --pos1lpt $POS_FILES_1LPT \
          --pos2lpt $POS_FILES_2LPT \
          --labels  $AINIT_LABELS \
          --grid 32 --box 1.0 \
          --out "$OUT_BASE/fig_displacements.png" || true
  else
      echo "  WARN: faltan archivos de posiciones — omitiendo figura de desplazamientos"
  fi

  # ── Figura 2: P(k) comparativo para a_init=0.02 ───────────────────────────
  PK1_INIT="$OUT_BASE/pk_a002_1lpt_init.json"
  PK2_INIT="$OUT_BASE/pk_a002_2lpt_init.json"
  if [ -f "$PK1_INIT" ] && [ -f "$PK2_INIT" ]; then
      echo "  Corriendo plot_pk.py (a_init=0.02)..."
      PK_ARGS="--pk1lpt $PK1_INIT --pk2lpt $PK2_INIT --box 100.0 --a-init 0.02"
      PK1_FINAL="$OUT_BASE/pk_a002_1lpt_final.json"
      PK2_FINAL="$OUT_BASE/pk_a002_2lpt_final.json"
      if [ -f "$PK1_FINAL" ] && [ -f "$PK2_FINAL" ]; then
          PK_ARGS="$PK_ARGS --pk1lpt-final $PK1_FINAL --pk2lpt-final $PK2_FINAL"
      fi
      python3 "$PY_SCRIPTS/plot_pk.py" \
          $PK_ARGS \
          --out "$OUT_BASE/fig_pk_a002.png" || true
  else
      echo "  WARN: faltan P(k) para a_init=0.02 — omitiendo figura espectral"
  fi

  # ── Figura 3: Crecimiento δ_rms(a) comparativo ────────────────────────────
  DIAG_1LPT_ARGS=""
  DIAG_2LPT_ARGS=""
  GROWTH_LABELS=""
  for ainit in a002 a005; do
      d1="$OUT_BASE/diag_${ainit}_1lpt.json"
      d2="$OUT_BASE/diag_${ainit}_2lpt.json"
      if [ -f "$d1" ] && [ -f "$d2" ]; then
          DIAG_1LPT_ARGS="$DIAG_1LPT_ARGS $d1"
          DIAG_2LPT_ARGS="$DIAG_2LPT_ARGS $d2"
          case "$ainit" in
              a002) GROWTH_LABELS="$GROWTH_LABELS a_init=0.02" ;;
              a005) GROWTH_LABELS="$GROWTH_LABELS a_init=0.05" ;;
          esac
      fi
  done

  if [ -n "$DIAG_1LPT_ARGS" ]; then
      echo "  Corriendo plot_growth.py..."
      python3 "$PY_SCRIPTS/plot_growth.py" \
          --diag1lpt $DIAG_1LPT_ARGS \
          --diag2lpt $DIAG_2LPT_ARGS \
          --labels   $GROWTH_LABELS \
          --omega-m 0.315 --omega-l 0.685 \
          --out "$OUT_BASE/fig_growth.png" || true
  else
      echo "  WARN: faltan diagnósticos — omitiendo figura de crecimiento"
  fi

fi  # fin if python3 disponible

# ════════════════════════════════════════════════════════════════════════════
# Resumen final
# ════════════════════════════════════════════════════════════════════════════

echo ""
echo "========================================================================"
echo "Resumen Phase 29 — 1LPT vs 2LPT"
echo "========================================================================"
echo ""
echo "Figuras generadas:"
ls -la "$OUT_BASE"/fig_*.png 2>/dev/null | awk '{print "  " $NF}' \
    || echo "  (ninguna figura generada)"
echo ""
echo "Métricas clave a verificar:"
echo "  1. |Ψ²|_rms / |Ψ¹|_rms ≈ 1–10%  (corrección subleading, > si σ₈ mayor)"
echo "  2. P(k) ratio 2LPT/1LPT < 5% para k < k_Nyq/2"
echo "  3. δ_rms(a) crecimiento: < 10% diferencia entre 1LPT y 2LPT"
echo "  4. v_rms: diferencia < 20% (corrección de velocidad 2LPT es subleading)"
echo "  5. PM ≈ TreePM: solver-independence de los resultados"
echo ""
echo "Logs de simulación:"
for case in "${CASES[@]}"; do
    log="$OUT_BASE/$case/run.log"
    if [ -f "$log" ]; then
        last=$(tail -3 "$log" 2>/dev/null || true)
        echo "  [$case]"
        echo "    $(echo "$last" | head -1)"
    fi
done
echo ""
echo "Para tests automáticos (no requieren binario):"
echo "  cargo test --package gadget-ng-physics --test phase29_lpt_comparison -- --nocapture"
echo ""
echo "========================================================================"
echo "Phase 29 completada"
echo "========================================================================"
