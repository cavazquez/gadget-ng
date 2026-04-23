#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Phase 79 — Script de corrida de validación N=128³
#
# Uso:
#   ./scripts/run_validation_128.sh [--resume] [--mpi N_RANKS] [--post]
#
# Opciones:
#   --resume        Reanudar desde el último checkpoint disponible
#   --mpi N_RANKS   Usar mpirun con N_RANKS procesos (default: 1)
#   --post          Ejecutar post-proceso Python al finalizar
#
# Variables de entorno:
#   GADGET_BIN      Ruta al binario gadget-ng (default: target/release/gadget-ng)
#   OUT_DIR         Directorio de salida (default: runs/validation_128)
#   LOG_DIR         Directorio de logs (default: runs/validation_128/logs)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

# ── Configuración ────────────────────────────────────────────────────────────
GADGET_BIN="${GADGET_BIN:-target/release/gadget-ng}"
OUT_DIR="${OUT_DIR:-runs/validation_128}"
LOG_DIR="${LOG_DIR:-${OUT_DIR}/logs}"
CONFIG="configs/validation_128.toml"
CHECKPOINT_DIR="${OUT_DIR}/checkpoints"
ANALYSIS_DIR="${OUT_DIR}/analysis"

N_RANKS=1
DO_RESUME=false
DO_POST=false

# ── Parsear argumentos ────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --resume)     DO_RESUME=true; shift ;;
        --mpi)        N_RANKS="$2"; shift 2 ;;
        --post)       DO_POST=true; shift ;;
        *) echo "Opción desconocida: $1"; exit 1 ;;
    esac
done

# ── Preparar directorios ──────────────────────────────────────────────────────
mkdir -p "$OUT_DIR" "$LOG_DIR" "$CHECKPOINT_DIR" "$ANALYSIS_DIR"

LOG_FILE="${LOG_DIR}/run_$(date +%Y%m%d_%H%M%S).log"
echo "[$(date)] Iniciando validación N=128³" | tee "$LOG_FILE"
echo "[$(date)] Config: $CONFIG" | tee -a "$LOG_FILE"
echo "[$(date)] Ranks MPI: $N_RANKS" | tee -a "$LOG_FILE"

# ── Compilar si es necesario ──────────────────────────────────────────────────
if [[ ! -f "$GADGET_BIN" ]]; then
    echo "[$(date)] Compilando gadget-ng (release)..." | tee -a "$LOG_FILE"
    cargo build --release -p gadget-ng-cli 2>&1 | tee -a "$LOG_FILE"
fi

# ── Detectar checkpoint para resume ──────────────────────────────────────────
RESUME_FLAG=""
if [[ "$DO_RESUME" == true ]]; then
    LATEST_CKPT=$(ls -t "${CHECKPOINT_DIR}"/checkpoint_*.toml 2>/dev/null | head -1 || true)
    if [[ -n "$LATEST_CKPT" ]]; then
        echo "[$(date)] Reanudando desde: $LATEST_CKPT" | tee -a "$LOG_FILE"
        RESUME_FLAG="--resume $LATEST_CKPT"
    else
        echo "[$(date)] No hay checkpoints disponibles, iniciando desde z=49" | tee -a "$LOG_FILE"
    fi
fi

# ── Ejecutar simulación ───────────────────────────────────────────────────────
echo "[$(date)] Iniciando simulación..." | tee -a "$LOG_FILE"

if [[ "$N_RANKS" -gt 1 ]]; then
    MPI_CMD="mpirun -n $N_RANKS"
else
    MPI_CMD=""
fi

TIME_START=$(date +%s)

$MPI_CMD "$GADGET_BIN" stepping \
    --config "$CONFIG" \
    $RESUME_FLAG \
    2>&1 | tee -a "$LOG_FILE"

TIME_END=$(date +%s)
ELAPSED=$((TIME_END - TIME_START))
echo "[$(date)] Simulación completada en ${ELAPSED}s ($(( ELAPSED / 3600 ))h $(( (ELAPSED % 3600) / 60 ))m)" | tee -a "$LOG_FILE"

# ── Post-proceso Python ───────────────────────────────────────────────────────
if [[ "$DO_POST" == true ]]; then
    echo "[$(date)] Ejecutando post-proceso..." | tee -a "$LOG_FILE"

    # P(k) vs CLASS
    if [[ -f "docs/notebooks/validate_pk_hmf.py" ]]; then
        python3 docs/notebooks/validate_pk_hmf.py \
            --analysis-dir "$ANALYSIS_DIR" \
            --out-dir "${OUT_DIR}/plots" \
            2>&1 | tee -a "$LOG_FILE" || echo "[WARN] Post-proceso falló (requiere numpy/matplotlib)"
    fi

    # P(k) evolución
    if [[ -f "docs/notebooks/postprocess_pk.py" ]]; then
        python3 docs/notebooks/postprocess_pk.py "$ANALYSIS_DIR" 2>&1 | tee -a "$LOG_FILE" || true
    fi

    echo "[$(date)] Post-proceso completado." | tee -a "$LOG_FILE"
fi

echo "[$(date)] ✓ Validación N=128³ completada. Resultados en: $OUT_DIR" | tee -a "$LOG_FILE"
