#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_production_256.sh — corrida de producción ΛCDM N=256³ con checkpointing
#
# Características:
#   - Detecta y reanuda desde checkpoint si existe
#   - Guarda checkpoint automático cada 2 h (configurable)
#   - Soporte MPI: usa mpirun si N_RANKS > 1
#   - Post-proceso opcional: P(k), ξ(r), HMF por snapshot
#   - Logging con timestamp a runs/production_256/run.log
#
# Variables de entorno:
#   N_RANKS         Número de ranks MPI (default: 1)
#   CONFIG          Archivo TOML de configuración (default: configs/production_256.toml)
#   OUT_DIR         Directorio de salida (default: runs/production_256)
#   POSTPROCESS     Si "1", ejecutar análisis Python tras cada snapshot (default: 0)
#   SKIP_BUILD      Si "1", omitir cargo build --release (default: 0)
#
# Uso:
#   bash scripts/run_production_256.sh
#   N_RANKS=4 bash scripts/run_production_256.sh
#   CONFIG=configs/mi_config.toml bash scripts/run_production_256.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Configuración ──────────────────────────────────────────────────────────
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

N_RANKS="${N_RANKS:-1}"
CONFIG="${CONFIG:-configs/production_256.toml}"
OUT_DIR="${OUT_DIR:-runs/production_256}"
POSTPROCESS="${POSTPROCESS:-0}"
SKIP_BUILD="${SKIP_BUILD:-0}"
BINARY="target/release/gadget-ng"
LOG_FILE="${OUT_DIR}/run.log"

# ── Colores ────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; NC='\033[0m'

log() { echo -e "${CYAN}[$(date '+%H:%M:%S')]${NC} $*" | tee -a "$LOG_FILE"; }
ok()  { echo -e "${GREEN}[✓]${NC} $*" | tee -a "$LOG_FILE"; }
warn(){ echo -e "${YELLOW}[!]${NC} $*" | tee -a "$LOG_FILE"; }
err() { echo -e "${RED}[✗]${NC} $*" | tee -a "$LOG_FILE"; exit 1; }

# ── Setup ──────────────────────────────────────────────────────────────────
mkdir -p "$OUT_DIR/frames" "$OUT_DIR/insitu" "$OUT_DIR/analysis"

echo "" | tee -a "$LOG_FILE"
log "════════════════════════════════════════════════════════"
log " gadget-ng — corrida de producción N=256³"
log " Config : $CONFIG"
log " Out    : $OUT_DIR"
log " Ranks  : $N_RANKS"
log " Fecha  : $(date '+%Y-%m-%d %H:%M:%S')"
log "════════════════════════════════════════════════════════"

# ── Build ──────────────────────────────────────────────────────────────────
if [ "${SKIP_BUILD}" != "1" ]; then
    log "Compilando en modo release..."
    cargo build --release --bin gadget-ng 2>&1 | tee -a "$LOG_FILE"
    ok "Build OK"
else
    warn "SKIP_BUILD=1 — usando binario existente"
fi

[ -f "$BINARY" ] || err "Binario no encontrado: $BINARY"

# ── Detectar checkpoint ────────────────────────────────────────────────────
CHECKPOINT_FLAG=""
if ls "${OUT_DIR}"/*.checkpoint 2>/dev/null | head -1 | grep -q "checkpoint"; then
    LATEST_CKPT=$(ls -t "${OUT_DIR}"/*.checkpoint | head -1)
    warn "Checkpoint detectado: $LATEST_CKPT"
    warn "Reanudando desde checkpoint..."
    CHECKPOINT_FLAG="--checkpoint $LATEST_CKPT"
else
    log "No hay checkpoint previo — corrida desde z=49"
fi

# ── Comando base ───────────────────────────────────────────────────────────
CMD="$BINARY run --config $CONFIG --out-dir $OUT_DIR $CHECKPOINT_FLAG"

if [ "$N_RANKS" -gt 1 ]; then
    if ! command -v mpirun &>/dev/null; then
        err "N_RANKS=$N_RANKS pero mpirun no encontrado. Instalar OpenMPI o MPICH."
    fi
    CMD="mpirun -n $N_RANKS $CMD"
    log "Modo MPI: $N_RANKS ranks"
else
    log "Modo serial (N_RANKS=1)"
fi

# ── Estimación de tiempo ───────────────────────────────────────────────────
log "Estimación de tiempo de corrida:"
log "  N=256³, TreePM + SFC + block timesteps"
log "  1 CPU: ~8-12 h  |  4 CPU MPI: ~3-4 h  |  1 GPU CUDA: ~2-4 h"

# ── Ejecución ──────────────────────────────────────────────────────────────
log "Ejecutando: $CMD"
START_TIME=$(date +%s)

if $CMD 2>&1 | tee -a "$LOG_FILE"; then
    END_TIME=$(date +%s)
    ELAPSED=$(( END_TIME - START_TIME ))
    ELAPSED_H=$(( ELAPSED / 3600 ))
    ELAPSED_M=$(( (ELAPSED % 3600) / 60 ))
    ok "Simulación completada en ${ELAPSED_H}h ${ELAPSED_M}m"
else
    err "La simulación falló. Revisar $LOG_FILE"
fi

# ── Post-proceso ───────────────────────────────────────────────────────────
if [ "${POSTPROCESS}" = "1" ]; then
    log "Ejecutando post-proceso..."

    if ! command -v python3 &>/dev/null; then
        warn "python3 no encontrado — omitiendo post-proceso"
    else
        NOTEBOOKS_DIR="docs/scripts"
        if [ -d "$NOTEBOOKS_DIR" ]; then
            log "Post-procesando P(k)..."
            python3 "$NOTEBOOKS_DIR/postprocess_pk.py" \
                --snapshots "${OUT_DIR}/frames" \
                --out "${OUT_DIR}/analysis/pk_evolution.json" \
                2>&1 | tee -a "$LOG_FILE" || warn "pk failed (no crítico)"

            log "Post-procesando HMF..."
            python3 "$NOTEBOOKS_DIR/postprocess_hmf.py" \
                --insitu "${OUT_DIR}/insitu" \
                --out "${OUT_DIR}/analysis/hmf_evolution.json" \
                2>&1 | tee -a "$LOG_FILE" || warn "hmf failed (no crítico)"
        else
            warn "Directorio $NOTEBOOKS_DIR no encontrado — omitiendo post-proceso"
        fi
    fi
fi

# ── Resumen final ──────────────────────────────────────────────────────────
log "════════════════════════════════════════════════════════"
log " Archivos generados en $OUT_DIR:"
log "  frames/     — snapshots HDF5 por paso"
log "  insitu/     — análisis in-situ (P(k), FoF, ξ(r))"
log "  analysis/   — post-proceso Python (si POSTPROCESS=1)"
log "  run.log     — log completo de la corrida"
log "════════════════════════════════════════════════════════"
ok "run_production_256.sh finalizado"
