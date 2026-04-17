#!/usr/bin/env bash
# run_strong_scaling.sh — Strong scaling: N=1000 fijo, variar ranks.
#
# Uso:
#   cd <repo_root>
#   bash experiments/nbody/phase3_gadget4_benchmark/mpi_strong_scaling/scripts/run_strong_scaling.sh
#
# Requiere mpirun. Sin MPI, ejecuta solo el caso de 1 rank (serial).
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$REPO_ROOT"

EXP_DIR="experiments/nbody/phase3_gadget4_benchmark/mpi_strong_scaling"
CONFIG="$EXP_DIR/config/strong_N1000.toml"
RESULTS_DIR="$EXP_DIR/results"
mkdir -p "$RESULTS_DIR"

# Detectar número máximo de cores disponibles.
MAX_CORES=$(nproc --all 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
echo "[strong_scaling] Cores disponibles: $MAX_CORES"

if command -v mpirun &>/dev/null; then
    echo "[strong_scaling] Compilando con feature MPI..."
    cargo build -p gadget-ng-cli --features mpi --release 2>&1 | grep -E "Compiling|Finished|error"
    BIN="./target/release/gadget-ng"

    # Lista de ranks a probar (hasta el número de cores disponibles).
    RANKS=(1 2 4 8)

    echo ""
    echo "rank,total_wall_s,mean_step_ms,gravity_frac,comm_frac" > "$RESULTS_DIR/strong_timing_raw.csv"

    for NRANKS in "${RANKS[@]}"; do
        if [ "$NRANKS" -gt "$MAX_CORES" ]; then
            echo "[strong_scaling] Omitiendo ranks=$NRANKS (máx. cores=$MAX_CORES)"
            continue
        fi
        OUT_DIR="$EXP_DIR/runs/ranks_$NRANKS"
        echo "[strong_scaling] Ejecutando con $NRANKS ranks..."
        mpirun -n "$NRANKS" "$BIN" stepping \
            --config "$CONFIG" \
            --out    "$OUT_DIR"

        if [ -f "$OUT_DIR/timings.json" ]; then
            python3 -c "
import json, sys
d = json.load(open('$OUT_DIR/timings.json'))
print(f\"{$NRANKS},{d['total_wall_s']:.6f},{d['mean_step_wall_s']*1000:.4f},{d.get('gravity_fraction',0):.6f},{d.get('comm_fraction',0):.6f}\")
" >> "$RESULTS_DIR/strong_timing_raw.csv"
            echo "  → ranks=$NRANKS done"
        else
            echo "  WARN: timings.json no generado para ranks=$NRANKS"
        fi
    done
else
    echo "[strong_scaling] WARN: mpirun no disponible."
    echo "  Ejecutando solo con 1 rank (serial)."
    cargo build -p gadget-ng-cli --release 2>&1 | grep -E "Compiling|Finished|error"
    BIN="./target/release/gadget-ng"
    OUT_DIR="$EXP_DIR/runs/ranks_1"
    "$BIN" stepping --config "$CONFIG" --out "$OUT_DIR"

    echo "rank,total_wall_s,mean_step_ms,gravity_frac,comm_frac" > "$RESULTS_DIR/strong_timing_raw.csv"
    if [ -f "$OUT_DIR/timings.json" ]; then
        python3 -c "
import json
d = json.load(open('$OUT_DIR/timings.json'))
print(f\"1,{d['total_wall_s']:.6f},{d['mean_step_wall_s']*1000:.4f},{d.get('gravity_fraction',0):.6f},{d.get('comm_fraction',0):.6f}\")
" >> "$RESULTS_DIR/strong_timing_raw.csv"
    fi
fi

echo ""
echo "[strong_scaling] Resultados en $RESULTS_DIR/strong_timing_raw.csv"
echo "Para analizar y plotear:"
echo "  python3 $EXP_DIR/scripts/analyze_strong_scaling.py"
