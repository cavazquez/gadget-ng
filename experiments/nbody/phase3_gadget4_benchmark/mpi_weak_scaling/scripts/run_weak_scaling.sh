#!/usr/bin/env bash
# run_weak_scaling.sh — Weak scaling: N proporcional al número de ranks.
#
# Configuración: N = 1000 × ranks, BH θ=0.5, 50 pasos.
# Ideal weak scaling → tiempo constante al aumentar ranks con N ∝ ranks.
#
# Uso:
#   cd <repo_root>
#   bash experiments/nbody/phase3_gadget4_benchmark/mpi_weak_scaling/scripts/run_weak_scaling.sh
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$REPO_ROOT"

EXP_DIR="experiments/nbody/phase3_gadget4_benchmark/mpi_weak_scaling"
RESULTS_DIR="$EXP_DIR/results"
CONFIGS_DIR="$EXP_DIR/configs"
mkdir -p "$RESULTS_DIR" "$CONFIGS_DIR"

N_PER_RANK=1000
STEPS=50

MAX_CORES=$(nproc --all 2>/dev/null || echo 4)
echo "[weak_scaling] Cores disponibles: $MAX_CORES"
echo "[weak_scaling] N por rank: $N_PER_RANK, pasos: $STEPS"

generate_config() {
    local n=$1
    local out_path=$2
    cat > "$out_path" << EOF
[simulation]
particle_count         = $n
box_size               = $(python3 -c "import math; print(f'{20.0 * math.cbrt($n / 1000.0):.2f}')")
dt                     = 0.01
num_steps              = $STEPS
softening              = 0.05
gravitational_constant = 1.0
seed                   = 42

[initial_conditions]
kind = { plummer = { a = 1.0 } }

[gravity]
solver = "barnes_hut"
theta  = 0.5

[output]
snapshot_interval   = 0
checkpoint_interval = 0

[performance]
deterministic = true
EOF
}

if command -v mpirun &>/dev/null; then
    cargo build -p gadget-ng-cli --features mpi --release 2>&1 | grep -E "Compiling|Finished|error"
    BIN="./target/release/gadget-ng"

    RANKS=(1 2 4)
    echo "rank,N,total_wall_s,mean_step_ms,gravity_frac,comm_frac,weak_efficiency" \
        > "$RESULTS_DIR/weak_timing_raw.csv"

    T1=""
    for NRANKS in "${RANKS[@]}"; do
        if [ "$NRANKS" -gt "$MAX_CORES" ]; then
            echo "[weak_scaling] Omitiendo ranks=$NRANKS (máx=$MAX_CORES)"
            continue
        fi
        N=$(( NRANKS * N_PER_RANK ))
        CONFIG="$CONFIGS_DIR/weak_N${N}_r${NRANKS}.toml"
        OUT_DIR="$EXP_DIR/runs/ranks_${NRANKS}_N${N}"

        generate_config "$N" "$CONFIG"
        echo "[weak_scaling] Ejecutando ranks=$NRANKS N=$N..."
        mpirun -n "$NRANKS" "$BIN" stepping --config "$CONFIG" --out "$OUT_DIR"

        if [ -f "$OUT_DIR/timings.json" ]; then
            ROW=$(python3 -c "
import json
d = json.load(open('$OUT_DIR/timings.json'))
w = d['total_wall_s']
ms = d['mean_step_wall_s']*1000
gf = d.get('gravity_fraction',0)
cf = d.get('comm_fraction',0)
print(f'{$NRANKS},{$N},{w:.6f},{ms:.4f},{gf:.6f},{cf:.6f}')
")
            # Calcular weak efficiency usando T(1) como referencia.
            if [ -z "$T1" ]; then
                T1=$(python3 -c "import json; d=json.load(open('$OUT_DIR/timings.json')); print(d['total_wall_s'])")
            fi
            WE=$(python3 -c "
import json
d = json.load(open('$OUT_DIR/timings.json'))
t1 = $T1
we = t1 / d['total_wall_s'] * 100
print(f'{we:.2f}')
")
            echo "${ROW},${WE}" >> "$RESULTS_DIR/weak_timing_raw.csv"
            echo "  → ranks=$NRANKS N=$N done (weak_eff=${WE}%)"
        fi
    done
else
    echo "[weak_scaling] WARN: mpirun no disponible. Ejecutando solo 1 rank con distintos N."
    cargo build -p gadget-ng-cli --release 2>&1 | grep -E "Compiling|Finished|error"
    BIN="./target/release/gadget-ng"

    echo "rank,N,total_wall_s,mean_step_ms,gravity_frac,comm_frac,weak_efficiency" \
        > "$RESULTS_DIR/weak_timing_raw.csv"

    N_LIST=(1000 2000 4000)
    T1=""
    for N in "${N_LIST[@]}"; do
        CONFIG="$CONFIGS_DIR/weak_N${N}_r1.toml"
        OUT_DIR="$EXP_DIR/runs/ranks_1_N${N}"
        generate_config "$N" "$CONFIG"
        echo "[weak_scaling] Ejecutando N=$N (1 rank)..."
        "$BIN" stepping --config "$CONFIG" --out "$OUT_DIR"

        if [ -f "$OUT_DIR/timings.json" ]; then
            if [ -z "$T1" ]; then
                T1=$(python3 -c "import json; d=json.load(open('$OUT_DIR/timings.json')); print(d['total_wall_s'])")
            fi
            python3 -c "
import json
d = json.load(open('$OUT_DIR/timings.json'))
t1 = $T1
we = t1 / d['total_wall_s'] * 100
print(f\"1,{$N},{d['total_wall_s']:.6f},{d['mean_step_wall_s']*1000:.4f},{d.get('gravity_fraction',0):.6f},{d.get('comm_fraction',0):.6f},{we:.2f}\")
" >> "$RESULTS_DIR/weak_timing_raw.csv"
        fi
    done
fi

echo ""
echo "[weak_scaling] Resultados en $RESULTS_DIR/weak_timing_raw.csv"
echo "Para analizar:"
echo "  python3 $EXP_DIR/scripts/analyze_weak_scaling.py"
