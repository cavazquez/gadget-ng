#!/usr/bin/env bash
# run_parity.sh — Compara resultados serial vs MPI deterministic vs MPI no-deterministic.
#
# Uso:
#   cd <repo_root>
#   bash experiments/nbody/phase3_gadget4_benchmark/serial_vs_mpi_parity/scripts/run_parity.sh
#
# Requiere: mpirun disponible (con feature "mpi") para los modos MPI.
# Si mpirun no está disponible, solo ejecuta el modo serial.
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$REPO_ROOT"

EXP_DIR="experiments/nbody/phase3_gadget4_benchmark/serial_vs_mpi_parity"
CONFIG="$EXP_DIR/config/parity.toml"

echo "[parity] Compilando serial (sin MPI)..."
cargo build -p gadget-ng-cli --release 2>&1 | grep -E "Compiling|Finished|error"
BIN_SERIAL="./target/release/gadget-ng"

echo ""
echo "[parity] Ejecutando: serial (1 rank)..."
"$BIN_SERIAL" stepping \
    --config "$CONFIG" \
    --out    "$EXP_DIR/runs/serial" \
    --snapshot

# Intentar MPI.
if command -v mpirun &>/dev/null; then
    echo ""
    echo "[parity] mpirun detectado. Compilando con feature MPI..."
    cargo build -p gadget-ng-cli --features mpi --release 2>&1 | grep -E "Compiling|Finished|error"
    BIN_MPI="./target/release/gadget-ng"

    echo ""
    echo "[parity] Ejecutando: MPI 2 ranks (deterministic)..."
    mpirun -n 2 "$BIN_MPI" stepping \
        --config "$CONFIG" \
        --out    "$EXP_DIR/runs/mpi_2rank_det" \
        --snapshot

    echo ""
    echo "[parity] Ejecutando: MPI 2 ranks (no-deterministic)..."
    # Crear config temporal sin deterministic.
    TMP_CFG=$(mktemp /tmp/parity_nondet_XXXX.toml)
    sed 's/deterministic = true/deterministic = false/' "$CONFIG" > "$TMP_CFG"
    mpirun -n 2 "$BIN_MPI" stepping \
        --config "$TMP_CFG" \
        --out    "$EXP_DIR/runs/mpi_2rank_nondet" \
        --snapshot
    rm -f "$TMP_CFG"

    echo ""
    echo "[parity] Ejecutando: MPI 4 ranks (deterministic)..."
    mpirun -n 4 "$BIN_MPI" stepping \
        --config "$CONFIG" \
        --out    "$EXP_DIR/runs/mpi_4rank_det" \
        --snapshot
else
    echo ""
    echo "[parity] WARN: mpirun no disponible en este sistema."
    echo "  Para experimentos MPI, instalar OpenMPI y compilar con --features mpi."
    echo "  El experimento de paridad serial-vs-serial-deterministic se puede hacer con:"
    echo "    cargo build -p gadget-ng-cli --release"
    echo "    (segunda ejecución serial produce resultado idéntico al bit)"
    echo ""
    echo "[parity] Ejecutando segunda corrida serial para verificar reproducibilidad..."
    "$BIN_SERIAL" stepping \
        --config "$CONFIG" \
        --out    "$EXP_DIR/runs/serial_run2" \
        --snapshot
fi

echo ""
echo "[parity] Runs completados en $EXP_DIR/runs/"
echo ""
echo "Para analizar:"
echo "  python3 $EXP_DIR/scripts/analyze_parity.py"
