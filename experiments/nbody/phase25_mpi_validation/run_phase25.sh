#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Phase 25: Benchmarks MPI reales — Fase 23 (clone+migrate) vs Fase 24 (scatter/gather PM)
#
# Uso:
#   ./run_phase25.sh [--mpi]            # --mpi: compilar y ejecutar con MPI
#   ./run_phase25.sh                    # sin --mpi: ejecutar serial (P=1 only)
#
# Requiere:
#   - Rust toolchain con cargo
#   - mpirun disponible en PATH (solo si --mpi)
#   - ~2 min de CPU (serial), ~5 min (P=4 MPI)
#
# Salida: experiments/nbody/phase25_mpi_validation/results/{variante}_N{N}_P{P}/
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT_DIR="$REPO_ROOT/experiments/nbody/phase25_mpi_validation"
CONFIGS_DIR="$SCRIPT_DIR/configs"
RESULTS_DIR="$SCRIPT_DIR/results"

# ── Opciones ─────────────────────────────────────────────────────────────────
USE_MPI=false
for arg in "$@"; do
    case "$arg" in
        --mpi) USE_MPI=true ;;
        *) echo "Opción desconocida: $arg"; exit 1 ;;
    esac
done

# ── Compilación ──────────────────────────────────────────────────────────────
cd "$REPO_ROOT"

if $USE_MPI; then
    echo "=== Compilando con --features mpi (release) ==="
    cargo build --release --features mpi -p gadget-ng-cli 2>&1
    BINARY="$REPO_ROOT/target/release/gadget-ng"
    LAUNCH_PREFIX="mpirun -n"
    P_VALUES=(1 2 4)
else
    echo "=== Compilando sin MPI (release) ==="
    cargo build --release -p gadget-ng-cli 2>&1
    BINARY="$REPO_ROOT/target/release/gadget-ng"
    LAUNCH_PREFIX=""
    P_VALUES=(1)
fi

echo "Binary: $BINARY"
echo "MPI mode: $USE_MPI"
echo ""

# ── Matriz de benchmarks ─────────────────────────────────────────────────────
# (variante, config_base, N)
declare -a CONFIGS=(
    "fase23 eds_N512_fase23   512"
    "fase24 eds_N512_fase24   512"
    "fase23 eds_N1000_fase23  1000"
    "fase24 eds_N1000_fase24  1000"
    "fase23 lcdm_N2000_fase23 2000"
    "fase24 lcdm_N2000_fase24 2000"
)

mkdir -p "$RESULTS_DIR"

total_runs=$(( ${#CONFIGS[@]} * ${#P_VALUES[@]} ))
run_num=0

for entry in "${CONFIGS[@]}"; do
    read -r variante config_name N <<< "$entry"
    config_file="$CONFIGS_DIR/${config_name}.toml"

    for P in "${P_VALUES[@]}"; do
        run_num=$(( run_num + 1 ))
        run_id="${variante}_N${N}_P${P}"
        out_dir="$RESULTS_DIR/$run_id"

        echo "─── [$run_num/$total_runs] $run_id ───────────────────────────────────"
        mkdir -p "$out_dir"

        t_start=$(date +%s%N)

        if $USE_MPI && [ "$P" -gt 1 ]; then
            $LAUNCH_PREFIX "$P" "$BINARY" stepping \
                --config "$config_file" \
                --out "$out_dir" \
                2>&1 | tee "$out_dir/run.log"
        else
            "$BINARY" stepping \
                --config "$config_file" \
                --out "$out_dir" \
                2>&1 | tee "$out_dir/run.log"
        fi

        t_end=$(date +%s%N)
        wall_ms=$(( (t_end - t_start) / 1000000 ))
        echo "  Completado en ${wall_ms} ms"
        echo ""

        # Guardar metadata del run
        cat > "$out_dir/run_meta.json" <<EOF
{
  "run_id":    "$run_id",
  "variante":  "$variante",
  "N":         $N,
  "P":         $P,
  "config":    "$config_name",
  "mpi_mode":  $USE_MPI,
  "wall_ms":   $wall_ms
}
EOF
    done
done

echo "════════════════════════════════════════════════════════════════"
echo "Todos los benchmarks completados."
echo "Resultados en: $RESULTS_DIR"
echo ""
echo "Siguiente paso:"
echo "  python3 $SCRIPT_DIR/scripts/compare_phase25.py"
echo "════════════════════════════════════════════════════════════════"
