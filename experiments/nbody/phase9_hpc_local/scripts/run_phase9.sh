#!/usr/bin/env bash
# Ejecuta benchmarks de Fase 9 (HPC local).
#
# Uso:
#   ./run_phase9.sh [--binary PATH] [--results DIR] [--max-procs P]
#
# Requiere:
#   - gadget-ng compilado con --features mpi (o sin mpi para P=1)
#   - mpirun disponible en el PATH
#   - python3 para generar configs

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PHASE9_DIR="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(cd "$PHASE9_DIR/../../../.." && pwd)"
CONFIG_DIR="$PHASE9_DIR/config"
RESULTS_DIR="${RESULTS_DIR:-$PHASE9_DIR/results}"
BINARY="${BINARY:-$REPO_ROOT/target/release/gadget-ng}"
MAX_PROCS="${MAX_PROCS:-4}"

mkdir -p "$RESULTS_DIR"

# Generar configs si no existen
if [ ! -d "$CONFIG_DIR" ] || [ -z "$(ls -A "$CONFIG_DIR" 2>/dev/null)" ]; then
    echo "[run_phase9] Generando configs..."
    python3 "$SCRIPT_DIR/generate_configs.py"
fi

# Compilar binario MPI release
echo "[run_phase9] Compilando gadget-ng con --features mpi (release)..."
cd "$REPO_ROOT"
cargo build --release --features mpi -p gadget-ng-cli 2>&1 | tail -5
BINARY="$REPO_ROOT/target/release/gadget-ng"

echo "[run_phase9] Binario: $BINARY"
echo "[run_phase9] Configs: $CONFIG_DIR"
echo "[run_phase9] Resultados: $RESULTS_DIR"
echo ""

# ── Strong scaling ───────────────────────────────────────────────────────────
for N in 2000 4000 8000; do
    for backend in allgather blocking overlap; do
        cfg="$CONFIG_DIR/N${N}_${backend}.toml"
        [ -f "$cfg" ] || continue

        for P in 1 2 4 8; do
            [ "$P" -le "$MAX_PROCS" ] || continue

            out_dir="$RESULTS_DIR/strong_N${N}_${backend}_P${P}"
            [ -d "$out_dir" ] && echo "[skip] $out_dir" && continue

            echo "[run] strong N=$N backend=$backend P=$P..."
            mkdir -p "$out_dir"

            t_start=$(date +%s%N)
            if [ "$P" -eq 1 ]; then
                "$BINARY" stepping --config "$cfg" --out "$out_dir" 2>"$out_dir/stderr.log"
            else
                mpirun -n "$P" "$BINARY" stepping --config "$cfg" --out "$out_dir" 2>"$out_dir/stderr.log"
            fi
            t_end=$(date +%s%N)
            wall_ms=$(( (t_end - t_start) / 1000000 ))
            echo "$wall_ms" > "$out_dir/wall_ms.txt"
            echo "  → ${wall_ms} ms"
        done
    done
done

# ── Weak scaling ─────────────────────────────────────────────────────────────
for np_ratio in 500 1000; do
    for P in 1 2 4 8; do
        [ "$P" -le "$MAX_PROCS" ] || continue
        N=$(( np_ratio * P ))

        for backend in blocking overlap; do
            cfg="$CONFIG_DIR/N${N}_${backend}_weak${np_ratio}_P${P}.toml"
            [ -f "$cfg" ] || continue

            out_dir="$RESULTS_DIR/weak${np_ratio}_N${N}_${backend}_P${P}"
            [ -d "$out_dir" ] && echo "[skip] $out_dir" && continue

            echo "[run] weak np_ratio=$np_ratio N=$N backend=$backend P=$P..."
            mkdir -p "$out_dir"

            t_start=$(date +%s%N)
            if [ "$P" -eq 1 ]; then
                "$BINARY" stepping --config "$cfg" --out "$out_dir" 2>"$out_dir/stderr.log"
            else
                mpirun -n "$P" "$BINARY" stepping --config "$cfg" --out "$out_dir" 2>"$out_dir/stderr.log"
            fi
            t_end=$(date +%s%N)
            wall_ms=$(( (t_end - t_start) / 1000000 ))
            echo "$wall_ms" > "$out_dir/wall_ms.txt"
            echo "  → ${wall_ms} ms"
        done
    done
done

echo ""
echo "[run_phase9] Todos los runs completados."
echo "Ejecuta: python3 $SCRIPT_DIR/analyze_phase9.py"
