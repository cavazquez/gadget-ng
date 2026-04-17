#!/usr/bin/env bash
# run_phase5.sh — Lanza los 40 runs dinámicos de la Fase 5.
#
# Cada run ejecuta `gadget-ng stepping` con uno de los 40 TOMLs generados por
# `generate_configs.py` y escribe salidas bajo `runs/<tag>/`. El script no
# re-ejecuta un run si su `timings.json` ya existe (idempotente): borra el
# directorio `runs/<tag>/` antes de relanzarlo si quieres forzar.
#
# Uso:
#   bash scripts/run_phase5.sh                 # todos los runs
#   bash scripts/run_phase5.sh plummer_a1      # filtra por prefijo
#   PARALLEL=4 bash scripts/run_phase5.sh      # paraleliza (si hay memoria suficiente)

set -euo pipefail

EXP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "$EXP_DIR/../../.." && pwd)"
CONFIG_DIR="$EXP_DIR/config/generated"
RUNS_DIR="$EXP_DIR/runs"
BIN="$REPO_ROOT/target/release/gadget-ng"
FILTER="${1:-}"
PARALLEL="${PARALLEL:-1}"

mkdir -p "$RUNS_DIR"

if [[ ! -x "$BIN" ]]; then
    echo "Compilando gadget-ng (release)..." >&2
    (cd "$REPO_ROOT" && cargo build --release -p gadget-ng-cli >/dev/null)
fi

if [[ ! -d "$CONFIG_DIR" ]]; then
    echo "Configs no generadas. Ejecuta primero:" >&2
    echo "    python3 $EXP_DIR/scripts/generate_configs.py" >&2
    exit 1
fi

run_one() {
    local cfg="$1"
    local tag
    tag="$(basename "$cfg" .toml)"
    local out="$RUNS_DIR/$tag"

    if [[ -f "$out/timings.json" ]]; then
        echo "[skip] $tag (ya completado)"
        return 0
    fi

    mkdir -p "$out"
    local t0 t1 dur
    t0=$(date +%s)
    if "$BIN" stepping --config "$cfg" --out "$out" --snapshot >"$out/run.log" 2>&1; then
        t1=$(date +%s); dur=$((t1 - t0))
        echo "[ok]   $tag (${dur}s)"
    else
        t1=$(date +%s); dur=$((t1 - t0))
        echo "[FAIL] $tag (${dur}s) — ver $out/run.log"
        return 1
    fi
}

export -f run_one
export RUNS_DIR BIN

mapfile -t CFGS < <(ls "$CONFIG_DIR"/*.toml | sort)

if [[ -n "$FILTER" ]]; then
    CFGS=("${CFGS[@]/#*/${FILTER}}")  # no-op; awk-style filter below
    mapfile -t CFGS < <(printf '%s\n' "${CFGS[@]}" | grep "$FILTER" || true)
fi

echo "Lanzando ${#CFGS[@]} runs (PARALLEL=$PARALLEL) — salida en $RUNS_DIR"
echo

if [[ "$PARALLEL" -le 1 ]]; then
    for cfg in "${CFGS[@]}"; do
        run_one "$cfg"
    done
else
    printf '%s\n' "${CFGS[@]}" | xargs -I{} -P "$PARALLEL" bash -c 'run_one "$@"' _ {}
fi

echo
echo "Fase 5 corrida completa. Runs en: $RUNS_DIR"
