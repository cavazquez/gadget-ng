#!/usr/bin/env bash
# run_phase7.sh — Lanza los 54 runs de la Fase 7 (Aarseth adaptive timesteps).
#
# Estructura de salida: runs/<tag>/ con diagnostics.jsonl, frames/, timings.json, run.log
#
# Uso:
#   bash scripts/run_phase7.sh                        # todos los runs
#   bash scripts/run_phase7.sh plummer_a1             # filtra por prefijo
#   bash scripts/run_phase7.sh hier_acc               # filtra por variante
#   PARALLEL=4 bash scripts/run_phase7.sh             # paraleliza N runs
#   FORCE=1 bash scripts/run_phase7.sh                # re-ejecuta aunque exista timings.json

set -euo pipefail

EXP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "$EXP_DIR/../../.." && pwd)"
CONFIG_DIR="$EXP_DIR/config/generated"
RUNS_DIR="$EXP_DIR/runs"
BIN="$REPO_ROOT/target/release/gadget-ng"
FILTER="${1:-}"
PARALLEL="${PARALLEL:-1}"
FORCE="${FORCE:-0}"

mkdir -p "$RUNS_DIR"

if [[ ! -x "$BIN" ]]; then
    echo "Compilando gadget-ng (release)..." >&2
    (cd "$REPO_ROOT" && cargo build --release -p gadget-ng-cli 2>&1 | tail -5)
fi

if [[ ! -d "$CONFIG_DIR" ]] || [[ -z "$(ls -A "$CONFIG_DIR" 2>/dev/null)" ]]; then
    echo "Configs no generadas. Ejecuta primero:" >&2
    echo "    python3 $EXP_DIR/scripts/generate_configs.py" >&2
    exit 1
fi

run_one() {
    local cfg="$1"
    local tag
    tag="$(basename "$cfg" .toml)"
    local out="$RUNS_DIR/$tag"

    if [[ "$FORCE" != "1" ]] && [[ -f "$out/timings.json" ]]; then
        echo "[skip] $tag (ya completado)"
        return 0
    fi

    mkdir -p "$out"
    local t0 t1 dur
    t0=$(date +%s%N 2>/dev/null || date +%s)
    if "$BIN" stepping --config "$cfg" --out "$out" --snapshot >"$out/run.log" 2>&1; then
        t1=$(date +%s%N 2>/dev/null || date +%s)
        # Calcular duración en segundos (compatible con y sin nanosegundos)
        if [[ ${#t0} -gt 10 ]]; then
            dur=$(( (t1 - t0) / 1000000000 ))
        else
            dur=$(( t1 - t0 ))
        fi
        echo "[ok]   $tag (${dur}s)"
    else
        echo "[FAIL] $tag — ver $out/run.log"
        return 1
    fi
}

export -f run_one
export RUNS_DIR BIN FORCE

mapfile -t CFGS < <(ls "$CONFIG_DIR"/*.toml 2>/dev/null | sort)

if [[ ${#CFGS[@]} -eq 0 ]]; then
    echo "No se encontraron TOMLs en $CONFIG_DIR" >&2
    exit 1
fi

if [[ -n "$FILTER" ]]; then
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
echo "Fase 7 corrida completa. Runs en: $RUNS_DIR"
echo "Siguiente paso: python3 $EXP_DIR/scripts/analyze_conservation.py"
