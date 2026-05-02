#!/usr/bin/env bash
# Valida que los TOML de ejemplo y configs listadas pasen `gadget-ng config`.
#
# Uso:
#   ./scripts/validate_example_configs.sh
# Variables:
#   GADGET_NG_BIN — ruta al binario (default: target/release/gadget-ng)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BIN="${GADGET_NG_BIN:-$ROOT/target/release/gadget-ng}"

if [[ ! -x "$BIN" ]]; then
  echo "[validate_example_configs] Compilando gadget-ng (release)…"
  cargo build --release -p gadget-ng-cli
  BIN="$ROOT/target/release/gadget-ng"
fi

# Lista explícita además de examples/*.toml (configs referenciadas por tests / CI).
EXTRA_CONFIGS=(
  "configs/validation_128.toml"
  "configs/validation_128_test.toml"
  "configs/production_256.toml"
  "configs/production_256_test.toml"
  "configs/eor_test.toml"
)

shopt -s nullglob
mapfile -t FILES < <(printf '%s\n' examples/*.toml "${EXTRA_CONFIGS[@]}" | sort -u)

if [[ ${#FILES[@]} -eq 0 ]]; then
  echo "[validate_example_configs] No hay ficheros que validar"
  exit 1
fi

for f in "${FILES[@]}"; do
  if [[ ! -f "$f" ]]; then
    echo "[validate_example_configs] WARN: no existe $f (omitido)"
    continue
  fi
  echo "[validate_example_configs] $f"
  "$BIN" config --config "$f" >/dev/null
done

echo "[validate_example_configs] OK (${#FILES[@]} rutas únicas)"
