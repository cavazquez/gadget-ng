#!/usr/bin/env bash
# Orquestador Phase 35 — modelado de R(N).
#
# 1. Ejecuta los 6 tests de Rust (vuelcan JSONs a `target/phase35/`).
# 2. Corre fit_r_n.py para generar `output/rn_model.json`.
# 3. Corre plot_r_n.py para generar las 5 figuras.
# 4. Demuestra la corrección de postproceso con apply_correction.py.
# 5. Copia los PNGs a `docs/reports/figures/phase35/`.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

REPO_ROOT="$(cd ../../.. && pwd)"
TARGET_DIR="$REPO_ROOT/target/phase35"
OUTPUT_DIR="$SCRIPT_DIR/output"
FIGURES_DIR="$SCRIPT_DIR/figures"
DOCS_FIGURES_DIR="$REPO_ROOT/docs/reports/figures/phase35"

mkdir -p "$OUTPUT_DIR" "$FIGURES_DIR" "$DOCS_FIGURES_DIR"

echo "[phase35] 1/5 — Ejecutando tests Rust (release)"
(
    cd "$REPO_ROOT"
    cargo test -p gadget-ng-physics --test phase35_rn_modeling --release -- --nocapture
)

echo "[phase35] 2/5 — Ajustando modelos A y B"
python3 "$SCRIPT_DIR/scripts/fit_r_n.py" \
    --input "$TARGET_DIR/rn_by_seed.json" \
    --output "$OUTPUT_DIR/rn_model.json" \
    --summary "$OUTPUT_DIR/fit_summary.md"

echo "[phase35] 3/5 — Generando figuras"
python3 "$SCRIPT_DIR/scripts/plot_r_n.py" \
    --target-dir "$TARGET_DIR" \
    --model "$OUTPUT_DIR/rn_model.json" \
    --output-dir "$FIGURES_DIR"

echo "[phase35] 4/5 — Demo de corrección (postproceso)"
# Construimos un P(k) de prueba a partir del fit para mostrar el pipeline.
python3 - <<'PY' > "$OUTPUT_DIR/pk_raw_demo.json"
import json
# P(k) sintético: 5 bins, amplitud arbitraria grande (como el estimador interno)
bins = [
    {"k": 0.5, "pk": 1.0e-4, "n_modes": 6},
    {"k": 1.0, "pk": 5.0e-5, "n_modes": 24},
    {"k": 2.0, "pk": 2.0e-5, "n_modes": 96},
    {"k": 4.0, "pk": 5.0e-6, "n_modes": 256},
    {"k": 8.0, "pk": 1.0e-6, "n_modes": 1024},
]
print(json.dumps(bins))
PY

python3 "$SCRIPT_DIR/scripts/apply_correction.py" \
    --input "$OUTPUT_DIR/pk_raw_demo.json" \
    --model "$OUTPUT_DIR/rn_model.json" \
    --output "$OUTPUT_DIR/pk_corrected_demo.json" \
    --n 64 --box-size 1.0 --box-mpc-h 100.0

echo "[phase35] 5/5 — Copiando figuras a docs/reports/figures/phase35/"
cp -f "$FIGURES_DIR"/*.png "$DOCS_FIGURES_DIR"/

echo "[phase35] ✅ Completado"
echo "  JSONs:   $TARGET_DIR/"
echo "  Output:  $OUTPUT_DIR/"
echo "  Figuras: $FIGURES_DIR/  →  $DOCS_FIGURES_DIR/"
