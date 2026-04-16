#!/usr/bin/env bash
# ─── Genera animación GIF de una simulación gadget-ng ───────────────────────
#
# Uso:
#   ./scripts/make_animation.sh [config] [out_dir] [gif_path] [projection] [color]
#
# Valores por defecto:
#   config      = examples/plummer_sphere.toml
#   out_dir     = runs/plummer
#   gif_path    = runs/plummer/animacion.gif
#   projection  = xy
#   color       = velocity
#
# Requiere: cargo, ffmpeg
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

CONFIG="${1:-examples/plummer_sphere.toml}"
OUT="${2:-runs/plummer}"
GIF="${3:-${OUT}/animacion.gif}"
PROJ="${4:-xy}"
COLOR="${5:-velocity}"

BINARY="./target/release/gadget-ng"
PNGS="${OUT}/pngs"

# ── 1. Compilar si no existe el binario ──────────────────────────────────────
if [ ! -x "$BINARY" ]; then
  echo "[1/4] Compilando binario release..."
  cargo build --release -p gadget-ng-cli
else
  echo "[1/4] Binario ya existe: $BINARY"
fi

# ── 2. Ejecutar simulación ───────────────────────────────────────────────────
echo "[2/4] Ejecutando simulación: $CONFIG → $OUT"
rm -rf "${OUT}/frames" "${OUT}/checkpoint" "${PNGS}"
"$BINARY" stepping --config "$CONFIG" --out "$OUT"

FRAMES_DIR="${OUT}/frames"
N_FRAMES=$(ls "$FRAMES_DIR" 2>/dev/null | wc -l)
echo "       $N_FRAMES snapshots guardados en $FRAMES_DIR"

if [ "$N_FRAMES" -eq 0 ]; then
  echo "ERROR: no hay frames. Asegúrate de que snapshot_interval > 0 en el TOML."
  exit 1
fi

# ── 3. Renderizar cada frame a PNG ───────────────────────────────────────────
echo "[3/4] Renderizando $N_FRAMES frames a PNG (paralelo)..."
mkdir -p "$PNGS"
for d in "${FRAMES_DIR}"/snap_*; do
  step=$(basename "$d" | sed 's/snap_//')
  "$BINARY" visualize \
    --snapshot "$d" \
    --output "${PNGS}/frame_${step}.png" \
    --width 600 --height 600 \
    --projection "$PROJ" --color "$COLOR" 2>/dev/null &
done
wait
echo "       $(ls "${PNGS}"/*.png | wc -l) PNGs generados"

# ── 4. Montar GIF con ffmpeg (paleta de 2 pasos) ─────────────────────────────
echo "[4/4] Generando GIF: $GIF"
ffmpeg -y -loglevel warning \
  -framerate 20 \
  -pattern_type glob -i "${PNGS}/frame_*.png" \
  -vf "fps=20,scale=600:600:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=256[p];[s1][p]paletteuse=dither=bayer:bayer_scale=3" \
  "$GIF"

SIZE=$(du -h "$GIF" | cut -f1)
echo ""
echo "✓ Animación generada: $GIF ($SIZE)"
echo "  Frames: $N_FRAMES  |  Proyección: $PROJ  |  Color: $COLOR"
