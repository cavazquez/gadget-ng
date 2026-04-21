#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Phase 38 — Generate external CLASS reference tables.
#
# Produces:
#   - pk_class_z0.dat   (linear P(k) at z=0,  σ_8=0.8)
#   - pk_class_z49.dat  (linear P(k) at z=49, same cosmology)
#
# Usage:
#   ./generate_reference.sh [VENV_DIR]
#
# If VENV_DIR is not provided, defaults to ./venv in this directory.
#
# The .dat files are tracked in git. This script only needs to be re-run if
# the cosmology changes or a new CLASS version is pinned.
#
# Tested with: Python 3.13, classy 3.3.4.0, CLASS core 3.3.4.
# -----------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="${1:-$SCRIPT_DIR/venv}"

if [[ ! -d "$VENV_DIR" ]]; then
    echo "[phase38] creating venv at $VENV_DIR"
    python3 -m venv "$VENV_DIR"
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    pip install --quiet --upgrade pip wheel setuptools
    pip install --quiet numpy cython scipy
    # classy 3.3.4.0 is the pinned reference version.
    pip install --quiet classy==3.3.4.0
else
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
fi

echo "[phase38] python : $(which python3)"
echo "[phase38] classy : $(python3 -c 'import classy; print(classy.__version__)')"

python3 "$SCRIPT_DIR/dump_class_pk.py" \
    --out-z0  "$SCRIPT_DIR/pk_class_z0.dat" \
    --out-z49 "$SCRIPT_DIR/pk_class_z49.dat"

echo "[phase38] wrote:"
ls -la "$SCRIPT_DIR/pk_class_z0.dat" "$SCRIPT_DIR/pk_class_z49.dat"

echo "[phase38] SHA256:"
( cd "$SCRIPT_DIR" && sha256sum pk_class_z0.dat pk_class_z49.dat )
