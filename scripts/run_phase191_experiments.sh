#!/usr/bin/env bash
set -euo pipefail

BIN="${BIN:-cargo run -p gadget-ng-cli --}"
BASE="configs/experiments/phase191_base.toml"
OUT_ROOT="${OUT_ROOT:-runs/phase191}"

mkdir -p "${OUT_ROOT}"

run_case() {
  local name="$1"
  shift
  echo "[phase191] ${name}"
  ${BIN} stepping --config "${BASE}" --out "${OUT_ROOT}/${name}" --snapshot "$@"
}

run_case baseline

run_case agn_pbh \
  --sph --gas-fraction 0.1 \
  --agn --agn-mergers \
  --pbh-seeding --pbh-n-seeds 4 --pbh-m-seed 1e3 --pbh-seed 191

run_case mhd_cr \
  --sph --gas-fraction 0.1 \
  --feedback --cr --cr-kappa 3e-3 --cr-anisotropic --cr-streaming 1e-2 \
  --mhd --bfield uniform --b0x 1e-9 \
  --turbulence --turb-amplitude 1e-3

run_case sidm_fr \
  --sidm --sidm-sigma-m 1e-5 \
  --fr --fr-f-r0 1e-5 --fr-n 1.0 --fr-nonlinear-mesh \
  --dark-matter warm --wdm-mass-kev 3.0

echo "[phase191] done: ${OUT_ROOT}"
