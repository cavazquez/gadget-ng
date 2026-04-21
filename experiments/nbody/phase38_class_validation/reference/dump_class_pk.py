#!/usr/bin/env python3
"""Dump the CLASS linear matter power spectrum to two .dat tables.

Usage (invoked by `generate_reference.sh`, not meant to be run directly):

    python3 dump_class_pk.py \\
        --out-z0  pk_class_z0.dat \\
        --out-z49 pk_class_z49.dat

Cosmology and output format are fixed to match Phase 38's requirements:

- Planck-like ΛCDM (Ω_m=0.315, Ω_b=0.049, h=0.674, n_s=0.965, σ_8=0.8,
  T_CMB=2.7255 K, N_ur=3.046, 0 non-CDM species).
- Linear P(k); `non_linear` empty.
- k grid: 512 log-spaced points in `k ∈ [1e-4, 20] h/Mpc` (covers gadget-ng's
  valid window up to N=64³ with box=100 Mpc/h; k_Nyq/2 ≈ 1 h/Mpc).
- Columns: `k [h/Mpc]   P(k) [(Mpc/h)^3]`, whitespace-separated, 6 decimals.
- Header rows start with `#` and document provenance (CLASS version,
  cosmology, redshift, columns).
"""

from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

import numpy as np
from classy import Class


COSMO = {
    "h": 0.674,
    "Omega_b": 0.049,
    "Omega_cdm": 0.266,        # = Ω_m − Ω_b
    "T_cmb": 2.7255,
    "N_ur": 3.046,
    "N_ncdm": 0,
    "n_s": 0.965,
    "sigma8": 0.8,
    "output": "mPk",
    "non_linear": "none",
    "z_pk": "0,49",
    "z_max_pk": 50.0,
    "P_k_max_h/Mpc": 20.0,
    "k_per_decade_for_pk": 50,
}

# Grid used to write the .dat. Must bracket gadget-ng's valid window.
K_MIN_H_MPC = 1.0e-4
K_MAX_H_MPC = 20.0
N_POINTS = 512


def _classy_version() -> str:
    import classy
    return getattr(classy, "__version__", "unknown")


def _header(z: float, sigma8_check: float) -> list[str]:
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return [
        "# Phase 38 — CLASS linear matter power spectrum",
        f"# Generated:    {now}",
        f"# classy:       {_classy_version()}",
        "# cosmology:    Planck-like ΛCDM",
        "#   Omega_m    = 0.315",
        "#   Omega_b    = 0.049",
        "#   Omega_cdm  = 0.266",
        "#   h          = 0.674",
        "#   n_s        = 0.965",
        "#   sigma_8    = 0.8",
        "#   T_CMB      = 2.7255 K",
        "#   N_ur       = 3.046",
        f"#   sigma8(z=0) check from CLASS = {sigma8_check:.10f}",
        f"# redshift:    z = {z:g}",
        "# grid:         512 log-spaced points in k ∈ [1e-4, 20] h/Mpc",
        "# columns:      k [h/Mpc]    P(k) [(Mpc/h)^3]",
    ]


def _write_dat(path: Path, z: float, ks_hmpc: np.ndarray, pks_mpch3: np.ndarray,
               sigma8_check: float) -> None:
    with path.open("w") as f:
        for line in _header(z, sigma8_check):
            f.write(line + "\n")
        for k, p in zip(ks_hmpc, pks_mpch3):
            f.write(f"{k:.8e}   {p:.8e}\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-z0", required=True)
    ap.add_argument("--out-z49", required=True)
    args = ap.parse_args()

    cosmo = Class()
    cosmo.set(COSMO)
    cosmo.compute()
    h = cosmo.h()
    sigma8 = cosmo.sigma8()
    print(f"[phase38] CLASS computed   σ8(z=0) = {sigma8:.10f}")
    print(f"[phase38] CLASS computed   h       = {h:.4f}")

    ln_kmin = np.log(K_MIN_H_MPC)
    ln_kmax = np.log(K_MAX_H_MPC)
    ks_hmpc = np.exp(np.linspace(ln_kmin, ln_kmax, N_POINTS))

    pks_z0 = np.array([cosmo.pk_lin(k * h, 0.0) * h**3 for k in ks_hmpc])
    pks_z49 = np.array([cosmo.pk_lin(k * h, 49.0) * h**3 for k in ks_hmpc])

    _write_dat(Path(args.out_z0), 0.0, ks_hmpc, pks_z0, sigma8)
    _write_dat(Path(args.out_z49), 49.0, ks_hmpc, pks_z49, sigma8)

    r = float(np.sqrt(pks_z49[N_POINTS // 2] / pks_z0[N_POINTS // 2]))
    print(f"[phase38] D(49)/D(0) ≈ √(P(49)/P(0)) @ k = "
          f"{ks_hmpc[N_POINTS//2]:.3e} h/Mpc  →  {r:.6e}")
    print(f"[phase38] wrote {args.out_z0}  and  {args.out_z49}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
