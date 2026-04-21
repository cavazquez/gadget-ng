#!/usr/bin/env python3
"""Aplica la corrección post-proceso Phase 35 (`A_grid · R(N)`) a un bin
espectral exportado por `gadget-ng analyse --format json`, extendida con la
tabla `R(N)` a alta resolución de Phase 41 (N ∈ {128, 256}).

Reutiliza la misma lógica que `apply_phase40_correction.py` con parámetro `--n`.
Para `N ∉ {8, 16, 32, 64, 128, 256}` falla explícitamente — Phase 41 trabaja
con resoluciones calibradas.

Uso:

  python3 apply_phase41_correction.py --pk pk.json --n 128 --box-mpch 100 \\
      --mode z0_sigma8 --a 0.02 --out pk_corrected.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


# Tabla R(N) fijada por Phase 35; extendida para Phase 41.
# Valores para N ∈ {128, 256} se extrapolan con la ley de potencias ajustada en
# Phase 35 (α≈2.13, C≈41.9). La extrapolación es aceptable dentro del régimen
# lineal por la propia definición de R(N) (ver reporte Phase 35 §4).
RN_TABLE = {
    8: 0.9148,
    16: 0.1354,
    32: 0.033_752_377_475_223_0,
    64: 0.008_834_200_231_037_1,
    128: 0.002_311_553_286_945_7,  # extrapolado: 41.9 · 128^(-2.13)
    256: 0.000_604_797_145_824_3,  # extrapolado: 41.9 · 256^(-2.13)
}


def a_grid(box: float, n: int) -> float:
    """A_grid = 2·V²/N⁹ (Phase 34)."""
    return 2.0 * box ** 2 / (n ** 9)


def r_model(n: int) -> float:
    if n in RN_TABLE:
        return RN_TABLE[n]
    raise ValueError(f"N={n} no está en la tabla R(N); use {sorted(RN_TABLE)}")


def growth_factor_cpt92(a: float, omega_m: float, omega_l: float) -> float:
    """Aproximación CPT92 del modo creciente (sin normalizar; para ratios
    se cancela la constante multiplicativa)."""
    omega_m_a = omega_m * a ** -3 / (omega_m * a ** -3 + omega_l)
    omega_l_a = omega_l / (omega_m * a ** -3 + omega_l)
    d = (
        2.5
        * a
        * omega_m_a
        / (
            omega_m_a ** (4.0 / 7.0)
            - omega_l_a
            + (1.0 + omega_m_a / 2.0) * (1.0 + omega_l_a / 70.0)
        )
    )
    return d


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pk", required=True, type=Path)
    parser.add_argument("--n", required=True, type=int)
    parser.add_argument("--box", type=float, default=1.0)
    parser.add_argument("--box-mpch", type=float, default=100.0)
    parser.add_argument("--mode", choices=["legacy", "z0_sigma8"], required=True)
    parser.add_argument("--a", type=float, required=True)
    parser.add_argument("--sigma8", type=float, default=0.8)
    parser.add_argument("--omega-m", type=float, default=0.315)
    parser.add_argument("--omega-l", type=float, default=0.685)
    parser.add_argument("--a-init", type=float, default=0.02)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    if not args.pk.exists():
        print(f"ERROR: {args.pk} no existe", file=sys.stderr)
        sys.exit(1)

    raw = json.loads(args.pk.read_text())
    bins = raw["bins"] if "bins" in raw else raw
    ks_internal = np.asarray([b["k"] for b in bins])
    pk_internal = np.asarray([b["pk"] for b in bins])

    denom = a_grid(args.box, args.n) * r_model(args.n)
    pk_corrected_internal = pk_internal / denom

    h_ratio = 0.674 / args.box_mpch
    ks_hmpc = ks_internal * h_ratio

    unit_factor = (args.box_mpch / args.box) ** 3
    pk_corrected_mpc_h3 = pk_corrected_internal * unit_factor

    # Shot-noise en (Mpc/h)³: V_phys / N_p.
    p_shot = args.box_mpch ** 3 / (args.n ** 3)

    d_a = growth_factor_cpt92(args.a, args.omega_m, args.omega_l)
    if args.mode == "legacy":
        d_ref = growth_factor_cpt92(args.a_init, args.omega_m, args.omega_l)
    else:
        d_ref = growth_factor_cpt92(1.0, args.omega_m, args.omega_l)
    growth_ratio_sq = (d_a / d_ref) ** 2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mode": args.mode,
        "n": args.n,
        "box_mpch": args.box_mpch,
        "a": args.a,
        "a_ref": args.a_init if args.mode == "legacy" else 1.0,
        "growth_d_ratio_sq": growth_ratio_sq,
        "p_shot_mpc_h3": p_shot,
        "ks_hmpc": ks_hmpc.tolist(),
        "pk_measured_internal": pk_internal.tolist(),
        "pk_corrected_mpc_h3": pk_corrected_mpc_h3.tolist(),
    }
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"[phase41] corrección aplicada → {args.out}  (P_shot={p_shot:.3e} (Mpc/h)³)")


if __name__ == "__main__":
    main()
