# Phase 38 — CLASS reference tables

Tables generated with [CLASS](https://lesgourg.github.io/class_public/class.html)
(`classy` 3.3.4.0) and committed to the repository. The Rust tests and the
Python CLI pipeline read these `.dat` files directly, so **CLASS is not a
runtime or CI dependency**. This directory is reproducible end-to-end via
`generate_reference.sh`.

## Contents

| File | Contents |
|------|----------|
| `class.ini`         | CLASS parameter file (reference for the `.dat`, not actually invoked by `dump_class_pk.py`). |
| `dump_class_pk.py`  | Python script that queries `classy.Class.pk_lin(k, z)` and writes the `.dat`. |
| `generate_reference.sh` | Entry point: creates a venv, pins `classy==3.3.4.0`, calls `dump_class_pk.py`. |
| `pk_class_z0.dat`   | Linear P(k) at z=0,  512 log-spaced k-bins in `[1e-4, 20] h/Mpc`. |
| `pk_class_z49.dat`  | Linear P(k) at z=49, same grid.                                 |

## Cosmology (identical to Phases 27–37)

| Parameter   | Value    |
|-------------|----------|
| Ω_m         | 0.315    |
| Ω_b         | 0.049    |
| Ω_cdm       | 0.266    |
| h           | 0.674    |
| n_s         | 0.965    |
| σ_8         | 0.8      |
| T_CMB       | 2.7255 K |
| N_ur        | 3.046    |
| N_ncdm      | 0        |
| non_linear  | none (linear theory only) |

## Units

- `k` in `h/Mpc`.
- `P(k)` in `(Mpc/h)^3`.
- Internally CLASS returns `P(k [Mpc^-1]) [Mpc^3]`; the script multiplies by
  `h^3` and feeds `k * h` to `pk_lin`, producing the `h/Mpc`/`(Mpc/h)^3`
  convention used by `gadget-ng` (same as Phases 34–37).

## Reproducibility

```bash
cd experiments/nbody/phase38_class_validation/reference
./generate_reference.sh                # creates ./venv and writes .dat
```

This requires Python ≥ 3.9 with `venv`, a C compiler (gcc/clang), and the
system libraries CLASS depends on (OpenMP). Re-running the script from a
clean venv with the same `classy` version reproduces the tables up to
bit-level floating-point noise.

## Sanity checks

- `sigma8(z=0)` reported by CLASS should round to `0.8000` within `~1e-6`
  (the rescaling is internal; our check is printed by `dump_class_pk.py`).
- `√(P(49)/P(0)) ≈ 2.56e-2`, which agrees with the CPT92 approximation
  `s = D(0.02)/D(1) ≈ 2.5413e-2` used by `gadget-ng` to ~0.8% (the
  residual difference is CPT92's fitting error relative to CLASS's full
  ODE integration and is discussed in the Phase 38 report).

## Integrity

| File | SHA-256 |
|------|---------|
| `pk_class_z0.dat`  | `cf8dc8ce62953404f06dc1b80e97a6df0206a786769ed1140fe1eac5e79f95b3` |
| `pk_class_z49.dat` | `539f880faed880f1056746a7d55fb4673fcb02645de5d4762b26b943ad723c1e` |

Re-compute at any time with:

```bash
sha256sum pk_class_z0.dat pk_class_z49.dat
```
