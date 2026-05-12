# Runtime Physics Overrides

Phase 191 adds `gadget-ng stepping` flags for fast experiment sweeps without
editing TOML files. CLI values override TOML; TOML values override defaults.

## Main Groups

| Phase range | CLI group | Key flags |
|-------------|-----------|-----------|
| 109-122 | SPH/baryons | `--sph`, `--gas-fraction`, `--cooling`, `--feedback`, `--sf-model`, `--winds` |
| 100/116/183/190 | AGN/PBH | `--agn`, `--agn-n-bh`, `--agn-m-seed`, `--agn-radio`, `--agn-spin`, `--agn-mergers`, `--pbh-seeding` |
| 117/170 | Cosmic rays | `--cr`, `--cr-kappa`, `--cr-anisotropic`, `--cr-streaming` |
| 123-150/161/172/194 | MHD/plasma | `--mhd`, `--bfield`, `--b0x`, `--b0y`, `--b0z`, `--turbulence`, `--two-fluid`, `--ambipolar` |
| 157-158/185 | Dark sector | `--sidm`, `--sidm-sigma-m`, `--fr`, `--fr-f-r0`, `--fr-nonlinear-mesh` |
| 81-95/181 | Radiation | `--rt`, `--rt-multifrequency`, `--reionization` |
| 184 | WDM/FDM | `--dark-matter`, `--wdm-mass-kev`, `--fdm-mass-22` |
| 192 | Active dust | `--dust`, `--dust-species`, `--dust-silicate-fraction`, `--dust-graphite-fraction`, `--dust-h2-shielding-boost` |

## Examples

```bash
cargo run -p gadget-ng-cli -- stepping \
  --config configs/experiments/phase191_base.toml \
  --out runs/phase191/agn_pbh \
  --sph --gas-fraction 0.1 \
  --agn --agn-mergers \
  --pbh-seeding --pbh-n-seeds 4 --pbh-m-seed 1e3
```

```bash
cargo run -p gadget-ng-cli -- stepping \
  --config configs/experiments/phase191_base.toml \
  --out runs/phase191/mhd_cr \
  --sph --gas-fraction 0.1 \
  --feedback --cr --cr-anisotropic --cr-streaming 1e-2 \
  --mhd --bfield uniform --b0x 1e-9 --turbulence
```

```bash
cargo run -p gadget-ng-cli -- stepping \
  --config configs/experiments/phase191_base.toml \
  --out runs/phase192/active_dust \
  --sph --gas-fraction 0.1 \
  --dust --dust-species silicate_graphite \
  --dust-silicate-fraction 0.54 \
  --dust-graphite-fraction 0.46 \
  --dust-h2-shielding-boost 3.0
```

```bash
cargo run -p gadget-ng-cli -- stepping \
  --config configs/experiments/phase191_base.toml \
  --out runs/phase194/ambipolar \
  --sph --gas-fraction 0.1 \
  --mhd --bfield uniform --b0x 1e-9 \
  --dust --dust-species silicate_graphite \
  --ambipolar --ambipolar-eta 0.05
```

For a smoke matrix, run:

```bash
bash scripts/run_phase191_experiments.sh
```
