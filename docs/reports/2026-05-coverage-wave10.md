# Oleada 10 — in-situ, f(R), AGN y dark matter

**Fecha:** 2026-05  
**Línea base (oleada 9):** ~65–70% estimado (`cargo tarpaulin --workspace --lib --features gadget-ng-io/hdf5`)

## Objetivo

Cubrir módulos CLI aún poco ejercitados tras oleada 9:

| Área | Acción |
|------|--------|
| `insitu.rs` | Smokes stepping con `[insitu_analysis]` base y flags extendidos |
| `stepping/context.rs` | `step_fr`, `step_agn`, `step_insitu` |
| ICs WDM | Smoke cosmo + `[dark_matter] warm` |

## Smokes (`tests/lib_stepping_smokes.rs`)

| Test | Rama |
|------|------|
| `smoke_stepping_insitu_basic` | P(k), FoF, ξ(r) → `insitu_000001.json` |
| `smoke_stepping_insitu_extended_flags` | SZ, Ly-α, WL, bispectrum (SPH+RT) |
| `maybe_run_insitu_igm_temp_and_cm21_with_hot_gas` | igm_temp + cm21 con gas caliente (`insitu.rs`) |
| `smoke_stepping_modified_gravity_fr` | `[modified_gravity]` en solver directo |
| `smoke_stepping_agn_with_insitu` | `[sph.agn]` + centros FoF in-situ |
| `smoke_stepping_dark_matter_wdm_cosmo` | cutoff WDM en ICs cosmológicas |

## Verificación

```bash
cargo test -p gadget-ng-cli --test lib_stepping_smokes
cargo clippy -p gadget-ng-cli --all-targets -- -D warnings
cargo tarpaulin --workspace --lib --features gadget-ng-io/hdf5
```

## Meta

+1–3 pp sobre oleada 9 → **~66–72%** global `--lib`.

## Fuera de alcance (oleada 10)

- Job CI `coverage-gpu` con lavapipe.
- MPI bajo tarpaulin.
- Assembly bias in-situ (requiere ≥4 halos FoF en retícula pequeña).
