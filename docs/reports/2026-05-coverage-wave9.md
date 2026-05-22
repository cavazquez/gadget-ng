# Oleada 9 — analyze extendido y checkpoint Phase 106

**Fecha:** 2026-05  
**Línea base (oleada 8):** ~63–67% estimado (`cargo tarpaulin --workspace --lib --features gadget-ng-io/hdf5`)

## Objetivo

Cubrir ramas de alto ROI aún sin ejercitar tras oleada 8:

| Área | Acción |
|------|--------|
| `analyze_cmd.rs` flags extendidos | Caminos con gas, SUBFIND, c(M), catálogo, luminosity/xray |
| `engine/checkpoint.rs` Phase 106 | Round-trip con `HierarchicalState`, AGN y química |

## Fase A — `analyze_cmd`

### Tests unitarios (`analyze_cmd.rs`)

| Test | Rama cubierta |
|------|---------------|
| `analyze_cm21_and_igm_temp_with_gas` | `--cm21`, `--igm-temp` con partículas de gas |
| `analyze_agn_and_eor_with_gas_particles` | `--agn-stats` (candidato BH), `--eor-state` ionizado |
| `analyze_luminosity_and_xray_with_stars_and_gas` | `--luminosity`, `--xray` |
| `analyze_subfind_on_dense_cluster` | SUBFIND sobre halo denso (64 partículas) |
| `analyze_cosmology_populates_concentration_mass` | c(M) con cosmología explícita |
| `analyze_hdf5_catalog_writes_halo_catalog` | catálogo HDF5 o JSONL según feature |

### Smoke de integración (`tests/lib_cmd_smokes.rs`)

| Test | Función |
|------|---------|
| `smoke_run_analyze_extended_flags_gas` | `run_analyze` con cm21 + igm + agn + eor + xray en retícula de gas |

## Fase B — checkpoint extendido

| Test | Estado persistido |
|------|-------------------|
| `save_load_checkpoint_with_hierarchical_state` | `hierarchical_state.json` (levels/elapsed) |
| `save_load_checkpoint_with_agn_and_chem` | `agn_bhs.json`, `chem_states.json` |

## Verificación

```bash
cargo test -p gadget-ng-cli
cargo clippy -p gadget-ng-cli --all-targets -- -D warnings
cargo tarpaulin --workspace --lib --features gadget-ng-io/hdf5
```

## Meta

+2–4 pp sobre oleada 8 → **~65–70%** global `--lib`.

## Fase C — smokes stepping (implementada)

| Test | Rama |
|------|------|
| `smoke_stepping_snapshot_interval_writes_frames` | `snapshot_interval = 1` → `frames/snap_*` |
| `smoke_stepping_resume_completes_remaining_steps` | resume 2→4 pasos vía checkpoint |
| `smoke_stepping_rt_reionization_minimal` | SPH + `[rt]` + `[reionization]` |
| `smoke_stepping_sidm_minimal` | `[sidm] enabled = true` |

## Fuera de alcance (post oleada 9)

- Job CI `coverage-gpu` con lavapipe.
- MPI bajo tarpaulin.
