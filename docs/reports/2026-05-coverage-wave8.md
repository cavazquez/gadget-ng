# Oleada 8 — handlers CLI y smokes stepping extendidos

**Fecha:** 2026-05  
**Línea base (oleada 7):** ~60–62% estimado (`cargo tarpaulin --workspace --lib --features gadget-ng-io/hdf5`)

## Objetivo

Subir cobertura en rutas CLI aún poco ejercitadas tras extraer `[lib]` en oleada 7:

| Área | Acción |
|------|--------|
| `engine/checkpoint.rs` | Round-trip `save_checkpoint` / `load_checkpoint` con `SerialRuntime` |
| Subcomandos (`analyze`, `fisher`, `mah`, `merge-tree`, `config`, `snapshot`, `visualize`) | Smokes vía API lib en `tests/lib_cmd_smokes.rs` |
| `run_stepping` | Smokes MHD, integrador jerárquico, snapshot final explícito |

## Cambios

### Checkpoint round-trip

- Test `save_load_checkpoint_roundtrip_serial` en `engine/checkpoint.rs`:
  - 8 partículas retícula, `completed_step = 2`, verifica JSON + partículas restauradas.

### `tests/lib_cmd_smokes.rs` (7 tests)

| Test | Función cubierta |
|------|------------------|
| `smoke_cmd_config_print_valid_toml` | `cmd_config_print` |
| `smoke_run_snapshot_lattice` | `run_snapshot` |
| `smoke_run_visualize_from_snapshot` | `run_visualize` |
| `smoke_run_analyze_lattice` | `run_analyze` |
| `smoke_run_fisher_writes_json` | `run_fisher` |
| `smoke_run_mah_from_minimal_forest` | `run_mah` |
| `smoke_run_merge_tree_single_epoch` | `run_merge_tree` (1 snapshot + catálogo) |

### `tests/lib_stepping_smokes.rs` (+3 tests)

| Test | Rama |
|------|------|
| `smoke_stepping_mhd_minimal` | SPH + MHD uniforme |
| `smoke_stepping_hierarchical_leapfrog` | `[timestep] hierarchical = true` |
| `smoke_run_snapshot_via_stepping` | `write_final_snapshot = true` |

## Verificación

```bash
cargo test -p gadget-ng-cli
cargo clippy -p gadget-ng-cli --all-targets -- -D warnings
cargo tarpaulin --workspace --lib --features gadget-ng-io/hdf5
```

## Meta

+3–5 pp sobre oleada 7 → **~63–67%** global `--lib`.

## Fuera de alcance (oleada 8)

- Job CI `coverage-gpu` con lavapipe (oleada 9 opcional).
- MPI bajo tarpaulin.
- Flags extendidos de `analyze` (`--cm21`, `--subfind`, etc.) — bajo ROI por coste de fixtures.
