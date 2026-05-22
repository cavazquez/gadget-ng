# Oleada 12 — tests in-lib para cmds, vis y stepping

**Fecha:** 2026-05  
**Línea base medida (post oleada 11):** **58,06%** (11 484 / 19 781 líneas)

## Objetivo

Los smokes en `crates/gadget-ng-cli/tests/` no cuentan en `cargo tarpaulin --workspace --lib`. Mover cobertura de alto ROI a módulos `#[cfg(test)]` embebidos en `src/`.

| Área | Acción |
|------|--------|
| `engine/cmds.rs` | `cmd_config_print`, `run_snapshot`, `run_visualize` (proyecciones + snapshot vacío) |
| `engine/mod.rs` | `render_snapshot_visualization` (PPM density + PNG points; skip sin dir) |
| `engine/stepping/mod.rs` | Smokes mínimos: direct 2 pasos, PM 2 pasos, snapshot final |

## Tests añadidos

### `engine/cmds.rs`

| Test | Rama |
|------|------|
| `cmd_config_print_emits_canonical_hash` | carga + hash canónico |
| `run_snapshot_writes_jsonl` | `run_snapshot` serial |
| `run_visualize_velocity_and_white_projections` | xy/velocity, xz/yz/white |
| `run_visualize_empty_snapshot_is_ok` | snapshot vacío → Ok sin PNG |

### `engine/mod.rs`

| Test | Rama |
|------|------|
| `render_snapshot_visualization_exports_ppm_and_png` | density PPM + points PNG |
| `render_snapshot_visualization_skips_missing_dir` | early return |

### `engine/stepping/mod.rs`

| Test | Rama |
|------|------|
| `smoke_direct_lattice_two_steps` | bucle direct, 2 pasos |
| `smoke_pm_lattice_two_steps` | solver PM |
| `smoke_direct_writes_final_snapshot` | `write_final_snapshot=true` |

## Verificación

```bash
cargo test -p gadget-ng-cli
cargo fmt --all
cargo clippy -p gadget-ng-cli -- -D warnings
cargo tarpaulin --workspace --lib --features gadget-ng-io/hdf5
```

## Meta

+3–6 pp sobre 58,06% → **~61–64%** global `--lib` (principalmente `cmds.rs`, `mod.rs`, `stepping/mod.rs`).

## Medición (2026-05-22)

**59,60%** (11 797 / 19 792 líneas, **+1,54 pp** vs 58,06% post-oleada 11).

| Módulo | Antes → Después | Notas |
|--------|-----------------|-------|
| `engine/cmds.rs` | 0/40 → **41/41** | 100% in-lib |
| `engine/mod.rs` | 0/23 → **22/23** | `render_snapshot_visualization` |
| `engine/stepping/mod.rs` | 0/1838 → **142/1840** | Smokes mínimos direct + PM + snapshot final |

Para seguir subiendo `stepping/mod.rs` hace falta migrar más smokes de `tests/lib_stepping_smokes.rs` (TreePM, SPH, cosmo, checkpoint, etc.) a `#[cfg(test)]` in-lib.

## Fuera de alcance

- Duplicar todos los smokes de `tests/lib_stepping_smokes.rs` in-lib (mantener ambos: integración + cobertura).
- MPI / GPU bajo tarpaulin.
