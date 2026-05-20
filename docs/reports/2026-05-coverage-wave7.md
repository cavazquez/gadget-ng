# Oleada 7 — cobertura archivos en 0%

**Fecha:** 2026-05  
**Línea base (oleada 6):** 54,6% (`cargo tarpaulin --workspace --lib`, 10168/18624 líneas)

## Objetivo

Eliminar los bloques 0% reales y subir ~+5–10 pp en `--workspace --lib`:

| Bloque | Líneas 0% | Causa |
|--------|-----------|-------|
| `gadget-ng-cli` (stepping, checkpoint, diagnostics…) | ~2100 | Solo `[[bin]]` en oleadas anteriores |
| `gadget-ng-io/hdf5_parallel_writer.rs` | 91 | Feature `hdf5` no activo en tarpaulin |
| `gadget-ng-gpu` (`bh_fmm`, `bh_monopole`, `treepm_short_wgsl`) | ~460 | Tests de integración fuera del lib |

## Cambios por fase

### Fase 0 — Higiene del informe

- `tarpaulin.toml`: `exclude-files` para `benches/`, `gadget-ng-physics/tests/phase*.rs`, `gadget-ng-integrators/tests/**`.
- Estas rutas no son código de librería productiva y distorsionaban el denominador de líneas totales.

### Fase 1 — CLI como librería

- Nuevo `[lib]` en `gadget-ng-cli/Cargo.toml`.
- `src/lib.rs` exporta `run_stepping`, `cmd_config_print`, `run_snapshot`, `run_visualize`.
- `src/main.rs` queda delgado (solo `use gadget_ng_cli::*` + dispatch).
- Tests unitarios en `diagnostics`, `checkpoint`, `provenance`, `gravity`, `error`.
- Smokes en `tests/` que llaman `run_stepping` directamente (Plummer, PM, cosmo, SPH, resume).

### Fase 2 — HDF5 parallel writer

- Tests unitarios en `#[cfg(test)]` de `hdf5_parallel_writer.rs` (activos con `--features hdf5`).
- CI tarpaulin: `--features gadget-ng-io/hdf5` en el comando de cobertura.

### Fase 3 — GPU WGPU

- `#[cfg(test)] mod tests` en `bh_monopole.rs`, `bh_fmm.rs`, `treepm_short_wgsl.rs`.
- Tests con `try_new()` + skip si no hay adaptador (mismo patrón que los tests de integración existentes).

## Verificación

```bash
cargo test -p gadget-ng-cli
cargo test -p gadget-ng-io --features hdf5
cargo test -p gadget-ng-gpu
cargo clippy --workspace -- -D warnings
cargo tarpaulin --workspace --lib --features gadget-ng-io/hdf5
```

Todos los comandos anteriores pasan limpiamente.

## Implementación completada

### Fase 0
- `tarpaulin.toml`: `exclude-files` para `*/benches/*`, `crates/gadget-ng-physics/tests/*`, `crates/gadget-ng-integrators/tests/*`.

### Fase 1 (CLI como librería)
- `gadget-ng-cli/Cargo.toml`: añadido `[lib]` (`gadget_ng_cli`, `src/lib.rs`).
- `src/lib.rs`: declara todos los módulos como `pub`, re-exporta `run_stepping`, `cmd_config_print`, etc., y publica `parse_*`, `RuntimeCliOverrides`, `apply_runtime_cli_overrides`, `run_with_runtime`.
- `src/main.rs`: reducido a Clap CLI + `main()` que delega en funciones del `[lib]`.
- Tests en `lib.rs` (27 tests): parse helpers, apply_runtime_cli_overrides, run_with_runtime.
- Smokes en `tests/lib_stepping_smokes.rs` (9 tests): Plummer BH, PM, TreePM, Direct, cosmo, checkpoint resume, snapshot final, SPH, Yoshida4.

### Fase 2 (HDF5)
- `hdf5_parallel_writer.rs`: 4 tests bajo `#[cfg(test)]` (default options, roundtrip P=1, empty, layout).
- `gadget-ng-io/Cargo.toml`: `dev-dependencies` con `hdf5-metno` y `ndarray`.
- CI `ci.yml`: tarpaulin con `--features gadget-ng-io/hdf5`.

### Fase 3 (GPU WGPU)
- `bh_monopole.rs`, `bh_fmm.rs`, `treepm_short_wgsl.rs`: `#[cfg(test)] mod tests` con `try_new_does_not_panic` + smoke con skip si no hay adaptador.

## Meta

~60–62% estimado tras fases 1–2 (`--lib` + `hdf5`); +~0,5 pp adicional con fase 3 si hay adaptador wgpu en CI.
