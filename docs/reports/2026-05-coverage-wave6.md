# Oleada 6 — cobertura unitaria (`--lib`)

**Fecha:** 2026-05  
**Objetivo:** +4–7 pp sobre la línea base ~52,6% (`cargo tarpaulin --workspace --lib`).

## Infra (Track 0)

- `tarpaulin.toml`: `timeout = "300s"` (humantime; evita parse error en 0.31.x).
- `gadget-ng-analysis`: `cfg(tarpaulin)` en `halofit` reduce cuadratura/bisección bajo instrumentación.
- CI Tests: `cargo test -p gadget-ng-analysis --features parallel` (rama `fof_parallel`).

## Tests añadidos

| Crate | Área |
|-------|------|
| `gadget-ng-analysis` | FoF: membership, combined, catalog, periodic wrap; SUBFIND smoke; Fisher correlación / `k_bins` |
| `gadget-ng-pm` | `PmSolver` smoke `GravitySolver` |
| `gadget-ng-treepm` | `TreePmSolver` smoke |
| `gadget-ng-core` | `validate` gas_fraction |
| `gadget-ng-sph` | fase Warm, `phase_fractions` |
| `gadget-ng-mhd` | `compute_dedner_div_b` vacío / gas |
| `gadget-ng-cuda` | `CudaCoolingSolver` try_new / vacío / `CoolingKind::None` |
| `gadget-ng-cli` | `tests/stepping_plummer_smoke.rs` (2 pasos; **no** cuenta en `--lib`) |

## Fuera de alcance inmediato

- Extraer `[lib]` en `gadget-ng-cli` para subir Codecov en `stepping/`.
- Track E GPU profundo (más allá de smoke CUDA).

## Verificación

```bash
cargo test -p gadget-ng-analysis --features parallel
cargo test -p gadget-ng-pm -p gadget-ng-treepm -p gadget-ng-core -p gadget-ng-sph -p gadget-ng-mhd -p gadget-ng-cuda
cargo test -p gadget-ng-cli
cargo clippy --workspace -- -D warnings
cargo tarpaulin --workspace --lib
```
