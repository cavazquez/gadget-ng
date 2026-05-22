# Oleada 11 — vis, render_snapshot_visualization e in-situ RSD

**Fecha:** 2026-05  
**Línea base medida (post oleada 10):** **57,45%** (11 365 / 19 781 líneas, +2,86 pp)

## Objetivo

Subir cobertura en módulos con +0,00% tras oleada 10:

| Área | Acción |
|------|--------|
| `gadget-ng-vis` (ppm, projection, renderer) | Unit tests in-lib: PPM XZ, density, PNG, perspective, Density mode |
| `engine/mod.rs` `render_snapshot_visualization` | Smoke export PPM density + PNG points |
| `run_visualize` | Smokes proyecciones XZ/YZ y color white |
| `insitu.rs` | `pk_rsd_bins`, `assembly_bias` con cluster denso |

## Higiene

- `.antigravitycli/` añadido a `.gitignore`

## Tests añadidos

### `gadget-ng-vis`

| Test | Rama |
|------|------|
| `ppm_projection_xz_marks_expected_pixel` | `render_ppm_projection` XZ |
| `density_ppm_cluster_pixel_brightest` | `render_density_ppm` |
| `write_png_roundtrip_magic_bytes` | `write_png` |
| `xz_and_yz_projections`, `perspective_scales_with_depth` | `Projection` |
| `density_mode_with_external_scalars` | `Renderer` + `ColorMode::Density` |

### `gadget-ng-cli`

| Test | Rama |
|------|------|
| `smoke_run_visualize_projections_and_white` | `run_visualize` xz/yz/white |
| `smoke_render_snapshot_visualization_exports` | PPM density + PNG desde `snapshot_final` |
| `maybe_run_insitu_pk_rsd_bins_populated` | P(k,μ) in-situ |
| `maybe_run_insitu_assembly_bias_on_dense_halos` | assembly bias + halo_centers |

## Verificación

```bash
cargo test -p gadget-ng-vis
cargo test -p gadget-ng-cli
cargo tarpaulin --workspace --lib --features gadget-ng-io/hdf5
```

## Meta

+2–4 pp sobre 57,45% → **~60–61%** global `--lib`.

## Medición (2026-05-22)

**58,06%** (11 484 / 19 781 líneas, **+0,60 pp** vs 57,45% post-oleada 10).

| Módulo | Antes → Después | Notas |
|--------|-----------------|-------|
| `gadget-ng-vis/ppm.rs` | 21/76 → **73/76** (+68 pp local) | Tests in-lib oleada 11 |
| `gadget-ng-vis/projection.rs` | 14/24 → **22/24** | |
| `gadget-ng-vis/renderer.rs` | 16/23 → **22/23** | |
| `gadget-ng-cli/insitu.rs` | — → **123/195** (+27 pp local) | pk_rsd + assembly_bias |
| `engine/cmds.rs` | 0/40 | Smokes en `tests/` no corren con `--lib` |
| `engine/mod.rs` | 0/23 | Idem (`render_snapshot_visualization`) |
| `stepping/mod.rs` | 0/1838 | Smokes en `tests/lib_stepping_smokes.rs` fuera de `--lib` |

**Implicación:** `cargo tarpaulin --workspace --lib` solo ejecuta unit tests embebidos en `src/`, no los integration tests de `tests/*.rs`. Para subir `cmds.rs` / `engine/mod.rs` hay que añadir `#[cfg(test)]` en el crate o ampliar el perfil de tarpaulin.

## Fuera de alcance

- Job CI `coverage-gpu` con lavapipe.
- MPI bajo tarpaulin.
