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

## Fuera de alcance

- Job CI `coverage-gpu` con lavapipe.
- MPI bajo tarpaulin.
