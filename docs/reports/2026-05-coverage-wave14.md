# Oleada 14 — gravity, stepping/context, SPH integrator, MHD alfven_dt

**Fecha:** 2026-05  
**Línea base medida (post oleada 13):** **62,48%** (12 367 / 19 792 líneas)

## Objetivo

Subir cobertura en módulos de física invocados desde stepping pero poco cubiertos in-lib:

| Área | Acción |
|------|--------|
| `engine/gravity.rs` | LET local/hierárquico/SFC, `make_solver` PM/TreePM/BH |
| `engine/stepping/context.rs` | Unit tests directos de `step_mhd`, `step_sph`, `step_rt`, `step_sidm`, `step_fr`, `step_agn`, checkpoint/snap |
| `gadget-ng-sph/integrator.rs` | `courant_dt` |
| `gadget-ng-mhd/induction.rs` | `alfven_dt` |

## Tests añadidos

### `engine/gravity.rs`

- `compute_forces_local_tree_non_empty`, `with_costs`, `hierarchical_let`, `sfc_let` (remoto)
- `local_bh_walk_params_geometric_bmax`
- `make_solver_pm_and_treepm_run`, `make_solver_barnes_hut_runs`

### `engine/stepping/context.rs`

- `step_mhd_*`, `step_sidm_*`, `step_fr_*`, `step_rt_*`, `step_reionization_*`
- `step_agn_*`, `step_sph_*`, `step_insitu_*`, `step_checkpoint_*`, `step_snap_frame_*`
- `step_mhd_chem_ambipolar_requires_matching_lengths`

### `gadget-ng-sph` / `gadget-ng-mhd`

- `courant_dt_finite_for_uniform_gas`
- `alfven_dt_finite_with_magnetized_gas`, `alfven_dt_infinite_without_b_field`

## Verificación

```bash
cargo test -p gadget-ng-cli -p gadget-ng-sph -p gadget-ng-mhd
cargo fmt --all
cargo clippy -p gadget-ng-cli -p gadget-ng-sph -p gadget-ng-mhd -- -D warnings
cargo tarpaulin --workspace --lib --features gadget-ng-io/hdf5
```

## Meta

+1–3 pp sobre 62,48% → **~64–65%** global `--lib`.

## Medición (2026-05-22)

**63,52%** (12 571 / 19 792 líneas, **+1,03 pp** vs 62,48% post-oleada 13).

| Módulo | Antes → Después | Notas |
|--------|-----------------|-------|
| `engine/gravity.rs` | 44/114 → **106/114** (+54 pp local) | LET, SFC, make_solver PM/TreePM |
| `engine/stepping/context.rs` | 172/344 → **179/344** | step_* directos |
| `gadget-ng-sph/integrator.rs` | 22/160 → **122/160** (+63 pp local) | `courant_dt` + KDK path |
| `gadget-ng-mhd/induction.rs` | — → **136/156** | `alfven_dt` |

Ramas pendientes: CUDA/GPU en `context.rs`, integrador SPH gadget-2 completo, Barnes–Hut vía `make_solver` (stack overflow en unit test aislado; cubierto por smokes stepping).

## Fuera de alcance

- GPU/CUDA paths en `context.rs` (requieren features + hardware).
- MPI / SFC-LET multirank.
