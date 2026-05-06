# Clippy all-targets backlog (2026-05)

> **Estado al 6 de mayo de 2026:** `cargo clippy --workspace --all-targets -- -D warnings` **pasa limpio** en `main`.
> Última limpieza masiva: fixes en tests/benches de `gadget-ng-physics`, `gadget-ng-parallel`, `gadget-ng-gpu`, `gadget-ng-treepm`, `gadget-ng-sph`.

## Deuda saldada en esta iteración (2026-05-06)

- `useless_vec` en tests (`sfc_hardening`, `pm_scatter_gather`).
- `deprecated` `criterion::black_box` → `std::hint::black_box` en benches GPU y SPH.
- `ptr_arg`: `&mut Vec<T>` → `&mut [T]` en ~20 firmas de tests physics.
- `needless_range_loop` en tests (`halo3d`, `zeldovich_pancake`, `magnetic_forces`, `pm_mesh_convergence`, `sfc_weighted`).
- `doc_overindented_list_items` / `doc_lazy_continuation` en doc comments de tests.
- `field_reassign_with_default` en tests (`gmc_collapse`, `dark_energy_wz`).
- `approx_constant` (`3.14`) en test HDF5 MHD.
- `too_many_arguments` en helpers de tests (`sidm_cross_section`, `long_growth`) — permitido con `#[allow]` donde es razonable.
- `erasing_op` en test `cosmo_pm_slab` — permitido con `#[allow]` por claridad documental.
- `cloned_ref_to_slice_refs` en tests `treepm_distributed` / `treepm_halo3d` — permitido con `#[allow]` porque las partículas se reutilizan en múltiples structs.
- Overly complex bool expr (`db > 0.0 || true`) en test MHD induction.

## Pipeline actual

| Job | Comando | Estado |
|-----|---------|--------|
| Bloqueante | `cargo clippy --workspace -- -D warnings` | ✅ |
| **Advisory → candidato a bloqueante** | `cargo clippy --workspace --all-targets -- -D warnings` | **✅ Limpio** |

## Criterio para graduar a bloqueante

1. ✅ `clippy-all-targets` pasa en `main` (ahora).
2. ⏳ Mantenerlo limpio por al menos 2 semanas consecutivas.
3. ⏳ Migrar el job advisory a bloqueante en `.github/workflows/ci.yml`.
4. ⏳ Mantener el workflow `physics-validation` para detectar regresiones lentas.

> **Nota para copilots:** Si introduces un warning nuevo bajo `--all-targets`, arréglalo inmediatamente o documenta la excepción en este archivo.
