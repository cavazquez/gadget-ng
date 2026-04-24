# Phase 100 — AGN con halos FoF

**Fecha:** 2026-04-23  
**Estado:** ✅ Completado

## Objetivo

Refactorizar el módulo AGN para colocar semillas de agujeros negros (BH seeds) en los centros de los N halos FoF más masivos identificados in-situ, en lugar de posiciones fijas en el centro de la caja.

## Cambios implementados

### `crates/gadget-ng-core/src/config.rs`
- Nuevo campo `n_agn_bh: usize` (default: 1) en `AgnSection`.
- Función `default_n_agn_bh()`.
- `TOML`: `[sph.agn] n_agn_bh = 3` para colocar BH en los 3 halos más masivos.

### `crates/gadget-ng-cli/src/insitu.rs`
- Nueva struct `InsituSideEffects { halo_centers: Vec<Vec3> }`.
- `maybe_run_insitu` ahora retorna `(bool, InsituSideEffects)`.
- Los halos se ordenan por masa DESC y sus centros de masa se retornan.

### `crates/gadget-ng-cli/src/engine.rs`
- Nueva variable `halo_centers: Vec<Vec3>` (antes de `maybe_insitu!`).
- `maybe_insitu!` actualiza `halo_centers` con los centros de halos.
- `maybe_agn!` refactorizado:
  - Si `halo_centers` no vacío: sincroniza `n_agn_bh` BHs con los primeros N centros.
  - Fallback: semilla en el centro de la caja hasta el primer paso de in-situ.

## Lógica del flujo

```
maybe_insitu!() → halos ordenados por masa → halo_centers[0..N]
                                                      ↓
maybe_agn!() → agn_bhs[i].pos = halo_centers[i]  (i < n_agn_bh)
```

## Tests (`crates/gadget-ng-physics/tests/phase100_agn_fof.rs`)

| Test | Descripción | Estado |
|------|-------------|--------|
| `fof_halos_sorted_by_mass_desc` | Ordenamiento por masa | ✅ |
| `fof_halo_centers_near_cluster_positions` | COM cerca de cluster | ✅ |
| `agn_bh_placed_at_halo_center` | BH en halo más masivo | ✅ |
| `agn_two_bhs_match_two_halos` | 2 BHs en 2 halos | ✅ |
| `agn_fallback_to_box_center_without_halos` | Fallback sin halos | ✅ |
| `agn_feedback_applies_near_bh` | Feedback térmico activo | ✅ |

**Total: 6/6 tests pasan**

## Configuración de ejemplo

```toml
[sph.agn]
enabled      = true
n_agn_bh     = 3        # BH en los 3 halos más masivos
eps_feedback = 0.05
m_seed       = 1e5
r_influence  = 1.0
```
