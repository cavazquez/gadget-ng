# Phase 142 — Engine: RMHD + Turbulencia Integrados

**Fecha:** 2026-04-23  
**Autor:** gadget-ng automated development  
**Estado:** ✅ Completado

## Objetivo

Integrar los módulos RMHD (Phase 139) y turbulencia Ornstein-Uhlenbeck (Phase 140) en el loop principal del simulador, activándolos mediante hooks en las macros `maybe_mhd!` y `maybe_sph!` de `engine.rs`.

## Cambios implementados

### `crates/gadget-ng-cli/src/engine.rs`

**`maybe_sph!`** — Nuevos hooks al final del bloque de física SPH:

```rust
// Phase 142: Forzado turbulento Ornstein-Uhlenbeck
if cfg.turbulence.enabled {
    gadget_ng_mhd::apply_turbulent_forcing(
        &mut local, &cfg.turbulence, cfg.simulation.dt, $sph_step as u64,
    );
}
// Phase 149: Acoplamiento electrón-ión (plasma de dos fluidos)
if cfg.two_fluid.enabled {
    gadget_ng_mhd::apply_electron_ion_coupling(&mut local, &cfg.two_fluid, cfg.simulation.dt);
}
```

**`maybe_mhd!`** — Nuevos hooks en el bloque MHD:

```rust
// Phase 142: correcciones SRMHD
if cfg.mhd.relativistic_mhd {
    gadget_ng_mhd::advance_srmhd(&mut local, dt_mhd, C_LIGHT, cfg.mhd.v_rel_threshold);
}
// Phase 142: flux-freeze ICM
let rho_ref = gadget_ng_mhd::mean_gas_density(&local);
gadget_ng_mhd::apply_flux_freeze(&mut local, cfg.sph.gamma, cfg.mhd.beta_freeze, rho_ref);
// Phase 146: viscosidad Braginskii
if cfg.mhd.eta_braginskii > 0.0 {
    gadget_ng_mhd::apply_braginskii_viscosity(&mut local, cfg.mhd.eta_braginskii, dt_mhd);
}
// Phase 145: reconexión magnética
if cfg.mhd.reconnection_enabled {
    gadget_ng_mhd::apply_magnetic_reconnection(&mut local, cfg.mhd.f_reconnection, ..., dt_mhd);
}
```

### `crates/gadget-ng-core/src/config.rs`

Nuevos campos en `MhdSection`:
- `reconnection_enabled: bool` (default `false`)
- `f_reconnection: f64` (default `0.01`)
- `eta_braginskii: f64` (default `0.0`)
- `jet_enabled: bool` (default `false`)
- `v_jet: f64` (default `0.3`)
- `n_jet_halos: usize` (default `1`)

Nueva sección `TwoFluidSection`:
- `enabled: bool` (default `false`)
- `nu_ei_coeff: f64` (default `1.0`)
- `t_e_init_k: f64` (default `0.0`)

### `crates/gadget-ng-core/src/particle.rs`

- Nuevo campo `t_electron: f64` para temperatura electrónica en plasma de dos fluidos

## Tests

6 tests en `phase142_engine_rmhd_turb.rs`:
1. `two_fluid_section_defaults` — defaults correctos
2. `turbulent_forcing_modifies_velocities` — forzado activa ≠ 0
3. `flux_freeze_in_engine_no_crash` — no panic con B uniforme
4. `srmhd_sub_threshold_no_position_change` — v < threshold sin efecto
5. `reconnection_releases_heat_reduces_b` — B antiparalelo libera calor
6. `braginskii_transfers_momentum_parallel_b` — viscosidad ∥B opera

**Resultado:** 6/6 tests pasan ✅
