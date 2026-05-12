# Phase 177 — Cooling+SF+Feedback de producción

**Fecha**: 2026-05-11  
**Estado**: ✅ Completada

## Objetivo

Consolidar la capa bariónica subgrid para régimen cosmológico de producción con tres piezas:

1. Cooling/heating neto con fondo UV cosmológico y auto-apantallamiento.
2. Star formation basada en presión como alternativa a la ley en densidad.
3. Feedback térmico estocástico con salto de temperatura configurable.

## Cambios implementados

### Configuración

Archivo: `crates/gadget-ng-core/src/config/sections/sph.rs`

- `CoolingKind::UvBackground`.
- `UvBackgroundModel::{None, Hm2012}`.
- Nuevos campos SPH:
  - `uv_background_model`
  - `reionization_redshift`
  - `self_shielding_nh_cm3`
- Nuevos enums de feedback/SF:
  - `StarFormationModel::{DensityLaw, PressureLaw}`
  - `StellarFeedbackMode::{Kinetic, ThermalStochastic}`
- Nuevos campos en `FeedbackSection`:
  - `sf_model`
  - `sf_pressure_norm`
  - `sf_pressure_index`
  - `feedback_mode`
  - `delta_t_heat_k`
  - `n_heat_neighbors`

### Cooling UVB

Archivo: `crates/gadget-ng-sph/src/cooling.rs`

- `cooling_rate_uvb(u, rho, Z, gamma, floor, cfg, z)`:
  - `Lambda_net = Lambda_cool - Gamma_photo`.
  - `Gamma_photo` modulado por transición de reionización (`reionization_switch`) y auto-apantallamiento.
- Nuevas variantes con redshift explícito:
  - `apply_cooling_with_redshift(...)`
  - `apply_cooling_mhd_with_redshift(...)`
- `apply_cooling(...)` y `apply_cooling_mhd(...)` mantienen compatibilidad y delegan con `z=0`.

### SF por presión + feedback térmico estocástico

Archivo: `crates/gadget-ng-sph/src/feedback.rs`

- `compute_sfr_pressure(...)` con ley `SFR ∝ (P/P0)^n`.
- `compute_sfr_model(...)` selecciona `density_law` o `pressure_law`.
- `apply_thermal_feedback_stochastic(...)` implementa inyección térmica estocástica usando:
  - `feedback_mode = thermal_stochastic`
  - `delta_t_heat_k`
  - `n_heat_neighbors`
  - semilla determinística externa del loop del engine.

### Integración en engine

Archivo: `crates/gadget-ng-cli/src/engine/stepping/context.rs`

- `step_sph` ahora:
  - Pasa redshift a cooling (`apply_cooling_with_redshift` / `apply_cooling_mhd_with_redshift`).
  - Usa `compute_sfr_model(...)` para ISM y feedback.
  - Selecciona feedback por modo:
    - `Kinetic` → `apply_sn_feedback`
    - `ThermalStochastic` → `apply_thermal_feedback_stochastic`

## Tests

### Nuevos tests de física

Archivo: `crates/gadget-ng-physics/tests/phase177_cooling_sf_feedback.rs`

- `uvb_reduces_net_cooling_after_reionization`
- `pressure_law_sfr_increases_with_pressure`
- `thermal_stochastic_feedback_injects_energy`

Resultado local: **3/3 OK**.

### Regresión relevante

Ejecutados:

- `phase108_galactic_winds` ✅
- `phase111_metal_cooling` ✅
- `phase112_stellar_spawning` ✅
- `phase114_ism_multiphase` ✅
- `phase115_stellar_winds` ✅
- `cargo test -p gadget-ng-sph` ✅

## Ejemplo de configuración

Se añade ejemplo:

- `examples/cosmo_128_sph_phase177_uvb.toml`

incluyendo activación de `cooling = "uv_background"`, `sf_model = "pressure_law"` y
`feedback_mode = "thermal_stochastic"`.

## Referencias

- Springel, V. & Hernquist, L. (2003), MNRAS 339, 289.
- Sutherland, R. & Dopita, M. (1993), ApJS 88, 253.
- Wiersma, R., Schaye, J. & Smith, B. (2009), MNRAS 393, 99.
- Schaye, J. & Dalla Vecchia, C. (2008), MNRAS 383, 1210.
- Dalla Vecchia, C. & Schaye, J. (2012), MNRAS 426, 140.
- Haardt, F. & Madau, P. (2012), ApJ 746, 125.
