# Phase 114 — ISM Multifase fría-caliente

**Fecha**: 23 de abril de 2026  
**Capa**: 2A — Física avanzada (Springel & Hernquist 2003)

## Resumen

Implementación del modelo de ISM multifase fría-caliente basado en Springel & Hernquist (2003).
El ISM se trata como una mezcla de nubes frías densas y gas caliente difuso, combinados
en una presión efectiva estabilizadora.

## Cambios técnicos

### `crates/gadget-ng-core/src/particle.rs`
- Agrega `u_cold: f64` con `#[serde(default)]` — energía interna de la fase fría del ISM.
- También se agrega `cr_energy: f64` (preparación para Phase 117).
- Todos los constructores (`new`, `new_gas`, `new_star`) inicializan los nuevos campos en `0.0`.

### `crates/gadget-ng-core/src/config.rs`
- Nueva struct `IsmSection` con:
  - `enabled: bool` (default: `false`)
  - `q_star: f64` (default: `2.5`) — parámetro de rigidez
  - `f_cold: f64` (default: `0.5`) — fracción fría en equilibrio
- Campo `ism: IsmSection` agregado a `SphSection`.

### `crates/gadget-ng-sph/src/ism.rs` (nuevo)
- `effective_pressure(rho, u, u_cold, q_star, gamma)` — presión efectiva combinada.
- `update_ism_phases(particles, sfr, rho_sf, cfg, dt)` — equilibración de fases:
  - Gas denso (sfr > 0): `u_cold` relaja hacia `f_cold × u_total × ρ/ρ_sf`.
  - Conserva `u + u_cold = const`.
  - Gas sub-umbral: `u_cold` se disipa exponencialmente hacia `u`.
- `effective_u(p, q_star)` — energía efectiva `u + q* × u_cold`.

## Modelo físico

```
P_eff = (γ - 1) × ρ × (u + q* × u_cold)
```

En equilibrio: `u_cold_eq = f_cold × (u + u_cold) × min(ρ/ρ_sf, 1)`

## Tests

7 tests en `phase114_ism_multiphase.rs` — todos pasan ✅:

1. `effective_pressure_exceeds_thermal` — P_eff > P_thermal con u_cold > 0
2. `effective_pressure_equals_thermal_when_no_cold` — P_eff = P_thermal con u_cold = 0
3. `u_cold_grows_for_dense_gas` — u_cold crece para gas denso con SFR activa
4. `total_energy_conserved` — conservación de u + u_cold
5. `u_cold_dissipates_below_threshold` — disipación en gas sub-umbral
6. `disabled_ism_no_change` — módulo desactivado no cambia estado
7. `effective_u_formula` — `effective_u` calcula correctamente

## Referencias

- Springel & Hernquist (2003) MNRAS 339, 289
