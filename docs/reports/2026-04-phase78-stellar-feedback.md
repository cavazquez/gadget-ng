# Phase 78 — Stellar Feedback: Kicks Estocásticos de Supernovas

**Crate**: `gadget-ng-sph/src/feedback.rs`  
**Fecha**: 2026-04

## Resumen

Implementación del modelo de feedback estelar por supernovas estocásticas, basado
en Springel & Hernquist (2003) y Dalla Vecchia & Schaye (2012). El módulo se
integra en el loop de stepping vía la macro `maybe_sph!` en `engine.rs`.

## Modelo físico

### Tasa de formación estelar (ley Schmidt-Kennicutt)

```
SFR(i) = A × (ρ_i / ρ_sf)^n   [masa/tiempo]
```

con `A = 0.1`, `n = 1.5`, aplicada solo a gas con `ρ > ρ_sf`.

La densidad se aproxima como `ρ ≈ m / (4π/3 × h³)` usando la longitud de suavizado.

### Kick estocástico

Para cada partícula de gas con `SFR > sfr_min`:
- Probabilidad de kick: `p = 1 - exp(-SFR × dt / m)`
- Si se aplica: velocidad ← velocidad + `v_kick × n̂` (dirección aleatoria)
- Energía interna: `ΔU = ε_SN × E_SN × SFR × dt / m`

La constante de SN: `E_SN ≈ ε_SN × 1.54e-3 (km/s)² / (10¹⁰ M_sun)` en unidades internas.

## Nuevos campos en `SphSection`

```toml
[sph.feedback]
enabled      = true
v_kick_km_s  = 350.0   # velocidad de kick en km/s
eps_sn       = 0.1     # fracción de E_SN transferida
rho_sf       = 0.1     # densidad umbral para SFR
sfr_min      = 1e-4    # SFR mínima para activar kick
```

## Funciones exportadas

| Función | Descripción |
|---|---|
| `compute_sfr(particles, cfg)` | Calcula SFR para cada partícula |
| `apply_sn_feedback(particles, sfr, cfg, dt, seed)` | Aplica kicks estocásticos |
| `total_sn_energy_injection(sfr, masses, cfg, dt)` | Energía total inyectada (monitoring) |

## Integración en `engine.rs`

El feedback se aplica dentro de la macro `maybe_sph!` después de SPH y cooling:

```rust
if cfg.sph.feedback.enabled {
    let sfr = compute_sfr(&local, &cfg.sph.feedback);
    apply_sn_feedback(&mut local, &sfr, &cfg.sph.feedback, dt, &mut seed);
}
```

## Tests

8 tests en `feedback::tests`:
- `sfr_zero_below_threshold`
- `sfr_positive_above_threshold`
- `sfr_zero_for_dm_particle`
- `feedback_disabled_no_kick`
- `feedback_enabled_probabilistic_kick`
- `kick_velocity_magnitude_correct`
- `energy_injection_positive`
- `unit_vector_on_sphere`
