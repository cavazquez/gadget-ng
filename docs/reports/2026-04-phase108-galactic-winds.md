# Phase 108 — Vientos Galácticos

**Fecha**: 2026-04-23  
**Crates**: `gadget-ng-core`, `gadget-ng-sph`  
**Archivos clave**: `crates/gadget-ng-core/src/config.rs`, `crates/gadget-ng-sph/src/feedback.rs`

## Problema

El módulo de feedback estelar (Phase 78) implementaba kicks SN estocásticos pero no modelaba
*vientos galácticos* sostenidos. Los vientos son el mecanismo principal de retroalimentación
para halos de baja masa (M < 10¹² M☉), complementarios al AGN (que domina en halos más masivos).

## Modelo Implementado

Basado en **Springel & Hernquist (2003)**, MNRAS 339, 289:

**Probabilidad de lanzamiento por paso**:
```
p_wind = 1 - exp(-η × SFR × dt / mass)
```
donde `η` es el factor de carga de masa (`mass_loading`).

**Kick de velocidad**: `v_wind` en dirección aleatoria uniforme sobre la esfera.

**Factor de carga de masa típico**: `η = 2.0` (2 unidades de masa en viento por unidad de
masa estelar formada).

## Cambios en Código

### `WindParams` en `config.rs`

```rust
pub struct WindParams {
    pub enabled: bool,
    pub v_wind_km_s: f64,   // default: 480 km/s
    pub mass_loading: f64,  // default: 2.0
    pub t_decoupling_myr: f64,  // default: 0.0
}
```

Integrado en `FeedbackSection` con `#[serde(default)]` para compatibilidad TOML retroactiva.

### `apply_galactic_winds` en `feedback.rs`

```rust
pub fn apply_galactic_winds(
    particles: &mut [Particle],
    sfr: &[f64],
    cfg: &WindParams,
    dt: f64,
    seed: &mut u64,
) -> Vec<usize>
```

Retorna los índices de partículas lanzadas para logging/estadísticas.

## Tests

Archivo: `crates/gadget-ng-physics/tests/phase108_galactic_winds.rs` (8 tests)

| Test | Descripción |
|------|-------------|
| `wind_disabled_no_launch` | Vientos desactivados → sin kick |
| `dm_particles_never_launched` | DM siempre ignorado |
| `zero_sfr_no_wind` | SFR=0 → sin viento |
| `high_sfr_most_particles_launched` | SFR alta → ≥90% lanzados |
| `wind_kick_magnitude_correct` | `|Δv| = v_wind` |
| `wind_params_serde_roundtrip` | JSON roundtrip de WindParams |
| `feedback_section_wind_default_in_toml` | TOML sin `[wind]` usa defaults |
| `sfr_and_wind_pipeline` | Pipeline SFR → vientos integrado |

## Uso en Configuración

```toml
[sph.feedback]
enabled = true
rho_sf = 0.1

[sph.feedback.wind]
enabled = true
v_wind_km_s = 480.0
mass_loading = 2.0
```

## Impacto

- Regulación de formación estelar en halos de baja masa.
- Complementa el feedback AGN (Phase 96) para la jerarquía completa de retroalimentación.
- Compatible con `maybe_sph!` existente en el motor; se puede invocar con `apply_galactic_winds`.
