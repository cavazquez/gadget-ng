# Phase 116 — Modo radio AGN (bubble feedback)

**Fecha**: 23 de abril de 2026  
**Capa**: 2B — Física avanzada (Croton et al. 2006)

## Resumen

Implementación del modo radio AGN para feedback mecánico mediante jets/burbujas.
Cuando la tasa de acreción es baja (`Ṁ/Ṁ_Edd < 0.01`), el AGN opera en modo radio:
inyecta jets mecánicos en una burbuja esférica en lugar de calentamiento térmico difuso.

## Cambios técnicos

### `crates/gadget-ng-core/src/config.rs`
Nuevos campos en `AgnSection`:
- `f_edd_threshold: f64` (default: `0.01`) — umbral de Eddington quasar/radio
- `r_bubble: f64` (default: `2.0`) — radio de la burbuja de jets
- `eps_radio: f64` (default: `0.2`) — eficiencia del modo radio

### `crates/gadget-ng-sph/src/agn.rs`
Nueva función `bubble_feedback_radio`:
```rust
pub fn bubble_feedback_radio(
    bh: &BlackHole, particles: &mut [Particle],
    _params: &AgnParams, r_bubble: f64, eps_radio: f64, dt: f64,
)
```
- Energía mecánica: `E_radio = eps_radio × Ṁ × c² × dt`.
- Distribuye como kicks tangenciales (perpendiculares a r_BH-partícula) ponderados por masa.
- Solo actúa sobre partículas Gas dentro de la burbuja.

Nueva función `apply_agn_feedback_bimodal`:
```rust
pub fn apply_agn_feedback_bimodal(
    particles, bhs, params, f_edd_threshold, r_bubble, eps_radio, dt
)
```
- Calcula `f_edd = Ṁ / Ṁ_Edd` (con `Ṁ_Edd ≈ M_BH × 1e-10`).
- `f_edd > threshold`: modo quasar (feedback térmico, comportamiento original).
- `f_edd ≤ threshold`: modo radio (kicks mecánicos en burbuja).

## Modelo físico

```
modo quasar:  f_edd > 0.01  →  E_th = eps_fb × Ṁ × c² × dt  (térmico difuso)
modo radio:   f_edd ≤ 0.01  →  E_mec = eps_radio × Ṁ × c² × dt  (kicks tangenciales)
```

Los kicks tangenciales evitan acumulación de momento lineal y modelan jets bipolares.

## Tests

7 tests en `phase116_radio_agn.rs` — todos pasan ✅:

1. `radio_bubble_changes_velocity` — bubble feedback cambia velocidad
2. `gas_outside_bubble_not_affected` — gas fuera del radio no cambia
3. `zero_eps_radio_no_kick` — eps_radio=0 sin kick
4. `quasar_mode_increases_internal_energy` — modo quasar inyecta energía térmica
5. `radio_mode_changes_velocity_not_energy` — modo radio usa kicks, no calor
6. `agn_section_radio_params_serde` — serialización correcta de nuevos parámetros
7. `dm_not_affected_by_radio_mode` — DM no recibe bubble feedback

## Referencias

- Croton et al. (2006) MNRAS 365, 11 — modo radio AGN en SAMs
- Sijacki & Springel (2006) MNRAS 366, 397 — bubble feedback en SPH
