# Phase 115 — Vientos estelares pre-SN

**Fecha**: 23 de abril de 2026  
**Capa**: 2A — Física avanzada (OB/Wolf-Rayet)

## Resumen

Implementación del feedback mecánico de vientos estelares pre-SN de estrellas OB y Wolf-Rayet.
Las estrellas masivas (~8–40 M_sun) emiten vientos potentes ~10–30 Myr antes de explotar como SN II,
inyectando energía mecánica en el ISM.

## Cambios técnicos

### `crates/gadget-ng-core/src/config.rs`
Nuevos campos en `FeedbackSection`:
- `stellar_wind_enabled: bool` (default: `false`)
- `v_stellar_wind_km_s: f64` (default: `2000.0`) — velocidad terminal del viento
- `eta_stellar_wind: f64` (default: `0.1`) — factor de carga másica η_w = Ṁ_wind / SFR

### `crates/gadget-ng-sph/src/feedback.rs`
Nueva función `apply_stellar_wind_feedback`:
```rust
pub fn apply_stellar_wind_feedback(
    particles: &mut [Particle], sfr: &[f64],
    cfg: &FeedbackSection, dt: f64, seed: &mut u64,
) -> Vec<usize>
```
- Solo actúa sobre partículas Gas con `sfr[i] > sfr_min`.
- Probabilidad de kick: `p = η_w × sfr[i] × dt / m_i`.
- Kick en dirección aleatoria con magnitud `v_stellar_wind_km_s`.
- Retorna índices de partículas que recibieron kick.

## Modelo físico

La fracción de masa barrida por vientos estelares respecto al gas recién formado:
```
η_w = Ṁ_wind / SFR ≈ 0.1  (de observaciones)
```

Velocidad típica: `v_wind ≈ 2000 km/s` para Wolf-Rayet.

## Tests

6 tests en `phase115_stellar_winds.rs` — todos pasan ✅:

1. `stellar_wind_kick_occurs_eventually` — kick ocurre con eta alta en ≤ 200 intentos
2. `dm_not_kicked_by_stellar_wind` — DM no recibe kick
3. `disabled_stellar_wind_no_kick` — desactivado = no-op
4. `zero_sfr_no_stellar_wind` — SFR=0 sin kick
5. `stellar_wind_changes_velocity` — la velocidad cambia tras un kick
6. `stellar_wind_params_serde` — serialización correcta de parámetros

## Referencias

- Leitherer et al. (1999) ApJS 123, 3 — Starburst99: propiedades de vientos estelares
- Scannapieco et al. (2006) MNRAS 371, 1125 — vientos estelares en SPH cosmológico
