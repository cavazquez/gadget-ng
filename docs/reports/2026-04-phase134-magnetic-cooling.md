# Phase 134 — Cooling Magnético: Supresión por β-Plasma

**Fecha:** 2026-04-23  
**Estado:** Completado ✓  
**Tests:** 6/6 passing

## Objetivo

Implementar la supresión del cooling radiativo por el campo magnético cuando la presión magnética
es comparable a la presión térmica (β-plasma ~ 1).

## Física

En un plasma magnetizado, el transporte de calor por electrones está suprimido en la dirección
perpendicular a B. Esto reduce efectivamente la tasa de enfriamiento por bremsstrahlung y
líneas de emisión en regiones de ICM con B fuerte:

```
Λ_eff = Λ(T) / (1 + f_mag / β)
```

donde:
- `Λ(T)`: tasa de cooling estándar
- `β = 2μ₀ P_th / |B|²`: parámetro β-plasma
- `f_mag`: parámetro de supresión magnética (default: 0.1)

## Implementación

### Modificación: `crates/gadget-ng-sph/src/cooling.rs`

Nueva función `apply_cooling_mhd(particles, cfg, dt)`:
- Calcula β-plasma local para cada partícula de gas
- Aplica factor de supresión `1/(1 + f_mag/β)` a la tasa de cooling
- Con `f_mag = 0.0` → idéntico a `apply_cooling` clásico
- Con `B = 0` → sin diferencia

### Modificaciones de configuración

- `SphSection.mag_suppress_cooling: f64` (default: `0.0`)
- Engine hook: si `mag_suppress_cooling > 0.0` y `mhd.enabled` → usa `apply_cooling_mhd`

## Tests

| Test | Descripción |
|------|-------------|
| `zero_f_mag_equals_classic` | f_mag=0 reproduce cooling estándar exactamente |
| `strong_b_suppresses_cooling` | B fuerte (B=1e9) reduce tasa de cooling |
| `zero_b_no_suppression` | B=0 sin diferencia entre versiones |
| `very_strong_b_max_suppression` | B muy fuerte → supresión máxima |
| `cold_gas_not_cooled_below_floor` | T_floor respetado con supresión magnética |
| `cooling_mhd_conserves_mass` | masa conservada durante cooling |

## Relevancia Física

Este efecto es relevante en:
- **ICM de cúmulos masivos**: campos B ~ μG, β ~ 100, supresión marginal
- **Halos de galaxias elípticas**: β ~ 1-10, supresión significativa
- **Jets AGN**: β << 1, cooling altamente suprimido (dominado por B)
