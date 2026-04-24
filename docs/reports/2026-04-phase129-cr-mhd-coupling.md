# Phase 129 — Acoplamiento CR–B: Difusión Suprimida por |B|

**Fecha:** 2026-04-23  
**Estado:** ✅ COMPLETADA  
**Tests:** 6/6 pasados

## Objetivo

Acoplar el transporte de rayos cósmicos (CRs) con el campo magnético: en un plasma magnetizado, la difusión CR está suprimida perpendicularmente a B por scattering de ondas de Alfvén. Se implementa una supresión isótropa simplificada como primer paso.

## Modelo físico

La difusividad CR efectiva:

```
κ_CR_eff = κ_CR / (1 + b_cr_suppress × |B|²)
```

donde:
- `κ_CR` — coeficiente de difusión isótropo base
- `b_cr_suppress` — factor de supresión (default `1.0`)
- `|B|²` — densidad de energía magnética (en unidades internas)

Para `b_cr_suppress = 0`: recupera comportamiento clásico sin supresión.  
Para `b_cr_suppress >> 0`: difusión CR fuertemente suprimida en regiones magnetizadas.

## Cambios

### `CrSection` (config.rs)

```toml
[sph.cr]
enabled       = true
cr_fraction   = 0.1
kappa_cr      = 3e-3
b_cr_suppress = 1.0   # nuevo en Phase 129
```

### `diffuse_cr` (cosmic_rays.rs)

Firma actualizada:
```rust
pub fn diffuse_cr(particles: &mut [Particle], kappa_cr: f64, b_suppress: f64, dt: f64)
```

Para cada partícula emisora `i`:
```rust
let b2_i = |B_i|²;
let kappa_eff = kappa_cr / (1.0 + b_suppress * b2_i);
delta_cr[i] += kappa_eff * (e_cr_j - e_cr_i) * w(r_ij) * dt;
```

### Compatibilidad

Todos los tests previos de CRs (Phase 117, 120) fueron actualizados para pasar `b_suppress = 0.0`, preservando el comportamiento original.

## Tests

| Test | Descripción |
|------|-------------|
| `zero_suppress_equals_classic` | `b_suppress=0` produce resultado idéntico |
| `nonzero_suppress_reduces_diffusion` | Con `B=10`, difusión menor que sin B |
| `very_strong_b_suppresses_diffusion` | `B=1000` → difusión casi nula |
| `b_suppress_does_not_affect_injection` | Inyección CR no depende de B |
| `suppression_scales_with_b_squared` | `f_suppress` escala como `1/(1+b×B²)` |
| `cr_pressure_independent_of_b` | Presión CR solo depende de `cr_energy` |

## Motivación física

El streaming de CRs a lo largo de líneas de campo y el scattering con turbulencia de Alfvén producen supresión de la difusión perpendicular. Este modelo simplificado (factor isótropo) es suficiente para estudios galácticos de gran escala.

## Referencias

- Jubelgas et al. (2008), A&A 481, 33 — CRs en SPH con MHD.
- Pfrommer et al. (2017), MNRAS 465, 4500 — transporte anisótropo de CRs.
- Zweibel (2013), Physics of Plasmas 20, 055501 — supresión por campo B.
