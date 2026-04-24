# Phase 118 — Función de luminosidad y colores galácticos

**Fecha**: 23 de abril de 2026  
**Capa**: Observables (SSP analítica)

## Resumen

Implementación de observables galácticos mediante síntesis de población estelar (SSP)
analítica simplificada: luminosidad total, índices de color B-V y g-r (SDSS).
Permite comparar simulaciones con observaciones fotométricas directamente.

## Cambios técnicos

### `crates/gadget-ng-analysis/src/luminosity.rs` (nuevo)

```rust
pub struct LuminosityResult {
    pub l_total: f64,   // L_sun
    pub bv: f64,        // mag B-V
    pub gr: f64,        // mag g-r (SDSS)
    pub n_stars: usize,
}

pub fn stellar_luminosity_solar(mass: f64, age_gyr: f64, metallicity: f64) -> f64
// L/L_sun = M/M_sun × age^{-0.8} × f_Z(Z)

pub fn bv_color(age_gyr: f64, metallicity: f64) -> f64
// B-V: 0.35 + 0.25 log(age) + 0.10 log(Z)

pub fn gr_color(age_gyr: f64, metallicity: f64) -> f64
// g-r: 0.24 + 0.18 log(age) + 0.07 log(Z)

pub fn galaxy_luminosity(particles: &[Particle]) -> LuminosityResult
// Suma sobre partículas estelares, promedia colores por luminosidad
```

### CLI `gadget-ng analyze --luminosity`
- Nuevo flag `--luminosity` en `AnalyzeParams`.
- Calcula `galaxy_luminosity(&data.particles)`.
- Escribe `analyze/luminosity.json` con L_total, B-V, g-r, N_stars.

## Modelo físico

SSP analítica BC03 de un parámetro:
```
L/L_sun = (M/M_sun) × age_Gyr^{-0.8} × f_Z(Z)
f_Z = 1 + 2.5 × log10(Z/Z_sun)  (corrección metálica)
```

Colores populaciones:
- Jóvenes (< 1 Gyr), metal-pobres: B-V ≈ 0.3, g-r ≈ 0.2 (galaxia azul)
- Viejas (> 5 Gyr), solar: B-V ≈ 0.8, g-r ≈ 0.6 (galaxia roja)

## Tests

7 tests en `phase118_luminosity.rs` — todos pasan ✅:

1. `luminosity_scales_with_mass` — L ∝ M verificado
2. `luminosity_decreases_with_age` — L decrece con edad
3. `dm_does_not_contribute_to_luminosity` — DM excluido
4. `galaxy_luminosity_sums_stars` — suma correcta de estrellas
5. `bv_color_in_physical_range` — B-V ∈ (-0.5, 2.0)
6. `gr_color_in_physical_range` — g-r ∈ (-0.5, 1.5)
7. `no_stars_returns_zero_luminosity` — sin estrellas L=0

## Referencias

- Bruzual & Charlot (2003) MNRAS 344, 1000 — BC03 SSP models
- Worthey (1994) ApJS 95, 107 — índices espectrales
- Baldry & Glazebrook (2003) ApJ 593, 258 — distribución de colores galácticos
