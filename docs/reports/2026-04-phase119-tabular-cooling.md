# Phase 119 — Enfriamiento tabulado mejorado (S&D93)

**Fecha**: 23 de abril de 2026  
**Capa**: 2D — Enfriamiento tabulado (Sutherland & Dopita 1993)

## Resumen

Implementación del enfriamiento tabulado con interpolación bilineal en (Z/Z_sun, log T),
basado en las tablas originales de Sutherland & Dopita (1993). Mejora la precisión del
enfriamiento metálico respecto al fitting analítico de la Phase 111.

## Cambios técnicos

### `crates/gadget-ng-core/src/config.rs`
Nueva variante `CoolingKind::MetalTabular` (las anteriores siguen funcionando).

### `crates/gadget-ng-sph/src/cooling.rs`

Tabla interna embebida 7×20:
```rust
const COOLING_TABLE_Z: [f64; 7] = [0.0, 1e-4, 1e-3, 0.01, 0.1, 1.0, 2.0];  // Z/Z_sun
const COOLING_TABLE_LOG_T: [f64; 20] = [4.0, 4.25, ..., 8.75];              // log10(T/K)
const COOLING_TABLE: [[f64; 20]; 7] = [...];  // Λ en unidades internas
```

Nueva función:
```rust
pub fn cooling_rate_tabular(
    u: f64, rho: f64, metallicity: f64, gamma: f64, t_floor_k: f64,
) -> f64
```
1. Convierte `u` → T mediante `u_to_temperature`.
2. Clamping de T al rango de la tabla.
3. Búsqueda de bins en log T y Z/Z_sun.
4. Interpolación bilineal 4 puntos.

`apply_cooling` ahora despacha `MetalTabular` → `cooling_rate_tabular`.

## Modelo físico

Interpolación bilineal:
```
Λ(T, Z) = Σ_{ij} Λ_ij × (1-fz)(1-ft) / fz×ft
```

La tabla cubre:
- Z/Z_sun de 0 (primordial) a 2 (super-solar)
- log T de 4.0 a 8.75 (10⁴ K a ~6×10⁸ K)

El enfriamiento metálico es máximo alrededor de T ~ 10⁵ K (pico de enfriamiento óptico) y
decrece en el régimen de bremsstrahlung (T > 10⁷ K, ∝ T^{0.5}).

## Backward Compatibility

`CoolingKind::MetalCooling` (fitting analítico Phase 111) sigue disponible y funcional.
Los archivos de configuración existentes no requieren actualización.

## Tests

6 tests en `phase119_metal_tabular.rs` — todos pasan ✅:

1. `tabular_rate_positive_above_floor` — Λ > 0 para T=1e6 K
2. `tabular_rate_zero_below_floor` — Λ = 0 para T < T_floor
3. `tabular_rate_increases_with_metallicity` — Λ(Z_solar) > Λ(Z_low)
4. `apply_cooling_tabular_reduces_u` — `u` decrece con MetalTabular
5. `backward_compat_metal_cooling_still_works` — MetalCooling analítico funciona
6. `cooling_kind_metal_tabular_serde` — serde correcto de la variante

## Referencias

- Sutherland & Dopita (1993) ApJS 88, 253 — tablas de enfriamiento metal
- Wiersma et al. (2009) MNRAS 393, 99 — enfriamiento tabular en OWLS/EAGLE
