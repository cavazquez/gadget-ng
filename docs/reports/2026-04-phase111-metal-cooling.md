# Phase 111 — Enfriamiento por Metales (MetalCooling)

**Fecha**: 2026-04-23
**Estado**: ✅ Completado

## Objetivo

Extender el modelo de enfriamiento radiativo con una contribución dependiente de la
metalicidad, siguiendo el fitting analítico de Sutherland & Dopita (1993).

## Física implementada

### Modelo de enfriamiento total

```
Λ(T, Z) = Λ_HHe(T) + (Z / Z_sun) × Λ_metal(T)
```

donde `Z_sun = 0.0127` y `Λ_HHe` es el enfriamiento H+He existente (Phase 66).

### `Λ_metal(T)` — Régimenes de temperatura

| Régimen | Rango T | Fórmula |
|---------|---------|---------|
| Frío | T < 10⁴ K | 0 |
| Tibio | 10⁴ ≤ T < 10⁷ K | `Λ_m0 × (T / 10⁵)^0.7` |
| Caliente | T ≥ 10⁷ K | `Λ_m1 × (T / 10⁷)^0.5` (bremsstrahlung) |

Con `Λ_m0 = 3×10⁻⁵` y `Λ_m1 = 10⁻⁵` en unidades internas.

### Despacho en `apply_cooling`

La función `apply_cooling` ahora despacha según `cfg.cooling`:

| `CoolingKind` | Función llamada |
|---------------|-----------------|
| `None` | No-op |
| `AtomicHHe` | `cooling_rate_atomic` |
| `MetalCooling` | `cooling_rate_metal` (usa `p.metallicity`) |

## Archivos modificados

| Archivo | Cambio |
|---------|--------|
| `gadget-ng-core/src/config.rs` | Nueva variante `CoolingKind::MetalCooling` |
| `gadget-ng-sph/src/cooling.rs` | `cooling_rate_metal`, dispatch en `apply_cooling` |
| `gadget-ng-sph/src/lib.rs` | Re-exporta `cooling_rate_metal` |

## Tests

`tests/phase111_metal_cooling.rs` — 6 tests, todos ✅:

| Test | Descripción |
|------|-------------|
| `metal_cooling_exceeds_atomic_at_high_z` | Λ_metal > Λ_HHe a T=10⁶K con Z=Z_sun |
| `metal_cooling_z_zero_equals_atomic` | Z=0 → mismo resultado que AtomicHHe |
| `metal_cooling_zero_below_floor` | Sin enfriamiento bajo T_floor |
| `cooling_rate_monotone_in_z` | Λ creciente en Z |
| `cooling_kind_metal_serde_roundtrip` | `MetalCooling` serializable/deserializable |
| `apply_cooling_metal_reduces_energy` | La energía interna baja con MetalCooling |

## Referencias

- Sutherland & Dopita (1993) ApJS 88, 253 — tablas de enfriamiento metálico
- Wiersma, Schaye & Smith (2009) MNRAS 393, 99 — implementación en OWLS/EAGLE
