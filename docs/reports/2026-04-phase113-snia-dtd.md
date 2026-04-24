# Phase 113 — SN Ia con Distribución de Retraso Temporal (DTD)

**Fecha**: 2026-04-23
**Estado**: ✅ Completado

## Objetivo

Implementar el feedback de supernovas de Tipo Ia (SN Ia) con distribución de retraso
temporal (DTD) power-law, inyectando energía térmica y hierro a las partículas de gas
vecinas de cada estrella.

## Física implementada

### DTD power-law (Maoz & Mannucci 2012)

```
R_Ia(t) = A_Ia × (t / 1 Gyr)^{-1}   [SN / Gyr / M_sun]
```

para `t > t_ia_min_gyr`.

### Número esperado de SN Ia por paso

```
N_exp = A_Ia × (t / Gyr)^{-1} × dt_gyr × m_star
```

Se sortea estocásticamente: `p = 1 - exp(-N_exp)`.

### Distribución de energía y Fe

Si ocurre una SN Ia:
- Se buscan vecinos de gas dentro de `r < 2 × h_i`
- Cada vecino recibe fracción ponderada de:
  - **Energía térmica**: `Δu_j = frac_j × e_ia_code`
  - **Hierro**: `ΔZ_j = frac_j × 0.002 / m_j` (yield Fe ~0.002 M_sun por SN Ia)
- Si no hay vecinos: la energía y el Fe van a la partícula de gas más cercana

### Integración DTD

```
∫_{t_min}^{10 Gyr} A_Ia × t^{-1} dt = A_Ia × ln(10/t_min) ≈ 9.2×10⁻³ SN/M_sun
```

Esto es consistente con las observaciones (Maoz & Mannucci 2012: ~1.3×10⁻³ SN/M_sun).

### `advance_stellar_ages`

Función auxiliar que incrementa `stellar_age` de todas las estrellas cada paso:

```rust
pub fn advance_stellar_ages(particles: &mut [Particle], dt_gyr: f64)
```

### Nuevos parámetros en `FeedbackSection`

| Campo | Default | Descripción |
|-------|---------|-------------|
| `a_ia` | `2e-3` | Normalización DTD [SN/Gyr/M_sun] |
| `t_ia_min_gyr` | `0.1` | Tiempo mínimo de retraso [Gyr] |
| `e_ia_code` | `1.54e-3 × 1.3` | Energía por SN Ia [unidades internas] |

## Archivos modificados

| Archivo | Cambio |
|---------|--------|
| `gadget-ng-core/src/config.rs` | `a_ia`, `t_ia_min_gyr`, `e_ia_code` en `FeedbackSection` |
| `gadget-ng-sph/src/feedback.rs` | `apply_snia_feedback`, `advance_stellar_ages` |
| `gadget-ng-sph/src/lib.rs` | Re-exporta ambas funciones |

## Tests

`tests/phase113_snia_dtd.rs` — 7 tests, todos ✅:

| Test | Descripción |
|------|-------------|
| `no_ia_for_young_stars` | Estrellas jóvenes (<t_min) no explotan como SN Ia |
| `disabled_no_ia` | Módulo desactivado = no-op |
| `energy_injected_to_neighbor` | Energía inyectada al gas vecino |
| `iron_distributed_to_neighbor` | Hierro distribuido al vecino |
| `dtd_integrated_fraction_reasonable` | Fracción integrada ∈ [10⁻³, 10⁻¹] SN/M_sun |
| `feedback_snia_params_serde` | Serde de nuevos parámetros |
| `advance_stellar_ages_test` | `stellar_age` incrementa; DM/gas no cambia |

## Referencias

- Maoz & Mannucci (2012) PASA 29, 447 — observaciones DTD SN Ia
- Scannapieco & Bildsten (2005) ApJL 629, 85 — DTD power-law
- Greggio (2005) A&A 441, 1055 — modelos analíticos de DTD
