# Phase 138 — Freeze-Out de B en ICM

**Fecha:** 2026-04-23  
**Estado:** Completado ✓  
**Tests:** 6/6 passing

## Objetivo

Implementar el criterio de flux-freeze magnético para gas difuso de alta β-plasma en el ICM
(Intracluster Medium), donde el campo B se "congela" con el fluido.

## Física

### Teorema de Alfvén (Flux Freezing)

En plasma ideal (resistividad → 0), el flujo magnético a través de cualquier superficie
co-moviente se conserva:

```
Φ = ∫ B · dA = cte
```

Para compresión isótropa en 3D:
```
B ∝ ρ^{2/3}
```

### Criterio β-plasma

El flujo magnético se puede considerar "congelado" cuando la presión térmica domina sobre
la magnética (`β >> 1`). El umbral `β_freeze` separa dos regímenes:

- `β > β_freeze`: gas difuso, flux-frozen, `B ∝ ρ^{2/3}`
- `β ≤ β_freeze`: campo dinámicamente importante, MHD completa

## Implementación

### Archivo nuevo: `crates/gadget-ng-mhd/src/flux_freeze.rs`

- **`apply_flux_freeze(particles, gamma, beta_freeze, rho_ref)`**: Escala B para cada
  partícula con `β > beta_freeze`: `B_new = B × (ρ/ρ_ref)^{2/3}`.

- **`mean_gas_density(particles) → f64`**: Densidad media del gas (referencia).

- **`flux_freeze_error(b_actual, b0, rho, rho0) → f64`**: Error relativo respecto a B ∝ ρ^{2/3}.

### Modificaciones de configuración

- `MhdSection.beta_freeze: f64` (default: `100.0`)

## Tests

| Test | Descripción |
|------|-------------|
| `zero_b_no_change` | B=0 → no-op |
| `high_beta_applies_scale` | β>100 aplica escala B ∝ ρ^{2/3} |
| `low_beta_no_change` | β<100 → sin cambio (B dinámico) |
| `scale_follows_rho_power_2_3` | Escala exacta para compresión 8× |
| `mean_density_uniform` | Densidad media correcta |
| `flux_freeze_error_exact_zero` | Error=0 para B=B₀(ρ/ρ₀)^{2/3} |

## Relevancia Astrofísica

- **ICM de cúmulos**: β ~ 10-1000, campos B primordiales amplifican con formación de estructura
- **Filamentos cosmológicos**: β ~ 100-10000, magnetización rastreada por compresión
- **Proto-estrellas**: β << 1 en disco, flux-freeze en envoltura externa
