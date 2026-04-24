# Phase 151 — Emisión de rayos X en cúmulos

**Fecha:** 2026-04-23  
**Crate afectado:** `gadget-ng-analysis`  
**Archivo nuevo:** `crates/gadget-ng-analysis/src/xray.rs`

## Resumen

Implementación del módulo de emisión de rayos X térmico por bremsstrahlung (free-free) para cúmulos de galaxias. El modelo sigue Sarazin (1988) y Mazzotta et al. (2004).

## Física implementada

### Bremsstrahlung térmico

```text
Λ_X(T) = 3×10⁻²⁷ × √T × n_e × n_i   [erg/s/cm³]
```

donde `n_e ≈ n_i ≈ ρ/(2 m_p)` para plasma H+He completamente ionizado.

### Temperatura espectroscópica (Mazzotta+2004)

```text
T_sl = Σ w_i T_i   con w_i = n_e²_i T_i^{-0.75} / Σ w_j
```

Esta ponderación reproduce la temperatura que mediría un observador real integrando el espectro en la banda ROSAT/Chandra.

## API pública

| Función | Descripción |
|---------|-------------|
| `bremsstrahlung_emissivity(p, gamma)` | Emissividad de una partícula de gas |
| `total_xray_luminosity(particles, gamma)` | L_X integrada sobre todas las partículas |
| `spectroscopic_temperature(particles, gamma)` | T_X ponderada por emissividad |
| `mass_weighted_temperature(particles, gamma)` | T_X ponderada por masa |
| `compute_xray_profile(particles, center, r_edges, gamma)` | Perfil radial de L_X y T_X |

## Tests (6/6 OK)

1. `lx_positive_hot_gas` — L_X > 0 para gas caliente
2. `tx_positive_hot_gas` — T_X > 0 para gas caliente
3. `lx_zero_no_hot_gas` — L_X = 0 para gas frío
4. `xray_profile_has_luminosity` — perfil radial tiene luminosidad positiva
5. `tx_spectroscopic_differs_from_mass_weighted` — T_sl ≠ T_mass
6. `xray_integration_n50` — N=50 sin panics

## Referencias

- Sarazin (1988) "X-ray Emission from Clusters of Galaxies"
- Mazzotta et al. (2004) MNRAS 354, 10
