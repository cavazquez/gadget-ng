# Phase 133 — MHD Anisótropo: Difusión ∥B

**Fecha:** 2026-04-23  
**Estado:** Completado ✓  
**Tests:** 6/6 passing

## Objetivo

Implementar conducción térmica y difusión CR anisótropa a lo largo de las líneas de campo magnético,
con tensor de difusión `D = κ_∥(B̂⊗B̂) + κ_⊥(I−B̂⊗B̂)`.

## Implementación

### Archivo nuevo: `crates/gadget-ng-mhd/src/anisotropic.rs`

- **`apply_anisotropic_conduction(particles, kappa_par, kappa_perp, gamma, dt)`**: Conducción
  térmica anisótropa SPH. La conductividad efectiva en la dirección r̂_ij es:
  `κ_eff(θ) = κ_⊥ + (κ_∥ − κ_⊥) cos²(θ)`.
  
- **`diffuse_cr_anisotropic(particles, kappa_cr, b_suppress, dt)`**: Difusión CR con flujo
  solo paralelo a B, incluyendo supresión por |B|² (Phase 129).

- **`beta_plasma(p_thermal, b) → f64`**: Calcula β = 2μ₀ P_th / |B|².

### Modificaciones

- `crates/gadget-ng-core/src/config.rs`: Nuevos campos `anisotropic: bool`, `kappa_par: f64`,
  `kappa_perp: f64` en `ConductionSection`.
- `crates/gadget-ng-mhd/src/lib.rs`: Re-exportación de las nuevas funciones.
- `crates/gadget-ng-cli/src/engine.rs`: Hook en `maybe_sph!` — si `conduction.anisotropic = true`
  usa `apply_anisotropic_conduction` en lugar del Spitzer isótropo.

## Física

### Modelo de Braginskii

En un plasma magnetizado, el transporte de calor y partículas cargadas está controlado por la
giración de electrones alrededor de las líneas de campo B. La conductividad paralela (∥B) es
del orden de la libre distancia media, mientras que la perpendicular (⊥B) está suprimida por
el radio de Larmor: `κ_⊥/κ_∥ ~ (ω_ci τ_i)^{-2} << 1`.

### Límites

- `κ_⊥ = 0, κ_∥ > 0`: difusión puramente paralela (límite fuerte campo B)
- `κ_∥ = κ_⊥ = κ`: recupera la conducción Spitzer isótropa

## Tests

| Test | Descripción |
|------|-------------|
| `heat_flows_parallel_to_b` | Calor fluye entre partículas alineadas con B (∥) |
| `no_heat_perpendicular_to_b` | Sin transferencia ⊥B con κ_⊥=0 |
| `isotropic_limit_kpar_eq_kperp` | Con κ_∥=κ_⊥ → conducción isótropa |
| `beta_plasma_formula` | β = 2μ₀P/|B|² correcto |
| `beta_infinite_with_zero_b` | β → ∞ con B=0 |
| `energy_conserved_anisotropic` | Conservación de energía en pares |

## Referencias

- Braginskii (1965), Rev. Plasma Phys. 1, 205
- Parrish & Stone (2005), ApJ 633, 334
- Sharma & Hammett (2007), J. Comput. Phys. 227, 123
