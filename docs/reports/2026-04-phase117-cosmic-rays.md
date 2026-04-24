# Phase 117 — Rayos cósmicos básicos

**Fecha**: 23 de abril de 2026  
**Capa**: 2C — Física avanzada (Jubelgas et al. 2008)

## Resumen

Implementación del módulo de rayos cósmicos (CRs) básico: inyección desde supernovas,
difusión isótropa entre partículas SPH vecinas, y contribución a la presión total.

## Cambios técnicos

### `crates/gadget-ng-core/src/particle.rs`
- Campo `cr_energy: f64` con `#[serde(default)]` — energía CR específica [(km/s)²].

### `crates/gadget-ng-core/src/config.rs`
Nueva struct `CrSection` en `SphSection`:
- `enabled: bool` (default: `false`)
- `cr_fraction: f64` (default: `0.1`) — fracción de E_SN → CRs
- `kappa_cr: f64` (default: `3e-3`) — difusividad isótropa

### `crates/gadget-ng-sph/src/cosmic_rays.rs` (nuevo)

```rust
pub fn inject_cr_from_sn(particles: &mut [Particle], sfr: &[f64], cr_fraction: f64, dt: f64)
// Δe_cr = cr_fraction × E_SN × sfr × dt  para cada gas con sfr > 0

pub fn diffuse_cr(particles: &mut [Particle], kappa_cr: f64, dt: f64)
// Δe_cr,i = κ_CR × Σ_j (e_cr,j - e_cr,i) × w(r_ij, h_i) × dt

pub fn cr_pressure(cr_energy: f64, rho: f64) -> f64
// P_cr = (γ_cr - 1) × ρ × e_cr,  γ_cr = 4/3 (relativista)
```

## Modelo físico

Los CRs son acelerados en frentes de choque de SN II. En un modelo simple:
- Fracción ε_CR ≈ 10% de E_SN va a CRs.
- Difusión isótropa: `∂e_cr/∂t = ∇·(κ_CR ∇e_cr)` — discretizada con kernel SPH.
- Presión CR con índice relativista γ_CR = 4/3: `P_CR = (1/3) ρ e_cr`.

## Tests

6 tests en `phase117_cosmic_rays.rs` — todos pasan ✅:

1. `injection_increases_cr_energy` — inyección CR desde SN aumenta cr_energy
2. `zero_sfr_no_cr_injection` — SFR=0 sin inyección
3. `dm_not_injected_with_cr` — DM no recibe CRs
4. `diffusion_equalizes_cr_energy` — difusión iguala energía entre vecinos
5. `cr_pressure_formula` — P_cr = (1/3) ρ e_cr correcto
6. `cr_section_serde` — serialización de CrSection

## Referencias

- Jubelgas et al. (2008) A&A 481, 33 — CRs en SPH cosmológico
- Pfrommer et al. (2017) MNRAS 465, 4500 — transporte CR
- Guo & Oh (2008) MNRAS 384, 251 — presión CR en halos
