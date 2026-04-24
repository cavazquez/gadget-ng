# Phase 148 — RMHD Cosmológica: Jets AGN Relativistas desde Halos FoF

**Fecha:** 2026-04-23  
**Autor:** gadget-ng automated development  
**Estado:** ✅ Completado

## Objetivo

Extender el módulo SRMHD con la capacidad de inyectar jets AGN relativistas bipolares desde los centros de halos FoF identificados in-situ, completando así el ciclo de retroalimentación AGN relativista.

## Física

### Modelo de jet bipolar

Para cada halo FoF seleccionado:
1. Identificar partícula de gas más cercana con z > z_centro (jet +z) y z < z_centro (jet -z)
2. Asignar velocidad: `v = ±v_jet ẑ`
3. Asignar campo B: `B = ±B_jet ẑ` (alineado con jet)
4. Energía interna: `u_jet = (γ − 1) c²` donde `γ = lorentz_factor(v_jet)`

### Factor de Lorentz

```
γ = 1 / sqrt(1 − |v|²/c²)
```

Para `v_jet = 0.9c`: `γ ≈ 2.294`, `E_jet = 1.294 × mc²`

## Archivos

### `crates/gadget-ng-mhd/src/relativistic.rs`

Nueva función `inject_relativistic_jet`:

```rust
pub fn inject_relativistic_jet(
    particles: &mut [Particle],
    halo_centers: &[Vec3],
    v_jet_frac: f64,   // v_jet en fracciones de c
    n_jet_halos: usize,
    c_light: f64,
    b_jet: f64,
)
```

### `crates/gadget-ng-core/src/config.rs`

- `MhdSection.jet_enabled: bool` (default `false`)
- `MhdSection.v_jet: f64` (default `0.3`)
- `MhdSection.n_jet_halos: usize` (default `1`)

## Tests

6 tests en `phase148_rmhd_jets.rs`:

| Test | Verificación | Estado |
|------|-------------|--------|
| `jet_injects_velocity` | v_z = ±v_jet en partículas más cercanas | ✅ |
| `jet_aligns_b_with_velocity` | B_z > 0 (jet +), B_z < 0 (jet -) | ✅ |
| `jet_energy_relativistic` | u ≥ (γ−1)c² con γ(0.9c) | ✅ |
| `zero_halos_no_jet` | n_jet=0 → no-op | ✅ |
| `zero_v_jet_no_injection` | v_jet=0 → no-op | ✅ |
| `lorentz_factor_jet_consistent` | γ(0.9c) ≈ 2.294 | ✅ |

**Resultado:** 6/6 tests pasan ✅

## Relevancia astrofísica

Los jets AGN relativistas son mecanismos de feedback clave en cúmulos masivos:
- Frenan el enfriamiento del gas en halos masivos (cooling flow problem)
- Amplifican el campo magnético del ICM mediante expansión de lóbulos
- Inyectan energía cinética y magnética a escalas de Mpc

## Referencias

- McNamara & Nulsen (2007), ARA&A 45, 117 — jets AGN en ICM
- Guo & Mathews (2011), ApJ 728, 121 — simulaciones de jets relativistas
- Sijacki et al. (2007), MNRAS 380, 877 — jets AGN en simulaciones cosmológicas
