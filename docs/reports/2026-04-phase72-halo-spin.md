# Phase 72 — Spin de halos λ (Peebles)

**Fecha**: Abril 2026  
**Crate**: `gadget-ng-analysis`  
**Módulo**: `src/halo_spin.rs`

---

## Motivación

El parámetro de spin λ cuantifica el momento angular específico de un halo en unidades de
lo que se necesitaría para soportarse por rotación. Es una predicción robusta de los modelos
de formación galáctica: galaxias de disco se forman en halos con λ ≳ 0.03, mientras que
esferoídales se asocian con halos de bajo spin.

---

## Definiciones

**Peebles (1971)**:

```
λ = |L| / (M × V_vir × R_vir)
```

**Bullock et al. (2001)**:

```
λ' = |L| / (√2 × M × V_vir × R_vir) = λ / √2
```

donde:
- `L = Σᵢ mᵢ (rᵢ − r_com) × (vᵢ − v_com)` — momento angular total.
- `R_vir = R₂₀₀ = [3M / (4π × 200 × ρ_crit)]^(1/3)`.
- `V_vir = sqrt(G × M / R₂₀₀)`.

---

## API Pública

```rust
pub struct HaloSpin {
    pub mass: f64,
    pub r200: f64,
    pub pos_com: [f64; 3],
    pub vel_com: [f64; 3],
    pub angular_momentum: [f64; 3],
    pub l_mag: f64,
    pub lambda_peebles: f64,
    pub lambda_bullock: f64,
}

pub struct SpinParams {
    pub g_newton: f64,   // 4.302e-3 kpc (km/s)² / M_sun
    pub delta_vir: f64,  // 200 (defecto)
    pub rho_crit: f64,   // 2.775e11 M_sun/Mpc³ para h=1
}

// Spin de un halo individual
pub fn halo_spin(
    positions: &[Vec3], velocities: &[Vec3], masses: &[f64],
    params: &SpinParams,
) -> Option<HaloSpin>

// Spin de múltiples halos dados por índices de partículas
pub fn compute_halo_spins(
    positions: &[Vec3], velocities: &[Vec3], masses: &[f64],
    halo_ids: &[Vec<usize>], params: &SpinParams,
) -> Vec<Option<HaloSpin>>
```

---

## Tests

| Test                                  | Verificación                                    |
|---------------------------------------|-------------------------------------------------|
| `spin_ring_positive`                  | Anillo circular → λ > 0, λ' > 0                |
| `spin_static_halo_zero`               | Sin velocidades → L = 0, λ = 0                 |
| `spin_empty_returns_none`             | Sin partículas → None                          |
| `lambda_bullock_smaller_than_peebles` | λ' = λ / √2 exactamente                        |
| `center_of_mass_symmetric`            | Distribución simétrica → COM en origen         |
| `spin_angular_momentum_direction`     | Rotación en XY → L apunta en +Z               |
| `compute_halo_spins_multi`            | Batch de 2 halos funciona correctamente        |

---

## Referencia

- Peebles (1971), A&A 11, 377.  
- Bullock et al. (2001), ApJ 555, 240.  
- Mo, Mao & White (1998), MNRAS 295, 319.
