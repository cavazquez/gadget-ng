# Phase 86 — Química de no-equilibrio HII/HeII/HeIII

**Fecha**: 2026-04-23  
**Crate**: `gadget-ng-rt`  
**Archivo nuevo**: `crates/gadget-ng-rt/src/chemistry.rs`  
**Tipo**: Nueva física

---

## Resumen

Red de química de no-equilibrio de 6 especies (HI, HII, HeI, HeII, HeIII, e⁻)
acoplada al campo de radiación M1 y al gas SPH. Implementa el solver implícito
subcíclico de Anninos et al. (1997) con tasas de Verner & Ferland (1996) y Cen (1992).

---

## Red de química

```
dx_hii/dt  = Γ_HI × x_hi  + β_HI(T) × x_hi × x_e  - α_HII(T) × x_hii × x_e
dx_heii/dt = Γ_HeI × x_hei + β_HeI(T) × x_hei × x_e - (α_HeII + β_HeII)(T) × x_heii × x_e
                                                       + α_HeIII(T) × x_heiii × x_e
dx_heiii/dt= β_HeII(T) × x_heii × x_e - α_HeIII(T) × x_heiii × x_e
x_hi      = 1 - x_hii                   (conservación H)
x_hei     = f_He - x_heii - x_heiii    (conservación He)
x_e       = x_hii + x_heii + 2·x_heiii (neutralidad de carga)
```

donde `f_He = 0.0789` (fracción He por número, Planck 2018).

---

## Tasas de reacción

### Recombinación (Verner & Ferland 1996)

| Función | Descripción |
|---------|-------------|
| `alpha_hii(T)` | Case-B HII: fit `2.753e-14 × (315614/T)^1.5 / (1 + ...)^2.242` |
| `alpha_heii(T)` | Case-B HeII: `1.26e-14 × (470000/T)^0.75` |
| `alpha_heiii(T)` | Análoga a HII escalada ×4 |

### Ionización colisional (Cen 1992)

| Función | Descripción |
|---------|-------------|
| `beta_hi(T)` | `5.85e-11 × √T × exp(-T_HI/T) / (1 + √(T/10⁵))` |
| `beta_hei(T)` | `2.38e-11 × √T × exp(-T_HeI/T) / (1 + √(T/10⁵))` |
| `beta_heii(T)` | `5.68e-12 × √T × exp(-T_HeII/T) / (1 + √(T/10⁵))` |

---

## Solver implícito subcíclico

```rust
pub fn solve_chemistry_implicit(
    state: &ChemState,
    gamma_hi: f64,   // tasa fotoionización HI [1/s]
    gamma_hei: f64,  // tasa fotoionización HeI [1/s]
    t: f64,          // temperatura [K]
    dt: f64,         // paso de tiempo [código]
) -> ChemState
```

**Algoritmo**:
1. Calcular escala de tiempo química: `dt_chem = min(0.1 / |rate|, dt)`
2. Actualización implícita linealizada: `x_new = (x_old + dt × I) / (1 + dt × R)`
3. `clamp_and_normalize()` para conservar H, He y neutralidad de carga
4. Iterar hasta `t_elapsed = dt` o convergencia

---

## Acoplamiento a gas SPH

```rust
pub fn apply_chemistry(
    particles: &mut [Particle],
    chem_states: &mut [ChemState],
    rad: &RadiationField,
    params: &ChemParams,
    dt: f64,
)
```

Para cada partícula de gas:
1. Temperatura desde `internal_energy` via μ adaptativo
2. Tasa Γ_HI desde campo M1 en la celda más cercana
3. `solve_chemistry_implicit` actualiza fracciones de ionización
4. Enfriamiento aproximado (bremsstrahlung + Lyα) ajusta `internal_energy`

### Temperatura desde energía interna

```
T = u × (γ-1) × μ × m_p / k_B
μ = 4 / (3 + x_hii + x_heii + 2·x_heiii + 3·x_hei)   [Osterbrock 2006]
```

### Enfriamiento aproximado

```rust
pub fn cooling_rate_approx(t: f64, x_e: f64, n_h: f64) -> f64
// Bremsstrahlung: 1.42e-27 × √T × n_H² × x_e
// Lyα: 7.5e-19 × exp(-118348/T) × n_H² × x_e
```

---

## Tests (13 tests)

| Test | Verifica |
|------|----------|
| `neutral_state_conserves_h` | x_hi + x_hii = 1 en estado neutro |
| `fully_ionized_conserves_charge` | x_e = x_hii + x_heii + 2·x_heiii |
| `recombination_rates_positive` | α > 0 para T ∈ {1e3, 1e4, 1e5, 1e6} K |
| `ionization_rates_positive` | β > 0 para T ∈ {1e4, 1e5, 1e6} K |
| `alpha_hii_decreases_with_temperature` | Recombinación se enfría a T alta |
| `beta_hi_increases_with_temperature` | Ionización aumenta con T (T < T_HI) |
| `solve_chemistry_neutral_no_photons_stays_neutral` | Sin UV, T=1e3 K → x_hi > 0.9 |
| `solve_chemistry_ionized_recombines` | Gas ionizado recombina con tiempo |
| `solve_chemistry_fractions_conserved` | H y He conservados tras evolución |
| `high_uv_field_ionizes_hydrogen` | UV fuerte → x_hii > 0.5 |
| `temperature_from_internal_energy_reasonable` | T > 10 K y T < 1e9 K |
| `cooling_rate_positive` | Λ_cool ≥ 0 |
| `clamp_normalize_prevents_negative` | Fracciones no negativas tras clamp |

Todos los 33 tests del crate `gadget-ng-rt` pasan ✅.

---

## Referencia

Anninos et al. (1997), New Astron. 2, 209;  
Cen (1992), ApJS 78, 341;  
Verner & Ferland (1996), ApJS 103, 467;  
Osterbrock (2006), Astrophysics of Gaseous Nebulae.
