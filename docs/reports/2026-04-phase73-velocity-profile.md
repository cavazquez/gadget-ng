# Phase 73 — Perfiles de velocidad σ_v(r)

**Fecha**: Abril 2026  
**Crate**: `gadget-ng-analysis`  
**Módulo**: `src/velocity_profile.rs`

---

## Motivación

El perfil de dispersión de velocidad σ_v(r) es una de las principales validaciones de la
dinámica interna de halos. Comparado con el perfil NFW cinético permite:

- Verificar que el halo está en equilibrio virial.
- Medir la anisotropía orbital β(r) (órbitas radiales vs tangenciales).
- Comparar con datos observacionales (cúmulos de galaxias, grupos).

---

## Estadísticas Calculadas por Bin Radial

| Cantidad         | Definición                                           |
|------------------|------------------------------------------------------|
| `v_r_mean`       | Velocidad radial media ⟨v_r⟩(r)                    |
| `sigma_r`        | Dispersión radial σ_r = √(⟨v_r²⟩ − ⟨v_r⟩²)        |
| `sigma_t`        | Dispersión tangencial σ_t = √(⟨v_t²⟩/2)           |
| `sigma_3d`       | Dispersión total σ₃D = √(σ_r² + 2σ_t²)            |
| `n_part`         | Partículas en el bin                                 |

---

## API Pública

```rust
pub struct VelocityProfileBin {
    pub r: f64, pub r_lo: f64, pub r_hi: f64,
    pub v_r_mean: f64,
    pub sigma_r: f64, pub sigma_t: f64, pub sigma_3d: f64,
    pub n_part: usize,
}

pub struct VelocityProfileParams {
    pub n_bins: usize,
    pub r_min: f64, pub r_max: f64,
    pub log_bins: bool,
}

// Perfil completo
pub fn velocity_profile(
    positions: &[Vec3], velocities: &[Vec3], masses: &[f64],
    center: Vec3, v_center: Vec3,
    params: &VelocityProfileParams,
) -> Vec<VelocityProfileBin>

// Dispersión 1D (línea de visión)
pub fn sigma_1d(sigma_3d: f64) -> f64

// Parámetro de anisotropía de Binney β(r) = 1 − σ_t²/σ_r²
pub fn velocity_anisotropy(profile: &[VelocityProfileBin]) -> Vec<(f64, f64)>
```

---

## Parámetro de Anisotropía β

- **β = 0**: órbitas isótropas.
- **β = 1**: órbitas puramente radiales.
- **β < 0**: dominan órbitas tangenciales (ej. post-merger).

Los halos NFW puros tienen β ≈ 0.2–0.5 en las regiones externas.

---

## Tests

| Test                               | Verificación                                      |
|------------------------------------|---------------------------------------------------|
| `profile_bins_nonempty`            | Perfil no vacío para distribución esférica       |
| `profile_sigma_positive`           | σ_r, σ₃D ≥ 0 siempre                            |
| `profile_radial_ordering`          | Bins en orden creciente de r                     |
| `profile_isotropic_beta_finite`    | β finito y ≤ 1 para distribución isótropa       |
| `sigma_1d_relationship`            | σ₁D = σ₃D / √3 exactamente                      |
| `lin_bins_cover_range`             | Bordes lineales cubren [r_min, r_max]            |
| `log_bins_cover_range`             | Bordes logarítmicos cubren [r_min, r_max]        |
| `radial_only_particles_high_beta`  | Velocidad radialmente sesgada → β > 0            |

---

## Referencia

- Springel et al. (2001), MNRAS 328, 726.  
- NFW (1997), ApJ 490, 493.  
- Binney & Tremaine (2008), "Galactic Dynamics", §4.8.
