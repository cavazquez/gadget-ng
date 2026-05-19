//! Rayos cósmicos básicos: inyección desde SN y difusión SPH (Phase 117).
//!
//! ## Modelo
//!
//! Los rayos cósmicos (CRs) son partículas relativistas (principalmente protones)
//! aceleradas en frentes de choque de supernovas. El módulo implementa:
//!
//! 1. **Inyección**: una fracción `cr_fraction` de la energía de SN II se deposita
//!    como energía CR en las partículas de gas con SFR activa.
//!
//! 2. **Difusión**: difusión isótropa simplificada usando SPH kernel:
//!    `Δe_cr,i = κ_CR × Σ_j (e_cr,j - e_cr,i) × w(r_ij, h_i) × dt`
//!
//! 3. **Presión de CRs**: contribución a la presión total con γ_CR = 4/3.
//!
//! ## Referencia
//!
//! Jubelgas et al. (2008) A&A 481, 33 — CRs en SPH cosmológico.
//! Pfrommer et al. (2017) MNRAS 465, 4500 — transporte de CRs.

use crate::periodic_delta;
use gadget_ng_core::{Particle, ParticleType};
#[cfg(feature = "rayon")]
use rayon::prelude::*;

/// Constante de energía de SN en unidades internas [(km/s)² por 10¹⁰ M_sun].
const E_SN_CODE: f64 = 1.54e-3;

/// Kernel SPH simple (Wendland C2) para difusión.
#[inline]
fn kernel_w_cr(r: f64, h: f64) -> f64 {
    if h <= 0.0 {
        return 0.0;
    }
    let q = r / h;
    if q > 2.0 {
        return 0.0;
    }
    let t = 1.0 - 0.5 * q;
    (21.0 / (2.0 * std::f64::consts::PI)) / (h * h * h) * t.powi(4) * (1.0 + 2.0 * q)
}

/// Presión de rayos cósmicos para γ_CR = 4/3 (relativista) (Phase 117).
///
/// `P_cr = (γ_cr - 1) × ρ × e_cr`
#[inline]
pub fn cr_pressure(cr_energy: f64, rho: f64) -> f64 {
    const GAMMA_CR: f64 = 4.0 / 3.0;
    (GAMMA_CR - 1.0) * rho * cr_energy.max(0.0)
}

/// Inyecta energía CR desde eventos de SN II (Phase 117).
///
/// Para cada partícula de gas con `sfr[i] > 0`:
/// `Δe_cr,i = cr_fraction × E_SN × sfr[i] × dt`
///
/// La energía CR se suma directamente al campo `cr_energy` de la partícula.
///
/// # Parámetros
///
/// - `particles`: slice mutable de partículas
/// - `sfr`: tasa de formación estelar por partícula
/// - `cr_fraction`: fracción de E_SN → CRs
/// - `dt`: paso de tiempo
pub fn inject_cr_from_sn(particles: &mut [Particle], sfr: &[f64], cr_fraction: f64, dt: f64) {
    assert_eq!(particles.len(), sfr.len());

    #[cfg(feature = "rayon")]
    {
        particles.par_iter_mut().enumerate().for_each(|(i, p)| {
            if p.ptype != ParticleType::Gas {
                return;
            }
            if sfr[i] <= 0.0 {
                return;
            }
            let delta_cr = cr_fraction * E_SN_CODE * sfr[i] * dt;
            p.cr_energy += delta_cr;
        });
    }

    #[cfg(not(feature = "rayon"))]
    for i in 0..particles.len() {
        if particles[i].ptype != ParticleType::Gas {
            continue;
        }
        if sfr[i] <= 0.0 {
            continue;
        }
        let delta_cr = cr_fraction * E_SN_CODE * sfr[i] * dt;
        particles[i].cr_energy += delta_cr;
    }
}

/// Difusión isótropa de CRs con supresión magnética opcional (Phase 117 + 129).
///
/// Implementa: `Δe_cr,i = κ_CR_eff × Σ_j (e_cr,j - e_cr,i) × w(r_ij, h_i) × dt`
///
/// donde `κ_CR_eff = κ_CR / (1 + b_suppress × |B_i|²)` (Phase 129).
/// Con `b_suppress = 0.0` se recupera el comportamiento clásico isótropo.
///
/// # Parámetros
///
/// - `particles`: slice mutable de partículas
/// - `kappa_cr`: coeficiente de difusión [unidades internas]
/// - `b_suppress`: factor de supresión por campo B (Phase 129); `0.0` = sin supresión
/// - `dt`: paso de tiempo
#[allow(clippy::needless_range_loop)]
pub fn diffuse_cr(particles: &mut [Particle], kappa_cr: f64, b_suppress: f64, dt: f64) {
    diffuse_cr_periodic(particles, kappa_cr, b_suppress, dt, None);
}

/// Igual que `diffuse_cr`, usando imagen mínima si `periodic_box = Some(L)`.
#[allow(clippy::needless_range_loop)]
pub fn diffuse_cr_periodic(
    particles: &mut [Particle],
    kappa_cr: f64,
    b_suppress: f64,
    dt: f64,
    periodic_box: Option<f64>,
) {
    let n = particles.len();
    if n == 0 {
        return;
    }

    #[cfg(feature = "rayon")]
    {
        let ptypes: Vec<ParticleType> = particles.iter().map(|p| p.ptype).collect();
        let pos: Vec<_> = particles.iter().map(|p| p.position).collect();
        let h: Vec<f64> = particles
            .iter()
            .map(|p| p.smoothing_length.max(1e-10))
            .collect();
        let e_cr: Vec<f64> = particles.iter().map(|p| p.cr_energy).collect();
        let b2: Vec<f64> = particles
            .iter()
            .map(|p| p.b_field.x.powi(2) + p.b_field.y.powi(2) + p.b_field.z.powi(2))
            .collect();

        let delta_cr: Vec<f64> = (0..n)
            .into_par_iter()
            .map(|i| {
                if ptypes[i] != ParticleType::Gas {
                    return 0.0;
                }
                let kappa_eff = kappa_cr / (1.0 + b_suppress * b2[i]);
                let mut sum = 0.0_f64;
                for j in 0..n {
                    if i == j || ptypes[j] != ParticleType::Gas {
                        continue;
                    }
                    let r = periodic_delta(pos[i], pos[j], periodic_box).norm();
                    let w = kernel_w_cr(r, 2.0 * h[i]);
                    if w > 0.0 {
                        sum += kappa_eff * (e_cr[j] - e_cr[i]) * w * dt;
                    }
                }
                sum
            })
            .collect();

        for i in 0..n {
            if particles[i].ptype == ParticleType::Gas {
                particles[i].cr_energy = (particles[i].cr_energy + delta_cr[i]).max(0.0);
            }
        }
    }

    #[cfg(not(feature = "rayon"))]
    {
        let mut delta_cr = vec![0.0_f64; n];

        for i in 0..n {
            if particles[i].ptype != ParticleType::Gas {
                continue;
            }
            let h_i = particles[i].smoothing_length.max(1e-10);
            let pos_i = particles[i].position;
            let e_i = particles[i].cr_energy;

            // Phase 129: κ efectiva modulada por |B|²
            let b2_i = particles[i].b_field.x.powi(2)
                + particles[i].b_field.y.powi(2)
                + particles[i].b_field.z.powi(2);
            let kappa_eff = kappa_cr / (1.0 + b_suppress * b2_i);

            for j in 0..n {
                if i == j {
                    continue;
                }
                if particles[j].ptype != ParticleType::Gas {
                    continue;
                }

                let r = periodic_delta(pos_i, particles[j].position, periodic_box).norm();

                let w = kernel_w_cr(r, 2.0 * h_i);
                if w > 0.0 {
                    delta_cr[i] += kappa_eff * (particles[j].cr_energy - e_i) * w * dt;
                }
            }
        }

        // Aplicar incrementos, asegurar no-negatividad
        for i in 0..n {
            if particles[i].ptype == ParticleType::Gas {
                particles[i].cr_energy = (particles[i].cr_energy + delta_cr[i]).max(0.0);
            }
        }
    }
}

/// Pérdidas hadrónicas aproximadas: `e_cr ← e_cr × exp(−k × ρ × dt)`.
///
/// `coeff` agrega secciones eficaces y normalización en un solo parámetro
/// [unidades internas]; `0` desactiva. La densidad se estima como `m / (4πh³/3)`.
pub fn apply_cr_hadronic_losses(particles: &mut [Particle], coeff: f64, dt: f64) {
    if coeff <= 0.0 {
        return;
    }

    #[cfg(feature = "rayon")]
    {
        const PI: f64 = std::f64::consts::PI;
        particles.par_iter_mut().for_each(|p| {
            if p.ptype != ParticleType::Gas {
                return;
            }
            let h = p.smoothing_length.max(1e-30);
            let rho = p.mass / ((4.0 / 3.0) * PI * h * h * h).max(1e-100);
            let x = (coeff * rho * dt).min(80.0);
            p.cr_energy *= (-x).exp();
        });
    }

    #[cfg(not(feature = "rayon"))]
    {
        const PI: f64 = std::f64::consts::PI;
        for p in particles.iter_mut() {
            if p.ptype != ParticleType::Gas {
                continue;
            }
            let h = p.smoothing_length.max(1e-30);
            let rho = p.mass / ((4.0 / 3.0) * PI * h * h * h).max(1e-100);
            let x = (coeff * rho * dt).min(80.0);
            p.cr_energy *= (-x).exp();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use gadget_ng_core::Vec3;

    #[test]
    fn cr_pressure_scales_with_energy_and_density() {
        let p_lo = cr_pressure(0.1, 2.0);
        let p_hi = cr_pressure(0.2, 2.0);
        assert!(p_hi > p_lo);
        assert_abs_diff_eq!(p_hi / p_lo, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn inject_cr_from_sn_adds_energy() {
        let mut particles = vec![Particle::new_gas(0, 1.0, Vec3::zero(), Vec3::zero(), 1.0, 0.1)];
        inject_cr_from_sn(&mut particles, &[1.0], 0.1, 0.01);
        assert!(particles[0].cr_energy > 0.0);
    }

    #[test]
    fn apply_cr_hadronic_losses_reduces_cr_energy() {
        let mut particles = vec![Particle::new_gas(0, 1.0, Vec3::zero(), Vec3::zero(), 1.0, 0.1)];
        particles[0].cr_energy = 1.0;
        apply_cr_hadronic_losses(&mut particles, 1.0, 0.1);
        assert!(particles[0].cr_energy < 1.0);
    }

    #[test]
    fn diffuse_cr_redistributes_energy_between_neighbors() {
        let hot = Particle::new_gas(0, 1.0, Vec3::zero(), Vec3::zero(), 1.0, 0.5);
        let mut cold = Particle::new_gas(1, 1.0, Vec3::new(0.05, 0.0, 0.0), Vec3::zero(), 1.0, 0.5);
        cold.cr_energy = 0.0;
        let mut particles = vec![hot, cold];
        particles[0].cr_energy = 10.0;
        diffuse_cr(&mut particles, 1.0, 0.0, 0.01);
        assert!(particles[0].cr_energy < 10.0);
        assert!(particles[1].cr_energy > 0.0);
    }
}
