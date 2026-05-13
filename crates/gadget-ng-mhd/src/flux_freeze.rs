//! Freeze-out del campo magnético en gas difuso de alta β-plasma (Phase 138).
//!
//! ## Modelo
//!
//! En plasma con β >> 1 (presión térmica domina sobre magnética), el campo B se "congela"
//! con el fluido (teorema de Alfvén: conservación de flujo magnético en plasma ideal).
//!
//! Al comprimir el gas, el flujo magnético se conserva:
//! `Φ = B × A = cte` → `B ∝ ρ^{2/3}` (en 3D, compresión isótropa).
//!
//! La función `apply_flux_freeze` corrige B en partículas con β > β_freeze
//! para mantener `B ∝ ρ^{2/3}` respecto a una densidad de referencia `ρ_ref`.
//!
//! ## Referencias
//!
//! Alfvén (1942) — conservación de flujo magnético.
//! Subramanian & Barrow (1998), PhysRevD 58 — amplificación de B en bariones.

use crate::MU0;
use gadget_ng_core::{Particle, ParticleType};
#[cfg(feature = "rayon")]
use rayon::prelude::*;

/// Aplica el criterio de flux-freeze a partículas de gas con β > beta_freeze (Phase 138).
///
/// Para cada partícula de gas:
/// 1. Calcula β = 2μ₀ P_th / |B|².
/// 2. Si β > beta_freeze: escala `B → B × (ρ/ρ_ref)^{2/3}` (compresión isótropa).
/// 3. Si β ≤ beta_freeze: el campo B es dinámicamente importante → no se aplica.
///
/// `rho_ref` es la densidad de referencia respecto a la cual se calcula la amplificación.
/// En la práctica suele ser la densidad inicial o la densidad media del halo.
pub fn apply_flux_freeze(particles: &mut [Particle], gamma: f64, beta_freeze: f64, rho_ref: f64) {
    #[cfg(feature = "rayon")]
    {
        particles
            .par_iter_mut()
            .for_each(|p| apply_flux_freeze_particle(p, gamma, beta_freeze, rho_ref));
    }

    #[cfg(not(feature = "rayon"))]
    for p in particles.iter_mut() {
        apply_flux_freeze_particle(p, gamma, beta_freeze, rho_ref);
    }
}

fn apply_flux_freeze_particle(p: &mut Particle, gamma: f64, beta_freeze: f64, rho_ref: f64) {
    if p.ptype != ParticleType::Gas {
        return;
    }

    let b2 = p.b_field.x * p.b_field.x + p.b_field.y * p.b_field.y + p.b_field.z * p.b_field.z;
    if b2 < 1e-60 {
        return;
    } // B=0: nada que congelar

    let h = p.smoothing_length.max(1e-10);
    let rho = (p.mass / (h * h * h)).max(1e-30);
    let p_th = (gamma - 1.0) * rho * p.internal_energy;
    let beta = 2.0 * MU0 * p_th / b2;

    if beta > beta_freeze && rho_ref > 0.0 {
        // Conservación de flujo: B ∝ ρ^{2/3}
        let scale = (rho / rho_ref).powf(2.0 / 3.0);
        p.b_field.x *= scale;
        p.b_field.y *= scale;
        p.b_field.z *= scale;
    }
}

/// Calcula la densidad media de las partículas de gas (densidad de referencia).
pub fn mean_gas_density(particles: &[Particle]) -> f64 {
    #[cfg(feature = "rayon")]
    {
        let (rho_sum, n) = particles
            .par_iter()
            .filter(|p| p.ptype == ParticleType::Gas)
            .map(|p| {
                let h = p.smoothing_length.max(1e-10);
                (p.mass / (h * h * h), 1usize)
            })
            .reduce(|| (0.0_f64, 0usize), |a, b| (a.0 + b.0, a.1 + b.1));
        if n == 0 { 1.0 } else { rho_sum / n as f64 }
    }

    #[cfg(not(feature = "rayon"))]
    {
        let mut rho_sum = 0.0_f64;
        let mut n = 0usize;
        for p in particles.iter() {
            if p.ptype != ParticleType::Gas {
                continue;
            }
            let h = p.smoothing_length.max(1e-10);
            rho_sum += p.mass / (h * h * h);
            n += 1;
        }
        if n == 0 { 1.0 } else { rho_sum / n as f64 }
    }
}

/// Valida que B ∝ ρ^{2/3} para una partícula dada respecto a valores de referencia.
///
/// Retorna el error relativo |B_actual / B_expected - 1|.
/// `b0` y `rho0` son los valores de referencia (estado inicial o densdiad del halo).
pub fn flux_freeze_error(b_actual: f64, b0: f64, rho: f64, rho0: f64) -> f64 {
    if b0 < 1e-30 || rho0 < 1e-30 {
        return 0.0;
    }
    let b_expected = b0 * (rho / rho0).powf(2.0 / 3.0);
    (b_actual / b_expected - 1.0).abs()
}

#[cfg(test)]
mod tests {
    use super::*;

    use approx::assert_abs_diff_eq;

    #[test]
    fn flux_freeze_error_at_reference_is_zero() {
        let b0 = 2.0;
        let rho0 = 5.0;
        assert_abs_diff_eq!(flux_freeze_error(b0, b0, rho0, rho0), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn flux_freeze_error_zero_when_bzero() {
        assert_abs_diff_eq!(
            flux_freeze_error(0.0, 0.0, 100.0, 100.0),
            0.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn flux_freeze_error_doubled_density() {
        let b0: f64 = 1.0;
        let rho0: f64 = 1.0;
        let rho: f64 = 8.0 * rho0;
        let b_expected = b0 * (rho / rho0).powf(2.0 / 3.0);
        assert_abs_diff_eq!(
            flux_freeze_error(b_expected, b0, rho, rho0),
            0.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn flux_freeze_error_known_mismatch() {
        let err = flux_freeze_error(1.0, 1.0, 8.0, 1.0);
        assert_abs_diff_eq!(err, 0.75, epsilon = 1e-12);
    }
}
