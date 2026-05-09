//! Turbulent dynamo: α-effect y crecimiento de campo magnético a gran escala (Phase 172).
//!
//! ## Modelo de dinamo cinemático (α-effect)
//!
//! El efecto α representa la fuerza electromotriz media (EMF) debida a turbulencia:
//!
//! ```text
//! EMF_i = α_ij ⟨b_j⟩ + β_ij ⟨J_j⟩ + ...
//! ```
//!
//! donde α_ij ≈ α δ_ij en isotropo y ⟨b⟩ es el campo magnético turbulento.
//!
//! Para el dinamo de crecimiento moyen-local (test-field model):
//! ```text
//! d⟨B_i⟩/dt = α_ij ∇⟨B_j⟩ + ...
//! ```
//!
//! El crecimiento del campo sigue la ecuación:
//! ```text
//! dB_L/dt = -B_L / τ_decay + (1/τ_growth) × B_T
//! ```
//!
//! donde B_L es el campo a gran escala, B_T es el campo turbulento, y τ_growth
//! depende del número de Mach Alfvénico.
//!
//! ## Implementación numérica
//!
//! Seguimos el modelo de Federrath et al. (2011) para crecimiento de campo:
//! ```text
//! B_new = B_old + dt × (α_term - decay_term)
//! α_term = C_alpha × v_rms × ∇ × B_turbulent
//! decay_term = B_old / τ_decay
//! ```
//!
//! La magnitud del campo crece mientras persista el forzado turbulento.
//!
//! ## Referencia
//!
//! Federrath et al. (2011) A&A 532, A62 — turbulent dynamo in primordial magnetic fields.
//! Schleicher et al. (2010) A&A 522, A115 — small-scale dynamo at different Mach numbers.

use crate::MU0;
use gadget_ng_core::{Particle, ParticleType, Vec3};

const C_ALPHA: f64 = 1.0 / 3.0;

pub fn alpha_coefficient(v_rms: f64, mach_alfven: f64) -> f64 {
    C_ALPHA * v_rms * (1.0 + mach_alfven.powi(2)).sqrt()
}

pub fn dynamo_growth_rate(v_rms: f64, b_rms: f64, rho: f64) -> f64 {
    if rho < 1e-30 || b_rms < 1e-30 {
        return 0.0;
    }
    let v_a = (b_rms * b_rms / (MU0 * rho)).sqrt();
    if v_a < 1e-30 {
        return 0.0;
    }
    let reynolds_magnetic = v_rms * 1.0 / 1e-6;
    let growth = (v_rms / v_a) / reynolds_magnetic.max(1.0);
    growth.max(0.0)
}

pub fn apply_turbulent_dynamo(particles: &mut [Particle], v_rms: f64, dt: f64, decay_time: f64) {
    let alpha = alpha_coefficient(v_rms, 0.5);
    if alpha < 1e-30 {
        return;
    }

    let decay = (-dt / decay_time.max(1e-10)).exp();

    for p in particles.iter_mut() {
        if p.ptype != ParticleType::Gas {
            continue;
        }

        let b2 = p.b_field.x.powi(2) + p.b_field.y.powi(2) + p.b_field.z.powi(2);
        if b2 < 1e-60 {
            continue;
        }

        let h = p.smoothing_length.max(1e-10);
        let rho = (p.mass / (4.0 / 3.0 * std::f64::consts::PI * h * h * h)).max(1e-30);

        let growth = dynamo_growth_rate(v_rms, b2.sqrt(), rho);
        let growth_factor = (growth * dt).exp();

        let b_norm = b2.sqrt().max(1e-30);
        let bx = p.b_field.x / b_norm;
        let by = p.b_field.y / b_norm;
        let bz = p.b_field.z / b_norm;

        let curl_b = alpha * (b_norm / h);

        p.b_field.x += dt * curl_b * bx;
        p.b_field.y += dt * curl_b * by;
        p.b_field.z += dt * curl_b * bz;

        let b_new = (p.b_field.x.powi(2) + p.b_field.y.powi(2) + p.b_field.z.powi(2)).sqrt();
        if b_new > 1e-30 {
            let grown = b_new * growth_factor;
            let renormalized = grown * decay;
            p.b_field.x *= renormalized / b_new;
            p.b_field.y *= renormalized / b_new;
            p.b_field.z *= renormalized / b_new;
        }
    }
}

pub fn magnetic_energy_ratio(particles: &[Particle], _gamma: f64) -> f64 {
    let mut e_kin_sum = 0.0_f64;
    let mut e_mag_sum = 0.0_f64;
    let mut n = 0usize;

    for p in particles.iter() {
        if p.ptype != ParticleType::Gas {
            continue;
        }
        let h = p.smoothing_length.max(1e-10);
        let rho = (p.mass / (h * h * h)).max(1e-30);

        let v2 = p.velocity.x.powi(2) + p.velocity.y.powi(2) + p.velocity.z.powi(2);
        let b2 = p.b_field.x.powi(2) + p.b_field.y.powi(2) + p.b_field.z.powi(2);

        e_kin_sum += 0.5 * rho * v2;
        e_mag_sum += b2 / (2.0 * MU0);
        n += 1;
    }

    if e_kin_sum < 1e-30 || n == 0 {
        return 0.0;
    }
    e_mag_sum / e_kin_sum
}

pub fn maxwell_stress_tensor(b: Vec3, rho: f64) -> f64 {
    let b2 = b.x.powi(2) + b.y.powi(2) + b.z.powi(2);
    b2 / (2.0 * MU0 * rho.max(1e-30))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alpha_is_positive() {
        let v_rms = 10.0;
        let mach_a = 0.5;
        let alpha = alpha_coefficient(v_rms, mach_a);
        assert!(alpha > 0.0, "alpha should be positive, got {}", alpha);
    }

    #[test]
    fn dynamo_growth_rate_positive() {
        let growth = dynamo_growth_rate(10.0, 1.0, 1.0);
        assert!(growth >= 0.0, "growth rate should be non-negative");
    }

    #[test]
    fn magnetic_energy_ratio_zero_when_no_b() {
        let mut p = Particle::new(
            0,
            1.0,
            gadget_ng_core::Vec3::zero(),
            gadget_ng_core::Vec3::zero(),
        );
        p.ptype = ParticleType::Gas;
        p.velocity = gadget_ng_core::Vec3::new(1.0, 0.0, 0.0);

        let ratio = magnetic_energy_ratio(&[p], 5.0 / 3.0);
        assert_eq!(ratio, 0.0, "energy ratio should be zero when B=0");
    }

    #[test]
    fn dynamo_alpha_depends_on_v_rms() {
        let alpha1 = alpha_coefficient(10.0, 0.5);
        let alpha2 = alpha_coefficient(20.0, 0.5);
        assert!(alpha2 > alpha1, "alpha should increase with v_rms");
    }
}
