//! MHD no ideal: difusión ambipolar dependiente de ionización (Phase 194).
//!
//! Modelo reducido: en gas poco ionizado, las partículas neutras desacoplan el
//! campo magnético del fluido. Representamos esto como una difusión local que
//! amortigua `B` con una tasa proporcional a `eta_ad / x_i`, suavizada por un
//! proxy de ionización térmica y por el contenido de polvo.

use gadget_ng_core::{Particle, ParticleType};

/// Proxy acotado de fracción ionizada local.
///
/// - `internal_energy` alto implica gas más ionizado.
/// - `dust_to_gas` alto reduce la ionización efectiva por recombinación/shielding.
pub fn ionization_fraction_proxy(p: &Particle, ion_floor: f64, dust_coupling: f64) -> f64 {
    if p.ptype != ParticleType::Gas {
        return 1.0;
    }
    let thermal = p.internal_energy.max(0.0);
    let collisional = thermal / (thermal + 1.0);
    let dust_suppression = 1.0 / (1.0 + dust_coupling.max(0.0) * p.dust_to_gas.max(0.0) * 100.0);
    (collisional * dust_suppression).clamp(ion_floor.max(1e-12), 1.0)
}

/// Aplica difusión ambipolar local sobre el campo magnético.
///
/// El amortiguamiento es `B(t+dt)=B(t) exp[-eta_ad dt (1/x_i - 1)]`, de modo que
/// gas completamente ionizado (`x_i≈1`) casi no cambia, mientras que gas neutro
/// difunde más rápido. La energía magnética disipada se deposita como calor.
pub fn apply_ambipolar_diffusion(
    particles: &mut [Particle],
    eta_ad: f64,
    ion_floor: f64,
    dust_coupling: f64,
    gamma: f64,
    dt: f64,
) {
    if eta_ad <= 0.0 || dt <= 0.0 {
        return;
    }
    let heat_eff = (gamma - 1.0).max(0.0);
    for p in particles.iter_mut() {
        if p.ptype != ParticleType::Gas {
            continue;
        }
        let b2_before = p.b_field.dot(p.b_field);
        if b2_before <= 0.0 {
            continue;
        }
        let x_i = ionization_fraction_proxy(p, ion_floor, dust_coupling);
        let rate = eta_ad.max(0.0) * (1.0 / x_i - 1.0).max(0.0);
        let damping = (-rate * dt).exp().clamp(0.0, 1.0);
        p.b_field *= damping;
        let b2_after = p.b_field.dot(p.b_field);
        let dissipated = 0.5 * (b2_before - b2_after).max(0.0);
        p.internal_energy += heat_eff * dissipated / p.mass.max(1e-30);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gadget_ng_core::{Particle, Vec3};

    fn gas(u: f64, dust: f64) -> Particle {
        let mut p = Particle::new_gas(0, 1.0, Vec3::zero(), Vec3::zero(), u, 0.1);
        p.b_field = Vec3::new(1.0, 0.0, 0.0);
        p.dust_to_gas = dust;
        p
    }

    #[test]
    fn ionization_proxy_is_lower_with_dust() {
        let clean = gas(1.0, 0.0);
        let dusty = gas(1.0, 0.02);
        assert!(
            ionization_fraction_proxy(&dusty, 1e-4, 1.0)
                < ionization_fraction_proxy(&clean, 1e-4, 1.0)
        );
    }

    #[test]
    fn ambipolar_diffusion_reduces_b_and_heats() {
        let mut particles = vec![gas(0.01, 0.02)];
        let b0 = particles[0].b_field.norm();
        let u0 = particles[0].internal_energy;
        apply_ambipolar_diffusion(&mut particles, 0.1, 1e-4, 1.0, 5.0 / 3.0, 1.0);
        assert!(particles[0].b_field.norm() < b0);
        assert!(particles[0].internal_energy > u0);
    }
}
