//! Gas molecular HI → H₂ (Phase 122).
//!
//! Placeholder — implementación completa en Phase 122.

use crate::dust::dust_h2_shielding_factor;
use gadget_ng_core::{DustSection, MolecularSection, Particle, ParticleType};
#[cfg(feature = "simd")]
use rayon::prelude::*;

/// Actualiza la fracción de gas molecular H₂ en cada partícula de gas (Phase 122).
pub fn update_h2_fraction(particles: &mut [Particle], cfg: &MolecularSection, dt: f64) {
    update_h2_fraction_with_dust(particles, cfg, None, dt);
}

/// Actualiza H2 incluyendo shielding dinámico por polvo activo (Phase 192).
pub fn update_h2_fraction_with_dust(
    particles: &mut [Particle],
    cfg: &MolecularSection,
    dust: Option<&DustSection>,
    dt: f64,
) {
    if !cfg.enabled {
        return;
    }
    let t_dissoc = 10.0; // tiempo de fotodisociación en unidades internas

    #[cfg(feature = "simd")]
    {
        particles
            .par_iter_mut()
            .for_each(|p| update_h2_particle(p, cfg, dust, dt, t_dissoc));
    }

    #[cfg(not(feature = "simd"))]
    for p in particles.iter_mut() {
        update_h2_particle(p, cfg, dust, dt, t_dissoc);
    }
}

fn update_h2_particle(
    p: &mut Particle,
    cfg: &MolecularSection,
    dust: Option<&DustSection>,
    dt: f64,
    t_dissoc: f64,
) {
    if p.ptype != ParticleType::Gas {
        return;
    }
    // Aproximación: densidad ∝ masa / h³ (SPH estándar)
    let h = p.smoothing_length.max(1e-10);
    let rho = p.mass / (h * h * h);
    let shielding = dust.map_or(1.0, |d| dust_h2_shielding_factor(p, d));
    if rho > cfg.rho_h2_threshold {
        let h2_eq = ((rho / cfg.rho_h2_threshold) * shielding).min(1.0);
        let tau = (dt / t_dissoc).min(1.0);
        p.h2_fraction = p.h2_fraction + tau * (h2_eq - p.h2_fraction);
    } else {
        let t_dissoc_eff = t_dissoc * shielding.max(1.0);
        p.h2_fraction *= (-dt / t_dissoc_eff).exp();
    }
    p.h2_fraction = p.h2_fraction.clamp(0.0, 1.0);
}
