//! Gas molecular HI → H₂ (Phase 122).
//!
//! Placeholder — implementación completa en Phase 122.

use gadget_ng_core::{MolecularSection, Particle, ParticleType};

/// Actualiza la fracción de gas molecular H₂ en cada partícula de gas (Phase 122).
pub fn update_h2_fraction(particles: &mut [Particle], cfg: &MolecularSection, dt: f64) {
    if !cfg.enabled {
        return;
    }
    let t_dissoc = 10.0; // tiempo de fotodisociación en unidades internas
    for p in particles.iter_mut() {
        if p.ptype != ParticleType::Gas {
            continue;
        }
        // Aproximación: densidad ∝ masa / h³ (SPH estándar)
        let h = p.smoothing_length.max(1e-10);
        let rho = p.mass / (h * h * h);
        if rho > cfg.rho_h2_threshold {
            let h2_eq = (rho / cfg.rho_h2_threshold).min(1.0);
            let tau = (dt / t_dissoc).min(1.0);
            p.h2_fraction = p.h2_fraction + tau * (h2_eq - p.h2_fraction);
        } else {
            p.h2_fraction *= (-dt / t_dissoc).exp();
        }
        p.h2_fraction = p.h2_fraction.clamp(0.0, 1.0);
    }
}
