//! Estadísticas del campo magnético para monitoreo cosmológico (Phase 136).

use gadget_ng_core::{Particle, ParticleType};

/// Estadísticas del campo magnético sobre todas las partículas de gas.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BFieldStats {
    /// Magnitud media de B: `<|B|>` en masa.
    pub b_mean: f64,
    /// Magnitud RMS de B: `sqrt(<|B|²>)` en masa.
    pub b_rms: f64,
    /// Máximo de `|B|` entre todas las partículas de gas.
    pub b_max: f64,
    /// Energía magnética total: `Σ m_i |B_i|² / (2 μ₀)`.
    pub e_mag: f64,
    /// Número de partículas de gas incluidas en el cálculo.
    pub n_gas: usize,
}

/// Calcula estadísticas del campo magnético sobre el slice de partículas (Phase 136).
///
/// Retorna `None` si no hay partículas de gas.
pub fn b_field_stats(particles: &[Particle]) -> Option<BFieldStats> {
    use crate::MU0;

    let mut m_total = 0.0_f64;
    let mut mb_sum = 0.0_f64;   // Σ m_i |B_i|
    let mut mb2_sum = 0.0_f64;  // Σ m_i |B_i|²
    let mut b_max = 0.0_f64;
    let mut e_mag = 0.0_f64;
    let mut n_gas = 0usize;

    for p in particles.iter() {
        if p.ptype != ParticleType::Gas { continue; }
        let b2 = p.b_field.x*p.b_field.x + p.b_field.y*p.b_field.y + p.b_field.z*p.b_field.z;
        let b_mag = b2.sqrt();

        m_total += p.mass;
        mb_sum += p.mass * b_mag;
        mb2_sum += p.mass * b2;
        b_max = b_max.max(b_mag);
        e_mag += p.mass * b2 / (2.0 * MU0);
        n_gas += 1;
    }

    if n_gas == 0 || m_total <= 0.0 { return None; }

    Some(BFieldStats {
        b_mean: mb_sum / m_total,
        b_rms: (mb2_sum / m_total).sqrt(),
        b_max,
        e_mag,
        n_gas,
    })
}
