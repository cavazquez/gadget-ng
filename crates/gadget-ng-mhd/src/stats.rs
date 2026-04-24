//! Estadísticas del campo magnético para monitoreo cosmológico (Phase 136 + 147).

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

/// Espectro de potencia magnético P_B(k) estimado por histograma de |B|² (Phase 147).
///
/// Asigna cada partícula a un bin de `k ∝ 2π/h_i` (inverso del smoothing length),
/// donde `h_i` se usa como escala local. Devuelve `(k_center, P_B)` para cada bin.
///
/// # Parámetros
///
/// - `particles`: slice de partículas de gas con campo magnético
/// - `box_size`: longitud de la caja [unidades del código]
/// - `n_bins`: número de bins logarítmicos en k
///
/// # Retorno
///
/// Vector de `(k [1/unidades], P_B(k) [B² × volumen])` para cada bin.
/// Bins vacíos se omiten del resultado.
pub fn magnetic_power_spectrum(
    particles: &[Particle],
    box_size: f64,
    n_bins: usize,
) -> Vec<(f64, f64)> {
    if n_bins == 0 || box_size <= 0.0 { return Vec::new(); }

    // Rango de k: de k_min = 2π/L hasta k_max = 2π/h_min
    let k_fund = 2.0 * std::f64::consts::PI / box_size;

    let mut k_vals: Vec<f64> = Vec::new();
    let mut b2_vals: Vec<f64> = Vec::new();

    for p in particles.iter() {
        if p.ptype != ParticleType::Gas { continue; }
        let h = p.smoothing_length;
        if h <= 0.0 { continue; }
        let b2 = p.b_field.x*p.b_field.x + p.b_field.y*p.b_field.y + p.b_field.z*p.b_field.z;
        let k_p = 2.0 * std::f64::consts::PI / h;
        k_vals.push(k_p);
        b2_vals.push(b2 * p.mass);
    }

    if k_vals.is_empty() { return Vec::new(); }

    let k_min = k_fund.min(k_vals.iter().cloned().fold(f64::INFINITY, f64::min));
    let k_max = k_vals.iter().cloned().fold(0.0_f64, f64::max);
    if k_max <= k_min { return Vec::new(); }

    let log_k_min = k_min.ln();
    let log_k_max = k_max.ln();
    let dlog_k = (log_k_max - log_k_min) / n_bins as f64;
    if dlog_k <= 0.0 { return Vec::new(); }

    let mut bin_power = vec![0.0_f64; n_bins];
    let mut bin_count = vec![0usize; n_bins];

    for (&k, &b2m) in k_vals.iter().zip(b2_vals.iter()) {
        let i = ((k.ln() - log_k_min) / dlog_k) as usize;
        let i = i.min(n_bins - 1);
        bin_power[i] += b2m;
        bin_count[i] += 1;
    }

    let mut result = Vec::new();
    for i in 0..n_bins {
        if bin_count[i] == 0 { continue; }
        let k_center = (log_k_min + (i as f64 + 0.5) * dlog_k).exp();
        result.push((k_center, bin_power[i]));
    }
    result
}
