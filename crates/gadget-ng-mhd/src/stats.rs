//! Estadísticas del campo magnético para monitoreo cosmológico (Phase 136 + 147).

use gadget_ng_core::{Particle, ParticleType};
#[cfg(feature = "rayon")]
use rayon::prelude::*;
#[cfg(all(
    not(feature = "rayon"),
    feature = "simd",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(all(
    not(feature = "rayon"),
    feature = "simd",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

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
    #[cfg(feature = "rayon")]
    use crate::MU0;

    #[cfg(feature = "rayon")]
    {
        let (m_total, mb_sum, mb2_sum, b_max, e_mag, n_gas) = particles
            .par_iter()
            .filter(|p| p.ptype == ParticleType::Gas)
            .map(|p| {
                let b2 = p.b_field.x * p.b_field.x
                    + p.b_field.y * p.b_field.y
                    + p.b_field.z * p.b_field.z;
                let b_mag = b2.sqrt();
                (
                    p.mass,
                    p.mass * b_mag,
                    p.mass * b2,
                    b_mag,
                    p.mass * b2 / (2.0 * MU0),
                    1usize,
                )
            })
            .reduce(
                || (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0usize),
                |a, b| {
                    (
                        a.0 + b.0,
                        a.1 + b.1,
                        a.2 + b.2,
                        a.3.max(b.3),
                        a.4 + b.4,
                        a.5 + b.5,
                    )
                },
            );

        if n_gas == 0 || m_total <= 0.0 {
            return None;
        }

        Some(BFieldStats {
            b_mean: mb_sum / m_total,
            b_rms: (mb2_sum / m_total).sqrt(),
            b_max,
            e_mag,
            n_gas,
        })
    }

    #[cfg(not(feature = "rayon"))]
    {
        #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
        {
            #[cfg(target_arch = "x86_64")]
            if is_x86_feature_detected!("avx512f") {
                // SAFETY: AVX-512F availability was checked at runtime.
                unsafe {
                    return b_field_stats_avx512(particles);
                }
            }
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                // SAFETY: AVX2+FMA availability was checked at runtime.
                unsafe {
                    return b_field_stats_avx2(particles);
                }
            }
        }
        b_field_stats_scalar(particles)
    }
}

#[cfg(not(feature = "rayon"))]
fn b_field_stats_scalar(particles: &[Particle]) -> Option<BFieldStats> {
    use crate::MU0;

    let mut m_total = 0.0_f64;
    let mut mb_sum = 0.0_f64; // Σ m_i |B_i|
    let mut mb2_sum = 0.0_f64; // Σ m_i |B_i|²
    let mut b_max = 0.0_f64;
    let mut e_mag = 0.0_f64;
    let mut n_gas = 0usize;

    for p in particles.iter() {
        if p.ptype != ParticleType::Gas {
            continue;
        }
        let b2 = p.b_field.x * p.b_field.x + p.b_field.y * p.b_field.y + p.b_field.z * p.b_field.z;
        let b_mag = b2.sqrt();

        m_total += p.mass;
        mb_sum += p.mass * b_mag;
        mb2_sum += p.mass * b2;
        b_max = b_max.max(b_mag);
        e_mag += p.mass * b2 / (2.0 * MU0);
        n_gas += 1;
    }

    if n_gas == 0 || m_total <= 0.0 {
        return None;
    }

    Some(BFieldStats {
        b_mean: mb_sum / m_total,
        b_rms: (mb2_sum / m_total).sqrt(),
        b_max,
        e_mag,
        n_gas,
    })
}

#[cfg(all(
    not(feature = "rayon"),
    feature = "simd",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn b_field_stats_avx2(particles: &[Particle]) -> Option<BFieldStats> {
    use crate::MU0;

    let lanes = 4;
    let chunks = particles.len() / lanes * lanes;
    let mut m_total_v = _mm256_setzero_pd();
    let mut mb_sum_v = _mm256_setzero_pd();
    let mut mb2_sum_v = _mm256_setzero_pd();
    let mut b_max_v = _mm256_setzero_pd();
    let mut n_gas = 0usize;
    let mut i = 0;
    while i < chunks {
        let mask_arr = [
            if particles[i].ptype == ParticleType::Gas {
                -1_i64
            } else {
                0
            },
            if particles[i + 1].ptype == ParticleType::Gas {
                -1_i64
            } else {
                0
            },
            if particles[i + 2].ptype == ParticleType::Gas {
                -1_i64
            } else {
                0
            },
            if particles[i + 3].ptype == ParticleType::Gas {
                -1_i64
            } else {
                0
            },
        ];
        n_gas += mask_arr.iter().filter(|&&m| m != 0).count();
        // SAFETY: fixed-size stack array has exactly four i64 lanes.
        let mask = unsafe { _mm256_castsi256_pd(_mm256_loadu_si256(mask_arr.as_ptr().cast())) };
        let m = _mm256_set_pd(
            particles[i + 3].mass,
            particles[i + 2].mass,
            particles[i + 1].mass,
            particles[i].mass,
        );
        let bx = _mm256_set_pd(
            particles[i + 3].b_field.x,
            particles[i + 2].b_field.x,
            particles[i + 1].b_field.x,
            particles[i].b_field.x,
        );
        let by = _mm256_set_pd(
            particles[i + 3].b_field.y,
            particles[i + 2].b_field.y,
            particles[i + 1].b_field.y,
            particles[i].b_field.y,
        );
        let bz = _mm256_set_pd(
            particles[i + 3].b_field.z,
            particles[i + 2].b_field.z,
            particles[i + 1].b_field.z,
            particles[i].b_field.z,
        );
        let b2 = _mm256_fmadd_pd(bx, bx, _mm256_fmadd_pd(by, by, _mm256_mul_pd(bz, bz)));
        let b = _mm256_sqrt_pd(b2);
        let m_masked = _mm256_and_pd(m, mask);
        let b_masked = _mm256_and_pd(b, mask);
        m_total_v = _mm256_add_pd(m_total_v, m_masked);
        mb_sum_v = _mm256_fmadd_pd(m_masked, b, mb_sum_v);
        mb2_sum_v = _mm256_fmadd_pd(m_masked, b2, mb2_sum_v);
        b_max_v = _mm256_max_pd(b_max_v, b_masked);
        i += lanes;
    }
    let mut mt = [0.0; 4];
    let mut mb = [0.0; 4];
    let mut mb2 = [0.0; 4];
    let mut bm = [0.0; 4];
    // SAFETY: fixed-size stack arrays have exactly four f64 lanes.
    unsafe {
        _mm256_storeu_pd(mt.as_mut_ptr(), m_total_v);
        _mm256_storeu_pd(mb.as_mut_ptr(), mb_sum_v);
        _mm256_storeu_pd(mb2.as_mut_ptr(), mb2_sum_v);
        _mm256_storeu_pd(bm.as_mut_ptr(), b_max_v);
    }
    let mut m_total = mt.into_iter().sum::<f64>();
    let mut mb_sum = mb.into_iter().sum::<f64>();
    let mut mb2_sum = mb2.into_iter().sum::<f64>();
    let mut b_max = bm.into_iter().fold(0.0_f64, f64::max);
    for p in &particles[chunks..] {
        if p.ptype != ParticleType::Gas {
            continue;
        }
        let b2 = p.b_field.x * p.b_field.x + p.b_field.y * p.b_field.y + p.b_field.z * p.b_field.z;
        let b = b2.sqrt();
        m_total += p.mass;
        mb_sum += p.mass * b;
        mb2_sum += p.mass * b2;
        b_max = b_max.max(b);
        n_gas += 1;
    }
    if n_gas == 0 || m_total <= 0.0 {
        return None;
    }
    Some(BFieldStats {
        b_mean: mb_sum / m_total,
        b_rms: (mb2_sum / m_total).sqrt(),
        b_max,
        e_mag: mb2_sum / (2.0 * MU0),
        n_gas,
    })
}

#[cfg(all(
    not(feature = "rayon"),
    feature = "simd",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
unsafe fn b_field_stats_avx512(particles: &[Particle]) -> Option<BFieldStats> {
    use crate::MU0;
    let lanes = 8;
    let chunks = particles.len() / lanes * lanes;
    let mut m_total_v = _mm512_setzero_pd();
    let mut mb_sum_v = _mm512_setzero_pd();
    let mut mb2_sum_v = _mm512_setzero_pd();
    let mut b_max_v = _mm512_setzero_pd();
    let mut n_gas = 0usize;
    let mut i = 0;
    while i < chunks {
        let mut mask_bits: u8 = 0;
        for lane in 0..lanes {
            if particles[i + lane].ptype == ParticleType::Gas {
                mask_bits |= 1 << lane;
            }
        }
        n_gas += mask_bits.count_ones() as usize;
        let m = _mm512_set_pd(
            particles[i + 7].mass,
            particles[i + 6].mass,
            particles[i + 5].mass,
            particles[i + 4].mass,
            particles[i + 3].mass,
            particles[i + 2].mass,
            particles[i + 1].mass,
            particles[i].mass,
        );
        let bx = _mm512_set_pd(
            particles[i + 7].b_field.x,
            particles[i + 6].b_field.x,
            particles[i + 5].b_field.x,
            particles[i + 4].b_field.x,
            particles[i + 3].b_field.x,
            particles[i + 2].b_field.x,
            particles[i + 1].b_field.x,
            particles[i].b_field.x,
        );
        let by = _mm512_set_pd(
            particles[i + 7].b_field.y,
            particles[i + 6].b_field.y,
            particles[i + 5].b_field.y,
            particles[i + 4].b_field.y,
            particles[i + 3].b_field.y,
            particles[i + 2].b_field.y,
            particles[i + 1].b_field.y,
            particles[i].b_field.y,
        );
        let bz = _mm512_set_pd(
            particles[i + 7].b_field.z,
            particles[i + 6].b_field.z,
            particles[i + 5].b_field.z,
            particles[i + 4].b_field.z,
            particles[i + 3].b_field.z,
            particles[i + 2].b_field.z,
            particles[i + 1].b_field.z,
            particles[i].b_field.z,
        );
        let b2 = _mm512_fmadd_pd(bx, bx, _mm512_fmadd_pd(by, by, _mm512_mul_pd(bz, bz)));
        let b = _mm512_sqrt_pd(b2);
        let m_masked = _mm512_maskz_add_pd(mask_bits, m, _mm512_setzero_pd());
        let b_masked = _mm512_maskz_add_pd(mask_bits, b, _mm512_setzero_pd());
        m_total_v = _mm512_add_pd(m_total_v, m_masked);
        mb_sum_v = _mm512_fmadd_pd(m_masked, b, mb_sum_v);
        mb2_sum_v = _mm512_fmadd_pd(m_masked, b2, mb2_sum_v);
        b_max_v = _mm512_max_pd(b_max_v, b_masked);
        i += lanes;
    }
    let mut mt = [0.0; 8];
    let mut mb = [0.0; 8];
    let mut mb2 = [0.0; 8];
    let mut bm = [0.0; 8];
    // SAFETY: fixed-size stack arrays have exactly eight f64 lanes.
    unsafe {
        _mm512_storeu_pd(mt.as_mut_ptr(), m_total_v);
        _mm512_storeu_pd(mb.as_mut_ptr(), mb_sum_v);
        _mm512_storeu_pd(mb2.as_mut_ptr(), mb2_sum_v);
        _mm512_storeu_pd(bm.as_mut_ptr(), b_max_v);
    }
    let mut m_total = mt.into_iter().sum::<f64>();
    let mut mb_sum = mb.into_iter().sum::<f64>();
    let mut mb2_sum = mb2.into_iter().sum::<f64>();
    let mut b_max = bm.into_iter().fold(0.0_f64, f64::max);
    for p in &particles[chunks..] {
        if p.ptype != ParticleType::Gas {
            continue;
        }
        let b2 = p.b_field.x * p.b_field.x + p.b_field.y * p.b_field.y + p.b_field.z * p.b_field.z;
        let b = b2.sqrt();
        m_total += p.mass;
        mb_sum += p.mass * b;
        mb2_sum += p.mass * b2;
        b_max = b_max.max(b);
        n_gas += 1;
    }
    if n_gas == 0 || m_total <= 0.0 {
        return None;
    }
    Some(BFieldStats {
        b_mean: mb_sum / m_total,
        b_rms: (mb2_sum / m_total).sqrt(),
        b_max,
        e_mag: mb2_sum / (2.0 * MU0),
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
#[cfg(not(feature = "rayon"))]
fn magnetic_power_spectrum_impl(
    particles: &[Particle],
    box_size: f64,
    n_bins: usize,
) -> Vec<(f64, f64)> {
    if n_bins == 0 || box_size <= 0.0 {
        return Vec::new();
    }

    let k_fund = 2.0 * std::f64::consts::PI / box_size;

    let mut k_vals: Vec<f64> = Vec::new();
    let mut b2_vals: Vec<f64> = Vec::new();

    for p in particles.iter() {
        if p.ptype != ParticleType::Gas {
            continue;
        }
        let h = p.smoothing_length;
        if h <= 0.0 {
            continue;
        }
        let b2 = p.b_field.x * p.b_field.x + p.b_field.y * p.b_field.y + p.b_field.z * p.b_field.z;
        let k_p = 2.0 * std::f64::consts::PI / h;
        k_vals.push(k_p);
        b2_vals.push(b2 * p.mass);
    }

    if k_vals.is_empty() {
        return Vec::new();
    }

    let k_min = k_fund.min(k_vals.iter().cloned().fold(f64::INFINITY, f64::min));
    let k_max = k_vals.iter().cloned().fold(0.0_f64, f64::max);
    if k_max <= k_min {
        return Vec::new();
    }

    let log_k_min = k_min.ln();
    let log_k_max = k_max.ln();
    let dlog_k = (log_k_max - log_k_min) / n_bins as f64;
    if dlog_k <= 0.0 {
        return Vec::new();
    }

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
        if bin_count[i] == 0 {
            continue;
        }
        let k_center = (log_k_min + (i as f64 + 0.5) * dlog_k).exp();
        result.push((k_center, bin_power[i]));
    }
    result
}

#[cfg(feature = "rayon")]
fn magnetic_power_spectrum_par(
    particles: &[Particle],
    box_size: f64,
    n_bins: usize,
) -> Vec<(f64, f64)> {
    if n_bins == 0 || box_size <= 0.0 {
        return Vec::new();
    }

    let k_fund = 2.0 * std::f64::consts::PI / box_size;

    let pairs: Vec<(f64, f64)> = particles
        .par_iter()
        .filter(|p| p.ptype == ParticleType::Gas && p.smoothing_length > 0.0)
        .map(|p| {
            let b2 =
                p.b_field.x * p.b_field.x + p.b_field.y * p.b_field.y + p.b_field.z * p.b_field.z;
            let k_p = 2.0 * std::f64::consts::PI / p.smoothing_length;
            (k_p, b2 * p.mass)
        })
        .collect();

    if pairs.is_empty() {
        return Vec::new();
    }

    let k_min = k_fund.min(pairs.iter().map(|(k, _)| *k).fold(f64::INFINITY, f64::min));
    let k_max = pairs.iter().map(|(k, _)| *k).fold(0.0_f64, f64::max);
    if k_max <= k_min {
        return Vec::new();
    }

    let log_k_min = k_min.ln();
    let log_k_max = k_max.ln();
    let dlog_k = (log_k_max - log_k_min) / n_bins as f64;
    if dlog_k <= 0.0 {
        return Vec::new();
    }

    let mut bin_power = vec![0.0_f64; n_bins];
    let mut bin_count = vec![0usize; n_bins];

    for (k, b2m) in &pairs {
        let i = ((k.ln() - log_k_min) / dlog_k) as usize;
        let i = i.min(n_bins - 1);
        bin_power[i] += b2m;
        bin_count[i] += 1;
    }

    let mut result = Vec::new();
    for i in 0..n_bins {
        if bin_count[i] == 0 {
            continue;
        }
        let k_center = (log_k_min + (i as f64 + 0.5) * dlog_k).exp();
        result.push((k_center, bin_power[i]));
    }
    result
}

pub fn magnetic_power_spectrum(
    particles: &[Particle],
    box_size: f64,
    n_bins: usize,
) -> Vec<(f64, f64)> {
    #[cfg(feature = "rayon")]
    {
        magnetic_power_spectrum_par(particles, box_size, n_bins)
    }

    #[cfg(not(feature = "rayon"))]
    {
        magnetic_power_spectrum_impl(particles, box_size, n_bins)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use approx::assert_abs_diff_eq;
    use gadget_ng_core::{Particle, Vec3};

    fn make_gas_particle(b: Vec3, mass: f64) -> Particle {
        let mut p = Particle::new_gas(0, mass, Vec3::zero(), Vec3::zero(), 0.0, 1.0);
        p.b_field = b;
        p
    }

    #[test]
    fn b_field_stats_none_for_empty_slice() {
        assert_eq!(b_field_stats(&[]), None);
    }

    #[test]
    fn b_field_stats_single_gas_particle() {
        let particles = vec![make_gas_particle(Vec3::new(3.0, 0.0, 4.0), 1.0)];
        let stats = b_field_stats(&particles).unwrap();
        assert_abs_diff_eq!(stats.b_mean, 5.0, epsilon = 1e-12);
        assert_abs_diff_eq!(stats.b_rms, 5.0, epsilon = 1e-12);
        assert_abs_diff_eq!(stats.b_max, 5.0, epsilon = 1e-12);
    }

    #[test]
    fn b_field_stats_ignores_dm() {
        let gas = make_gas_particle(Vec3::new(3.0, 0.0, 4.0), 1.0);
        let dm = Particle::new(1, 2.0, Vec3::zero(), Vec3::zero());
        let particles = vec![gas, dm];
        let stats = b_field_stats(&particles).unwrap();
        assert_abs_diff_eq!(stats.b_mean, 5.0, epsilon = 1e-12);
        assert_eq!(stats.n_gas, 1);
    }

    #[test]
    fn b_field_stats_b_rms_gte_b_mean() {
        let particles = vec![
            make_gas_particle(Vec3::new(3.0, 0.0, 4.0), 1.0),
            make_gas_particle(Vec3::new(1.0, 1.0, 1.0), 2.0),
            make_gas_particle(Vec3::new(0.0, 5.0, 0.0), 1.5),
        ];
        let stats = b_field_stats(&particles).unwrap();
        assert!(stats.b_rms >= stats.b_mean);
    }
}
