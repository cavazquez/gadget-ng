//! Gas molecular HI → H₂ (Phase 122).
//!
//! Placeholder — implementación completa en Phase 122.

use crate::dust::dust_h2_shielding_factor;
use gadget_ng_core::{DustSection, MolecularSection, Particle, ParticleType};
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

    #[cfg(feature = "rayon")]
    {
        particles
            .par_iter_mut()
            .for_each(|p| update_h2_particle(p, cfg, dust, dt, t_dissoc));
    }

    #[cfg(not(feature = "rayon"))]
    {
        update_h2_serial(particles, cfg, dust, dt, t_dissoc);
    }
}

#[cfg(not(feature = "rayon"))]
fn update_h2_serial(
    particles: &mut [Particle],
    cfg: &MolecularSection,
    dust: Option<&DustSection>,
    dt: f64,
    t_dissoc: f64,
) {
    if dust.is_some() {
        return update_h2_scalar(particles, cfg, dust, dt, t_dissoc);
    }

    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    {
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx512f") {
            // SAFETY: AVX-512F availability was checked at runtime.
            unsafe {
                return update_h2_avx512(particles, cfg, dt, t_dissoc);
            }
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: AVX2+FMA availability was checked at runtime.
            unsafe {
                return update_h2_avx2(particles, cfg, dt, t_dissoc);
            }
        }
    }

    update_h2_scalar(particles, cfg, dust, dt, t_dissoc);
}

#[cfg(not(feature = "rayon"))]
fn update_h2_scalar(
    particles: &mut [Particle],
    cfg: &MolecularSection,
    dust: Option<&DustSection>,
    dt: f64,
    t_dissoc: f64,
) {
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

#[cfg(all(
    not(feature = "rayon"),
    feature = "simd",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn update_h2_avx2(
    particles: &mut [Particle],
    cfg: &MolecularSection,
    dt: f64,
    t_dissoc: f64,
) {
    let lanes = 4;
    let chunks = particles.len() / lanes * lanes;
    let threshold_v = _mm256_set1_pd(cfg.rho_h2_threshold);
    let tau_v = _mm256_set1_pd((dt / t_dissoc).min(1.0));
    let zero_v = _mm256_set1_pd(0.0);
    let one_v = _mm256_set1_pd(1.0);
    let min_h_v = _mm256_set1_pd(1e-10);
    let decay = (-dt / t_dissoc).exp();
    let decay_v = _mm256_set1_pd(decay);

    let mut i = 0;
    while i < chunks {
        let all_gas = particles[i..i + lanes]
            .iter()
            .all(|p| p.ptype == ParticleType::Gas);
        if !all_gas {
            update_h2_scalar(&mut particles[i..i + lanes], cfg, None, dt, t_dissoc);
            i += lanes;
            continue;
        }
        let h = _mm256_max_pd(
            min_h_v,
            _mm256_set_pd(
                particles[i + 3].smoothing_length,
                particles[i + 2].smoothing_length,
                particles[i + 1].smoothing_length,
                particles[i].smoothing_length,
            ),
        );
        let mass = _mm256_set_pd(
            particles[i + 3].mass,
            particles[i + 2].mass,
            particles[i + 1].mass,
            particles[i].mass,
        );
        let rho = _mm256_div_pd(mass, _mm256_mul_pd(h, _mm256_mul_pd(h, h)));
        let old = _mm256_set_pd(
            particles[i + 3].h2_fraction,
            particles[i + 2].h2_fraction,
            particles[i + 1].h2_fraction,
            particles[i].h2_fraction,
        );
        let h2_eq = _mm256_min_pd(one_v, _mm256_div_pd(rho, threshold_v));
        let grow = _mm256_fmadd_pd(tau_v, _mm256_sub_pd(h2_eq, old), old);
        let decay_new = _mm256_mul_pd(old, decay_v);
        let dense_mask = _mm256_cmp_pd(rho, threshold_v, _CMP_GT_OQ);
        let new_h2 = _mm256_min_pd(
            one_v,
            _mm256_max_pd(zero_v, _mm256_blendv_pd(decay_new, grow, dense_mask)),
        );
        let mut out = [0.0; 4];
        // SAFETY: fixed-size stack array has exactly four f64 lanes.
        unsafe { _mm256_storeu_pd(out.as_mut_ptr(), new_h2) };
        for lane in 0..lanes {
            particles[i + lane].h2_fraction = out[lane];
        }
        i += lanes;
    }
    update_h2_scalar(&mut particles[chunks..], cfg, None, dt, t_dissoc);
}

#[cfg(all(
    not(feature = "rayon"),
    feature = "simd",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
unsafe fn update_h2_avx512(
    particles: &mut [Particle],
    cfg: &MolecularSection,
    dt: f64,
    t_dissoc: f64,
) {
    // SAFETY: the caller checked AVX-512F; AVX2 arithmetic preserves the same update semantics.
    unsafe { update_h2_avx2(particles, cfg, dt, t_dissoc) }
}
