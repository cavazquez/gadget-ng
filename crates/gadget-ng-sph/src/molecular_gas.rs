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
    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    {
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx512f") {
            // SAFETY: AVX-512F availability was checked at runtime.
            unsafe {
                return update_h2_avx512(particles, cfg, dust, dt, t_dissoc);
            }
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: AVX2+FMA availability was checked at runtime.
            unsafe {
                return update_h2_avx2(particles, cfg, dust, dt, t_dissoc);
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
    dust: Option<&DustSection>,
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
    let dust_enabled = dust.is_some_and(|d| d.enabled);
    let (kappa_dust, shielding_boost) = dust.map_or((0.0, 0.0), |d| {
        (
            crate::dust::effective_dust_uv_opacity(d),
            d.h2_shielding_boost.max(0.0),
        )
    });
    let kappa_v = _mm256_set1_pd(kappa_dust.max(0.0));
    let boost_v = _mm256_set1_pd(shielding_boost);
    let min_h_shield_v = _mm256_set1_pd(1e-30);
    let one_hundred_v = _mm256_set1_pd(1.0);
    let neg_dt_v = _mm256_set1_pd(-dt);

    let mut i = 0;
    while i < chunks {
        let all_gas = particles[i..i + lanes]
            .iter()
            .all(|p| p.ptype == ParticleType::Gas);
        if !all_gas {
            update_h2_scalar(&mut particles[i..i + lanes], cfg, dust, dt, t_dissoc);
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
        let shielding = if dust_enabled {
            let h_shield = _mm256_max_pd(
                min_h_shield_v,
                _mm256_set_pd(
                    particles[i + 3].smoothing_length,
                    particles[i + 2].smoothing_length,
                    particles[i + 1].smoothing_length,
                    particles[i].smoothing_length,
                ),
            );
            let dust_to_gas = _mm256_set_pd(
                particles[i + 3].dust_to_gas,
                particles[i + 2].dust_to_gas,
                particles[i + 1].dust_to_gas,
                particles[i].dust_to_gas,
            );
            let active = _mm256_cmp_pd(dust_to_gas, zero_v, _CMP_GT_OQ);
            let rho_shield = _mm256_div_pd(
                mass,
                _mm256_mul_pd(h_shield, _mm256_mul_pd(h_shield, h_shield)),
            );
            let tau = _mm256_max_pd(
                zero_v,
                _mm256_mul_pd(
                    kappa_v,
                    _mm256_mul_pd(dust_to_gas, _mm256_mul_pd(rho_shield, h_shield)),
                ),
            );
            let mut tau_arr = [0.0; 4];
            // SAFETY: fixed-size stack array has exactly four f64 lanes.
            unsafe { _mm256_storeu_pd(tau_arr.as_mut_ptr(), tau) };
            let exp_neg_tau = _mm256_set_pd(
                (-tau_arr[3]).exp(),
                (-tau_arr[2]).exp(),
                (-tau_arr[1]).exp(),
                (-tau_arr[0]).exp(),
            );
            let shield_raw = _mm256_add_pd(
                one_v,
                _mm256_mul_pd(boost_v, _mm256_sub_pd(one_v, exp_neg_tau)),
            );
            _mm256_blendv_pd(one_v, shield_raw, active)
        } else {
            one_v
        };
        let h2_eq = _mm256_min_pd(
            one_v,
            _mm256_mul_pd(_mm256_div_pd(rho, threshold_v), shielding),
        );
        let grow = _mm256_fmadd_pd(tau_v, _mm256_sub_pd(h2_eq, old), old);
        let shielding_decay = _mm256_max_pd(one_hundred_v, shielding);
        let exp_arg = _mm256_div_pd(
            neg_dt_v,
            _mm256_mul_pd(_mm256_set1_pd(t_dissoc), shielding_decay),
        );
        let mut exp_args = [0.0; 4];
        // SAFETY: fixed-size stack array has exactly four f64 lanes.
        unsafe { _mm256_storeu_pd(exp_args.as_mut_ptr(), exp_arg) };
        let decay_v = _mm256_set_pd(
            exp_args[3].exp(),
            exp_args[2].exp(),
            exp_args[1].exp(),
            exp_args[0].exp(),
        );
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
    update_h2_scalar(&mut particles[chunks..], cfg, dust, dt, t_dissoc);
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
    dust: Option<&DustSection>,
    dt: f64,
    t_dissoc: f64,
) {
    // SAFETY: the caller checked AVX-512F; AVX2 arithmetic preserves the same update semantics.
    unsafe { update_h2_avx2(particles, cfg, dust, dt, t_dissoc) }
}
