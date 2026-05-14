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
    {
        #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
        {
            #[cfg(target_arch = "x86_64")]
            if is_x86_feature_detected!("avx512f") {
                // SAFETY: AVX-512F availability was checked at runtime.
                unsafe {
                    return apply_flux_freeze_avx512(particles, gamma, beta_freeze, rho_ref);
                }
            }
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                // SAFETY: AVX2+FMA availability was checked at runtime.
                unsafe {
                    return apply_flux_freeze_avx2(particles, gamma, beta_freeze, rho_ref);
                }
            }
        }
        for p in particles.iter_mut() {
            apply_flux_freeze_particle(p, gamma, beta_freeze, rho_ref);
        }
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
        #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
        {
            #[cfg(target_arch = "x86_64")]
            if is_x86_feature_detected!("avx512f") {
                // SAFETY: AVX-512F availability was checked at runtime.
                unsafe {
                    return mean_gas_density_avx512(particles);
                }
            }
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                // SAFETY: AVX2+FMA availability was checked at runtime.
                unsafe {
                    return mean_gas_density_avx2(particles);
                }
            }
        }
        mean_gas_density_scalar(particles)
    }
}

#[cfg(not(feature = "rayon"))]
fn mean_gas_density_scalar(particles: &[Particle]) -> f64 {
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

#[cfg(all(
    not(feature = "rayon"),
    feature = "simd",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn mean_gas_density_avx2(particles: &[Particle]) -> f64 {
    let lanes = 4;
    let chunks = particles.len() / lanes * lanes;
    let min_h_v = _mm256_set1_pd(1e-10);
    let mut rho_v = _mm256_setzero_pd();
    let mut n_gas = 0usize;
    let mut i = 0;
    while i < chunks {
        let all_gas = particles[i..i + lanes]
            .iter()
            .all(|p| p.ptype == ParticleType::Gas);
        if !all_gas {
            for lane in 0..lanes {
                if particles[i + lane].ptype == ParticleType::Gas {
                    let h = particles[i + lane].smoothing_length.max(1e-10);
                    rho_v = _mm256_add_pd(
                        rho_v,
                        _mm256_set_pd(0.0, 0.0, 0.0, particles[i + lane].mass / (h * h * h)),
                    );
                    n_gas += 1;
                }
            }
            i += lanes;
            continue;
        }
        n_gas += lanes;
        let h = _mm256_max_pd(
            min_h_v,
            _mm256_set_pd(
                particles[i + 3].smoothing_length,
                particles[i + 2].smoothing_length,
                particles[i + 1].smoothing_length,
                particles[i].smoothing_length,
            ),
        );
        let m = _mm256_set_pd(
            particles[i + 3].mass,
            particles[i + 2].mass,
            particles[i + 1].mass,
            particles[i].mass,
        );
        let rho_lane = _mm256_div_pd(m, _mm256_mul_pd(h, _mm256_mul_pd(h, h)));
        rho_v = _mm256_add_pd(rho_v, rho_lane);
        i += lanes;
    }
    let mut out = [0.0; 4];
    // SAFETY: fixed-size stack array has exactly four f64 lanes.
    unsafe {
        _mm256_storeu_pd(out.as_mut_ptr(), rho_v);
    }
    let mut rho_sum = out.into_iter().sum::<f64>();
    for p in &particles[chunks..] {
        if p.ptype != ParticleType::Gas {
            continue;
        }
        let h = p.smoothing_length.max(1e-10);
        rho_sum += p.mass / (h * h * h);
        n_gas += 1;
    }
    if n_gas == 0 {
        1.0
    } else {
        rho_sum / n_gas as f64
    }
}

#[cfg(all(
    not(feature = "rayon"),
    feature = "simd",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
unsafe fn mean_gas_density_avx512(particles: &[Particle]) -> f64 {
    let lanes = 8;
    let chunks = particles.len() / lanes * lanes;
    let min_h_v = _mm512_set1_pd(1e-10);
    let mut rho_v = _mm512_setzero_pd();
    let mut n_gas = 0usize;
    let mut i = 0;
    while i < chunks {
        let all_gas = particles[i..i + lanes]
            .iter()
            .all(|p| p.ptype == ParticleType::Gas);
        if !all_gas {
            for lane in 0..lanes {
                if particles[i + lane].ptype == ParticleType::Gas {
                    let h = particles[i + lane].smoothing_length.max(1e-10);
                    rho_v = _mm512_add_pd(
                        rho_v,
                        _mm512_set_pd(
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            particles[i + lane].mass / (h * h * h),
                        ),
                    );
                    n_gas += 1;
                }
            }
            i += lanes;
            continue;
        }
        n_gas += lanes;
        let h = _mm512_max_pd(
            min_h_v,
            _mm512_set_pd(
                particles[i + 7].smoothing_length,
                particles[i + 6].smoothing_length,
                particles[i + 5].smoothing_length,
                particles[i + 4].smoothing_length,
                particles[i + 3].smoothing_length,
                particles[i + 2].smoothing_length,
                particles[i + 1].smoothing_length,
                particles[i].smoothing_length,
            ),
        );
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
        let rho_lane = _mm512_div_pd(m, _mm512_mul_pd(h, _mm512_mul_pd(h, h)));
        rho_v = _mm512_add_pd(rho_v, rho_lane);
        i += lanes;
    }
    let mut out = [0.0; 8];
    // SAFETY: fixed-size stack array has exactly eight f64 lanes.
    unsafe {
        _mm512_storeu_pd(out.as_mut_ptr(), rho_v);
    }
    let mut rho_sum = out.into_iter().sum::<f64>();
    for p in &particles[chunks..] {
        if p.ptype != ParticleType::Gas {
            continue;
        }
        let h = p.smoothing_length.max(1e-10);
        rho_sum += p.mass / (h * h * h);
        n_gas += 1;
    }
    if n_gas == 0 {
        1.0
    } else {
        rho_sum / n_gas as f64
    }
}

#[cfg(all(
    not(feature = "rayon"),
    feature = "simd",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn apply_flux_freeze_avx2(
    particles: &mut [Particle],
    gamma: f64,
    beta_freeze: f64,
    rho_ref: f64,
) {
    let lanes = 4;
    let chunks = particles.len() / lanes * lanes;
    let gamma_m1 = gamma - 1.0;
    let gm1_v = _mm256_set1_pd(gamma_m1);
    let two_mu0_v = _mm256_set1_pd(2.0 * MU0);
    let beta_freeze_v = _mm256_set1_pd(beta_freeze);
    let rho_ref_v = _mm256_set1_pd(rho_ref);
    let min_h_v = _mm256_set1_pd(1e-10);
    let min_rho_v = _mm256_set1_pd(1e-30);
    let min_b2_v = _mm256_set1_pd(1e-60);
    let zero_v = _mm256_set1_pd(0.0);
    let four_thirds_pi_v = _mm256_set1_pd(4.0 / 3.0 * std::f64::consts::PI);
    let mut i = 0;
    while i < chunks {
        let all_gas = particles[i..i + lanes]
            .iter()
            .all(|p| p.ptype == ParticleType::Gas);
        if !all_gas {
            for lane in 0..lanes {
                apply_flux_freeze_particle(&mut particles[i + lane], gamma, beta_freeze, rho_ref);
            }
            i += lanes;
            continue;
        }
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
        let has_b = _mm256_cmp_pd(b2, min_b2_v, _CMP_GT_OQ);
        let h = _mm256_max_pd(
            min_h_v,
            _mm256_set_pd(
                particles[i + 3].smoothing_length,
                particles[i + 2].smoothing_length,
                particles[i + 1].smoothing_length,
                particles[i].smoothing_length,
            ),
        );
        let m = _mm256_set_pd(
            particles[i + 3].mass,
            particles[i + 2].mass,
            particles[i + 1].mass,
            particles[i].mass,
        );
        let rho = _mm256_max_pd(
            min_rho_v,
            _mm256_div_pd(
                m,
                _mm256_mul_pd(four_thirds_pi_v, _mm256_mul_pd(h, _mm256_mul_pd(h, h))),
            ),
        );
        let u = _mm256_set_pd(
            particles[i + 3].internal_energy,
            particles[i + 2].internal_energy,
            particles[i + 1].internal_energy,
            particles[i].internal_energy,
        );
        let p_th = _mm256_mul_pd(gm1_v, _mm256_mul_pd(rho, u));
        let beta = _mm256_div_pd(_mm256_mul_pd(two_mu0_v, p_th), _mm256_max_pd(min_b2_v, b2));
        let ratio = _mm256_div_pd(rho, rho_ref_v);
        let mut ratio_arr = [0.0f64; 4];
        // SAFETY: fixed-size stack array has exactly four f64 lanes.
        unsafe {
            _mm256_storeu_pd(ratio_arr.as_mut_ptr(), ratio);
        }
        let scale_v = _mm256_set_pd(
            ratio_arr[3].powf(2.0 / 3.0),
            ratio_arr[2].powf(2.0 / 3.0),
            ratio_arr[1].powf(2.0 / 3.0),
            ratio_arr[0].powf(2.0 / 3.0),
        );
        let should_freeze = _mm256_and_pd(
            _mm256_cmp_pd(beta, beta_freeze_v, _CMP_GT_OQ),
            _mm256_cmp_pd(rho_ref_v, zero_v, _CMP_GT_OQ),
        );
        let masked_scale = _mm256_blendv_pd(_mm256_set1_pd(1.0), scale_v, should_freeze);
        let active_mask = _mm256_and_pd(has_b, should_freeze);
        let final_scale = _mm256_blendv_pd(_mm256_set1_pd(1.0), masked_scale, active_mask);
        let new_bx = _mm256_mul_pd(bx, final_scale);
        let new_by = _mm256_mul_pd(by, final_scale);
        let new_bz = _mm256_mul_pd(bz, final_scale);
        let mut out_bx = [0.0; 4];
        let mut out_by = [0.0; 4];
        let mut out_bz = [0.0; 4];
        // SAFETY: fixed-size stack arrays have exactly four f64 lanes.
        unsafe {
            _mm256_storeu_pd(out_bx.as_mut_ptr(), new_bx);
            _mm256_storeu_pd(out_by.as_mut_ptr(), new_by);
            _mm256_storeu_pd(out_bz.as_mut_ptr(), new_bz);
        }
        for lane in 0..lanes {
            let b2_local = out_bx[lane] * out_bx[lane]
                + out_by[lane] * out_by[lane]
                + out_bz[lane] * out_bz[lane];
            if b2_local >= 1e-60 {
                particles[i + lane].b_field.x = out_bx[lane];
                particles[i + lane].b_field.y = out_by[lane];
                particles[i + lane].b_field.z = out_bz[lane];
            }
        }
        i += lanes;
    }
    for p in particles[chunks..].iter_mut() {
        apply_flux_freeze_particle(p, gamma, beta_freeze, rho_ref);
    }
}

#[cfg(all(
    not(feature = "rayon"),
    feature = "simd",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
unsafe fn apply_flux_freeze_avx512(
    particles: &mut [Particle],
    gamma: f64,
    beta_freeze: f64,
    rho_ref: f64,
) {
    let lanes = 8;
    let chunks = particles.len() / lanes * lanes;
    let gamma_m1 = gamma - 1.0;
    let gm1_v = _mm512_set1_pd(gamma_m1);
    let two_mu0_v = _mm512_set1_pd(2.0 * MU0);
    let beta_freeze_v = _mm512_set1_pd(beta_freeze);
    let rho_ref_v = _mm512_set1_pd(rho_ref);
    let min_h_v = _mm512_set1_pd(1e-10);
    let min_rho_v = _mm512_set1_pd(1e-30);
    let min_b2_v = _mm512_set1_pd(1e-60);
    let zero_v = _mm512_set1_pd(0.0);
    let four_thirds_pi_v = _mm512_set1_pd(4.0 / 3.0 * std::f64::consts::PI);
    let mut i = 0;
    while i < chunks {
        let all_gas = particles[i..i + lanes]
            .iter()
            .all(|p| p.ptype == ParticleType::Gas);
        if !all_gas {
            for lane in 0..lanes {
                apply_flux_freeze_particle(&mut particles[i + lane], gamma, beta_freeze, rho_ref);
            }
            i += lanes;
            continue;
        }
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
        let has_b_mask = _mm512_cmp_pd_mask(b2, min_b2_v, _CMP_GT_OQ);
        let h = _mm512_max_pd(
            min_h_v,
            _mm512_set_pd(
                particles[i + 7].smoothing_length,
                particles[i + 6].smoothing_length,
                particles[i + 5].smoothing_length,
                particles[i + 4].smoothing_length,
                particles[i + 3].smoothing_length,
                particles[i + 2].smoothing_length,
                particles[i + 1].smoothing_length,
                particles[i].smoothing_length,
            ),
        );
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
        let rho = _mm512_max_pd(
            min_rho_v,
            _mm512_div_pd(
                m,
                _mm512_mul_pd(four_thirds_pi_v, _mm512_mul_pd(h, _mm512_mul_pd(h, h))),
            ),
        );
        let u = _mm512_set_pd(
            particles[i + 7].internal_energy,
            particles[i + 6].internal_energy,
            particles[i + 5].internal_energy,
            particles[i + 4].internal_energy,
            particles[i + 3].internal_energy,
            particles[i + 2].internal_energy,
            particles[i + 1].internal_energy,
            particles[i].internal_energy,
        );
        let p_th = _mm512_mul_pd(gm1_v, _mm512_mul_pd(rho, u));
        let beta = _mm512_div_pd(_mm512_mul_pd(two_mu0_v, p_th), _mm512_max_pd(min_b2_v, b2));
        let should_freeze_mask = _mm512_cmp_pd_mask(beta, beta_freeze_v, _CMP_GT_OQ)
            & _mm512_cmp_pd_mask(rho_ref_v, zero_v, _CMP_GT_OQ);
        let ratio = _mm512_div_pd(rho, rho_ref_v);
        let mut ratio_arr = [0.0f64; 8];
        // SAFETY: fixed-size stack array has exactly eight f64 lanes.
        unsafe {
            _mm512_storeu_pd(ratio_arr.as_mut_ptr(), ratio);
        }
        let scale_v = _mm512_set_pd(
            ratio_arr[7].powf(2.0 / 3.0),
            ratio_arr[6].powf(2.0 / 3.0),
            ratio_arr[5].powf(2.0 / 3.0),
            ratio_arr[4].powf(2.0 / 3.0),
            ratio_arr[3].powf(2.0 / 3.0),
            ratio_arr[2].powf(2.0 / 3.0),
            ratio_arr[1].powf(2.0 / 3.0),
            ratio_arr[0].powf(2.0 / 3.0),
        );
        let new_bx = _mm512_mask_mul_pd(bx, should_freeze_mask & has_b_mask, bx, scale_v);
        let new_by = _mm512_mask_mul_pd(by, should_freeze_mask & has_b_mask, by, scale_v);
        let new_bz = _mm512_mask_mul_pd(bz, should_freeze_mask & has_b_mask, bz, scale_v);
        let mut out_bx = [0.0; 8];
        let mut out_by = [0.0; 8];
        let mut out_bz = [0.0; 8];
        // SAFETY: fixed-size stack arrays have exactly eight f64 lanes.
        unsafe {
            _mm512_storeu_pd(out_bx.as_mut_ptr(), new_bx);
            _mm512_storeu_pd(out_by.as_mut_ptr(), new_by);
            _mm512_storeu_pd(out_bz.as_mut_ptr(), new_bz);
        }
        for lane in 0..lanes {
            particles[i + lane].b_field.x = out_bx[lane];
            particles[i + lane].b_field.y = out_by[lane];
            particles[i + lane].b_field.z = out_bz[lane];
        }
        i += lanes;
    }
    for p in particles[chunks..].iter_mut() {
        apply_flux_freeze_particle(p, gamma, beta_freeze, rho_ref);
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
