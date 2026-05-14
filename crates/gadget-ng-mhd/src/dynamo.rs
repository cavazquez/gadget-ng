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

fn apply_turbulent_dynamo_particle(p: &mut Particle, v_rms: f64, dt: f64, decay: f64, alpha: f64) {
    if p.ptype != ParticleType::Gas {
        return;
    }

    let b2 = p.b_field.x * p.b_field.x + p.b_field.y * p.b_field.y + p.b_field.z * p.b_field.z;
    if b2 < 1e-60 {
        return;
    }

    let h = p.smoothing_length.max(1e-10);
    let rho = (p.mass / (4.0 / 3.0 * std::f64::consts::PI * h * h * h)).max(1e-30);

    let growth = dynamo_growth_rate(v_rms, b2.sqrt(), rho);
    let growth_factor = (growth * dt).exp();

    let b_norm = b2.sqrt().max(1e-30);
    let bhat_x = p.b_field.x / b_norm;
    let bhat_y = p.b_field.y / b_norm;
    let bhat_z = p.b_field.z / b_norm;

    let curl_b = alpha * (b_norm / h);

    p.b_field.x += dt * curl_b * bhat_x;
    p.b_field.y += dt * curl_b * bhat_y;
    p.b_field.z += dt * curl_b * bhat_z;

    let b_new =
        (p.b_field.x * p.b_field.x + p.b_field.y * p.b_field.y + p.b_field.z * p.b_field.z).sqrt();
    if b_new > 1e-30 {
        let grown = b_new * growth_factor;
        let renormalized = grown * decay;
        p.b_field.x *= renormalized / b_new;
        p.b_field.y *= renormalized / b_new;
        p.b_field.z *= renormalized / b_new;
    }
}

pub fn apply_turbulent_dynamo(particles: &mut [Particle], v_rms: f64, dt: f64, decay_time: f64) {
    let alpha = alpha_coefficient(v_rms, 0.5);
    if alpha < 1e-30 {
        return;
    }

    let decay = (-dt / decay_time.max(1e-10)).exp();

    #[cfg(feature = "rayon")]
    {
        particles
            .par_iter_mut()
            .for_each(|p| apply_turbulent_dynamo_particle(p, v_rms, dt, decay, alpha));
    }

    #[cfg(not(feature = "rayon"))]
    {
        #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
        {
            #[cfg(target_arch = "x86_64")]
            if is_x86_feature_detected!("avx512f") {
                // SAFETY: AVX-512F availability was checked at runtime.
                unsafe {
                    apply_turbulent_dynamo_avx512(particles, v_rms, dt, decay, alpha);
                    return;
                }
            }
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                // SAFETY: AVX2+FMA availability was checked at runtime.
                unsafe {
                    apply_turbulent_dynamo_avx2(particles, v_rms, dt, decay, alpha);
                    return;
                }
            }
        }
        for p in particles.iter_mut() {
            apply_turbulent_dynamo_particle(p, v_rms, dt, decay, alpha);
        }
    }
}

#[cfg(all(
    not(feature = "rayon"),
    feature = "simd",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn apply_turbulent_dynamo_avx2(
    particles: &mut [Particle],
    v_rms: f64,
    dt: f64,
    decay: f64,
    alpha: f64,
) {
    let lanes = 4;
    let n = particles.len();
    let chunks = n / lanes * lanes;
    let one_v = _mm256_set1_pd(1.0);
    let min_h_v = _mm256_set1_pd(1e-10);
    let min_rho_v = _mm256_set1_pd(1e-30);
    let min_b2_v = _mm256_set1_pd(1e-60);
    let four_thirds_pi_v = _mm256_set1_pd(4.0 / 3.0 * std::f64::consts::PI);
    let mut i = 0;
    while i + lanes <= chunks {
        let all_gas = particles[i..i + lanes]
            .iter()
            .all(|p| p.ptype == ParticleType::Gas);
        if !all_gas {
            for lane in 0..lanes {
                apply_turbulent_dynamo_particle(&mut particles[i + lane], v_rms, dt, decay, alpha);
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
        let h3 = _mm256_mul_pd(h, _mm256_mul_pd(h, h));
        let rho = _mm256_max_pd(
            min_rho_v,
            _mm256_div_pd(m, _mm256_mul_pd(four_thirds_pi_v, h3)),
        );
        let active_mask = _mm256_cmp_pd(b2, min_b2_v, _CMP_GT_OQ);
        let mut b2_arr = [0.0f64; 4];
        let mut rho_arr = [0.0f64; 4];
        let mut h_arr = [0.0f64; 4];
        // SAFETY: fixed-size stack arrays have exactly four f64 lanes.
        unsafe {
            _mm256_storeu_pd(b2_arr.as_mut_ptr(), b2);
            _mm256_storeu_pd(rho_arr.as_mut_ptr(), rho);
            _mm256_storeu_pd(h_arr.as_mut_ptr(), h);
        }
        let mut scale_arr = [1.0f64; 4];
        for lane in 0..lanes {
            if b2_arr[lane] >= 1e-60 {
                let b_rms = b2_arr[lane].sqrt();
                let growth = dynamo_growth_rate(v_rms, b_rms, rho_arr[lane]);
                let growth_factor = (growth * dt).exp();
                let alpha_over_h = alpha / h_arr[lane];
                scale_arr[lane] = (1.0 + dt * alpha_over_h) * growth_factor * decay;
            }
        }
        let scale_v = _mm256_set_pd(scale_arr[3], scale_arr[2], scale_arr[1], scale_arr[0]);
        let final_scale = _mm256_blendv_pd(one_v, scale_v, active_mask);
        let new_bx = _mm256_mul_pd(bx, final_scale);
        let new_by = _mm256_mul_pd(by, final_scale);
        let new_bz = _mm256_mul_pd(bz, final_scale);
        let mut out_bx = [0.0f64; 4];
        let mut out_by = [0.0f64; 4];
        let mut out_bz = [0.0f64; 4];
        // SAFETY: fixed-size stack arrays have exactly four f64 lanes.
        unsafe {
            _mm256_storeu_pd(out_bx.as_mut_ptr(), new_bx);
            _mm256_storeu_pd(out_by.as_mut_ptr(), new_by);
            _mm256_storeu_pd(out_bz.as_mut_ptr(), new_bz);
        }
        for lane in 0..lanes {
            particles[i + lane].b_field.x = out_bx[lane];
            particles[i + lane].b_field.y = out_by[lane];
            particles[i + lane].b_field.z = out_bz[lane];
        }
        i += lanes;
    }
    for p in particles[chunks..].iter_mut() {
        apply_turbulent_dynamo_particle(p, v_rms, dt, decay, alpha);
    }
}

#[cfg(all(
    not(feature = "rayon"),
    feature = "simd",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
unsafe fn apply_turbulent_dynamo_avx512(
    particles: &mut [Particle],
    v_rms: f64,
    dt: f64,
    decay: f64,
    alpha: f64,
) {
    let lanes = 8;
    let n = particles.len();
    let chunks = n / lanes * lanes;
    let min_h_v = _mm512_set1_pd(1e-10);
    let min_rho_v = _mm512_set1_pd(1e-30);
    let min_b2_v = _mm512_set1_pd(1e-60);
    let four_thirds_pi_v = _mm512_set1_pd(4.0 / 3.0 * std::f64::consts::PI);
    let mut i = 0;
    while i + lanes <= chunks {
        let all_gas = particles[i..i + lanes]
            .iter()
            .all(|p| p.ptype == ParticleType::Gas);
        if !all_gas {
            for lane in 0..lanes {
                apply_turbulent_dynamo_particle(&mut particles[i + lane], v_rms, dt, decay, alpha);
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
        let active_mask = _mm512_cmp_pd_mask(b2, min_b2_v, _CMP_GT_OQ);
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
        let h3 = _mm512_mul_pd(h, _mm512_mul_pd(h, h));
        let rho = _mm512_max_pd(
            min_rho_v,
            _mm512_div_pd(m, _mm512_mul_pd(four_thirds_pi_v, h3)),
        );
        let mut b2_arr = [0.0f64; 8];
        let mut rho_arr = [0.0f64; 8];
        let mut h_arr = [0.0f64; 8];
        // SAFETY: fixed-size stack arrays have exactly eight f64 lanes.
        unsafe {
            _mm512_storeu_pd(b2_arr.as_mut_ptr(), b2);
            _mm512_storeu_pd(rho_arr.as_mut_ptr(), rho);
            _mm512_storeu_pd(h_arr.as_mut_ptr(), h);
        }
        let mut scale_arr = [1.0f64; 8];
        for lane in 0..lanes {
            if b2_arr[lane] >= 1e-60 {
                let b_rms = b2_arr[lane].sqrt();
                let growth = dynamo_growth_rate(v_rms, b_rms, rho_arr[lane]);
                let growth_factor = (growth * dt).exp();
                let alpha_over_h = alpha / h_arr[lane];
                scale_arr[lane] = (1.0 + dt * alpha_over_h) * growth_factor * decay;
            }
        }
        let scale_v = _mm512_set_pd(
            scale_arr[7],
            scale_arr[6],
            scale_arr[5],
            scale_arr[4],
            scale_arr[3],
            scale_arr[2],
            scale_arr[1],
            scale_arr[0],
        );
        let new_bx = _mm512_mask_mul_pd(bx, active_mask, bx, scale_v);
        let new_by = _mm512_mask_mul_pd(by, active_mask, by, scale_v);
        let new_bz = _mm512_mask_mul_pd(bz, active_mask, bz, scale_v);
        let mut out_bx = [0.0f64; 8];
        let mut out_by = [0.0f64; 8];
        let mut out_bz = [0.0f64; 8];
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
        apply_turbulent_dynamo_particle(p, v_rms, dt, decay, alpha);
    }
}

#[cfg(not(feature = "rayon"))]
fn magnetic_energy_ratio_scalar(particles: &[Particle], _gamma: f64) -> f64 {
    let mut e_kin_sum = 0.0_f64;
    let mut e_mag_sum = 0.0_f64;
    let mut n = 0usize;

    for p in particles.iter() {
        if p.ptype != ParticleType::Gas {
            continue;
        }
        let h = p.smoothing_length.max(1e-10);
        let rho = (p.mass / (h * h * h)).max(1e-30);

        let v2 =
            p.velocity.x * p.velocity.x + p.velocity.y * p.velocity.y + p.velocity.z * p.velocity.z;
        let b2 = p.b_field.x * p.b_field.x + p.b_field.y * p.b_field.y + p.b_field.z * p.b_field.z;

        e_kin_sum += 0.5 * rho * v2;
        e_mag_sum += b2 / (2.0 * MU0);
        n += 1;
    }

    if e_kin_sum < 1e-30 || n == 0 {
        0.0
    } else {
        e_mag_sum / e_kin_sum
    }
}

#[cfg(feature = "rayon")]
fn magnetic_energy_ratio_par(particles: &[Particle], _gamma: f64) -> f64 {
    let (e_kin, e_mag, n) = particles
        .par_iter()
        .filter(|p| p.ptype == ParticleType::Gas)
        .fold(
            || (0.0_f64, 0.0_f64, 0usize),
            |(ek, em, cnt), p| {
                let h = p.smoothing_length.max(1e-10);
                let rho = (p.mass / (h * h * h)).max(1e-30);
                let v2 = p.velocity.x * p.velocity.x
                    + p.velocity.y * p.velocity.y
                    + p.velocity.z * p.velocity.z;
                let b2 = p.b_field.x * p.b_field.x
                    + p.b_field.y * p.b_field.y
                    + p.b_field.z * p.b_field.z;
                (ek + 0.5 * rho * v2, em + b2 / (2.0 * MU0), cnt + 1)
            },
        )
        .reduce(|| (0.0, 0.0, 0), |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2));

    if e_kin < 1e-30 || n == 0 {
        0.0
    } else {
        e_mag / e_kin
    }
}

pub fn magnetic_energy_ratio(particles: &[Particle], gamma: f64) -> f64 {
    #[cfg(feature = "rayon")]
    {
        magnetic_energy_ratio_par(particles, gamma)
    }

    #[cfg(not(feature = "rayon"))]
    {
        #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
        {
            #[cfg(target_arch = "x86_64")]
            if is_x86_feature_detected!("avx512f") {
                // SAFETY: AVX-512F availability was checked at runtime.
                unsafe {
                    return magnetic_energy_ratio_avx512(particles);
                }
            }
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                // SAFETY: AVX2+FMA availability was checked at runtime.
                unsafe {
                    return magnetic_energy_ratio_avx2(particles);
                }
            }
        }
        magnetic_energy_ratio_scalar(particles, gamma)
    }
}

#[cfg(all(
    not(feature = "rayon"),
    feature = "simd",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn magnetic_energy_ratio_avx2(particles: &[Particle]) -> f64 {
    let lanes = 4;
    let chunks = particles.len() / lanes * lanes;
    let half_v = _mm256_set1_pd(0.5);
    let inv_2mu0_v = _mm256_set1_pd(1.0 / (2.0 * MU0));
    let min_h_v = _mm256_set1_pd(1e-10);
    let min_rho_v = _mm256_set1_pd(1e-30);
    let mut e_kin_v = _mm256_setzero_pd();
    let mut e_mag_v = _mm256_setzero_pd();
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
                    let rho = (particles[i + lane].mass / (h * h * h)).max(1e-30);
                    let v2 = particles[i + lane].velocity.x * particles[i + lane].velocity.x
                        + particles[i + lane].velocity.y * particles[i + lane].velocity.y
                        + particles[i + lane].velocity.z * particles[i + lane].velocity.z;
                    let b2 = particles[i + lane].b_field.x * particles[i + lane].b_field.x
                        + particles[i + lane].b_field.y * particles[i + lane].b_field.y
                        + particles[i + lane].b_field.z * particles[i + lane].b_field.z;
                    let ek_lane = 0.5 * rho * v2;
                    let em_lane = b2 / (2.0 * MU0);
                    e_kin_v = _mm256_add_pd(e_kin_v, _mm256_set_pd(0.0, 0.0, 0.0, ek_lane));
                    e_mag_v = _mm256_add_pd(e_mag_v, _mm256_set_pd(0.0, 0.0, 0.0, em_lane));
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
        let rho = _mm256_max_pd(
            min_rho_v,
            _mm256_div_pd(m, _mm256_mul_pd(h, _mm256_mul_pd(h, h))),
        );
        let vx = _mm256_set_pd(
            particles[i + 3].velocity.x,
            particles[i + 2].velocity.x,
            particles[i + 1].velocity.x,
            particles[i].velocity.x,
        );
        let vy = _mm256_set_pd(
            particles[i + 3].velocity.y,
            particles[i + 2].velocity.y,
            particles[i + 1].velocity.y,
            particles[i].velocity.y,
        );
        let vz = _mm256_set_pd(
            particles[i + 3].velocity.z,
            particles[i + 2].velocity.z,
            particles[i + 1].velocity.z,
            particles[i].velocity.z,
        );
        let v2 = _mm256_fmadd_pd(vx, vx, _mm256_fmadd_pd(vy, vy, _mm256_mul_pd(vz, vz)));
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
        e_kin_v = _mm256_fmadd_pd(half_v, _mm256_mul_pd(rho, v2), e_kin_v);
        e_mag_v = _mm256_fmadd_pd(inv_2mu0_v, b2, e_mag_v);
        i += lanes;
    }
    let mut e_kin_arr = [0.0f64; 4];
    let mut e_mag_arr = [0.0f64; 4];
    // SAFETY: fixed-size stack arrays have exactly four f64 lanes.
    unsafe {
        _mm256_storeu_pd(e_kin_arr.as_mut_ptr(), e_kin_v);
        _mm256_storeu_pd(e_mag_arr.as_mut_ptr(), e_mag_v);
    }
    let mut e_kin_sum = e_kin_arr.into_iter().sum::<f64>();
    let mut e_mag_sum = e_mag_arr.into_iter().sum::<f64>();
    for p in &particles[chunks..] {
        if p.ptype != ParticleType::Gas {
            continue;
        }
        let h = p.smoothing_length.max(1e-10);
        let rho = (p.mass / (h * h * h)).max(1e-30);
        let v2 =
            p.velocity.x * p.velocity.x + p.velocity.y * p.velocity.y + p.velocity.z * p.velocity.z;
        let b2 = p.b_field.x * p.b_field.x + p.b_field.y * p.b_field.y + p.b_field.z * p.b_field.z;
        e_kin_sum += 0.5 * rho * v2;
        e_mag_sum += b2 / (2.0 * MU0);
        n_gas += 1;
    }
    if e_kin_sum < 1e-30 || n_gas == 0 {
        0.0
    } else {
        e_mag_sum / e_kin_sum
    }
}

#[cfg(all(
    not(feature = "rayon"),
    feature = "simd",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
unsafe fn magnetic_energy_ratio_avx512(particles: &[Particle]) -> f64 {
    let lanes = 8;
    let chunks = particles.len() / lanes * lanes;
    let half_v = _mm512_set1_pd(0.5);
    let inv_2mu0_v = _mm512_set1_pd(1.0 / (2.0 * MU0));
    let min_h_v = _mm512_set1_pd(1e-10);
    let min_rho_v = _mm512_set1_pd(1e-30);
    let mut e_kin_v = _mm512_setzero_pd();
    let mut e_mag_v = _mm512_setzero_pd();
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
            _mm512_div_pd(m, _mm512_mul_pd(h, _mm512_mul_pd(h, h))),
        );
        let vx = _mm512_set_pd(
            particles[i + 7].velocity.x,
            particles[i + 6].velocity.x,
            particles[i + 5].velocity.x,
            particles[i + 4].velocity.x,
            particles[i + 3].velocity.x,
            particles[i + 2].velocity.x,
            particles[i + 1].velocity.x,
            particles[i].velocity.x,
        );
        let vy = _mm512_set_pd(
            particles[i + 7].velocity.y,
            particles[i + 6].velocity.y,
            particles[i + 5].velocity.y,
            particles[i + 4].velocity.y,
            particles[i + 3].velocity.y,
            particles[i + 2].velocity.y,
            particles[i + 1].velocity.y,
            particles[i].velocity.y,
        );
        let vz = _mm512_set_pd(
            particles[i + 7].velocity.z,
            particles[i + 6].velocity.z,
            particles[i + 5].velocity.z,
            particles[i + 4].velocity.z,
            particles[i + 3].velocity.z,
            particles[i + 2].velocity.z,
            particles[i + 1].velocity.z,
            particles[i].velocity.z,
        );
        let v2 = _mm512_fmadd_pd(vx, vx, _mm512_fmadd_pd(vy, vy, _mm512_mul_pd(vz, vz)));
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
        let rho_masked = _mm512_maskz_add_pd(mask_bits, rho, _mm512_setzero_pd());
        let v2_masked = _mm512_maskz_add_pd(mask_bits, v2, _mm512_setzero_pd());
        let b2_masked = _mm512_maskz_add_pd(mask_bits, b2, _mm512_setzero_pd());
        e_kin_v = _mm512_fmadd_pd(half_v, _mm512_mul_pd(rho_masked, v2_masked), e_kin_v);
        e_mag_v = _mm512_fmadd_pd(inv_2mu0_v, b2_masked, e_mag_v);
        i += lanes;
    }
    let mut e_kin_arr = [0.0f64; 8];
    let mut e_mag_arr = [0.0f64; 8];
    // SAFETY: fixed-size stack arrays have exactly eight f64 lanes.
    unsafe {
        _mm512_storeu_pd(e_kin_arr.as_mut_ptr(), e_kin_v);
        _mm512_storeu_pd(e_mag_arr.as_mut_ptr(), e_mag_v);
    }
    let mut e_kin_sum = e_kin_arr.into_iter().sum::<f64>();
    let mut e_mag_sum = e_mag_arr.into_iter().sum::<f64>();
    for p in &particles[chunks..] {
        if p.ptype != ParticleType::Gas {
            continue;
        }
        let h = p.smoothing_length.max(1e-10);
        let rho = (p.mass / (h * h * h)).max(1e-30);
        let v2 =
            p.velocity.x * p.velocity.x + p.velocity.y * p.velocity.y + p.velocity.z * p.velocity.z;
        let b2 = p.b_field.x * p.b_field.x + p.b_field.y * p.b_field.y + p.b_field.z * p.b_field.z;
        e_kin_sum += 0.5 * rho * v2;
        e_mag_sum += b2 / (2.0 * MU0);
        n_gas += 1;
    }
    if e_kin_sum < 1e-30 || n_gas == 0 {
        0.0
    } else {
        e_mag_sum / e_kin_sum
    }
}

pub fn maxwell_stress_tensor(b: Vec3, rho: f64) -> f64 {
    let b2 = b.x * b.x + b.y * b.y + b.z * b.z;
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
