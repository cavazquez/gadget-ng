//! Reconexión magnética: liberación de energía de campo B antiparalelos (Phase 145).
//!
//! ## Modelo Sweet-Parker
//!
//! La tasa de reconexión Sweet-Parker es:
//!
//! ```text
//! v_rec = v_A / sqrt(Rm)    donde Rm = L × v_A / η_eff
//! ```
//!
//! En la implementación SPH se usa una versión simplificada basada en la detección de
//! pares de partículas con líneas de B antiparalelas (B_i · B_j < 0) dentro del radio
//! de suavizado 2h. La energía magnética de las líneas antiparalelas se libera como
//! calor local:
//!
//! ```text
//! ΔE_heat = (|B_i|² + |B_j|²) / (2 μ₀) × f_rec × dt
//! ```
//!
//! El campo B de las partículas involucradas se reduce proporcionalmente:
//!
//! ```text
//! |B_i|_new = |B_i| × sqrt(1 − f_rec × dt)   (conservación de flujo)
//! ```
//!
//! ## Referencias
//!
//! Sweet (1958), Nuovo Cim. Suppl. 8 — modelo original de reconexión.
//! Parker (1957), JGR 62, 509 — tasa de reconexión Sweet-Parker.
//! Lazarian & Vishniac (1999), ApJ 517, 700 — reconexión en turbulencia MHD.

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

#[cfg(not(feature = "rayon"))]
fn apply_magnetic_reconnection_impl(
    particles: &mut [Particle],
    f_reconnection: f64,
    _gamma: f64,
    dt: f64,
) {
    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    {
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx512f") {
            // SAFETY: AVX-512F availability was checked at runtime immediately above.
            unsafe {
                return apply_magnetic_reconnection_avx512(particles, f_reconnection, dt);
            }
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: AVX2+FMA availability was checked at runtime immediately above.
            unsafe {
                return apply_magnetic_reconnection_avx2(particles, f_reconnection, dt);
            }
        }
    }
    apply_magnetic_reconnection_scalar(particles, f_reconnection, dt);
}

#[cfg(not(feature = "rayon"))]
fn apply_magnetic_reconnection_scalar(particles: &mut [Particle], f_reconnection: f64, dt: f64) {
    if f_reconnection <= 0.0 {
        return;
    }
    let n = particles.len();
    if n < 2 {
        return;
    }

    let mut delta_u = vec![0.0_f64; n];
    let mut b_scale = vec![1.0_f64; n];

    for i in 0..n {
        if particles[i].ptype != ParticleType::Gas {
            continue;
        }
        let b_i = particles[i].b_field;
        let b2_i = b_i.x * b_i.x + b_i.y * b_i.y + b_i.z * b_i.z;
        if b2_i < 1e-60 {
            continue;
        }

        for j in (i + 1)..n {
            reconnection_pair_scalar(
                particles,
                &mut delta_u,
                &mut b_scale,
                i,
                j,
                f_reconnection,
                dt,
                (1.0 - f_reconnection * dt).max(0.0).sqrt(),
            );
        }
    }

    apply_reconnection_updates(particles, &delta_u, &b_scale);
}

#[cfg(not(feature = "rayon"))]
#[expect(
    clippy::too_many_arguments,
    reason = "pair update needs cached reconnection scalars and output buffers"
)]
fn reconnection_pair_scalar(
    particles: &[Particle],
    delta_u: &mut [f64],
    b_scale: &mut [f64],
    i: usize,
    j: usize,
    f_reconnection: f64,
    dt: f64,
    b_decay: f64,
) {
    if particles[j].ptype != ParticleType::Gas {
        return;
    }
    let h_i = particles[i].smoothing_length.max(1e-10);
    let h_j = particles[j].smoothing_length.max(1e-10);
    let b_i = particles[i].b_field;
    let b_j = particles[j].b_field;
    let b2_i = b_i.x * b_i.x + b_i.y * b_i.y + b_i.z * b_i.z;
    let b2_j = b_j.x * b_j.x + b_j.y * b_j.y + b_j.z * b_j.z;
    if b2_i < 1e-60 || b2_j < 1e-60 {
        return;
    }

    let dx = particles[j].position.x - particles[i].position.x;
    let dy = particles[j].position.y - particles[i].position.y;
    let dz = particles[j].position.z - particles[i].position.z;
    let r2 = dx * dx + dy * dy + dz * dz;
    let h_avg = 0.5 * (h_i + h_j);
    if r2 > 4.0 * h_avg * h_avg {
        return;
    }

    let b_dot = b_i.x * b_j.x + b_i.y * b_j.y + b_i.z * b_j.z;
    if b_dot >= 0.0 {
        return;
    }

    let e_mag_pair = (b2_i + b2_j) / (2.0 * MU0);
    let de_heat = e_mag_pair * f_reconnection * dt;
    let rho_i = (particles[i].mass / (h_i * h_i * h_i)).max(1e-30);
    let rho_j = (particles[j].mass / (h_j * h_j * h_j)).max(1e-30);

    delta_u[i] += 0.5 * de_heat / rho_i;
    delta_u[j] += 0.5 * de_heat / rho_j;
    b_scale[i] = b_scale[i].min(b_decay);
    b_scale[j] = b_scale[j].min(b_decay);
}

#[cfg(not(feature = "rayon"))]
fn apply_reconnection_updates(particles: &mut [Particle], delta_u: &[f64], b_scale: &[f64]) {
    for (i, p) in particles.iter_mut().enumerate() {
        if p.ptype == ParticleType::Gas {
            p.internal_energy = (p.internal_energy + delta_u[i]).max(0.0);
            if b_scale[i] < 1.0 {
                p.b_field.x *= b_scale[i];
                p.b_field.y *= b_scale[i];
                p.b_field.z *= b_scale[i];
            }
        }
    }
}

#[cfg(all(
    not(feature = "rayon"),
    feature = "simd",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn apply_magnetic_reconnection_avx2(
    particles: &mut [Particle],
    f_reconnection: f64,
    dt: f64,
) {
    if f_reconnection <= 0.0 {
        return;
    }
    let n = particles.len();
    if n < 2 {
        return;
    }

    let mut delta_u = vec![0.0_f64; n];
    let mut b_scale = vec![1.0_f64; n];
    let b_decay = (1.0 - f_reconnection * dt).max(0.0).sqrt();

    for i in 0..n {
        if particles[i].ptype != ParticleType::Gas {
            continue;
        }
        let h_i = particles[i].smoothing_length.max(1e-10);
        let b_i = particles[i].b_field;
        let b2_i = b_i.x * b_i.x + b_i.y * b_i.y + b_i.z * b_i.z;
        if b2_i < 1e-60 {
            continue;
        }
        let rho_i = (particles[i].mass / (h_i * h_i * h_i)).max(1e-30);
        let pos_i = particles[i].position;
        let mut j = i + 1;
        while j + 4 <= n {
            let all_gas = particles[j..j + 4]
                .iter()
                .all(|p| p.ptype == ParticleType::Gas);
            if !all_gas {
                for lane in 0..4 {
                    reconnection_pair_scalar(
                        particles,
                        &mut delta_u,
                        &mut b_scale,
                        i,
                        j + lane,
                        f_reconnection,
                        dt,
                        b_decay,
                    );
                }
                j += 4;
                continue;
            }

            let px = _mm256_set_pd(
                particles[j + 3].position.x,
                particles[j + 2].position.x,
                particles[j + 1].position.x,
                particles[j].position.x,
            );
            let py = _mm256_set_pd(
                particles[j + 3].position.y,
                particles[j + 2].position.y,
                particles[j + 1].position.y,
                particles[j].position.y,
            );
            let pz = _mm256_set_pd(
                particles[j + 3].position.z,
                particles[j + 2].position.z,
                particles[j + 1].position.z,
                particles[j].position.z,
            );
            let dx = _mm256_sub_pd(px, _mm256_set1_pd(pos_i.x));
            let dy = _mm256_sub_pd(py, _mm256_set1_pd(pos_i.y));
            let dz = _mm256_sub_pd(pz, _mm256_set1_pd(pos_i.z));
            let r2 = _mm256_fmadd_pd(dx, dx, _mm256_fmadd_pd(dy, dy, _mm256_mul_pd(dz, dz)));
            let h_j = _mm256_max_pd(
                _mm256_set1_pd(1e-10),
                _mm256_set_pd(
                    particles[j + 3].smoothing_length,
                    particles[j + 2].smoothing_length,
                    particles[j + 1].smoothing_length,
                    particles[j].smoothing_length,
                ),
            );
            let h_avg = _mm256_mul_pd(_mm256_set1_pd(0.5), _mm256_add_pd(_mm256_set1_pd(h_i), h_j));
            let within = _mm256_cmp_pd(
                r2,
                _mm256_mul_pd(_mm256_set1_pd(4.0), _mm256_mul_pd(h_avg, h_avg)),
                _CMP_LE_OQ,
            );
            let bx = _mm256_set_pd(
                particles[j + 3].b_field.x,
                particles[j + 2].b_field.x,
                particles[j + 1].b_field.x,
                particles[j].b_field.x,
            );
            let by = _mm256_set_pd(
                particles[j + 3].b_field.y,
                particles[j + 2].b_field.y,
                particles[j + 1].b_field.y,
                particles[j].b_field.y,
            );
            let bz = _mm256_set_pd(
                particles[j + 3].b_field.z,
                particles[j + 2].b_field.z,
                particles[j + 1].b_field.z,
                particles[j].b_field.z,
            );
            let b2_j = _mm256_fmadd_pd(bx, bx, _mm256_fmadd_pd(by, by, _mm256_mul_pd(bz, bz)));
            let b_dot = _mm256_fmadd_pd(
                _mm256_set1_pd(b_i.x),
                bx,
                _mm256_fmadd_pd(
                    _mm256_set1_pd(b_i.y),
                    by,
                    _mm256_mul_pd(_mm256_set1_pd(b_i.z), bz),
                ),
            );
            let active = _mm256_and_pd(
                within,
                _mm256_and_pd(
                    _mm256_cmp_pd(b2_j, _mm256_set1_pd(1e-60), _CMP_GE_OQ),
                    _mm256_cmp_pd(b_dot, _mm256_setzero_pd(), _CMP_LT_OQ),
                ),
            );
            let e_mag_pair = _mm256_div_pd(
                _mm256_add_pd(_mm256_set1_pd(b2_i), b2_j),
                _mm256_set1_pd(2.0 * MU0),
            );
            let de_heat = _mm256_mul_pd(e_mag_pair, _mm256_set1_pd(f_reconnection * dt));
            let du_i = _mm256_and_pd(_mm256_mul_pd(de_heat, _mm256_set1_pd(0.5 / rho_i)), active);
            let mass_j = _mm256_set_pd(
                particles[j + 3].mass,
                particles[j + 2].mass,
                particles[j + 1].mass,
                particles[j].mass,
            );
            let rho_j = _mm256_max_pd(
                _mm256_set1_pd(1e-30),
                _mm256_div_pd(mass_j, _mm256_mul_pd(h_j, _mm256_mul_pd(h_j, h_j))),
            );
            let du_j = _mm256_and_pd(
                _mm256_mul_pd(de_heat, _mm256_div_pd(_mm256_set1_pd(0.5), rho_j)),
                active,
            );

            let mut du_i_arr = [0.0; 4];
            let mut du_j_arr = [0.0; 4];
            let mut active_arr = [0.0; 4];
            // SAFETY: fixed-size stack arrays have exactly four f64 lanes.
            unsafe {
                _mm256_storeu_pd(du_i_arr.as_mut_ptr(), du_i);
                _mm256_storeu_pd(du_j_arr.as_mut_ptr(), du_j);
                _mm256_storeu_pd(active_arr.as_mut_ptr(), active);
            }
            delta_u[i] += du_i_arr.into_iter().sum::<f64>();
            for lane in 0..4 {
                if active_arr[lane].to_bits() != 0 {
                    delta_u[j + lane] += du_j_arr[lane];
                    b_scale[j + lane] = b_scale[j + lane].min(b_decay);
                    b_scale[i] = b_scale[i].min(b_decay);
                }
            }
            j += 4;
        }
        while j < n {
            reconnection_pair_scalar(
                particles,
                &mut delta_u,
                &mut b_scale,
                i,
                j,
                f_reconnection,
                dt,
                b_decay,
            );
            j += 1;
        }
    }

    apply_reconnection_updates(particles, &delta_u, &b_scale);
}

#[cfg(all(
    not(feature = "rayon"),
    feature = "simd",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
unsafe fn apply_magnetic_reconnection_avx512(
    particles: &mut [Particle],
    f_reconnection: f64,
    dt: f64,
) {
    if f_reconnection <= 0.0 {
        return;
    }
    let n = particles.len();
    if n < 2 {
        return;
    }
    let mut delta_u = vec![0.0_f64; n];
    let mut b_scale = vec![1.0_f64; n];
    let b_decay = (1.0 - f_reconnection * dt).max(0.0).sqrt();

    for i in 0..n {
        if particles[i].ptype != ParticleType::Gas {
            continue;
        }
        let h_i = particles[i].smoothing_length.max(1e-10);
        let b_i = particles[i].b_field;
        let b2_i = b_i.x * b_i.x + b_i.y * b_i.y + b_i.z * b_i.z;
        if b2_i < 1e-60 {
            continue;
        }
        let rho_i = (particles[i].mass / (h_i * h_i * h_i)).max(1e-30);
        let pos_i = particles[i].position;
        let mut j = i + 1;
        while j + 8 <= n {
            let all_gas = particles[j..j + 8]
                .iter()
                .all(|p| p.ptype == ParticleType::Gas);
            if !all_gas {
                for lane in 0..8 {
                    reconnection_pair_scalar(
                        particles,
                        &mut delta_u,
                        &mut b_scale,
                        i,
                        j + lane,
                        f_reconnection,
                        dt,
                        b_decay,
                    );
                }
                j += 8;
                continue;
            }
            let px = _mm512_set_pd(
                particles[j + 7].position.x,
                particles[j + 6].position.x,
                particles[j + 5].position.x,
                particles[j + 4].position.x,
                particles[j + 3].position.x,
                particles[j + 2].position.x,
                particles[j + 1].position.x,
                particles[j].position.x,
            );
            let py = _mm512_set_pd(
                particles[j + 7].position.y,
                particles[j + 6].position.y,
                particles[j + 5].position.y,
                particles[j + 4].position.y,
                particles[j + 3].position.y,
                particles[j + 2].position.y,
                particles[j + 1].position.y,
                particles[j].position.y,
            );
            let pz = _mm512_set_pd(
                particles[j + 7].position.z,
                particles[j + 6].position.z,
                particles[j + 5].position.z,
                particles[j + 4].position.z,
                particles[j + 3].position.z,
                particles[j + 2].position.z,
                particles[j + 1].position.z,
                particles[j].position.z,
            );
            let dx = _mm512_sub_pd(px, _mm512_set1_pd(pos_i.x));
            let dy = _mm512_sub_pd(py, _mm512_set1_pd(pos_i.y));
            let dz = _mm512_sub_pd(pz, _mm512_set1_pd(pos_i.z));
            let r2 = _mm512_fmadd_pd(dx, dx, _mm512_fmadd_pd(dy, dy, _mm512_mul_pd(dz, dz)));
            let h_j = _mm512_max_pd(
                _mm512_set1_pd(1e-10),
                _mm512_set_pd(
                    particles[j + 7].smoothing_length,
                    particles[j + 6].smoothing_length,
                    particles[j + 5].smoothing_length,
                    particles[j + 4].smoothing_length,
                    particles[j + 3].smoothing_length,
                    particles[j + 2].smoothing_length,
                    particles[j + 1].smoothing_length,
                    particles[j].smoothing_length,
                ),
            );
            let h_avg = _mm512_mul_pd(_mm512_set1_pd(0.5), _mm512_add_pd(_mm512_set1_pd(h_i), h_j));
            let within = _mm512_cmp_pd_mask(
                r2,
                _mm512_mul_pd(_mm512_set1_pd(4.0), _mm512_mul_pd(h_avg, h_avg)),
                _CMP_LE_OQ,
            );
            let bx = _mm512_set_pd(
                particles[j + 7].b_field.x,
                particles[j + 6].b_field.x,
                particles[j + 5].b_field.x,
                particles[j + 4].b_field.x,
                particles[j + 3].b_field.x,
                particles[j + 2].b_field.x,
                particles[j + 1].b_field.x,
                particles[j].b_field.x,
            );
            let by = _mm512_set_pd(
                particles[j + 7].b_field.y,
                particles[j + 6].b_field.y,
                particles[j + 5].b_field.y,
                particles[j + 4].b_field.y,
                particles[j + 3].b_field.y,
                particles[j + 2].b_field.y,
                particles[j + 1].b_field.y,
                particles[j].b_field.y,
            );
            let bz = _mm512_set_pd(
                particles[j + 7].b_field.z,
                particles[j + 6].b_field.z,
                particles[j + 5].b_field.z,
                particles[j + 4].b_field.z,
                particles[j + 3].b_field.z,
                particles[j + 2].b_field.z,
                particles[j + 1].b_field.z,
                particles[j].b_field.z,
            );
            let b2_j = _mm512_fmadd_pd(bx, bx, _mm512_fmadd_pd(by, by, _mm512_mul_pd(bz, bz)));
            let b_dot = _mm512_fmadd_pd(
                _mm512_set1_pd(b_i.x),
                bx,
                _mm512_fmadd_pd(
                    _mm512_set1_pd(b_i.y),
                    by,
                    _mm512_mul_pd(_mm512_set1_pd(b_i.z), bz),
                ),
            );
            let active = within
                & _mm512_cmp_pd_mask(b2_j, _mm512_set1_pd(1e-60), _CMP_GE_OQ)
                & _mm512_cmp_pd_mask(b_dot, _mm512_setzero_pd(), _CMP_LT_OQ);
            let e_mag_pair = _mm512_div_pd(
                _mm512_add_pd(_mm512_set1_pd(b2_i), b2_j),
                _mm512_set1_pd(2.0 * MU0),
            );
            let de_heat = _mm512_mul_pd(e_mag_pair, _mm512_set1_pd(f_reconnection * dt));
            let du_i = _mm512_maskz_mul_pd(active, de_heat, _mm512_set1_pd(0.5 / rho_i));
            let mass_j = _mm512_set_pd(
                particles[j + 7].mass,
                particles[j + 6].mass,
                particles[j + 5].mass,
                particles[j + 4].mass,
                particles[j + 3].mass,
                particles[j + 2].mass,
                particles[j + 1].mass,
                particles[j].mass,
            );
            let rho_j = _mm512_max_pd(
                _mm512_set1_pd(1e-30),
                _mm512_div_pd(mass_j, _mm512_mul_pd(h_j, _mm512_mul_pd(h_j, h_j))),
            );
            let du_j =
                _mm512_maskz_mul_pd(active, de_heat, _mm512_div_pd(_mm512_set1_pd(0.5), rho_j));
            let mut du_i_arr = [0.0; 8];
            let mut du_j_arr = [0.0; 8];
            // SAFETY: fixed-size stack arrays have exactly eight f64 lanes.
            unsafe {
                _mm512_storeu_pd(du_i_arr.as_mut_ptr(), du_i);
                _mm512_storeu_pd(du_j_arr.as_mut_ptr(), du_j);
            }
            delta_u[i] += du_i_arr.into_iter().sum::<f64>();
            for lane in 0..8 {
                if active & (1 << lane) != 0 {
                    delta_u[j + lane] += du_j_arr[lane];
                    b_scale[j + lane] = b_scale[j + lane].min(b_decay);
                    b_scale[i] = b_scale[i].min(b_decay);
                }
            }
            j += 8;
        }
        while j < n {
            reconnection_pair_scalar(
                particles,
                &mut delta_u,
                &mut b_scale,
                i,
                j,
                f_reconnection,
                dt,
                b_decay,
            );
            j += 1;
        }
    }

    apply_reconnection_updates(particles, &delta_u, &b_scale);
}

#[cfg(feature = "rayon")]
fn apply_magnetic_reconnection_par(
    particles: &mut [Particle],
    f_reconnection: f64,
    _gamma: f64,
    dt: f64,
) {
    if f_reconnection <= 0.0 {
        return;
    }
    let n = particles.len();
    if n < 2 {
        return;
    }

    let pos: Vec<gadget_ng_core::Vec3> = particles.iter().map(|p| p.position).collect();
    let mass: Vec<f64> = particles.iter().map(|p| p.mass).collect();
    let h_sml: Vec<f64> = particles
        .iter()
        .map(|p| p.smoothing_length.max(1e-10))
        .collect();
    let b_field: Vec<gadget_ng_core::Vec3> = particles.iter().map(|p| p.b_field).collect();
    let is_gas: Vec<bool> = particles
        .iter()
        .map(|p| p.ptype == ParticleType::Gas)
        .collect();

    let updates: Vec<Option<(f64, f64)>> = (0..n)
        .into_par_iter()
        .map(|i| {
            if !is_gas[i] {
                return None;
            }
            let h_i = h_sml[i];
            let b_i = b_field[i];
            let b2_i = b_i.x * b_i.x + b_i.y * b_i.y + b_i.z * b_i.z;
            if b2_i < 1e-60 {
                return Some((0.0, 1.0));
            }

            let rho_i = (mass[i] / (h_i * h_i * h_i)).max(1e-30);
            let mut delta_u_i = 0.0_f64;
            let mut b_scale_i = 1.0_f64;
            let b_decay = (1.0 - f_reconnection * dt).max(0.0).sqrt();

            for j in 0..n {
                if j == i || !is_gas[j] {
                    continue;
                }
                let b_j = b_field[j];
                let b2_j = b_j.x * b_j.x + b_j.y * b_j.y + b_j.z * b_j.z;
                if b2_j < 1e-60 {
                    continue;
                }

                let dx = pos[j].x - pos[i].x;
                let dy = pos[j].y - pos[i].y;
                let dz = pos[j].z - pos[i].z;
                let r2 = dx * dx + dy * dy + dz * dz;
                let h_avg = 0.5 * (h_i + h_sml[j]);

                if r2 > 4.0 * h_avg * h_avg {
                    continue;
                }

                let b_dot = b_i.x * b_j.x + b_i.y * b_j.y + b_i.z * b_j.z;
                if b_dot >= 0.0 {
                    continue;
                }

                let e_mag_pair = (b2_i + b2_j) / (2.0 * MU0);
                let de_heat = e_mag_pair * f_reconnection * dt;

                delta_u_i += 0.5 * de_heat / rho_i;
                b_scale_i = b_scale_i.min(b_decay);
            }
            Some((delta_u_i, b_scale_i))
        })
        .collect();

    for (p, update) in particles.iter_mut().zip(updates) {
        if let (true, Some((du, bs))) = (p.ptype == ParticleType::Gas, update) {
            p.internal_energy = (p.internal_energy + du).max(0.0);
            if bs < 1.0 {
                p.b_field.x *= bs;
                p.b_field.y *= bs;
                p.b_field.z *= bs;
            }
        }
    }
}

/// Aplica reconexión magnética entre pares de partículas antiparalelas (Phase 145).
///
/// Detecta pares `(i, j)` donde `B_i · B_j < 0` (campos antiparalelos) dentro de `2h`.
/// Libera una fracción `f_rec` de la energía magnética como calor en cada paso.
///
/// # Parámetros
///
/// - `particles`: slice mutable de partículas de gas
/// - `f_reconnection`: fracción de energía magnética liberada por paso (típico: 0.01)
/// - `gamma`: índice adiabático (para convertir ΔE a Δu)
/// - `dt`: paso de tiempo
pub fn apply_magnetic_reconnection(
    particles: &mut [Particle],
    f_reconnection: f64,
    gamma: f64,
    dt: f64,
) {
    #[cfg(feature = "rayon")]
    {
        apply_magnetic_reconnection_par(particles, f_reconnection, gamma, dt);
    }

    #[cfg(not(feature = "rayon"))]
    {
        apply_magnetic_reconnection_impl(particles, f_reconnection, gamma, dt);
    }
}

/// Tasa de reconexión Sweet-Parker teórica: `v_rec = v_A / sqrt(Rm)`.
///
/// - `v_a`: velocidad de Alfvén [unidades del código]
/// - `l_rec`: escala de la corriente de reconexión
/// - `eta_eff`: resistividad efectiva (numérica o física)
pub fn sweet_parker_rate(v_a: f64, l_rec: f64, eta_eff: f64) -> f64 {
    if eta_eff <= 0.0 || l_rec <= 0.0 {
        return 0.0;
    }
    let rm = l_rec * v_a / eta_eff;
    if rm <= 0.0 {
        return 0.0;
    }
    v_a / rm.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use gadget_ng_core::Vec3;

    fn make_reconnection_particles(n: usize, with_dm: bool) -> Vec<Particle> {
        (0..n)
            .map(|idx| {
                let t = idx as f64;
                let mut p = Particle::new_gas(
                    idx,
                    1.0 + 0.01 * t,
                    Vec3::new(0.08 * t, 0.03 * (0.5 * t).sin(), 0.02 * (0.7 * t).cos()),
                    Vec3::zero(),
                    0.5 + 0.01 * (idx % 3) as f64,
                    0.55 + 0.005 * (idx % 4) as f64,
                );
                let sign = if idx % 2 == 0 { 1.0 } else { -1.0 };
                p.b_field = Vec3::new(
                    sign * (0.2 + 0.003 * t),
                    -sign * (0.04 + 0.001 * t),
                    sign * 0.03,
                );
                if with_dm && matches!(idx, 5 | 12) {
                    let mut dm = Particle::new(idx, p.mass, p.position, Vec3::zero());
                    dm.internal_energy = 7.0;
                    dm.b_field = Vec3::new(9.0, -4.0, 2.0);
                    dm
                } else {
                    p
                }
            })
            .collect()
    }

    fn assert_reconnection_close(actual: &[Particle], expected: &[Particle]) {
        for (a, e) in actual.iter().zip(expected.iter()) {
            // SIMD batches reduce i-side pair heat horizontally; equivalent arithmetic can
            // differ from scalar by roundoff while preserving the pair update.
            assert!((a.internal_energy - e.internal_energy).abs() < 1e-10);
            assert!((a.b_field.x - e.b_field.x).abs() < 1e-12);
            assert!((a.b_field.y - e.b_field.y).abs() < 1e-12);
            assert!((a.b_field.z - e.b_field.z).abs() < 1e-12);
        }
    }

    #[test]
    #[cfg(not(feature = "rayon"))]
    fn reconnection_dispatch_matches_scalar_for_all_gas() {
        let mut scalar = make_reconnection_particles(16, false);
        let mut dispatched = scalar.clone();

        apply_magnetic_reconnection_scalar(&mut scalar, 0.04, 0.02);
        apply_magnetic_reconnection_impl(&mut dispatched, 0.04, 5.0 / 3.0, 0.02);

        assert_reconnection_close(&dispatched, &scalar);
    }

    #[test]
    #[cfg(not(feature = "rayon"))]
    fn reconnection_dispatch_matches_scalar_with_dark_matter() {
        let mut scalar = make_reconnection_particles(16, true);
        let mut dispatched = scalar.clone();
        let dm_before: Vec<(usize, f64, Vec3)> = dispatched
            .iter()
            .enumerate()
            .filter_map(|(idx, p)| {
                (p.ptype == ParticleType::DarkMatter).then_some((idx, p.internal_energy, p.b_field))
            })
            .collect();

        apply_magnetic_reconnection_scalar(&mut scalar, 0.03, 0.015);
        apply_magnetic_reconnection_impl(&mut dispatched, 0.03, 5.0 / 3.0, 0.015);

        assert_reconnection_close(&dispatched, &scalar);
        for (idx, u_before, b_before) in dm_before {
            assert_eq!(dispatched[idx].internal_energy, u_before);
            assert_eq!(dispatched[idx].b_field, b_before);
        }
    }
}
