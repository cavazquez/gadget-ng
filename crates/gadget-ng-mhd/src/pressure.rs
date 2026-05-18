//! Presión magnética y tensor de Maxwell en fuerzas SPH (Phase 124).
//!
//! ## Formulación
//!
//! La presión magnética es `P_B = |B|² / (2μ₀)` y el tensor de Maxwell es:
//!
//! ```text
//! M_ij = B_i B_j / μ₀ - P_B δ_ij
//! ```
//!
//! La contribución a la aceleración SPH es:
//!
//! ```text
//! (dv_i/dt)_B = Σ_j m_j [M_i/ρ_i² + M_j/ρ_j²] · ∇W_ij
//! ```
//!
//! ## Referencia
//!
//! Price & Monaghan (2005), MNRAS 364, 384.

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

/// Calcula la presión magnética escalar `P_B = |B|² / (2μ₀)`.
pub fn magnetic_pressure(b: Vec3) -> f64 {
    (b.x * b.x + b.y * b.y + b.z * b.z) / (2.0 * MU0)
}

/// Calcula el tensor de Maxwell 3×3: `M = B⊗B/μ₀ - P_B·I`.
///
/// Retorna la matriz como `[[f64; 3]; 3]` en orden fila (row-major).
pub fn maxwell_stress(b: Vec3) -> [[f64; 3]; 3] {
    let p_b = magnetic_pressure(b);
    [
        [b.x * b.x / MU0 - p_b, b.x * b.y / MU0, b.x * b.z / MU0],
        [b.y * b.x / MU0, b.y * b.y / MU0 - p_b, b.y * b.z / MU0],
        [b.z * b.x / MU0, b.z * b.y / MU0, b.z * b.z / MU0 - p_b],
    ]
}

/// Gradiente del kernel SPH (cúbico) — duplicado de `induction.rs` para evitar ciclos.
fn kernel_gradient(r_vec: Vec3, h: f64) -> Vec3 {
    let r2 = r_vec.x * r_vec.x + r_vec.y * r_vec.y + r_vec.z * r_vec.z;
    let r = r2.sqrt();
    if r < 1e-10 || h <= 0.0 {
        return Vec3::zero();
    }
    let q = r / h;
    let dw_dq = if q < 1.0 {
        8.0 / (std::f64::consts::PI * h.powi(3)) * (-6.0 * q + 9.0 * q * q)
    } else if q < 2.0 {
        8.0 / (std::f64::consts::PI * h.powi(3)) * (-6.0 * (2.0 - q).powi(2)) / 4.0
    } else {
        0.0
    };
    let dw_dr = dw_dq / h;
    Vec3 {
        x: dw_dr * r_vec.x / r,
        y: dw_dr * r_vec.y / r,
        z: dw_dr * r_vec.z / r,
    }
}

#[cfg(not(feature = "rayon"))]
fn apply_magnetic_forces_impl(particles: &mut [Particle], dt: f64) {
    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    {
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx512f") {
            // SAFETY: AVX-512F availability was checked at runtime immediately above.
            unsafe {
                return apply_magnetic_forces_avx512(particles, dt);
            }
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: AVX2+FMA availability was checked at runtime immediately above.
            unsafe {
                return apply_magnetic_forces_avx2(particles, dt);
            }
        }
    }
    apply_magnetic_forces_scalar(particles, dt);
}

#[cfg(not(feature = "rayon"))]
fn apply_magnetic_forces_scalar(particles: &mut [Particle], dt: f64) {
    let n = particles.len();
    if n == 0 {
        return;
    }

    let rho: Vec<f64> = particles
        .iter()
        .map(|p| {
            let h = p.smoothing_length.max(1e-10);
            (p.mass / (h * h * h)).max(1e-30)
        })
        .collect();
    let maxwell: Vec<[[f64; 3]; 3]> = particles
        .iter()
        .map(|p| maxwell_stress(p.b_field))
        .collect();

    let mut acc_mag = vec![Vec3::zero(); n];

    for i in 0..n {
        if particles[i].ptype != ParticleType::Gas {
            continue;
        }
        let rho_i2 = rho[i] * rho[i];
        let m_i = &maxwell[i];
        let pos_i = particles[i].position;
        let h_i = particles[i].smoothing_length;

        for j in 0..n {
            let acc = magnetic_force_pair_contribution(
                particles, &rho, &maxwell, i, j, pos_i, h_i, rho_i2, m_i,
            );
            acc_mag[i].x += acc.x;
            acc_mag[i].y += acc.y;
            acc_mag[i].z += acc.z;
        }
    }

    apply_magnetic_force_updates(particles, &acc_mag, dt);
}

#[cfg(not(feature = "rayon"))]
fn apply_magnetic_force_updates(particles: &mut [Particle], acc_mag: &[Vec3], dt: f64) {
    for (p, acc) in particles.iter_mut().zip(acc_mag.iter()) {
        if p.ptype == ParticleType::Gas {
            p.velocity.x += acc.x * dt;
            p.velocity.y += acc.y * dt;
            p.velocity.z += acc.z * dt;
        }
    }
}

#[cfg(not(feature = "rayon"))]
#[expect(
    clippy::too_many_arguments,
    reason = "hot pair kernel carries cached i-side data"
)]
fn magnetic_force_pair_contribution(
    particles: &[Particle],
    rho: &[f64],
    maxwell: &[[[f64; 3]; 3]],
    i: usize,
    j: usize,
    pos_i: Vec3,
    h_i: f64,
    rho_i2: f64,
    m_i: &[[f64; 3]; 3],
) -> Vec3 {
    if j == i || particles[j].ptype != ParticleType::Gas {
        return Vec3::zero();
    }
    let rho_j2 = rho[j] * rho[j];
    let m_j = &maxwell[j];

    let r_ij = Vec3 {
        x: particles[j].position.x - pos_i.x,
        y: particles[j].position.y - pos_i.y,
        z: particles[j].position.z - pos_i.z,
    };
    let h_ij = 0.5 * (h_i + particles[j].smoothing_length).max(1e-10);
    let grad_w = kernel_gradient(r_ij, h_ij);
    let gw = [grad_w.x, grad_w.y, grad_w.z];

    let mut a = [0.0_f64; 3];
    for k in 0..3 {
        for (l, &gw_l) in gw.iter().enumerate() {
            a[k] += (m_i[k][l] / rho_i2 + m_j[k][l] / rho_j2) * gw_l;
        }
    }

    Vec3 {
        x: particles[j].mass * a[0],
        y: particles[j].mass * a[1],
        z: particles[j].mass * a[2],
    }
}

#[cfg(all(
    not(feature = "rayon"),
    feature = "simd",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn apply_magnetic_forces_avx2(particles: &mut [Particle], dt: f64) {
    let n = particles.len();
    if n == 0 {
        return;
    }
    let rho: Vec<f64> = particles
        .iter()
        .map(|p| {
            let h = p.smoothing_length.max(1e-10);
            (p.mass / (h * h * h)).max(1e-30)
        })
        .collect();
    let maxwell: Vec<[[f64; 3]; 3]> = particles
        .iter()
        .map(|p| maxwell_stress(p.b_field))
        .collect();
    let mut acc_mag = vec![Vec3::zero(); n];
    for i in 0..n {
        if particles[i].ptype != ParticleType::Gas {
            continue;
        }
        // SAFETY: caller reached this AVX2 function only after runtime AVX2+FMA dispatch.
        acc_mag[i] = unsafe { magnetic_force_sum_avx2(particles, &rho, &maxwell, i) };
    }
    apply_magnetic_force_updates(particles, &acc_mag, dt);
}

#[cfg(all(
    not(feature = "rayon"),
    feature = "simd",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
unsafe fn apply_magnetic_forces_avx512(particles: &mut [Particle], dt: f64) {
    let n = particles.len();
    if n == 0 {
        return;
    }
    let rho: Vec<f64> = particles
        .iter()
        .map(|p| {
            let h = p.smoothing_length.max(1e-10);
            (p.mass / (h * h * h)).max(1e-30)
        })
        .collect();
    let maxwell: Vec<[[f64; 3]; 3]> = particles
        .iter()
        .map(|p| maxwell_stress(p.b_field))
        .collect();
    let mut acc_mag = vec![Vec3::zero(); n];
    for i in 0..n {
        if particles[i].ptype != ParticleType::Gas {
            continue;
        }
        // SAFETY: caller reached this AVX-512F function only after runtime AVX-512F dispatch.
        acc_mag[i] = unsafe { magnetic_force_sum_avx512(particles, &rho, &maxwell, i) };
    }
    apply_magnetic_force_updates(particles, &acc_mag, dt);
}

#[cfg(all(
    not(feature = "rayon"),
    feature = "simd",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn magnetic_force_sum_avx2(
    particles: &[Particle],
    rho: &[f64],
    maxwell: &[[[f64; 3]; 3]],
    i: usize,
) -> Vec3 {
    let lanes = 4;
    let chunks = particles.len() / lanes * lanes;
    let pos_i = particles[i].position;
    let h_i = particles[i].smoothing_length;
    let rho_i2 = rho[i] * rho[i];
    let m_i = &maxwell[i];
    let mut sum_x = _mm256_setzero_pd();
    let mut sum_y = _mm256_setzero_pd();
    let mut sum_z = _mm256_setzero_pd();
    let mut j = 0;
    while j < chunks {
        let all_valid = i < j || i >= j + lanes;
        let all_gas = particles[j..j + lanes]
            .iter()
            .all(|p| p.ptype == ParticleType::Gas);
        if !all_valid || !all_gas {
            for lane in 0..lanes {
                let contrib = magnetic_force_pair_contribution(
                    particles,
                    rho,
                    maxwell,
                    i,
                    j + lane,
                    pos_i,
                    h_i,
                    rho_i2,
                    m_i,
                );
                sum_x = _mm256_add_pd(sum_x, _mm256_set_pd(0.0, 0.0, 0.0, contrib.x));
                sum_y = _mm256_add_pd(sum_y, _mm256_set_pd(0.0, 0.0, 0.0, contrib.y));
                sum_z = _mm256_add_pd(sum_z, _mm256_set_pd(0.0, 0.0, 0.0, contrib.z));
            }
            j += lanes;
            continue;
        }
        let (cx, cy, cz) =
            magnetic_force_batch_avx2(particles, rho, maxwell, j, pos_i, h_i, rho_i2, m_i);
        sum_x = _mm256_add_pd(sum_x, cx);
        sum_y = _mm256_add_pd(sum_y, cy);
        sum_z = _mm256_add_pd(sum_z, cz);
        j += lanes;
    }
    let mut out_x = [0.0; 4];
    let mut out_y = [0.0; 4];
    let mut out_z = [0.0; 4];
    // SAFETY: fixed-size stack arrays have exactly four f64 lanes.
    unsafe {
        _mm256_storeu_pd(out_x.as_mut_ptr(), sum_x);
        _mm256_storeu_pd(out_y.as_mut_ptr(), sum_y);
        _mm256_storeu_pd(out_z.as_mut_ptr(), sum_z);
    }
    let mut acc = Vec3::new(
        out_x.into_iter().sum(),
        out_y.into_iter().sum(),
        out_z.into_iter().sum(),
    );
    for j_tail in chunks..particles.len() {
        let contrib = magnetic_force_pair_contribution(
            particles, rho, maxwell, i, j_tail, pos_i, h_i, rho_i2, m_i,
        );
        acc.x += contrib.x;
        acc.y += contrib.y;
        acc.z += contrib.z;
    }
    acc
}

#[cfg(all(
    not(feature = "rayon"),
    feature = "simd",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx2", enable = "fma")]
#[expect(
    clippy::too_many_arguments,
    reason = "hot batch kernel keeps i-side cached"
)]
fn magnetic_force_batch_avx2(
    particles: &[Particle],
    rho: &[f64],
    maxwell: &[[[f64; 3]; 3]],
    j: usize,
    pos_i: Vec3,
    h_i: f64,
    rho_i2: f64,
    m_i: &[[f64; 3]; 3],
) -> (__m256d, __m256d, __m256d) {
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
    let r = _mm256_sqrt_pd(r2);
    let h_j = _mm256_set_pd(
        particles[j + 3].smoothing_length,
        particles[j + 2].smoothing_length,
        particles[j + 1].smoothing_length,
        particles[j].smoothing_length,
    );
    let h_ij = _mm256_mul_pd(
        _mm256_set1_pd(0.5),
        _mm256_max_pd(
            _mm256_set1_pd(1e-10),
            _mm256_add_pd(_mm256_set1_pd(h_i), h_j),
        ),
    );
    let q = _mm256_div_pd(r, h_ij);
    let norm = _mm256_div_pd(
        _mm256_set1_pd(8.0 / std::f64::consts::PI),
        _mm256_mul_pd(h_ij, _mm256_mul_pd(h_ij, h_ij)),
    );
    let q2 = _mm256_mul_pd(q, q);
    let dw_inner = _mm256_mul_pd(
        norm,
        _mm256_fmadd_pd(
            _mm256_set1_pd(9.0),
            q2,
            _mm256_mul_pd(_mm256_set1_pd(-6.0), q),
        ),
    );
    let two_minus_q = _mm256_sub_pd(_mm256_set1_pd(2.0), q);
    let dw_outer = _mm256_mul_pd(
        norm,
        _mm256_mul_pd(
            _mm256_set1_pd(-1.5),
            _mm256_mul_pd(two_minus_q, two_minus_q),
        ),
    );
    let dw_dq = _mm256_blendv_pd(
        _mm256_setzero_pd(),
        _mm256_blendv_pd(
            dw_outer,
            dw_inner,
            _mm256_cmp_pd(q, _mm256_set1_pd(1.0), _CMP_LT_OQ),
        ),
        _mm256_cmp_pd(q, _mm256_set1_pd(2.0), _CMP_LT_OQ),
    );
    let scale = _mm256_div_pd(
        _mm256_div_pd(dw_dq, h_ij),
        _mm256_blendv_pd(
            r,
            _mm256_set1_pd(1.0),
            _mm256_cmp_pd(r, _mm256_set1_pd(1e-10), _CMP_LT_OQ),
        ),
    );
    let valid_r = _mm256_cmp_pd(r, _mm256_set1_pd(1e-10), _CMP_GE_OQ);
    let gwx = _mm256_and_pd(_mm256_mul_pd(scale, dx), valid_r);
    let gwy = _mm256_and_pd(_mm256_mul_pd(scale, dy), valid_r);
    let gwz = _mm256_and_pd(_mm256_mul_pd(scale, dz), valid_r);
    let mass = _mm256_set_pd(
        particles[j + 3].mass,
        particles[j + 2].mass,
        particles[j + 1].mass,
        particles[j].mass,
    );
    let inv_rho_j2 = _mm256_div_pd(
        _mm256_set1_pd(1.0),
        _mm256_set_pd(
            rho[j + 3] * rho[j + 3],
            rho[j + 2] * rho[j + 2],
            rho[j + 1] * rho[j + 1],
            rho[j] * rho[j],
        ),
    );
    let inv_rho_i2 = 1.0 / rho_i2;
    let row = |k: usize| {
        let mj0 = _mm256_set_pd(
            maxwell[j + 3][k][0],
            maxwell[j + 2][k][0],
            maxwell[j + 1][k][0],
            maxwell[j][k][0],
        );
        let mj1 = _mm256_set_pd(
            maxwell[j + 3][k][1],
            maxwell[j + 2][k][1],
            maxwell[j + 1][k][1],
            maxwell[j][k][1],
        );
        let mj2 = _mm256_set_pd(
            maxwell[j + 3][k][2],
            maxwell[j + 2][k][2],
            maxwell[j + 1][k][2],
            maxwell[j][k][2],
        );
        let c0 = _mm256_fmadd_pd(mj0, inv_rho_j2, _mm256_set1_pd(m_i[k][0] * inv_rho_i2));
        let c1 = _mm256_fmadd_pd(mj1, inv_rho_j2, _mm256_set1_pd(m_i[k][1] * inv_rho_i2));
        let c2 = _mm256_fmadd_pd(mj2, inv_rho_j2, _mm256_set1_pd(m_i[k][2] * inv_rho_i2));
        _mm256_mul_pd(
            mass,
            _mm256_fmadd_pd(c0, gwx, _mm256_fmadd_pd(c1, gwy, _mm256_mul_pd(c2, gwz))),
        )
    };
    (row(0), row(1), row(2))
}

#[cfg(all(
    not(feature = "rayon"),
    feature = "simd",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
unsafe fn magnetic_force_sum_avx512(
    particles: &[Particle],
    rho: &[f64],
    maxwell: &[[[f64; 3]; 3]],
    i: usize,
) -> Vec3 {
    let lanes = 8;
    let chunks = particles.len() / lanes * lanes;
    let pos_i = particles[i].position;
    let h_i = particles[i].smoothing_length;
    let rho_i2 = rho[i] * rho[i];
    let m_i = &maxwell[i];
    let mut sum_x = _mm512_setzero_pd();
    let mut sum_y = _mm512_setzero_pd();
    let mut sum_z = _mm512_setzero_pd();
    let mut j = 0;
    while j < chunks {
        let all_valid = i < j || i >= j + lanes;
        let all_gas = particles[j..j + lanes]
            .iter()
            .all(|p| p.ptype == ParticleType::Gas);
        if !all_valid || !all_gas {
            for lane in 0..lanes {
                let contrib = magnetic_force_pair_contribution(
                    particles,
                    rho,
                    maxwell,
                    i,
                    j + lane,
                    pos_i,
                    h_i,
                    rho_i2,
                    m_i,
                );
                sum_x = _mm512_add_pd(
                    sum_x,
                    _mm512_set_pd(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, contrib.x),
                );
                sum_y = _mm512_add_pd(
                    sum_y,
                    _mm512_set_pd(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, contrib.y),
                );
                sum_z = _mm512_add_pd(
                    sum_z,
                    _mm512_set_pd(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, contrib.z),
                );
            }
            j += lanes;
            continue;
        }
        let (cx, cy, cz) =
            magnetic_force_batch_avx512(particles, rho, maxwell, j, pos_i, h_i, rho_i2, m_i);
        sum_x = _mm512_add_pd(sum_x, cx);
        sum_y = _mm512_add_pd(sum_y, cy);
        sum_z = _mm512_add_pd(sum_z, cz);
        j += lanes;
    }
    let mut out_x = [0.0; 8];
    let mut out_y = [0.0; 8];
    let mut out_z = [0.0; 8];
    // SAFETY: fixed-size stack arrays have exactly eight f64 lanes.
    unsafe {
        _mm512_storeu_pd(out_x.as_mut_ptr(), sum_x);
        _mm512_storeu_pd(out_y.as_mut_ptr(), sum_y);
        _mm512_storeu_pd(out_z.as_mut_ptr(), sum_z);
    }
    let mut acc = Vec3::new(
        out_x.into_iter().sum(),
        out_y.into_iter().sum(),
        out_z.into_iter().sum(),
    );
    for j_tail in chunks..particles.len() {
        let contrib = magnetic_force_pair_contribution(
            particles, rho, maxwell, i, j_tail, pos_i, h_i, rho_i2, m_i,
        );
        acc.x += contrib.x;
        acc.y += contrib.y;
        acc.z += contrib.z;
    }
    acc
}

#[cfg(all(
    not(feature = "rayon"),
    feature = "simd",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
#[expect(
    clippy::too_many_arguments,
    reason = "hot batch kernel keeps i-side cached"
)]
fn magnetic_force_batch_avx512(
    particles: &[Particle],
    rho: &[f64],
    maxwell: &[[[f64; 3]; 3]],
    j: usize,
    pos_i: Vec3,
    h_i: f64,
    rho_i2: f64,
    m_i: &[[f64; 3]; 3],
) -> (__m512d, __m512d, __m512d) {
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
    let r = _mm512_sqrt_pd(r2);
    let h_j = _mm512_set_pd(
        particles[j + 7].smoothing_length,
        particles[j + 6].smoothing_length,
        particles[j + 5].smoothing_length,
        particles[j + 4].smoothing_length,
        particles[j + 3].smoothing_length,
        particles[j + 2].smoothing_length,
        particles[j + 1].smoothing_length,
        particles[j].smoothing_length,
    );
    let h_ij = _mm512_mul_pd(
        _mm512_set1_pd(0.5),
        _mm512_max_pd(
            _mm512_set1_pd(1e-10),
            _mm512_add_pd(_mm512_set1_pd(h_i), h_j),
        ),
    );
    let q = _mm512_div_pd(r, h_ij);
    let norm = _mm512_div_pd(
        _mm512_set1_pd(8.0 / std::f64::consts::PI),
        _mm512_mul_pd(h_ij, _mm512_mul_pd(h_ij, h_ij)),
    );
    let q2 = _mm512_mul_pd(q, q);
    let dw_inner = _mm512_mul_pd(
        norm,
        _mm512_fmadd_pd(
            _mm512_set1_pd(9.0),
            q2,
            _mm512_mul_pd(_mm512_set1_pd(-6.0), q),
        ),
    );
    let two_minus_q = _mm512_sub_pd(_mm512_set1_pd(2.0), q);
    let dw_outer = _mm512_mul_pd(
        norm,
        _mm512_mul_pd(
            _mm512_set1_pd(-1.5),
            _mm512_mul_pd(two_minus_q, two_minus_q),
        ),
    );
    let inner_mask = _mm512_cmp_pd_mask(q, _mm512_set1_pd(1.0), _CMP_LT_OQ);
    let support_mask = _mm512_cmp_pd_mask(q, _mm512_set1_pd(2.0), _CMP_LT_OQ);
    let dw_dq = _mm512_mask_blend_pd(
        support_mask,
        _mm512_setzero_pd(),
        _mm512_mask_blend_pd(inner_mask, dw_outer, dw_inner),
    );
    let safe_r = _mm512_mask_blend_pd(
        _mm512_cmp_pd_mask(r, _mm512_set1_pd(1e-10), _CMP_LT_OQ),
        r,
        _mm512_set1_pd(1.0),
    );
    let scale = _mm512_div_pd(_mm512_div_pd(dw_dq, h_ij), safe_r);
    let valid_r = _mm512_cmp_pd_mask(r, _mm512_set1_pd(1e-10), _CMP_GE_OQ);
    let gwx = _mm512_maskz_mul_pd(valid_r, scale, dx);
    let gwy = _mm512_maskz_mul_pd(valid_r, scale, dy);
    let gwz = _mm512_maskz_mul_pd(valid_r, scale, dz);
    let mass = _mm512_set_pd(
        particles[j + 7].mass,
        particles[j + 6].mass,
        particles[j + 5].mass,
        particles[j + 4].mass,
        particles[j + 3].mass,
        particles[j + 2].mass,
        particles[j + 1].mass,
        particles[j].mass,
    );
    let inv_rho_j2 = _mm512_div_pd(
        _mm512_set1_pd(1.0),
        _mm512_set_pd(
            rho[j + 7] * rho[j + 7],
            rho[j + 6] * rho[j + 6],
            rho[j + 5] * rho[j + 5],
            rho[j + 4] * rho[j + 4],
            rho[j + 3] * rho[j + 3],
            rho[j + 2] * rho[j + 2],
            rho[j + 1] * rho[j + 1],
            rho[j] * rho[j],
        ),
    );
    let inv_rho_i2 = 1.0 / rho_i2;
    let row = |k: usize| {
        let mj0 = _mm512_set_pd(
            maxwell[j + 7][k][0],
            maxwell[j + 6][k][0],
            maxwell[j + 5][k][0],
            maxwell[j + 4][k][0],
            maxwell[j + 3][k][0],
            maxwell[j + 2][k][0],
            maxwell[j + 1][k][0],
            maxwell[j][k][0],
        );
        let mj1 = _mm512_set_pd(
            maxwell[j + 7][k][1],
            maxwell[j + 6][k][1],
            maxwell[j + 5][k][1],
            maxwell[j + 4][k][1],
            maxwell[j + 3][k][1],
            maxwell[j + 2][k][1],
            maxwell[j + 1][k][1],
            maxwell[j][k][1],
        );
        let mj2 = _mm512_set_pd(
            maxwell[j + 7][k][2],
            maxwell[j + 6][k][2],
            maxwell[j + 5][k][2],
            maxwell[j + 4][k][2],
            maxwell[j + 3][k][2],
            maxwell[j + 2][k][2],
            maxwell[j + 1][k][2],
            maxwell[j][k][2],
        );
        let c0 = _mm512_fmadd_pd(mj0, inv_rho_j2, _mm512_set1_pd(m_i[k][0] * inv_rho_i2));
        let c1 = _mm512_fmadd_pd(mj1, inv_rho_j2, _mm512_set1_pd(m_i[k][1] * inv_rho_i2));
        let c2 = _mm512_fmadd_pd(mj2, inv_rho_j2, _mm512_set1_pd(m_i[k][2] * inv_rho_i2));
        _mm512_mul_pd(
            mass,
            _mm512_fmadd_pd(c0, gwx, _mm512_fmadd_pd(c1, gwy, _mm512_mul_pd(c2, gwz))),
        )
    };
    (row(0), row(1), row(2))
}

#[cfg(feature = "rayon")]
fn apply_magnetic_forces_par(particles: &mut [Particle], dt: f64) {
    let n = particles.len();
    if n == 0 {
        return;
    }

    let pos: Vec<Vec3> = particles.iter().map(|p| p.position).collect();
    let mass: Vec<f64> = particles.iter().map(|p| p.mass).collect();
    let h_sml: Vec<f64> = particles
        .iter()
        .map(|p| p.smoothing_length.max(1e-10))
        .collect();
    let rho: Vec<f64> = h_sml
        .iter()
        .zip(mass.iter())
        .map(|(&h, &m)| (m / (h * h * h)).max(1e-30))
        .collect();
    let maxwell: Vec<[[f64; 3]; 3]> = particles
        .iter()
        .map(|p| maxwell_stress(p.b_field))
        .collect();
    let is_gas: Vec<bool> = particles
        .iter()
        .map(|p| p.ptype == ParticleType::Gas)
        .collect();

    let updates: Vec<Option<Vec3>> = (0..n)
        .into_par_iter()
        .map(|i| {
            if !is_gas[i] {
                return None;
            }
            let rho_i2 = rho[i] * rho[i];
            let m_i = &maxwell[i];
            let mut acc = Vec3::zero();

            for j in 0..n {
                if j == i || !is_gas[j] {
                    continue;
                }
                let rho_j2 = rho[j] * rho[j];
                let m_j = &maxwell[j];

                let r_ij = Vec3 {
                    x: pos[j].x - pos[i].x,
                    y: pos[j].y - pos[i].y,
                    z: pos[j].z - pos[i].z,
                };
                let h_ij = 0.5 * (h_sml[i] + h_sml[j]);
                let grad_w = kernel_gradient(r_ij, h_ij);
                let gw = [grad_w.x, grad_w.y, grad_w.z];

                let mut a = [0.0_f64; 3];
                for k in 0..3 {
                    for (l, &gw_l) in gw.iter().enumerate() {
                        a[k] += (m_i[k][l] / rho_i2 + m_j[k][l] / rho_j2) * gw_l;
                    }
                }

                acc.x += mass[j] * a[0];
                acc.y += mass[j] * a[1];
                acc.z += mass[j] * a[2];
            }
            Some(acc)
        })
        .collect();

    for (p, update) in particles.iter_mut().zip(updates) {
        if let (true, Some(acc)) = (p.ptype == ParticleType::Gas, update) {
            p.velocity.x += acc.x * dt;
            p.velocity.y += acc.y * dt;
            p.velocity.z += acc.z * dt;
        }
    }
}

/// Aplica las fuerzas magnéticas (tensor de Maxwell SPH) a las partículas de gas (Phase 124).
///
/// Para cada par (i, j) de partículas de gas, acumula la aceleración magnética:
///
/// ```text
/// a_i += m_j (M_i/ρ_i² + M_j/ρ_j²) · ∇W_ij
/// ```
pub fn apply_magnetic_forces(particles: &mut [Particle], dt: f64) {
    #[cfg(feature = "rayon")]
    {
        apply_magnetic_forces_par(particles, dt);
    }

    #[cfg(not(feature = "rayon"))]
    {
        apply_magnetic_forces_impl(particles, dt);
    }
}

#[cfg(test)]
#[cfg_attr(feature = "rayon", allow(dead_code, unused_imports))]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn magnetic_pressure_zero_when_b_zero() {
        assert_abs_diff_eq!(magnetic_pressure(Vec3::zero()), 0.0, epsilon = 1e-15);
    }

    #[test]
    fn magnetic_pressure_unit_b_equals_one_half() {
        let pb = magnetic_pressure(Vec3::new(1.0, 0.0, 0.0));
        assert_abs_diff_eq!(pb, 0.5, epsilon = 1e-15);
    }

    #[test]
    fn magnetic_pressure_scales_with_b_squared() {
        let pb1 = magnetic_pressure(Vec3::new(1.0, 0.0, 0.0));
        let pb2 = magnetic_pressure(Vec3::new(2.0, 0.0, 0.0));
        assert_abs_diff_eq!(pb1 * 4.0, pb2, epsilon = 1e-15);
    }

    #[test]
    fn magnetic_pressure_sign_invariant() {
        let pb1 = magnetic_pressure(Vec3::new(1.0, 2.0, 3.0));
        let pb2 = magnetic_pressure(Vec3::new(-1.0, -2.0, -3.0));
        assert_abs_diff_eq!(pb1, pb2, epsilon = 1e-15);
    }

    #[test]
    fn magnetic_pressure_isotropic() {
        let pb = magnetic_pressure(Vec3::new(1.0, 2.0, 3.0));
        let expected = (1.0_f64 * 1.0 + 2.0 * 2.0 + 3.0 * 3.0) / (2.0 * MU0);
        assert_abs_diff_eq!(pb, expected, epsilon = 1e-15);
    }

    #[test]
    fn maxwell_stress_zero_when_b_zero() {
        let m = maxwell_stress(Vec3::zero());
        for row in &m {
            for value in row {
                assert_abs_diff_eq!(*value, 0.0, epsilon = 1e-15);
            }
        }
    }

    #[test]
    fn maxwell_stress_trace_equals_negative_pb() {
        let b = Vec3::new(3.0, 4.0, 0.0);
        let m = maxwell_stress(b);
        let trace = m[0][0] + m[1][1] + m[2][2];
        let pb = magnetic_pressure(b);
        assert_abs_diff_eq!(trace, -pb, epsilon = 1e-14);
    }

    #[test]
    fn maxwell_stress_is_symmetric() {
        let b = Vec3::new(1.0, 2.0, 3.0);
        let m = maxwell_stress(b);
        for (i, row) in m.iter().enumerate() {
            for (j, value) in row.iter().enumerate() {
                assert_abs_diff_eq!(*value, m[j][i], epsilon = 1e-15);
            }
        }
    }

    #[test]
    fn maxwell_stress_diagonal_for_b_along_x() {
        let b = Vec3::new(5.0, 0.0, 0.0);
        let m = maxwell_stress(b);
        let pb = magnetic_pressure(b);
        assert_abs_diff_eq!(m[0][0], b.x * b.x / MU0 - pb, epsilon = 1e-14);
        assert_abs_diff_eq!(m[1][1], -pb, epsilon = 1e-14);
        assert_abs_diff_eq!(m[2][2], -pb, epsilon = 1e-14);
        assert_abs_diff_eq!(m[0][1], 0.0, epsilon = 1e-15);
        assert_abs_diff_eq!(m[0][2], 0.0, epsilon = 1e-15);
        assert_abs_diff_eq!(m[1][2], 0.0, epsilon = 1e-15);
    }

    fn make_force_particles(n: usize, with_dm: bool) -> Vec<Particle> {
        (0..n)
            .map(|idx| {
                let t = idx as f64;
                let mut p = Particle::new_gas(
                    idx,
                    1.0 + 0.02 * t,
                    Vec3::new(0.16 * t, 0.05 * (0.7 * t).sin(), 0.03 * (1.3 * t).cos()),
                    Vec3::new(0.01 * t, -0.02 * t.sin(), 0.015 * t.cos()),
                    1.0,
                    0.42 + 0.003 * (idx % 7) as f64,
                );
                p.b_field = Vec3::new(
                    0.12 + 0.004 * t,
                    -0.08 + 0.003 * (0.5 * t).sin(),
                    0.05 + 0.002 * (0.25 * t).cos(),
                );
                if with_dm && matches!(idx, 2 | 9) {
                    Particle::new(idx, p.mass, p.position, Vec3::new(3.0, -2.0, 1.0))
                } else {
                    p
                }
            })
            .collect()
    }

    fn assert_velocities_close(actual: &[Particle], expected: &[Particle]) {
        for (a, e) in actual.iter().zip(expected.iter()) {
            // SIMD force batches use FMA and horizontal reductions; this tolerance covers
            // roundoff from equivalent accumulation while catching physical regressions.
            assert_abs_diff_eq!(a.velocity.x, e.velocity.x, epsilon = 1e-10);
            assert_abs_diff_eq!(a.velocity.y, e.velocity.y, epsilon = 1e-10);
            assert_abs_diff_eq!(a.velocity.z, e.velocity.z, epsilon = 1e-10);
        }
    }

    #[test]
    #[cfg(not(feature = "rayon"))]
    fn magnetic_forces_dispatch_matches_scalar_for_all_gas() {
        let mut scalar = make_force_particles(16, false);
        let mut dispatched = scalar.clone();

        apply_magnetic_forces_scalar(&mut scalar, 0.01);
        apply_magnetic_forces_impl(&mut dispatched, 0.01);

        assert_velocities_close(&dispatched, &scalar);
    }

    #[test]
    #[cfg(not(feature = "rayon"))]
    fn magnetic_forces_dispatch_matches_scalar_with_dark_matter() {
        let mut scalar = make_force_particles(16, true);
        let mut dispatched = scalar.clone();
        let dm_before: Vec<(usize, Vec3)> = dispatched
            .iter()
            .enumerate()
            .filter_map(|(idx, p)| {
                (p.ptype == ParticleType::DarkMatter).then_some((idx, p.velocity))
            })
            .collect();

        apply_magnetic_forces_scalar(&mut scalar, 0.008);
        apply_magnetic_forces_impl(&mut dispatched, 0.008);

        assert_velocities_close(&dispatched, &scalar);
        for (idx, v_before) in dm_before {
            assert_abs_diff_eq!(dispatched[idx].velocity.x, v_before.x, epsilon = 0.0);
            assert_abs_diff_eq!(dispatched[idx].velocity.y, v_before.y, epsilon = 0.0);
            assert_abs_diff_eq!(dispatched[idx].velocity.z, v_before.z, epsilon = 0.0);
        }
    }
}
