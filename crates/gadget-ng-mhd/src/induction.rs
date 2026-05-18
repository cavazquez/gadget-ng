//! Ecuación de inducción SPH: `dB/dt = ∇×(v×B)` (Phase 123).
//!
//! ## Formulación
//!
//! En la discretización SPH de Morris & Monaghan (1997), la ecuación de inducción
//! para la partícula `i` es:
//!
//! ```text
//! dB_i/dt = Σ_j (m_j/ρ_j) [(B_i·∇W_ij) v_ij - (v_ij·∇W_ij) B_i]
//! ```
//!
//! donde `v_ij = v_i - v_j` y `∇W_ij` es el gradiente del kernel SPH.
//!
//! Esta formulación preserva `∇·B = 0` mejor que la formulación no conservativa
//! y es simétrica respecto al intercambio i↔j.
//!
//! ## Referencia
//!
//! Morris & Monaghan (1997), J. Comput. Phys. 136, 41–60.
//! Price & Monaghan (2005), MNRAS 364, 384–406.

use crate::MU0;
use gadget_ng_core::{BFieldKind, MhdSection, Particle, ParticleType, Vec3};
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

/// Gradiente del kernel SPH cúbico (en 3D): `∇W(r, h)`.
///
/// Devuelve el gradiente evaluado en la dirección `r_ij = r_j - r_i`.
fn kernel_gradient(r_vec: Vec3, h: f64) -> Vec3 {
    let r2 = r_vec.x * r_vec.x + r_vec.y * r_vec.y + r_vec.z * r_vec.z;
    let r = r2.sqrt();
    if r < 1e-10 || h <= 0.0 {
        return Vec3::zero();
    }
    let q = r / h;
    // Derivada del kernel cúbico B-spline
    let dw_dq = if q < 1.0 {
        let norm = 8.0 / (std::f64::consts::PI * h * h * h);
        norm * (-6.0 * q + 9.0 * q * q) // d/dq [1 - 6q² + 6q³] / norm_h
    } else if q < 2.0 {
        let norm = 8.0 / (std::f64::consts::PI * h * h * h);
        norm * (-6.0 * (2.0 - q).powi(2)) / 4.0
    } else {
        0.0
    };
    let dw_dr = dw_dq / h;
    // ∇W = (dW/dr) * r_hat = (dW/dr) * r_vec/r
    Vec3 {
        x: dw_dr * r_vec.x / r,
        y: dw_dr * r_vec.y / r,
        z: dw_dr * r_vec.z / r,
    }
}

#[cfg(not(feature = "rayon"))]
fn advance_induction_impl(particles: &mut [Particle], dt: f64) {
    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    {
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx512f") {
            // SAFETY: AVX-512F availability was checked at runtime immediately above.
            unsafe {
                return advance_induction_avx512(particles, dt);
            }
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: AVX2+FMA availability was checked at runtime immediately above.
            unsafe {
                return advance_induction_avx2(particles, dt);
            }
        }
    }
    advance_induction_scalar(particles, dt);
}

#[cfg(not(feature = "rayon"))]
fn advance_induction_scalar(particles: &mut [Particle], dt: f64) {
    let n = particles.len();
    let mut db = vec![Vec3::zero(); n];

    for i in 0..n {
        if particles[i].ptype != ParticleType::Gas {
            continue;
        }
        let h_i = particles[i].smoothing_length.max(1e-10);
        let rho_i = (particles[i].mass / (h_i * h_i * h_i)).max(1e-30);
        let b_i = particles[i].b_field;
        let v_i = particles[i].velocity;
        let pos_i = particles[i].position;

        for j in 0..n {
            let contrib = induction_pair_contribution(particles, i, j, pos_i, v_i, b_i, h_i);
            db[i].x += contrib.x;
            db[i].y += contrib.y;
            db[i].z += contrib.z;
        }
        let _ = rho_i;
    }

    apply_induction_updates(particles, &db, dt);
}

#[cfg(not(feature = "rayon"))]
#[inline]
fn induction_pair_contribution(
    particles: &[Particle],
    i: usize,
    j: usize,
    pos_i: Vec3,
    v_i: Vec3,
    b_i: Vec3,
    h_i: f64,
) -> Vec3 {
    if j == i || particles[j].ptype != ParticleType::Gas {
        return Vec3::zero();
    }

    let h_j = particles[j].smoothing_length.max(1e-10);
    let rho_j = (particles[j].mass / (h_j * h_j * h_j)).max(1e-30);
    let v_j = particles[j].velocity;
    let b_j = particles[j].b_field;

    let r_ij = Vec3 {
        x: particles[j].position.x - pos_i.x,
        y: particles[j].position.y - pos_i.y,
        z: particles[j].position.z - pos_i.z,
    };

    let h_ij = 0.5 * (h_i + h_j);
    let grad_w = kernel_gradient(r_ij, h_ij);

    let v_ij = Vec3 {
        x: v_i.x - v_j.x,
        y: v_i.y - v_j.y,
        z: v_i.z - v_j.z,
    };
    let b_ij = Vec3 {
        x: b_i.x - b_j.x,
        y: b_i.y - b_j.y,
        z: b_i.z - b_j.z,
    };

    let b_dot_grad = b_ij.x * grad_w.x + b_ij.y * grad_w.y + b_ij.z * grad_w.z;
    let v_dot_grad = v_ij.x * grad_w.x + v_ij.y * grad_w.y + v_ij.z * grad_w.z;
    let factor = particles[j].mass / rho_j;

    Vec3 {
        x: factor * (b_dot_grad * v_ij.x - v_dot_grad * b_ij.x),
        y: factor * (b_dot_grad * v_ij.y - v_dot_grad * b_ij.y),
        z: factor * (b_dot_grad * v_ij.z - v_dot_grad * b_ij.z),
    }
}

#[cfg(all(
    not(feature = "rayon"),
    feature = "simd",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn advance_induction_avx2(particles: &mut [Particle], dt: f64) {
    let n = particles.len();
    let mut db = vec![Vec3::zero(); n];
    for i in 0..n {
        if particles[i].ptype != ParticleType::Gas {
            continue;
        }
        // SAFETY: caller reached this AVX2 function only after runtime AVX2+FMA dispatch.
        db[i] = unsafe { induction_sum_avx2(particles, i) };
    }
    apply_induction_updates(particles, &db, dt);
}

#[cfg(all(
    not(feature = "rayon"),
    feature = "simd",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
unsafe fn advance_induction_avx512(particles: &mut [Particle], dt: f64) {
    let n = particles.len();
    let mut db = vec![Vec3::zero(); n];
    for i in 0..n {
        if particles[i].ptype != ParticleType::Gas {
            continue;
        }
        // SAFETY: caller reached this AVX-512F function only after runtime AVX-512F dispatch.
        db[i] = unsafe { induction_sum_avx512(particles, i) };
    }
    apply_induction_updates(particles, &db, dt);
}

#[cfg(not(feature = "rayon"))]
fn apply_induction_updates(particles: &mut [Particle], db: &[Vec3], dt: f64) {
    for (p, db_i) in particles.iter_mut().zip(db.iter()) {
        if p.ptype == ParticleType::Gas {
            p.b_field.x += db_i.x * dt;
            p.b_field.y += db_i.y * dt;
            p.b_field.z += db_i.z * dt;
        }
    }
}

#[cfg(all(
    not(feature = "rayon"),
    feature = "simd",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn induction_sum_avx2(particles: &[Particle], i: usize) -> Vec3 {
    let lanes = 4;
    let chunks = particles.len() / lanes * lanes;
    let h_i = particles[i].smoothing_length.max(1e-10);
    let pos_i = particles[i].position;
    let v_i = particles[i].velocity;
    let b_i = particles[i].b_field;
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
                let contrib =
                    induction_pair_contribution(particles, i, j + lane, pos_i, v_i, b_i, h_i);
                sum_x = _mm256_add_pd(sum_x, _mm256_set_pd(0.0, 0.0, 0.0, contrib.x));
                sum_y = _mm256_add_pd(sum_y, _mm256_set_pd(0.0, 0.0, 0.0, contrib.y));
                sum_z = _mm256_add_pd(sum_z, _mm256_set_pd(0.0, 0.0, 0.0, contrib.z));
            }
            j += lanes;
            continue;
        }

        let (cx, cy, cz) = induction_batch_avx2(particles, j, pos_i, v_i, b_i, h_i);
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
    let mut db = Vec3::new(
        out_x.into_iter().sum(),
        out_y.into_iter().sum(),
        out_z.into_iter().sum(),
    );
    for j_tail in chunks..particles.len() {
        let contrib = induction_pair_contribution(particles, i, j_tail, pos_i, v_i, b_i, h_i);
        db.x += contrib.x;
        db.y += contrib.y;
        db.z += contrib.z;
    }
    db
}

#[cfg(all(
    not(feature = "rayon"),
    feature = "simd",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx2", enable = "fma")]
fn induction_batch_avx2(
    particles: &[Particle],
    j: usize,
    pos_i: Vec3,
    v_i: Vec3,
    b_i: Vec3,
    h_i: f64,
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

    let h_j = _mm256_max_pd(
        _mm256_set1_pd(1e-10),
        _mm256_set_pd(
            particles[j + 3].smoothing_length,
            particles[j + 2].smoothing_length,
            particles[j + 1].smoothing_length,
            particles[j].smoothing_length,
        ),
    );
    let h_ij = _mm256_mul_pd(_mm256_set1_pd(0.5), _mm256_add_pd(_mm256_set1_pd(h_i), h_j));
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
    let grad_x = _mm256_and_pd(_mm256_mul_pd(scale, dx), valid_r);
    let grad_y = _mm256_and_pd(_mm256_mul_pd(scale, dy), valid_r);
    let grad_z = _mm256_and_pd(_mm256_mul_pd(scale, dz), valid_r);

    let vx = _mm256_sub_pd(
        _mm256_set1_pd(v_i.x),
        _mm256_set_pd(
            particles[j + 3].velocity.x,
            particles[j + 2].velocity.x,
            particles[j + 1].velocity.x,
            particles[j].velocity.x,
        ),
    );
    let vy = _mm256_sub_pd(
        _mm256_set1_pd(v_i.y),
        _mm256_set_pd(
            particles[j + 3].velocity.y,
            particles[j + 2].velocity.y,
            particles[j + 1].velocity.y,
            particles[j].velocity.y,
        ),
    );
    let vz = _mm256_sub_pd(
        _mm256_set1_pd(v_i.z),
        _mm256_set_pd(
            particles[j + 3].velocity.z,
            particles[j + 2].velocity.z,
            particles[j + 1].velocity.z,
            particles[j].velocity.z,
        ),
    );
    let bx = _mm256_sub_pd(
        _mm256_set1_pd(b_i.x),
        _mm256_set_pd(
            particles[j + 3].b_field.x,
            particles[j + 2].b_field.x,
            particles[j + 1].b_field.x,
            particles[j].b_field.x,
        ),
    );
    let by = _mm256_sub_pd(
        _mm256_set1_pd(b_i.y),
        _mm256_set_pd(
            particles[j + 3].b_field.y,
            particles[j + 2].b_field.y,
            particles[j + 1].b_field.y,
            particles[j].b_field.y,
        ),
    );
    let bz = _mm256_sub_pd(
        _mm256_set1_pd(b_i.z),
        _mm256_set_pd(
            particles[j + 3].b_field.z,
            particles[j + 2].b_field.z,
            particles[j + 1].b_field.z,
            particles[j].b_field.z,
        ),
    );
    let b_dot_grad = _mm256_fmadd_pd(
        bx,
        grad_x,
        _mm256_fmadd_pd(by, grad_y, _mm256_mul_pd(bz, grad_z)),
    );
    let v_dot_grad = _mm256_fmadd_pd(
        vx,
        grad_x,
        _mm256_fmadd_pd(vy, grad_y, _mm256_mul_pd(vz, grad_z)),
    );
    let mass = _mm256_set_pd(
        particles[j + 3].mass,
        particles[j + 2].mass,
        particles[j + 1].mass,
        particles[j].mass,
    );
    let rho = _mm256_max_pd(
        _mm256_set1_pd(1e-30),
        _mm256_div_pd(mass, _mm256_mul_pd(h_j, _mm256_mul_pd(h_j, h_j))),
    );
    let factor = _mm256_div_pd(mass, rho);
    (
        _mm256_mul_pd(
            factor,
            _mm256_sub_pd(_mm256_mul_pd(b_dot_grad, vx), _mm256_mul_pd(v_dot_grad, bx)),
        ),
        _mm256_mul_pd(
            factor,
            _mm256_sub_pd(_mm256_mul_pd(b_dot_grad, vy), _mm256_mul_pd(v_dot_grad, by)),
        ),
        _mm256_mul_pd(
            factor,
            _mm256_sub_pd(_mm256_mul_pd(b_dot_grad, vz), _mm256_mul_pd(v_dot_grad, bz)),
        ),
    )
}

#[cfg(all(
    not(feature = "rayon"),
    feature = "simd",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
unsafe fn induction_sum_avx512(particles: &[Particle], i: usize) -> Vec3 {
    let lanes = 8;
    let chunks = particles.len() / lanes * lanes;
    let h_i = particles[i].smoothing_length.max(1e-10);
    let pos_i = particles[i].position;
    let v_i = particles[i].velocity;
    let b_i = particles[i].b_field;
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
                let contrib =
                    induction_pair_contribution(particles, i, j + lane, pos_i, v_i, b_i, h_i);
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

        let (cx, cy, cz) = induction_batch_avx512(particles, j, pos_i, v_i, b_i, h_i);
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
    let mut db = Vec3::new(
        out_x.into_iter().sum(),
        out_y.into_iter().sum(),
        out_z.into_iter().sum(),
    );
    for j_tail in chunks..particles.len() {
        let contrib = induction_pair_contribution(particles, i, j_tail, pos_i, v_i, b_i, h_i);
        db.x += contrib.x;
        db.y += contrib.y;
        db.z += contrib.z;
    }
    db
}

#[cfg(all(
    not(feature = "rayon"),
    feature = "simd",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
fn induction_batch_avx512(
    particles: &[Particle],
    j: usize,
    pos_i: Vec3,
    v_i: Vec3,
    b_i: Vec3,
    h_i: f64,
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
    let h_ij = _mm512_mul_pd(_mm512_set1_pd(0.5), _mm512_add_pd(_mm512_set1_pd(h_i), h_j));
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
    let grad_x = _mm512_maskz_mul_pd(valid_r, scale, dx);
    let grad_y = _mm512_maskz_mul_pd(valid_r, scale, dy);
    let grad_z = _mm512_maskz_mul_pd(valid_r, scale, dz);

    let vx = _mm512_sub_pd(
        _mm512_set1_pd(v_i.x),
        _mm512_set_pd(
            particles[j + 7].velocity.x,
            particles[j + 6].velocity.x,
            particles[j + 5].velocity.x,
            particles[j + 4].velocity.x,
            particles[j + 3].velocity.x,
            particles[j + 2].velocity.x,
            particles[j + 1].velocity.x,
            particles[j].velocity.x,
        ),
    );
    let vy = _mm512_sub_pd(
        _mm512_set1_pd(v_i.y),
        _mm512_set_pd(
            particles[j + 7].velocity.y,
            particles[j + 6].velocity.y,
            particles[j + 5].velocity.y,
            particles[j + 4].velocity.y,
            particles[j + 3].velocity.y,
            particles[j + 2].velocity.y,
            particles[j + 1].velocity.y,
            particles[j].velocity.y,
        ),
    );
    let vz = _mm512_sub_pd(
        _mm512_set1_pd(v_i.z),
        _mm512_set_pd(
            particles[j + 7].velocity.z,
            particles[j + 6].velocity.z,
            particles[j + 5].velocity.z,
            particles[j + 4].velocity.z,
            particles[j + 3].velocity.z,
            particles[j + 2].velocity.z,
            particles[j + 1].velocity.z,
            particles[j].velocity.z,
        ),
    );
    let bx = _mm512_sub_pd(
        _mm512_set1_pd(b_i.x),
        _mm512_set_pd(
            particles[j + 7].b_field.x,
            particles[j + 6].b_field.x,
            particles[j + 5].b_field.x,
            particles[j + 4].b_field.x,
            particles[j + 3].b_field.x,
            particles[j + 2].b_field.x,
            particles[j + 1].b_field.x,
            particles[j].b_field.x,
        ),
    );
    let by = _mm512_sub_pd(
        _mm512_set1_pd(b_i.y),
        _mm512_set_pd(
            particles[j + 7].b_field.y,
            particles[j + 6].b_field.y,
            particles[j + 5].b_field.y,
            particles[j + 4].b_field.y,
            particles[j + 3].b_field.y,
            particles[j + 2].b_field.y,
            particles[j + 1].b_field.y,
            particles[j].b_field.y,
        ),
    );
    let bz = _mm512_sub_pd(
        _mm512_set1_pd(b_i.z),
        _mm512_set_pd(
            particles[j + 7].b_field.z,
            particles[j + 6].b_field.z,
            particles[j + 5].b_field.z,
            particles[j + 4].b_field.z,
            particles[j + 3].b_field.z,
            particles[j + 2].b_field.z,
            particles[j + 1].b_field.z,
            particles[j].b_field.z,
        ),
    );
    let b_dot_grad = _mm512_fmadd_pd(
        bx,
        grad_x,
        _mm512_fmadd_pd(by, grad_y, _mm512_mul_pd(bz, grad_z)),
    );
    let v_dot_grad = _mm512_fmadd_pd(
        vx,
        grad_x,
        _mm512_fmadd_pd(vy, grad_y, _mm512_mul_pd(vz, grad_z)),
    );
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
    let rho = _mm512_max_pd(
        _mm512_set1_pd(1e-30),
        _mm512_div_pd(mass, _mm512_mul_pd(h_j, _mm512_mul_pd(h_j, h_j))),
    );
    let factor = _mm512_div_pd(mass, rho);
    (
        _mm512_mul_pd(
            factor,
            _mm512_sub_pd(_mm512_mul_pd(b_dot_grad, vx), _mm512_mul_pd(v_dot_grad, bx)),
        ),
        _mm512_mul_pd(
            factor,
            _mm512_sub_pd(_mm512_mul_pd(b_dot_grad, vy), _mm512_mul_pd(v_dot_grad, by)),
        ),
        _mm512_mul_pd(
            factor,
            _mm512_sub_pd(_mm512_mul_pd(b_dot_grad, vz), _mm512_mul_pd(v_dot_grad, bz)),
        ),
    )
}

#[cfg(feature = "rayon")]
fn advance_induction_par(particles: &mut [Particle], dt: f64) {
    let n = particles.len();
    let pos: Vec<Vec3> = particles.iter().map(|p| p.position).collect();
    let vel: Vec<Vec3> = particles.iter().map(|p| p.velocity).collect();
    let b_field: Vec<Vec3> = particles.iter().map(|p| p.b_field).collect();
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
            let h_i = h_sml[i];
            let b_i = b_field[i];
            let v_i = vel[i];
            let mut db_i = Vec3::zero();

            for j in 0..n {
                if j == i || !is_gas[j] {
                    continue;
                }
                let r_ij = Vec3 {
                    x: pos[j].x - pos[i].x,
                    y: pos[j].y - pos[i].y,
                    z: pos[j].z - pos[i].z,
                };
                let h_ij = 0.5 * (h_i + h_sml[j]);
                let grad_w = kernel_gradient(r_ij, h_ij);

                let v_ij = Vec3 {
                    x: v_i.x - vel[j].x,
                    y: v_i.y - vel[j].y,
                    z: v_i.z - vel[j].z,
                };
                let b_ij = Vec3 {
                    x: b_i.x - b_field[j].x,
                    y: b_i.y - b_field[j].y,
                    z: b_i.z - b_field[j].z,
                };

                let b_dot_grad = b_ij.x * grad_w.x + b_ij.y * grad_w.y + b_ij.z * grad_w.z;
                let v_dot_grad = v_ij.x * grad_w.x + v_ij.y * grad_w.y + v_ij.z * grad_w.z;
                let factor = mass[j] / rho[j];

                db_i.x += factor * (b_dot_grad * v_ij.x - v_dot_grad * b_ij.x);
                db_i.y += factor * (b_dot_grad * v_ij.y - v_dot_grad * b_ij.y);
                db_i.z += factor * (b_dot_grad * v_ij.z - v_dot_grad * b_ij.z);
            }
            Some(db_i)
        })
        .collect();

    for (p, update) in particles.iter_mut().zip(updates) {
        if let (true, Some(db)) = (p.ptype == ParticleType::Gas, update) {
            p.b_field.x += db.x * dt;
            p.b_field.y += db.y * dt;
            p.b_field.z += db.z * dt;
        }
    }
}

pub fn advance_induction(particles: &mut [Particle], dt: f64) {
    #[cfg(feature = "rayon")]
    {
        advance_induction_par(particles, dt);
    }

    #[cfg(not(feature = "rayon"))]
    {
        advance_induction_impl(particles, dt);
    }
}

/// Inicializa el campo magnético de las partículas de gas según `cfg.b0_kind` (Phase 127).
///
/// - `Uniform`:  B = b0_uniform para todas las partículas de gas.
/// - `Random`:   B = amplitud aleatoria con |B| ≈ |b0_uniform| (usando global_id como semilla).
/// - `Spiral`:   B = B0 × (sin(2πy/L), cos(2πx/L), 0).
/// - `None`:     no-op.
///
/// `box_size` se usa para normalizar la posición en el modo espiral.
pub fn init_b_field(particles: &mut [Particle], cfg: &MhdSection, box_size: f64) {
    let b0 = Vec3::new(cfg.b0_uniform[0], cfg.b0_uniform[1], cfg.b0_uniform[2]);
    let b_mag = (b0.x * b0.x + b0.y * b0.y + b0.z * b0.z).sqrt();
    let l = box_size.max(1e-10);

    for p in particles.iter_mut() {
        if p.ptype != ParticleType::Gas {
            continue;
        }
        p.b_field = match cfg.b0_kind {
            BFieldKind::None => Vec3::zero(),
            BFieldKind::Uniform => b0,
            BFieldKind::Random => {
                let seed = p.global_id as u64;
                let rx = lcg_uniform(seed);
                let ry = lcg_uniform(seed.wrapping_add(1));
                let rz = lcg_uniform(seed.wrapping_add(2));
                let norm = (rx * rx + ry * ry + rz * rz).sqrt().max(1e-10);
                Vec3::new(rx / norm * b_mag, ry / norm * b_mag, rz / norm * b_mag)
            }
            BFieldKind::Spiral => {
                let x = p.position.x / l;
                let y = p.position.y / l;
                let two_pi = 2.0 * std::f64::consts::PI;
                Vec3::new(b_mag * (two_pi * y).sin(), b_mag * (two_pi * x).cos(), 0.0)
            }
        };
    }
}

/// Genera un número pseudo-aleatorio ∈ [−1, 1] a partir de una semilla u64.
#[inline]
fn lcg_uniform(seed: u64) -> f64 {
    let s = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let bits = (s >> 33) as u32;
    (bits as f64 / u32::MAX as f64) * 2.0 - 1.0
}

/// Calcula el paso de tiempo máximo permitido por el criterio CFL de Alfvén (Phase 127).
///
/// `dt_A = cfl × min_i(h_i) / max_i(v_{A,i})`
///
/// donde `v_{A,i} = |B_i| / sqrt(μ₀ ρ_i)`.
///
/// Retorna `f64::INFINITY` si no hay partículas de gas o si todos los B son cero.
pub fn alfven_dt(particles: &[Particle], cfl: f64) -> f64 {
    let mut h_min = f64::INFINITY;
    let mut v_a_max = 0.0_f64;

    for p in particles.iter() {
        if p.ptype != ParticleType::Gas {
            continue;
        }
        let h = p.smoothing_length.max(1e-10);
        let rho = (p.mass / (h * h * h)).max(1e-30);
        let b2 = p.b_field.x * p.b_field.x + p.b_field.y * p.b_field.y + p.b_field.z * p.b_field.z;
        let v_a = (b2 / (MU0 * rho)).sqrt();

        h_min = h_min.min(h);
        v_a_max = v_a_max.max(v_a);
    }

    if v_a_max < 1e-30 || !h_min.is_finite() {
        return f64::INFINITY;
    }
    cfl * h_min / v_a_max
}

#[cfg(not(feature = "rayon"))]
fn apply_artificial_resistivity_impl(particles: &mut [Particle], alpha_b: f64, dt: f64) {
    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    {
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx512f") {
            // SAFETY: AVX-512F availability was checked at runtime immediately above.
            unsafe {
                return apply_artificial_resistivity_avx512(particles, alpha_b, dt);
            }
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: AVX2+FMA availability was checked at runtime immediately above.
            unsafe {
                return apply_artificial_resistivity_avx2(particles, alpha_b, dt);
            }
        }
    }
    apply_artificial_resistivity_scalar(particles, alpha_b, dt);
}

#[cfg(not(feature = "rayon"))]
fn apply_artificial_resistivity_scalar(particles: &mut [Particle], alpha_b: f64, dt: f64) {
    if alpha_b <= 0.0 {
        return;
    }
    let n = particles.len();
    if n == 0 {
        return;
    }

    let mut db = vec![Vec3::zero(); n];

    for i in 0..n {
        if particles[i].ptype != ParticleType::Gas {
            continue;
        }
        let h_i = particles[i].smoothing_length.max(1e-10);
        let pos_i = particles[i].position;
        let b_i = particles[i].b_field;
        let vel_i = particles[i].velocity;

        for j in 0..n {
            let contrib =
                resistivity_pair_contribution(particles, i, j, pos_i, vel_i, b_i, h_i, alpha_b);
            db[i].x += contrib.x;
            db[i].y += contrib.y;
            db[i].z += contrib.z;
        }
    }

    apply_induction_updates(particles, &db, dt);
}

#[cfg(not(feature = "rayon"))]
#[expect(
    clippy::too_many_arguments,
    reason = "hot pair helper keeps scalar context explicit and allocation-free"
)]
#[inline]
fn resistivity_pair_contribution(
    particles: &[Particle],
    i: usize,
    j: usize,
    pos_i: Vec3,
    vel_i: Vec3,
    b_i: Vec3,
    h_i: f64,
    alpha_b: f64,
) -> Vec3 {
    if i == j || particles[j].ptype != ParticleType::Gas {
        return Vec3::zero();
    }

    let dx = particles[j].position.x - pos_i.x;
    let dy = particles[j].position.y - pos_i.y;
    let dz = particles[j].position.z - pos_i.z;
    let r2 = dx * dx + dy * dy + dz * dz;
    let r = r2.sqrt();
    if r < 1e-14 {
        return Vec3::zero();
    }

    let h_j = particles[j].smoothing_length.max(1e-10);
    let h_avg = 0.5 * (h_i + h_j);
    if r > 2.0 * h_avg {
        return Vec3::zero();
    }

    let dvx = particles[j].velocity.x - vel_i.x;
    let dvy = particles[j].velocity.y - vel_i.y;
    let dvz = particles[j].velocity.z - vel_i.z;
    let v_sig = (dvx * dvx + dvy * dvy + dvz * dvz).sqrt();
    let eta_art = alpha_b * h_i * v_sig;

    let q = r / h_avg;
    let dw_dr = if q <= 1.0 {
        -3.0 * q * (2.0 - q) / (h_avg.powi(4))
    } else if q <= 2.0 {
        -3.0 * (2.0 - q).powi(2) / (2.0 * h_avg.powi(4))
    } else {
        0.0
    };
    let grad_w_mag = dw_dr.abs();

    let rho_j = (particles[j].mass / (h_j * h_j * h_j)).max(1e-30);
    let factor = eta_art * particles[j].mass / rho_j * 2.0 * grad_w_mag / r;

    Vec3 {
        x: factor * (particles[j].b_field.x - b_i.x),
        y: factor * (particles[j].b_field.y - b_i.y),
        z: factor * (particles[j].b_field.z - b_i.z),
    }
}

#[cfg(all(
    not(feature = "rayon"),
    feature = "simd",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn apply_artificial_resistivity_avx2(particles: &mut [Particle], alpha_b: f64, dt: f64) {
    if alpha_b <= 0.0 {
        return;
    }
    let n = particles.len();
    let mut db = vec![Vec3::zero(); n];
    for i in 0..n {
        if particles[i].ptype != ParticleType::Gas {
            continue;
        }
        // SAFETY: caller reached this AVX2 function only after runtime AVX2+FMA dispatch.
        db[i] = unsafe { resistivity_sum_avx2(particles, i, alpha_b) };
    }
    apply_induction_updates(particles, &db, dt);
}

#[cfg(all(
    not(feature = "rayon"),
    feature = "simd",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
unsafe fn apply_artificial_resistivity_avx512(particles: &mut [Particle], alpha_b: f64, dt: f64) {
    if alpha_b <= 0.0 {
        return;
    }
    let n = particles.len();
    let mut db = vec![Vec3::zero(); n];
    for i in 0..n {
        if particles[i].ptype != ParticleType::Gas {
            continue;
        }
        // SAFETY: caller reached this AVX-512F function only after runtime AVX-512F dispatch.
        db[i] = unsafe { resistivity_sum_avx512(particles, i, alpha_b) };
    }
    apply_induction_updates(particles, &db, dt);
}

#[cfg(all(
    not(feature = "rayon"),
    feature = "simd",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn resistivity_sum_avx2(particles: &[Particle], i: usize, alpha_b: f64) -> Vec3 {
    let lanes = 4;
    let chunks = particles.len() / lanes * lanes;
    let h_i = particles[i].smoothing_length.max(1e-10);
    let pos_i = particles[i].position;
    let vel_i = particles[i].velocity;
    let b_i = particles[i].b_field;
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
                let contrib = resistivity_pair_contribution(
                    particles,
                    i,
                    j + lane,
                    pos_i,
                    vel_i,
                    b_i,
                    h_i,
                    alpha_b,
                );
                sum_x = _mm256_add_pd(sum_x, _mm256_set_pd(0.0, 0.0, 0.0, contrib.x));
                sum_y = _mm256_add_pd(sum_y, _mm256_set_pd(0.0, 0.0, 0.0, contrib.y));
                sum_z = _mm256_add_pd(sum_z, _mm256_set_pd(0.0, 0.0, 0.0, contrib.z));
            }
            j += lanes;
            continue;
        }

        let (cx, cy, cz) = resistivity_batch_avx2(particles, j, pos_i, vel_i, b_i, h_i, alpha_b);
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
    let mut db = Vec3::new(
        out_x.into_iter().sum(),
        out_y.into_iter().sum(),
        out_z.into_iter().sum(),
    );
    for j_tail in chunks..particles.len() {
        let contrib =
            resistivity_pair_contribution(particles, i, j_tail, pos_i, vel_i, b_i, h_i, alpha_b);
        db.x += contrib.x;
        db.y += contrib.y;
        db.z += contrib.z;
    }
    db
}

#[cfg(all(
    not(feature = "rayon"),
    feature = "simd",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx2", enable = "fma")]
fn resistivity_batch_avx2(
    particles: &[Particle],
    j: usize,
    pos_i: Vec3,
    vel_i: Vec3,
    b_i: Vec3,
    h_i: f64,
    alpha_b: f64,
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

    let h_j = _mm256_max_pd(
        _mm256_set_pd(
            particles[j + 3].smoothing_length,
            particles[j + 2].smoothing_length,
            particles[j + 1].smoothing_length,
            particles[j].smoothing_length,
        ),
        _mm256_set1_pd(1e-10),
    );
    let h_avg = _mm256_mul_pd(_mm256_add_pd(_mm256_set1_pd(h_i), h_j), _mm256_set1_pd(0.5));
    let q = _mm256_div_pd(r, h_avg);
    let q_le_1 = _mm256_cmp_pd(q, _mm256_set1_pd(1.0), _CMP_LE_OQ);
    let q_le_2 = _mm256_cmp_pd(q, _mm256_set1_pd(2.0), _CMP_LE_OQ);
    let h2 = _mm256_mul_pd(h_avg, h_avg);
    let h4 = _mm256_mul_pd(h2, h2);
    let dw_inner = _mm256_div_pd(
        _mm256_mul_pd(
            _mm256_set1_pd(-3.0),
            _mm256_mul_pd(q, _mm256_sub_pd(_mm256_set1_pd(2.0), q)),
        ),
        h4,
    );
    let two_minus_q = _mm256_sub_pd(_mm256_set1_pd(2.0), q);
    let dw_outer = _mm256_div_pd(
        _mm256_mul_pd(
            _mm256_set1_pd(-1.5),
            _mm256_mul_pd(two_minus_q, two_minus_q),
        ),
        h4,
    );
    let dw = _mm256_or_pd(
        _mm256_and_pd(q_le_1, dw_inner),
        _mm256_andnot_pd(q_le_1, _mm256_and_pd(q_le_2, dw_outer)),
    );
    let grad_w_mag = _mm256_andnot_pd(_mm256_set1_pd(-0.0), dw);

    let vx = _mm256_set_pd(
        particles[j + 3].velocity.x,
        particles[j + 2].velocity.x,
        particles[j + 1].velocity.x,
        particles[j].velocity.x,
    );
    let vy = _mm256_set_pd(
        particles[j + 3].velocity.y,
        particles[j + 2].velocity.y,
        particles[j + 1].velocity.y,
        particles[j].velocity.y,
    );
    let vz = _mm256_set_pd(
        particles[j + 3].velocity.z,
        particles[j + 2].velocity.z,
        particles[j + 1].velocity.z,
        particles[j].velocity.z,
    );
    let dvx = _mm256_sub_pd(vx, _mm256_set1_pd(vel_i.x));
    let dvy = _mm256_sub_pd(vy, _mm256_set1_pd(vel_i.y));
    let dvz = _mm256_sub_pd(vz, _mm256_set1_pd(vel_i.z));
    let v_sig = _mm256_sqrt_pd(_mm256_fmadd_pd(
        dvx,
        dvx,
        _mm256_fmadd_pd(dvy, dvy, _mm256_mul_pd(dvz, dvz)),
    ));
    let eta_art = _mm256_mul_pd(_mm256_set1_pd(alpha_b * h_i), v_sig);
    let rho_j = _mm256_max_pd(
        _mm256_div_pd(
            _mm256_set_pd(
                particles[j + 3].mass,
                particles[j + 2].mass,
                particles[j + 1].mass,
                particles[j].mass,
            ),
            _mm256_mul_pd(_mm256_mul_pd(h_j, h_j), h_j),
        ),
        _mm256_set1_pd(1e-30),
    );
    let mass_over_rho = _mm256_div_pd(
        _mm256_set_pd(
            particles[j + 3].mass,
            particles[j + 2].mass,
            particles[j + 1].mass,
            particles[j].mass,
        ),
        rho_j,
    );
    let r_safe = _mm256_max_pd(r, _mm256_set1_pd(1e-14));
    let factor = _mm256_mul_pd(
        _mm256_mul_pd(eta_art, mass_over_rho),
        _mm256_div_pd(_mm256_mul_pd(_mm256_set1_pd(2.0), grad_w_mag), r_safe),
    );
    let valid = _mm256_and_pd(
        _mm256_cmp_pd(r, _mm256_set1_pd(1e-14), _CMP_GE_OQ),
        _mm256_cmp_pd(r, _mm256_mul_pd(_mm256_set1_pd(2.0), h_avg), _CMP_LE_OQ),
    );
    let factor = _mm256_and_pd(factor, valid);
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
    (
        _mm256_mul_pd(factor, _mm256_sub_pd(bx, _mm256_set1_pd(b_i.x))),
        _mm256_mul_pd(factor, _mm256_sub_pd(by, _mm256_set1_pd(b_i.y))),
        _mm256_mul_pd(factor, _mm256_sub_pd(bz, _mm256_set1_pd(b_i.z))),
    )
}

#[cfg(all(
    not(feature = "rayon"),
    feature = "simd",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
unsafe fn resistivity_sum_avx512(particles: &[Particle], i: usize, alpha_b: f64) -> Vec3 {
    let lanes = 8;
    let chunks = particles.len() / lanes * lanes;
    let h_i = particles[i].smoothing_length.max(1e-10);
    let pos_i = particles[i].position;
    let vel_i = particles[i].velocity;
    let b_i = particles[i].b_field;
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
                let contrib = resistivity_pair_contribution(
                    particles,
                    i,
                    j + lane,
                    pos_i,
                    vel_i,
                    b_i,
                    h_i,
                    alpha_b,
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

        let (cx, cy, cz) = resistivity_batch_avx512(particles, j, pos_i, vel_i, b_i, h_i, alpha_b);
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
    let mut db = Vec3::new(
        out_x.into_iter().sum(),
        out_y.into_iter().sum(),
        out_z.into_iter().sum(),
    );
    for j_tail in chunks..particles.len() {
        let contrib =
            resistivity_pair_contribution(particles, i, j_tail, pos_i, vel_i, b_i, h_i, alpha_b);
        db.x += contrib.x;
        db.y += contrib.y;
        db.z += contrib.z;
    }
    db
}

#[cfg(all(
    not(feature = "rayon"),
    feature = "simd",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
fn resistivity_batch_avx512(
    particles: &[Particle],
    j: usize,
    pos_i: Vec3,
    vel_i: Vec3,
    b_i: Vec3,
    h_i: f64,
    alpha_b: f64,
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

    let h_j = _mm512_max_pd(
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
        _mm512_set1_pd(1e-10),
    );
    let h_avg = _mm512_mul_pd(_mm512_add_pd(_mm512_set1_pd(h_i), h_j), _mm512_set1_pd(0.5));
    let q = _mm512_div_pd(r, h_avg);
    let q_le_1 = _mm512_cmp_pd_mask(q, _mm512_set1_pd(1.0), _CMP_LE_OQ);
    let q_le_2 = _mm512_cmp_pd_mask(q, _mm512_set1_pd(2.0), _CMP_LE_OQ);
    let h2 = _mm512_mul_pd(h_avg, h_avg);
    let h4 = _mm512_mul_pd(h2, h2);
    let dw_inner = _mm512_div_pd(
        _mm512_mul_pd(
            _mm512_set1_pd(-3.0),
            _mm512_mul_pd(q, _mm512_sub_pd(_mm512_set1_pd(2.0), q)),
        ),
        h4,
    );
    let two_minus_q = _mm512_sub_pd(_mm512_set1_pd(2.0), q);
    let dw_outer = _mm512_div_pd(
        _mm512_mul_pd(
            _mm512_set1_pd(-1.5),
            _mm512_mul_pd(two_minus_q, two_minus_q),
        ),
        h4,
    );
    let dw = _mm512_mask_blend_pd(q_le_1, _mm512_maskz_mov_pd(q_le_2, dw_outer), dw_inner);
    let grad_w_mag = _mm512_sub_pd(_mm512_setzero_pd(), dw);

    let vx = _mm512_set_pd(
        particles[j + 7].velocity.x,
        particles[j + 6].velocity.x,
        particles[j + 5].velocity.x,
        particles[j + 4].velocity.x,
        particles[j + 3].velocity.x,
        particles[j + 2].velocity.x,
        particles[j + 1].velocity.x,
        particles[j].velocity.x,
    );
    let vy = _mm512_set_pd(
        particles[j + 7].velocity.y,
        particles[j + 6].velocity.y,
        particles[j + 5].velocity.y,
        particles[j + 4].velocity.y,
        particles[j + 3].velocity.y,
        particles[j + 2].velocity.y,
        particles[j + 1].velocity.y,
        particles[j].velocity.y,
    );
    let vz = _mm512_set_pd(
        particles[j + 7].velocity.z,
        particles[j + 6].velocity.z,
        particles[j + 5].velocity.z,
        particles[j + 4].velocity.z,
        particles[j + 3].velocity.z,
        particles[j + 2].velocity.z,
        particles[j + 1].velocity.z,
        particles[j].velocity.z,
    );
    let dvx = _mm512_sub_pd(vx, _mm512_set1_pd(vel_i.x));
    let dvy = _mm512_sub_pd(vy, _mm512_set1_pd(vel_i.y));
    let dvz = _mm512_sub_pd(vz, _mm512_set1_pd(vel_i.z));
    let v_sig = _mm512_sqrt_pd(_mm512_fmadd_pd(
        dvx,
        dvx,
        _mm512_fmadd_pd(dvy, dvy, _mm512_mul_pd(dvz, dvz)),
    ));
    let eta_art = _mm512_mul_pd(_mm512_set1_pd(alpha_b * h_i), v_sig);
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
    let rho_j = _mm512_max_pd(
        _mm512_div_pd(mass, _mm512_mul_pd(_mm512_mul_pd(h_j, h_j), h_j)),
        _mm512_set1_pd(1e-30),
    );
    let r_safe = _mm512_max_pd(r, _mm512_set1_pd(1e-14));
    let factor = _mm512_mul_pd(
        _mm512_mul_pd(eta_art, _mm512_div_pd(mass, rho_j)),
        _mm512_div_pd(_mm512_mul_pd(_mm512_set1_pd(2.0), grad_w_mag), r_safe),
    );
    let valid = _mm512_cmp_pd_mask(r, _mm512_set1_pd(1e-14), _CMP_GE_OQ)
        & _mm512_cmp_pd_mask(r, _mm512_mul_pd(_mm512_set1_pd(2.0), h_avg), _CMP_LE_OQ);
    let factor = _mm512_maskz_mov_pd(valid, factor);
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
    (
        _mm512_mul_pd(factor, _mm512_sub_pd(bx, _mm512_set1_pd(b_i.x))),
        _mm512_mul_pd(factor, _mm512_sub_pd(by, _mm512_set1_pd(b_i.y))),
        _mm512_mul_pd(factor, _mm512_sub_pd(bz, _mm512_set1_pd(b_i.z))),
    )
}

#[cfg(feature = "rayon")]
fn apply_artificial_resistivity_par(particles: &mut [Particle], alpha_b: f64, dt: f64) {
    if alpha_b <= 0.0 {
        return;
    }
    let n = particles.len();
    if n == 0 {
        return;
    }

    let pos: Vec<Vec3> = particles.iter().map(|p| p.position).collect();
    let vel: Vec<Vec3> = particles.iter().map(|p| p.velocity).collect();
    let b_field: Vec<Vec3> = particles.iter().map(|p| p.b_field).collect();
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
            let h_i = h_sml[i];
            let b_i = b_field[i];
            let vel_i = vel[i];
            let mut db_i = Vec3::zero();

            for j in 0..n {
                if j == i || !is_gas[j] {
                    continue;
                }
                let r_ij = Vec3 {
                    x: pos[j].x - pos[i].x,
                    y: pos[j].y - pos[i].y,
                    z: pos[j].z - pos[i].z,
                };
                let r = (r_ij.x * r_ij.x + r_ij.y * r_ij.y + r_ij.z * r_ij.z).sqrt();
                if r < 1e-14 {
                    continue;
                }
                let h_avg = 0.5 * (h_i + h_sml[j]);
                if r > 2.0 * h_avg {
                    continue;
                }

                let dv = Vec3 {
                    x: vel[j].x - vel_i.x,
                    y: vel[j].y - vel_i.y,
                    z: vel[j].z - vel_i.z,
                };
                let v_sig = (dv.x * dv.x + dv.y * dv.y + dv.z * dv.z).sqrt();
                let eta_art = alpha_b * h_i * v_sig;

                let q = r / h_avg;
                let dw_dr = if q <= 1.0 {
                    -3.0 * q * (2.0 - q) / (h_avg.powi(4))
                } else if q <= 2.0 {
                    -3.0 * (2.0 - q).powi(2) / (2.0 * h_avg.powi(4))
                } else {
                    0.0
                };
                let grad_w_mag = dw_dr.abs();

                let factor = eta_art * mass[j] / rho[j] * 2.0 * grad_w_mag / r;

                db_i.x += factor * (b_field[j].x - b_i.x);
                db_i.y += factor * (b_field[j].y - b_i.y);
                db_i.z += factor * (b_field[j].z - b_i.z);
            }
            Some(db_i)
        })
        .collect();

    for (p, update) in particles.iter_mut().zip(updates) {
        if let (true, Some(db)) = (p.ptype == ParticleType::Gas, update) {
            p.b_field.x += db.x * dt;
            p.b_field.y += db.y * dt;
            p.b_field.z += db.z * dt;
        }
    }
}

/// Aplica resistividad numérica artificial al campo magnético (Phase 135).
///
/// Suaviza las discontinuidades de B usando el esquema de Price (2008):
///
/// ```text
/// (∂B_i/∂t)_η = η_art × Σ_j m_j/ρ_j × (B_j − B_i) × 2|∇W_ij|/|r_ij|
/// ```
///
/// donde `η_art = alpha_b × h_i × v_sig` y `v_sig = |v_ij|` es la señal de velocidad relativa.
///
/// Con `alpha_b = 0.0` es un no-op.
pub fn apply_artificial_resistivity(particles: &mut [Particle], alpha_b: f64, dt: f64) {
    #[cfg(feature = "rayon")]
    {
        apply_artificial_resistivity_par(particles, alpha_b, dt);
    }

    #[cfg(not(feature = "rayon"))]
    {
        apply_artificial_resistivity_impl(particles, alpha_b, dt);
    }
}

#[cfg(test)]
#[cfg_attr(feature = "rayon", allow(dead_code, unused_imports))]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn make_induction_particles(n: usize, with_dm: bool) -> Vec<Particle> {
        (0..n)
            .map(|idx| {
                let t = idx as f64;
                let mut p = Particle::new_gas(
                    idx,
                    1.0 + 0.03 * t,
                    Vec3::new(0.19 * t, 0.07 * (t * 1.7).sin(), 0.05 * (t * 0.9).cos()),
                    Vec3::new(0.02 * t.cos(), -0.015 * t.sin(), 0.01 * (t + 1.0)),
                    0.8 + 0.01 * t,
                    0.45 + 0.004 * (idx % 5) as f64,
                );
                p.b_field = Vec3::new(
                    0.08 + 0.003 * t,
                    -0.04 + 0.002 * (t * 0.5).cos(),
                    0.02 + 0.001 * (t * 0.25).sin(),
                );
                if with_dm && matches!(idx, 3 | 11) {
                    let mut dm = Particle::new(idx, p.mass, p.position, Vec3::new(-0.2, 0.1, 0.05));
                    dm.b_field = Vec3::new(9.0 + t, -3.0, 2.0);
                    dm
                } else {
                    p
                }
            })
            .collect()
    }

    fn assert_b_fields_close(actual: &[Particle], expected: &[Particle]) {
        for (a, e) in actual.iter().zip(expected.iter()) {
            // SIMD lanes use FMA and reduce batches horizontally, so the exact last bits can
            // differ from scalar accumulation while preserving the induction update numerically.
            assert_abs_diff_eq!(a.b_field.x, e.b_field.x, epsilon = 1e-10);
            assert_abs_diff_eq!(a.b_field.y, e.b_field.y, epsilon = 1e-10);
            assert_abs_diff_eq!(a.b_field.z, e.b_field.z, epsilon = 1e-10);
        }
    }

    #[test]
    #[cfg(not(feature = "rayon"))]
    fn induction_dispatch_matches_scalar_for_all_gas() {
        let mut scalar = make_induction_particles(16, false);
        let mut dispatched = scalar.clone();

        advance_induction_scalar(&mut scalar, 0.025);
        advance_induction_impl(&mut dispatched, 0.025);

        assert_b_fields_close(&dispatched, &scalar);
    }

    #[test]
    #[cfg(not(feature = "rayon"))]
    fn induction_dispatch_matches_scalar_with_dark_matter() {
        let mut scalar = make_induction_particles(16, true);
        let mut dispatched = scalar.clone();
        let dm_before: Vec<(usize, Vec3)> = dispatched
            .iter()
            .enumerate()
            .filter_map(|(idx, p)| {
                (p.ptype == ParticleType::DarkMatter).then_some((idx, p.b_field))
            })
            .collect();

        advance_induction_scalar(&mut scalar, 0.018);
        advance_induction_impl(&mut dispatched, 0.018);

        assert_b_fields_close(&dispatched, &scalar);
        for (idx, b_before) in dm_before {
            assert_abs_diff_eq!(dispatched[idx].b_field.x, b_before.x, epsilon = 0.0);
            assert_abs_diff_eq!(dispatched[idx].b_field.y, b_before.y, epsilon = 0.0);
            assert_abs_diff_eq!(dispatched[idx].b_field.z, b_before.z, epsilon = 0.0);
        }
    }

    #[test]
    #[cfg(not(feature = "rayon"))]
    fn resistivity_dispatch_matches_scalar_for_all_gas() {
        let mut scalar = make_induction_particles(16, false);
        let mut dispatched = scalar.clone();

        apply_artificial_resistivity_scalar(&mut scalar, 0.35, 0.02);
        apply_artificial_resistivity_impl(&mut dispatched, 0.35, 0.02);

        assert_b_fields_close(&dispatched, &scalar);
    }

    #[test]
    #[cfg(not(feature = "rayon"))]
    fn resistivity_dispatch_matches_scalar_with_dark_matter() {
        let mut scalar = make_induction_particles(16, true);
        let mut dispatched = scalar.clone();
        let dm_before: Vec<(usize, Vec3)> = dispatched
            .iter()
            .enumerate()
            .filter_map(|(idx, p)| {
                (p.ptype == ParticleType::DarkMatter).then_some((idx, p.b_field))
            })
            .collect();

        apply_artificial_resistivity_scalar(&mut scalar, 0.28, 0.018);
        apply_artificial_resistivity_impl(&mut dispatched, 0.28, 0.018);

        assert_b_fields_close(&dispatched, &scalar);
        for (idx, b_before) in dm_before {
            assert_abs_diff_eq!(dispatched[idx].b_field.x, b_before.x, epsilon = 0.0);
            assert_abs_diff_eq!(dispatched[idx].b_field.y, b_before.y, epsilon = 0.0);
            assert_abs_diff_eq!(dispatched[idx].b_field.z, b_before.z, epsilon = 0.0);
        }
    }
}
