//! Viscosidad anisótropa de Braginskii (Phase 146).
//!
//! ## Modelo
//!
//! En plasma magnetizado con alto β, el transporte de momento es anisótropo.
//! El tensor de presión viscosa de Braginskii tiene la forma:
//!
//! ```text
//! π_ij = −η_visc × (b̂_i b̂_j − δ_ij/3) × (∇·v)
//! ```
//!
//! donde `b̂ = B/|B|` es el versor del campo magnético. Esto produce:
//! - Viscosidad **máxima** en la dirección ∥ a B
//! - Viscosidad **nula** en la dirección ⊥ a B
//!
//! El efecto neto es una aceleración viscosa:
//!
//! ```text
//! a_visc,i = (1/ρ) ∇·π = η_visc/ρ × (b̂·∇)(b̂·∇v·b̂) b̂ + ...
//! ```
//!
//! En la discretización SPH usamos la forma simplificada:
//!
//! ```text
//! Δv_i = η_visc × Σ_j m_j/ρ_j × (b̂_i · r̂_ij)² × (v_j − v_i) · b̂_i × b̂_i × W_ij × dt
//! ```
//!
//! ## Referencias
//!
//! Braginskii (1965), Rev. Plasma Phys. 1, 205 — tensor de transporte viscoso.
//! Kunz et al. (2011), MNRAS 410, 2446 — viscosidad Braginskii en ICM.
//! Schekochihin & Cowley (2006), Phys. Plasmas 13, 056501 — MHD con Braginskii.

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

/// Kernel SPH compacto para difusión de momento.
#[inline]
fn kernel_w(r: f64, h: f64) -> f64 {
    if h <= 0.0 {
        return 0.0;
    }
    let q = r / h;
    if q > 2.0 {
        return 0.0;
    }
    let t = 1.0 - 0.5 * q;
    (21.0 / (2.0 * std::f64::consts::PI * h * h * h)) * t.powi(4) * (1.0 + 2.0 * q)
}

#[cfg(not(feature = "rayon"))]
fn apply_braginskii_viscosity_impl(particles: &mut [Particle], eta_visc: f64, dt: f64) {
    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    {
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx512f") {
            // SAFETY: AVX-512F availability was checked at runtime immediately above.
            unsafe {
                return apply_braginskii_viscosity_avx512(particles, eta_visc, dt);
            }
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: AVX2+FMA availability was checked at runtime immediately above.
            unsafe {
                return apply_braginskii_viscosity_avx2(particles, eta_visc, dt);
            }
        }
    }
    apply_braginskii_viscosity_scalar(particles, eta_visc, dt);
}

#[cfg(not(feature = "rayon"))]
fn apply_braginskii_viscosity_scalar(particles: &mut [Particle], eta_visc: f64, dt: f64) {
    if eta_visc <= 0.0 {
        return;
    }
    let n = particles.len();
    if n == 0 {
        return;
    }

    let mut dv = vec![Vec3::zero(); n];

    for i in 0..n {
        if particles[i].ptype != ParticleType::Gas {
            continue;
        }
        let h_i = particles[i].smoothing_length.max(1e-10);
        let pos_i = particles[i].position;
        let vel_i = particles[i].velocity;
        let b_i = particles[i].b_field;
        let b2_i = b_i.x * b_i.x + b_i.y * b_i.y + b_i.z * b_i.z;
        if b2_i < 1e-60 {
            continue;
        }
        let b_mag = b2_i.sqrt();
        let bhat = Vec3::new(b_i.x / b_mag, b_i.y / b_mag, b_i.z / b_mag);
        for j in 0..n {
            let contrib = braginskii_pair_contribution(
                particles, i, j, eta_visc, dt, h_i, pos_i, vel_i, bhat,
            );
            dv[i].x += contrib.x;
            dv[i].y += contrib.y;
            dv[i].z += contrib.z;
        }
    }

    apply_braginskii_updates(particles, &dv);
}

#[cfg(not(feature = "rayon"))]
fn apply_braginskii_updates(particles: &mut [Particle], dv: &[Vec3]) {
    for (p, dv_i) in particles.iter_mut().zip(dv.iter()) {
        if p.ptype == ParticleType::Gas {
            p.velocity.x += dv_i.x;
            p.velocity.y += dv_i.y;
            p.velocity.z += dv_i.z;
        }
    }
}

#[cfg(not(feature = "rayon"))]
#[expect(
    clippy::too_many_arguments,
    reason = "hot pair kernel keeps i-side cached"
)]
fn braginskii_pair_contribution(
    particles: &[Particle],
    i: usize,
    j: usize,
    eta_visc: f64,
    dt: f64,
    h_i: f64,
    pos_i: Vec3,
    vel_i: Vec3,
    bhat: Vec3,
) -> Vec3 {
    if j == i || particles[j].ptype != ParticleType::Gas {
        return Vec3::zero();
    }

    let dx = particles[j].position.x - pos_i.x;
    let dy = particles[j].position.y - pos_i.y;
    let dz = particles[j].position.z - pos_i.z;
    let r = (dx * dx + dy * dy + dz * dz).sqrt();
    if r < 1e-14 {
        return Vec3::zero();
    }

    let h_j = particles[j].smoothing_length.max(1e-10);
    let h_avg = 0.5 * (h_i + h_j);
    let w = kernel_w(r, 2.0 * h_avg);
    if w <= 0.0 {
        return Vec3::zero();
    }

    let rhat_x = dx / r;
    let rhat_y = dy / r;
    let rhat_z = dz / r;

    let cos_theta = bhat.x * rhat_x + bhat.y * rhat_y + bhat.z * rhat_z;
    let cos2 = cos_theta * cos_theta;

    let dvx = particles[j].velocity.x - vel_i.x;
    let dvy = particles[j].velocity.y - vel_i.y;
    let dvz = particles[j].velocity.z - vel_i.z;
    let dv_par = dvx * bhat.x + dvy * bhat.y + dvz * bhat.z;

    let rho_j = (particles[j].mass / (h_j * h_j * h_j)).max(1e-30);
    let factor = eta_visc * particles[j].mass / rho_j * cos2 * w * dv_par * dt;

    Vec3::new(factor * bhat.x, factor * bhat.y, factor * bhat.z)
}

#[cfg(all(
    not(feature = "rayon"),
    feature = "simd",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn apply_braginskii_viscosity_avx2(particles: &mut [Particle], eta_visc: f64, dt: f64) {
    if eta_visc <= 0.0 {
        return;
    }
    let n = particles.len();
    if n == 0 {
        return;
    }
    let mut dv = vec![Vec3::zero(); n];
    for i in 0..n {
        if particles[i].ptype != ParticleType::Gas {
            continue;
        }
        let h_i = particles[i].smoothing_length.max(1e-10);
        let pos_i = particles[i].position;
        let vel_i = particles[i].velocity;
        let b_i = particles[i].b_field;
        let b2_i = b_i.x * b_i.x + b_i.y * b_i.y + b_i.z * b_i.z;
        if b2_i < 1e-60 {
            continue;
        }
        let b_mag = b2_i.sqrt();
        let bhat = Vec3::new(b_i.x / b_mag, b_i.y / b_mag, b_i.z / b_mag);
        // SAFETY: caller reached this function only after runtime AVX2+FMA dispatch.
        dv[i] = unsafe { braginskii_sum_avx2(particles, i, eta_visc, dt, h_i, pos_i, vel_i, bhat) };
    }
    apply_braginskii_updates(particles, &dv);
}

#[cfg(all(
    not(feature = "rayon"),
    feature = "simd",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
unsafe fn apply_braginskii_viscosity_avx512(particles: &mut [Particle], eta_visc: f64, dt: f64) {
    if eta_visc <= 0.0 {
        return;
    }
    let n = particles.len();
    if n == 0 {
        return;
    }
    let mut dv = vec![Vec3::zero(); n];
    for i in 0..n {
        if particles[i].ptype != ParticleType::Gas {
            continue;
        }
        let h_i = particles[i].smoothing_length.max(1e-10);
        let pos_i = particles[i].position;
        let vel_i = particles[i].velocity;
        let b_i = particles[i].b_field;
        let b2_i = b_i.x * b_i.x + b_i.y * b_i.y + b_i.z * b_i.z;
        if b2_i < 1e-60 {
            continue;
        }
        let b_mag = b2_i.sqrt();
        let bhat = Vec3::new(b_i.x / b_mag, b_i.y / b_mag, b_i.z / b_mag);
        // SAFETY: caller reached this function only after runtime AVX-512F dispatch.
        dv[i] =
            unsafe { braginskii_sum_avx512(particles, i, eta_visc, dt, h_i, pos_i, vel_i, bhat) };
    }
    apply_braginskii_updates(particles, &dv);
}

#[cfg(all(
    not(feature = "rayon"),
    feature = "simd",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx2", enable = "fma")]
#[expect(
    clippy::too_many_arguments,
    reason = "hot pair reduction keeps i-side cached"
)]
unsafe fn braginskii_sum_avx2(
    particles: &[Particle],
    i: usize,
    eta_visc: f64,
    dt: f64,
    h_i: f64,
    pos_i: Vec3,
    vel_i: Vec3,
    bhat: Vec3,
) -> Vec3 {
    let lanes = 4;
    let chunks = particles.len() / lanes * lanes;
    let mut sum = _mm256_setzero_pd();
    let mut j = 0;
    while j < chunks {
        let all_valid = i < j || i >= j + lanes;
        let all_gas = particles[j..j + lanes]
            .iter()
            .all(|p| p.ptype == ParticleType::Gas);
        if !all_valid || !all_gas {
            for lane in 0..lanes {
                let contrib = braginskii_pair_contribution(
                    particles,
                    i,
                    j + lane,
                    eta_visc,
                    dt,
                    h_i,
                    pos_i,
                    vel_i,
                    bhat,
                );
                let scalar = if bhat.x.abs() > 0.0 {
                    contrib.x / bhat.x
                } else if bhat.y.abs() > 0.0 {
                    contrib.y / bhat.y
                } else {
                    contrib.z / bhat.z
                };
                sum = _mm256_add_pd(sum, _mm256_set_pd(0.0, 0.0, 0.0, scalar));
            }
            j += lanes;
            continue;
        }
        let contrib = braginskii_batch_avx2(particles, j, eta_visc, dt, h_i, pos_i, vel_i, bhat);
        sum = _mm256_add_pd(sum, contrib);
        j += lanes;
    }
    let mut out = [0.0; 4];
    // SAFETY: fixed-size stack array has exactly four f64 lanes.
    unsafe { _mm256_storeu_pd(out.as_mut_ptr(), sum) };
    let mut scalar_sum = out.into_iter().sum::<f64>();
    for j_tail in chunks..particles.len() {
        let contrib = braginskii_pair_contribution(
            particles, i, j_tail, eta_visc, dt, h_i, pos_i, vel_i, bhat,
        );
        scalar_sum += if bhat.x.abs() > 0.0 {
            contrib.x / bhat.x
        } else if bhat.y.abs() > 0.0 {
            contrib.y / bhat.y
        } else {
            contrib.z / bhat.z
        };
    }
    Vec3::new(
        scalar_sum * bhat.x,
        scalar_sum * bhat.y,
        scalar_sum * bhat.z,
    )
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
fn braginskii_batch_avx2(
    particles: &[Particle],
    j: usize,
    eta_visc: f64,
    dt: f64,
    h_i: f64,
    pos_i: Vec3,
    vel_i: Vec3,
    bhat: Vec3,
) -> __m256d {
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
    let valid_r = _mm256_cmp_pd(r, _mm256_set1_pd(1e-14), _CMP_GE_OQ);
    let h_j = _mm256_max_pd(
        _mm256_set1_pd(1e-10),
        _mm256_set_pd(
            particles[j + 3].smoothing_length,
            particles[j + 2].smoothing_length,
            particles[j + 1].smoothing_length,
            particles[j].smoothing_length,
        ),
    );
    let h_kernel = _mm256_add_pd(_mm256_set1_pd(h_i), h_j);
    let q = _mm256_div_pd(r, h_kernel);
    let support = _mm256_cmp_pd(q, _mm256_set1_pd(2.0), _CMP_LE_OQ);
    let t = _mm256_sub_pd(_mm256_set1_pd(1.0), _mm256_mul_pd(_mm256_set1_pd(0.5), q));
    let t2 = _mm256_mul_pd(t, t);
    let t4 = _mm256_mul_pd(t2, t2);
    let norm = _mm256_div_pd(
        _mm256_set1_pd(21.0 / (2.0 * std::f64::consts::PI)),
        _mm256_mul_pd(h_kernel, _mm256_mul_pd(h_kernel, h_kernel)),
    );
    let w = _mm256_and_pd(
        _mm256_mul_pd(
            norm,
            _mm256_mul_pd(
                t4,
                _mm256_add_pd(_mm256_set1_pd(1.0), _mm256_mul_pd(_mm256_set1_pd(2.0), q)),
            ),
        ),
        _mm256_and_pd(valid_r, support),
    );
    let inv_r = _mm256_div_pd(
        _mm256_set1_pd(1.0),
        _mm256_blendv_pd(
            r,
            _mm256_set1_pd(1.0),
            _mm256_cmp_pd(r, _mm256_set1_pd(1e-14), _CMP_LT_OQ),
        ),
    );
    let rhat_x = _mm256_mul_pd(dx, inv_r);
    let rhat_y = _mm256_mul_pd(dy, inv_r);
    let rhat_z = _mm256_mul_pd(dz, inv_r);
    let cos_theta = _mm256_fmadd_pd(
        _mm256_set1_pd(bhat.x),
        rhat_x,
        _mm256_fmadd_pd(
            _mm256_set1_pd(bhat.y),
            rhat_y,
            _mm256_mul_pd(_mm256_set1_pd(bhat.z), rhat_z),
        ),
    );
    let cos2 = _mm256_mul_pd(cos_theta, cos_theta);
    let vx = _mm256_sub_pd(
        _mm256_set_pd(
            particles[j + 3].velocity.x,
            particles[j + 2].velocity.x,
            particles[j + 1].velocity.x,
            particles[j].velocity.x,
        ),
        _mm256_set1_pd(vel_i.x),
    );
    let vy = _mm256_sub_pd(
        _mm256_set_pd(
            particles[j + 3].velocity.y,
            particles[j + 2].velocity.y,
            particles[j + 1].velocity.y,
            particles[j].velocity.y,
        ),
        _mm256_set1_pd(vel_i.y),
    );
    let vz = _mm256_sub_pd(
        _mm256_set_pd(
            particles[j + 3].velocity.z,
            particles[j + 2].velocity.z,
            particles[j + 1].velocity.z,
            particles[j].velocity.z,
        ),
        _mm256_set1_pd(vel_i.z),
    );
    let dv_par = _mm256_fmadd_pd(
        _mm256_set1_pd(bhat.x),
        vx,
        _mm256_fmadd_pd(
            _mm256_set1_pd(bhat.y),
            vy,
            _mm256_mul_pd(_mm256_set1_pd(bhat.z), vz),
        ),
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
    _mm256_mul_pd(
        _mm256_set1_pd(eta_visc * dt),
        _mm256_mul_pd(
            _mm256_div_pd(mass, rho),
            _mm256_mul_pd(cos2, _mm256_mul_pd(w, dv_par)),
        ),
    )
}

#[cfg(all(
    not(feature = "rayon"),
    feature = "simd",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
#[expect(
    clippy::too_many_arguments,
    reason = "hot pair reduction keeps i-side cached"
)]
unsafe fn braginskii_sum_avx512(
    particles: &[Particle],
    i: usize,
    eta_visc: f64,
    dt: f64,
    h_i: f64,
    pos_i: Vec3,
    vel_i: Vec3,
    bhat: Vec3,
) -> Vec3 {
    let lanes = 8;
    let chunks = particles.len() / lanes * lanes;
    let mut sum = _mm512_setzero_pd();
    let mut j = 0;
    while j < chunks {
        let all_valid = i < j || i >= j + lanes;
        let all_gas = particles[j..j + lanes]
            .iter()
            .all(|p| p.ptype == ParticleType::Gas);
        if !all_valid || !all_gas {
            for lane in 0..lanes {
                let contrib = braginskii_pair_contribution(
                    particles,
                    i,
                    j + lane,
                    eta_visc,
                    dt,
                    h_i,
                    pos_i,
                    vel_i,
                    bhat,
                );
                let scalar = if bhat.x.abs() > 0.0 {
                    contrib.x / bhat.x
                } else if bhat.y.abs() > 0.0 {
                    contrib.y / bhat.y
                } else {
                    contrib.z / bhat.z
                };
                sum = _mm512_add_pd(
                    sum,
                    _mm512_set_pd(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, scalar),
                );
            }
            j += lanes;
            continue;
        }
        let contrib = braginskii_batch_avx512(particles, j, eta_visc, dt, h_i, pos_i, vel_i, bhat);
        sum = _mm512_add_pd(sum, contrib);
        j += lanes;
    }
    let mut out = [0.0; 8];
    // SAFETY: fixed-size stack array has exactly eight f64 lanes.
    unsafe { _mm512_storeu_pd(out.as_mut_ptr(), sum) };
    let mut scalar_sum = out.into_iter().sum::<f64>();
    for j_tail in chunks..particles.len() {
        let contrib = braginskii_pair_contribution(
            particles, i, j_tail, eta_visc, dt, h_i, pos_i, vel_i, bhat,
        );
        scalar_sum += if bhat.x.abs() > 0.0 {
            contrib.x / bhat.x
        } else if bhat.y.abs() > 0.0 {
            contrib.y / bhat.y
        } else {
            contrib.z / bhat.z
        };
    }
    Vec3::new(
        scalar_sum * bhat.x,
        scalar_sum * bhat.y,
        scalar_sum * bhat.z,
    )
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
fn braginskii_batch_avx512(
    particles: &[Particle],
    j: usize,
    eta_visc: f64,
    dt: f64,
    h_i: f64,
    pos_i: Vec3,
    vel_i: Vec3,
    bhat: Vec3,
) -> __m512d {
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
    let valid_r = _mm512_cmp_pd_mask(r, _mm512_set1_pd(1e-14), _CMP_GE_OQ);
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
    let h_kernel = _mm512_add_pd(_mm512_set1_pd(h_i), h_j);
    let q = _mm512_div_pd(r, h_kernel);
    let support = _mm512_cmp_pd_mask(q, _mm512_set1_pd(2.0), _CMP_LE_OQ);
    let t = _mm512_sub_pd(_mm512_set1_pd(1.0), _mm512_mul_pd(_mm512_set1_pd(0.5), q));
    let t2 = _mm512_mul_pd(t, t);
    let t4 = _mm512_mul_pd(t2, t2);
    let norm = _mm512_div_pd(
        _mm512_set1_pd(21.0 / (2.0 * std::f64::consts::PI)),
        _mm512_mul_pd(h_kernel, _mm512_mul_pd(h_kernel, h_kernel)),
    );
    let w = _mm512_maskz_mul_pd(
        valid_r & support,
        norm,
        _mm512_mul_pd(
            t4,
            _mm512_add_pd(_mm512_set1_pd(1.0), _mm512_mul_pd(_mm512_set1_pd(2.0), q)),
        ),
    );
    let safe_r = _mm512_mask_blend_pd(
        _mm512_cmp_pd_mask(r, _mm512_set1_pd(1e-14), _CMP_LT_OQ),
        r,
        _mm512_set1_pd(1.0),
    );
    let inv_r = _mm512_div_pd(_mm512_set1_pd(1.0), safe_r);
    let rhat_x = _mm512_mul_pd(dx, inv_r);
    let rhat_y = _mm512_mul_pd(dy, inv_r);
    let rhat_z = _mm512_mul_pd(dz, inv_r);
    let cos_theta = _mm512_fmadd_pd(
        _mm512_set1_pd(bhat.x),
        rhat_x,
        _mm512_fmadd_pd(
            _mm512_set1_pd(bhat.y),
            rhat_y,
            _mm512_mul_pd(_mm512_set1_pd(bhat.z), rhat_z),
        ),
    );
    let cos2 = _mm512_mul_pd(cos_theta, cos_theta);
    let vx = _mm512_sub_pd(
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
        _mm512_set1_pd(vel_i.x),
    );
    let vy = _mm512_sub_pd(
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
        _mm512_set1_pd(vel_i.y),
    );
    let vz = _mm512_sub_pd(
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
        _mm512_set1_pd(vel_i.z),
    );
    let dv_par = _mm512_fmadd_pd(
        _mm512_set1_pd(bhat.x),
        vx,
        _mm512_fmadd_pd(
            _mm512_set1_pd(bhat.y),
            vy,
            _mm512_mul_pd(_mm512_set1_pd(bhat.z), vz),
        ),
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
    _mm512_mul_pd(
        _mm512_set1_pd(eta_visc * dt),
        _mm512_mul_pd(
            _mm512_div_pd(mass, rho),
            _mm512_mul_pd(cos2, _mm512_mul_pd(w, dv_par)),
        ),
    )
}

#[cfg(feature = "rayon")]
fn apply_braginskii_viscosity_par(particles: &mut [Particle], eta_visc: f64, dt: f64) {
    if eta_visc <= 0.0 {
        return;
    }
    let n = particles.len();
    if n == 0 {
        return;
    }

    let pos: Vec<Vec3> = particles.iter().map(|p| p.position).collect();
    let vel: Vec<Vec3> = particles.iter().map(|p| p.velocity).collect();
    let mass: Vec<f64> = particles.iter().map(|p| p.mass).collect();
    let h_sml: Vec<f64> = particles
        .iter()
        .map(|p| p.smoothing_length.max(1e-10))
        .collect();
    let b_field: Vec<Vec3> = particles.iter().map(|p| p.b_field).collect();
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
            let b2_i = b_i.x * b_i.x + b_i.y * b_i.y + b_i.z * b_i.z;
            if b2_i < 1e-60 {
                return Some(Vec3::zero());
            }
            let b_mag = b2_i.sqrt();
            let bhat_x = b_i.x / b_mag;
            let bhat_y = b_i.y / b_mag;
            let bhat_z = b_i.z / b_mag;
            let vel_i = vel[i];

            let mut dv_i = Vec3::zero();

            for j in 0..n {
                if j == i || !is_gas[j] {
                    continue;
                }
                let dx = pos[j].x - pos[i].x;
                let dy = pos[j].y - pos[i].y;
                let dz = pos[j].z - pos[i].z;
                let r = (dx * dx + dy * dy + dz * dz).sqrt();
                if r < 1e-14 {
                    continue;
                }

                let h_avg = 0.5 * (h_i + h_sml[j]);
                let w = kernel_w(r, 2.0 * h_avg);
                if w <= 0.0 {
                    continue;
                }

                let rhat_x = dx / r;
                let rhat_y = dy / r;
                let rhat_z = dz / r;

                let cos_theta = bhat_x * rhat_x + bhat_y * rhat_y + bhat_z * rhat_z;
                let cos2 = cos_theta * cos_theta;

                let dvx = vel[j].x - vel_i.x;
                let dvy = vel[j].y - vel_i.y;
                let dvz = vel[j].z - vel_i.z;
                let dv_par = dvx * bhat_x + dvy * bhat_y + dvz * bhat_z;

                let factor = eta_visc * mass[j] / rho[j] * cos2 * w;

                dv_i.x += factor * dv_par * bhat_x * dt;
                dv_i.y += factor * dv_par * bhat_y * dt;
                dv_i.z += factor * dv_par * bhat_z * dt;
            }
            Some(dv_i)
        })
        .collect();

    for (p, update) in particles.iter_mut().zip(updates) {
        if let (true, Some(dv)) = (p.ptype == ParticleType::Gas, update) {
            p.velocity.x += dv.x;
            p.velocity.y += dv.y;
            p.velocity.z += dv.z;
        }
    }
}

/// Aplica la viscosidad anisótropa de Braginskii al campo de velocidades (Phase 146).
///
/// El tensor de presión viscosa `π_ij = −η_visc (b̂_i b̂_j − δ_ij/3) ∇·v`
/// se discretiza en SPH como un intercambio de momento anisótropo entre pares,
/// proyectado sobre la dirección del campo magnético local.
///
/// # Parámetros
///
/// - `particles`: slice mutable de partículas de gas
/// - `eta_visc`: coeficiente de viscosidad de Braginskii [unidades internas]
/// - `dt`: paso de tiempo
pub fn apply_braginskii_viscosity(particles: &mut [Particle], eta_visc: f64, dt: f64) {
    #[cfg(feature = "rayon")]
    {
        apply_braginskii_viscosity_par(particles, eta_visc, dt);
    }

    #[cfg(not(feature = "rayon"))]
    {
        apply_braginskii_viscosity_impl(particles, eta_visc, dt);
    }
}

#[cfg(test)]
#[cfg_attr(feature = "rayon", allow(dead_code, unused_imports))]
mod tests {
    use super::*;

    fn make_braginskii_particles(n: usize, with_dm: bool) -> Vec<Particle> {
        (0..n)
            .map(|idx| {
                let t = idx as f64;
                let mut p = Particle::new_gas(
                    idx,
                    1.0 + 0.015 * t,
                    Vec3::new(0.11 * t, 0.04 * (0.6 * t).sin(), 0.03 * (0.4 * t).cos()),
                    Vec3::new(0.02 * t.sin(), -0.015 * t.cos(), 0.01 * t),
                    1.0,
                    0.4 + 0.006 * (idx % 5) as f64,
                );
                p.b_field = Vec3::new(
                    0.1 + 0.005 * t,
                    -0.03 + 0.002 * (0.3 * t).sin(),
                    0.06 + 0.001 * (idx % 4) as f64,
                );
                if with_dm && matches!(idx, 4 | 11) {
                    Particle::new(idx, p.mass, p.position, Vec3::new(3.0, -2.0, 1.0))
                } else {
                    p
                }
            })
            .collect()
    }

    fn assert_velocities_close(actual: &[Particle], expected: &[Particle]) {
        for (a, e) in actual.iter().zip(expected.iter()) {
            // SIMD batches use FMA and horizontal reductions; tolerance is roundoff-only
            // against the scalar Braginskii pair loop.
            assert!((a.velocity.x - e.velocity.x).abs() < 1e-10);
            assert!((a.velocity.y - e.velocity.y).abs() < 1e-10);
            assert!((a.velocity.z - e.velocity.z).abs() < 1e-10);
        }
    }

    #[test]
    #[cfg(not(feature = "rayon"))]
    fn braginskii_dispatch_matches_scalar_for_all_gas() {
        let mut scalar = make_braginskii_particles(16, false);
        let mut dispatched = scalar.clone();

        apply_braginskii_viscosity_scalar(&mut scalar, 0.08, 0.03);
        apply_braginskii_viscosity_impl(&mut dispatched, 0.08, 0.03);

        assert_velocities_close(&dispatched, &scalar);
    }

    #[test]
    #[cfg(not(feature = "rayon"))]
    fn braginskii_dispatch_matches_scalar_with_dark_matter() {
        let mut scalar = make_braginskii_particles(16, true);
        let mut dispatched = scalar.clone();
        let dm_before: Vec<(usize, Vec3)> = dispatched
            .iter()
            .enumerate()
            .filter_map(|(idx, p)| {
                (p.ptype == ParticleType::DarkMatter).then_some((idx, p.velocity))
            })
            .collect();

        apply_braginskii_viscosity_scalar(&mut scalar, 0.05, 0.025);
        apply_braginskii_viscosity_impl(&mut dispatched, 0.05, 0.025);

        assert_velocities_close(&dispatched, &scalar);
        for (idx, v_before) in dm_before {
            assert_eq!(dispatched[idx].velocity, v_before);
        }
    }
}
