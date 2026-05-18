//! Conducción térmica y difusión CR anisótropa a lo largo de líneas de campo B (Phase 133).
//!
//! ## Modelo
//!
//! En un plasma magnetizado, el transporte de calor y de rayos cósmicos es altamente
//! anisótropo: la difusión es eficiente paralela a B y suprimida en la dirección
//! perpendicular por la giración de las partículas en torno a las líneas de campo.
//!
//! ### Tensor de difusión
//!
//! ```text
//! D = κ_∥ (B̂ ⊗ B̂) + κ_⊥ (I − B̂ ⊗ B̂)
//! ```
//!
//! El flujo de calor entre partículas `i` y `j`:
//!
//! ```text
//! q_ij = (κ_∥ − κ_⊥) (B̂_i · r̂_ij)² + κ_⊥) × (T_j − T_i) × W(r_ij, h_i) × dt
//! ```
//!
//! Para `κ_⊥ = 0` y `κ_∥ >> 0`: conducción puramente paralela a B.
//! Para `κ_∥ = κ_⊥ = κ`: recupera conducción isótropa de Spitzer.
//!
//! ## Referencias
//!
//! Braginskii (1965), Rev. Plasma Phys. 1, 205 — tensor de transporte anisótropo.
//! Parrish & Stone (2005), ApJ 633, 334 — conducción anisótropa en SPH.
//! Sharma & Hammett (2007), J. Comput. Phys. 227, 123 — implementación discreta.

use crate::MU0;
use gadget_ng_core::{Particle, ParticleType, Vec3};
#[cfg(feature = "rayon")]
use rayon::prelude::*;
#[cfg(all(
    feature = "simd",
    not(feature = "rayon"),
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(all(
    feature = "simd",
    not(feature = "rayon"),
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Kernel SPH simple para difusión (Wendland C2).
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

/// Convierte energía interna `u` a temperatura con `γ = gamma`.
#[inline]
fn u_to_t(u: f64, gamma: f64) -> f64 {
    const MU_MEAN: f64 = 0.588;
    const KB_OVER_MU: f64 = 8.314e7;
    u.max(0.0) * (gamma - 1.0) / (KB_OVER_MU / MU_MEAN)
}

#[cfg(not(feature = "rayon"))]
fn apply_anisotropic_conduction_impl(
    particles: &mut [Particle],
    kappa_par: f64,
    kappa_perp: f64,
    gamma: f64,
    dt: f64,
) {
    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    {
        if is_x86_feature_detected!("avx512f") {
            // SAFETY: runtime dispatch checked `avx512f` immediately above.
            unsafe {
                apply_anisotropic_conduction_avx512(particles, kappa_par, kappa_perp, gamma, dt);
            }
            return;
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: runtime dispatch checked `avx2` and `fma` immediately above.
            unsafe {
                apply_anisotropic_conduction_avx2(particles, kappa_par, kappa_perp, gamma, dt);
            }
            return;
        }
    }

    apply_anisotropic_conduction_scalar(particles, kappa_par, kappa_perp, gamma, dt);
}

#[cfg(not(feature = "rayon"))]
fn apply_anisotropic_conduction_scalar(
    particles: &mut [Particle],
    kappa_par: f64,
    kappa_perp: f64,
    gamma: f64,
    dt: f64,
) {
    let n = particles.len();
    if n == 0 {
        return;
    }

    let mut delta_u = vec![0.0_f64; n];

    for i in 0..n {
        if particles[i].ptype != ParticleType::Gas {
            continue;
        }
        let h_i = particles[i].smoothing_length.max(1e-10);
        let pos_i = particles[i].position;
        let t_i = u_to_t(particles[i].internal_energy, gamma);
        let b_i = particles[i].b_field;
        let b_mag_i = (b_i.x * b_i.x + b_i.y * b_i.y + b_i.z * b_i.z)
            .sqrt()
            .max(1e-30);
        let bhat_i = Vec3::new(b_i.x / b_mag_i, b_i.y / b_mag_i, b_i.z / b_mag_i);

        for j in 0..n {
            if j == i {
                continue;
            }
            if particles[j].ptype != ParticleType::Gas {
                continue;
            }
            delta_u[i] += anisotropic_conduction_pair_contribution(
                particles, j, h_i, pos_i, t_i, bhat_i, kappa_par, kappa_perp, gamma, dt,
            );
        }
    }

    apply_conduction_updates(particles, &delta_u);
}

#[cfg(not(feature = "rayon"))]
#[expect(
    clippy::too_many_arguments,
    reason = "hot pair helper keeps scalar context explicit and allocation-free"
)]
#[inline]
fn anisotropic_conduction_pair_contribution(
    particles: &[Particle],
    j: usize,
    h_i: f64,
    pos_i: Vec3,
    t_i: f64,
    bhat_i: Vec3,
    kappa_par: f64,
    kappa_perp: f64,
    gamma: f64,
    dt: f64,
) -> f64 {
    let dx = particles[j].position.x - pos_i.x;
    let dy = particles[j].position.y - pos_i.y;
    let dz = particles[j].position.z - pos_i.z;
    let r = (dx * dx + dy * dy + dz * dz).sqrt();
    if r < 1e-14 {
        return 0.0;
    }

    let h_j = particles[j].smoothing_length.max(1e-10);
    let w = kernel_w(r, h_i + h_j);
    if w <= 0.0 {
        return 0.0;
    }

    let rhat_x = dx / r;
    let rhat_y = dy / r;
    let rhat_z = dz / r;
    let cos_theta = bhat_i.x * rhat_x + bhat_i.y * rhat_y + bhat_i.z * rhat_z;
    let cos2 = cos_theta * cos_theta;

    let kappa_eff = kappa_perp + (kappa_par - kappa_perp) * cos2;
    let t_j = u_to_t(particles[j].internal_energy, gamma);
    kappa_eff * (t_j - t_i) * w * dt
}

#[cfg(all(
    feature = "simd",
    not(feature = "rayon"),
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx2,fma")]
unsafe fn apply_anisotropic_conduction_avx2(
    particles: &mut [Particle],
    kappa_par: f64,
    kappa_perp: f64,
    gamma: f64,
    dt: f64,
) {
    let n = particles.len();
    let mut delta_u = vec![0.0_f64; n];

    for i in 0..n {
        if particles[i].ptype != ParticleType::Gas {
            continue;
        }
        let h_i = particles[i].smoothing_length.max(1e-10);
        let pos_i = particles[i].position;
        let t_i = u_to_t(particles[i].internal_energy, gamma);
        let b_i = particles[i].b_field;
        let b_mag_i = (b_i.x * b_i.x + b_i.y * b_i.y + b_i.z * b_i.z)
            .sqrt()
            .max(1e-30);
        let bhat_i = Vec3::new(b_i.x / b_mag_i, b_i.y / b_mag_i, b_i.z / b_mag_i);

        let mut j = 0;
        while j + 4 <= n {
            // SAFETY: this function is entered only after AVX2+FMA runtime dispatch.
            delta_u[i] += unsafe {
                anisotropic_conduction_sum4_avx2(
                    particles, i, j, h_i, pos_i, t_i, bhat_i, kappa_par, kappa_perp, gamma, dt,
                )
            };
            j += 4;
        }
        for tail in j..n {
            if tail != i && particles[tail].ptype == ParticleType::Gas {
                delta_u[i] += anisotropic_conduction_pair_contribution(
                    particles, tail, h_i, pos_i, t_i, bhat_i, kappa_par, kappa_perp, gamma, dt,
                );
            }
        }
    }

    apply_conduction_updates(particles, &delta_u);
}

#[cfg(all(
    feature = "simd",
    not(feature = "rayon"),
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
unsafe fn apply_anisotropic_conduction_avx512(
    particles: &mut [Particle],
    kappa_par: f64,
    kappa_perp: f64,
    gamma: f64,
    dt: f64,
) {
    let n = particles.len();
    let mut delta_u = vec![0.0_f64; n];

    for i in 0..n {
        if particles[i].ptype != ParticleType::Gas {
            continue;
        }
        let h_i = particles[i].smoothing_length.max(1e-10);
        let pos_i = particles[i].position;
        let t_i = u_to_t(particles[i].internal_energy, gamma);
        let b_i = particles[i].b_field;
        let b_mag_i = (b_i.x * b_i.x + b_i.y * b_i.y + b_i.z * b_i.z)
            .sqrt()
            .max(1e-30);
        let bhat_i = Vec3::new(b_i.x / b_mag_i, b_i.y / b_mag_i, b_i.z / b_mag_i);

        let mut j = 0;
        while j + 8 <= n {
            // SAFETY: this function is entered only after AVX512F runtime dispatch.
            delta_u[i] += unsafe {
                anisotropic_conduction_sum8_avx512(
                    particles, i, j, h_i, pos_i, t_i, bhat_i, kappa_par, kappa_perp, gamma, dt,
                )
            };
            j += 8;
        }
        for tail in j..n {
            if tail != i && particles[tail].ptype == ParticleType::Gas {
                delta_u[i] += anisotropic_conduction_pair_contribution(
                    particles, tail, h_i, pos_i, t_i, bhat_i, kappa_par, kappa_perp, gamma, dt,
                );
            }
        }
    }

    apply_conduction_updates(particles, &delta_u);
}

#[cfg(not(feature = "rayon"))]
fn apply_conduction_updates(particles: &mut [Particle], delta_u: &[f64]) {
    for (particle, du) in particles.iter_mut().zip(delta_u) {
        if particle.ptype == ParticleType::Gas {
            particle.internal_energy = (particle.internal_energy + du).max(0.0);
        }
    }
}

#[cfg(all(
    feature = "simd",
    not(feature = "rayon"),
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[expect(
    clippy::too_many_arguments,
    reason = "SIMD batch helper mirrors scalar pair context"
)]
#[target_feature(enable = "avx2,fma")]
unsafe fn anisotropic_conduction_sum4_avx2(
    particles: &[Particle],
    i: usize,
    start: usize,
    h_i: f64,
    pos_i: Vec3,
    t_i: f64,
    bhat_i: Vec3,
    kappa_par: f64,
    kappa_perp: f64,
    gamma: f64,
    dt: f64,
) -> f64 {
    let mut px = [0.0; 4];
    let mut py = [0.0; 4];
    let mut pz = [0.0; 4];
    let mut h = [0.0; 4];
    let mut temp = [0.0; 4];
    let mut active = [0.0; 4];
    for lane in 0..4 {
        let idx = start + lane;
        let p = &particles[idx];
        px[lane] = p.position.x;
        py[lane] = p.position.y;
        pz[lane] = p.position.z;
        h[lane] = (h_i + p.smoothing_length.max(1e-10)).max(1e-10);
        temp[lane] = u_to_t(p.internal_energy, gamma);
        active[lane] = f64::from(idx != i && p.ptype == ParticleType::Gas);
    }

    // SAFETY: stack arrays have at least four contiguous f64 lanes for unaligned loads.
    let px_v = unsafe { _mm256_loadu_pd(px.as_ptr()) };
    // SAFETY: stack arrays have at least four contiguous f64 lanes for unaligned loads.
    let py_v = unsafe { _mm256_loadu_pd(py.as_ptr()) };
    // SAFETY: stack arrays have at least four contiguous f64 lanes for unaligned loads.
    let pz_v = unsafe { _mm256_loadu_pd(pz.as_ptr()) };
    // SAFETY: stack arrays have at least four contiguous f64 lanes for unaligned loads.
    let h_v = unsafe { _mm256_loadu_pd(h.as_ptr()) };
    // SAFETY: stack arrays have at least four contiguous f64 lanes for unaligned loads.
    let temp_v = unsafe { _mm256_loadu_pd(temp.as_ptr()) };
    // SAFETY: stack arrays have at least four contiguous f64 lanes for unaligned loads.
    let active_v = unsafe { _mm256_loadu_pd(active.as_ptr()) };

    let dx = _mm256_sub_pd(px_v, _mm256_set1_pd(pos_i.x));
    let dy = _mm256_sub_pd(py_v, _mm256_set1_pd(pos_i.y));
    let dz = _mm256_sub_pd(pz_v, _mm256_set1_pd(pos_i.z));
    let r2 = _mm256_fmadd_pd(dx, dx, _mm256_fmadd_pd(dy, dy, _mm256_mul_pd(dz, dz)));
    let r = _mm256_sqrt_pd(r2);
    let q = _mm256_div_pd(r, h_v);
    let q_clamped = _mm256_min_pd(q, _mm256_set1_pd(2.0));
    let t = _mm256_sub_pd(
        _mm256_set1_pd(1.0),
        _mm256_mul_pd(_mm256_set1_pd(0.5), q_clamped),
    );
    let t2 = _mm256_mul_pd(t, t);
    let t4 = _mm256_mul_pd(t2, t2);
    let h2 = _mm256_mul_pd(h_v, h_v);
    let h3 = _mm256_mul_pd(h2, h_v);
    let norm = _mm256_div_pd(_mm256_set1_pd(21.0 / (2.0 * std::f64::consts::PI)), h3);
    let w = _mm256_mul_pd(
        _mm256_mul_pd(norm, t4),
        _mm256_add_pd(
            _mm256_set1_pd(1.0),
            _mm256_mul_pd(_mm256_set1_pd(2.0), q_clamped),
        ),
    );

    let r_safe = _mm256_max_pd(r, _mm256_set1_pd(1e-14));
    let inv_r = _mm256_div_pd(_mm256_set1_pd(1.0), r_safe);
    let cos_theta = _mm256_fmadd_pd(
        _mm256_set1_pd(bhat_i.x),
        _mm256_mul_pd(dx, inv_r),
        _mm256_fmadd_pd(
            _mm256_set1_pd(bhat_i.y),
            _mm256_mul_pd(dy, inv_r),
            _mm256_mul_pd(_mm256_set1_pd(bhat_i.z), _mm256_mul_pd(dz, inv_r)),
        ),
    );
    let cos2 = _mm256_mul_pd(cos_theta, cos_theta);
    let kappa_eff = _mm256_fmadd_pd(
        _mm256_set1_pd(kappa_par - kappa_perp),
        cos2,
        _mm256_set1_pd(kappa_perp),
    );
    let temp_delta = _mm256_sub_pd(temp_v, _mm256_set1_pd(t_i));
    let flux = _mm256_mul_pd(
        _mm256_mul_pd(_mm256_mul_pd(kappa_eff, temp_delta), w),
        _mm256_mul_pd(
            active_v,
            _mm256_and_pd(
                _mm256_cmp_pd(r, _mm256_set1_pd(1e-14), _CMP_GE_OQ),
                _mm256_set1_pd(dt),
            ),
        ),
    );
    let mut out = [0.0; 4];
    // SAFETY: `out` has four contiguous f64 slots for the unaligned store.
    unsafe {
        _mm256_storeu_pd(out.as_mut_ptr(), flux);
    }
    out.iter().sum()
}

#[cfg(all(
    feature = "simd",
    not(feature = "rayon"),
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[expect(
    clippy::too_many_arguments,
    reason = "SIMD batch helper mirrors scalar pair context"
)]
#[target_feature(enable = "avx512f")]
unsafe fn anisotropic_conduction_sum8_avx512(
    particles: &[Particle],
    i: usize,
    start: usize,
    h_i: f64,
    pos_i: Vec3,
    t_i: f64,
    bhat_i: Vec3,
    kappa_par: f64,
    kappa_perp: f64,
    gamma: f64,
    dt: f64,
) -> f64 {
    let mut px = [0.0; 8];
    let mut py = [0.0; 8];
    let mut pz = [0.0; 8];
    let mut h = [0.0; 8];
    let mut temp = [0.0; 8];
    let mut active = [0.0; 8];
    for lane in 0..8 {
        let idx = start + lane;
        let p = &particles[idx];
        px[lane] = p.position.x;
        py[lane] = p.position.y;
        pz[lane] = p.position.z;
        h[lane] = (h_i + p.smoothing_length.max(1e-10)).max(1e-10);
        temp[lane] = u_to_t(p.internal_energy, gamma);
        active[lane] = f64::from(idx != i && p.ptype == ParticleType::Gas);
    }

    // SAFETY: stack arrays have at least eight contiguous f64 lanes for unaligned loads.
    let px_v = unsafe { _mm512_loadu_pd(px.as_ptr()) };
    // SAFETY: stack arrays have at least eight contiguous f64 lanes for unaligned loads.
    let py_v = unsafe { _mm512_loadu_pd(py.as_ptr()) };
    // SAFETY: stack arrays have at least eight contiguous f64 lanes for unaligned loads.
    let pz_v = unsafe { _mm512_loadu_pd(pz.as_ptr()) };
    // SAFETY: stack arrays have at least eight contiguous f64 lanes for unaligned loads.
    let h_v = unsafe { _mm512_loadu_pd(h.as_ptr()) };
    // SAFETY: stack arrays have at least eight contiguous f64 lanes for unaligned loads.
    let temp_v = unsafe { _mm512_loadu_pd(temp.as_ptr()) };
    // SAFETY: stack arrays have at least eight contiguous f64 lanes for unaligned loads.
    let active_v = unsafe { _mm512_loadu_pd(active.as_ptr()) };

    let dx = _mm512_sub_pd(px_v, _mm512_set1_pd(pos_i.x));
    let dy = _mm512_sub_pd(py_v, _mm512_set1_pd(pos_i.y));
    let dz = _mm512_sub_pd(pz_v, _mm512_set1_pd(pos_i.z));
    let r2 = _mm512_fmadd_pd(dx, dx, _mm512_fmadd_pd(dy, dy, _mm512_mul_pd(dz, dz)));
    let r = _mm512_sqrt_pd(r2);
    let q = _mm512_div_pd(r, h_v);
    let q_clamped = _mm512_min_pd(q, _mm512_set1_pd(2.0));
    let t = _mm512_sub_pd(
        _mm512_set1_pd(1.0),
        _mm512_mul_pd(_mm512_set1_pd(0.5), q_clamped),
    );
    let t2 = _mm512_mul_pd(t, t);
    let t4 = _mm512_mul_pd(t2, t2);
    let h2 = _mm512_mul_pd(h_v, h_v);
    let h3 = _mm512_mul_pd(h2, h_v);
    let norm = _mm512_div_pd(_mm512_set1_pd(21.0 / (2.0 * std::f64::consts::PI)), h3);
    let w = _mm512_mul_pd(
        _mm512_mul_pd(norm, t4),
        _mm512_add_pd(
            _mm512_set1_pd(1.0),
            _mm512_mul_pd(_mm512_set1_pd(2.0), q_clamped),
        ),
    );

    let r_safe = _mm512_max_pd(r, _mm512_set1_pd(1e-14));
    let inv_r = _mm512_div_pd(_mm512_set1_pd(1.0), r_safe);
    let cos_theta = _mm512_fmadd_pd(
        _mm512_set1_pd(bhat_i.x),
        _mm512_mul_pd(dx, inv_r),
        _mm512_fmadd_pd(
            _mm512_set1_pd(bhat_i.y),
            _mm512_mul_pd(dy, inv_r),
            _mm512_mul_pd(_mm512_set1_pd(bhat_i.z), _mm512_mul_pd(dz, inv_r)),
        ),
    );
    let cos2 = _mm512_mul_pd(cos_theta, cos_theta);
    let kappa_eff = _mm512_fmadd_pd(
        _mm512_set1_pd(kappa_par - kappa_perp),
        cos2,
        _mm512_set1_pd(kappa_perp),
    );
    let temp_delta = _mm512_sub_pd(temp_v, _mm512_set1_pd(t_i));
    let flux = _mm512_mul_pd(
        _mm512_mul_pd(_mm512_mul_pd(kappa_eff, temp_delta), w),
        _mm512_mul_pd(
            active_v,
            _mm512_mask_blend_pd(
                _mm512_cmp_pd_mask(r, _mm512_set1_pd(1e-14), _CMP_GE_OQ),
                _mm512_setzero_pd(),
                _mm512_set1_pd(dt),
            ),
        ),
    );
    let mut out = [0.0; 8];
    // SAFETY: `out` has eight contiguous f64 slots for the unaligned store.
    unsafe {
        _mm512_storeu_pd(out.as_mut_ptr(), flux);
    }
    out.iter().sum()
}

#[cfg(feature = "rayon")]
fn apply_anisotropic_conduction_par(
    particles: &mut [Particle],
    kappa_par: f64,
    kappa_perp: f64,
    gamma: f64,
    dt: f64,
) {
    let n = particles.len();
    if n == 0 {
        return;
    }

    let pos: Vec<Vec3> = particles.iter().map(|p| p.position).collect();
    let h_sml: Vec<f64> = particles
        .iter()
        .map(|p| p.smoothing_length.max(1e-10))
        .collect();
    let internal_energy: Vec<f64> = particles.iter().map(|p| p.internal_energy).collect();
    let b_field: Vec<Vec3> = particles.iter().map(|p| p.b_field).collect();
    let is_gas: Vec<bool> = particles
        .iter()
        .map(|p| p.ptype == ParticleType::Gas)
        .collect();

    let updates: Vec<Option<f64>> = (0..n)
        .into_par_iter()
        .map(|i| {
            if !is_gas[i] {
                return None;
            }
            let h_i = h_sml[i];
            let t_i = u_to_t(internal_energy[i], gamma);
            let b_i = b_field[i];
            let b_mag_i = (b_i.x * b_i.x + b_i.y * b_i.y + b_i.z * b_i.z)
                .sqrt()
                .max(1e-30);
            let bhat_x = b_i.x / b_mag_i;
            let bhat_y = b_i.y / b_mag_i;
            let bhat_z = b_i.z / b_mag_i;

            let mut delta_u_i = 0.0_f64;

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

                let kappa_eff = kappa_perp + (kappa_par - kappa_perp) * cos2;
                let t_j = u_to_t(internal_energy[j], gamma);
                delta_u_i += kappa_eff * (t_j - t_i) * w * dt;
            }
            Some(delta_u_i)
        })
        .collect();

    for (p, update) in particles.iter_mut().zip(updates) {
        if let (true, Some(du)) = (p.ptype == ParticleType::Gas, update) {
            p.internal_energy = (p.internal_energy + du).max(0.0);
        }
    }
}

/// Conducción térmica anisótropa ∥B: `D = κ_∥ (B̂⊗B̂) + κ_⊥ (I − B̂⊗B̂)` (Phase 133).
///
/// El flujo de calor entre dos partículas depende del coseno del ángulo entre `r_ij` y `B_i`.
/// La conductividad efectiva en la dirección r̂_ij es:
///
/// ```text
/// κ_eff(θ) = κ_⊥ + (κ_∥ − κ_⊥) cos²(θ)
/// ```
///
/// donde `θ` es el ángulo entre r̂_ij y B̂_i.
pub fn apply_anisotropic_conduction(
    particles: &mut [Particle],
    kappa_par: f64,
    kappa_perp: f64,
    gamma: f64,
    dt: f64,
) {
    #[cfg(feature = "rayon")]
    {
        apply_anisotropic_conduction_par(particles, kappa_par, kappa_perp, gamma, dt);
    }

    #[cfg(not(feature = "rayon"))]
    {
        apply_anisotropic_conduction_impl(particles, kappa_par, kappa_perp, gamma, dt);
    }
}

#[cfg(not(feature = "rayon"))]
fn diffuse_cr_anisotropic_impl(
    particles: &mut [Particle],
    kappa_cr: f64,
    b_suppress: f64,
    dt: f64,
) {
    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    {
        if is_x86_feature_detected!("avx512f") {
            // SAFETY: runtime dispatch checked `avx512f` immediately above.
            unsafe {
                diffuse_cr_anisotropic_avx512(particles, kappa_cr, b_suppress, dt);
            }
            return;
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: runtime dispatch checked `avx2` and `fma` immediately above.
            unsafe {
                diffuse_cr_anisotropic_avx2(particles, kappa_cr, b_suppress, dt);
            }
            return;
        }
    }

    diffuse_cr_anisotropic_scalar(particles, kappa_cr, b_suppress, dt);
}

#[cfg(not(feature = "rayon"))]
fn diffuse_cr_anisotropic_scalar(
    particles: &mut [Particle],
    kappa_cr: f64,
    b_suppress: f64,
    dt: f64,
) {
    let n = particles.len();
    if n == 0 {
        return;
    }

    let mut delta_cr = vec![0.0_f64; n];

    for i in 0..n {
        if particles[i].ptype != ParticleType::Gas {
            continue;
        }
        let h_i = particles[i].smoothing_length.max(1e-10);
        let pos_i = particles[i].position;
        let e_i = particles[i].cr_energy;
        let b_i = particles[i].b_field;
        let b2_i = b_i.x * b_i.x + b_i.y * b_i.y + b_i.z * b_i.z;
        let b_mag_i = b2_i.sqrt().max(1e-30);
        let bhat_i = Vec3::new(b_i.x / b_mag_i, b_i.y / b_mag_i, b_i.z / b_mag_i);

        let kappa_eff_base = kappa_cr / (1.0 + b_suppress * b2_i);

        for j in 0..n {
            if i == j {
                continue;
            }
            if particles[j].ptype != ParticleType::Gas {
                continue;
            }
            delta_cr[i] += cr_diffusion_pair_contribution(
                particles,
                j,
                2.0 * h_i,
                pos_i,
                e_i,
                bhat_i,
                kappa_eff_base,
                dt,
            );
        }
    }

    apply_cr_updates(particles, &delta_cr);
}

#[cfg(not(feature = "rayon"))]
#[expect(
    clippy::too_many_arguments,
    reason = "hot pair helper keeps scalar context explicit and allocation-free"
)]
#[inline]
fn cr_diffusion_pair_contribution(
    particles: &[Particle],
    j: usize,
    h_kernel: f64,
    pos_i: Vec3,
    e_i: f64,
    bhat_i: Vec3,
    kappa_eff_base: f64,
    dt: f64,
) -> f64 {
    let dx = particles[j].position.x - pos_i.x;
    let dy = particles[j].position.y - pos_i.y;
    let dz = particles[j].position.z - pos_i.z;
    let r = (dx * dx + dy * dy + dz * dz).sqrt();
    if r < 1e-14 {
        return 0.0;
    }

    let w = kernel_w(r, h_kernel);
    if w <= 0.0 {
        return 0.0;
    }

    let rhat_x = dx / r;
    let rhat_y = dy / r;
    let rhat_z = dz / r;
    let cos_theta = bhat_i.x * rhat_x + bhat_i.y * rhat_y + bhat_i.z * rhat_z;
    let cos2 = cos_theta * cos_theta;

    let kappa_aniso = kappa_eff_base * cos2;
    kappa_aniso * (particles[j].cr_energy - e_i) * w * dt
}

#[cfg(all(
    feature = "simd",
    not(feature = "rayon"),
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx2,fma")]
unsafe fn diffuse_cr_anisotropic_avx2(
    particles: &mut [Particle],
    kappa_cr: f64,
    b_suppress: f64,
    dt: f64,
) {
    let n = particles.len();
    let mut delta_cr = vec![0.0_f64; n];

    for i in 0..n {
        if particles[i].ptype != ParticleType::Gas {
            continue;
        }
        let h_i = particles[i].smoothing_length.max(1e-10);
        let pos_i = particles[i].position;
        let e_i = particles[i].cr_energy;
        let b_i = particles[i].b_field;
        let b2_i = b_i.x * b_i.x + b_i.y * b_i.y + b_i.z * b_i.z;
        let b_mag_i = b2_i.sqrt().max(1e-30);
        let bhat_i = Vec3::new(b_i.x / b_mag_i, b_i.y / b_mag_i, b_i.z / b_mag_i);
        let kappa_eff_base = kappa_cr / (1.0 + b_suppress * b2_i);

        let mut j = 0;
        while j + 4 <= n {
            // SAFETY: this function is entered only after AVX2+FMA runtime dispatch.
            delta_cr[i] += unsafe {
                cr_diffusion_sum4_avx2(
                    particles,
                    i,
                    j,
                    2.0 * h_i,
                    pos_i,
                    e_i,
                    bhat_i,
                    kappa_eff_base,
                    dt,
                )
            };
            j += 4;
        }
        for tail in j..n {
            if tail != i && particles[tail].ptype == ParticleType::Gas {
                delta_cr[i] += cr_diffusion_pair_contribution(
                    particles,
                    tail,
                    2.0 * h_i,
                    pos_i,
                    e_i,
                    bhat_i,
                    kappa_eff_base,
                    dt,
                );
            }
        }
    }

    apply_cr_updates(particles, &delta_cr);
}

#[cfg(all(
    feature = "simd",
    not(feature = "rayon"),
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
unsafe fn diffuse_cr_anisotropic_avx512(
    particles: &mut [Particle],
    kappa_cr: f64,
    b_suppress: f64,
    dt: f64,
) {
    let n = particles.len();
    let mut delta_cr = vec![0.0_f64; n];

    for i in 0..n {
        if particles[i].ptype != ParticleType::Gas {
            continue;
        }
        let h_i = particles[i].smoothing_length.max(1e-10);
        let pos_i = particles[i].position;
        let e_i = particles[i].cr_energy;
        let b_i = particles[i].b_field;
        let b2_i = b_i.x * b_i.x + b_i.y * b_i.y + b_i.z * b_i.z;
        let b_mag_i = b2_i.sqrt().max(1e-30);
        let bhat_i = Vec3::new(b_i.x / b_mag_i, b_i.y / b_mag_i, b_i.z / b_mag_i);
        let kappa_eff_base = kappa_cr / (1.0 + b_suppress * b2_i);

        let mut j = 0;
        while j + 8 <= n {
            // SAFETY: this function is entered only after AVX512F runtime dispatch.
            delta_cr[i] += unsafe {
                cr_diffusion_sum8_avx512(
                    particles,
                    i,
                    j,
                    2.0 * h_i,
                    pos_i,
                    e_i,
                    bhat_i,
                    kappa_eff_base,
                    dt,
                )
            };
            j += 8;
        }
        for tail in j..n {
            if tail != i && particles[tail].ptype == ParticleType::Gas {
                delta_cr[i] += cr_diffusion_pair_contribution(
                    particles,
                    tail,
                    2.0 * h_i,
                    pos_i,
                    e_i,
                    bhat_i,
                    kappa_eff_base,
                    dt,
                );
            }
        }
    }

    apply_cr_updates(particles, &delta_cr);
}

#[cfg(not(feature = "rayon"))]
fn apply_cr_updates(particles: &mut [Particle], delta_cr: &[f64]) {
    for (particle, dc) in particles.iter_mut().zip(delta_cr) {
        if particle.ptype == ParticleType::Gas {
            particle.cr_energy = (particle.cr_energy + dc).max(0.0);
        }
    }
}

#[cfg(all(
    feature = "simd",
    not(feature = "rayon"),
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[expect(
    clippy::too_many_arguments,
    reason = "SIMD batch helper mirrors scalar pair context"
)]
#[target_feature(enable = "avx2,fma")]
unsafe fn cr_diffusion_sum4_avx2(
    particles: &[Particle],
    i: usize,
    start: usize,
    h_kernel: f64,
    pos_i: Vec3,
    e_i: f64,
    bhat_i: Vec3,
    kappa_eff_base: f64,
    dt: f64,
) -> f64 {
    let mut px = [0.0; 4];
    let mut py = [0.0; 4];
    let mut pz = [0.0; 4];
    let mut cr = [0.0; 4];
    let mut active = [0.0; 4];
    for lane in 0..4 {
        let idx = start + lane;
        let p = &particles[idx];
        px[lane] = p.position.x;
        py[lane] = p.position.y;
        pz[lane] = p.position.z;
        cr[lane] = p.cr_energy;
        active[lane] = f64::from(idx != i && p.ptype == ParticleType::Gas);
    }

    // SAFETY: stack arrays have at least four contiguous f64 lanes for unaligned loads.
    let px_v = unsafe { _mm256_loadu_pd(px.as_ptr()) };
    // SAFETY: stack arrays have at least four contiguous f64 lanes for unaligned loads.
    let py_v = unsafe { _mm256_loadu_pd(py.as_ptr()) };
    // SAFETY: stack arrays have at least four contiguous f64 lanes for unaligned loads.
    let pz_v = unsafe { _mm256_loadu_pd(pz.as_ptr()) };
    // SAFETY: stack arrays have at least four contiguous f64 lanes for unaligned loads.
    let cr_v = unsafe { _mm256_loadu_pd(cr.as_ptr()) };
    // SAFETY: stack arrays have at least four contiguous f64 lanes for unaligned loads.
    let active_v = unsafe { _mm256_loadu_pd(active.as_ptr()) };

    let dx = _mm256_sub_pd(px_v, _mm256_set1_pd(pos_i.x));
    let dy = _mm256_sub_pd(py_v, _mm256_set1_pd(pos_i.y));
    let dz = _mm256_sub_pd(pz_v, _mm256_set1_pd(pos_i.z));
    let r2 = _mm256_fmadd_pd(dx, dx, _mm256_fmadd_pd(dy, dy, _mm256_mul_pd(dz, dz)));
    let r = _mm256_sqrt_pd(r2);
    let q = _mm256_div_pd(r, _mm256_set1_pd(h_kernel));
    let q_clamped = _mm256_min_pd(q, _mm256_set1_pd(2.0));
    let t = _mm256_sub_pd(
        _mm256_set1_pd(1.0),
        _mm256_mul_pd(_mm256_set1_pd(0.5), q_clamped),
    );
    let t2 = _mm256_mul_pd(t, t);
    let t4 = _mm256_mul_pd(t2, t2);
    let norm = 21.0 / (2.0 * std::f64::consts::PI * h_kernel * h_kernel * h_kernel);
    let w = _mm256_mul_pd(
        _mm256_mul_pd(_mm256_set1_pd(norm), t4),
        _mm256_add_pd(
            _mm256_set1_pd(1.0),
            _mm256_mul_pd(_mm256_set1_pd(2.0), q_clamped),
        ),
    );

    let r_safe = _mm256_max_pd(r, _mm256_set1_pd(1e-14));
    let inv_r = _mm256_div_pd(_mm256_set1_pd(1.0), r_safe);
    let cos_theta = _mm256_fmadd_pd(
        _mm256_set1_pd(bhat_i.x),
        _mm256_mul_pd(dx, inv_r),
        _mm256_fmadd_pd(
            _mm256_set1_pd(bhat_i.y),
            _mm256_mul_pd(dy, inv_r),
            _mm256_mul_pd(_mm256_set1_pd(bhat_i.z), _mm256_mul_pd(dz, inv_r)),
        ),
    );
    let cos2 = _mm256_mul_pd(cos_theta, cos_theta);
    let flux = _mm256_mul_pd(
        _mm256_mul_pd(
            _mm256_mul_pd(
                _mm256_mul_pd(_mm256_set1_pd(kappa_eff_base), cos2),
                _mm256_sub_pd(cr_v, _mm256_set1_pd(e_i)),
            ),
            w,
        ),
        _mm256_mul_pd(
            active_v,
            _mm256_and_pd(
                _mm256_cmp_pd(r, _mm256_set1_pd(1e-14), _CMP_GE_OQ),
                _mm256_set1_pd(dt),
            ),
        ),
    );
    let mut out = [0.0; 4];
    // SAFETY: `out` has four contiguous f64 slots for the unaligned store.
    unsafe {
        _mm256_storeu_pd(out.as_mut_ptr(), flux);
    }
    out.iter().sum()
}

#[cfg(all(
    feature = "simd",
    not(feature = "rayon"),
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[expect(
    clippy::too_many_arguments,
    reason = "SIMD batch helper mirrors scalar pair context"
)]
#[target_feature(enable = "avx512f")]
unsafe fn cr_diffusion_sum8_avx512(
    particles: &[Particle],
    i: usize,
    start: usize,
    h_kernel: f64,
    pos_i: Vec3,
    e_i: f64,
    bhat_i: Vec3,
    kappa_eff_base: f64,
    dt: f64,
) -> f64 {
    let mut px = [0.0; 8];
    let mut py = [0.0; 8];
    let mut pz = [0.0; 8];
    let mut cr = [0.0; 8];
    let mut active = [0.0; 8];
    for lane in 0..8 {
        let idx = start + lane;
        let p = &particles[idx];
        px[lane] = p.position.x;
        py[lane] = p.position.y;
        pz[lane] = p.position.z;
        cr[lane] = p.cr_energy;
        active[lane] = f64::from(idx != i && p.ptype == ParticleType::Gas);
    }

    // SAFETY: stack arrays have at least eight contiguous f64 lanes for unaligned loads.
    let px_v = unsafe { _mm512_loadu_pd(px.as_ptr()) };
    // SAFETY: stack arrays have at least eight contiguous f64 lanes for unaligned loads.
    let py_v = unsafe { _mm512_loadu_pd(py.as_ptr()) };
    // SAFETY: stack arrays have at least eight contiguous f64 lanes for unaligned loads.
    let pz_v = unsafe { _mm512_loadu_pd(pz.as_ptr()) };
    // SAFETY: stack arrays have at least eight contiguous f64 lanes for unaligned loads.
    let cr_v = unsafe { _mm512_loadu_pd(cr.as_ptr()) };
    // SAFETY: stack arrays have at least eight contiguous f64 lanes for unaligned loads.
    let active_v = unsafe { _mm512_loadu_pd(active.as_ptr()) };

    let dx = _mm512_sub_pd(px_v, _mm512_set1_pd(pos_i.x));
    let dy = _mm512_sub_pd(py_v, _mm512_set1_pd(pos_i.y));
    let dz = _mm512_sub_pd(pz_v, _mm512_set1_pd(pos_i.z));
    let r2 = _mm512_fmadd_pd(dx, dx, _mm512_fmadd_pd(dy, dy, _mm512_mul_pd(dz, dz)));
    let r = _mm512_sqrt_pd(r2);
    let q = _mm512_div_pd(r, _mm512_set1_pd(h_kernel));
    let q_clamped = _mm512_min_pd(q, _mm512_set1_pd(2.0));
    let t = _mm512_sub_pd(
        _mm512_set1_pd(1.0),
        _mm512_mul_pd(_mm512_set1_pd(0.5), q_clamped),
    );
    let t2 = _mm512_mul_pd(t, t);
    let t4 = _mm512_mul_pd(t2, t2);
    let norm = 21.0 / (2.0 * std::f64::consts::PI * h_kernel * h_kernel * h_kernel);
    let w = _mm512_mul_pd(
        _mm512_mul_pd(_mm512_set1_pd(norm), t4),
        _mm512_add_pd(
            _mm512_set1_pd(1.0),
            _mm512_mul_pd(_mm512_set1_pd(2.0), q_clamped),
        ),
    );

    let r_safe = _mm512_max_pd(r, _mm512_set1_pd(1e-14));
    let inv_r = _mm512_div_pd(_mm512_set1_pd(1.0), r_safe);
    let cos_theta = _mm512_fmadd_pd(
        _mm512_set1_pd(bhat_i.x),
        _mm512_mul_pd(dx, inv_r),
        _mm512_fmadd_pd(
            _mm512_set1_pd(bhat_i.y),
            _mm512_mul_pd(dy, inv_r),
            _mm512_mul_pd(_mm512_set1_pd(bhat_i.z), _mm512_mul_pd(dz, inv_r)),
        ),
    );
    let cos2 = _mm512_mul_pd(cos_theta, cos_theta);
    let flux = _mm512_mul_pd(
        _mm512_mul_pd(
            _mm512_mul_pd(
                _mm512_mul_pd(_mm512_set1_pd(kappa_eff_base), cos2),
                _mm512_sub_pd(cr_v, _mm512_set1_pd(e_i)),
            ),
            w,
        ),
        _mm512_mul_pd(
            active_v,
            _mm512_mask_blend_pd(
                _mm512_cmp_pd_mask(r, _mm512_set1_pd(1e-14), _CMP_GE_OQ),
                _mm512_setzero_pd(),
                _mm512_set1_pd(dt),
            ),
        ),
    );
    let mut out = [0.0; 8];
    // SAFETY: `out` has eight contiguous f64 slots for the unaligned store.
    unsafe {
        _mm512_storeu_pd(out.as_mut_ptr(), flux);
    }
    out.iter().sum()
}

#[cfg(feature = "rayon")]
fn diffuse_cr_anisotropic_par(particles: &mut [Particle], kappa_cr: f64, b_suppress: f64, dt: f64) {
    let n = particles.len();
    if n == 0 {
        return;
    }

    let pos: Vec<Vec3> = particles.iter().map(|p| p.position).collect();
    let h_sml: Vec<f64> = particles
        .iter()
        .map(|p| p.smoothing_length.max(1e-10))
        .collect();
    let b_field: Vec<Vec3> = particles.iter().map(|p| p.b_field).collect();
    let cr_energy: Vec<f64> = particles.iter().map(|p| p.cr_energy).collect();
    let is_gas: Vec<bool> = particles
        .iter()
        .map(|p| p.ptype == ParticleType::Gas)
        .collect();

    let updates: Vec<Option<f64>> = (0..n)
        .into_par_iter()
        .map(|i| {
            if !is_gas[i] {
                return None;
            }
            let h_i = h_sml[i];
            let e_i = cr_energy[i];
            let b_i = b_field[i];
            let b2_i = b_i.x * b_i.x + b_i.y * b_i.y + b_i.z * b_i.z;
            let b_mag_i = b2_i.sqrt().max(1e-30);
            let bhat_x = b_i.x / b_mag_i;
            let bhat_y = b_i.y / b_mag_i;
            let bhat_z = b_i.z / b_mag_i;

            let kappa_eff_base = kappa_cr / (1.0 + b_suppress * b2_i);
            let mut delta_cr_i = 0.0_f64;
            let two_h = 2.0 * h_i;

            for j in 0..n {
                if j == i || !is_gas[j] {
                    continue;
                }
                let dx = pos[j].x - pos[i].x;
                let dy = pos[j].y - pos[i].y;
                let dz = pos[j].z - pos[i].z;
                let r2 = dx * dx + dy * dy + dz * dz;
                if r2 >= two_h * two_h {
                    continue;
                }
                let r = r2.sqrt();
                if r < 1e-14 {
                    continue;
                }

                let w = kernel_w_branchfree(r, two_h);
                if w <= 0.0 {
                    continue;
                }

                let rhat_x = dx / r;
                let rhat_y = dy / r;
                let rhat_z = dz / r;
                let cos_theta = bhat_x * rhat_x + bhat_y * rhat_y + bhat_z * rhat_z;
                let cos2 = cos_theta * cos_theta;

                let kappa_aniso = kappa_eff_base * cos2;
                delta_cr_i += kappa_aniso * (cr_energy[j] - e_i) * w * dt;
            }
            Some(delta_cr_i)
        })
        .collect();

    for (p, update) in particles.iter_mut().zip(updates) {
        if let (true, Some(dc)) = (p.ptype == ParticleType::Gas, update) {
            p.cr_energy = (p.cr_energy + dc).max(0.0);
        }
    }
}

/// Wendland C2 kernel (branch-free inner) para evaluación batch con fixed h.
///
/// `q = r/h`, `t = max(1 - q/2, 0)`, `W = σ/h³ · t⁴ · (1 + 2q)`.
///
/// La formulación branch-free (`q.min(2.0)`) hace que `t = 0` automáticamente
/// para `q > 2`, eliminando ramas en el inner loop.
#[cfg(feature = "rayon")]
#[inline]
fn kernel_w_branchfree(r: f64, h: f64) -> f64 {
    if h <= 0.0 {
        return 0.0;
    }
    let q = r / h;
    let q_clamped = if q > 2.0 { 2.0 } else { q };
    let t = 1.0 - 0.5 * q_clamped;
    (21.0 / (2.0 * std::f64::consts::PI * h * h * h)) * t * t * t * t * (1.0 + 2.0 * q_clamped)
}

/// Difusión CR anisótropa a lo largo de B (Phase 133).
///
/// El flujo CR en la dirección r̂_ij tiene un factor geométrico cos²(θ_B):
/// ```text
/// ΔE_cr,i = κ_CR × cos²(θ_B) × (e_cr,j − e_cr,i) × W(r_ij) × dt
/// ```
///
/// Con `B = 0` degenera en difusión isótropa.
pub fn diffuse_cr_anisotropic(particles: &mut [Particle], kappa_cr: f64, b_suppress: f64, dt: f64) {
    #[cfg(feature = "rayon")]
    {
        diffuse_cr_anisotropic_par(particles, kappa_cr, b_suppress, dt);
    }

    #[cfg(not(feature = "rayon"))]
    {
        diffuse_cr_anisotropic_impl(particles, kappa_cr, b_suppress, dt);
    }
}

/// Calcula el factor β-plasma: `β = 2μ₀ P_th / |B|²`.
///
/// Un β grande (>1) indica que la presión térmica domina sobre la presión magnética.
/// Un β pequeño (<1) indica que el campo magnético domina.
pub fn beta_plasma(p_thermal: f64, b: Vec3) -> f64 {
    let b2 = b.x * b.x + b.y * b.y + b.z * b.z;
    if b2 < 1e-60 {
        return f64::INFINITY;
    }
    2.0 * MU0 * p_thermal / b2
}

#[cfg(test)]
#[cfg_attr(feature = "rayon", allow(dead_code, unused_imports))]
mod tests {
    use super::*;

    use approx::assert_abs_diff_eq;
    use gadget_ng_core::{Particle, Vec3};

    #[test]
    fn beta_plasma_infinite_when_b_zero() {
        assert_eq!(beta_plasma(1.0, Vec3::zero()), f64::INFINITY);
    }

    #[test]
    fn beta_plasma_one_at_equipartition() {
        let b = Vec3::new(2.0, 0.0, 0.0);
        let p_th = 4.0 / (2.0 * MU0);
        assert_abs_diff_eq!(beta_plasma(p_th, b), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn beta_plasma_doubles_with_double_pth() {
        let b = Vec3::new(1.0, 2.0, 3.0);
        let p_th = 1.0;
        assert_abs_diff_eq!(
            beta_plasma(2.0 * p_th, b),
            2.0 * beta_plasma(p_th, b),
            epsilon = 1e-12
        );
    }

    #[test]
    fn beta_plasma_small_for_strong_b() {
        let b = Vec3::new(100.0, 0.0, 0.0);
        assert!(beta_plasma(1.0, b) < 1.0);
    }

    #[cfg(not(feature = "rayon"))]
    fn anisotropic_particles(mixed_dm: bool) -> Vec<Particle> {
        (0..12)
            .map(|i| {
                let mut particle = Particle::new_gas(
                    i,
                    1.0 + 0.01 * i as f64,
                    Vec3::new(
                        0.17 * i as f64 + 0.03 * (i % 3) as f64,
                        0.11 * (i % 5) as f64 + 0.02 * i as f64,
                        0.07 * (i % 7) as f64,
                    ),
                    Vec3::zero(),
                    2.0e12 + 1.0e10 * i as f64,
                    1.2,
                );
                particle.cr_energy = 0.4 + 0.03 * i as f64;
                particle.b_field = Vec3::new(
                    0.2 + 0.01 * i as f64,
                    -0.1 + 0.02 * (i % 4) as f64,
                    0.3 - 0.01 * (i % 5) as f64,
                );
                if mixed_dm && i % 4 == 0 {
                    particle.ptype = ParticleType::DarkMatter;
                    particle.internal_energy = -9.0;
                    particle.cr_energy = -3.0;
                }
                particle
            })
            .collect()
    }

    #[cfg(not(feature = "rayon"))]
    fn assert_particle_energies_close(actual: &[Particle], expected: &[Particle]) {
        for (a, e) in actual.iter().zip(expected) {
            // SIMD changes only local pair summation grouping, so a tight absolute
            // epsilon is enough for these small deterministic parity setups.
            assert_abs_diff_eq!(a.internal_energy, e.internal_energy, epsilon = 1e-6);
            assert_abs_diff_eq!(a.cr_energy, e.cr_energy, epsilon = 1e-12);
            if e.ptype != ParticleType::Gas {
                assert_eq!(a.internal_energy, e.internal_energy);
                assert_eq!(a.cr_energy, e.cr_energy);
            }
        }
    }

    #[test]
    #[cfg(not(feature = "rayon"))]
    fn anisotropic_conduction_dispatch_matches_scalar_for_all_gas() {
        let mut scalar = anisotropic_particles(false);
        let mut dispatch = scalar.clone();

        apply_anisotropic_conduction_scalar(&mut scalar, 0.7, 0.05, 5.0 / 3.0, 0.02);
        apply_anisotropic_conduction(&mut dispatch, 0.7, 0.05, 5.0 / 3.0, 0.02);

        assert_particle_energies_close(&dispatch, &scalar);
    }

    #[test]
    #[cfg(not(feature = "rayon"))]
    fn anisotropic_conduction_dispatch_matches_scalar_with_dark_matter() {
        let mut scalar = anisotropic_particles(true);
        let mut dispatch = scalar.clone();

        apply_anisotropic_conduction_scalar(&mut scalar, 0.5, 0.02, 1.4, 0.015);
        apply_anisotropic_conduction(&mut dispatch, 0.5, 0.02, 1.4, 0.015);

        assert_particle_energies_close(&dispatch, &scalar);
    }

    #[test]
    #[cfg(not(feature = "rayon"))]
    fn cr_diffusion_dispatch_matches_scalar_for_all_gas() {
        let mut scalar = anisotropic_particles(false);
        let mut dispatch = scalar.clone();

        diffuse_cr_anisotropic_scalar(&mut scalar, 0.4, 0.2, 0.03);
        diffuse_cr_anisotropic(&mut dispatch, 0.4, 0.2, 0.03);

        assert_particle_energies_close(&dispatch, &scalar);
    }

    #[test]
    #[cfg(not(feature = "rayon"))]
    fn cr_diffusion_dispatch_matches_scalar_with_dark_matter() {
        let mut scalar = anisotropic_particles(true);
        let mut dispatch = scalar.clone();

        diffuse_cr_anisotropic_scalar(&mut scalar, 0.6, 0.15, 0.025);
        diffuse_cr_anisotropic(&mut dispatch, 0.6, 0.15, 0.025);

        assert_particle_energies_close(&dispatch, &scalar);
    }
}
