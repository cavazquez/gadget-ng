//! Esquema de Dedner div-B cleaning (Phase 125).
//!
//! ## Formulación
//!
//! El esquema de Dedner et al. (2002) introduce un campo escalar `ψ` que
//! transporta y disipa el error de divergencia `∇·B`:
//!
//! ```text
//! ∂B/∂t + ∇ψ = 0
//! ∂ψ/∂t + c_h² ∇·B = −c_r ψ
//! ```
//!
//! donde:
//! - `c_h` es la velocidad de propagación de las ondas de limpieza (típicamente la
//!   velocidad de Alfvén máxima en la caja).
//! - `c_r` es la tasa de amortiguamiento (control la disipación de `ψ`).
//!
//! En la integración explícita de Euler:
//!
//! ```text
//! ψ_new = ψ × exp(−c_r × dt)  [disipación]
//! B_new  = B − ∇ψ × dt         [corrección del campo]
//! ```
//!
//! ## Referencia
//!
//! Dedner et al. (2002), J. Comput. Phys. 175, 645–673.
//! Tricco & Price (2012), J. Comput. Phys. 231, 7214.

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

/// Gradiente SPH del campo escalar ψ para un par (i, j).
fn grad_w_scalar(r_vec: Vec3, h: f64) -> Vec3 {
    let r2 = r_vec.x * r_vec.x + r_vec.y * r_vec.y + r_vec.z * r_vec.z;
    let r = r2.sqrt();
    if r < 1e-10 || h <= 0.0 {
        return Vec3::zero();
    }
    let q = r / h;
    let dw_dr = if q < 1.0 {
        let norm = 8.0 / (std::f64::consts::PI * h.powi(3));
        norm * (-6.0 * q + 9.0 * q * q) / h
    } else if q < 2.0 {
        let norm = 8.0 / (std::f64::consts::PI * h.powi(3));
        norm * (-6.0 * (2.0 - q).powi(2)) / (4.0 * h)
    } else {
        0.0
    };
    Vec3 {
        x: dw_dr * r_vec.x / r,
        y: dw_dr * r_vec.y / r,
        z: dw_dr * r_vec.z / r,
    }
}

#[cfg(not(feature = "rayon"))]
fn dedner_cleaning_step_impl(particles: &mut [Particle], c_h: f64, c_r: f64, dt: f64) {
    let n = particles.len();

    let rho: Vec<f64> = particles
        .iter()
        .map(|p| {
            let h = p.smoothing_length.max(1e-10);
            (p.mass / (h * h * h)).max(1e-30)
        })
        .collect();

    let mut div_b = vec![0.0_f64; n];
    let mut grad_psi = vec![Vec3::zero(); n];

    for i in 0..n {
        if particles[i].ptype != ParticleType::Gas {
            continue;
        }
        let b_i = particles[i].b_field;
        let psi_i = particles[i].psi_div;

        for j in 0..n {
            if i == j {
                continue;
            }
            if particles[j].ptype != ParticleType::Gas {
                continue;
            }

            let b_j = particles[j].b_field;
            let psi_j = particles[j].psi_div;
            let h_ij =
                0.5 * (particles[i].smoothing_length + particles[j].smoothing_length).max(1e-10);
            let r_ij = Vec3 {
                x: particles[j].position.x - particles[i].position.x,
                y: particles[j].position.y - particles[i].position.y,
                z: particles[j].position.z - particles[i].position.z,
            };
            let grad_w = grad_w_scalar(r_ij, h_ij);
            let factor = particles[j].mass / rho[j];

            let db = Vec3 {
                x: b_j.x - b_i.x,
                y: b_j.y - b_i.y,
                z: b_j.z - b_i.z,
            };
            div_b[i] += factor * (db.x * grad_w.x + db.y * grad_w.y + db.z * grad_w.z);

            let dpsi = psi_j - psi_i;
            grad_psi[i].x += factor * dpsi * grad_w.x;
            grad_psi[i].y += factor * dpsi * grad_w.y;
            grad_psi[i].z += factor * dpsi * grad_w.z;
        }
    }

    let decay = (-c_r * dt).exp();

    #[cfg(all(
        not(feature = "rayon"),
        feature = "simd",
        any(target_arch = "x86", target_arch = "x86_64")
    ))]
    {
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx512f") {
            // SAFETY: AVX-512F availability was checked at runtime.
            // Only the final update step is vectorized;
            // the O(N²) pairwise loop stays scalar.
            unsafe {
                dedner_cleaning_update_simd(
                    particles,
                    &div_b,
                    &grad_psi,
                    c_h * c_h * dt,
                    decay,
                    dt,
                );
                return;
            }
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: AVX2+FMA availability was checked at runtime.
            unsafe {
                dedner_cleaning_update_simd(
                    particles,
                    &div_b,
                    &grad_psi,
                    c_h * c_h * dt,
                    decay,
                    dt,
                );
                return;
            }
        }
    }

    for i in 0..n {
        if particles[i].ptype == ParticleType::Gas {
            particles[i].psi_div = particles[i].psi_div * decay - c_h * c_h * div_b[i] * dt;
            particles[i].b_field.x -= grad_psi[i].x * dt;
            particles[i].b_field.y -= grad_psi[i].y * dt;
            particles[i].b_field.z -= grad_psi[i].z * dt;
        }
    }
}

#[cfg(feature = "rayon")]
fn dedner_cleaning_step_par(particles: &mut [Particle], c_h: f64, c_r: f64, dt: f64) {
    let n = particles.len();

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
    let b_field: Vec<Vec3> = particles.iter().map(|p| p.b_field).collect();
    let psi_div: Vec<f64> = particles.iter().map(|p| p.psi_div).collect();
    let is_gas: Vec<bool> = particles
        .iter()
        .map(|p| p.ptype == ParticleType::Gas)
        .collect();

    let updates: Vec<Option<(f64, Vec3)>> = (0..n)
        .into_par_iter()
        .map(|i| {
            if !is_gas[i] {
                return None;
            }
            let b_i = b_field[i];
            let psi_i = psi_div[i];
            let mut div_b_i = 0.0_f64;
            let mut grad_psi_i = Vec3::zero();

            for j in 0..n {
                if j == i || !is_gas[j] {
                    continue;
                }
                let b_j = b_field[j];
                let psi_j = psi_div[j];
                let h_ij = 0.5 * (h_sml[i] + h_sml[j]);
                let r_ij = Vec3 {
                    x: pos[j].x - pos[i].x,
                    y: pos[j].y - pos[i].y,
                    z: pos[j].z - pos[i].z,
                };
                let grad_w = grad_w_scalar(r_ij, h_ij);
                let factor = mass[j] / rho[j];

                let db = Vec3 {
                    x: b_j.x - b_i.x,
                    y: b_j.y - b_i.y,
                    z: b_j.z - b_i.z,
                };
                div_b_i += factor * (db.x * grad_w.x + db.y * grad_w.y + db.z * grad_w.z);

                let dpsi = psi_j - psi_i;
                grad_psi_i.x += factor * dpsi * grad_w.x;
                grad_psi_i.y += factor * dpsi * grad_w.y;
                grad_psi_i.z += factor * dpsi * grad_w.z;
            }
            Some((div_b_i, grad_psi_i))
        })
        .collect();

    let decay = (-c_r * dt).exp();
    for (p, update) in particles.iter_mut().zip(updates) {
        if let (true, Some((div_b, grad_psi))) = (p.ptype == ParticleType::Gas, update) {
            p.psi_div = p.psi_div * decay - c_h * c_h * div_b * dt;
            p.b_field.x -= grad_psi.x * dt;
            p.b_field.y -= grad_psi.y * dt;
            p.b_field.z -= grad_psi.z * dt;
        }
    }
}

/// Aplica un paso del esquema de limpieza de Dedner para div-B (Phase 125).
///
/// # Parámetros
///
/// - `particles` — slice mutable de partículas.
/// - `c_h`       — velocidad de las ondas de limpieza (típicamente velocidad de Alfvén máx.).
/// - `c_r`       — tasa de amortiguamiento de ψ (s⁻¹).
/// - `dt`        — paso de tiempo.
///
/// # Algoritmo
///
/// 1. Calcula la divergencia SPH de B para cada partícula: `div_B_i = Σ_j (m_j/ρ_j) (B_j − B_i)·∇W_ij`.
/// 2. Actualiza ψ: `ψ_new = ψ × exp(−c_r × dt) − c_h² × div_B × dt`.
/// 3. Calcula el gradiente SPH de ψ: `∇ψ_i = Σ_j (m_j/ρ_j) (ψ_j − ψ_i) ∇W_ij`.
/// 4. Corrige B: `B_new = B − ∇ψ × dt`.
pub fn dedner_cleaning_step(particles: &mut [Particle], c_h: f64, c_r: f64, dt: f64) {
    #[cfg(feature = "rayon")]
    {
        dedner_cleaning_step_par(particles, c_h, c_r, dt);
    }

    #[cfg(not(feature = "rayon"))]
    {
        dedner_cleaning_step_impl(particles, c_h, c_r, dt);
    }
}

#[cfg(all(
    not(feature = "rayon"),
    feature = "simd",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn dedner_cleaning_update_simd(
    particles: &mut [Particle],
    div_b: &[f64],
    grad_psi: &[Vec3],
    c_h_sq_dt: f64,
    decay: f64,
    dt: f64,
) {
    let lanes = 4;
    let n = particles.len();
    let chunks = n / lanes * lanes;
    let c_h_sq_dt_v = _mm256_set1_pd(c_h_sq_dt);
    let decay_v = _mm256_set1_pd(decay);
    let dt_v = _mm256_set1_pd(dt);
    let mut i = 0;
    while i + lanes <= chunks {
        let all_gas = particles[i..i + lanes]
            .iter()
            .all(|p| p.ptype == ParticleType::Gas);
        if !all_gas {
            for lane in 0..lanes {
                if particles[i + lane].ptype == ParticleType::Gas {
                    particles[i + lane].psi_div =
                        particles[i + lane].psi_div * decay - c_h_sq_dt * div_b[i + lane];
                    particles[i + lane].b_field.x -= grad_psi[i + lane].x * dt;
                    particles[i + lane].b_field.y -= grad_psi[i + lane].y * dt;
                    particles[i + lane].b_field.z -= grad_psi[i + lane].z * dt;
                }
            }
            i += lanes;
            continue;
        }
        let psi = _mm256_set_pd(
            particles[i + 3].psi_div,
            particles[i + 2].psi_div,
            particles[i + 1].psi_div,
            particles[i].psi_div,
        );
        let div_b_v = _mm256_set_pd(div_b[i + 3], div_b[i + 2], div_b[i + 1], div_b[i]);
        let new_psi = _mm256_sub_pd(
            _mm256_mul_pd(psi, decay_v),
            _mm256_mul_pd(c_h_sq_dt_v, div_b_v),
        );
        let gp_x = _mm256_set_pd(
            grad_psi[i + 3].x,
            grad_psi[i + 2].x,
            grad_psi[i + 1].x,
            grad_psi[i].x,
        );
        let gp_y = _mm256_set_pd(
            grad_psi[i + 3].y,
            grad_psi[i + 2].y,
            grad_psi[i + 1].y,
            grad_psi[i].y,
        );
        let gp_z = _mm256_set_pd(
            grad_psi[i + 3].z,
            grad_psi[i + 2].z,
            grad_psi[i + 1].z,
            grad_psi[i].z,
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
        let new_bx = _mm256_sub_pd(bx, _mm256_mul_pd(dt_v, gp_x));
        let new_by = _mm256_sub_pd(by, _mm256_mul_pd(dt_v, gp_y));
        let new_bz = _mm256_sub_pd(bz, _mm256_mul_pd(dt_v, gp_z));
        let mut out_psi = [0.0f64; 4];
        let mut out_bx = [0.0f64; 4];
        let mut out_by = [0.0f64; 4];
        let mut out_bz = [0.0f64; 4];
        // SAFETY: fixed-size stack arrays have exactly four f64 lanes.
        unsafe {
            _mm256_storeu_pd(out_psi.as_mut_ptr(), new_psi);
            _mm256_storeu_pd(out_bx.as_mut_ptr(), new_bx);
            _mm256_storeu_pd(out_by.as_mut_ptr(), new_by);
            _mm256_storeu_pd(out_bz.as_mut_ptr(), new_bz);
        }
        for lane in 0..lanes {
            particles[i + lane].psi_div = out_psi[lane];
            particles[i + lane].b_field.x = out_bx[lane];
            particles[i + lane].b_field.y = out_by[lane];
            particles[i + lane].b_field.z = out_bz[lane];
        }
        i += lanes;
    }
    for j in chunks..n {
        if particles[j].ptype == ParticleType::Gas {
            particles[j].psi_div = particles[j].psi_div * decay - c_h_sq_dt * div_b[j];
            particles[j].b_field.x -= grad_psi[j].x * dt;
            particles[j].b_field.y -= grad_psi[j].y * dt;
            particles[j].b_field.z -= grad_psi[j].z * dt;
        }
    }
}
