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
#[cfg(feature = "simd")]
use rayon::prelude::*;

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

#[cfg(not(feature = "simd"))]
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
    for i in 0..n {
        if particles[i].ptype == ParticleType::Gas {
            particles[i].psi_div = particles[i].psi_div * decay - c_h * c_h * div_b[i] * dt;
            particles[i].b_field.x -= grad_psi[i].x * dt;
            particles[i].b_field.y -= grad_psi[i].y * dt;
            particles[i].b_field.z -= grad_psi[i].z * dt;
        }
    }
}

#[cfg(feature = "simd")]
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
    #[cfg(feature = "simd")]
    {
        dedner_cleaning_step_par(particles, c_h, c_r, dt);
    }

    #[cfg(not(feature = "simd"))]
    {
        dedner_cleaning_step_impl(particles, c_h, c_r, dt);
    }
}
