//! Esquema de Dedner div-B cleaning (Phase 125).
//!
//! ## Formulación
//!
//! El esquema de Dedner et al. (2002) introduce un campo escalar `ψ` que
//! transporta y disipa el error de divergencia `∇·B`:
//!
//! ```text
//! ∂B/∂t + ∇ψ = 0
//! ∂ψ/∂t + c_h² ∇·B = -c_r ψ
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
//! ψ_new = ψ × exp(-c_r × dt)  [disipación]
//! B_new  = B − ∇ψ × dt         [corrección del campo]
//! ```
//!
//! ## Referencia
//!
//! Dedner et al. (2002), J. Comput. Phys. 175, 645–673.
//! Tricco & Price (2012), J. Comput. Phys. 231, 7214.

use gadget_ng_core::{Particle, ParticleType, Vec3};

/// Gradiente SPH del campo escalar ψ para un par (i, j).
fn grad_w_scalar(r_vec: Vec3, h: f64) -> Vec3 {
    let r2 = r_vec.x * r_vec.x + r_vec.y * r_vec.y + r_vec.z * r_vec.z;
    let r = r2.sqrt();
    if r < 1e-10 || h <= 0.0 { return Vec3::zero(); }
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
    Vec3 { x: dw_dr * r_vec.x / r, y: dw_dr * r_vec.y / r, z: dw_dr * r_vec.z / r }
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
/// 1. Calcula la divergencia SPH de B para cada partícula: `div_B_i = Σ_j (m_j/ρ_j) (B_j - B_i)·∇W_ij`.
/// 2. Actualiza ψ: `ψ_new = ψ × exp(-c_r × dt) − c_h² × div_B × dt`.
/// 3. Calcula el gradiente SPH de ψ: `∇ψ_i = Σ_j (m_j/ρ_j) (ψ_j - ψ_i) ∇W_ij`.
/// 4. Corrige B: `B_new = B − ∇ψ × dt`.
pub fn dedner_cleaning_step(
    particles: &mut [Particle],
    c_h: f64,
    c_r: f64,
    dt: f64,
) {
    let n = particles.len();

    // Precalcular densidades
    let rho: Vec<f64> = particles.iter().map(|p| {
        let h = p.smoothing_length.max(1e-10);
        (p.mass / (h * h * h)).max(1e-30)
    }).collect();

    // Paso 1: div_B SPH para cada partícula i
    let mut div_b = vec![0.0_f64; n];
    let mut grad_psi = vec![Vec3::zero(); n];

    for i in 0..n {
        if particles[i].ptype != ParticleType::Gas { continue; }
        let b_i = particles[i].b_field;
        let psi_i = particles[i].psi_div;

        for j in 0..n {
            if i == j { continue; }
            if particles[j].ptype != ParticleType::Gas { continue; }

            let b_j = particles[j].b_field;
            let psi_j = particles[j].psi_div;
            let h_ij = 0.5 * (particles[i].smoothing_length + particles[j].smoothing_length).max(1e-10);
            let r_ij = Vec3 {
                x: particles[j].position.x - particles[i].position.x,
                y: particles[j].position.y - particles[i].position.y,
                z: particles[j].position.z - particles[i].position.z,
            };
            let grad_w = grad_w_scalar(r_ij, h_ij);
            let factor = particles[j].mass / rho[j];

            // div_B_i += (m_j/ρ_j) (B_j - B_i) · ∇W_ij
            let db = Vec3 { x: b_j.x - b_i.x, y: b_j.y - b_i.y, z: b_j.z - b_i.z };
            div_b[i] += factor * (db.x * grad_w.x + db.y * grad_w.y + db.z * grad_w.z);

            // grad_ψ_i += (m_j/ρ_j) (ψ_j - ψ_i) ∇W_ij
            let dpsi = psi_j - psi_i;
            grad_psi[i].x += factor * dpsi * grad_w.x;
            grad_psi[i].y += factor * dpsi * grad_w.y;
            grad_psi[i].z += factor * dpsi * grad_w.z;
        }
    }

    // Paso 2 & 3: actualizar ψ y B
    let decay = (-c_r * dt).exp();
    for i in 0..n {
        if particles[i].ptype != ParticleType::Gas { continue; }
        // ψ_new = ψ × exp(-c_r dt) − c_h² × div_B × dt
        particles[i].psi_div = particles[i].psi_div * decay - c_h * c_h * div_b[i] * dt;
        // B_new = B − ∇ψ × dt
        particles[i].b_field.x -= grad_psi[i].x * dt;
        particles[i].b_field.y -= grad_psi[i].y * dt;
        particles[i].b_field.z -= grad_psi[i].z * dt;
    }
}
