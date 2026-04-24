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

use gadget_ng_core::{Particle, ParticleType, Vec3};

/// Gradiente del kernel SPH cúbico (en 3D): `∇W(r, h)`.
///
/// Devuelve el gradiente evaluado en la dirección `r_ij = r_j - r_i`.
fn kernel_gradient(r_vec: Vec3, h: f64) -> Vec3 {
    let r2 = r_vec.x * r_vec.x + r_vec.y * r_vec.y + r_vec.z * r_vec.z;
    let r = r2.sqrt();
    if r < 1e-10 || h <= 0.0 { return Vec3::zero(); }
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

/// Avanza el campo magnético de todas las partículas de gas un paso `dt`
/// usando la ecuación de inducción SPH (Phase 123).
///
/// La densidad SPH se aproxima como `ρ ≈ mass/h³` (consistente con el resto del código).
pub fn advance_induction(particles: &mut [Particle], dt: f64) {
    let n = particles.len();
    let mut db = vec![Vec3::zero(); n];

    for i in 0..n {
        if particles[i].ptype != ParticleType::Gas { continue; }
        let h_i = particles[i].smoothing_length.max(1e-10);
        let rho_i = (particles[i].mass / (h_i * h_i * h_i)).max(1e-30);
        let b_i = particles[i].b_field;
        let v_i = particles[i].velocity;

        for j in 0..n {
            if i == j { continue; }
            if particles[j].ptype != ParticleType::Gas { continue; }

            let h_j = particles[j].smoothing_length.max(1e-10);
            let rho_j = (particles[j].mass / (h_j * h_j * h_j)).max(1e-30);
            let v_j = particles[j].velocity;
            let b_j = particles[j].b_field;

            let r_ij = Vec3 {
                x: particles[j].position.x - particles[i].position.x,
                y: particles[j].position.y - particles[i].position.y,
                z: particles[j].position.z - particles[i].position.z,
            };

            // ∇W evaluado en r_ij con h promedio
            let h_ij = 0.5 * (h_i + h_j);
            let grad_w = kernel_gradient(r_ij, h_ij);

            // v_ij = v_i - v_j
            let v_ij = Vec3 { x: v_i.x - v_j.x, y: v_i.y - v_j.y, z: v_i.z - v_j.z };
            let b_ij = Vec3 { x: b_i.x - b_j.x, y: b_i.y - b_j.y, z: b_i.z - b_j.z };

            // Contribución de j a dB_i/dt (formulación simétrica):
            // dB_i/dt += (m_j/ρ_j) [(B_ij·∇W_ij) v_ij - (v_ij·∇W_ij) B_ij]
            let b_dot_grad = b_ij.x * grad_w.x + b_ij.y * grad_w.y + b_ij.z * grad_w.z;
            let v_dot_grad = v_ij.x * grad_w.x + v_ij.y * grad_w.y + v_ij.z * grad_w.z;
            let factor = particles[j].mass / rho_j;

            db[i].x += factor * (b_dot_grad * v_ij.x - v_dot_grad * b_ij.x);
            db[i].y += factor * (b_dot_grad * v_ij.y - v_dot_grad * b_ij.y);
            db[i].z += factor * (b_dot_grad * v_ij.z - v_dot_grad * b_ij.z);
        }
        // Factor de normalización para consistencia con SPH estándar
        let _ = rho_i; // rho_i usada para verificación futura (supresión de warning)
    }

    // Integración explícita de Euler
    for i in 0..n {
        if particles[i].ptype != ParticleType::Gas { continue; }
        particles[i].b_field.x += db[i].x * dt;
        particles[i].b_field.y += db[i].y * dt;
        particles[i].b_field.z += db[i].z * dt;
    }
}
