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

/// Aplica las fuerzas magnéticas (tensor de Maxwell SPH) a las partículas de gas (Phase 124).
///
/// Para cada par (i, j) de partículas de gas, acumula la aceleración magnética:
///
/// ```text
/// a_i += m_j (M_i/ρ_i² + M_j/ρ_j²) · ∇W_ij
/// ```
#[allow(clippy::needless_range_loop)]
pub fn apply_magnetic_forces(particles: &mut [Particle], dt: f64) {
    let n = particles.len();
    let mut acc_mag = vec![Vec3::zero(); n];

    // Precalcular densidades y tensores de Maxwell
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

    for i in 0..n {
        if particles[i].ptype != ParticleType::Gas {
            continue;
        }
        let rho_i2 = rho[i] * rho[i];
        let m_i = &maxwell[i];

        for j in (i + 1)..n {
            if particles[j].ptype != ParticleType::Gas {
                continue;
            }
            let rho_j2 = rho[j] * rho[j];
            let m_j = &maxwell[j];

            let r_ij = Vec3 {
                x: particles[j].position.x - particles[i].position.x,
                y: particles[j].position.y - particles[i].position.y,
                z: particles[j].position.z - particles[i].position.z,
            };
            let h_ij =
                0.5 * (particles[i].smoothing_length + particles[j].smoothing_length).max(1e-10);
            let grad_w = kernel_gradient(r_ij, h_ij);
            let gw = [grad_w.x, grad_w.y, grad_w.z];

            // a_contrib = (M_i/ρ_i² + M_j/ρ_j²) · ∇W_ij
            let mut a = [0.0_f64; 3];
            for k in 0..3 {
                for l in 0..3 {
                    a[k] += (m_i[k][l] / rho_i2 + m_j[k][l] / rho_j2) * gw[l];
                }
            }

            let m_j_mass = particles[j].mass;
            let m_i_mass = particles[i].mass;

            acc_mag[i].x += m_j_mass * a[0];
            acc_mag[i].y += m_j_mass * a[1];
            acc_mag[i].z += m_j_mass * a[2];

            acc_mag[j].x -= m_i_mass * a[0];
            acc_mag[j].y -= m_i_mass * a[1];
            acc_mag[j].z -= m_i_mass * a[2];
        }
    }

    // Integración de Euler: v += a * dt
    for i in 0..n {
        if particles[i].ptype != ParticleType::Gas {
            continue;
        }
        particles[i].velocity.x += acc_mag[i].x * dt;
        particles[i].velocity.y += acc_mag[i].y * dt;
        particles[i].velocity.z += acc_mag[i].z * dt;
    }
}

#[cfg(test)]
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
}
