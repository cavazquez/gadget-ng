//! Factor de Balsara para viscosidad artificial selectiva (Balsara 1995).
//!
//! ## Motivación
//!
//! La viscosidad artificial de Monaghan controla bien los shocks pero introduce
//! disipación excesiva en flujos de cizallamiento (rotación diferencial, vórtices).
//! El limitador de Balsara multiplica la viscosidad por `(f_i + f_j)/2` donde:
//!
//! ```text
//! f_i = |∇·v_i| / (|∇·v_i| + |∇×v_i| + ε_bal · c_s_i / h_i)
//! ```
//!
//! - En un shock puro: `|∇·v| >> |∇×v|` → `f_i ≈ 1` (viscosidad completa).
//! - En cizallamiento puro: `|∇×v| >> |∇·v|` → `f_i ≈ 0` (viscosidad suprimida).
//!
//! ## Estimadores SPH
//!
//! Se usan los estimadores consistentes (Monaghan & Price 2004):
//!
//! ```text
//! (∇·v)_i  = (1/ρ_i) Σ_j m_j (v_j − v_i) · ∇_i W(r_ij, h_i)
//! (∇×v)_i  = (1/ρ_i) Σ_j m_j ∇_i W(r_ij, h_i) × (v_j − v_i)
//! ```
//!
//! donde `∇_i W(r_ij, h_i) = grad_w(r, h_i) / h_i · r̂_ij`.

use crate::density::GAMMA;
use crate::kernel::grad_w;
use crate::particle::SphParticle;
use gadget_ng_core::Vec3;

/// Regularización para el factor Balsara (evita división por cero).
const EPS_BAL: f64 = 0.0001;

/// Producto vectorial a × b (Vec3 no incluye este método en gadget-ng-core).
#[inline]
fn cross(a: Vec3, b: Vec3) -> Vec3 {
    Vec3::new(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    )
}

/// Calcula el factor Balsara `f_i` para cada partícula de gas.
///
/// Actualiza `gas.balsara` in-place. Requiere que `compute_density` haya sido
/// llamado previamente (necesita `rho`, `pressure`, `h_sml`).
pub fn compute_balsara_factors(particles: &mut [SphParticle]) {
    let n = particles.len();

    // Extrae datos inmutables para evitar borrow doble.
    let pos: Vec<Vec3> = particles.iter().map(|p| p.position).collect();
    let vel: Vec<Vec3> = particles.iter().map(|p| p.velocity).collect();
    let mass: Vec<f64> = particles.iter().map(|p| p.mass).collect();
    let is_gas: Vec<bool> = particles.iter().map(|p| p.gas.is_some()).collect();
    let rho: Vec<f64> = particles
        .iter()
        .map(|p| p.gas.as_ref().map(|g| g.rho).unwrap_or(0.0))
        .collect();
    let pressure: Vec<f64> = particles
        .iter()
        .map(|p| p.gas.as_ref().map(|g| g.pressure).unwrap_or(0.0))
        .collect();
    let h_sml: Vec<f64> = particles
        .iter()
        .map(|p| p.gas.as_ref().map(|g| g.h_sml).unwrap_or(1.0))
        .collect();

    for i in 0..n {
        if !is_gas[i] {
            continue;
        }

        let pi = pos[i];
        let vi = vel[i];
        let hi = h_sml[i];

        if rho[i] < 1e-200 {
            if let Some(gas) = particles[i].gas.as_mut() {
                gas.balsara = 1.0;
            }
            continue;
        }

        let inv_rho_i = 1.0 / rho[i];

        let mut div_v = 0.0_f64;         // ∇·v  (escalar)
        let mut curl_v = Vec3::zero();   // ∇×v  (vector 3D)

        for j in 0..n {
            if j == i || !is_gas[j] || rho[j] < 1e-200 {
                continue;
            }
            let r_ij = pi - pos[j];
            let r = r_ij.norm();

            let gw = grad_w(r, hi);
            if gw == 0.0 {
                continue;
            }

            let r_hat = if r > 1e-300 { r_ij * (1.0 / r) } else { Vec3::zero() };
            // ∇_i W(r_ij, h_i)  (apunta de j hacia i)
            let nabla_w = r_hat * (gw / hi);

            let dv = vel[j] - vi; // v_j − v_i

            // ∇·v  =  (1/ρ_i) Σ m_j (v_j−v_i)·∇W_ij
            div_v += mass[j] * dv.dot(nabla_w);

            // ∇×v  =  (1/ρ_i) Σ m_j ∇W_ij × (v_j−v_i)
            curl_v += cross(nabla_w, dv) * mass[j];
        }

        div_v *= inv_rho_i;
        curl_v = curl_v * inv_rho_i;

        let abs_div = div_v.abs();
        let abs_curl = curl_v.norm();
        let cs_i = (GAMMA * pressure[i] / rho[i]).sqrt().max(0.0);
        let eps_term = EPS_BAL * cs_i / hi;

        let balsara_val = abs_div / (abs_div + abs_curl + eps_term);
        if let Some(gas) = particles[i].gas.as_mut() {
            gas.balsara = balsara_val;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::density::compute_density;
    use crate::particle::SphParticle;
    use gadget_ng_core::Vec3;

    /// En un flujo de cizallamiento puro (v_x = y), el factor Balsara debe ser
    /// significativamente menor que 1 (viscosidad suprimida).
    #[test]
    fn balsara_suppressed_in_shear_flow() {
        let n_side = 4usize;
        let box_size = 4.0_f64;
        let dx = box_size / n_side as f64;

        // Grilla cúbica con velocidad de cizallamiento v_x = y.
        let mut parts: Vec<SphParticle> = (0..n_side.pow(3))
            .map(|k| {
                let iz = k / (n_side * n_side);
                let iy = (k / n_side) % n_side;
                let ix = k % n_side;
                let x = (ix as f64 + 0.5) * dx;
                let y = (iy as f64 + 0.5) * dx;
                let z = (iz as f64 + 0.5) * dx;
                let vx = y; // cizallamiento puro: dv_x/dy = 1
                SphParticle::new_gas(k, 1.0, Vec3::new(x, y, z), Vec3::new(vx, 0.0, 0.0), 1.0, 2.0 * dx)
            })
            .collect();

        compute_density(&mut parts);
        compute_balsara_factors(&mut parts);

        // Las partículas internas (lejos del borde) deben tener f < 0.3
        let mean_f: f64 = parts
            .iter()
            .filter(|p| {
                let pos = p.position;
                pos.x > dx && pos.x < box_size - dx &&
                pos.y > dx && pos.y < box_size - dx &&
                pos.z > dx && pos.z < box_size - dx
            })
            .filter_map(|p| p.gas.as_ref().map(|g| g.balsara))
            .sum::<f64>()
            / 8.0; // 2³ partículas internas en un cubo 4³

        assert!(
            mean_f < 0.5,
            "Balsara no suprimido en cizallamiento: f_mean = {mean_f:.4}"
        );
    }

    /// En un flujo compresivo puro (convergente uniforme), el factor Balsara
    /// debe ser cercano a 1 (viscosidad activa).
    #[test]
    fn balsara_active_in_compression() {
        let n_side = 4usize;
        let box_size = 4.0_f64;
        let dx = box_size / n_side as f64;

        // Compresión radial: v = -0.5 · r (∇·v = -1.5, ∇×v = 0)
        let center = Vec3::new(2.0, 2.0, 2.0);
        let mut parts: Vec<SphParticle> = (0..n_side.pow(3))
            .map(|k| {
                let iz = k / (n_side * n_side);
                let iy = (k / n_side) % n_side;
                let ix = k % n_side;
                let pos = Vec3::new(
                    (ix as f64 + 0.5) * dx,
                    (iy as f64 + 0.5) * dx,
                    (iz as f64 + 0.5) * dx,
                );
                let vel = (pos - center) * (-0.5); // convergente
                SphParticle::new_gas(k, 1.0, pos, vel, 1.0, 2.0 * dx)
            })
            .collect();

        compute_density(&mut parts);
        compute_balsara_factors(&mut parts);

        let mean_f: f64 = parts
            .iter()
            .filter_map(|p| p.gas.as_ref().map(|g| g.balsara))
            .sum::<f64>()
            / n_side.pow(3) as f64;

        assert!(
            mean_f > 0.5,
            "Balsara inactivo en compresión: f_mean = {mean_f:.4}"
        );
    }
}
