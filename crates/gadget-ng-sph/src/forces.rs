//! Aceleraciones y tasa de cambio de energía interna / entropía SPH.
//!
//! ## Ecuaciones de movimiento SPH (forma simétrica de Springel & Hernquist 2002)
//!
//! ```text
//! dv_i/dt = −Σ_j m_j [f_i P_i/ρ_i² ∇W(r_ij, h_i) + f_j P_j/ρ_j² ∇W(r_ij, h_j)]
//! du_i/dt =  Σ_j m_j f_i P_i/ρ_i² v_ij · ∇W(r_ij, h_i)
//! ```
//!
//! donde `∇W(r, h) = grad_w(r, h) · r̂ / h` y `v_ij = v_i − v_j`.
//!
//! ## Viscosidad artificial clásica (Monaghan 1997)
//!
//! Se añade un término de viscosidad para evitar interpenetración en shocks:
//!
//! ```text
//! Π_ij = −α c_s_ij μ_ij / ρ̄_ij   si v_ij · r_ij < 0
//! μ_ij = h̄ v_ij · r_ij / (|r_ij|² + ε² h̄²)
//! c_s   = sqrt(γ P / ρ)
//! ```
//!
//! ## Viscosidad Gadget-2 (velocidad de señal + limitador de Balsara)
//!
//! En `compute_sph_forces_gadget2` se utiliza la viscosidad por velocidad de señal
//! (Gadget-2, ec. 14) con el limitador de Balsara:
//!
//! ```text
//! v_sig_ij = α (c_i + c_j − 3 w_ij) / 2    [solo si v_ij·r_ij < 0]
//! w_ij     = |v_ij · r̂_ij|
//! Π_ij     = −v_sig_ij · w_ij / ρ̄_ij  · (f_i + f_j)/2
//! dA_i/dt  = (γ-1)/(2ρ_i^(γ-1)) Σ_j m_j Π_ij v_ij · ∇W̄_ij
//! ```

use crate::density::GAMMA;
use crate::kernel::grad_w;
use crate::particle::SphParticle;
use gadget_ng_core::Vec3;

/// Regularización de μ_ij (viscosidad clásica de Monaghan).
const EPS_VISC: f64 = 0.01;

/// Calcula `acc_sph` y `du_dt` para todas las partículas de gas.
///
/// Requiere que `compute_density` haya sido llamado previamente.
pub fn compute_sph_forces(particles: &mut [SphParticle]) {
    let n = particles.len();
    let pos: Vec<Vec3> = particles.iter().map(|p| p.position).collect();
    let vel: Vec<Vec3> = particles.iter().map(|p| p.velocity).collect();
    let mass: Vec<f64> = particles.iter().map(|p| p.mass).collect();
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
        if particles[i].gas.is_none() {
            continue;
        }
        if rho[i] < 1e-200 {
            continue;
        }

        let pi_rho2 = pressure[i] / (rho[i] * rho[i]); // P_i / ρ_i²
        let cs_i = (GAMMA * pressure[i] / rho[i]).sqrt().max(0.0); // velocidad del sonido i

        let mut acc = Vec3::zero();
        let mut dudt = 0.0_f64;

        for j in 0..n {
            if j == i || particles[j].gas.is_none() || rho[j] < 1e-200 {
                continue;
            }

            let r_ij = pos[i] - pos[j];
            let r = r_ij.norm();
            let h_i = h_sml[i];
            let h_j = h_sml[j];

            // Gradientes del kernel (simétrico)
            let gw_i = grad_w(r, h_i);
            let gw_j = grad_w(r, h_j);
            if gw_i == 0.0 && gw_j == 0.0 {
                continue;
            }

            let r_hat = if r > 1e-300 {
                r_ij * (1.0 / r)
            } else {
                Vec3::zero()
            };
            let nabla_w_i = r_hat * (gw_i / h_i);
            let nabla_w_j = r_hat * (gw_j / h_j);

            let pj_rho2 = pressure[j] / (rho[j] * rho[j]);

            // Viscosidad artificial
            let v_ij = vel[i] - vel[j];
            let h_bar = 0.5 * (h_i + h_j);
            let mu_ij = if v_ij.dot(r_ij) < 0.0 {
                let cs_j = (GAMMA * pressure[j] / rho[j]).sqrt().max(0.0);
                let cs_bar = 0.5 * (cs_i + cs_j);
                let rho_bar = 0.5 * (rho[i] + rho[j]);
                let mu = h_bar * v_ij.dot(r_ij) / (r * r + EPS_VISC * h_bar * h_bar);
                -ALPHA_VISC * cs_bar * mu / rho_bar
            } else {
                0.0
            };

            // Aceleración SPH simétrica
            let coeff_i = pi_rho2 + 0.5 * mu_ij;
            let coeff_j = pj_rho2 + 0.5 * mu_ij;
            acc -= (nabla_w_i * coeff_i + nabla_w_j * coeff_j) * mass[j];

            // Tasa de energía interna (sólo término i)
            dudt += mass[j] * pi_rho2 * v_ij.dot(nabla_w_i);
        }

        if let Some(gas) = particles[i].gas.as_mut() {
            gas.acc_sph = acc;
            gas.du_dt = dudt;
        }
    }
}

/// Fuerzas SPH con formulación Gadget-2 completa.
///
/// Implementa la viscosidad artificial por **velocidad de señal** (Gadget-2, ec. 14)
/// combinada con el **limitador de Balsara** y evolución de la **función entrópica**
/// A = P/ρ^γ (Springel & Hernquist 2002).
///
/// ## Prerrequisitos
///
/// Antes de llamar a esta función deben haberse ejecutado:
/// 1. `compute_density` — calcula `rho`, `pressure`, `entropy`, `h_sml`.
/// 2. `compute_balsara_factors` — calcula `balsara` para cada partícula.
///
/// ## Salidas (por partícula de gas)
///
/// - `gas.acc_sph` — aceleración hidrodinámica.
/// - `gas.da_dt`   — tasa de cambio de entropía A por calentamiento viscoso.
/// - `gas.max_vsig` — velocidad de señal máxima (para timestep de Courant).
pub fn compute_sph_forces_gadget2(particles: &mut [SphParticle]) {
    let n = particles.len();

    let pos: Vec<Vec3> = particles.iter().map(|p| p.position).collect();
    let vel: Vec<Vec3> = particles.iter().map(|p| p.velocity).collect();
    let mass: Vec<f64> = particles.iter().map(|p| p.mass).collect();
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
    let balsara: Vec<f64> = particles
        .iter()
        .map(|p| p.gas.as_ref().map(|g| g.balsara).unwrap_or(1.0))
        .collect();

    for i in 0..n {
        if particles[i].gas.is_none() || rho[i] < 1e-200 {
            continue;
        }

        let pi_rho2 = pressure[i] / (rho[i] * rho[i]);
        let cs_i = (GAMMA * pressure[i] / rho[i]).sqrt().max(0.0);
        let fi = balsara[i];

        let mut acc = Vec3::zero();
        let mut da_dt = 0.0_f64;
        let mut max_vsig = 0.0_f64;

        for j in 0..n {
            if j == i || particles[j].gas.is_none() || rho[j] < 1e-200 {
                continue;
            }

            let r_ij = pos[i] - pos[j];
            let r = r_ij.norm();
            let hi = h_sml[i];
            let hj = h_sml[j];

            let gw_i = grad_w(r, hi);
            let gw_j = grad_w(r, hj);
            if gw_i == 0.0 && gw_j == 0.0 {
                continue;
            }

            let r_hat = if r > 1e-300 { r_ij * (1.0 / r) } else { Vec3::zero() };
            let nabla_w_i = r_hat * (gw_i / hi);
            let nabla_w_j = r_hat * (gw_j / hj);
            // Gradiente promediado para el término viscoso
            let nabla_w_bar = (nabla_w_i + nabla_w_j) * 0.5;

            let v_ij = vel[i] - vel[j];
            let w_ij = v_ij.dot(r_hat); // proyección radial de v_ij

            // ── Viscosidad por velocidad de señal (Gadget-2 ec. 14) ──────────
            let (pi_visc, vsig) = if w_ij < 0.0 {
                let cs_j = (GAMMA * pressure[j] / rho[j]).sqrt().max(0.0);
                // v_sig = α(c_i + c_j − 3 w_ij)/2  [Gadget-2 eq. 14]
                let vsig_ij = ALPHA_VISC * (cs_i + cs_j - 3.0 * w_ij) * 0.5;
                let rho_bar = 0.5 * (rho[i] + rho[j]);
                let fij = 0.5 * (fi + balsara[j]); // Balsara promediado
                let pi_ij = -fij * vsig_ij * w_ij / rho_bar;
                (pi_ij, vsig_ij)
            } else {
                (0.0, 0.0)
            };

            max_vsig = max_vsig.max(vsig);

            let pj_rho2 = pressure[j] / (rho[j] * rho[j]);

            // ── Aceleración hidrodinámica (forma simétrica SH02) ─────────────
            let coeff_i = pi_rho2 + 0.5 * pi_visc;
            let coeff_j = pj_rho2 + 0.5 * pi_visc;
            acc -= (nabla_w_i * coeff_i + nabla_w_j * coeff_j) * mass[j];

            // ── Calentamiento viscoso: dA/dt = (γ-1)/(2ρ^(γ-1)) Σ m_j Π_ij v_ij·∇W̄
            da_dt += mass[j] * pi_visc * v_ij.dot(nabla_w_bar);
        }

        // Factor (γ-1)/(2ρ_i^(γ-1)) para convertir a tasa de entropía
        let da_factor = if rho[i] > 0.0 {
            (GAMMA - 1.0) * 0.5 / rho[i].powf(GAMMA - 1.0)
        } else {
            0.0
        };

        if let Some(gas) = particles[i].gas.as_mut() {
            gas.acc_sph = acc;
            gas.da_dt = da_factor * da_dt;
            gas.max_vsig = max_vsig;
            // du_dt también: para compatibilidad con integrador clásico
            gas.du_dt = pressure[i] / (rho[i] * rho[i]) * da_dt;
        }
    }
}

/// Parámetro de viscosidad artificial (compartido entre ambos integradores).
const ALPHA_VISC: f64 = 1.0;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::density::compute_density;
    use crate::particle::SphParticle;
    use gadget_ng_core::Vec3;

    /// En un sistema en reposo con gradiente de presión nulo, la aceleración SPH neta
    /// debe ser pequeña (simetría: cada par i-j contribuye en sentidos opuestos).
    #[test]
    fn uniform_rest_gas_near_zero_acceleration() {
        let n_side = 4usize;
        let box_size = 4.0_f64;
        let dx = box_size / n_side as f64;
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
                SphParticle::new_gas(k, 1.0, pos, Vec3::zero(), 1.0, 2.0 * dx)
            })
            .collect();

        compute_density(&mut parts);
        compute_sph_forces(&mut parts);

        let max_acc = parts
            .iter()
            .filter_map(|p| p.gas.as_ref())
            .map(|g| g.acc_sph.norm())
            .fold(0.0_f64, f64::max);

        // Para un gas uniforme en reposo el desequilibrio de borde es pequeño.
        assert!(
            max_acc < 5.0,
            "aceleración SPH máxima para gas en reposo = {max_acc:.4}"
        );
    }
}
