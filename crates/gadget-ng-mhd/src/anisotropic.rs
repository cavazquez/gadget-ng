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
#[cfg(feature = "simd")]
use rayon::prelude::*;

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

#[cfg(not(feature = "simd"))]
#[expect(
    clippy::needless_range_loop,
    reason = "hot MHD pair loop indexes multiple SoA arrays"
)]
fn apply_anisotropic_conduction_impl(
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

            let dx = particles[j].position.x - pos_i.x;
            let dy = particles[j].position.y - pos_i.y;
            let dz = particles[j].position.z - pos_i.z;
            let r = (dx * dx + dy * dy + dz * dz).sqrt();
            if r < 1e-14 {
                continue;
            }

            let h_j = particles[j].smoothing_length.max(1e-10);
            let h_avg = 0.5 * (h_i + h_j);
            let w = kernel_w(r, 2.0 * h_avg);
            if w <= 0.0 {
                continue;
            }

            let rhat_x = dx / r;
            let rhat_y = dy / r;
            let rhat_z = dz / r;
            let cos_theta = bhat_i.x * rhat_x + bhat_i.y * rhat_y + bhat_i.z * rhat_z;
            let cos2 = cos_theta * cos_theta;

            let kappa_eff = kappa_perp + (kappa_par - kappa_perp) * cos2;
            let t_j = u_to_t(particles[j].internal_energy, gamma);
            let flux = kappa_eff * (t_j - t_i) * w * dt;

            delta_u[i] += flux;
        }
    }

    for i in 0..n {
        if particles[i].ptype == ParticleType::Gas {
            particles[i].internal_energy = (particles[i].internal_energy + delta_u[i]).max(0.0);
        }
    }
}

#[cfg(feature = "simd")]
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
    #[cfg(feature = "simd")]
    {
        apply_anisotropic_conduction_par(particles, kappa_par, kappa_perp, gamma, dt);
    }

    #[cfg(not(feature = "simd"))]
    {
        apply_anisotropic_conduction_impl(particles, kappa_par, kappa_perp, gamma, dt);
    }
}

#[cfg(not(feature = "simd"))]
#[expect(
    clippy::needless_range_loop,
    reason = "hot MHD pair loop indexes multiple SoA arrays"
)]
fn diffuse_cr_anisotropic_impl(
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

            let dx = particles[j].position.x - pos_i.x;
            let dy = particles[j].position.y - pos_i.y;
            let dz = particles[j].position.z - pos_i.z;
            let r = (dx * dx + dy * dy + dz * dz).sqrt();
            if r < 1e-14 {
                continue;
            }

            let w = kernel_w(r, 2.0 * h_i);
            if w <= 0.0 {
                continue;
            }

            let rhat_x = dx / r;
            let rhat_y = dy / r;
            let rhat_z = dz / r;
            let cos_theta = bhat_i.x * rhat_x + bhat_i.y * rhat_y + bhat_i.z * rhat_z;
            let cos2 = cos_theta * cos_theta;

            let kappa_aniso = kappa_eff_base * cos2;
            delta_cr[i] += kappa_aniso * (particles[j].cr_energy - e_i) * w * dt;
        }
    }

    for i in 0..n {
        if particles[i].ptype == ParticleType::Gas {
            particles[i].cr_energy = (particles[i].cr_energy + delta_cr[i]).max(0.0);
        }
    }
}

#[cfg(feature = "simd")]
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
#[cfg(feature = "simd")]
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
    #[cfg(feature = "simd")]
    {
        diffuse_cr_anisotropic_par(particles, kappa_cr, b_suppress, dt);
    }

    #[cfg(not(feature = "simd"))]
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
mod tests {
    use super::*;

    use approx::assert_abs_diff_eq;
    use gadget_ng_core::Vec3;

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
}
