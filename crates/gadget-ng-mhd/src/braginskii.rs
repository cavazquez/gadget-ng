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
#[expect(
    clippy::needless_range_loop,
    reason = "hot MHD pair loop indexes multiple SoA arrays"
)]
fn apply_braginskii_viscosity_impl(particles: &mut [Particle], eta_visc: f64, dt: f64) {
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

            let cos_theta = bhat.x * rhat_x + bhat.y * rhat_y + bhat.z * rhat_z;
            let cos2 = cos_theta * cos_theta;

            let dvx = particles[j].velocity.x - vel_i.x;
            let dvy = particles[j].velocity.y - vel_i.y;
            let dvz = particles[j].velocity.z - vel_i.z;
            let dv_par = dvx * bhat.x + dvy * bhat.y + dvz * bhat.z;

            let rho_j = (particles[j].mass / (h_j * h_j * h_j)).max(1e-30);
            let factor = eta_visc * particles[j].mass / rho_j * cos2 * w;

            dv[i].x += factor * dv_par * bhat.x * dt;
            dv[i].y += factor * dv_par * bhat.y * dt;
            dv[i].z += factor * dv_par * bhat.z * dt;
        }
    }

    for i in 0..n {
        if particles[i].ptype == ParticleType::Gas {
            particles[i].velocity.x += dv[i].x;
            particles[i].velocity.y += dv[i].y;
            particles[i].velocity.z += dv[i].z;
        }
    }
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
