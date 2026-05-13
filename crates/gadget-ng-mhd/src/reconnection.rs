//! Reconexión magnética: liberación de energía de campo B antiparalelos (Phase 145).
//!
//! ## Modelo Sweet-Parker
//!
//! La tasa de reconexión Sweet-Parker es:
//!
//! ```text
//! v_rec = v_A / sqrt(Rm)    donde Rm = L × v_A / η_eff
//! ```
//!
//! En la implementación SPH se usa una versión simplificada basada en la detección de
//! pares de partículas con líneas de B antiparalelas (B_i · B_j < 0) dentro del radio
//! de suavizado 2h. La energía magnética de las líneas antiparalelas se libera como
//! calor local:
//!
//! ```text
//! ΔE_heat = (|B_i|² + |B_j|²) / (2 μ₀) × f_rec × dt
//! ```
//!
//! El campo B de las partículas involucradas se reduce proporcionalmente:
//!
//! ```text
//! |B_i|_new = |B_i| × sqrt(1 − f_rec × dt)   (conservación de flujo)
//! ```
//!
//! ## Referencias
//!
//! Sweet (1958), Nuovo Cim. Suppl. 8 — modelo original de reconexión.
//! Parker (1957), JGR 62, 509 — tasa de reconexión Sweet-Parker.
//! Lazarian & Vishniac (1999), ApJ 517, 700 — reconexión en turbulencia MHD.

use crate::MU0;
use gadget_ng_core::{Particle, ParticleType};
#[cfg(feature = "rayon")]
use rayon::prelude::*;

#[cfg(not(feature = "rayon"))]
fn apply_magnetic_reconnection_impl(
    particles: &mut [Particle],
    f_reconnection: f64,
    _gamma: f64,
    dt: f64,
) {
    if f_reconnection <= 0.0 {
        return;
    }
    let n = particles.len();
    if n < 2 {
        return;
    }

    let mut delta_u = vec![0.0_f64; n];
    let mut b_scale = vec![1.0_f64; n];

    for i in 0..n {
        if particles[i].ptype != ParticleType::Gas {
            continue;
        }
        let h_i = particles[i].smoothing_length.max(1e-10);
        let b_i = particles[i].b_field;
        let b2_i = b_i.x * b_i.x + b_i.y * b_i.y + b_i.z * b_i.z;
        if b2_i < 1e-60 {
            continue;
        }
        let pos_i = particles[i].position;

        for j in (i + 1)..n {
            if particles[j].ptype != ParticleType::Gas {
                continue;
            }
            let h_j = particles[j].smoothing_length.max(1e-10);
            let b_j = particles[j].b_field;
            let b2_j = b_j.x * b_j.x + b_j.y * b_j.y + b_j.z * b_j.z;
            if b2_j < 1e-60 {
                continue;
            }

            let dx = particles[j].position.x - pos_i.x;
            let dy = particles[j].position.y - pos_i.y;
            let dz = particles[j].position.z - pos_i.z;
            let r2 = dx * dx + dy * dy + dz * dz;
            let h_avg = 0.5 * (h_i + h_j);

            if r2 > 4.0 * h_avg * h_avg {
                continue;
            }

            let b_dot = b_i.x * b_j.x + b_i.y * b_j.y + b_i.z * b_j.z;
            if b_dot >= 0.0 {
                continue;
            }

            let e_mag_pair = (b2_i + b2_j) / (2.0 * MU0);
            let de_heat = e_mag_pair * f_reconnection * dt;

            let rho_i = (particles[i].mass / (h_i * h_i * h_i)).max(1e-30);
            let rho_j = (particles[j].mass / (h_j * h_j * h_j)).max(1e-30);

            delta_u[i] += 0.5 * de_heat / rho_i;
            delta_u[j] += 0.5 * de_heat / rho_j;

            let b_decay = (1.0 - f_reconnection * dt).max(0.0).sqrt();
            b_scale[i] = b_scale[i].min(b_decay);
            b_scale[j] = b_scale[j].min(b_decay);
        }
    }

    for i in 0..n {
        if particles[i].ptype == ParticleType::Gas {
            particles[i].internal_energy = (particles[i].internal_energy + delta_u[i]).max(0.0);
            if b_scale[i] < 1.0 {
                particles[i].b_field.x *= b_scale[i];
                particles[i].b_field.y *= b_scale[i];
                particles[i].b_field.z *= b_scale[i];
            }
        }
    }
}

#[cfg(feature = "rayon")]
fn apply_magnetic_reconnection_par(
    particles: &mut [Particle],
    f_reconnection: f64,
    _gamma: f64,
    dt: f64,
) {
    if f_reconnection <= 0.0 {
        return;
    }
    let n = particles.len();
    if n < 2 {
        return;
    }

    let pos: Vec<gadget_ng_core::Vec3> = particles.iter().map(|p| p.position).collect();
    let mass: Vec<f64> = particles.iter().map(|p| p.mass).collect();
    let h_sml: Vec<f64> = particles
        .iter()
        .map(|p| p.smoothing_length.max(1e-10))
        .collect();
    let b_field: Vec<gadget_ng_core::Vec3> = particles.iter().map(|p| p.b_field).collect();
    let is_gas: Vec<bool> = particles
        .iter()
        .map(|p| p.ptype == ParticleType::Gas)
        .collect();

    let updates: Vec<Option<(f64, f64)>> = (0..n)
        .into_par_iter()
        .map(|i| {
            if !is_gas[i] {
                return None;
            }
            let h_i = h_sml[i];
            let b_i = b_field[i];
            let b2_i = b_i.x * b_i.x + b_i.y * b_i.y + b_i.z * b_i.z;
            if b2_i < 1e-60 {
                return Some((0.0, 1.0));
            }

            let rho_i = (mass[i] / (h_i * h_i * h_i)).max(1e-30);
            let mut delta_u_i = 0.0_f64;
            let mut b_scale_i = 1.0_f64;
            let b_decay = (1.0 - f_reconnection * dt).max(0.0).sqrt();

            for j in 0..n {
                if j == i || !is_gas[j] {
                    continue;
                }
                let b_j = b_field[j];
                let b2_j = b_j.x * b_j.x + b_j.y * b_j.y + b_j.z * b_j.z;
                if b2_j < 1e-60 {
                    continue;
                }

                let dx = pos[j].x - pos[i].x;
                let dy = pos[j].y - pos[i].y;
                let dz = pos[j].z - pos[i].z;
                let r2 = dx * dx + dy * dy + dz * dz;
                let h_avg = 0.5 * (h_i + h_sml[j]);

                if r2 > 4.0 * h_avg * h_avg {
                    continue;
                }

                let b_dot = b_i.x * b_j.x + b_i.y * b_j.y + b_i.z * b_j.z;
                if b_dot >= 0.0 {
                    continue;
                }

                let e_mag_pair = (b2_i + b2_j) / (2.0 * MU0);
                let de_heat = e_mag_pair * f_reconnection * dt;

                delta_u_i += 0.5 * de_heat / rho_i;
                b_scale_i = b_scale_i.min(b_decay);
            }
            Some((delta_u_i, b_scale_i))
        })
        .collect();

    for (p, update) in particles.iter_mut().zip(updates) {
        if let (true, Some((du, bs))) = (p.ptype == ParticleType::Gas, update) {
            p.internal_energy = (p.internal_energy + du).max(0.0);
            if bs < 1.0 {
                p.b_field.x *= bs;
                p.b_field.y *= bs;
                p.b_field.z *= bs;
            }
        }
    }
}

/// Aplica reconexión magnética entre pares de partículas antiparalelas (Phase 145).
///
/// Detecta pares `(i, j)` donde `B_i · B_j < 0` (campos antiparalelos) dentro de `2h`.
/// Libera una fracción `f_rec` de la energía magnética como calor en cada paso.
///
/// # Parámetros
///
/// - `particles`: slice mutable de partículas de gas
/// - `f_reconnection`: fracción de energía magnética liberada por paso (típico: 0.01)
/// - `gamma`: índice adiabático (para convertir ΔE a Δu)
/// - `dt`: paso de tiempo
pub fn apply_magnetic_reconnection(
    particles: &mut [Particle],
    f_reconnection: f64,
    gamma: f64,
    dt: f64,
) {
    #[cfg(feature = "rayon")]
    {
        apply_magnetic_reconnection_par(particles, f_reconnection, gamma, dt);
    }

    #[cfg(not(feature = "rayon"))]
    {
        apply_magnetic_reconnection_impl(particles, f_reconnection, gamma, dt);
    }
}

/// Tasa de reconexión Sweet-Parker teórica: `v_rec = v_A / sqrt(Rm)`.
///
/// - `v_a`: velocidad de Alfvén [unidades del código]
/// - `l_rec`: escala de la corriente de reconexión
/// - `eta_eff`: resistividad efectiva (numérica o física)
pub fn sweet_parker_rate(v_a: f64, l_rec: f64, eta_eff: f64) -> f64 {
    if eta_eff <= 0.0 || l_rec <= 0.0 {
        return 0.0;
    }
    let rm = l_rec * v_a / eta_eff;
    if rm <= 0.0 {
        return 0.0;
    }
    v_a / rm.sqrt()
}
