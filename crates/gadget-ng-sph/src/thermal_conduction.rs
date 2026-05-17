//! Conducción térmica del gas intracúmulo (ICM) — Spitzer con supresión (Phase 121).
//!
//! ## Modelo
//!
//! La conducción térmica de Spitzer (1962) transporta calor desde regiones calientes
//! a frías en el plasma del ICM. La tasa de transferencia de calor entre dos partículas:
//!
//! ```text
//! q_ij = κ_eff × (T_j − T_i) × W(r_ij, h_i) × Δt
//! ```
//!
//! donde la conductividad efectiva es:
//! ```text
//! κ_eff = κ_Spitzer × ψ × T_mean^{5/2}
//! ```
//!
//! con `ψ ∈ \[0,1\]` el factor de supresión por campo magnético o turbulencia.
//! En cúmulos de galaxias, ψ ≈ 0.1–0.3 (Narayan & Medvedev 2001).
//!
//! ## Conservación de energía
//!
//! El calor fluye simétricamente: lo que gana `i` lo pierde `j`.
//! Para evitar aliasing, acumulamos el flujo neto antes de aplicar.
//!
//! ## Referencia
//!
//! Spitzer (1962), Physics of Fully Ionized Gases.
//! Narayan & Medvedev (2001) ApJ 562, L129.
//! Dolag et al. (2004) ApJ 606, L97.

use crate::cooling::{temperature_to_u, u_to_temperature};
use crate::periodic_delta;
use gadget_ng_core::{ConductionSection, Particle, ParticleType};
#[cfg(feature = "rayon")]
use rayon::prelude::*;

/// Logaritmo de Coulomb típico para plasma del ICM.
const COULOMB_LOG: f64 = 37.0;

/// Kernel SPH suavizado (Wendland C2 simplificado) para conducción.
#[inline]
fn kernel_cond(r: f64, h: f64) -> f64 {
    if h <= 0.0 || r >= 2.0 * h {
        return 0.0;
    }
    let q = r / h;
    let t = 1.0 - 0.5 * q;
    (21.0 / (2.0 * std::f64::consts::PI)) / (h * h * h) * t.powi(4) * (1.0 + 2.0 * q)
}

/// Aplica conducción térmica de Spitzer entre partículas de gas vecinas (Phase 121).
///
/// Para cada par (i, j) de partículas de gas dentro del radio de suavizado:
/// 1. Convierte `u` a temperatura `T`.
/// 2. Calcula `κ_eff = kappa_spitzer × ψ × T_mean^{5/2} / log_Coulomb`.
/// 3. Calcula flujo: `q_ij = κ_eff × (T_j - T_i) × w(r_ij)`.
/// 4. Aplica simétricamente conservando energía total.
///
/// La temperatura de floor impide enfriamiento excesivo.
pub fn apply_thermal_conduction(
    particles: &mut [Particle],
    cfg: &ConductionSection,
    gamma: f64,
    t_floor_k: f64,
    dt: f64,
) {
    apply_thermal_conduction_periodic(particles, cfg, gamma, t_floor_k, dt, None);
}

/// Igual que `apply_thermal_conduction`, usando imagen mínima si `periodic_box = Some(L)`.
pub fn apply_thermal_conduction_periodic(
    particles: &mut [Particle],
    cfg: &ConductionSection,
    gamma: f64,
    t_floor_k: f64,
    dt: f64,
    periodic_box: Option<f64>,
) {
    if !cfg.enabled {
        return;
    }

    let n = particles.len();

    let u_floor = temperature_to_u(t_floor_k, gamma);

    #[cfg(feature = "rayon")]
    {
        let ptypes: Vec<ParticleType> = particles.iter().map(|p| p.ptype).collect();
        let pos: Vec<_> = particles.iter().map(|p| p.position).collect();
        let h: Vec<f64> = particles
            .iter()
            .map(|p| p.smoothing_length.max(1e-10))
            .collect();
        let t_arr: Vec<f64> = particles
            .iter()
            .map(|p| u_to_temperature(p.internal_energy.max(0.0), gamma))
            .collect();

        let delta_u: Vec<f64> = (0..n)
            .into_par_iter()
            .map(|i| {
                if ptypes[i] != ParticleType::Gas {
                    return 0.0;
                }
                let mut sum = 0.0_f64;
                for j in 0..n {
                    if i == j || ptypes[j] != ParticleType::Gas {
                        continue;
                    }
                    let r = periodic_delta(pos[i], pos[j], periodic_box).norm();
                    let h_ij = h[i].max(h[j]);
                    let w = kernel_cond(r, h_ij);
                    if w <= 0.0 {
                        continue;
                    }
                    let t_mean = 0.5 * (t_arr[i] + t_arr[j]);
                    let kappa_eff =
                        cfg.kappa_spitzer * cfg.psi_suppression * t_mean.powf(2.5) / COULOMB_LOG;
                    sum += kappa_eff * (t_arr[j] - t_arr[i]) * w * dt;
                }
                sum
            })
            .collect();

        for i in 0..n {
            if particles[i].ptype == ParticleType::Gas && delta_u[i] != 0.0 {
                let u_new = particles[i].internal_energy + delta_u[i];
                particles[i].internal_energy = u_new.max(u_floor);
            }
        }
    }

    #[cfg(not(feature = "rayon"))]
    {
        let mut delta_u = vec![0.0_f64; n];

        // Loop sobre pares únicos (i < j) para garantizar conservación de energía exacta.
        // Calor que gana i proviene de j y viceversa: Δu_i = −Δu_j.
        for i in 0..n {
            if particles[i].ptype != ParticleType::Gas {
                continue;
            }
            let h_i = particles[i].smoothing_length.max(1e-10);
            let t_i = u_to_temperature(particles[i].internal_energy.max(0.0), gamma);

            for j in (i + 1)..n {
                if particles[j].ptype != ParticleType::Gas {
                    continue;
                }
                let t_j = u_to_temperature(particles[j].internal_energy.max(0.0), gamma);

                let r = periodic_delta(particles[i].position, particles[j].position, periodic_box)
                    .norm();

                // Usa el máximo de los dos radios de suavizado
                let h_ij = h_i.max(particles[j].smoothing_length.max(1e-10));
                let w = kernel_cond(r, h_ij);
                if w <= 0.0 {
                    continue;
                }

                // Conductividad efectiva con dependencia T_mean^{5/2}
                let t_mean = 0.5 * (t_i + t_j);
                let kappa_eff =
                    cfg.kappa_spitzer * cfg.psi_suppression * t_mean.powf(2.5) / COULOMB_LOG;

                // Flujo neto: q > 0 significa que i gana calor de j (j > i en temperatura)
                let q_ij = kappa_eff * (t_j - t_i) * w * dt;
                delta_u[i] += q_ij; // i recibe
                delta_u[j] -= q_ij; // j cede (conservación exacta)
            }
        }

        // Aplicar incrementos: solo clampear si la conducción enfría por debajo del floor
        for i in 0..n {
            if particles[i].ptype == ParticleType::Gas && delta_u[i] != 0.0 {
                let u_new = particles[i].internal_energy + delta_u[i];
                particles[i].internal_energy = u_new.max(u_floor);
            }
        }
    }
}
