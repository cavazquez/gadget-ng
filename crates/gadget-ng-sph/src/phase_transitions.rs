//! Phase transitions y thermal instability para ISM multifase (Phase 171).
//!
//! ## Modelo de tres fases
//!
//! El ISM se divide en tres fases thermally:
//! - **Frío (cold)**: T < 10⁴ K — nubes moleculares, formación estelar
//! - **Tibio (warm)**: 10⁴ K ≤ T < 10⁵.5 K — gas ionizado T~10⁴ K, CIV,
//!   gas intermedio
//! - **Caliente (hot)**: T ≥ 10⁵.5 K — ICM, gas en shocks, Cooling flow
//!
//! ## Thermal instability (Field length)
//!
//! El criterio de Field (1965) para inestabilidad térmica:
//! ```text
//! λ_F = π^(1/2) * κ^(1/2) / (ρ * |dΛ/dT|)^(1/2)
//! ```
//! donde κ es la conductividad térmica. Si λ_F > resolution scale → gas termalmente inestable.
//!
//! En la práctica, usamos el criterio simplificado:
//! ```text
//! t_cool < t_ff  →  inestable (formación de nubes)
//! ```
//!
//! ## Transitions de fase
//!
//! - Cold → Warm: heating por CRs, conducción, o shocks
//! - Warm → Hot: cooling radiativo fuerte, especialmente en regiones de alta densidad
//! - Hot → Warm: cooling hasta T~10⁵ K
//!
//! ## Referencia
//!
//! Field (1965) ApJ 142, 531 — thermal instability.
//! McKee & Ostriker (1977) ApJ 218, 148 — three-phase ISM.

use gadget_ng_core::{Particle, ParticleType};
#[cfg(feature = "simd")]
use rayon::prelude::*;

const KB_OVER_MH_MU: f64 = 8.254e-3 / 0.6;

const T_COLD_MAX: f64 = 1.0e4;
const T_WARM_MAX: f64 = 3.2e5;

pub enum GasPhase {
    Cold,
    Warm,
    Hot,
}

pub fn temperature_from_u(u: f64, gamma: f64) -> f64 {
    KB_OVER_MH_MU * (gamma - 1.0) * u.max(0.0)
}

pub fn u_from_temperature(t: f64, gamma: f64) -> f64 {
    t / (KB_OVER_MH_MU * (gamma - 1.0))
}

pub fn classify_phase(u: f64, u_cold: f64, gamma: f64) -> GasPhase {
    let t = temperature_from_u(u + u_cold, gamma);
    if t < T_COLD_MAX {
        GasPhase::Cold
    } else if t < T_WARM_MAX {
        GasPhase::Warm
    } else {
        GasPhase::Hot
    }
}

pub fn phase_fractions(particles: &[Particle], gamma: f64) -> (f64, f64, f64) {
    #[cfg(feature = "simd")]
    {
        let (n_cold, n_warm, n_hot, n_gas) = particles
            .par_iter()
            .filter(|p| p.ptype == ParticleType::Gas)
            .fold(
                || (0u64, 0u64, 0u64, 0u64),
                |(c, w, h, n), p| match classify_phase(p.internal_energy, p.u_cold, gamma) {
                    GasPhase::Cold => (c + 1, w, h, n + 1),
                    GasPhase::Warm => (c, w + 1, h, n + 1),
                    GasPhase::Hot => (c, w, h + 1, n + 1),
                },
            )
            .reduce(
                || (0, 0, 0, 0),
                |(ac, aw, ah, an), (bc, bw, bh, bn)| (ac + bc, aw + bw, ah + bh, an + bn),
            );

        if n_gas == 0 {
            return (0.0, 0.0, 0.0);
        }

        (
            n_cold as f64 / n_gas as f64,
            n_warm as f64 / n_gas as f64,
            n_hot as f64 / n_gas as f64,
        )
    }

    #[cfg(not(feature = "simd"))]
    {
        let mut n_cold = 0u64;
        let mut n_warm = 0u64;
        let mut n_hot = 0u64;
        let mut n_gas = 0u64;

        for p in particles.iter() {
            if p.ptype != ParticleType::Gas {
                continue;
            }
            n_gas += 1;
            match classify_phase(p.internal_energy, p.u_cold, gamma) {
                GasPhase::Cold => n_cold += 1,
                GasPhase::Warm => n_warm += 1,
                GasPhase::Hot => n_hot += 1,
            }
        }

        if n_gas == 0 {
            return (0.0, 0.0, 0.0);
        }

        (
            n_cold as f64 / n_gas as f64,
            n_warm as f64 / n_gas as f64,
            n_hot as f64 / n_gas as f64,
        )
    }
}

pub fn field_length(kappa: f64, rho: f64, lambda_cooling: f64, gamma: f64) -> f64 {
    let dlambda_dt = if lambda_cooling > 0.0 {
        lambda_cooling / 1e5_f64.max(temperature_from_u(1.0, gamma))
    } else {
        1e-10
    };

    let numerator = std::f64::consts::PI * kappa.sqrt();
    let denominator = (rho * dlambda_dt.abs().max(1e-30)).sqrt();

    if denominator < 1e-30 {
        return f64::MAX;
    }

    numerator / denominator
}

pub fn cooling_time(t: f64, rho: f64, lambda_cool: f64) -> f64 {
    if lambda_cool <= 0.0 || rho <= 0.0 {
        return f64::MAX;
    }
    let n_h2 = 0.76 * rho;
    let cool_per_vol = lambda_cool * n_h2 * n_h2;
    if cool_per_vol < 1e-30 {
        return f64::MAX;
    }
    let gamma = 5.0 / 3.0;
    let u = t / (KB_OVER_MH_MU * (gamma - 1.0));
    u.abs() / cool_per_vol
}

pub fn free_fall_time(rho: f64) -> f64 {
    if rho <= 0.0 {
        return f64::MAX;
    }
    let g_code = 3.76e-4;
    (std::f64::consts::PI / 8.0) / (g_code * rho).sqrt()
}

pub fn thermal_instability_criterion(t: f64, rho: f64, lambda_cool: f64) -> bool {
    let t_cool = cooling_time(t, rho, lambda_cool);
    let t_ff = free_fall_time(rho);
    t_cool < t_ff
}

pub fn apply_phase_transitions(particles: &mut [Particle], dt: f64, gamma: f64, t_transition: f64) {
    #[cfg(feature = "simd")]
    {
        particles.par_iter_mut().for_each(|p| {
            apply_phase_transition_particle(p, dt, gamma, t_transition);
        });
    }

    #[cfg(not(feature = "simd"))]
    for p in particles.iter_mut() {
        apply_phase_transition_particle(p, dt, gamma, t_transition);
    }
}

fn apply_phase_transition_particle(p: &mut Particle, dt: f64, gamma: f64, t_transition: f64) {
    if p.ptype != ParticleType::Gas {
        return;
    }

    let t = temperature_from_u(p.internal_energy + p.u_cold, gamma);

    if t < T_COLD_MAX {
        let u_hot = u_from_temperature(T_WARM_MAX, gamma);
        let delta_u = (u_hot - p.u_cold).min(dt / t_transition);
        p.internal_energy += delta_u;
        p.u_cold -= delta_u;
    } else if t > T_WARM_MAX {
        let u_cold_target = u_from_temperature(T_COLD_MAX, gamma);
        let delta_u = (p.u_cold - u_cold_target).min(dt / t_transition);
        p.u_cold = u_cold_target.max(0.0);
        p.internal_energy -= delta_u;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn temperature_conversion_roundtrip() {
        let u = 1.0;
        let t = temperature_from_u(u, 5.0 / 3.0);
        let u_back = u_from_temperature(t, 5.0 / 3.0);
        assert!((u_back - u).abs() < 1e-10);
    }

    #[test]
    fn classify_cold_phase() {
        let u_cold = u_from_temperature(5e3, 5.0 / 3.0);
        let phase = classify_phase(0.0, u_cold, 5.0 / 3.0);
        assert!(matches!(phase, GasPhase::Cold));
    }

    #[test]
    fn classify_hot_phase() {
        let u_hot = u_from_temperature(1e6, 5.0 / 3.0);
        let phase = classify_phase(u_hot, 0.0, 5.0 / 3.0);
        assert!(matches!(phase, GasPhase::Hot));
    }

    #[test]
    fn phase_transitions_run_without_panic() {
        let mut p = Particle::new(
            0,
            1.0,
            gadget_ng_core::Vec3::zero(),
            gadget_ng_core::Vec3::zero(),
        );
        p.ptype = ParticleType::Gas;
        p.u_cold = u_from_temperature(5e3, 5.0 / 3.0);
        p.internal_energy = 0.1;

        apply_phase_transitions(&mut [p], 0.01, 5.0 / 3.0, 0.1);
    }
}
