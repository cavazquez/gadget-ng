//! Modelo ISM multifase fría-caliente (Phase 114).
//!
//! ## Modelo
//!
//! Basado en Springel & Hernquist (2003) MNRAS 339, 289.
//!
//! El ISM se trata como una mezcla de dos fases:
//! - **Fase caliente**: gas difuso a alta temperatura, trackeado por `internal_energy` (`u`).
//! - **Fase fría**: nubes moleculares densas, trackeadas por `u_cold`.
//!
//! La **presión efectiva** combina ambas fases:
//!
//! ```text
//! P_eff = (γ - 1) × ρ × (u + q* × u_cold)
//! ```
//!
//! donde `q*` es el parámetro de escala que controla la rigidez del ISM.
//!
//! ## Dinámica de fases
//!
//! La transferencia de energía entre fases sigue:
//! - Gas denso (sobre umbral de SFR) gana componente fría: `u_cold` crece a expensas de `u`.
//! - La fracción fría en equilibrio es `f_cold × min(rho / rho_sf, 1)`.
//! - Tiempo de equilibración: `t_eq = dt × 0.1` (relajación rápida hacia el equilibrio).
//!
//! ## Referencia
//!
//! Springel & Hernquist (2003) MNRAS 339, 289

use gadget_ng_core::{IsmSection, Particle, ParticleType};
#[cfg(feature = "rayon")]
use rayon::prelude::*;

/// Calcula la presión efectiva del ISM multifase (Phase 114).
///
/// `P_eff = (γ - 1) × ρ × (u + q_star × u_cold)`
///
/// # Parámetros
///
/// - `rho`: densidad local de la partícula.
/// - `u`: energía interna específica de la fase caliente.
/// - `u_cold`: energía interna específica de la fase fría.
/// - `q_star`: parámetro de rigidez del ISM (típicamente 2.5).
/// - `gamma`: índice adiabático.
pub fn effective_pressure(rho: f64, u: f64, u_cold: f64, q_star: f64, gamma: f64) -> f64 {
    (gamma - 1.0) * rho * (u + q_star * u_cold)
}

/// Actualiza las fases frías y calientes del ISM para gas denso (Phase 114).
///
/// Para cada partícula de gas con densidad sobre el umbral de formación estelar:
/// 1. Calcula la energía fría objetivo: `u_cold_eq = f_cold × u_total × clamp(ρ/ρ_sf, 0, 1)`.
/// 2. Relaja `u_cold` hacia ese valor: `u_cold += (u_cold_eq - u_cold) × dt / t_relax`.
/// 3. Conserva energía total: `u += (u_cold_old - u_cold_new)` (la fase caliente cede a la fría).
///
/// Fuera del umbral de densidad, `u_cold` se disipa exponencialmente: `u_cold *= exp(-dt / t_relax)`.
///
/// # Parámetros
///
/// - `particles`: slice mutable de partículas.
/// - `sfr`: tasa de formación estelar por partícula [mismas unidades que densidad/tiempo].
/// - `rho_sf`: densidad umbral de formación estelar.
/// - `cfg`: configuración del módulo ISM.
/// - `dt`: paso de tiempo.
pub fn update_ism_phases(
    particles: &mut [Particle],
    sfr: &[f64],
    rho_sf: f64,
    cfg: &IsmSection,
    dt: f64,
) {
    if !cfg.enabled {
        return;
    }

    let n = particles.len();
    assert_eq!(sfr.len(), n, "sfr.len() debe ser igual a particles.len()");

    #[cfg(feature = "rayon")]
    {
        particles
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, p)| update_ism_particle(p, i, sfr, rho_sf, cfg, dt));
    }

    #[cfg(not(feature = "rayon"))]
    {
        for (i, p) in particles.iter_mut().enumerate() {
            update_ism_particle(p, i, sfr, rho_sf, cfg, dt);
        }
    }
}

fn update_ism_particle(
    p: &mut Particle,
    i: usize,
    sfr: &[f64],
    rho_sf: f64,
    cfg: &IsmSection,
    dt: f64,
) {
    const T_RELAX_FACTOR: f64 = 0.1;

    if p.ptype != ParticleType::Gas {
        return;
    }

    let h = p.smoothing_length.max(1e-10);
    let rho_local = p.mass / (4.0 / 3.0 * std::f64::consts::PI * h * h * h);

    let u_total = p.internal_energy + p.u_cold;

    if sfr[i] > 0.0 && rho_sf > 0.0 {
        let density_factor = (rho_local / rho_sf).min(1.0);
        let u_cold_eq = cfg.f_cold * u_total * density_factor;

        let u_cold_old = p.u_cold;
        let t_relax = (T_RELAX_FACTOR * dt).max(dt * 0.01);
        let alpha = (dt / t_relax).min(1.0);

        let u_cold_new = u_cold_old + alpha * (u_cold_eq - u_cold_old);
        let du_cold = u_cold_new - u_cold_old;

        p.u_cold = u_cold_new.max(0.0);
        p.internal_energy = (p.internal_energy - du_cold).max(0.0);
    } else {
        let decay = (-dt / (T_RELAX_FACTOR * 10.0 * dt.max(1e-20))).exp();
        let u_cold_new = p.u_cold * decay;
        let released = p.u_cold - u_cold_new;
        p.u_cold = u_cold_new;
        p.internal_energy += released;
    }
}

/// Aplica la presión efectiva ISM en el cálculo de fuerzas SPH (Phase 114).
///
/// Modifica el campo `internal_energy` temporalmente para que `compute_sph_forces`
/// use la presión efectiva en lugar de la termal pura. La energía real se restaura
/// después del paso de fuerzas.
///
/// En la práctica, para el integrador simplificado, calculamos directamente
/// la presión efectiva sin modificar el estado interno.
#[inline]
pub fn effective_u(p: &Particle, q_star: f64) -> f64 {
    p.internal_energy + q_star * p.u_cold
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use gadget_ng_core::{IsmSection, Vec3};

    #[test]
    fn effective_pressure_includes_cold_phase() {
        let p = effective_pressure(2.0, 1.0, 0.5, 2.5, 5.0 / 3.0);
        let p_hot_only = effective_pressure(2.0, 1.0, 0.0, 2.5, 5.0 / 3.0);
        assert!(p > p_hot_only);
        assert_abs_diff_eq!(p / p_hot_only, 2.25, epsilon = 1e-12);
    }

    #[test]
    fn update_ism_phases_transfers_to_cold_component() {
        let mut p = gadget_ng_core::Particle::new_gas(0, 1.0, Vec3::zero(), Vec3::zero(), 1.0, 0.1);
        p.u_cold = 0.0;
        let mut particles = vec![p];
        let sfr = vec![1.0];
        let cfg = IsmSection {
            enabled: true,
            q_star: 2.5,
            f_cold: 0.5,
        };
        update_ism_phases(&mut particles, &sfr, 1.0, &cfg, 0.1);
        assert!(particles[0].u_cold > 0.0);
        let u_total = particles[0].internal_energy + particles[0].u_cold;
        assert_abs_diff_eq!(u_total, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn effective_u_matches_pressure_formula() {
        let mut p = gadget_ng_core::Particle::new_gas(0, 1.0, Vec3::zero(), Vec3::zero(), 2.0, 0.1);
        p.u_cold = 1.0;
        assert_abs_diff_eq!(effective_u(&p, 2.5), 4.5, epsilon = 1e-12);
    }
}
