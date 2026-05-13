//! Plasma de dos fluidos: temperatura electrónica ≠ temperatura iónica (Phase 149).
//!
//! ## Modelo
//!
//! En plasmas de alta temperatura (ICM de cúmulos, shocks fuertes), el tiempo de
//! termalización entre electrones e iones puede ser mayor que el tiempo dinámico.
//! Los electrones y los iones tienen temperaturas distintas:
//!
//! - `T_i` = temperatura iónica, derivada de `internal_energy` (como siempre)
//! - `T_e` = temperatura electrónica, almacenada en `Particle.t_electron`
//!
//! ### Acoplamiento Coulomb
//!
//! La transferencia de calor entre electrones e iones se da por colisiones Coulomb:
//!
//! ```text
//! dT_e/dt = −ν_ei(T_e − T_i)
//! ```
//!
//! donde la frecuencia de acoplamiento es:
//!
//! ```text
//! ν_ei = ν_coeff × n_e / T_e^{3/2}
//! ```
//!
//! Con `ν_coeff` en unidades internas. En el límite `ν_ei → ∞` se recupera T_e = T_i.
//!
//! ### Calentamiento electrónico por shocks
//!
//! En un shock de velocidad `v_sh`, los electrones reciben una fracción `β_e ~ m_e/m_p`
//! del calentamiento cinético (muy poco). La mayor parte va a los iones. Esto produce
//! `T_e << T_i` justo detrás del shock (relevante para observaciones de X-ray).
//!
//! ## Referencias
//!
//! Spitzer (1962) — frecuencia de colisión Coulomb en plasma.
//! Fox & Loeb (1997), ApJ 491, 460 — dos fluidos en ICM de cúmulos.
//! Rudd & Nagai (2009), ApJL 701, L16 — T_e/T_i en simulaciones de cúmulos.

use gadget_ng_core::{Particle, ParticleType, TwoFluidSection};
#[cfg(feature = "rayon")]
use rayon::prelude::*;

/// Convierte energía interna `u` a temperatura proporcional (en unidades arbitrarias).
///
/// `T ∝ (γ−1) × u / k_B`. Usamos γ=5/3 implícito y k_B=1 en unidades del código.
#[inline]
fn u_to_t_code(u: f64, gamma: f64) -> f64 {
    (gamma - 1.0) * u.max(0.0)
}

/// Aplica el acoplamiento Coulomb electrón-ión a partículas de gas (Phase 149).
///
/// Para cada partícula de gas:
/// 1. Calcula `T_i` desde `internal_energy` con `γ = 5/3`
/// 2. Si `t_electron = 0`, inicializa `t_electron = T_i`
/// 3. Calcula `ν_ei = nu_ei_coeff × ρ / T_e^{3/2}` (ρ ≈ m/h³)
/// 4. Actualiza `T_e` con el paso implícito:
///    `T_e_new = T_e + (T_i − T_e) × (1 − exp(−ν_ei × dt))`
#[cfg(not(feature = "rayon"))]
fn apply_electron_ion_coupling_impl(particles: &mut [Particle], cfg: &TwoFluidSection, dt: f64) {
    const GAMMA: f64 = 5.0 / 3.0;

    for p in particles.iter_mut() {
        if p.ptype != ParticleType::Gas {
            continue;
        }

        let t_i = u_to_t_code(p.internal_energy, GAMMA);

        if p.t_electron <= 0.0 {
            p.t_electron = t_i;
            continue;
        }

        let t_e = p.t_electron;
        let h = p.smoothing_length.max(1e-10);
        let rho = (p.mass / (h * h * h)).max(1e-30);

        let t_e_eff = t_e.abs().max(1e-30);
        let nu_ei = cfg.nu_ei_coeff * rho / (t_e_eff * t_e_eff.sqrt());

        let factor = 1.0 - (-nu_ei * dt).exp();
        p.t_electron = t_e + (t_i - t_e) * factor;

        p.t_electron = p.t_electron.max(0.0);
    }
}

#[cfg(feature = "rayon")]
fn apply_electron_ion_coupling_par(particles: &mut [Particle], cfg: &TwoFluidSection, dt: f64) {
    const GAMMA: f64 = 5.0 / 3.0;

    particles.par_iter_mut().for_each(|p| {
        if p.ptype != ParticleType::Gas {
            return;
        }

        let t_i = u_to_t_code(p.internal_energy, GAMMA);

        if p.t_electron <= 0.0 {
            p.t_electron = t_i;
            return;
        }

        let t_e = p.t_electron;
        let h = p.smoothing_length.max(1e-10);
        let rho = (p.mass / (h * h * h)).max(1e-30);

        let t_e_eff = t_e.abs().max(1e-30);
        let nu_ei = cfg.nu_ei_coeff * rho / (t_e_eff * t_e_eff.sqrt());

        let factor = 1.0 - (-nu_ei * dt).exp();
        p.t_electron = t_e + (t_i - t_e) * factor;

        p.t_electron = p.t_electron.max(0.0);
    });
}

pub fn apply_electron_ion_coupling(particles: &mut [Particle], cfg: &TwoFluidSection, dt: f64) {
    #[cfg(feature = "rayon")]
    {
        apply_electron_ion_coupling_par(particles, cfg, dt);
    }

    #[cfg(not(feature = "rayon"))]
    {
        apply_electron_ion_coupling_impl(particles, cfg, dt);
    }
}

/// Relación T_e/T_i promediada sobre todas las partículas de gas.
///
/// Útil para monitoreo: en ICM sin shocks → T_e/T_i ≈ 1.
/// Detrás de shocks fuertes → T_e/T_i << 1.
#[cfg(not(feature = "rayon"))]
fn mean_te_over_ti_impl(particles: &[Particle]) -> f64 {
    const GAMMA: f64 = 5.0 / 3.0;
    let mut sum = 0.0_f64;
    let mut n = 0usize;
    for p in particles.iter() {
        if p.ptype != ParticleType::Gas {
            continue;
        }
        let t_i = u_to_t_code(p.internal_energy, GAMMA).max(1e-30);
        sum += p.t_electron / t_i;
        n += 1;
    }
    if n == 0 { 1.0 } else { sum / n as f64 }
}

#[cfg(feature = "rayon")]
fn mean_te_over_ti_par(particles: &[Particle]) -> f64 {
    const GAMMA: f64 = 5.0 / 3.0;
    let (sum, n) = particles
        .par_iter()
        .filter(|p| p.ptype == ParticleType::Gas)
        .fold(
            || (0.0_f64, 0usize),
            |(s, c), p| {
                let t_i = u_to_t_code(p.internal_energy, GAMMA).max(1e-30);
                (s + p.t_electron / t_i, c + 1)
            },
        )
        .reduce(|| (0.0, 0), |a, b| (a.0 + b.0, a.1 + b.1));
    if n == 0 { 1.0 } else { sum / n as f64 }
}

pub fn mean_te_over_ti(particles: &[Particle]) -> f64 {
    #[cfg(feature = "rayon")]
    {
        mean_te_over_ti_par(particles)
    }

    #[cfg(not(feature = "rayon"))]
    {
        mean_te_over_ti_impl(particles)
    }
}
