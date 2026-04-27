//! Polvo intersticial: acreción D/G y destrucción por sputtering (Phase 130).
//!
//! ## Modelo
//!
//! La relación polvo/gas D/G evoluciona por dos procesos competitivos:
//!
//! ### Acreción (T < T_destroy)
//!
//! El polvo crece por acumulación de metales del gas:
//! ```text
//! dD/G/dt = Z × (d_to_g_max − D/G) / τ_grow
//! ```
//!
//! La tasa es proporcional a la metalicidad Z (más metales → más polvo posible)
//! y se satura en `d_to_g_max` (límite solar).
//!
//! ### Sputtering térmico (T > T_destroy)
//!
//! A temperaturas > 10⁶ K el plasma destruye el polvo por bombardeo iónico:
//! ```text
//! D/G → D/G × exp(−dt / τ_sputter)
//! ```
//!
//! donde `τ_sputter = τ_grow × (T/T_destroy)^{-2}` (tasa proporcional a T²).
//!
//! ## Referencia
//!
//! McKee (1989), Astrophys. J. 345, 782 — sputtering térmico.
//! Dwek (1998), Astrophys. J. 501, 643 — ciclo de vida del polvo.
//! Zhukovska et al. (2014), A&A 562, A76 — modelo D/G en SPH.

use crate::cooling::u_to_temperature;
use gadget_ng_core::{DustSection, Particle, ParticleType};

/// Actualiza la relación polvo/gas D/G de cada partícula de gas (Phase 130).
///
/// # Proceso
///
/// 1. Convierte `u` → temperatura T usando γ.
/// 2. Si `T < t_destroy_k`: acreción → `D/G += Z × (d_to_g_max − D/G) × dt / τ_grow`.
/// 3. Si `T ≥ t_destroy_k`: sputtering → `D/G × exp(−dt / τ_sputter)`.
/// 4. Clampea D/G ∈ [0, d_to_g_max].
///
/// Las partículas de DM y estrellas no se modifican.
pub fn update_dust(particles: &mut [Particle], cfg: &DustSection, gamma: f64, dt: f64) {
    if !cfg.enabled {
        return;
    }

    for p in particles.iter_mut() {
        if p.ptype != ParticleType::Gas {
            continue;
        }

        let t = u_to_temperature(p.internal_energy.max(0.0), gamma);
        let z = p.metallicity.clamp(0.0, 1.0);

        if t < cfg.t_destroy_k {
            // Acreción: D/G crece hacia d_to_g_max × Z con tiempo τ_grow
            let d_target = cfg.d_to_g_max * z;
            let tau = cfg.tau_grow.max(1e-10);
            p.dust_to_gas += z * (d_target - p.dust_to_gas) * dt / tau;
        } else {
            // Sputtering térmico: τ_sputter ∝ (T_destroy/T)²
            let t_ratio = cfg.t_destroy_k / t.max(cfg.t_destroy_k);
            let tau_sputter = cfg.tau_grow * (t_ratio * t_ratio).max(1e-6);
            p.dust_to_gas *= (-dt / tau_sputter).exp();
        }

        p.dust_to_gas = p.dust_to_gas.clamp(0.0, cfg.d_to_g_max);
    }
}

/// Profundidad óptica del polvo en UV (Phase 137).
///
/// Aproximación local: `τ_dust = κ_dust × (D/G) × ρ × h`
///
/// - `κ_dust`: opacidad del polvo en UV [cm²/g] (típico ISM: 1000 cm²/g)
/// - `dust_to_gas`: relación D/G de la partícula
/// - `rho`: densidad local [g/cm³ en unidades del código]
/// - `h`: longitud de suavizado [cm en unidades del código]
///
/// El flujo UV atenuado: `J_UV_eff = J_UV × exp(−τ_dust)`.
pub fn dust_uv_opacity(kappa_dust_uv: f64, dust_to_gas: f64, rho: f64, h: f64) -> f64 {
    kappa_dust_uv * dust_to_gas * rho * h
}
