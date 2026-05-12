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
use gadget_ng_core::{DustSection, DustSpeciesModel, Particle, ParticleType, Vec3};

/// Radiation constant in cgs, `a_rad` [erg cm^-3 K^-4].
const A_RAD_CGS: f64 = 7.5657e-15;
/// Stefan-Boltzmann constant in cgs [erg cm^-2 s^-1 K^-4].
const SIGMA_SB_CGS: f64 = 5.670_374_419e-5;

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

/// Fracciones normalizadas `(silicato, grafito)` para una mezcla activa.
pub fn dust_species_fractions(cfg: &DustSection) -> (f64, f64) {
    if cfg.species_model == DustSpeciesModel::Single {
        return (1.0, 0.0);
    }
    let sil = cfg.silicate_fraction.max(0.0);
    let gra = cfg.graphite_fraction.max(0.0);
    let sum = sil + gra;
    if sum <= 0.0 {
        (1.0, 0.0)
    } else {
        (sil / sum, gra / sum)
    }
}

/// Opacidad UV efectiva para el modelo de especies de polvo activo.
pub fn effective_dust_uv_opacity(cfg: &DustSection) -> f64 {
    match cfg.species_model {
        DustSpeciesModel::Single => cfg.kappa_dust_uv.max(0.0),
        DustSpeciesModel::SilicateGraphite => {
            let (sil, gra) = dust_species_fractions(cfg);
            sil * cfg.kappa_silicate_uv.max(0.0) + gra * cfg.kappa_graphite_uv.max(0.0)
        }
    }
}

/// Profundidad óptica UV local usando la mezcla activa de polvo.
pub fn dust_uv_opacity_active(cfg: &DustSection, dust_to_gas: f64, rho: f64, h: f64) -> f64 {
    dust_uv_opacity(effective_dust_uv_opacity(cfg), dust_to_gas, rho, h)
}

/// Factor de shielding para H2 producido por polvo activo.
///
/// Retorna `1 + boost × (1 − exp(−tau_dust))`, acotado y suave. El resultado
/// multiplica la fracción de equilibrio H2 y alarga el tiempo efectivo de
/// fotodisociación en [`update_h2_fraction_with_dust`].
pub fn dust_h2_shielding_factor(p: &Particle, cfg: &DustSection) -> f64 {
    if !cfg.enabled || p.ptype != ParticleType::Gas || p.dust_to_gas <= 0.0 {
        return 1.0;
    }
    let h = p.smoothing_length.max(1e-30);
    let rho = p.mass / (h * h * h).max(1e-100);
    let tau = dust_uv_opacity_active(cfg, p.dust_to_gas, rho, h).max(0.0);
    1.0 + cfg.h2_shielding_boost.max(0.0) * (1.0 - (-tau).exp())
}

/// Impulso de “presión de radiación” en el gas acoplado al polvo (modelo mínimo Fase A).
///
/// Ajusta `velocity` (misma variable que el integrador SPH cosmológico) con
/// `Δv ≈ (κ × (D/G) × J / ρ) · û`, con û = signo(z − z_ref)·ẑ para un escape
/// vertical simplificado. Requiere `DustSection::radiation_pressure_enabled` y
/// `dust_to_gas` &gt; 0.
pub fn apply_dust_radiation_pressure_kick(
    particles: &mut [Particle],
    cfg: &DustSection,
    z_reference: f64,
    dt: f64,
) {
    if !cfg.enabled || !cfg.radiation_pressure_enabled {
        return;
    }
    const PI: f64 = std::f64::consts::PI;
    for p in particles.iter_mut() {
        if p.ptype != ParticleType::Gas {
            continue;
        }
        if p.dust_to_gas <= 0.0 {
            continue;
        }
        let h = p.smoothing_length.max(1e-30);
        let rho = p.mass / ((4.0 / 3.0) * PI * h * h * h).max(1e-100);
        let a_mag = cfg.radiation_pressure_kappa * p.dust_to_gas * cfg.radiation_pressure_j_uv
            / rho.max(1e-30);
        let dir = if p.position.z >= z_reference {
            Vec3::new(0.0, 0.0, 1.0)
        } else {
            Vec3::new(0.0, 0.0, -1.0)
        };
        p.velocity += dir * (a_mag * dt);
    }
}

/// Equilibrium dust temperature for a greybody with emissivity index `beta=2`.
///
/// The model solves a reduced balance `E_rad κ_uv ~ a_rad T^(4+beta)` and clamps
/// the result to the configured floor/cap. It is intentionally local and smooth:
/// suitable for production diagnostics and for feeding the reduced IR RT group.
pub fn dust_equilibrium_temperature(radiation_energy_density: f64, cfg: &DustSection) -> f64 {
    let floor = cfg.dust_temperature_floor_k.max(0.0);
    let cap = cfg.dust_temperature_cap_k.max(floor + 1e-12);
    if !cfg.enabled || !cfg.ir_emission_enabled {
        return floor;
    }

    let e_rad = radiation_energy_density.max(0.0);
    let opacity_ratio = (effective_dust_uv_opacity(cfg) / cfg.kappa_dust_ir.max(1e-30)).max(0.0);
    let emissivity = cfg.ir_emissivity.max(1e-30);
    let t_rad = (e_rad * opacity_ratio / (A_RAD_CGS * emissivity)).powf(1.0 / 6.0);
    floor.max(t_rad).min(cap)
}

/// Greybody IR luminosity proxy for one gas particle with dust.
///
/// Returns energy per unit time in internal RT units. The `T^6` dependence is the
/// same modified-blackbody scaling used by [`dust_equilibrium_temperature`].
pub fn dust_ir_luminosity(p: &Particle, dust_temperature_k: f64, cfg: &DustSection) -> f64 {
    if !cfg.enabled || !cfg.ir_emission_enabled || p.ptype != ParticleType::Gas {
        return 0.0;
    }
    if p.dust_to_gas <= 0.0 || p.mass <= 0.0 {
        return 0.0;
    }

    let t = dust_temperature_k.clamp(
        cfg.dust_temperature_floor_k.max(0.0),
        cfg.dust_temperature_cap_k.max(0.0),
    );
    let modified_blackbody = SIGMA_SB_CGS * t.powi(6) / 100.0_f64.powi(2);
    cfg.ir_emissivity.max(0.0)
        * cfg.kappa_dust_ir.max(0.0)
        * p.dust_to_gas.max(0.0)
        * p.mass.max(0.0)
        * modified_blackbody
}
