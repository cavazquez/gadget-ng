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
#[cfg(feature = "rayon")]
use rayon::prelude::*;
#[cfg(all(
    not(feature = "rayon"),
    feature = "simd",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(all(
    not(feature = "rayon"),
    feature = "simd",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

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

    #[cfg(feature = "rayon")]
    {
        particles
            .par_iter_mut()
            .for_each(|p| update_dust_particle(p, cfg, gamma, dt));
    }

    #[cfg(not(feature = "rayon"))]
    {
        update_dust_serial(particles, cfg, gamma, dt);
    }
}

#[cfg(not(feature = "rayon"))]
fn update_dust_serial(particles: &mut [Particle], cfg: &DustSection, gamma: f64, dt: f64) {
    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    {
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx512f") {
            // SAFETY: AVX-512F availability was checked at runtime.
            unsafe {
                return update_dust_avx512(particles, cfg, gamma, dt);
            }
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: AVX2+FMA availability was checked at runtime.
            unsafe {
                return update_dust_avx2(particles, cfg, gamma, dt);
            }
        }
    }

    update_dust_scalar(particles, cfg, gamma, dt);
}

#[cfg(not(feature = "rayon"))]
fn update_dust_scalar(particles: &mut [Particle], cfg: &DustSection, gamma: f64, dt: f64) {
    for p in particles.iter_mut() {
        update_dust_particle(p, cfg, gamma, dt);
    }
}

fn update_dust_particle(p: &mut Particle, cfg: &DustSection, gamma: f64, dt: f64) {
    if p.ptype != ParticleType::Gas {
        return;
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
    #[cfg(feature = "rayon")]
    {
        particles
            .par_iter_mut()
            .for_each(|p| apply_dust_radiation_pressure_kick_particle(p, cfg, z_reference, dt));
    }

    #[cfg(not(feature = "rayon"))]
    {
        apply_dust_radiation_pressure_kick_serial(particles, cfg, z_reference, dt);
    }
}

#[cfg(not(feature = "rayon"))]
fn apply_dust_radiation_pressure_kick_serial(
    particles: &mut [Particle],
    cfg: &DustSection,
    z_reference: f64,
    dt: f64,
) {
    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    {
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx512f") {
            // SAFETY: AVX-512F availability was checked at runtime.
            unsafe {
                return apply_dust_radiation_pressure_kick_avx512(particles, cfg, z_reference, dt);
            }
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: AVX2+FMA availability was checked at runtime.
            unsafe {
                return apply_dust_radiation_pressure_kick_avx2(particles, cfg, z_reference, dt);
            }
        }
    }

    apply_dust_radiation_pressure_kick_scalar(particles, cfg, z_reference, dt);
}

#[cfg(not(feature = "rayon"))]
fn apply_dust_radiation_pressure_kick_scalar(
    particles: &mut [Particle],
    cfg: &DustSection,
    z_reference: f64,
    dt: f64,
) {
    for p in particles.iter_mut() {
        apply_dust_radiation_pressure_kick_particle(p, cfg, z_reference, dt);
    }
}

fn apply_dust_radiation_pressure_kick_particle(
    p: &mut Particle,
    cfg: &DustSection,
    z_reference: f64,
    dt: f64,
) {
    if p.ptype != ParticleType::Gas {
        return;
    }
    if p.dust_to_gas <= 0.0 {
        return;
    }
    const PI: f64 = std::f64::consts::PI;
    let h = p.smoothing_length.max(1e-30);
    let rho = p.mass / ((4.0 / 3.0) * PI * h * h * h).max(1e-100);
    let a_mag =
        cfg.radiation_pressure_kappa * p.dust_to_gas * cfg.radiation_pressure_j_uv / rho.max(1e-30);
    let dir = if p.position.z >= z_reference {
        Vec3::new(0.0, 0.0, 1.0)
    } else {
        Vec3::new(0.0, 0.0, -1.0)
    };
    p.velocity += dir * (a_mag * dt);
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

#[cfg(all(
    not(feature = "rayon"),
    feature = "simd",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn update_dust_avx2(particles: &mut [Particle], cfg: &DustSection, gamma: f64, dt: f64) {
    let lanes = 4;
    let chunks = particles.len() / lanes * lanes;
    let t_factor = (gamma - 1.0) / crate::cooling::KB_OVER_MH_MU_FOR_SIMD;
    let t_factor_v = _mm256_set1_pd(t_factor);
    let t_destroy_v = _mm256_set1_pd(cfg.t_destroy_k);
    let zero_v = _mm256_set1_pd(0.0);
    let one_v = _mm256_set1_pd(1.0);
    let dmax_v = _mm256_set1_pd(cfg.d_to_g_max);
    let grow_v = _mm256_set1_pd(dt / cfg.tau_grow.max(1e-10));
    let tau_grow_v = _mm256_set1_pd(cfg.tau_grow);
    let min_tau_ratio2_v = _mm256_set1_pd(1e-6);
    let neg_dt_v = _mm256_set1_pd(-dt);

    let mut i = 0;
    while i < chunks {
        let all_gas = particles[i..i + lanes]
            .iter()
            .all(|p| p.ptype == ParticleType::Gas);
        if !all_gas {
            update_dust_scalar(&mut particles[i..i + lanes], cfg, gamma, dt);
            i += lanes;
            continue;
        }
        let u = _mm256_set_pd(
            particles[i + 3].internal_energy.max(0.0),
            particles[i + 2].internal_energy.max(0.0),
            particles[i + 1].internal_energy.max(0.0),
            particles[i].internal_energy.max(0.0),
        );
        let t = _mm256_mul_pd(u, t_factor_v);
        let cold_mask = _mm256_cmp_pd(t, t_destroy_v, _CMP_LT_OQ);
        let z = _mm256_min_pd(
            one_v,
            _mm256_max_pd(
                zero_v,
                _mm256_set_pd(
                    particles[i + 3].metallicity,
                    particles[i + 2].metallicity,
                    particles[i + 1].metallicity,
                    particles[i].metallicity,
                ),
            ),
        );
        let d = _mm256_set_pd(
            particles[i + 3].dust_to_gas,
            particles[i + 2].dust_to_gas,
            particles[i + 1].dust_to_gas,
            particles[i].dust_to_gas,
        );
        let d_target = _mm256_mul_pd(dmax_v, z);
        let delta = _mm256_mul_pd(z, _mm256_mul_pd(_mm256_sub_pd(d_target, d), grow_v));
        let cold_d = _mm256_add_pd(d, delta);

        let t_hot = _mm256_max_pd(t, t_destroy_v);
        let t_ratio = _mm256_div_pd(t_destroy_v, t_hot);
        let tau_sputter = _mm256_mul_pd(
            tau_grow_v,
            _mm256_max_pd(_mm256_mul_pd(t_ratio, t_ratio), min_tau_ratio2_v),
        );
        let exp_arg = _mm256_div_pd(neg_dt_v, tau_sputter);
        let mut exp_args = [0.0; 4];
        // SAFETY: fixed-size stack array has exactly four f64 lanes.
        unsafe { _mm256_storeu_pd(exp_args.as_mut_ptr(), exp_arg) };
        let hot_factor = _mm256_set_pd(
            exp_args[3].exp(),
            exp_args[2].exp(),
            exp_args[1].exp(),
            exp_args[0].exp(),
        );
        let hot_d = _mm256_mul_pd(d, hot_factor);
        let new_d = _mm256_min_pd(
            dmax_v,
            _mm256_max_pd(zero_v, _mm256_blendv_pd(hot_d, cold_d, cold_mask)),
        );
        let mut out = [0.0; 4];
        // SAFETY: fixed-size stack array has exactly four f64 lanes.
        unsafe { _mm256_storeu_pd(out.as_mut_ptr(), new_d) };
        for lane in 0..lanes {
            particles[i + lane].dust_to_gas = out[lane];
        }
        i += lanes;
    }
    update_dust_scalar(&mut particles[chunks..], cfg, gamma, dt);
}

#[cfg(all(
    not(feature = "rayon"),
    feature = "simd",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
unsafe fn update_dust_avx512(particles: &mut [Particle], cfg: &DustSection, gamma: f64, dt: f64) {
    // SAFETY: the caller checked AVX-512F; AVX2 arithmetic preserves the same update semantics.
    unsafe { update_dust_avx2(particles, cfg, gamma, dt) }
}

#[cfg(all(
    not(feature = "rayon"),
    feature = "simd",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn apply_dust_radiation_pressure_kick_avx2(
    particles: &mut [Particle],
    cfg: &DustSection,
    z_reference: f64,
    dt: f64,
) {
    const FOUR_THIRDS_PI: f64 = 4.0 / 3.0 * std::f64::consts::PI;
    let lanes = 4;
    let chunks = particles.len() / lanes * lanes;
    let coeff_v = _mm256_set1_pd(cfg.radiation_pressure_kappa * cfg.radiation_pressure_j_uv * dt);
    let zref_v = _mm256_set1_pd(z_reference);
    let zero_v = _mm256_set1_pd(0.0);
    let min_h_v = _mm256_set1_pd(1e-30);
    let min_rho_v = _mm256_set1_pd(1e-30);
    let rho_norm_v = _mm256_set1_pd(FOUR_THIRDS_PI);

    let mut i = 0;
    while i < chunks {
        let all_gas = particles[i..i + lanes]
            .iter()
            .all(|p| p.ptype == ParticleType::Gas);
        if !all_gas {
            apply_dust_radiation_pressure_kick_scalar(
                &mut particles[i..i + lanes],
                cfg,
                z_reference,
                dt,
            );
            i += lanes;
            continue;
        }
        let d = _mm256_set_pd(
            particles[i + 3].dust_to_gas,
            particles[i + 2].dust_to_gas,
            particles[i + 1].dust_to_gas,
            particles[i].dust_to_gas,
        );
        let active = _mm256_cmp_pd(d, zero_v, _CMP_GT_OQ);
        let h = _mm256_max_pd(
            min_h_v,
            _mm256_set_pd(
                particles[i + 3].smoothing_length,
                particles[i + 2].smoothing_length,
                particles[i + 1].smoothing_length,
                particles[i].smoothing_length,
            ),
        );
        let m = _mm256_set_pd(
            particles[i + 3].mass,
            particles[i + 2].mass,
            particles[i + 1].mass,
            particles[i].mass,
        );
        let rho = _mm256_max_pd(
            min_rho_v,
            _mm256_div_pd(
                m,
                _mm256_mul_pd(rho_norm_v, _mm256_mul_pd(h, _mm256_mul_pd(h, h))),
            ),
        );
        let amag_dt = _mm256_div_pd(_mm256_mul_pd(coeff_v, d), rho);
        let z = _mm256_set_pd(
            particles[i + 3].position.z,
            particles[i + 2].position.z,
            particles[i + 1].position.z,
            particles[i].position.z,
        );
        let sign = _mm256_blendv_pd(
            _mm256_set1_pd(-1.0),
            _mm256_set1_pd(1.0),
            _mm256_cmp_pd(z, zref_v, _CMP_GE_OQ),
        );
        let dv = _mm256_and_pd(_mm256_mul_pd(amag_dt, sign), active);
        let vz = _mm256_set_pd(
            particles[i + 3].velocity.z,
            particles[i + 2].velocity.z,
            particles[i + 1].velocity.z,
            particles[i].velocity.z,
        );
        let new_vz = _mm256_add_pd(vz, dv);
        let mut out = [0.0; 4];
        // SAFETY: fixed-size stack array has exactly four f64 lanes.
        unsafe { _mm256_storeu_pd(out.as_mut_ptr(), new_vz) };
        for lane in 0..lanes {
            particles[i + lane].velocity.z = out[lane];
        }
        i += lanes;
    }
    apply_dust_radiation_pressure_kick_scalar(&mut particles[chunks..], cfg, z_reference, dt);
}

#[cfg(all(
    not(feature = "rayon"),
    feature = "simd",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
unsafe fn apply_dust_radiation_pressure_kick_avx512(
    particles: &mut [Particle],
    cfg: &DustSection,
    z_reference: f64,
    dt: f64,
) {
    // SAFETY: the caller checked AVX-512F; AVX2 arithmetic preserves the same update semantics.
    unsafe { apply_dust_radiation_pressure_kick_avx2(particles, cfg, z_reference, dt) }
}
