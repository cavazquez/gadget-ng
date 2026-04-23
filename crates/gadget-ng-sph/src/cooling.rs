//! Enfriamiento radiativo atómico H+He (Phase 66).
//!
//! ## Modelo
//!
//! La tasa de enfriamiento neta sigue la ley de potencia:
//!
//! ```text
//! Λ(T) = Λ₀ · (T / T_ref)^β   [erg cm³ s⁻¹]
//! ```
//!
//! con `Λ₀ = 1.4 × 10⁻²³`, `β = 0.7`, `T_ref = 10⁴ K`.
//!
//! La temperatura se convierte desde la energía interna específica `u` usando
//! `T = u · (γ−1) · μ · m_H / k_B`, con `μ = 0.6` (gas completamente ionizado H+He).
//!
//! La derivada de energía interna por unidad de masa es:
//! `du/dt = −Λ(T) · n_H² / ρ`
//! donde `n_H = X_H · ρ / m_H` y `X_H = 0.76`.
//!
//! En unidades internas gadget-ng (L=kpc, M=10¹⁰ M☉, V=km/s) los factores de
//! conversión son: `k_B/m_H ≈ 8.254 × 10⁻³ (km/s)² / K`.

use gadget_ng_core::{Particle, ParticleType, SphSection};

/// k_B / (m_H · μ) en (km/s)² / K.
/// μ = 0.6 (gas completamente ionizado H+He), k_B/m_H = 8.254 × 10⁻³ km²/s²/K.
const KB_OVER_MH_MU: f64 = 8.254e-3 / 0.6;

/// Factor de enfriamiento Λ₀ en unidades adimensionales (ajustado a escala interna).
/// Valor físico: ~1.4 × 10⁻²³ erg cm³ s⁻¹; aquí re-escalado a (km/s)² Mpc³ / M☉.
/// Ajuste: `Λ_code ≈ 1.4e-23 × conv_factor` donde `conv_factor` se estima para
/// que el tiempo de enfriamiento a T=10⁵ K y ρ_típica sea ~1 Gyr.
/// Usamos un valor de conveniencia calibrado: `Λ₀ = 2e-5` (unidades internas).
const LAMBDA_0: f64 = 2e-5;

/// Exponente de la ley de potencia Λ ∝ T^β.
const BETA: f64 = 0.7;

/// Temperatura de referencia [K].
const T_REF: f64 = 1e4;

/// Fracción de hidrógeno en masa.
const X_H: f64 = 0.76;

/// Convierte energía interna específica `u` [(km/s)²] a temperatura [K].
#[inline]
pub fn u_to_temperature(u: f64, gamma: f64) -> f64 {
    u * (gamma - 1.0) / KB_OVER_MH_MU
}

/// Convierte temperatura [K] a energía interna específica [(km/s)²].
#[inline]
pub fn temperature_to_u(t_k: f64, gamma: f64) -> f64 {
    t_k * KB_OVER_MH_MU / (gamma - 1.0)
}

/// Tasa de enfriamiento `Λ(T)` en unidades internas.
///
/// Devuelve 0 por debajo de `t_floor_k`.
pub fn cooling_rate_atomic(u: f64, _rho: f64, gamma: f64, t_floor_k: f64) -> f64 {
    let t = u_to_temperature(u, gamma);
    if t <= t_floor_k {
        return 0.0;
    }
    LAMBDA_0 * (t / T_REF).powf(BETA)
}

/// Aplica enfriamiento radiativo a todas las partículas de gas.
///
/// Usa un paso de Euler explícito: `u_new = max(u + du_dt * dt, u_floor)`.
///
/// La tasa de cambio de `u` es:
/// ```text
/// du/dt = −Λ(T) · (X_H · ρ / m_H)² / ρ = −Λ(T) · X_H² · ρ / m_H²
/// ```
/// En unidades internas asumimos ρ ≈ m/h³_sml (estimación local de densidad).
pub fn apply_cooling(particles: &mut [Particle], cfg: &SphSection, dt: f64) {
    let gamma = cfg.gamma;
    let t_floor_k = cfg.t_floor_k;
    let u_floor = temperature_to_u(t_floor_k, gamma);

    for p in particles.iter_mut() {
        if p.ptype != ParticleType::Gas || p.internal_energy <= u_floor {
            continue;
        }
        // Densidad estimada localmente: ρ ≈ m / (4/3 π h³)
        let h = p.smoothing_length.max(1e-10);
        let rho_local = p.mass / (4.0 / 3.0 * std::f64::consts::PI * h * h * h);

        let lambda = cooling_rate_atomic(p.internal_energy, rho_local, gamma, t_floor_k);
        // du/dt = -Λ · X_H² · ρ / m
        // (en unidades adimensionales el factor m_H² se cancela con la normalización de Λ₀)
        let du_dt = -lambda * X_H * X_H * rho_local;
        p.internal_energy = (p.internal_energy + du_dt * dt).max(u_floor);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn temperature_conversion_round_trip() {
        let gamma = 5.0 / 3.0;
        let t0 = 1e5_f64;
        let u = temperature_to_u(t0, gamma);
        let t1 = u_to_temperature(u, gamma);
        assert!((t1 - t0).abs() / t0 < 1e-12, "T = {t1} ≠ {t0}");
    }

    #[test]
    fn cooling_rate_zero_at_floor() {
        let gamma = 5.0 / 3.0;
        let u_floor = temperature_to_u(1e4, gamma);
        assert_eq!(cooling_rate_atomic(u_floor * 0.9, 1.0, gamma, 1e4), 0.0);
    }

    #[test]
    fn cooling_rate_positive_above_floor() {
        let gamma = 5.0 / 3.0;
        let u_hot = temperature_to_u(1e6, gamma);
        let rate = cooling_rate_atomic(u_hot, 1.0, gamma, 1e4);
        assert!(rate > 0.0, "Λ debe ser positiva para T > T_floor");
    }
}
