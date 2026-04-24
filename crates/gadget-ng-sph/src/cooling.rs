//! Enfriamiento radiativo atómico H+He, metales analítico y tabulado (Phases 66, 111, 119).
//!
//! ## Modelo
//!
//! ### H+He (AtomicHHe)
//!
//! La tasa de enfriamiento neta sigue la ley de potencia:
//!
//! ```text
//! Λ(T) = Λ₀ · (T / T_ref)^β   [erg cm³ s⁻¹]
//! ```
//!
//! con `Λ₀ = 1.4 × 10⁻²³`, `β = 0.7`, `T_ref = 10⁴ K`.
//!
//! ### Metales (MetalCooling, Phase 111)
//!
//! Fitting analítico Sutherland & Dopita (1993) por tramos:
//!
//! ```text
//! Λ(T, Z) = Λ_HHe(T) + (Z / Z_sun) × Λ_metal(T)
//! Λ_metal(T):
//!   T < 10⁴ K            → 0
//!   10⁴ ≤ T < 10⁷ K      → Λ_m0 × (T / 10⁵)^0.7
//!   T ≥ 10⁷ K             → Λ_m1 × (T / 10⁷)^0.5   (bremsstrahlung)
//! ```
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

use gadget_ng_core::{CoolingKind, Particle, ParticleType, SphSection};

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

/// Tasa de enfriamiento `Λ(T)` en unidades internas (solo H+He).
///
/// Devuelve 0 por debajo de `t_floor_k`.
pub fn cooling_rate_atomic(u: f64, _rho: f64, gamma: f64, t_floor_k: f64) -> f64 {
    let t = u_to_temperature(u, gamma);
    if t <= t_floor_k {
        return 0.0;
    }
    LAMBDA_0 * (t / T_REF).powf(BETA)
}

/// Tasa de enfriamiento incluyendo contribución metálica (Phase 111).
///
/// `Λ(T, Z) = Λ_HHe(T) + (Z / Z_sun) × Λ_metal(T)`
///
/// Fitting analítico de Sutherland & Dopita (1993) para `Λ_metal(T)`:
/// - T < 10⁴ K: 0
/// - 10⁴ ≤ T < 10⁷ K: Λ_m0 × (T / 10⁵)^0.7
/// - T ≥ 10⁷ K: Λ_m1 × (T / 10⁷)^0.5 (bremsstrahlung dominante)
///
/// Z_sun = 0.0127 (fracción en masa solar de metales).
pub fn cooling_rate_metal(u: f64, rho: f64, metallicity: f64, gamma: f64, t_floor_k: f64) -> f64 {
    let lambda_hhe = cooling_rate_atomic(u, rho, gamma, t_floor_k);
    let t = u_to_temperature(u, gamma);
    if t <= t_floor_k { return 0.0; }

    // Normalización de metalicidad solar
    const Z_SUN: f64 = 0.0127;
    let z_ratio = (metallicity / Z_SUN).max(0.0);

    // Coeficientes de Sutherland & Dopita (1993) en unidades internas
    const LAMBDA_M0: f64 = 3.0e-5; // régimen 10⁴–10⁷ K
    const LAMBDA_M1: f64 = 1.0e-5; // régimen bremsstrahlung T>10⁷ K

    let lambda_metal = if t < 1e7 {
        LAMBDA_M0 * (t / 1e5_f64).powf(0.7)
    } else {
        LAMBDA_M1 * (t / 1e7_f64).powf(0.5)
    };

    lambda_hhe + z_ratio * lambda_metal
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase 119 — Enfriamiento tabulado S&D93 con interpolación bilineal
// ─────────────────────────────────────────────────────────────────────────────

/// Tabla de metalicidades Z/Z_sun (7 bins).
const COOLING_TABLE_Z: [f64; 7] = [0.0, 1e-4, 1e-3, 0.01, 0.1, 1.0, 2.0];

/// Tabla de log10(T/K) (20 bins, de 4.0 a 8.5).
const COOLING_TABLE_LOG_T: [f64; 20] = [
    4.0, 4.25, 4.5, 4.75, 5.0, 5.25, 5.5, 5.75, 6.0, 6.25,
    6.5, 6.75, 7.0, 7.25, 7.5, 7.75, 8.0, 8.25, 8.5, 8.75,
];

/// Tabla de tasas de enfriamiento Λ en unidades internas [(km/s)² / tiempo].
/// Dimensiones: [7 bins Z] × [20 bins log T].
/// Valores calibrados a partir de S&D93 + escala interna gadget-ng.
///
/// Z=0 (primordial H+He):
const COOLING_TABLE: [[f64; 20]; 7] = [
    // Z/Z_sun = 0.0 (primordial)
    [
        0.0,    0.0,    1e-7, 5e-7, 3e-6, 1e-5, 2e-5, 3e-5, 2.5e-5, 2e-5,
        1.5e-5, 1e-5,   8e-6, 6e-6, 5e-6, 4.5e-6, 4e-6, 4e-6, 4e-6, 4e-6,
    ],
    // Z/Z_sun = 1e-4
    [
        0.0,    0.0,    1.1e-7, 5.2e-7, 3.1e-6, 1.05e-5, 2.1e-5, 3.1e-5, 2.6e-5, 2.1e-5,
        1.6e-5, 1.1e-5, 8.2e-6, 6.2e-6, 5.2e-6, 4.6e-6, 4.1e-6, 4.1e-6, 4.1e-6, 4.1e-6,
    ],
    // Z/Z_sun = 1e-3
    [
        0.0,    0.0,    2e-7, 8e-7, 4e-6, 1.3e-5, 2.5e-5, 3.5e-5, 3e-5, 2.4e-5,
        1.8e-5, 1.3e-5, 1e-5, 8e-6, 6.5e-6, 5.5e-6, 5e-6, 5e-6, 5e-6, 5e-6,
    ],
    // Z/Z_sun = 0.01
    [
        0.0,    0.0,    5e-7, 2e-6, 8e-6, 2.5e-5, 4e-5, 5e-5, 4.5e-5, 3.5e-5,
        2.5e-5, 1.8e-5, 1.4e-5, 1.1e-5, 9e-6, 8e-6, 7e-6, 7e-6, 7e-6, 7e-6,
    ],
    // Z/Z_sun = 0.1
    [
        0.0,    0.0,    2e-6, 8e-6, 3e-5, 8e-5, 1.2e-4, 1.4e-4, 1.2e-4, 9e-5,
        6.5e-5, 4.5e-5, 3.5e-5, 2.8e-5, 2.3e-5, 2e-5, 1.8e-5, 1.8e-5, 1.8e-5, 1.8e-5,
    ],
    // Z/Z_sun = 1.0 (solar)
    [
        0.0,    0.0,    1.5e-5, 6e-5, 2e-4, 5e-4, 7e-4, 8e-4, 7e-4, 5e-4,
        3.5e-4, 2.5e-4, 1.8e-4, 1.4e-4, 1.1e-4, 9e-5, 8e-5, 8e-5, 8e-5, 8e-5,
    ],
    // Z/Z_sun = 2.0 (super-solar)
    [
        0.0,    0.0,    2.5e-5, 1e-4, 3.5e-4, 8e-4, 1.1e-3, 1.2e-3, 1.05e-3, 7.5e-4,
        5e-4,   3.5e-4, 2.5e-4, 1.9e-4, 1.5e-4, 1.25e-4, 1.1e-4, 1.1e-4, 1.1e-4, 1.1e-4,
    ],
];

/// Tasa de enfriamiento usando interpolación bilineal en tabla S&D93 (Phase 119).
///
/// Realiza interpolación bilineal en el espacio (Z/Z_sun, log10 T):
/// 1. Convierte `u` a temperatura T usando la relación γ.
/// 2. Encuentra los 4 nodos más cercanos en la tabla (Z_lo, Z_hi, T_lo, T_hi).
/// 3. Interpola bilinealmente para obtener Λ(T, Z).
///
/// # Parámetros
///
/// - `u`: energía interna específica [(km/s)²]
/// - `rho`: densidad local (actualmente no usada, reservada para normalización futura)
/// - `metallicity`: fracción de masa en metales Z
/// - `gamma`: índice adiabático
/// - `t_floor_k`: temperatura mínima [K]; por debajo retorna 0
pub fn cooling_rate_tabular(
    u: f64, _rho: f64, metallicity: f64, gamma: f64, t_floor_k: f64,
) -> f64 {
    let t = u_to_temperature(u, gamma);
    if t <= t_floor_k || t <= 0.0 { return 0.0; }

    let log_t = t.log10();

    // Clamping de log_t al rango de la tabla
    let log_t_min = COOLING_TABLE_LOG_T[0];
    let log_t_max = COOLING_TABLE_LOG_T[COOLING_TABLE_LOG_T.len() - 1];
    let log_t_cl = log_t.clamp(log_t_min, log_t_max);

    // Buscar índice en log_t
    let i_t = COOLING_TABLE_LOG_T
        .windows(2)
        .position(|w| log_t_cl >= w[0] && log_t_cl <= w[1])
        .unwrap_or(COOLING_TABLE_LOG_T.len() - 2);
    let dt = COOLING_TABLE_LOG_T[i_t + 1] - COOLING_TABLE_LOG_T[i_t];
    let ft = if dt > 0.0 { (log_t_cl - COOLING_TABLE_LOG_T[i_t]) / dt } else { 0.0 };

    // Normalización de metalicidad
    const Z_SUN: f64 = 0.0127;
    let z_over_zsun = (metallicity / Z_SUN).max(0.0);
    let z_cl = z_over_zsun.clamp(COOLING_TABLE_Z[0], COOLING_TABLE_Z[COOLING_TABLE_Z.len() - 1]);

    // Buscar índice en Z
    let i_z = COOLING_TABLE_Z
        .windows(2)
        .position(|w| z_cl >= w[0] && z_cl <= w[1])
        .unwrap_or(COOLING_TABLE_Z.len() - 2);
    let dz = COOLING_TABLE_Z[i_z + 1] - COOLING_TABLE_Z[i_z];
    let fz = if dz > 0.0 { (z_cl - COOLING_TABLE_Z[i_z]) / dz } else { 0.0 };

    // Interpolación bilineal
    let l00 = COOLING_TABLE[i_z][i_t];
    let l10 = COOLING_TABLE[i_z + 1][i_t];
    let l01 = COOLING_TABLE[i_z][i_t + 1];
    let l11 = COOLING_TABLE[i_z + 1][i_t + 1];

    let lambda = l00 * (1.0 - fz) * (1.0 - ft)
        + l10 * fz * (1.0 - ft)
        + l01 * (1.0 - fz) * ft
        + l11 * fz * ft;

    lambda.max(0.0)
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

        let lambda = match cfg.cooling {
            CoolingKind::None => 0.0,
            CoolingKind::AtomicHHe => {
                cooling_rate_atomic(p.internal_energy, rho_local, gamma, t_floor_k)
            }
            CoolingKind::MetalCooling => {
                cooling_rate_metal(p.internal_energy, rho_local, p.metallicity, gamma, t_floor_k)
            }
            CoolingKind::MetalTabular => {
                cooling_rate_tabular(p.internal_energy, rho_local, p.metallicity, gamma, t_floor_k)
            }
        };
        if lambda == 0.0 { continue; }
        // du/dt = -Λ · X_H² · ρ / m
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
