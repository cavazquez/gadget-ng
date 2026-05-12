//! Warm and fuzzy dark-matter transfer suppressions (Phase 184).

use crate::config::{DarkMatterModel, DarkMatterSection};

/// Bode et al.-style WDM transfer-function suppression.
///
/// Returns `T_wdm(k)` for `k` in `h/Mpc`. The matter power spectrum is suppressed
/// by `T_wdm(k)^2`; IC amplitude closures should multiply their Gaussian sigma by
/// this returned value.
pub fn wdm_transfer_suppression(k_hmpc: f64, m_wdm_kev: f64, omega_m: f64, h: f64) -> f64 {
    if k_hmpc <= 0.0 || m_wdm_kev <= 0.0 {
        return 1.0;
    }
    let nu = 1.12_f64;
    let alpha = 0.049
        * m_wdm_kev.powf(-1.11)
        * (omega_m.max(1e-30) / 0.25).powf(0.11)
        * (h.max(1e-30) / 0.7).powf(1.22);
    (1.0 + (alpha * k_hmpc).powf(2.0 * nu)).powf(-5.0 / nu)
}

/// Approximate WDM half-mode scale where the power suppression is 1/2.
pub fn wdm_half_mode_k(m_wdm_kev: f64, omega_m: f64, h: f64) -> f64 {
    if m_wdm_kev <= 0.0 {
        return f64::INFINITY;
    }
    let nu = 1.12_f64;
    let alpha = 0.049
        * m_wdm_kev.powf(-1.11)
        * (omega_m.max(1e-30) / 0.25).powf(0.11)
        * (h.max(1e-30) / 0.7).powf(1.22);
    ((2.0_f64.powf(nu / 10.0) - 1.0).powf(1.0 / (2.0 * nu))) / alpha
}

/// Smooth fuzzy-DM transfer envelope.
///
/// This captures the high-k cutoff controlled by the de Broglie/Jeans scale while
/// avoiding oscillatory sign changes in IC amplitudes. `m_fdm_22` is the particle
/// mass in units of `10^-22 eV`.
pub fn fdm_transfer_suppression(k_hmpc: f64, m_fdm_22: f64, omega_m: f64, h: f64) -> f64 {
    if k_hmpc <= 0.0 || m_fdm_22 <= 0.0 {
        return 1.0;
    }
    let k_half = fdm_half_mode_k(m_fdm_22, omega_m, h);
    (1.0 + (k_hmpc / k_half).powi(8)).powf(-0.5)
}

/// Approximate fuzzy-DM half-mode scale in `h/Mpc`.
pub fn fdm_half_mode_k(m_fdm_22: f64, omega_m: f64, h: f64) -> f64 {
    if m_fdm_22 <= 0.0 {
        return f64::INFINITY;
    }
    9.1 * m_fdm_22.powf(4.0 / 9.0) * (omega_m.max(1e-30) / 0.27).powf(0.11) * (h / 0.7).powf(1.22)
}

/// Effective fuzzy-DM quantum-pressure sound speed squared proxy.
///
/// Scales as `k^4 / (m^2 a^2)` and is intended as a diagnostic or future source
/// term coefficient, not a full Schrödinger-Poisson solver.
pub fn fdm_quantum_pressure_cs2(k_hmpc: f64, m_fdm_22: f64, a: f64) -> f64 {
    if k_hmpc <= 0.0 || m_fdm_22 <= 0.0 || a <= 0.0 {
        return 0.0;
    }
    1.0e-8 * k_hmpc.powi(4) / (m_fdm_22 * m_fdm_22 * a * a)
}

/// Transfer factor selected by [`DarkMatterSection`].
pub fn dark_matter_transfer_suppression(
    cfg: &DarkMatterSection,
    k_hmpc: f64,
    omega_m: f64,
    h: f64,
) -> f64 {
    if !cfg.enabled {
        return 1.0;
    }
    match cfg.model {
        DarkMatterModel::Cold => 1.0,
        DarkMatterModel::Warm => wdm_transfer_suppression(k_hmpc, cfg.m_wdm_kev, omega_m, h),
        DarkMatterModel::Fuzzy => fdm_transfer_suppression(k_hmpc, cfg.m_fdm_22, omega_m, h),
    }
}
