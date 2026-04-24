//! Líneas de emisión nebular — Phase 152.
//!
//! ## Modelo físico
//!
//! Las emissividades se calculan para gas ionizado (T > 5000 K) usando
//! recombinación case B (Hα) y excitación colisional ([OIII], [NII]):
//!
//! - **Hα 6563Å**: `j_Hα = 1.37×10⁻²⁵ × n_e² × T^{-0.9}` (Osterbrock 2006).
//! - **[OIII] 5007Å**: `j_OIII = 1.8×10⁻²⁴ × n_e² × (Z/Z_sun) × exp(-29000/T)`.
//! - **[NII] 6583Å**: `j_NII = 4.5×10⁻²⁵ × n_e² × (Z/Z_sun) × exp(-21500/T)`.
//!
//! El diagrama BPT (Baldwin, Phillips & Terlevich 1981) usa los cocientes:
//! - **y-axis**: log₁₀([OIII]/Hβ)  donde Hβ ≈ Hα/2.86 (Balmer decrement)
//! - **x-axis**: log₁₀([NII]/Hα)
//!
//! ## Referencias
//!
//! - Osterbrock & Ferland (2006) "Astrophysics of Gaseous Nebulae".
//! - Baldwin, Phillips & Terlevich (1981) PASP 93, 5.
//! - Kauffmann et al. (2003) MNRAS 346, 1055 (línea de demarcación SF/AGN).

use gadget_ng_core::Particle;

/// Temperatura mínima para gas ionizado [K].
const T_ION_MIN: f64 = 5.0e3;

/// Temperatura solar de referencia para metalicidad [Z_sun = 0.02].
const Z_SUN: f64 = 0.02;

/// k_B / (m_H · μ) para conversión u→T (misma que en cooling.rs).
const KB_OVER_MH_MU: f64 = 8.254e-3 / 0.6;

#[inline]
fn u_to_temperature(u: f64, gamma: f64) -> f64 {
    u * (gamma - 1.0) / KB_OVER_MH_MU
}

/// Resultado de la emisión de líneas para una partícula de gas ionizado (Phase 152).
#[derive(Debug, Clone, PartialEq)]
pub struct EmissionLine {
    /// Emissividad Hα en unidades internas.
    pub h_alpha: f64,
    /// Emissividad [OIII] 5007Å en unidades internas.
    pub oiii: f64,
    /// Emissividad [NII] 6583Å en unidades internas.
    pub nii: f64,
    /// Cociente [NII]/Hα (diagnóstico BPT eje-x).
    pub r_nii_halpha: f64,
    /// Cociente [OIII]/Hβ (diagnóstico BPT eje-y, con Hβ = Hα/2.86).
    pub r_oiii_hbeta: f64,
}

/// Emissividad Hα [case B] (Phase 152).
///
/// `j_Ha ∝ n_e² × T^{-0.9}`, Osterbrock (2006).
///
/// # Parámetros
/// - `rho`: densidad en unidades internas (proporcional a n_e²)
/// - `t_k`: temperatura en Kelvin
pub fn emissivity_halpha(rho: f64, t_k: f64) -> f64 {
    if t_k < T_ION_MIN || rho <= 0.0 { return 0.0; }
    1.37e-25 * rho * rho * t_k.powf(-0.9)
}

/// Emissividad [OIII] 5007Å (excitación colisional) (Phase 152).
///
/// `j_OIII ∝ n_e² × (Z/Z_sun) × exp(-29000/T)`.
///
/// # Parámetros
/// - `rho`: densidad en unidades internas
/// - `t_k`: temperatura en Kelvin
/// - `metallicity`: fracción de masa en metales (Z)
pub fn emissivity_oiii(rho: f64, t_k: f64, metallicity: f64) -> f64 {
    if t_k < T_ION_MIN || rho <= 0.0 { return 0.0; }
    let z_rel = (metallicity / Z_SUN).max(0.0);
    1.8e-24 * rho * rho * z_rel * (-29_000.0 / t_k).exp()
}

/// Emissividad [NII] 6583Å (excitación colisional) (Phase 152).
///
/// `j_NII ∝ n_e² × (Z/Z_sun) × exp(-21500/T)`.
///
/// # Parámetros
/// - `rho`: densidad en unidades internas
/// - `t_k`: temperatura en Kelvin
/// - `metallicity`: fracción de masa en metales (Z)
pub fn emissivity_nii(rho: f64, t_k: f64, metallicity: f64) -> f64 {
    if t_k < T_ION_MIN || rho <= 0.0 { return 0.0; }
    let z_rel = (metallicity / Z_SUN).max(0.0);
    4.5e-25 * rho * rho * z_rel * (-21_500.0 / t_k).exp()
}

/// Calcula las líneas de emisión para cada partícula de gas ionizado (Phase 152).
///
/// Solo procesa partículas de gas con T > T_ION_MIN.
///
/// # Parámetros
/// - `particles`: slice de partículas
/// - `gamma`: índice adiabático
///
/// # Retorna
/// `Vec<EmissionLine>` — un elemento por partícula (en el mismo orden).
/// Las partículas no-gas o frías tienen todas las emissividades a 0.
pub fn compute_emission_lines(particles: &[Particle], gamma: f64) -> Vec<EmissionLine> {
    particles
        .iter()
        .map(|p| {
            if !p.is_gas() {
                return EmissionLine {
                    h_alpha: 0.0, oiii: 0.0, nii: 0.0,
                    r_nii_halpha: 0.0, r_oiii_hbeta: 0.0,
                };
            }
            let rho = if p.smoothing_length > 0.0 {
                p.mass / p.smoothing_length.powi(3)
            } else {
                0.0
            };
            let t_k = u_to_temperature(p.internal_energy, gamma);
        let ha = emissivity_halpha(rho, t_k);
        let oiii = emissivity_oiii(rho, t_k, p.metallicity);
        let nii = emissivity_nii(rho, t_k, p.metallicity);

            let r_nii_halpha = if ha > 0.0 { nii / ha } else { 0.0 };
            // Hβ ≈ Hα / 2.86 (Balmer decrement)
            let h_beta = ha / 2.86;
            let r_oiii_hbeta = if h_beta > 0.0 { oiii / h_beta } else { 0.0 };

            EmissionLine { h_alpha: ha, oiii, nii, r_nii_halpha, r_oiii_hbeta }
        })
        .collect()
}

/// Genera puntos para el diagrama BPT (Phase 152).
///
/// # Retorna
/// `Vec<(f64, f64)>` — (log₁₀([NII]/Hα), log₁₀([OIII]/Hβ)) para cada línea
/// con Hα > 0. Galaxias SF se encuentran bajo la línea de Kauffmann+2003.
pub fn bpt_diagram(lines: &[EmissionLine]) -> Vec<(f64, f64)> {
    lines
        .iter()
        .filter(|l| l.h_alpha > 0.0 && l.oiii > 0.0)
        .map(|l| {
            let x = if l.r_nii_halpha > 0.0 { l.r_nii_halpha.log10() } else { f64::NEG_INFINITY };
            let y = if l.r_oiii_hbeta > 0.0 { l.r_oiii_hbeta.log10() } else { f64::NEG_INFINITY };
            (x, y)
        })
        .collect()
}
