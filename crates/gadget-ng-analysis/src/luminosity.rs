//! Función de luminosidad y colores galácticos — SSP analítica (Phase 118).
//!
//! ## Modelo
//!
//! Se implementa una síntesis de población estelar (SSP) analítica simplificada
//! basada en un fitting de un parámetro de Bruzual & Charlot (2003).
//!
//! ### Luminosidad estelar
//!
//! ```text
//! L/L_sun = (M/M_sun) × age_Gyr^{-0.8} × f_Z(Z)
//! ```
//!
//! donde `f_Z(Z) = 1 + 2.5 × log10(max(Z, 0.0004) / 0.02)` es la corrección
//! de metalicidad respecto al Solar.
//!
//! ### Índices de color
//!
//! Ajustes empíricos de Worthey (1994) + BC03 simplificado:
//!
//! - **B-V** ≈ 0.35 + 0.25 × log10(age_Gyr + 0.01) + 0.1 × log10(max(Z, 0.001))
//! - **g-r** ≈ 0.24 + 0.18 × log10(age_Gyr + 0.01) + 0.07 × log10(max(Z, 0.001))
//!
//! Los colores son más rojos para poblaciones más viejas y más metálicas.
//!
//! ## Referencia
//!
//! Bruzual & Charlot (2003) MNRAS 344, 1000 — modelos de población estelar.
//! Worthey (1994) ApJS 95, 107 — índices espectrales.

use gadget_ng_core::{Particle, ParticleType};

/// Resultado del cálculo de luminosidad para una galaxia o cúmulo (Phase 118).
#[derive(Debug, Clone, PartialEq)]
pub struct LuminosityResult {
    /// Luminosidad total en unidades solares (L_sun).
    pub l_total: f64,
    /// Índice de color B-V (mag). Galaxias azules: B-V ≈ 0.4; rojas: B-V ≈ 0.9.
    pub bv: f64,
    /// Índice de color g-r (SDSS) (mag). Galaxias azules: g-r ≈ 0.3; rojas: g-r ≈ 0.7.
    pub gr: f64,
    /// Número de partículas estelares contribuyentes.
    pub n_stars: usize,
}

/// Luminosidad de una población estelar simple en unidades solares (Phase 118).
///
/// Modelo analítico BC03 de un parámetro:
/// `L/L_sun = (M/M_sun) × age^{-0.8} × f_Z(metallicity)`
///
/// # Parámetros
///
/// - `mass`: masa de la población en M_sun
/// - `age_gyr`: edad de la población en Gyr (debe ser > 0)
/// - `metallicity`: fracción de masa en metales Z (típicamente 0.001–0.04)
///
/// # Retorna
///
/// Luminosidad en unidades de L_sun.
pub fn stellar_luminosity_solar(mass: f64, age_gyr: f64, metallicity: f64) -> f64 {
    if mass <= 0.0 { return 0.0; }
    let age_safe = age_gyr.max(1e-3); // evitar singularidad en age = 0
    let z_safe = metallicity.max(4e-4); // mínimo sub-solar

    // Corrección de metalicidad respecto a Z_sun = 0.02
    let f_z = 1.0 + 2.5 * (z_safe / 0.02).log10();

    mass * age_safe.powf(-0.8) * f_z.max(0.1)
}

/// Índice de color B-V para una población estelar (Phase 118).
///
/// Fitting empírico BC03 + Worthey (1994).
///
/// # Parámetros
///
/// - `age_gyr`: edad en Gyr
/// - `metallicity`: fracción de masa en metales
///
/// # Retorna
///
/// B-V en magnitudes (typically 0.0–1.0).
pub fn bv_color(age_gyr: f64, metallicity: f64) -> f64 {
    let log_age = (age_gyr.max(1e-3) + 0.01).log10();
    let log_z = metallicity.max(1e-3).log10();
    (0.35 + 0.25 * log_age + 0.10 * log_z).clamp(-0.3, 1.5)
}

/// Índice de color g-r (SDSS) para una población estelar (Phase 118).
///
/// Fitting empírico BC03 + calibración SDSS.
///
/// # Parámetros
///
/// - `age_gyr`: edad en Gyr
/// - `metallicity`: fracción de masa en metales
///
/// # Retorna
///
/// g-r en magnitudes (typically 0.0–0.9).
pub fn gr_color(age_gyr: f64, metallicity: f64) -> f64 {
    let log_age = (age_gyr.max(1e-3) + 0.01).log10();
    let log_z = metallicity.max(1e-3).log10();
    (0.24 + 0.18 * log_age + 0.07 * log_z).clamp(-0.2, 1.2)
}

/// Calcula la luminosidad total y colores de una galaxia desde sus partículas (Phase 118).
///
/// Solo las partículas de tipo `Star` contribuyen.
/// El color medio ponderado por luminosidad se calcula para B-V y g-r.
///
/// # Parámetros
///
/// - `particles`: slice de partículas (se filtran las estelares)
///
/// # Retorna
///
/// `LuminosityResult` con luminosidad total, colores promedio y número de estrellas.
pub fn galaxy_luminosity(particles: &[Particle]) -> LuminosityResult {
    let mut l_total = 0.0_f64;
    let mut bv_weighted = 0.0_f64;
    let mut gr_weighted = 0.0_f64;
    let mut n_stars = 0_usize;

    for p in particles {
        if p.ptype != ParticleType::Star { continue; }
        let age = p.stellar_age.max(1e-4);
        let z = p.metallicity;
        let l_i = stellar_luminosity_solar(p.mass, age, z);

        l_total += l_i;
        bv_weighted += l_i * bv_color(age, z);
        gr_weighted += l_i * gr_color(age, z);
        n_stars += 1;
    }

    let (bv, gr) = if l_total > 0.0 {
        (bv_weighted / l_total, gr_weighted / l_total)
    } else {
        (0.0, 0.0)
    };

    LuminosityResult { l_total, bv, gr, n_stars }
}
