//! Función de luminosidad y colores galácticos — SSP analítica (Phase 118) + SED SPS (Phase 153).
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

use crate::sps_tables::{Spsband, sps_luminosity};
use gadget_ng_core::{Particle, ParticleType};
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
    if mass <= 0.0 {
        return 0.0;
    }
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

/// Resultado SED multicolor para una galaxia (Phase 153).
#[derive(Debug, Clone, PartialEq)]
pub struct SedResult {
    /// Luminosidad total en banda U [L☉].
    pub l_u: f64,
    /// Luminosidad total en banda B [L☉].
    pub l_b: f64,
    /// Luminosidad total en banda V [L☉].
    pub l_v: f64,
    /// Luminosidad total en banda R [L☉].
    pub l_r: f64,
    /// Luminosidad total en banda I [L☉].
    pub l_i: f64,
    /// Color B-V ponderado por L_B.
    pub bv: f64,
    /// Color V-R ponderado por L_V.
    pub vr: f64,
    /// Edad media ponderada por masa [Gyr].
    pub mass_weighted_age: f64,
    /// Número de partículas estelares.
    pub n_stars: usize,
}

/// Calcula la SED multicolor de una galaxia usando tablas SPS BC03-lite (Phase 153).
///
/// Solo las partículas `Star` contribuyen. La luminosidad de cada partícula
/// se calcula interpolando en la grilla SPS por edad y metalicidad.
///
/// # Parámetros
/// - `particles`: slice de partículas (se filtran las estelares)
///
/// # Retorna
/// `SedResult` con luminosidades por banda, colores y edad media ponderada.
pub fn galaxy_sed(particles: &[Particle]) -> SedResult {
    #[cfg(feature = "rayon")]
    {
        let acc = particles
            .par_iter()
            .map(sed_contribution)
            .reduce(SedAccumulator::default, SedAccumulator::combine);
        acc.into_result()
    }

    #[cfg(not(feature = "rayon"))]
    {
        galaxy_sed_serial(particles)
    }
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
    #[cfg(feature = "rayon")]
    {
        let acc = particles.par_iter().map(luminosity_contribution).reduce(
            LuminosityAccumulator::default,
            LuminosityAccumulator::combine,
        );
        acc.into_result()
    }

    #[cfg(not(feature = "rayon"))]
    {
        galaxy_luminosity_serial(particles)
    }
}

#[derive(Default)]
struct SedAccumulator {
    l_u: f64,
    l_b: f64,
    l_v: f64,
    l_r: f64,
    l_i: f64,
    mass_age: f64,
    mass_tot: f64,
    n_stars: usize,
}

impl SedAccumulator {
    #[cfg(feature = "rayon")]
    fn combine(self, other: Self) -> Self {
        Self {
            l_u: self.l_u + other.l_u,
            l_b: self.l_b + other.l_b,
            l_v: self.l_v + other.l_v,
            l_r: self.l_r + other.l_r,
            l_i: self.l_i + other.l_i,
            mass_age: self.mass_age + other.mass_age,
            mass_tot: self.mass_tot + other.mass_tot,
            n_stars: self.n_stars + other.n_stars,
        }
    }

    fn into_result(self) -> SedResult {
        let bv = if self.l_b > 0.0 && self.l_v > 0.0 {
            -2.5 * (self.l_b / self.l_v).log10()
        } else {
            0.0
        };
        let vr = if self.l_v > 0.0 && self.l_r > 0.0 {
            -2.5 * (self.l_v / self.l_r).log10()
        } else {
            0.0
        };
        let mass_weighted_age = if self.mass_tot > 0.0 {
            self.mass_age / self.mass_tot
        } else {
            0.0
        };

        SedResult {
            l_u: self.l_u,
            l_b: self.l_b,
            l_v: self.l_v,
            l_r: self.l_r,
            l_i: self.l_i,
            bv,
            vr,
            mass_weighted_age,
            n_stars: self.n_stars,
        }
    }
}

#[cfg(feature = "rayon")]
fn sed_contribution(p: &Particle) -> SedAccumulator {
    if p.ptype != ParticleType::Star {
        return SedAccumulator::default();
    }
    let age = p.stellar_age.max(1e-3);
    let z = p.metallicity;
    let m = p.mass;
    SedAccumulator {
        l_u: m * sps_luminosity(age, z, Spsband::U),
        l_b: m * sps_luminosity(age, z, Spsband::B),
        l_v: m * sps_luminosity(age, z, Spsband::V),
        l_r: m * sps_luminosity(age, z, Spsband::R),
        l_i: m * sps_luminosity(age, z, Spsband::I),
        mass_age: m * age,
        mass_tot: m,
        n_stars: 1,
    }
}

#[cfg(feature = "rayon")]
#[derive(Default)]
struct LuminosityAccumulator {
    l_total: f64,
    bv_weighted: f64,
    gr_weighted: f64,
    n_stars: usize,
}

#[cfg(feature = "rayon")]
impl LuminosityAccumulator {
    fn combine(self, other: Self) -> Self {
        Self {
            l_total: self.l_total + other.l_total,
            bv_weighted: self.bv_weighted + other.bv_weighted,
            gr_weighted: self.gr_weighted + other.gr_weighted,
            n_stars: self.n_stars + other.n_stars,
        }
    }

    fn into_result(self) -> LuminosityResult {
        let (bv, gr) = if self.l_total > 0.0 {
            (
                self.bv_weighted / self.l_total,
                self.gr_weighted / self.l_total,
            )
        } else {
            (0.0, 0.0)
        };

        LuminosityResult {
            l_total: self.l_total,
            bv,
            gr,
            n_stars: self.n_stars,
        }
    }
}

#[cfg(feature = "rayon")]
fn luminosity_contribution(p: &Particle) -> LuminosityAccumulator {
    if p.ptype != ParticleType::Star {
        return LuminosityAccumulator::default();
    }
    let age = p.stellar_age.max(1e-4);
    let z = p.metallicity;
    let l_i = stellar_luminosity_solar(p.mass, age, z);
    LuminosityAccumulator {
        l_total: l_i,
        bv_weighted: l_i * bv_color(age, z),
        gr_weighted: l_i * gr_color(age, z),
        n_stars: 1,
    }
}

#[cfg(not(feature = "rayon"))]
fn galaxy_luminosity_serial(particles: &[Particle]) -> LuminosityResult {
    let mut l = Vec::new();
    let mut lbv = Vec::new();
    let mut lgr = Vec::new();

    for p in particles {
        if p.ptype != ParticleType::Star {
            continue;
        }
        let age = p.stellar_age.max(1e-4);
        let z = p.metallicity;
        let l_i = stellar_luminosity_solar(p.mass, age, z);
        l.push(l_i);
        lbv.push(l_i * bv_color(age, z));
        lgr.push(l_i * gr_color(age, z));
    }

    let l_total = sum_f64(&l);
    let bv_weighted = sum_f64(&lbv);
    let gr_weighted = sum_f64(&lgr);
    let (bv, gr) = if l_total > 0.0 {
        (bv_weighted / l_total, gr_weighted / l_total)
    } else {
        (0.0, 0.0)
    };

    LuminosityResult {
        l_total,
        bv,
        gr,
        n_stars: l.len(),
    }
}

#[cfg(not(feature = "rayon"))]
fn galaxy_sed_serial(particles: &[Particle]) -> SedResult {
    let mut l_u = Vec::new();
    let mut l_b = Vec::new();
    let mut l_v = Vec::new();
    let mut l_r = Vec::new();
    let mut l_i = Vec::new();
    let mut mass_age = Vec::new();
    let mut mass = Vec::new();

    for p in particles {
        if p.ptype != ParticleType::Star {
            continue;
        }
        let age = p.stellar_age.max(1e-3);
        let z = p.metallicity;
        let m = p.mass;

        l_u.push(m * sps_luminosity(age, z, Spsband::U));
        l_b.push(m * sps_luminosity(age, z, Spsband::B));
        l_v.push(m * sps_luminosity(age, z, Spsband::V));
        l_r.push(m * sps_luminosity(age, z, Spsband::R));
        l_i.push(m * sps_luminosity(age, z, Spsband::I));
        mass_age.push(m * age);
        mass.push(m);
    }

    let acc = SedAccumulator {
        l_u: sum_f64(&l_u),
        l_b: sum_f64(&l_b),
        l_v: sum_f64(&l_v),
        l_r: sum_f64(&l_r),
        l_i: sum_f64(&l_i),
        mass_age: sum_f64(&mass_age),
        mass_tot: sum_f64(&mass),
        n_stars: mass.len(),
    };
    acc.into_result()
}

#[cfg(not(feature = "rayon"))]
fn sum_f64(values: &[f64]) -> f64 {
    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    {
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx512f") {
            // SAFETY: AVX-512F availability was checked at runtime.
            unsafe {
                return sum_f64_avx512(values);
            }
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: AVX2+FMA availability was checked at runtime.
            unsafe {
                return sum_f64_avx2(values);
            }
        }
    }

    values.iter().sum()
}

#[cfg(all(
    not(feature = "rayon"),
    feature = "simd",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn sum_f64_avx2(values: &[f64]) -> f64 {
    let lanes = 4;
    let chunks = values.len() / lanes * lanes;
    let mut acc = _mm256_setzero_pd();
    let mut i = 0;
    while i < chunks {
        // SAFETY: `i..i+4` is in-bounds by construction and unaligned loads are permitted.
        let v = unsafe { _mm256_loadu_pd(values.as_ptr().add(i)) };
        acc = _mm256_add_pd(acc, v);
        i += lanes;
    }
    let mut tmp = [0.0; 4];
    // SAFETY: fixed-size stack array has exactly four f64 lanes.
    unsafe { _mm256_storeu_pd(tmp.as_mut_ptr(), acc) };
    tmp.into_iter().sum::<f64>() + values[chunks..].iter().sum::<f64>()
}

#[cfg(all(
    not(feature = "rayon"),
    feature = "simd",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[target_feature(enable = "avx512f")]
unsafe fn sum_f64_avx512(values: &[f64]) -> f64 {
    let lanes = 8;
    let chunks = values.len() / lanes * lanes;
    let mut acc = _mm512_setzero_pd();
    let mut i = 0;
    while i < chunks {
        // SAFETY: `i..i+8` is in-bounds by construction and unaligned loads are permitted.
        let v = unsafe { _mm512_loadu_pd(values.as_ptr().add(i)) };
        acc = _mm512_add_pd(acc, v);
        i += lanes;
    }
    let mut tmp = [0.0; 8];
    // SAFETY: fixed-size stack array has exactly eight f64 lanes.
    unsafe { _mm512_storeu_pd(tmp.as_mut_ptr(), acc) };
    tmp.into_iter().sum::<f64>() + values[chunks..].iter().sum::<f64>()
}
