//! Perfil de temperatura del gas intergaláctico T(z) (Phase 90).
//!
//! Calcula estadísticas de temperatura del IGM (gas de baja densidad) desde
//! los estados de energía interna y química de las partículas SPH de gas.
//!
//! ## Definición del IGM
//!
//! Se consideran partículas del IGM aquellas con densidad comóvil δ < δ_threshold
//! (default: 10×densidad media), siguiendo la convención de Lukić et al. (2015).
//!
//! ## Temperatura desde energía interna
//!
//! La temperatura se obtiene de `internal_energy` usando la masa molecular media
//! μ calculada desde el estado de ionización (`ChemState`):
//!
//! ```text
//! T = u × (γ-1) × μ × m_p / k_B
//! μ = 4 / (3 + x_hii + x_heii + 2·x_heiii + 3·x_hei)
//! ```
//!
//! ## Output
//!
//! El perfil `IgmTempBin` se añade al JSON de análisis in-situ como:
//!
//! ```json
//! "igm_temp": { "z": 7.5, "t_mean": 12000.0, "t_median": 11500.0, "t_sigma": 3200.0, "n_particles": 1024 }
//! ```
//!
//! ## Referencia
//!
//! Lukić et al. (2015), MNRAS 446, 3697;
//! Springel (2005), MNRAS 364, 1105.

use std::ops::Range;

use gadget_ng_core::Particle;

use crate::chemistry::ChemState;
#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
use crate::chemistry::{F_D, F_HE, U_CODE_TO_ERG_G};
#[cfg(feature = "rayon")]
use rayon::prelude::*;
#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
use std::arch::is_x86_feature_detected;
#[cfg(all(
    feature = "simd",
    any(target_arch = "x86", target_arch = "x86_64"),
    not(target_arch = "x86_64")
))]
use std::arch::x86::*;
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
use std::arch::x86_64::*;

// ── Structs ───────────────────────────────────────────────────────────────────

/// Estadísticas de temperatura del IGM para un instante de tiempo (un redshift z).
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct IgmTempBin {
    /// Redshift del instante.
    pub z: f64,
    /// Temperatura media del IGM [K].
    pub t_mean: f64,
    /// Temperatura mediana del IGM [K].
    pub t_median: f64,
    /// Desviación estándar de la temperatura [K].
    pub t_sigma: f64,
    /// Temperatura del 16° percentil [K].
    pub t_p16: f64,
    /// Temperatura del 84° percentil [K].
    pub t_p84: f64,
    /// Número de partículas de gas IGM usadas en el cálculo.
    pub n_particles: usize,
}

/// Parámetros para el cálculo del perfil de temperatura IGM.
#[derive(Debug, Clone)]
pub struct IgmTempParams {
    /// Umbral de densidad: partículas con `mass/h_sml³ < rho_max` se consideran IGM.
    /// En unidades de la densidad media del universo; usar `0.0` para incluir todas.
    /// Default: 10.0 (10× la densidad media).
    pub delta_max: f64,
    /// Índice adiabático del gas (default: 5/3).
    pub gamma: f64,
}

impl Default for IgmTempParams {
    fn default() -> Self {
        Self {
            delta_max: 10.0,
            gamma: 5.0 / 3.0,
        }
    }
}

// ── SIMD dispatch: AVX-512F (8-wide) → AVX2+FMA (4-wide) → escalar ─────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(dead_code)]
enum IgmSimdLevel {
    Scalar,
    /// AVX2 + FMA (256 bits).
    Avx2,
    /// AVX-512F en x86_64 (512 bits, 8 `f64` por bloque).
    Avx512,
}

#[inline]
fn igm_simd_level() -> IgmSimdLevel {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx512f") {
            return IgmSimdLevel::Avx512;
        }
    }
    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return IgmSimdLevel::Avx2;
        }
    }
    IgmSimdLevel::Scalar
}

// ── Funciones principales ─────────────────────────────────────────────────────

/// Calcula la temperatura de una partícula de gas desde su energía interna y estado químico.
///
/// Usa la masa molecular media μ adaptativa calculada desde las fracciones de ionización.
///
/// # Argumentos
/// - `internal_energy` — energía interna [unidades internas, erg/g]
/// - `chem`            — estado de ionización de la partícula
/// - `gamma`           — índice adiabático (típicamente 5/3)
///
/// # Retorna
/// Temperatura en Kelvin.
pub fn temperature_from_particle(internal_energy: f64, chem: &ChemState, gamma: f64) -> f64 {
    chem.temperature_from_internal_energy(internal_energy, gamma)
}

fn collect_igm_temperatures_scalar_range(
    particles: &[Particle],
    chem_states: &[ChemState],
    range: Range<usize>,
    mean_density: f64,
    params: &IgmTempParams,
) -> Vec<f64> {
    let mut temperatures = Vec::new();
    for i in range {
        let p = &particles[i];
        if p.internal_energy <= 0.0 {
            continue;
        }
        if mean_density > 0.0 && params.delta_max > 0.0 && p.smoothing_length > 0.0 {
            let rho_sph = p.mass / (p.smoothing_length * p.smoothing_length * p.smoothing_length);
            let delta_threshold = params.delta_max * mean_density;
            if rho_sph > delta_threshold {
                continue;
            }
        }
        let t = temperature_from_particle(p.internal_energy, &chem_states[i], params.gamma);
        if t > 0.0 && t.is_finite() {
            temperatures.push(t);
        }
    }
    temperatures
}

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn igm_temperature_avx2_from_chem(
    chem: &[ChemState],
    base: usize,
    u: __m256d,
    gamma: f64,
) -> __m256d {
    const K_B_CGS: f64 = 1.380649e-16;
    const M_P_CGS: f64 = 1.672623e-24;

    let x_hi = _mm256_set_pd(
        chem[base + 3].x_hi,
        chem[base + 2].x_hi,
        chem[base + 1].x_hi,
        chem[base].x_hi,
    );
    let x_hii = _mm256_set_pd(
        chem[base + 3].x_hii,
        chem[base + 2].x_hii,
        chem[base + 1].x_hii,
        chem[base].x_hii,
    );
    let x_hm = _mm256_set_pd(
        chem[base + 3].x_hm,
        chem[base + 2].x_hm,
        chem[base + 1].x_hm,
        chem[base].x_hm,
    );
    let x_e = _mm256_set_pd(
        chem[base + 3].x_e,
        chem[base + 2].x_e,
        chem[base + 1].x_e,
        chem[base].x_e,
    );
    let x_h2p = _mm256_set_pd(
        chem[base + 3].x_h2p,
        chem[base + 2].x_h2p,
        chem[base + 1].x_h2p,
        chem[base].x_h2p,
    );
    let x_d = _mm256_set_pd(
        chem[base + 3].x_d,
        chem[base + 2].x_d,
        chem[base + 1].x_d,
        chem[base].x_d,
    );
    let x_dp = _mm256_set_pd(
        chem[base + 3].x_dp,
        chem[base + 2].x_dp,
        chem[base + 1].x_dp,
        chem[base].x_dp,
    );
    let x_h2 = _mm256_set_pd(
        chem[base + 3].x_h2,
        chem[base + 2].x_h2,
        chem[base + 1].x_h2,
        chem[base].x_h2,
    );
    let x_hd = _mm256_set_pd(
        chem[base + 3].x_hd,
        chem[base + 2].x_hd,
        chem[base + 1].x_hd,
        chem[base].x_hd,
    );
    let x_hei = _mm256_set_pd(
        chem[base + 3].x_hei,
        chem[base + 2].x_hei,
        chem[base + 1].x_hei,
        chem[base].x_hei,
    );
    let x_heii = _mm256_set_pd(
        chem[base + 3].x_heii,
        chem[base + 2].x_heii,
        chem[base + 1].x_heii,
        chem[base].x_heii,
    );
    let x_heiii = _mm256_set_pd(
        chem[base + 3].x_heiii,
        chem[base + 2].x_heiii,
        chem[base + 1].x_heiii,
        chem[base].x_heiii,
    );

    let free_h = _mm256_add_pd(
        _mm256_add_pd(_mm256_add_pd(_mm256_add_pd(x_hi, x_hii), x_hm), x_e),
        x_h2p,
    );
    let deut = _mm256_add_pd(x_d, x_dp);
    let molec = _mm256_add_pd(x_h2, x_hd);
    let he = _mm256_add_pd(_mm256_add_pd(x_hei, x_heii), x_heiii);
    let sum_denom = _mm256_add_pd(_mm256_add_pd(free_h, deut), _mm256_add_pd(molec, he));
    let min_den = _mm256_set1_pd(1e-30);
    let denom = _mm256_max_pd(sum_denom, min_den);
    let mu_num = _mm256_set1_pd(1.0 + 4.0 * F_HE + 2.0 * F_D);
    let mu = _mm256_div_pd(mu_num, denom);
    let u_cgs = _mm256_mul_pd(u, _mm256_set1_pd(U_CODE_TO_ERG_G));
    let gm1 = _mm256_set1_pd(gamma - 1.0);
    let mp_over_kb = _mm256_set1_pd(M_P_CGS / K_B_CGS);
    let t = _mm256_mul_pd(_mm256_mul_pd(_mm256_mul_pd(u_cgs, gm1), mu), mp_over_kb);
    _mm256_max_pd(t, _mm256_set1_pd(1.0))
}

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn collect_igm_temperatures_avx2_range(
    particles: &[Particle],
    chem_states: &[ChemState],
    range: Range<usize>,
    mean_density: f64,
    params: &IgmTempParams,
) -> Vec<f64> {
    let mut out = Vec::new();
    let lo = range.start;
    let hi = range.end;
    let use_density = mean_density > 0.0 && params.delta_max > 0.0;
    let delta_threshold = params.delta_max * mean_density;
    let gamma = params.gamma;

    let mut i = lo;
    while i + 4 <= hi {
        let u = _mm256_set_pd(
            particles[i + 3].internal_energy,
            particles[i + 2].internal_energy,
            particles[i + 1].internal_energy,
            particles[i].internal_energy,
        );
        let zero = _mm256_setzero_pd();
        let m_energy = _mm256_movemask_pd(_mm256_cmp_pd(u, zero, _CMP_GT_OQ)) as u8 & 0x0f;

        let dens_ok: u8 = if use_density {
            let mass = _mm256_set_pd(
                particles[i + 3].mass,
                particles[i + 2].mass,
                particles[i + 1].mass,
                particles[i].mass,
            );
            let h = _mm256_set_pd(
                particles[i + 3].smoothing_length,
                particles[i + 2].smoothing_length,
                particles[i + 1].smoothing_length,
                particles[i].smoothing_length,
            );
            let tiny = _mm256_set1_pd(1e-300);
            let h_safe = _mm256_max_pd(h, tiny);
            let h3 = _mm256_mul_pd(h_safe, _mm256_mul_pd(h_safe, h_safe));
            let rho = _mm256_div_pd(mass, h3);
            let thresh_v = _mm256_set1_pd(delta_threshold);
            let m_hgt0 = _mm256_movemask_pd(_mm256_cmp_pd(h, zero, _CMP_GT_OQ)) as u8 & 0x0f;
            let m_rho_le =
                _mm256_movemask_pd(_mm256_cmp_pd(rho, thresh_v, _CMP_LE_OQ)) as u8 & 0x0f;
            // Pasa densidad si h≤0 (no se aplica corte) o si ρ≤umbral cuando h>0.
            (!m_hgt0 | (m_hgt0 & m_rho_le)) & 0x0f
        } else {
            0x0f
        };

        let active_before_t = (m_energy & dens_ok) as i32;
        if active_before_t != 0 {
            // SAFETY: AVX2+FMA enabled for this function; `i` is aligned to 4-lane chunks.
            let t_vec = unsafe { igm_temperature_avx2_from_chem(chem_states, i, u, gamma) };
            let mut t_arr = [0.0_f64; 4];
            // SAFETY: fixed-size stack buffer for four lanes.
            unsafe {
                _mm256_storeu_pd(t_arr.as_mut_ptr(), t_vec);
            }
            for (lane, &t) in t_arr.iter().enumerate() {
                if (active_before_t >> lane) & 1 == 0 {
                    continue;
                }
                if t > 0.0 && t.is_finite() {
                    out.push(t);
                }
            }
        }
        i += 4;
    }

    out.extend(collect_igm_temperatures_scalar_range(
        particles,
        chem_states,
        i..hi,
        mean_density,
        params,
    ));
    out
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn igm_temperature_avx512_from_chem(
    chem: &[ChemState],
    base: usize,
    u: __m512d,
    gamma: f64,
) -> __m512d {
    const K_B_CGS: f64 = 1.380649e-16;
    const M_P_CGS: f64 = 1.672623e-24;

    macro_rules! chem8 {
        ($field:ident) => {
            _mm512_set_pd(
                chem[base + 7].$field,
                chem[base + 6].$field,
                chem[base + 5].$field,
                chem[base + 4].$field,
                chem[base + 3].$field,
                chem[base + 2].$field,
                chem[base + 1].$field,
                chem[base].$field,
            )
        };
    }

    let x_hi = chem8!(x_hi);
    let x_hii = chem8!(x_hii);
    let x_hm = chem8!(x_hm);
    let x_e = chem8!(x_e);
    let x_h2p = chem8!(x_h2p);
    let x_d = chem8!(x_d);
    let x_dp = chem8!(x_dp);
    let x_h2 = chem8!(x_h2);
    let x_hd = chem8!(x_hd);
    let x_hei = chem8!(x_hei);
    let x_heii = chem8!(x_heii);
    let x_heiii = chem8!(x_heiii);

    let free_h = _mm512_add_pd(
        _mm512_add_pd(_mm512_add_pd(_mm512_add_pd(x_hi, x_hii), x_hm), x_e),
        x_h2p,
    );
    let deut = _mm512_add_pd(x_d, x_dp);
    let molec = _mm512_add_pd(x_h2, x_hd);
    let he = _mm512_add_pd(_mm512_add_pd(x_hei, x_heii), x_heiii);
    let sum_denom = _mm512_add_pd(_mm512_add_pd(free_h, deut), _mm512_add_pd(molec, he));
    let min_den = _mm512_set1_pd(1e-30);
    let denom = _mm512_max_pd(sum_denom, min_den);
    let mu_num = _mm512_set1_pd(1.0 + 4.0 * F_HE + 2.0 * F_D);
    let mu = _mm512_div_pd(mu_num, denom);
    let u_cgs = _mm512_mul_pd(u, _mm512_set1_pd(U_CODE_TO_ERG_G));
    let gm1 = _mm512_set1_pd(gamma - 1.0);
    let mp_over_kb = _mm512_set1_pd(M_P_CGS / K_B_CGS);
    let t = _mm512_mul_pd(_mm512_mul_pd(_mm512_mul_pd(u_cgs, gm1), mu), mp_over_kb);
    _mm512_max_pd(t, _mm512_set1_pd(1.0))
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn collect_igm_temperatures_avx512_range(
    particles: &[Particle],
    chem_states: &[ChemState],
    range: Range<usize>,
    mean_density: f64,
    params: &IgmTempParams,
) -> Vec<f64> {
    let mut out = Vec::new();
    let lo = range.start;
    let hi = range.end;
    let use_density = mean_density > 0.0 && params.delta_max > 0.0;
    let delta_threshold = params.delta_max * mean_density;
    let gamma = params.gamma;

    let mut i = lo;
    while i + 8 <= hi {
        let u = _mm512_set_pd(
            particles[i + 7].internal_energy,
            particles[i + 6].internal_energy,
            particles[i + 5].internal_energy,
            particles[i + 4].internal_energy,
            particles[i + 3].internal_energy,
            particles[i + 2].internal_energy,
            particles[i + 1].internal_energy,
            particles[i].internal_energy,
        );
        let zero = _mm512_setzero_pd();
        let m_energy = _mm512_cmp_pd_mask(u, zero, _CMP_GT_OQ) as u8;

        let dens_ok: u8 = if use_density {
            let mass = _mm512_set_pd(
                particles[i + 7].mass,
                particles[i + 6].mass,
                particles[i + 5].mass,
                particles[i + 4].mass,
                particles[i + 3].mass,
                particles[i + 2].mass,
                particles[i + 1].mass,
                particles[i].mass,
            );
            let h = _mm512_set_pd(
                particles[i + 7].smoothing_length,
                particles[i + 6].smoothing_length,
                particles[i + 5].smoothing_length,
                particles[i + 4].smoothing_length,
                particles[i + 3].smoothing_length,
                particles[i + 2].smoothing_length,
                particles[i + 1].smoothing_length,
                particles[i].smoothing_length,
            );
            let tiny = _mm512_set1_pd(1e-300);
            let h_safe = _mm512_max_pd(h, tiny);
            let h3 = _mm512_mul_pd(h_safe, _mm512_mul_pd(h_safe, h_safe));
            let rho = _mm512_div_pd(mass, h3);
            let thresh_v = _mm512_set1_pd(delta_threshold);
            let m_hgt0 = _mm512_cmp_pd_mask(h, zero, _CMP_GT_OQ) as u8;
            let m_rho_le = _mm512_cmp_pd_mask(rho, thresh_v, _CMP_LE_OQ) as u8;
            (!m_hgt0 | (m_hgt0 & m_rho_le)) as u8
        } else {
            0xff
        };

        let active_before_t = m_energy & dens_ok;
        if active_before_t != 0 {
            let t_vec = unsafe { igm_temperature_avx512_from_chem(chem_states, i, u, gamma) };
            let mut t_arr = [0.0_f64; 8];
            // SAFETY: fixed-size stack buffer for eight lanes.
            unsafe {
                _mm512_storeu_pd(t_arr.as_mut_ptr(), t_vec);
            }
            for (lane, &t) in t_arr.iter().enumerate() {
                if (active_before_t >> lane) & 1 == 0 {
                    continue;
                }
                if t > 0.0 && t.is_finite() {
                    out.push(t);
                }
            }
        }
        i += 8;
    }

    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    {
        if i + 4 <= hi {
            // SAFETY: caller verified AVX-512F; AVX2+FMA is a subset on the same CPU.
            out.extend(unsafe {
                collect_igm_temperatures_avx2_range(
                    particles,
                    chem_states,
                    i..hi,
                    mean_density,
                    params,
                )
            });
            return out;
        }
    }

    out.extend(collect_igm_temperatures_scalar_range(
        particles,
        chem_states,
        i..hi,
        mean_density,
        params,
    ));
    out
}

fn collect_igm_temperatures_dispatch_range(
    particles: &[Particle],
    chem_states: &[ChemState],
    range: Range<usize>,
    mean_density: f64,
    params: &IgmTempParams,
    level: IgmSimdLevel,
) -> Vec<f64> {
    match level {
        IgmSimdLevel::Scalar => collect_igm_temperatures_scalar_range(
            particles,
            chem_states,
            range,
            mean_density,
            params,
        ),
        IgmSimdLevel::Avx512 => {
            #[cfg(all(feature = "simd", target_arch = "x86_64"))]
            {
                // SAFETY: `Avx512` implies runtime `avx512f` detection on x86_64.
                unsafe {
                    collect_igm_temperatures_avx512_range(
                        particles,
                        chem_states,
                        range,
                        mean_density,
                        params,
                    )
                }
            }
            #[cfg(not(all(feature = "simd", target_arch = "x86_64")))]
            {
                #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
                {
                    // SAFETY: fallback AVX2+FMA when AVX-512 is unavailable at compile time.
                    unsafe {
                        collect_igm_temperatures_avx2_range(
                            particles,
                            chem_states,
                            range,
                            mean_density,
                            params,
                        )
                    }
                }
                #[cfg(not(all(
                    feature = "simd",
                    any(target_arch = "x86", target_arch = "x86_64")
                )))]
                {
                    collect_igm_temperatures_scalar_range(
                        particles,
                        chem_states,
                        range,
                        mean_density,
                        params,
                    )
                }
            }
        }
        IgmSimdLevel::Avx2 => {
            #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
            {
                // SAFETY: `Avx2` implies runtime detection of AVX2 and FMA.
                unsafe {
                    collect_igm_temperatures_avx2_range(
                        particles,
                        chem_states,
                        range,
                        mean_density,
                        params,
                    )
                }
            }
            #[cfg(not(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64"))))]
            {
                collect_igm_temperatures_scalar_range(
                    particles,
                    chem_states,
                    range,
                    mean_density,
                    params,
                )
            }
        }
    }
}

/// Tamaño de rango por hilo Rayon: equilibrio entre overhead y reutilización SIMD.
#[cfg(feature = "rayon")]
const IGM_TEMP_RAYON_CHUNK: usize = 2048;

fn collect_igm_temperatures(
    particles: &[Particle],
    chem_states: &[ChemState],
    mean_density: f64,
    params: &IgmTempParams,
) -> Vec<f64> {
    let n = particles.len().min(chem_states.len());
    let level = igm_simd_level();

    #[cfg(feature = "rayon")]
    {
        let starts: Vec<usize> = (0..n).step_by(IGM_TEMP_RAYON_CHUNK).collect();
        let parts: Vec<Vec<f64>> = starts
            .par_iter()
            .map(|&start| {
                let end = (start + IGM_TEMP_RAYON_CHUNK).min(n);
                collect_igm_temperatures_dispatch_range(
                    particles,
                    chem_states,
                    start..end,
                    mean_density,
                    params,
                    level,
                )
            })
            .collect();
        parts.into_iter().flatten().collect()
    }

    #[cfg(not(feature = "rayon"))]
    {
        collect_igm_temperatures_dispatch_range(
            particles,
            chem_states,
            0..n,
            mean_density,
            params,
            level,
        )
    }
}

#[cfg(not(feature = "rayon"))]
fn compute_igm_temp_profile_impl(
    particles: &[Particle],
    chem_states: &[ChemState],
    mean_density: f64,
    z: f64,
    params: &IgmTempParams,
) -> IgmTempBin {
    if particles.is_empty() || chem_states.is_empty() {
        return IgmTempBin {
            z,
            ..Default::default()
        };
    }

    let mut temperatures = collect_igm_temperatures(particles, chem_states, mean_density, params);
    compute_igm_temp_stats(&mut temperatures, z)
}

#[cfg(feature = "rayon")]
fn compute_igm_temp_profile_par(
    particles: &[Particle],
    chem_states: &[ChemState],
    mean_density: f64,
    z: f64,
    params: &IgmTempParams,
) -> IgmTempBin {
    if particles.is_empty() || chem_states.is_empty() {
        return IgmTempBin {
            z,
            ..Default::default()
        };
    }

    let mut temperatures = collect_igm_temperatures(particles, chem_states, mean_density, params);
    compute_igm_temp_stats(&mut temperatures, z)
}

fn compute_igm_temp_stats(temperatures: &mut [f64], z: f64) -> IgmTempBin {
    if temperatures.is_empty() {
        return IgmTempBin {
            z,
            ..Default::default()
        };
    }

    let n_p = temperatures.len();
    let t_mean = temperatures.iter().sum::<f64>() / n_p as f64;
    let t_var = temperatures
        .iter()
        .map(|&t| (t - t_mean).powi(2))
        .sum::<f64>()
        / n_p as f64;
    let t_sigma = t_var.sqrt();

    temperatures.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let t_median = percentile(temperatures, 0.50);
    let t_p16 = percentile(temperatures, 0.16);
    let t_p84 = percentile(temperatures, 0.84);

    IgmTempBin {
        z,
        t_mean,
        t_median,
        t_sigma,
        t_p16,
        t_p84,
        n_particles: n_p,
    }
}

pub fn compute_igm_temp_profile(
    particles: &[Particle],
    chem_states: &[ChemState],
    mean_density: f64,
    z: f64,
    params: &IgmTempParams,
) -> IgmTempBin {
    #[cfg(feature = "rayon")]
    {
        compute_igm_temp_profile_par(particles, chem_states, mean_density, z, params)
    }

    #[cfg(not(feature = "rayon"))]
    {
        compute_igm_temp_profile_impl(particles, chem_states, mean_density, z, params)
    }
}

/// Versión simplificada sin filtro de densidad, útil para pruebas y análisis rápidos.
pub fn compute_igm_temp_all(
    particles: &[Particle],
    chem_states: &[ChemState],
    z: f64,
    gamma: f64,
) -> IgmTempBin {
    let params = IgmTempParams {
        delta_max: f64::MAX,
        gamma,
    };
    compute_igm_temp_profile(particles, chem_states, 0.0, z, &params)
}

// ── Función auxiliar ──────────────────────────────────────────────────────────

/// Percentil p ∈ [0, 1] de un slice ya ordenado.
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use gadget_ng_core::Vec3;

    fn make_gas_particle(internal_energy: f64, target_density: f64) -> Particle {
        let mass = 1e-6f64;
        let h = if target_density > 0.0 {
            (mass / target_density).cbrt()
        } else {
            0.1
        };
        Particle::new_gas(
            0,
            mass,
            Vec3::new(0.5, 0.5, 0.5),
            Vec3::new(0.0, 0.0, 0.0),
            internal_energy,
            h,
        )
    }

    fn warm_ionized_chem() -> ChemState {
        let mut s = ChemState::neutral();
        s.x_hii = 0.9;
        s.x_hi = 0.1;
        s.x_e = 0.9;
        s
    }

    #[test]
    fn temperature_from_particle_reasonable() {
        let chem = warm_ionized_chem();
        let u_code = 2100.0_f64;
        let t = temperature_from_particle(u_code, &chem, 5.0 / 3.0);
        assert!(t > 1e3 && t < 1e7, "T = {t:.2e} K fuera del rango esperado");
    }

    #[test]
    fn compute_igm_temp_profile_empty_returns_default() {
        let result = compute_igm_temp_profile(&[], &[], 1e-4, 7.0, &IgmTempParams::default());
        assert_eq!(result.n_particles, 0);
        assert_eq!(result.z, 7.0);
    }

    #[test]
    fn compute_igm_temp_profile_filters_high_density() {
        let params = IgmTempParams {
            delta_max: 10.0,
            gamma: 5.0 / 3.0,
        };
        let mean_density = 1e-4;

        let u_code = 2100.0_f64;
        let igm_particle = make_gas_particle(u_code, mean_density * 5.0);
        let halo_particle = make_gas_particle(u_code, mean_density * 100.0);

        let chem = vec![warm_ionized_chem(), warm_ionized_chem()];
        let particles = vec![igm_particle, halo_particle];

        let result = compute_igm_temp_profile(&particles, &chem, mean_density, 7.0, &params);
        assert_eq!(result.n_particles, 1, "Solo debe incluir la partícula IGM");
    }

    #[test]
    fn compute_igm_temp_profile_mean_and_median_reasonable() {
        let n = 100;
        let params = IgmTempParams::default();
        let u_code = 2100.0_f64;
        let particles: Vec<Particle> = (0..n).map(|_| make_gas_particle(u_code, 0.0)).collect();
        let chems: Vec<ChemState> = (0..n).map(|_| warm_ionized_chem()).collect();

        let result = compute_igm_temp_profile(&particles, &chems, 0.0, 7.0, &params);
        assert_eq!(result.n_particles, n);
        assert!(
            (result.t_mean - result.t_median).abs() / result.t_mean < 0.01,
            "Con energía uniforme, media ≈ mediana"
        );
        assert!(
            result.t_sigma < result.t_mean * 0.01,
            "Con energía uniforme, sigma debe ser muy pequeña"
        );
    }

    #[test]
    fn compute_igm_temp_all_includes_all_particles() {
        let u_code = 2100.0_f64;
        let particles = vec![
            make_gas_particle(u_code, 1e-2),
            make_gas_particle(u_code, 1e-6),
        ];
        let chems = vec![warm_ionized_chem(), warm_ionized_chem()];
        let result = compute_igm_temp_all(&particles, &chems, 7.0, 5.0 / 3.0);
        assert_eq!(
            result.n_particles, 2,
            "compute_igm_temp_all incluye todas las partículas"
        );
    }

    #[test]
    fn percentile_correct() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(percentile(&data, 0.0), 1.0);
        assert_eq!(percentile(&data, 1.0), 5.0);
        assert_eq!(percentile(&data, 0.5), 3.0);
    }

    #[test]
    fn igm_temp_bin_default_is_zero() {
        let bin = IgmTempBin::default();
        assert_eq!(bin.t_mean, 0.0);
        assert_eq!(bin.n_particles, 0);
    }

    #[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
    #[test]
    fn igm_collect_simd_matches_scalar_multiset() {
        let level = igm_simd_level();
        if level == IgmSimdLevel::Scalar {
            return;
        }
        let n = 99_usize;
        let mean_density = 1e-4_f64;
        let params = IgmTempParams {
            delta_max: 10.0,
            gamma: 5.0 / 3.0,
        };
        let particles: Vec<Particle> = (0..n)
            .map(|i| {
                let rho = mean_density * (1.5 + (i % 7) as f64 * 0.2);
                make_gas_particle(1800.0 + (i % 5) as f64 * 50.0, rho)
            })
            .collect();
        let mut chems: Vec<ChemState> = (0..n)
            .map(|i| {
                let mut s = ChemState::neutral();
                let x = (i % 11) as f64 * 0.05;
                s.x_hii = x.clamp(0.0, 0.95);
                s.x_hi = (1.0 - s.x_hii - 0.01).max(0.0);
                s.x_e = s.x_hii + 0.01;
                s
            })
            .collect();
        for c in &mut chems {
            c.clamp_and_normalize();
        }

        let mut scalar =
            collect_igm_temperatures_scalar_range(&particles, &chems, 0..n, mean_density, &params);
        let mut vector = collect_igm_temperatures_dispatch_range(
            &particles,
            &chems,
            0..n,
            mean_density,
            &params,
            level,
        );

        scalar.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        vector.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        assert_eq!(scalar.len(), vector.len(), "mismo cardinal IGM");
        for (a, b) in scalar.iter().zip(vector.iter()) {
            assert!(
                (a - b).abs() < 1e-9_f64.max(*a * 1e-12),
                "T scalar {a} vs simd {b}"
            );
        }
    }
}
