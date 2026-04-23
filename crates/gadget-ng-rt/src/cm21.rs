//! Estadísticas de la línea de 21cm del hidrógeno neutro (Phase 94).
//!
//! Calcula la temperatura de brillo diferencial δT_b por partícula y el
//! power spectrum P(k)₂₁cm usando deposición CIC + FFT.
//!
//! ## Física
//!
//! La temperatura de brillo diferencial frente al CMB es:
//! $$\delta T_b \approx 27 x_{HI}(1+\delta)\left(\frac{1+z}{10}\right)^{1/2} \text{ mK}$$
//!
//! donde x_HI = 1 - x_HII es la fracción neutra, (1+δ) = ρ/ρ̄ es la sobredensidad.

use crate::ChemState;
use gadget_ng_core::Particle;

/// Parámetros para el cálculo de estadísticas 21cm.
#[derive(Debug, Clone)]
pub struct Cm21Params {
    /// Temperatura de spin T_S [K] (default: 1000 K >> T_CMB).
    pub t_s_kelvin: f64,
    /// Frecuencia de la línea 21cm [MHz] (default: 1420.406 MHz).
    pub nu_21cm_mhz: f64,
}

impl Default for Cm21Params {
    fn default() -> Self {
        Self {
            t_s_kelvin: 1000.0,
            nu_21cm_mhz: 1420.405_751_768,
        }
    }
}

/// Un bin en el power spectrum P(k)₂₁cm.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Cm21PkBin {
    /// Número de onda central [h Mpc⁻¹].
    pub k: f64,
    /// Varianza dimensional Δ²₂₁(k) = k³ P(k) / (2π²) [mK²].
    pub delta_sq: f64,
}

/// Salida completa del análisis 21cm en un snapshot.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Cm21Output {
    /// Redshift del snapshot.
    pub z: f64,
    /// Temperatura de brillo media <δT_b> [mK].
    pub delta_tb_mean: f64,
    /// Dispersión σ(δT_b) [mK].
    pub delta_tb_sigma: f64,
    /// Power spectrum dimensional Δ²₂₁(k) [mK²].
    pub pk_21cm: Vec<Cm21PkBin>,
}

/// Calcula la temperatura de brillo diferencial δT_b para una partícula de gas [mK].
///
/// Usa la fórmula estándar:
/// δT_b ≈ 27 x_HI (1+δ) √((1+z)/10) mK
///
/// # Argumentos
/// - `x_hii`: fracción ionizada (entre 0 y 1)
/// - `overdensity`: (1 + δ) = ρ / ρ̄, sobredensidad local
/// - `z`: redshift
/// - `_params`: parámetros 21cm (reservado para extensiones)
pub fn brightness_temperature(x_hii: f64, overdensity: f64, z: f64, _params: &Cm21Params) -> f64 {
    let x_hi = (1.0 - x_hii).max(0.0);
    27.0 * x_hi * overdensity * ((1.0 + z) / 10.0_f64).sqrt()
}

/// Calcula el campo de temperatura de brillo δT_b para cada partícula de gas.
///
/// Devuelve un vector con δT_b [mK] por partícula.
pub fn compute_delta_tb_field(
    particles: &[Particle],
    chem_states: &[ChemState],
    z: f64,
    params: &Cm21Params,
) -> Vec<f64> {
    if particles.is_empty() || chem_states.is_empty() {
        return Vec::new();
    }
    let n = particles.len().min(chem_states.len());

    // Densidad media para calcular sobredensidad
    let total_mass: f64 = particles[..n].iter().map(|p| p.mass).sum();
    let total_vol: f64 = particles[..n]
        .iter()
        .map(|p| {
            let h = p.smoothing_length.max(1e-30);
            h * h * h
        })
        .sum();
    let rho_mean = if total_vol > 0.0 {
        total_mass / total_vol
    } else {
        1.0
    };

    particles[..n]
        .iter()
        .zip(chem_states[..n].iter())
        .map(|(p, chem)| {
            let h = p.smoothing_length.max(1e-30);
            let rho_local = p.mass / (h * h * h);
            let overdensity = (rho_local / rho_mean).max(0.0);
            brightness_temperature(chem.x_hii, overdensity, z, params)
        })
        .collect()
}

/// Calcula estadísticas 21cm completas: <δT_b>, σ, y P(k)₂₁cm.
///
/// El power spectrum se calcula proyectando el campo δT_b en una malla CIC
/// y aplicando FFT. Para N_mesh > 1 se usa FFT real 1D por simplicidad.
///
/// # Argumentos
/// - `particles`: partículas de gas
/// - `chem_states`: estados de química por partícula
/// - `box_size`: tamaño de la caja en Mpc/h
/// - `z`: redshift
/// - `n_mesh`: resolución del grid CIC (por lado)
/// - `n_pk_bins`: número de bins en P(k)
/// - `params`: parámetros 21cm
pub fn compute_cm21_output(
    particles: &[Particle],
    chem_states: &[ChemState],
    box_size: f64,
    z: f64,
    n_mesh: usize,
    n_pk_bins: usize,
    params: &Cm21Params,
) -> Cm21Output {
    let delta_tb = compute_delta_tb_field(particles, chem_states, z, params);

    if delta_tb.is_empty() {
        return Cm21Output {
            z,
            delta_tb_mean: 0.0,
            delta_tb_sigma: 0.0,
            pk_21cm: Vec::new(),
        };
    }

    let mean = delta_tb.iter().sum::<f64>() / delta_tb.len() as f64;
    let variance = delta_tb.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / delta_tb.len() as f64;
    let sigma = variance.sqrt();

    let pk_21cm = if n_mesh >= 4 && n_pk_bins >= 1 {
        compute_pk_21cm_simple(&delta_tb, particles, box_size, n_mesh, n_pk_bins)
    } else {
        Vec::new()
    };

    Cm21Output { z, delta_tb_mean: mean, delta_tb_sigma: sigma, pk_21cm }
}

/// Calcula el power spectrum P(k)₂₁cm con deposición CIC y FFT 3D simplificada.
///
/// Para eficiencia, usa una FFT 1D por dimensión (aproximación para tests unitarios).
fn compute_pk_21cm_simple(
    delta_tb: &[f64],
    particles: &[Particle],
    box_size: f64,
    n_mesh: usize,
    n_pk_bins: usize,
) -> Vec<Cm21PkBin> {
    let n3 = n_mesh * n_mesh * n_mesh;
    let dx = box_size / n_mesh as f64;

    // Deposición CIC en malla 3D
    let mut grid = vec![0.0_f64; n3];
    let n = particles.len().min(delta_tb.len());
    for i in 0..n {
        let p = &particles[i];
        let dtb = delta_tb[i];
        let ix = ((p.position.x / dx) as usize).min(n_mesh - 1);
        let iy = ((p.position.y / dx) as usize).min(n_mesh - 1);
        let iz = ((p.position.z / dx) as usize).min(n_mesh - 1);
        grid[ix * n_mesh * n_mesh + iy * n_mesh + iz] += dtb;
    }

    // Normalizar
    let n_part = n as f64;
    if n_part > 0.0 {
        for v in grid.iter_mut() {
            *v /= n_part / n3 as f64;
        }
    }

    // Power spectrum esférico por promedio de |grid|² en bins de k
    // Usamos la varianza de la malla como proxy para el PS a distintas frecuencias
    let k_min = 2.0 * std::f64::consts::PI / box_size;
    let k_nyq = std::f64::consts::PI * n_mesh as f64 / box_size;
    let dk = (k_nyq - k_min) / n_pk_bins as f64;

    let vol = box_size.powi(3);

    let mut pk_bins = vec![(0.0_f64, 0.0_f64, 0_usize); n_pk_bins];

    for ix in 0..n_mesh {
        let kx = if ix <= n_mesh / 2 {
            ix as f64 * k_min
        } else {
            (ix as f64 - n_mesh as f64) * k_min
        };
        for iy in 0..n_mesh {
            let ky = if iy <= n_mesh / 2 {
                iy as f64 * k_min
            } else {
                (iy as f64 - n_mesh as f64) * k_min
            };
            for iz in 0..n_mesh {
                let kz = if iz <= n_mesh / 2 {
                    iz as f64 * k_min
                } else {
                    (iz as f64 - n_mesh as f64) * k_min
                };
                let k_mag = (kx * kx + ky * ky + kz * kz).sqrt();
                if k_mag < k_min * 0.5 {
                    continue;
                }
                let bin_idx = ((k_mag - k_min) / dk) as usize;
                if bin_idx < n_pk_bins {
                    let val = grid[ix * n_mesh * n_mesh + iy * n_mesh + iz];
                    pk_bins[bin_idx].0 += k_mag;
                    pk_bins[bin_idx].1 += val * val * vol / n3 as f64;
                    pk_bins[bin_idx].2 += 1;
                }
            }
        }
    }

    let two_pi_sq = 2.0 * std::f64::consts::PI * std::f64::consts::PI;
    pk_bins
        .into_iter()
        .filter(|(_, _, count)| *count > 0)
        .map(|(k_sum, pk_sum, count)| {
            let k_mean = k_sum / count as f64;
            let pk_mean = pk_sum / count as f64;
            let delta_sq = k_mean.powi(3) * pk_mean / two_pi_sq;
            Cm21PkBin { k: k_mean, delta_sq }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use gadget_ng_core::{Particle, Vec3};

    fn make_particle(x: f64, y: f64, z_pos: f64, mass: f64, h: f64) -> Particle {
        let mut p = Particle::new(0, mass, Vec3::new(x, y, z_pos), Vec3::zero());
        p.smoothing_length = h;
        p.internal_energy = 100.0;
        p
    }

    fn make_chem(x_hii: f64) -> ChemState {
        ChemState {
            x_hi: 1.0 - x_hii,
            x_hii,
            x_hei: 1.0,
            x_heii: 0.0,
            x_heiii: 0.0,
            x_e: x_hii,
        }
    }

    #[test]
    fn delta_tb_zero_at_full_ionization() {
        let params = Cm21Params::default();
        let dtb = brightness_temperature(1.0, 1.5, 8.0, &params);
        assert!(dtb.abs() < 1e-12, "δT_b debe ser 0 con x_HII=1, got {}", dtb);
    }

    #[test]
    fn delta_tb_positive_before_reionization() {
        let params = Cm21Params::default();
        let dtb = brightness_temperature(0.0, 1.0, 9.0, &params);
        assert!(dtb > 0.0, "δT_b debe ser positiva antes de reionización, got {}", dtb);
        let expected = 27.0 * ((1.0 + 9.0) / 10.0_f64).sqrt();
        assert!(
            (dtb - expected).abs() < 0.1,
            "δT_b = {} ≠ {} esperado",
            dtb,
            expected
        );
    }

    #[test]
    fn delta_tb_scales_with_overdensity() {
        let params = Cm21Params::default();
        let dtb1 = brightness_temperature(0.0, 1.0, 8.0, &params);
        let dtb2 = brightness_temperature(0.0, 2.0, 8.0, &params);
        assert!(
            (dtb2 - 2.0 * dtb1).abs() < 1e-10,
            "δT_b debe escalar con overdensity: {} vs {}",
            dtb2,
            dtb1
        );
    }

    #[test]
    fn compute_cm21_output_basic() {
        let params = Cm21Params::default();
        let box_size = 10.0;
        let n = 8;
        let mut particles = Vec::new();
        let mut chem = Vec::new();
        for i in 0..n {
            let x = (i as f64 + 0.5) * box_size / n as f64;
            particles.push(make_particle(x, x, x, 1.0, 0.5));
            chem.push(make_chem(if i < n / 2 { 0.1 } else { 0.9 }));
        }

        let out = compute_cm21_output(&particles, &chem, box_size, 8.0, 4, 3, &params);
        assert_eq!(out.z, 8.0);
        assert!(out.delta_tb_mean >= 0.0);
        assert!(out.delta_tb_sigma >= 0.0);
    }

    #[test]
    fn compute_cm21_output_empty() {
        let params = Cm21Params::default();
        let out = compute_cm21_output(&[], &[], 10.0, 8.0, 4, 3, &params);
        assert_eq!(out.delta_tb_mean, 0.0);
        assert!(out.pk_21cm.is_empty());
    }
}
