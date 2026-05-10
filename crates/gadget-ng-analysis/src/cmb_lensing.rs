//! CMB secondary lensing: convergence and shear maps (Phase TBD).
//!
//! Projects N-body particle positions onto a 2D angular grid at the
//! CMB source plane (z ~ 1100), computing convergence κ and shear
//! (γ₁, γ₂) using Cloud-in-Cell (CIC) deposition with the CMB lensing
//! kernel W(χ).
//!
//! ## Physical model
//!
//! The CMB convergence κ(θ) is the projected overdensity weighted by
//! the lensing kernel:
//!
//! ```text
//! κ(θ) ∝ Σ_p [m_p × W(χ_p)] / χ_p²
//! ```
//!
//! where `W(χ) = χ(χ_s − χ)/χ_s` is the dimensionless lensing efficiency
//! kernel, `χ_s` is the comoving distance to the CMB last-scattering surface,
//! and the sum runs over all particles projected onto the angular pixel.
//!
//! The full physical convergence including cosmological prefactors is:
//!
//! ```text
//! κ = (3Ω_m H₀²) / (2c²) × ∫ dχ (1+z) W(χ) δ(θ,χ) / a(χ)
//! ```
//!
//! The code stores κ proportional to the projected surface mass density;
//! multiply by `(3Ω_m H₀²)/(2c²ρ̄)` for physical units.
//!
//! ## CIC deposition
//!
//! Each particle is deposited onto 4 neighboring angular pixels with
//! bilinear (Cloud-in-Cell) weights. The angular position is computed
//! from the flat-sky approximation: `θ_x = (x − x_obs)/χ`, `θ_y = (y − y_obs)/χ`.
//!
//! ## Angular power spectrum
//!
//! `compute_cmb_angular_cl` computes `C_ℓ^{κκ}` via 2D FFT of the
//! convergence field, binned in multipole ℓ, following the same method
//! as `convergence_angular_cl` in `lightcone.rs`.

use gadget_ng_core::Particle;
use rustfft::{FftPlanner, num_complex::Complex};

/// c / (H₀ × 100 km/s/Mpc) in Mpc/h.
///
/// Comoving distance: `χ(z) = C_H0_INV × ∫₀^z dz'/E(z')` with
/// E(z) = √(Ω_m(1+z)³ + Ω_Λ) for Planck-like defaults.
const C_H0_INV_MPC_H: f64 = 2997.9;

/// Default cosmological parameters (Planck 2018).
const OMEGA_M_DEFAULT: f64 = 0.315;
const OMEGA_LAMBDA_DEFAULT: f64 = 0.685;

/// Parámetros para la proyección de lensing CMB.
///
/// El observador se sitúa en la cara z = 0 de la caja, con eje de
/// visión a lo largo de +z. El campo de visión `fov_rad` define la
/// extensión angular del mapa en radianes (lado completo, no half-width).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CmbLensingParams {
    /// Número de píxeles por lado (n_pixels × n_pixels).
    pub n_pixels: usize,
    /// Redshift del plano fuente CMB (típicamente z ~ 1100).
    pub z_source: f64,
    /// Campo de visión en radianes (lado).
    pub fov_rad: f64,
}

impl Default for CmbLensingParams {
    fn default() -> Self {
        Self {
            n_pixels: 256,
            z_source: 1100.0,
            fov_rad: 0.1,
        }
    }
}

/// Mapa de convergencia y shear CMB (κ, γ₁, γ₂).
///
/// Cada campo es un array aplanado fila-major de tamaño `n_pixels²`.
/// La convergencia κ es proporcional a la densidad superficial de masa
/// pesada por el kernel de lensing W(χ). Para obtener κ adimensional,
/// multiplicar por `(3Ω_m H₀²)/(2c² ρ̄)`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CmbLensingMap {
    /// Número de píxeles por lado.
    pub n_pixels: usize,
    /// Convergencia κ (n_pixels², fila-major).
    pub kappa: Vec<f64>,
    /// Shear γ₁ (n_pixels², fila-major).
    pub gamma1: Vec<f64>,
    /// Shear γ₂ (n_pixels², fila-major).
    pub gamma2: Vec<f64>,
    /// Convergencia media del mapa.
    pub mean_kappa: f64,
}

/// Bin del espectro angular de potencia C_ℓ^{κκ}.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CmbClBin {
    /// Multipolo ℓ central del bin.
    pub ell: f64,
    /// Potencia C_ℓ^{κκ}.
    pub cl_kk: f64,
}

/// Distancia comóvil al redshift z en universo ΛCDM plano.
///
/// Usa integración del rectángulo medio con 512 pasos y parámetros
/// Planck-like (Ω_m = 0.315, Ω_Λ = 0.685). Retorna χ en Mpc/h.
///
/// ```text
/// χ(z) = (c/H₀) × ∫₀^z dz'/E(z') × h
/// ```
///
/// donde `E(z) = √(Ω_m(1+z)³ + Ω_Λ)` y `(c/H₀)h ≈ 2997.9 Mpc/h`.
fn comoving_distance_to_z(z: f64) -> f64 {
    if z <= 0.0 {
        return 0.0;
    }
    const N_STEPS: usize = 512;
    let dz = z / N_STEPS as f64;
    let mut chi = 0.0;
    for i in 0..N_STEPS {
        let zi = (i as f64 + 0.5) * dz;
        let ez = (OMEGA_M_DEFAULT * (1.0 + zi).powi(3) + OMEGA_LAMBDA_DEFAULT).sqrt();
        chi += dz / ez;
    }
    C_H0_INV_MPC_H * chi
}

/// Kernel de eficiencia de lensing CMB: W(χ) = χ(χ_s − χ)/χ_s.
///
/// Es adimensional y vale 0 en χ = 0 y χ = χ_s, con un máximo
/// en χ = χ_s/2 donde W_max = χ_s/4.
#[inline]
fn lensing_kernel_w(chi: f64, chi_s: f64) -> f64 {
    if chi <= 0.0 || chi_s <= 0.0 || chi >= chi_s {
        0.0
    } else {
        chi * (chi_s - chi) / chi_s
    }
}

/// Proyecta partículas N-body en un mapa angular de convergencia CMB.
///
/// Usa deposición CIC (Cloud-in-Cell) en una malla `n_pixels × n_pixels`
/// con kernel de lensing `W(χ) = χ(χ_s − χ)/χ_s`. La posición angular
/// de cada partícula se calcula en aproximación de cielo plano:
///
/// ```text
/// θ_x = (x − box_size/2) / χ
/// θ_y = (y − box_size/2) / χ
/// ```
///
/// donde `χ = z` es la distancia comóvil a lo largo de la línea de visión
/// (eje +z). Partículas con `χ < χ_min` o `χ ≥ χ_s` se descartan.
///
/// El peso de convergencia por partícula es:
///
/// ```text
/// w_κ = m_p × W(χ) / (χ² × pixel_area)
/// ```
///
/// y los pesos de shear se modulan por cosenos direccionales transversales.
///
/// # Parámetros
///
/// - `particles`: slice de partículas (todos los tipos contribuyen a κ).
/// - `box_size`: tamaño de la caja en unidades internas (Mpc/h).
/// - `params`: parámetros de proyección (n_pixels, z_source, fov_rad).
///
/// # Retorna
///
/// `CmbLensingMap` con campos κ, γ₁, γ₂ y `mean_kappa`.
pub fn compute_cmb_convergence(
    particles: &[Particle],
    box_size: f64,
    params: &CmbLensingParams,
) -> CmbLensingMap {
    let n = params.n_pixels;
    let nn = n * n;
    let chi_s = comoving_distance_to_z(params.z_source);
    let pixel_ang = params.fov_rad / n as f64;
    let pixel_area = pixel_ang * pixel_ang;

    let obs_x = box_size / 2.0;
    let obs_y = box_size / 2.0;
    let min_chi = box_size * 1e-6;

    let mut kappa = vec![0.0f64; nn];
    let mut gamma1 = vec![0.0f64; nn];
    let mut gamma2 = vec![0.0f64; nn];

    for p in particles {
        let chi = p.position.z;
        if chi < min_chi || chi >= chi_s {
            continue;
        }

        let w = lensing_kernel_w(chi, chi_s);
        if w <= 0.0 {
            continue;
        }

        let theta_x = (p.position.x - obs_x) / chi;
        let theta_y = (p.position.y - obs_y) / chi;

        let fx = (theta_x / params.fov_rad + 0.5) * n as f64;
        let fy = (theta_y / params.fov_rad + 0.5) * n as f64;

        let ix = fx.floor() as isize;
        let iy = fy.floor() as isize;
        let dx = fx - ix as f64;
        let dy = fy - iy as f64;

        let weight = p.mass * w / (chi * chi * pixel_area);

        let r_perp_sq = theta_x * theta_x + theta_y * theta_y;
        if r_perp_sq < 1e-30 {
            for &(di, dj) in &[(0isize, 0isize), (1, 0), (0, 1), (1, 1)] {
                let ci = ix + di;
                let cj = iy + dj;
                if ci < 0 || ci >= n as isize || cj < 0 || cj >= n as isize {
                    continue;
                }
                let wx = if di == 0 { 1.0 - dx } else { dx };
                let wy = if dj == 0 { 1.0 - dy } else { dy };
                let idx = (cj as usize) * n + (ci as usize);
                kappa[idx] += weight * wx * wy;
            }
            continue;
        }

        let r_perp = r_perp_sq.sqrt();
        let n_x = theta_x / r_perp;
        let n_y = theta_y / r_perp;
        let shear_cos1 = n_x * n_x - n_y * n_y;
        let shear_cos2 = 2.0 * n_x * n_y;

        for &(di, dj) in &[(0isize, 0isize), (1, 0), (0, 1), (1, 1)] {
            let ci = ix + di;
            let cj = iy + dj;
            if ci < 0 || ci >= n as isize || cj < 0 || cj >= n as isize {
                continue;
            }
            let wx = if di == 0 { 1.0 - dx } else { dx };
            let wy = if dj == 0 { 1.0 - dy } else { dy };
            let w_cic = wx * wy;
            let idx = (cj as usize) * n + (ci as usize);
            kappa[idx] += weight * w_cic;
            gamma1[idx] += weight * w_cic * shear_cos1;
            gamma2[idx] += weight * w_cic * shear_cos2;
        }
    }

    let total: f64 = kappa.iter().sum();
    let mean_kappa = if nn > 0 { total / nn as f64 } else { 0.0 };

    CmbLensingMap {
        n_pixels: n,
        kappa,
        gamma1,
        gamma2,
        mean_kappa,
    }
}

/// Calcula el espectro angular de potencia C_ℓ^{κκ} del mapa de convergencia.
///
/// Usa FFT 2D sobre la malla angular de κ (restando la media primero)
/// y promedia en anillos de ℓ, análogo a `convergence_angular_cl` en
/// `lightcone.rs`.
///
/// # Parámetros
///
/// - `map`: mapa de convergencia CMB con campo κ.
/// - `params`: parámetros de proyección (se usa `fov_rad`).
/// - `n_ell_bins`: número de bins logarítmicos en ℓ.
///
/// # Retorna
///
/// Vec de `CmbClBin` con ℓ central y potencia C_ℓ^{κκ} de cada bin.
pub fn compute_cmb_angular_cl(
    map: &CmbLensingMap,
    params: &CmbLensingParams,
    n_ell_bins: usize,
) -> Vec<CmbClBin> {
    let n = map.n_pixels;
    let nn = n * n;
    let mean = map.mean_kappa;

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_inverse(n);

    let mut kappa_freq: Vec<Complex<f64>> = (0..nn)
        .map(|i| Complex::new(map.kappa[i] - mean, 0.0))
        .collect();

    for row in 0..n {
        let mut row_k: Vec<Complex<f64>> = (0..n).map(|col| kappa_freq[row * n + col]).collect();
        fft.process(&mut row_k);
        for col in 0..n {
            kappa_freq[row * n + col] = row_k[col];
        }
    }
    for col in 0..n {
        let mut col_k: Vec<Complex<f64>> = (0..n).map(|row| kappa_freq[row * n + col]).collect();
        fft.process(&mut col_k);
        for row in 0..n {
            kappa_freq[row * n + col] = col_k[row];
        }
    }

    let pixel_area = (params.fov_rad / n as f64).powi(2);
    let dl = 2.0 * std::f64::consts::PI / params.fov_rad;
    let omega_pixel = params.fov_rad * params.fov_rad / nn as f64;

    let ell_max = (n as f64 / 2.0) * dl;
    let d_ell = ell_max / n_ell_bins as f64;
    let mut cl_sum = vec![0.0f64; n_ell_bins];
    let mut cl_count = vec![0usize; n_ell_bins];

    for row in 0..n {
        let l2 = if row <= n / 2 {
            row as f64
        } else {
            (row as f64) - n as f64
        };
        for col in 0..n {
            let l1 = if col <= n / 2 {
                col as f64
            } else {
                (col as f64) - n as f64
            };
            let ell = (l1 * l1 + l2 * l2).sqrt() * dl;
            let power = kappa_freq[row * n + col].norm_sqr() * omega_pixel * omega_pixel;
            let bin = ((ell / d_ell) as usize).min(n_ell_bins - 1);
            cl_sum[bin] += power;
            cl_count[bin] += 1;
        }
    }

    let norm = pixel_area / nn as f64;
    cl_sum
        .iter()
        .zip(cl_count.iter())
        .enumerate()
        .filter(|(_, (_, c))| **c > 0)
        .map(|(i, (&s, &c))| CmbClBin {
            ell: (i as f64 + 0.5) * d_ell,
            cl_kk: s / c as f64 * norm,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use gadget_ng_core::{Particle, Vec3};

    fn make_dm_particle(x: f64, y: f64, z: f64, mass: f64, id: usize) -> Particle {
        let mut p = Particle::new(id, mass, Vec3::new(x, y, z), Vec3::zero());
        p.smoothing_length = 1.0;
        p
    }

    #[test]
    fn empty_particles_zero_convergence() {
        let particles: Vec<Particle> = Vec::new();
        let params = CmbLensingParams {
            n_pixels: 32,
            z_source: 1100.0,
            fov_rad: 0.1,
        };
        let result = compute_cmb_convergence(&particles, 100.0, &params);
        assert!(
            result.kappa.iter().all(|&k| k == 0.0),
            "Empty particles should give zero convergence"
        );
        assert_eq!(result.mean_kappa, 0.0);
        assert!(result.gamma1.iter().all(|&g| g == 0.0));
        assert!(result.gamma2.iter().all(|&g| g == 0.0));
    }

    #[test]
    fn cmb_map_dimensions() {
        let params = CmbLensingParams {
            n_pixels: 64,
            z_source: 1100.0,
            fov_rad: 0.1,
        };
        let p = make_dm_particle(50.0, 50.0, 50.0, 1.0, 0);
        let result = compute_cmb_convergence(&[p], 100.0, &params);
        assert_eq!(result.kappa.len(), 64 * 64, "kappa should be n_pixels²");
        assert_eq!(result.gamma1.len(), 64 * 64, "gamma1 should be n_pixels²");
        assert_eq!(result.gamma2.len(), 64 * 64, "gamma2 should be n_pixels²");
        assert_eq!(result.n_pixels, 64);
    }

    #[test]
    fn cl_bins_non_empty() {
        let params = CmbLensingParams {
            n_pixels: 32,
            z_source: 1100.0,
            fov_rad: 0.1,
        };
        let mut map = CmbLensingMap {
            n_pixels: 32,
            kappa: vec![0.0; 32 * 32],
            gamma1: vec![0.0; 32 * 32],
            gamma2: vec![0.0; 32 * 32],
            mean_kappa: 0.0,
        };
        map.kappa[16 * 32 + 16] = 1.0;
        let cl = compute_cmb_angular_cl(&map, &params, 4);
        assert!(!cl.is_empty(), "C_ell bins should not be empty");
        assert!(
            cl.iter().all(|b| b.ell > 0.0),
            "All ell values should be positive"
        );
    }

    #[test]
    fn kappa_positive_for_particles() {
        let params = CmbLensingParams {
            n_pixels: 64,
            z_source: 1100.0,
            fov_rad: 0.1,
        };
        let box_size = 100.0;
        let p1 = make_dm_particle(box_size / 2.0, box_size / 2.0, 50.0, 1.0, 0);
        let p2 = make_dm_particle(box_size / 2.0 + 1.0, box_size / 2.0, 50.0, 2.0, 1);
        let result = compute_cmb_convergence(&[p1, p2], box_size, &params);
        let total_kappa: f64 = result.kappa.iter().sum();
        assert!(
            total_kappa > 0.0,
            "Total kappa should be positive for particles with positive mass: got {total_kappa}"
        );
    }
}
