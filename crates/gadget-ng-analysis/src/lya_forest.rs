//! Bosque Lyman-α: absorción GPI y flujo transmitido (Phase 176).
//!
//! Calcula el flujo transmitido F = exp(-τ_GP) a lo largo de líneas de visión
//! (sightlines) a través de partículas de gas, usando la aproximación
//! Gunn-Peterson para la profundidad óptica:
//!
//! ```text
//! τ_GP = (π e² f_α λ_α) / (m_e c H(z)) × n_HI × (1 + δ)
//! ```
//!
//! donde `f_α = 0.4164` es la fuerza de oscilador Ly-α, `λ_α = 1215.67 Å`,
//! y `H(z)` es el parámetro de Hubble al redshift correspondiente.
//!
//! El módulo produce:
//! - Espectros transmitidos F(λ) por sightline
//! - P(k) del campo de flujo δ_F = F/⟨F⟩ - 1
//! - Estadísticas汇总: ⟨F⟩, σ(F), τ_ef

use gadget_ng_core::Particle;
use rustfft::{FftPlanner, num_complex::Complex};

/// Estado de ionización simplificado para Ly-α.
///
/// Si `x_hi = 1.0` el gas es completamente neutral (absorción máxima).
/// Si `x_hi = 0.0` el gas está completamente ionizado (transparente).
#[derive(Debug, Clone, Copy)]
pub struct LyaChemState {
    /// Fracción de hidrógeno neutro (0–1).
    pub x_hi: f64,
}

impl LyaChemState {
    /// Gas completamente neutro.
    pub fn neutral() -> Self {
        Self { x_hi: 1.0 }
    }
    /// Gas completamente ionizado.
    pub fn fully_ionized() -> Self {
        Self { x_hi: 0.0 }
    }
}

// ── Constantes físicas ──────────────────────────────────────────────────────

/// Fuerza de oscilador Ly-α.
#[allow(dead_code)]
const F_ALPHA: f64 = 0.4164;
/// Longitud de onda Ly-α en Å.
const LAMBDA_ALPHA_AA: f64 = 1215.67;
/// Factor de conversión GPI: π e² f_α λ_α / (m_e c).
/// En unidades prácticas: σ_α × c / √π ≈ 1.039 × 10⁻² (cgs).
/// Multiplicado por n_HI / H(z) da τ por unidad de longitud, integrado
/// sobre la línea de visión da τ adimensional.
///
/// Factor completo: (π e² f_α λ_α) / (m_e c √π) en cgs
/// = 1.04 × 10⁻² cm² Å / (g × cm/s) × (1 Å = 10⁻⁸ cm)
#[allow(dead_code)]
const GP_TAU_FACTOR: f64 = 5.88e-6;
/// Conversión de unidades internas a cgs para densidad:
/// 1 (10¹⁰ M☉/h) / (Mpc/h)³ en g/cm³ ≈ 6.77 × 10⁻²²
#[allow(dead_code)]
const DENSITY_UNIT_CGS: f64 = 6.77e-22;
/// Conversión de longitud: 1 Mpc/h → km (usado para H(z) · L)
#[allow(dead_code)]
const MPC_PER_H_TO_KM: f64 = 3.086e19;
/// Masa del protón [g].
#[allow(dead_code)]
const M_PROTON_CGS: f64 = 1.6726e-24;
/// Velocidad de la luz [km/s].
const C_LIGHT_KMS: f64 = 2.998e5;

// ── Tipos ────────────────────────────────────────────────────────────────────

/// Parámetros cosmológicos para el cálculo Ly-α.
#[derive(Debug, Clone, Copy)]
pub struct LyaCosmoParams {
    /// H₀ [km/s/Mpc].
    pub h0: f64,
    /// Ω_m (materia total).
    pub omega_m: f64,
    /// Ω_Λ (energía oscura).
    pub omega_lambda: f64,
}

impl Default for LyaCosmoParams {
    fn default() -> Self {
        Self {
            h0: 67.0,
            omega_m: 0.31,
            omega_lambda: 0.69,
        }
    }
}

/// Parámetros para el cálculo del bosque Ly-α.
#[derive(Debug, Clone)]
pub struct LyaParams {
    /// Número de sightlines (líneas de visión) en el mapa. Default: 256.
    pub n_sightlines: usize,
    /// Número de celdas de velocidad a lo largo de cada sightline. Default: 512.
    pub n_velocity_cells: usize,
    /// Redshift de la fuente de fondo (QSO). Default: 3.0.
    pub z_source: f64,
    /// Resolución de velocidad (km/s) por celda. Default: 25.0.
    pub dv_kms: f64,
    /// Temperatura del IGM para ensanchamiento térmico [K]. Default: 1e4.
    pub t_igm_kelvin: f64,
}

impl Default for LyaParams {
    fn default() -> Self {
        Self {
            n_sightlines: 256,
            n_velocity_cells: 512,
            z_source: 3.0,
            dv_kms: 25.0,
            t_igm_kelvin: 1e4,
        }
    }
}

/// Resultado del análisis Ly-α para una sightline.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LyaSightline {
    /// Flujo transmitido F(v) a lo largo de la sightline.
    pub flux: Vec<f64>,
    /// Profundidad óptica τ(v).
    pub tau: Vec<f64>,
    /// Velocidad de cada celda [km/s] (desde el observador).
    pub velocity: Vec<f64>,
    /// Redshift medio de absorción.
    pub z_mean: f64,
    /// Flujo medio ⟨F⟩.
    pub mean_flux: f64,
    /// Flujo RMS σ(F).
    pub sigma_flux: f64,
    /// Profundidad óptica efectiva τ_ef = -ln(⟨F⟩).
    pub tau_effective: f64,
}

/// Resultado agregado del bosque Ly-α.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LyaForestResult {
    /// Número de sightlines analizadas.
    pub n_sightlines: usize,
    /// P(k) del campo de flujo δ_F: P(k) [Mpc/h]³.
    pub pk_flux: Vec<LyaPkBin>,
    /// ⟨F⟩ promediado sobre todas las sightlines.
    pub mean_flux: f64,
    /// τ_ef = -ln(⟨F⟩) promediado.
    pub tau_effective: f64,
}

/// Bin en el power spectrum del campo de flujo Ly-α.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LyaPkBin {
    /// Número de onda [h/Mpc].
    pub k: f64,
    /// P(k) [(Mpc/h)³].
    pub pk: f64,
    /// Número de modos en el bin.
    pub n_modes: usize,
}

// ── Funciones auxiliares ─────────────────────────────────────────────────────

/// H(z) en km/s/Mpc usando cosmológica plana ΛCDM.
///
/// `H(z) = H₀ √(Ω_m (1+z)³ + Ω_Λ)`
fn hubble_z(z: f64, h0: f64, omega_m: f64, omega_lambda: f64) -> f64 {
    h0 * (omega_m * (1.0 + z).powi(3) + omega_lambda).sqrt()
}

/// Conversión redshift → velocidad peculiar [km/s] desde el QSO.
///
/// `v = c (z_source - z) / (1 + z_source)`
#[allow(dead_code)]
fn z_to_velocity(z: f64, z_source: f64) -> f64 {
    C_LIGHT_KMS * (z_source - z) / (1.0 + z_source)
}

/// Conversión velocidad → longitud de onda observada [Å].
///
/// `λ_obs = λ_α (1 + z)`
#[allow(dead_code)]
fn z_to_wavelength(z: f64) -> f64 {
    LAMBDA_ALPHA_AA * (1.0 + z)
}

/// Enanchamiento térmico-Doppler del Ly-α [km/s].
///
/// `b_thermal = √(2 k_B T / m_H) = 12.9 √(T / 10⁴ K) km/s`
fn thermal_b(t_kelvin: f64) -> f64 {
    12.9 * (t_kelvin / 1e4).sqrt()
}

// ── Cálculo de τ_GP por sightline ──────────────────────────────────────────

/// Calcula la profundidad óptica Gunn-Peterson a lo largo de una sightline.
///
/// Asigna las partículas de gas a celdas de posición (comoving distance)
/// a lo largo de la línea de visión, usando deposición SPH con
/// ensanchamiento térmico convertido a distancia comóvil.
///
/// # Argumentos
/// - `particles`: todas las partículas (solo Gas contribuye)
/// - `sightline_dir`: eje de la línea de visión ('x', 'y' o 'z')
/// - `impact_x`, `impact_y`: coordenadas transversales de la sightline
/// - `box_size`: tamaño de la caja en Mpc/h
/// - `params`: parámetros Ly-α
/// - `h0`, `omega_m`, `omega_lambda`: parámetros cosmológicos
/// - `chem_states`: estados de ionización (None → completamente neutral)
#[allow(clippy::too_many_arguments)]
pub fn compute_tau_along_sightline(
    particles: &[Particle],
    sightline_dir: char,
    impact_x: f64,
    impact_y: f64,
    box_size: f64,
    params: &LyaParams,
    cosmo: &LyaCosmoParams,
    chem_states: Option<&[LyaChemState]>,
) -> LyaSightline {
    let h0 = cosmo.h0;
    let omega_m = cosmo.omega_m;
    let omega_lambda = cosmo.omega_lambda;
    let n_cells = params.n_velocity_cells;
    let dx = box_size / n_cells as f64;
    let _b_thermal = thermal_b(params.t_igm_kelvin);

    let (axis_los, _ax1, _ax2) = match sightline_dir {
        'x' | 'X' => (0usize, 1, 2),
        'y' | 'Y' => (1, 0, 2),
        _ => (2, 0, 1),
    };

    // Calcular ρ_mean a partir de todas las partículas de gas
    let mut total_mass = 0.0f64;
    let mut total_vol_inv = 0.0f64;
    for p in particles {
        if !p.is_gas() {
            continue;
        }
        total_mass += p.mass;
        let h = p.smoothing_length.max(1e-30);
        total_vol_inv += 1.0 / (h * h * h);
    }
    let rho_mean = if total_vol_inv > 0.0 {
        total_mass * total_vol_inv / particles.len().max(1) as f64
    } else {
        1.0
    };

    let mut tau = vec![0.0f64; n_cells];
    let mut velocity = vec![0.0f64; n_cells];

    for (i, vel) in velocity.iter_mut().enumerate() {
        let pos_comoving = (i as f64 + 0.5) * dx;
        let z_cell = params.z_source * pos_comoving / box_size;
        let _hz = hubble_z(z_cell.max(0.0), h0, omega_m, omega_lambda).max(1e-30);
        *vel = C_LIGHT_KMS * z_cell / (1.0 + z_cell.max(1e-10));
    }

    // Prefactor GPI adimensional por celda:
    // τ_cell ≈ τ₀ × x_HI × (1+δ) × (Δl / l_0) × W_perp
    // donde τ₀ ≈ 5.2e-3 × (1+z)^{3/2} × (Ω_b h²/0.02) para IGM medio
    let omega_b_h2 = 0.022;
    let tau_prefactor = 5.2e-3;
    let _h2 = (h0 / 100.0) * (h0 / 100.0);
    let omega_b_factor = omega_b_h2 / 0.02;

    for (idx, p) in particles.iter().enumerate() {
        if !p.is_gas() || p.internal_energy <= 0.0 {
            continue;
        }

        let x_hi = if let Some(chem) = chem_states {
            if idx < chem.len() { chem[idx].x_hi } else { 1.0 }
        } else {
            1.0
        };

        // Posición de la partícula
        let (pos_los, pos_t1, pos_t2) = match axis_los {
            0 => (p.position.x, p.position.y, p.position.z),
            1 => (p.position.y, p.position.x, p.position.z),
            _ => (p.position.z, p.position.x, p.position.y),
        };

        // Distancia transversal al impacto
        let dx1 = pos_t1 - impact_x;
        let dx2 = pos_t2 - impact_y;
        let d_perp_sq = dx1 * dx1 + dx2 * dx2;
        let h = p.smoothing_length.max(1e-30);

        if d_perp_sq > (4.0 * h * h) {
            continue;
        }

        // Sobredensidad local
        let rho_local = p.mass / h.powi(3).max(1e-60);
        let overdensity = if rho_mean > 0.0 { rho_local / rho_mean } else { 1.0 };

        // Peso transversal SPH (kernel gaussiano 2D)
        let weight_perp = (-0.5 * d_perp_sq / (h * h)).exp();

        // Redshift de la partícula
        let z_p = (params.z_source * pos_los / box_size).max(0.0);
        let tau_z = tau_prefactor * (1.0 + z_p).powf(1.5) * omega_b_factor * x_hi * overdensity * weight_perp;

        // Velocidad peculiar de la partícula (reservado para extensión futura con perfiles Voigt)
        let _v_pec = match axis_los {
            0 => p.velocity.x,
            1 => p.velocity.y,
            _ => p.velocity.z,
        };

        // Convertir térmico y peculiar a celdas de posición (Δx = v/b × σ_x)
        // Spread τ en celdas adyacentes con kernel gaussiano térmico + peculiar
        let sigma_h = h;
        let cell_center = pos_los;
        let i_min = ((cell_center - 4.0 * sigma_h) / dx).floor() as isize;
        let i_max = ((cell_center + 4.0 * sigma_h) / dx).ceil() as isize;

        for ic in i_min..=i_max {
            if ic < 0 || ic >= n_cells as isize {
                continue;
            }
            let iu = ic as usize;
            let cell_pos = (iu as f64 + 0.5) * dx;
            let d_pos = cell_pos - cell_center;
            let gauss_weight = (-0.5 * (d_pos / sigma_h).powi(2)).exp();
            tau[iu] += tau_z * gauss_weight * dx / sigma_h;
        }
    }

    let flux: Vec<f64> = tau.iter().map(|&t| (-t).exp()).collect();
    let mean_flux = if flux.is_empty() { 0.0 } else { flux.iter().sum::<f64>() / flux.len() as f64 };
    let sigma_flux = if flux.is_empty() { 0.0 } else {
        let var = flux.iter().map(|f| (f - mean_flux).powi(2)).sum::<f64>() / flux.len().max(1) as f64;
        var.sqrt()
    };
    let tau_effective = if mean_flux > 0.0 { -mean_flux.ln() } else { f64::INFINITY };
    let z_mean = params.z_source * 0.5;

    LyaSightline {
        flux,
        tau,
        velocity,
        z_mean,
        mean_flux,
        sigma_flux,
        tau_effective,
    }
}

// ── P(k) del campo de flujo ────────────────────────────────────────────────

/// Calcula P(k) del campo de contraste de flujo δ_F a partir de múltiples sightlines.
///
/// `δ_F = F / ⟨F⟩ - 1`, y se calcula el power spectrum 1D:
/// `P_F(k) = ⟨|δ̃_F(k)|²⟩` prome diado sobre sightlines.
pub fn compute_lya_pk_1d(
    sightlines: &[LyaSightline],
    n_k_bins: usize,
) -> LyaForestResult {
    if sightlines.is_empty() {
        return LyaForestResult {
            n_sightlines: 0,
            pk_flux: Vec::new(),
            mean_flux: 0.0,
            tau_effective: 0.0,
        };
    }

    let n_cells = sightlines[0].flux.len();
    if n_cells == 0 || n_k_bins == 0 {
        return LyaForestResult {
            n_sightlines: sightlines.len(),
            pk_flux: Vec::new(),
            mean_flux: 0.0,
            tau_effective: 0.0,
        };
    }

    let global_mean: f64 = sightlines.iter().map(|s| s.mean_flux).sum::<f64>() / sightlines.len() as f64;
    let global_tau_eff = if global_mean > 0.0 { -global_mean.ln() } else { f64::INFINITY };

    let dv = if n_cells > 1 {
        sightlines[0].velocity[1] - sightlines[0].velocity[0]
    } else {
        25.0
    };
    let dv_abs = dv.abs().max(1.0);

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n_cells);

    let mut k_sum = vec![0.0f64; n_k_bins];
    let mut pk_sum = vec![0.0f64; n_k_bins];
    let mut count = vec![0usize; n_k_bins];

    let k_fund = 2.0 * std::f64::consts::PI / (n_cells as f64 * dv_abs);
    let k_max = std::f64::consts::PI / dv_abs;
    let dk = (k_max - k_fund) / n_k_bins as f64;

    for s in sightlines {
        if s.flux.len() != n_cells {
            continue;
        }
        // δ_F = F/⟨F⟩ - 1
        let mean_f = s.mean_flux.max(1e-30);
        let mut delta_f: Vec<Complex<f64>> = s
            .flux
            .iter()
            .map(|&f| Complex::new(f / mean_f - 1.0, 0.0))
            .collect();

        fft.process(&mut delta_f);

        let norm = 1.0 / (n_cells as f64);
        for (i, c) in delta_f.iter().enumerate() {
            let ki = if i <= n_cells / 2 { i as f64 } else { (i as isize - n_cells as isize) as f64 };
            let k = (ki * k_fund).abs();
            if k < k_fund * 0.5 || k > k_max {
                continue;
            }
            let bin = (((k - k_fund) / dk).floor() as usize).min(n_k_bins - 1);
            let power = (c.re * c.re + c.im * c.im) * norm;
            k_sum[bin] += k;
            pk_sum[bin] += power;
            count[bin] += 1;
        }
    }

    let pk_flux: Vec<LyaPkBin> = k_sum
        .iter()
        .zip(pk_sum.iter())
        .zip(count.iter())
        .filter(|((_, _), c)| **c > 0)
        .map(|((&ks, &ps), &c)| LyaPkBin {
            k: ks / c as f64,
            pk: ps / c as f64,
            n_modes: c,
        })
        .collect();

    LyaForestResult {
        n_sightlines: sightlines.len(),
        pk_flux,
        mean_flux: global_mean,
        tau_effective: global_tau_eff,
    }
}

// ── Generación de sightlines ───────────────────────────────────────────────

/// Genera posiciones de impacto para N sightlines en una grilla regular.
///
/// Las sightlines se distribuyen uniformemente en el plano transversal
/// al eje de visión.
pub fn generate_impact_positions(n_sightlines: usize, box_size: f64) -> Vec<(f64, f64)> {
    let n_side = ((n_sightlines as f64).sqrt().ceil() as usize).max(1);
    let spacing = box_size / n_side as f64;
    let mut positions = Vec::with_capacity(n_side * n_side);
    for i in 0..n_side {
        for j in 0..n_side {
            if positions.len() >= n_sightlines {
                break;
            }
            positions.push(((i as f64 + 0.5) * spacing, (j as f64 + 0.5) * spacing));
        }
    }
    positions.truncate(n_sightlines);
    positions
}

/// Analiza el bosque Ly-α completo: genera sightlines, calcula τ y F, P(k).
///
/// Pipeline completo in-situ:
/// 1. Genera `n_sightlines` líneas de visión en grilla regular
/// 2. Calcula τ_GP y F para cada sightline
/// 3. Calcula P_F(k) a partir de los espectros δ_F
pub fn analyze_lya_forest(
    particles: &[Particle],
    box_size: f64,
    params: &LyaParams,
    cosmo: &LyaCosmoParams,
    sightline_dir: char,
    chem_states: Option<&[LyaChemState]>,
) -> LyaForestResult {
    let impacts = generate_impact_positions(params.n_sightlines, box_size);
    let sightlines: Vec<LyaSightline> = impacts
        .iter()
        .map(|&(ix, iy)| {
            compute_tau_along_sightline(
                particles,
                sightline_dir,
                ix,
                iy,
                box_size,
                params,
                cosmo,
                chem_states,
            )
        })
        .collect();

    compute_lya_pk_1d(&sightlines, params.n_velocity_cells / 2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use gadget_ng_core::{Particle, Vec3};

    fn make_gas_particle(x: f64, y: f64, z: f64, mass: f64, u: f64, h: f64) -> Particle {
        Particle::new_gas(0, mass, Vec3::new(x, y, z), Vec3::zero(), u, h)
    }

    #[test]
    fn empty_particles_zero_flux() {
        let params = LyaParams::default();
        let cosmo = LyaCosmoParams::default();
        let result = compute_tau_along_sightline(
            &[],
            'z',
            50.0,
            50.0,
            100.0,
            &params,
            &cosmo,
            None,
        );
        assert!(result.flux.iter().all(|&f| (f - 1.0).abs() < 1e-10),
            "Empty gas should have F=1 (τ=0)");
    }

#[test]
    fn neutral_gas_produces_absorption() {
        let params = LyaParams {
            n_sightlines: 4,
            n_velocity_cells: 64,
            z_source: 3.0,
            dv_kms: 25.0,
            t_igm_kelvin: 1e4,
        };
        let cosmo = LyaCosmoParams::default();
        let h = 5.0;
        let p = make_gas_particle(50.0, 50.0, 50.0, 1e10, 1e4, h);
        let result = compute_tau_along_sightline(
            &[p],
            'z',
            50.0,
            50.0,
            100.0,
            &params,
            &cosmo,
            None,
        );
        let min_flux = result.flux.iter().cloned().fold(1.0f64, f64::min);
        assert!(min_flux < 1.0, "Neutral gas should absorb: min F = {min_flux}");
    }

    #[test]
    fn ionized_gas_transparent() {
        let params = LyaParams {
            n_sightlines: 4,
            n_velocity_cells: 64,
            z_source: 3.0,
            dv_kms: 25.0,
            t_igm_kelvin: 1e4,
        };
        let cosmo = LyaCosmoParams::default();
        let h = 5.0;
        let p = make_gas_particle(50.0, 50.0, 50.0, 1e10, 1e4, h);
        let chem = LyaChemState::fully_ionized();
        let result = compute_tau_along_sightline(
            &[p],
            'z',
            50.0,
            50.0,
            100.0,
            &params,
            &cosmo,
            Some(&[chem]),
        );
        let mean_flux = result.mean_flux;
        assert!(mean_flux > 0.99, "Fully ionized should be transparent: <F> = {mean_flux}");
    }

    #[test]
    fn generate_impact_positions_count() {
        let positions = generate_impact_positions(9, 100.0);
        assert_eq!(positions.len(), 9);
        for &(x, y) in &positions {
            assert!(x > 0.0 && x < 100.0, "Impact x out of range: {x}");
            assert!(y > 0.0 && y < 100.0, "Impact y out of range: {y}");
        }
    }

    #[test]
    fn pk_1d_empty_gives_empty() {
        let result = compute_lya_pk_1d(&[], 10);
        assert_eq!(result.n_sightlines, 0);
        assert!(result.pk_flux.is_empty());
    }

    #[test]
    fn hubble_z_increases_with_redshift() {
        let h0 = hubble_z(0.0, 67.0, 0.31, 0.69);
        let h1 = hubble_z(3.0, 67.0, 0.31, 0.69);
        assert!(h1 > h0, "H(z) should increase: H(0)={h0}, H(3)={h1}");
    }
}