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

use gadget_ng_core::Particle;

use crate::chemistry::ChemState;

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

/// Calcula el perfil de temperatura del IGM para un conjunto de partículas de gas.
///
/// Filtra las partículas del IGM (baja densidad) y computa estadísticas de temperatura.
///
/// # Argumentos
/// - `particles`    — partículas de gas SPH
/// - `chem_states`  — estados de ionización por partícula (misma longitud que `particles`)
/// - `mean_density` — densidad media del universo en unidades internas (para calcular δ)
/// - `z`            — redshift actual
/// - `params`       — parámetros del cálculo
///
/// # Retorna
/// `IgmTempBin` con las estadísticas de temperatura del IGM.
pub fn compute_igm_temp_profile(
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

    let n = particles.len().min(chem_states.len());
    // Filtrar partículas del IGM y calcular temperaturas
    // Proxy de densidad: ρ ≈ mass / h_sml³ (densidad SPH estimada)
    let mut temperatures: Vec<f64> = Vec::new();
    for i in 0..n {
        let p = &particles[i];
        // Solo partículas de gas (internal_energy > 0)
        if p.internal_energy <= 0.0 {
            continue;
        }
        // Filtro de densidad vía smoothing_length (mayor h → menor densidad)
        if mean_density > 0.0 && params.delta_max > 0.0 && p.smoothing_length > 0.0 {
            let rho_sph = p.mass / (p.smoothing_length * p.smoothing_length * p.smoothing_length);
            let delta_threshold = params.delta_max * mean_density;
            if rho_sph > delta_threshold {
                continue; // partícula en halo denso
            }
        }
        let t = temperature_from_particle(p.internal_energy, &chem_states[i], params.gamma);
        if t > 0.0 && t.is_finite() {
            temperatures.push(t);
        }
    }

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

    // Percentiles (sortear)
    temperatures.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let t_median = percentile(&temperatures, 0.50);
    let t_p16 = percentile(&temperatures, 0.16);
    let t_p84 = percentile(&temperatures, 0.84);

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

    /// Crea partícula de gas con `smoothing_length` tal que ρ_SPH ≈ target_density.
    /// ρ ≈ m / h³ → h = (m / ρ)^(1/3)
    fn make_gas_particle(internal_energy: f64, target_density: f64) -> Particle {
        let mass = 1e-6f64;
        let h = if target_density > 0.0 {
            (mass / target_density).cbrt()
        } else {
            0.1 // default
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

    fn neutral_chem() -> ChemState {
        ChemState::neutral()
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
        // Partícula con energía interna equivalente a ~10⁴ K en unidades internas (km²/s²)
        // T ~ 10⁴ K → u ~ k_B T / ((γ-1) μ m_p) / U_CODE_TO_ERG_G
        // U_CODE_TO_ERG_G = 1e10 (gadget-ng units: km²/s² → erg/g)
        // u_code ≈ 2.1e13 erg/g / 1e10 = 2100 km²/s²
        let chem = warm_ionized_chem();
        let u_code = 2100.0_f64; // ~10⁴ K en unidades internas
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

        // u_code en unidades internas (km²/s²)
        let u_code = 2100.0_f64;
        // Partícula IGM: ρ_SPH ≈ mean_density * 5 < delta_max * mean_density → incluida
        let igm_particle = make_gas_particle(u_code, mean_density * 5.0);
        // Partícula en halo: ρ_SPH ≈ mean_density * 100 > delta_max * mean_density → excluida
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
        // Partículas con energía interna uniforme en unidades internas
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
            make_gas_particle(u_code, 1e-2), // alta densidad
            make_gas_particle(u_code, 1e-6), // baja densidad
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
}
