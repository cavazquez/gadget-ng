//! Emisión de rayos X térmica en cúmulos de galaxias — Phase 151.
//!
//! ## Modelo físico
//!
//! Bremsstrahlung térmico (free-free): la emissividad volumétrica es
//!
//! ```text
//! Λ_X(T) = 3×10⁻²⁷ × √T × n_e × n_i   [erg/s/cm³]
//! ```
//!
//! donde `T` es la temperatura en Kelvin, y `n_e ≈ n_i ≈ ρ/(2 m_p)` para
//! plasma totalmente ionizado de hidrógeno puro (Sarazin 1988).
//!
//! La temperatura espectroscópica ponderada por emissividad sigue
//! Mazzotta et al. (2004):
//!
//! ```text
//! T_sl = ∫ n_e² T^{-0.75} dV / ∫ n_e² T^{-1.75} dV
//! ```
//!
//! ## Referencias
//!
//! - Sarazin (1988) "X-ray Emission from Clusters of Galaxies".
//! - Mazzotta et al. (2004) MNRAS 354, 10.

use gadget_ng_core::Particle;

/// k_B / (m_H · μ) en (km/s)² / K — misma constante que en gadget-ng-sph/cooling.rs.
const KB_OVER_MH_MU: f64 = 8.254e-3 / 0.6;

/// Convierte energía interna específica `u` [(km/s)²] a temperatura [K].
#[inline]
fn u_to_temperature(u: f64, gamma: f64) -> f64 {
    u * (gamma - 1.0) / KB_OVER_MH_MU
}

/// Constante de bremsstrahlung: Λ_0 = 3×10⁻²⁷ erg·s⁻¹·cm³·K^{-1/2}.
const LAMBDA_0: f64 = 3.0e-27;

/// Temperatura de corte mínima [K]: gas frío no emite en X.
const T_X_MIN_K: f64 = 1.0e5;

/// Resultado de un bin radial de emisión de rayos X.
#[derive(Debug, Clone, PartialEq)]
pub struct XrayBin {
    /// Radio central del bin en unidades internas.
    pub r_center: f64,
    /// Luminosidad de rayos X en el bin [unidades internas].
    pub luminosity_x: f64,
    /// Temperatura promedio ponderada por emissividad [K].
    pub temperature_x: f64,
    /// Número de partículas de gas caliente en el bin.
    pub n_particles: usize,
}

/// Calcula la emissividad de bremsstrahlung para una partícula de gas (Phase 151).
///
/// # Parámetros
///
/// - `p`: partícula de gas con `internal_energy` y `smoothing_length`
/// - `gamma`: índice adiabático (típicamente 5/3)
///
/// # Retorna
///
/// Emissividad en unidades internas (proporcional a `ρ² √T`).
pub fn bremsstrahlung_emissivity(p: &Particle, gamma: f64) -> f64 {
    if !p.is_gas() { return 0.0; }
    let rho = if p.smoothing_length > 0.0 { p.mass / p.smoothing_length.powi(3) } else { 0.0 };
    if rho <= 0.0 { return 0.0; }
    let t_k = u_to_temperature(p.internal_energy, gamma);
    if t_k < T_X_MIN_K { return 0.0; }
    LAMBDA_0 * rho * rho * t_k.sqrt()
}

/// Luminosidad de rayos X total de un conjunto de partículas (Phase 151).
///
/// Suma las contribuciones de bremsstrahlung sobre todo el gas caliente.
///
/// # Parámetros
///
/// - `particles`: slice de todas las partículas
/// - `gamma`: índice adiabático
///
/// # Retorna
///
/// Luminosidad X total en unidades internas.
pub fn total_xray_luminosity(particles: &[Particle], gamma: f64) -> f64 {
    particles.iter().map(|p| bremsstrahlung_emissivity(p, gamma) * p.mass).sum()
}

/// Temperatura espectroscópica ponderada por emissividad (Mazzotta+2004) (Phase 151).
///
/// ```text
/// T_sl = Σ w_i T_i   con   w_i = n_e²_i T_i^{-0.75} / Σ n_e²_j T_j^{-0.75}
/// ```
///
/// # Parámetros
///
/// - `particles`: slice de partículas
/// - `gamma`: índice adiabático
///
/// # Retorna
///
/// T_sl en Kelvin; 0.0 si no hay gas caliente.
pub fn spectroscopic_temperature(particles: &[Particle], gamma: f64) -> f64 {
    let mut num = 0.0_f64;
    let mut den = 0.0_f64;
    for p in particles {
        if !p.is_gas() { continue; }
        let rho = if p.smoothing_length > 0.0 { p.mass / p.smoothing_length.powi(3) } else { 0.0 };
        if rho <= 0.0 { continue; }
        let t_k = u_to_temperature(p.internal_energy, gamma);
        if t_k < T_X_MIN_K { continue; }
        let w = rho * rho * t_k.powf(-0.75);
        num += w * t_k;
        den += w;
    }
    if den > 0.0 { num / den } else { 0.0 }
}

/// Temperatura media ponderada por masa (Phase 151).
pub fn mass_weighted_temperature(particles: &[Particle], gamma: f64) -> f64 {
    let mut num = 0.0_f64;
    let mut den = 0.0_f64;
    for p in particles {
        if !p.is_gas() { continue; }
        let t_k = u_to_temperature(p.internal_energy, gamma);
        if t_k < T_X_MIN_K { continue; }
        num += p.mass * t_k;
        den += p.mass;
    }
    if den > 0.0 { num / den } else { 0.0 }
}

/// Calcula el perfil radial de emisión X alrededor de un centro (Phase 151).
///
/// Las partículas se asignan a bins logarítmicos de radio.
///
/// # Parámetros
///
/// - `particles`: slice de partículas
/// - `center`: posición del centro [x, y, z]
/// - `r_edges`: bordes de los bins radiales (debe ser monótono creciente)
/// - `gamma`: índice adiabático
///
/// # Retorna
///
/// `Vec<XrayBin>` con un bin por intervalo en `r_edges`.
pub fn compute_xray_profile(
    particles: &[Particle],
    center: [f64; 3],
    r_edges: &[f64],
    gamma: f64,
) -> Vec<XrayBin> {
    let n_bins = r_edges.len().saturating_sub(1);
    let mut lx = vec![0.0_f64; n_bins];
    let mut tx_num = vec![0.0_f64; n_bins];
    let mut tx_den = vec![0.0_f64; n_bins];
    let mut counts = vec![0_usize; n_bins];

    for p in particles {
        if !p.is_gas() { continue; }
        let dx = p.position.x - center[0];
        let dy = p.position.y - center[1];
        let dz = p.position.z - center[2];
        let r = (dx * dx + dy * dy + dz * dz).sqrt();

        let bin = r_edges.partition_point(|&e| e <= r).saturating_sub(1);
        if bin >= n_bins { continue; }

        let rho = if p.smoothing_length > 0.0 { p.mass / p.smoothing_length.powi(3) } else { 0.0 };
        if rho <= 0.0 { continue; }
        let t_k = u_to_temperature(p.internal_energy, gamma);
        if t_k < T_X_MIN_K { continue; }

        let emiss = LAMBDA_0 * rho * rho * t_k.sqrt() * p.mass;
        let w = rho * rho * t_k.powf(-0.75);
        lx[bin] += emiss;
        tx_num[bin] += w * t_k;
        tx_den[bin] += w;
        counts[bin] += 1;
    }

    (0..n_bins)
        .map(|i| {
            let r_center = 0.5 * (r_edges[i] + r_edges[i + 1]);
            let temperature_x = if tx_den[i] > 0.0 { tx_num[i] / tx_den[i] } else { 0.0 };
            XrayBin {
                r_center,
                luminosity_x: lx[i],
                temperature_x,
                n_particles: counts[i],
            }
        })
        .collect()
}
