//! Efecto Sunyaev-Zel'dovich: mapa Compton-y y kinetic SZ (Phase 174).
//!
//! Calcula mapas 2D de efecto térmico SZ (Compton-y) y cinético SZ (kSZ)
//! a partir de partículas de gas proyectadas a lo largo de la línea de visión.
//!
//! ## Modelo físico
//!
//! **tSZ (térmico):** El parámetro Compton-y integraliza la presión electrónica:
//!
//! ```text
//! y = (σ_T / (m_e c²)) ∫ P_e dl
//! ```
//!
//! Para gas totalmente ionizado con abundancia cosmológica (X_H = 0.76):
//!
//! ```text
//! P_e = n_e k_B T_e = (X_e ρ / μ_e m_p) k_B T_e
//! ```
//!
//! donde `μ_e = 2/(1+X_H) ≈ 1.143` y `X_e ≈ 1.16` (ionización completa H+He).
//!
//! **kSZ (cinético):** El término de temperatura modulado por velocidad peculiar:
//!
//! ```text
//! ΔT/T_CMB = -σ_T ∫ n_e (v·n̂ / c) dl
//! ```
//!
//! ## Unidades internas
//!
//! El código opera en unidades internas del simulador. Las constantes físicas
//! (`σ_T`, `m_e c²`, `k_B`) se combinan en un factor de conversión para
//! producir `y` adimensional y `ΔT/T_CMB` adimensional.

use gadget_ng_core::Particle;

/// Parámetros para la proyección SZ.
#[derive(Debug, Clone)]
pub struct SzParams {
    /// Número de píxeles por lado del mapa (n_pixels × n_pixels).
    pub n_pixels: usize,
    /// Eje de proyección: 'x', 'y' o 'z' (default: 'z').
    pub axis: char,
}

impl Default for SzParams {
    fn default() -> Self {
        Self {
            n_pixels: 256,
            axis: 'z',
        }
    }
}

/// Mapa Compton-y (tSZ) 2D.
///
/// Cada píxel contiene el valor integral de `y = (σ_T / m_e c²) ∫ P_e dl`
/// a lo largo de la línea de visión.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ComptonYMap {
    /// Mapa 2D aplanado (fila-major), tamaño n_pixels².
    pub map: Vec<f64>,
    /// Número de píxeles por lado.
    pub n_pixels: usize,
    /// Tamaño del píxel en unidades internas (box_size / n_pixels).
    pub pixel_size: f64,
    /// Valor medio de y en el mapa.
    pub mean_y: f64,
    /// Valor máximo de y en el mapa.
    pub y_max: f64,
}

/// Mapa kinetic SZ 2D.
///
/// Cada píxel contiene `ΔT/T_CMB = -σ_T ∫ n_e (v·n̂/c) dl`,
/// donde `n̂` es la dirección de la línea de visión.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct KineticSzMap {
    /// Mapa 2D aplanado (fila-major), tamaño n_pixels².
    pub map: Vec<f64>,
    /// Número de píxeles por lado.
    pub n_pixels: usize,
    /// Tamaño del píxel en unidades internas.
    pub pixel_size: f64,
    /// RMS del mapa kSZ.
    pub rms_ksz: f64,
}

/// Presión electrónica `P_e = n_e k_B T_e` en unidades internas.
///
/// Para gas completamente ionizado con fracción de helio Y = 0.24:
/// - `μ_e = 2/(1+X_H) ≈ 1.143` donde `X_H = 0.76`
/// - `n_e = X_e ρ / (μ_e m_H)` donde `X_e ≈ 1.16`
/// - `P_e = n_e k_B T_e`
///
/// En unidades internas donde `u = (γ-1)^{-1} P/ρ`:
/// `P_e = ρ u (γ-1) X_e / μ_e`
#[inline]
fn electron_pressure(p: &Particle, gamma: f64) -> f64 {
    if !p.is_gas() || p.internal_energy <= 0.0 {
        return 0.0;
    }
    let x_e = 1.16_f64;
    let mu_e = 2.0 / (1.0 + 0.76);
    p.mass / p.smoothing_length.powi(3).max(1e-30) * p.internal_energy * (gamma - 1.0) * x_e / mu_e
}

/// Densidad electrónica `n_e` en unidades internas.
///
/// `n_e = X_e ρ / (μ_e m_p)` donde `m_p ≈ 1.673×10⁻²⁴ g` se expresa
/// en unidades internas (km/s)² / (Mpc/h).
#[inline]
fn electron_density(p: &Particle, _gamma: f64) -> f64 {
    if !p.is_gas() || p.internal_energy <= 0.0 {
        return 0.0;
    }
    let rho = p.mass / p.smoothing_length.powi(3).max(1e-30);
    let x_e = 1.16_f64;
    let mu_e = 2.0 / (1.0 + 0.76);
    rho * x_e / mu_e
}

/// Factor de conversión a Compton-y adimensional.
///
/// `y = (σ_T k_B) / (m_e c²) × ∫ dl × (P_e / k_B)` donde la integral
/// se convierte a CIC sobre el mapa. En unidades internas:
///
/// `σ_T / (m_e c²) ≈ 1.685×10⁻²⁴ cm² / (8.187×10⁻⁷ erg) = 2.058×10⁻¹⁸ cm²/erg`
///
/// Convertimos usando: `1 (km/s)² = 1.040×10⁻⁶ erg/g` y
/// box_size está en Mpc/h. El factor resultante es adimensional cuando
/// multiplicamos por la longitud de línea de visión en unidades de box.
const Y_CONVERSION: f64 = 2.058e-18 * 1.040e-6;

/// Factor de conversión para kSZ: `ΔT/T_CMB = -σ_T ∫ n_e (v_los/c) dl`.
///
/// `σ_T = 6.652×10⁻²⁵ cm²`, `c = 2.998×10⁵ km/s`.
/// Factor: `σ_T / c = 2.219×10⁻³⁰ cm²·s/km`
/// Convertimos con unidades internas: `1 Mpc/h ≈ 3.086×10²⁴ cm`.
const KSZ_CONVERSION: f64 = 6.652e-25 / 2.998e5 * 3.086e24;

/// Calcula el mapa Compton-y (tSZ) 2D proyectando la presión electrónica.
///
/// Asigna cada partícula de gas a un píxel del mapa 2D usando
/// CIC (Cloud-in-Cell). El eje de proyección es configurable.
///
/// # Parámetros
/// - `particles`: slice de partículas (solo las de tipo Gas contribuyen).
/// - `box_size`: tamaño de la caja en unidades internas (para normalización).
/// - `params`: parámetros de proyección (n_pixels, axis).
/// - `gamma`: índice adiabático (típicamente 5/3).
pub fn compute_compton_y_map(
    particles: &[Particle],
    box_size: f64,
    params: &SzParams,
    gamma: f64,
) -> ComptonYMap {
    let n = params.n_pixels;
    let pixel_size = box_size / n as f64;
    let mut map = vec![0.0f64; n * n];

    let axis_idx = match params.axis {
        'x' | 'X' => 0,
        'y' | 'Y' => 1,
        _ => 2,
    };
    let (proj0, proj1) = match axis_idx {
        0 => (1usize, 2),
        1 => (0, 2),
        _ => (0, 1),
    };

    for p in particles {
        if !p.is_gas() || p.internal_energy <= 0.0 {
            continue;
        }
        let pe = electron_pressure(p, gamma);
        if pe <= 0.0 {
            continue;
        }
        let weight = Y_CONVERSION * pe * pixel_size;

        let v = p.position;
        let (px, py) = match proj0 {
            0 => (v.x, v.y),
            1 => (
                v.x,
                match proj1 {
                    2 => v.z,
                    _ => v.y,
                },
            ),
            _ => (v.y, v.z),
        };

        let ix = (px / pixel_size) as usize;
        let iy = (py / pixel_size) as usize;
        if ix < n && iy < n {
            map[iy * n + ix] += weight;
        }
    }

    let total: f64 = map.iter().sum();
    let mean_y = if n * n > 0 {
        total / (n * n) as f64
    } else {
        0.0
    };
    let y_max = map.iter().cloned().fold(0.0f64, f64::max);

    ComptonYMap {
        map,
        n_pixels: n,
        pixel_size,
        mean_y,
        y_max,
    }
}

/// Calcula el mapa kinetic SZ 2D proyectando `n_e × v_los`.
///
/// El valor en cada píxel es `ΔT/T_CMB = -σ_T ∫ n_e (v·n̂/c) dl`,
/// donde `n̂` es el eje de proyección.
pub fn compute_kinetic_sz_map(
    particles: &[Particle],
    box_size: f64,
    params: &SzParams,
    gamma: f64,
) -> KineticSzMap {
    let n = params.n_pixels;
    let pixel_size = box_size / n as f64;
    let mut map = vec![0.0f64; n * n];

    let axis_idx = match params.axis {
        'x' | 'X' => 0,
        'y' | 'Y' => 1,
        _ => 2,
    };
    let (proj0, proj1) = match axis_idx {
        0 => (1usize, 2),
        1 => (0, 2),
        _ => (0, 1),
    };

    for p in particles {
        if !p.is_gas() || p.internal_energy <= 0.0 {
            continue;
        }
        let ne = electron_density(p, gamma);
        if ne <= 0.0 {
            continue;
        }
        let v_los = match axis_idx {
            0 => p.velocity.x,
            1 => p.velocity.y,
            _ => p.velocity.z,
        };
        let weight = KSZ_CONVERSION * ne * v_los * pixel_size;

        let v = p.position;
        let (px, py) = match proj0 {
            0 => (v.x, v.y),
            1 => (
                v.x,
                match proj1 {
                    2 => v.z,
                    _ => v.y,
                },
            ),
            _ => (v.y, v.z),
        };

        let ix = (px / pixel_size) as usize;
        let iy = (py / pixel_size) as usize;
        if ix < n && iy < n {
            map[iy * n + ix] += weight;
        }
    }

    let mean: f64 = map.iter().sum::<f64>() / (n * n).max(1) as f64;
    let variance: f64 =
        map.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / (n * n).max(1) as f64;
    let rms = variance.sqrt();

    KineticSzMap {
        map,
        n_pixels: n,
        pixel_size,
        rms_ksz: rms,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gadget_ng_core::{Particle, ParticleType, Vec3};

    fn make_gas_particle(
        x: f64,
        y: f64,
        z: f64,
        mass: f64,
        u: f64,
        h: f64,
        vx: f64,
        id: usize,
    ) -> Particle {
        Particle::new_gas(id, mass, Vec3::new(x, y, z), Vec3::new(vx, 0.0, 0.0), u, h)
    }

    #[test]
    fn zero_gas_zero_y() {
        let particles: Vec<Particle> = Vec::new();
        let params = SzParams::default();
        let result = compute_compton_y_map(&particles, 100.0, &params, 5.0 / 3.0);
        assert!(
            result.map.iter().all(|&y| y == 0.0),
            "Empty particles should give zero y"
        );
        assert_eq!(result.mean_y, 0.0);
    }

    #[test]
    fn electron_pressure_scales_with_density_and_temperature() {
        let p1 = make_gas_particle(50.0, 50.0, 50.0, 1.0, 1.0, 10.0, 0.0, 0);
        let p2 = make_gas_particle(50.0, 50.0, 50.0, 1.0, 4.0, 10.0, 0.0, 1);
        let pe1 = electron_pressure(&p1, 5.0 / 3.0);
        let pe2 = electron_pressure(&p2, 5.0 / 3.0);
        assert!(
            pe2 > pe1,
            "Higher internal_energy should give higher P_e: {pe2} > {pe1}"
        );
    }

    #[test]
    fn map_has_correct_dimensions() {
        let params = SzParams {
            n_pixels: 64,
            axis: 'z',
        };
        let p = make_gas_particle(50.0, 50.0, 50.0, 1.0, 100.0, 5.0, 10.0, 0);
        let result_y = compute_compton_y_map(&[p], 100.0, &params, 5.0 / 3.0);
        assert_eq!(result_y.map.len(), 64 * 64, "Map should be n_pixels^2");
        assert!((result_y.pixel_size - 100.0 / 64.0).abs() < 1e-10);

        let p2 = make_gas_particle(50.0, 50.0, 50.0, 1.0, 100.0, 5.0, 10.0, 1);
        let result_ksz = compute_kinetic_sz_map(&[p2], 100.0, &params, 5.0 / 3.0);
        assert_eq!(
            result_ksz.map.len(),
            64 * 64,
            "kSZ map should be n_pixels^2"
        );
    }
}
