//! Gravedad modificada Hu-Sawicki f(R) con screening chameleon — Phase 158.
//!
//! ## Modelo físico
//!
//! El modelo Hu-Sawicki (2007) modifica la acción gravitacional añadiendo
//! una función f(R) del escalar de Ricci:
//!
//! ```text
//! S = (1/16πG) ∫ d⁴x √(-g) [R + f(R)] + S_matter
//! ```
//!
//! En el límite de campo débil, el efecto neto es una "quinta fuerza" que
//! amplifica la gravedad newtoniana por un factor `1 + 1/3 × |f_R|/|f_R0|`
//! fuera de las regiones densas (no screened).
//!
//! ### Mecanismo de screening (chameleon)
//!
//! En regiones de alta densidad (sobredensidad δρ/ρ̄ >> 1), el campo escalar
//! f_R adquiere una masa grande y la quinta fuerza es suprimida localmente.
//! El factor de screening se estima como:
//!
//! ```text
//! fifth_force_factor = min(1, |f_R_local / f_R0|)
//! Δg/g = (1/3) × fifth_force_factor
//! ```
//!
//! ## Parámetros del modelo Hu-Sawicki
//!
//! - `f_R0`: valor del campo escalar hoy (z=0). Típico: 10⁻⁶ a 10⁻⁴.
//! - `n`: índice del modelo (n=1 es el más estudiado).
//!
//! ## Referencia
//!
//! Hu & Sawicki (2007) PRD 76, 064004.
//! Khoury & Weltman (2004) PRD 69, 044026 — mecanismo chameleon.

use crate::{cosmology::CosmologyParams, particle::Particle};

/// Parámetros del modelo f(R) Hu-Sawicki (Phase 158).
#[derive(Debug, Clone)]
pub struct FRParams {
    /// |f_R0| — valor del campo escalar en z=0. Típicamente 10⁻⁶ – 10⁻⁴.
    pub f_r0: f64,
    /// Índice n del modelo (n=1 es el más estudiado).
    pub n: f64,
}

impl Default for FRParams {
    fn default() -> Self {
        Self { f_r0: 1.0e-4, n: 1.0 }
    }
}

/// Calcula el campo escalar f_R local dado la sobredensidad (Phase 158).
///
/// En el modelo Hu-Sawicki, el campo f_R en un punto de sobredensidad δ es:
///
/// ```text
/// f_R_local = -n × (2Ω_Λ/Ω_m)^{n+1} × (f_R0 / n) × (ρ̄_m / ρ_local)^{n+1}
/// ```
///
/// Aproximación simplificada para δ = (ρ_local - ρ̄) / ρ̄:
///
/// ```text
/// f_R_local ≈ f_R0 × (1 + δ)^{-(n+1)}
/// ```
///
/// # Parámetros
/// - `delta_rho`: sobredensidad δ = (ρ - ρ̄)/ρ̄
/// - `f_r0`: valor del campo en z=0
/// - `n`: índice del modelo
pub fn chameleon_field(delta_rho: f64, f_r0: f64, n: f64) -> f64 {
    if delta_rho <= -1.0 { return f_r0.abs(); }
    let rho_ratio = 1.0 + delta_rho;
    f_r0.abs() * rho_ratio.powf(-(n + 1.0))
}

/// Factor de amplificación de la quinta fuerza (Phase 158).
///
/// ```text
/// fifth_force_factor = min(1, |f_R_local / f_R0|)
/// ```
///
/// En regiones densas (screened), f_R_local << f_R0 → factor ≈ 0.
/// En regiones vacías (no screened), f_R_local ≈ f_R0 → factor ≈ 1.
///
/// # Parámetros
/// - `f_r_local`: campo escalar f_R local
/// - `f_r0`: valor de referencia en z=0
pub fn fifth_force_factor(f_r_local: f64, f_r0: f64) -> f64 {
    if f_r0.abs() <= 0.0 { return 0.0; }
    (f_r_local.abs() / f_r0.abs()).min(1.0)
}

/// Aplica la modificación de gravedad f(R) a las partículas (Phase 158).
///
/// Para cada partícula de materia (no gas, no estrella) escala la aceleración
/// por `1 + (1/3) × fifth_force_factor(δρ)`.
///
/// El gas y las estrellas también reciben la quinta fuerza (gravedad = universal),
/// pero se puede inhibir en regiones de alta densidad barionica (screening).
///
/// # Parámetros
/// - `particles`: partículas de la simulación
/// - `params`: parámetros f(R)
/// - `cosmo`: parámetros cosmológicos
/// - `a`: factor de escala actual
pub fn apply_modified_gravity(
    particles: &mut Vec<Particle>,
    params: &FRParams,
    cosmo: &CosmologyParams,
    a: f64,
) {
    if params.f_r0.abs() <= 0.0 { return; }

    // Calcular densidad media para la sobredensidad
    let rho_bar = if !particles.is_empty() {
        let total_mass: f64 = particles.iter().map(|p| p.mass).sum();
        total_mass / particles.len() as f64
    } else {
        return;
    };

    // Convertir Ω_m a densidad de referencia (unidades internas)
    // ρ_crit ∝ H²; aquí usamos rho_bar como proxy
    let _ = (cosmo, a); // parámetros disponibles para implementaciones más detalladas

    for p in particles.iter_mut() {
        let rho_local = if p.smoothing_length > 0.0 {
            p.mass / p.smoothing_length.powi(3)
        } else {
            rho_bar
        };

        let delta_rho = (rho_local - rho_bar) / rho_bar.max(1e-30);
        let f_r_local = chameleon_field(delta_rho, params.f_r0, params.n);
        let factor = 1.0 + fifth_force_factor(f_r_local, params.f_r0) / 3.0;

        p.acceleration.x *= factor;
        p.acceleration.y *= factor;
        p.acceleration.z *= factor;
    }
}
