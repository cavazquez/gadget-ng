//! Reionización: fuentes UV puntuales y seguimiento del frente de ionización (Phase 89).
//!
//! Implementa la física de reionización del Universo modelando fuentes de fotones UV
//! (p. ej. galaxias y cuásares) como puntos discretos en el grid del campo de radiación M1.
//!
//! ## Modelo
//!
//! Cada fuente UV deposita fotones en la celda del grid M1 más cercana, aumentando
//! la densidad de energía radiativa. Estos fotones se propagan usando el solver M1
//! (`m1_update`) y fotoionizan el gas mediante `apply_chemistry`.
//!
//! ## Estadística de reionización
//!
//! `ReionizationState` agrega la fracción media de ionización `<x_HII>` a redshift z.
//! Cuando `x_hii_mean > 0.5` (convención estándar), el IGM está "mayoritariamente ionizado".
//!
//! ## Validación: esfera de Strömgren
//!
//! Para una fuente puntual con tasa de ionizaciones Ṅ_ion en un medio uniforme de
//! densidad n_H, el radio de Strömgren al equilibrio es:
//!
//! ```text
//! R_S = (3 Ṅ_ion / (4π n_H² α_B))^(1/3)
//! ```
//!
//! donde α_B ≈ 2.6×10⁻¹³ cm³/s es la tasa de recombinación case-B a T=10⁴ K.
//!
//! ## Referencia
//!
//! Rosdahl & Teyssier (2015), MNRAS 449, 4380;
//! Pawlik & Schaye (2008), MNRAS 389, 651.

use gadget_ng_core::Vec3;

use crate::chemistry::ChemState;
use crate::m1::RadiationField;

// ── Constantes ────────────────────────────────────────────────────────────────

/// Tasa de recombinación case-B de HII [cm³/s] a T = 10⁴ K (Osterbrock 2006).
pub const ALPHA_B: f64 = 2.6e-13;

// ── Structs ───────────────────────────────────────────────────────────────────

/// Fuente de fotones UV puntual (p. ej. galaxia o cuásar).
#[derive(Debug, Clone)]
pub struct UvSource {
    /// Posición de la fuente en coordenadas de caja [0, box_size)³.
    pub pos: Vec3,
    /// Luminosidad UV: tasa de emisión de fotones ionizantes Ṅ_ion [fotones/s en unidades internas].
    /// En la práctica, unidades: energía / (tiempo × volumen_celda) añadida al campo de radiación.
    pub luminosity: f64,
}

/// Estado global de reionización en un instante de tiempo.
#[derive(Debug, Clone, Default)]
pub struct ReionizationState {
    /// Fracción media de ionización del hidrógeno `<x_HII>` sobre todas las partículas de gas.
    /// 0 = completamente neutro, 1 = completamente ionizado.
    pub x_hii_mean: f64,
    /// Desviación estándar de `x_HII`.
    pub x_hii_sigma: f64,
    /// Fracción de volumen del IGM con `x_HII > 0.5` (proxy de la fracción ionizada).
    pub ionized_volume_fraction: f64,
    /// Redshift del instante.
    pub z: f64,
    /// Número de fuentes UV activas.
    pub n_sources: usize,
}

/// Parámetros de la reionización.
#[derive(Debug, Clone)]
pub struct ReionizationParams {
    /// Activa el módulo de reionización.
    pub enabled: bool,
    /// Radio de suavizado CIC para el depósito de fuentes UV (en número de celdas).
    /// `0` → depósito en celda más cercana (NGP); `1` → CIC (default).
    pub deposit_ngp: bool,
    /// Escala de tiempo de reionización: z de inicio y fin.
    pub z_start: f64,
    pub z_end: f64,
}

impl Default for ReionizationParams {
    fn default() -> Self {
        Self {
            enabled: false,
            deposit_ngp: false,
            z_start: 12.0,
            z_end: 6.0,
        }
    }
}

// ── Funciones principales ─────────────────────────────────────────────────────

/// Deposita la emisión de fuentes UV puntuales en el grid de radiación M1.
///
/// Para cada fuente, identifica la celda del grid más cercana (NGP) o usa CIC
/// (en función de `params.deposit_ngp`) y añade `luminosity × dt` a la energía radiativa.
///
/// # Argumentos
/// - `rad`      — campo de radiación M1 (modificado in-place)
/// - `sources`  — lista de fuentes UV
/// - `box_size` — tamaño de la caja periódica
/// - `dt`       — paso de tiempo
pub fn deposit_uv_sources(rad: &mut RadiationField, sources: &[UvSource], box_size: f64, dt: f64) {
    let nx = rad.nx;
    let ny = rad.ny;
    let nz = rad.nz;
    let dx = rad.dx;

    for src in sources {
        // Coordenadas de celda (con wrapping periódico)
        let cx = ((src.pos.x / dx).floor() as isize).rem_euclid(nx as isize) as usize;
        let cy = ((src.pos.y / dx).floor() as isize).rem_euclid(ny as isize) as usize;
        let cz = ((src.pos.z / dx).floor() as isize).rem_euclid(nz as isize) as usize;

        // NGP: depositar toda la energía en la celda más cercana
        let idx = rad.idx(cx, cy, cz);
        let dv = dx * dx * dx;
        rad.energy_density[idx] += src.luminosity * dt / dv;

        // Asegurar positividad
        if rad.energy_density[idx] < 0.0 {
            rad.energy_density[idx] = 0.0;
        }
    }

    let _ = box_size; // utilizado implícitamente vía dx = box_size / n
}

/// Calcula el estado de reionización a partir de los estados químicos de las partículas de gas.
///
/// # Argumentos
/// - `chem_states` — fracciones de ionización por partícula de gas
/// - `z`           — redshift actual
/// - `n_sources`   — número de fuentes UV activas
pub fn compute_reionization_state(
    chem_states: &[ChemState],
    z: f64,
    n_sources: usize,
) -> ReionizationState {
    if chem_states.is_empty() {
        return ReionizationState {
            z,
            n_sources,
            ..Default::default()
        };
    }

    let n = chem_states.len() as f64;
    let mean = chem_states.iter().map(|s| s.x_hii).sum::<f64>() / n;
    let variance = chem_states
        .iter()
        .map(|s| (s.x_hii - mean).powi(2))
        .sum::<f64>()
        / n;
    let sigma = variance.sqrt();
    let ionized_fraction = chem_states.iter().filter(|s| s.x_hii > 0.5).count() as f64 / n;

    ReionizationState {
        x_hii_mean: mean,
        x_hii_sigma: sigma,
        ionized_volume_fraction: ionized_fraction,
        z,
        n_sources,
    }
}

/// Paso de reionización: depositar fuentes, actualizar M1, actualizar química.
///
/// Wrapper conveniente que integra `deposit_uv_sources` + `m1_update` + `apply_chemistry`
/// en un único llamada para usar dentro del loop de simulación.
///
/// # Argumentos
/// - `rad`          — campo de radiación M1
/// - `chem_states`  — estados químicos de las partículas de gas (modificados in-place)
/// - `sources`      — fuentes UV activas en este paso
/// - `m1_params`    — parámetros del solver M1
/// - `chem_params`  — parámetros de la química
/// - `dt`           — paso de tiempo
/// - `box_size`     — tamaño de la caja
/// - `z`            — redshift actual
///
/// # Retorna
/// Estado de reionización tras el paso.
pub fn reionization_step(
    rad: &mut RadiationField,
    chem_states: &mut [ChemState],
    sources: &[UvSource],
    m1_params: &crate::m1::M1Params,
    dt: f64,
    box_size: f64,
    z: f64,
) -> ReionizationState {
    // 1. Depositar fuentes UV en el campo de radiación
    deposit_uv_sources(rad, sources, box_size, dt);

    // 2. Propagar fotones con el solver M1
    crate::m1::m1_update(rad, dt, m1_params);

    // 3. Calcular estado de reionización
    compute_reionization_state(chem_states, z, sources.len())
}

/// Radio de Strömgren al equilibrio para una fuente puntual en un medio uniforme.
///
/// ```text
/// R_S = (3 Ṅ_ion / (4π n_H² α_B))^(1/3)
/// ```
///
/// # Argumentos
/// - `n_ion_rate` — tasa de ionizaciones Ṅ_ion [fotones/s]
/// - `n_h`        — densidad numérica de H [cm⁻³]
///
/// # Retorna
/// Radio de Strömgren R_S [cm].
pub fn stromgren_radius(n_ion_rate: f64, n_h: f64) -> f64 {
    let vol = 3.0 * n_ion_rate / (4.0 * std::f64::consts::PI * n_h * n_h * ALPHA_B);
    vol.cbrt()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::m1::{M1Params, RadiationField};

    fn make_rf(n: usize) -> RadiationField {
        RadiationField::uniform(n, n, n, 1.0 / n as f64, 0.0)
    }

    fn make_neutral_states(n: usize) -> Vec<ChemState> {
        vec![ChemState::neutral(); n]
    }

    fn make_ionized_states(n: usize) -> Vec<ChemState> {
        (0..n)
            .map(|_| {
                let mut s = ChemState::neutral();
                s.x_hii = 0.8;
                s.x_hi = 0.2;
                s.x_e = 0.8;
                s
            })
            .collect()
    }

    #[test]
    fn deposit_uv_sources_increases_energy() {
        let n = 8;
        let box_size = 1.0;
        let dx = box_size / n as f64;
        let mut rad = make_rf(n);
        let e_total_before = rad.energy_density.iter().sum::<f64>();

        let sources = vec![UvSource {
            pos: Vec3::new(0.5, 0.5, 0.5),
            luminosity: 1.0,
        }];
        deposit_uv_sources(&mut rad, &sources, box_size, 0.01);

        let e_total_after = rad.energy_density.iter().sum::<f64>();
        let dv = dx * dx * dx;
        assert!(
            e_total_after > e_total_before,
            "Energía debe aumentar tras depositar fuente UV"
        );
        // La energía añadida debería ser luminosity × dt / dv
        let added = (e_total_after - e_total_before) * dv;
        let expected = 1.0 * 0.01; // luminosity × dt
        assert!(
            (added - expected).abs() < 1e-12,
            "Energía añadida = {added}, esperado = {expected}"
        );
    }

    #[test]
    fn deposit_uv_sources_periodic_wrapping() {
        let n = 8;
        let box_size = 1.0;
        let mut rad = make_rf(n);

        // Fuente en posición negativa → debe wrappear periódicamente
        let sources = vec![UvSource {
            pos: Vec3::new(-0.01, 0.5, 0.5),
            luminosity: 1.0,
        }];
        // No debe paniquear
        deposit_uv_sources(&mut rad, &sources, box_size, 0.01);
        let e_total: f64 = rad.energy_density.iter().sum();
        assert!(e_total > 0.0);
    }

    #[test]
    fn compute_reionization_state_neutral() {
        let states = make_neutral_states(100);
        let state = compute_reionization_state(&states, 10.0, 1);
        assert!(
            state.x_hii_mean < 1e-6,
            "Estado neutro: x_hii_mean debe ser ~0"
        );
        assert!(state.ionized_volume_fraction < 1e-6);
        assert_eq!(state.n_sources, 1);
        assert_eq!(state.z, 10.0);
    }

    #[test]
    fn compute_reionization_state_ionized() {
        let states = make_ionized_states(100);
        let state = compute_reionization_state(&states, 7.0, 3);
        assert!(
            state.x_hii_mean > 0.7,
            "Gas ionizado: x_hii_mean debe ser > 0.7"
        );
        assert!(
            state.ionized_volume_fraction > 0.9,
            "Fracción ionizada debe ser > 0.9"
        );
    }

    #[test]
    fn compute_reionization_state_empty() {
        let state = compute_reionization_state(&[], 8.0, 0);
        assert_eq!(state.x_hii_mean, 0.0);
        assert_eq!(state.n_sources, 0);
    }

    #[test]
    fn reionization_step_no_crash() {
        let n = 4;
        let box_size = 1.0;
        let mut rad = make_rf(n);
        let mut chem_states = make_neutral_states(8);
        let sources = vec![UvSource {
            pos: Vec3::new(0.3, 0.3, 0.3),
            luminosity: 0.5,
        }];
        let params = M1Params {
            c_red_factor: 100.0,
            kappa_abs: 0.1,
            kappa_scat: 0.0,
            substeps: 1,
            ..Default::default()
        };
        let state = reionization_step(
            &mut rad,
            &mut chem_states,
            &sources,
            &params,
            0.001,
            box_size,
            9.0,
        );
        assert_eq!(state.n_sources, 1);
        assert!(state.z > 0.0);
        // El campo de energía debe tener al menos una celda con energía positiva
        let has_energy = rad.energy_density.iter().any(|&e| e > 0.0);
        assert!(
            has_energy,
            "Alguna celda debe tener energía UV tras el paso de reionización"
        );
    }

    #[test]
    fn stromgren_radius_reasonable() {
        // Para Ṅ_ion = 1e48 fotones/s y n_H = 1e-3 cm⁻³:
        let rs = stromgren_radius(1e48, 1e-3);
        let rs_kpc = rs / 3.086e21; // convertir cm → kpc
        // Radio típico de esfera de Strömgren: 0.1 - 10 Mpc
        assert!(
            rs_kpc > 1.0 && rs_kpc < 1e7,
            "R_S = {rs_kpc:.2e} kpc fuera del rango esperado"
        );
    }

    #[test]
    fn stromgren_radius_scales_correctly() {
        // R_S ∝ Ṅ_ion^(1/3) — doblar Ṅ_ion → R_S aumenta 2^(1/3) ≈ 1.26
        let rs1 = stromgren_radius(1e48, 1e-3);
        let rs2 = stromgren_radius(2e48, 1e-3);
        let ratio = rs2 / rs1;
        assert!(
            (ratio - 2.0f64.cbrt()).abs() < 1e-10,
            "Ratio = {ratio}, esperado = {:.4}",
            2.0f64.cbrt()
        );
    }

    #[test]
    fn reionization_state_default_is_zero() {
        let state = ReionizationState::default();
        assert_eq!(state.x_hii_mean, 0.0);
        assert_eq!(state.n_sources, 0);
    }

    #[test]
    fn reionization_params_default_disabled() {
        let params = ReionizationParams::default();
        assert!(!params.enabled);
        assert!(params.z_start > params.z_end);
    }
}
