use serde::{Deserialize, Serialize};

/// Configuración del solver de transferencia radiativa M1 (Phase 81).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RtSection {
    /// Activa el solver de transferencia radiativa M1 (default: `false`).
    #[serde(default)]
    pub enabled: bool,
    /// Factor de reducción de la velocidad de la luz (default: 100.0).
    #[serde(default = "default_c_red_factor")]
    pub c_red_factor: f64,
    /// Opacidad de absorción κ_abs en unidades internas (default: 1.0).
    #[serde(default = "default_kappa_abs")]
    pub kappa_abs: f64,
    /// Número de celdas del grid de radiación por lado (default: 32).
    #[serde(default = "default_rt_mesh")]
    pub rt_mesh: usize,
    /// Número de sub-pasos del solver M1 por paso de simulación (default: 5).
    #[serde(default = "default_rt_substeps")]
    pub substeps: usize,
}

fn default_c_red_factor() -> f64 {
    100.0
}
fn default_kappa_abs() -> f64 {
    1.0
}
fn default_rt_mesh() -> usize {
    32
}
fn default_rt_substeps() -> usize {
    5
}

impl Default for RtSection {
    fn default() -> Self {
        Self {
            enabled: false,
            c_red_factor: default_c_red_factor(),
            kappa_abs: default_kappa_abs(),
            rt_mesh: default_rt_mesh(),
            substeps: default_rt_substeps(),
        }
    }
}

// ── ReionizationSection ───────────────────────────────────────────────────────

/// Configuración del módulo de reionización del Universo (Phase 89).
///
/// ```toml
/// [reionization]
/// enabled = true
/// n_sources = 4          # número de fuentes UV
/// uv_luminosity = 1.0    # luminosidad por fuente [unidades internas]
/// z_start = 12.0
/// z_end   = 6.0
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReionizationSection {
    /// Activa el módulo (default: `false`).
    #[serde(default)]
    pub enabled: bool,
    /// Número de fuentes UV homogéneamente distribuidas (default: 0).
    #[serde(default)]
    pub n_sources: usize,
    /// Luminosidad UV por fuente en unidades internas (default: 1.0).
    #[serde(default = "default_uv_luminosity")]
    pub uv_luminosity: f64,
    /// Redshift de inicio de la reionización (default: 12.0).
    #[serde(default = "default_z_reion_start")]
    pub z_start: f64,
    /// Redshift de fin de la reionización (default: 6.0).
    #[serde(default = "default_z_reion_end")]
    pub z_end: f64,
    /// Si `true`, las fuentes UV se colocan en los halos FoF más masivos del análisis in-situ.
    /// Requiere `insitu_analysis.enabled = true`. Default: false (fuentes uniformes).
    #[serde(default)]
    pub uv_from_halos: bool,
}

fn default_uv_luminosity() -> f64 {
    1.0
}
fn default_z_reion_start() -> f64 {
    12.0
}
fn default_z_reion_end() -> f64 {
    6.0
}

impl Default for ReionizationSection {
    fn default() -> Self {
        Self {
            enabled: false,
            n_sources: 0,
            uv_luminosity: default_uv_luminosity(),
            z_start: default_z_reion_start(),
            z_end: default_z_reion_end(),
            uv_from_halos: false,
        }
    }
}
