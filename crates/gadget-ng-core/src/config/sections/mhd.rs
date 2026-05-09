use serde::{Deserialize, Serialize};

/// Configuración del módulo MHD ideal (Phase 126).
///
/// ```toml
/// [mhd]
/// enabled = true
/// c_h     = 1.0
/// c_r     = 0.5
/// ```
/// Tipo de condición inicial para el campo magnético (Phase 127).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum BFieldKind {
    /// Sin campo magnético inicial (default).
    #[default]
    None,
    /// Campo uniforme en la dirección `b0_uniform`.
    Uniform,
    /// Campo aleatorio con amplitud `|b0_uniform|`.
    Random,
    /// Campo espiral: `B = B0 × (sin(2πy/L), cos(2πx/L), 0)`.
    Spiral,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MhdSection {
    /// Activa el solver MHD (default: `false`).
    #[serde(default)]
    pub enabled: bool,
    /// Velocidad de propagación de ondas de limpieza Dedner (default: `1.0`).
    #[serde(default = "default_mhd_c_h")]
    pub c_h: f64,
    /// Tasa de amortiguamiento Dedner (default: `0.5`).
    #[serde(default = "default_mhd_c_r")]
    pub c_r: f64,
    /// Tipo de condición inicial para B (Phase 127). Default: `None`.
    #[serde(default)]
    pub b0_kind: BFieldKind,
    /// Campo magnético inicial en unidades internas (Phase 127). Default: `[0,0,0]`.
    #[serde(default)]
    pub b0_uniform: [f64; 3],
    /// Número de Courant magnético para el límite CFL de Alfvén (Phase 127). Default: `0.3`.
    #[serde(default = "default_cfl_mhd")]
    pub cfl_mhd: f64,
    /// Coeficiente de resistividad numérica artificial (Phase 135). Default: `0.5`.
    ///
    /// `η_art = alpha_b × h_i × v_sig`. Con `alpha_b = 0.0` se desactiva la resistividad.
    #[serde(default = "default_alpha_b")]
    pub alpha_b: f64,
    /// Intervalo de pasos para calcular estadísticas de B (Phase 136). Default: `0` (desactivado).
    #[serde(default)]
    pub stats_interval: usize,
    /// β-plasma umbral para flux-freeze en ICM (Phase 138). Default: `100.0`.
    ///
    /// Si `β > beta_freeze`, el campo B se "congela" con el fluido
    /// (resistividad desactivada para esa partícula).
    #[serde(default = "default_beta_freeze")]
    pub beta_freeze: f64,
    /// Activa SRMHD especial-relativista (Phase 139). Default: `false`.
    #[serde(default)]
    pub relativistic_mhd: bool,
    /// Umbral de |v|/c para aplicar correcciones relativistas (Phase 139). Default: `0.1`.
    #[serde(default = "default_v_rel_threshold")]
    pub v_rel_threshold: f64,
    /// Activa reconexión magnética Sweet-Parker (Phase 145). Default: `false`.
    #[serde(default)]
    pub reconnection_enabled: bool,
    /// Fracción de energía magnética liberada por reconexión por paso (Phase 145). Default: `0.01`.
    #[serde(default = "default_f_reconnection")]
    pub f_reconnection: f64,
    /// Coeficiente de viscosidad Braginskii anisótropa (Phase 146). Default: `0.0` (desactivado).
    #[serde(default)]
    pub eta_braginskii: f64,
    /// Activa jets AGN relativistas desde halos FoF (Phase 148). Default: `false`.
    #[serde(default)]
    pub jet_enabled: bool,
    /// Velocidad del jet en unidades de c (Phase 148). Default: `0.3`.
    #[serde(default = "default_v_jet")]
    pub v_jet: f64,
    /// Número de halos FoF que inyectan jets (Phase 148). Default: `1`.
    #[serde(default = "default_n_jet_halos")]
    pub n_jet_halos: usize,
    /// Acta el dinamo turbulento α-effect (Phase 172). Default: `false`.
    #[serde(default)]
    pub dynamo_enabled: bool,
    /// Tiempo de decaimiento del campo magnético en unidades internas (Phase 172). Default: `10.0`.
    #[serde(default = "default_dynamo_decay_time")]
    pub dynamo_decay_time: f64,
}

fn default_mhd_c_h() -> f64 {
    1.0
}
fn default_mhd_c_r() -> f64 {
    0.5
}
fn default_cfl_mhd() -> f64 {
    0.3
}
fn default_alpha_b() -> f64 {
    0.5
}
fn default_beta_freeze() -> f64 {
    100.0
}
fn default_v_rel_threshold() -> f64 {
    0.1
}
fn default_f_reconnection() -> f64 {
    0.01
}
fn default_v_jet() -> f64 {
    0.3
}
fn default_n_jet_halos() -> usize {
    1
}
fn default_dynamo_decay_time() -> f64 {
    10.0
}

impl Default for MhdSection {
    fn default() -> Self {
        Self {
            enabled: false,
            c_h: default_mhd_c_h(),
            c_r: default_mhd_c_r(),
            b0_kind: BFieldKind::None,
            b0_uniform: [0.0; 3],
            cfl_mhd: default_cfl_mhd(),
            alpha_b: default_alpha_b(),
            stats_interval: 0,
            beta_freeze: default_beta_freeze(),
            relativistic_mhd: false,
            v_rel_threshold: default_v_rel_threshold(),
            reconnection_enabled: false,
            f_reconnection: default_f_reconnection(),
            eta_braginskii: 0.0,
            jet_enabled: false,
            v_jet: default_v_jet(),
            n_jet_halos: default_n_jet_halos(),
            dynamo_enabled: false,
            dynamo_decay_time: default_dynamo_decay_time(),
        }
    }
}
