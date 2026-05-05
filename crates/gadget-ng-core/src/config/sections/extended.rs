use serde::{Deserialize, Serialize};

/// Forzado de turbulencia MHD estocástico Ornstein-Uhlenbeck (Phase 140).
///
/// Genera turbulencia Alfvénica con espectro de Kolmogorov `E(k) ∝ k^{-5/3}`
/// o Goldreich-Sridhar `E(k) ∝ k^{-3/2}` en presencia de campo B₀ de fondo.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurbulenceSection {
    /// Activa el forzado turbulento (default: `false`).
    #[serde(default)]
    pub enabled: bool,
    /// Amplitud del forzado (default: `1e-3`).
    #[serde(default = "default_turb_amplitude")]
    pub amplitude: f64,
    /// Tiempo de correlación del proceso OU [unidades internas] (default: `1.0`).
    #[serde(default = "default_turb_tau")]
    pub correlation_time: f64,
    /// Número de onda mínimo de la banda de forzado (default: `1.0`).
    #[serde(default = "default_turb_k_min")]
    pub k_min: f64,
    /// Número de onda máximo de la banda de forzado (default: `4.0`).
    #[serde(default = "default_turb_k_max")]
    pub k_max: f64,
    /// Índice espectral: `5/3` (Kolmogorov) o `3/2` (Goldreich-Sridhar) (default: `1.6667`).
    #[serde(default = "default_turb_spectral_index")]
    pub spectral_index: f64,
}

fn default_turb_amplitude() -> f64 {
    1e-3
}
fn default_turb_tau() -> f64 {
    1.0
}
fn default_turb_k_min() -> f64 {
    1.0
}
fn default_turb_k_max() -> f64 {
    4.0
}
fn default_turb_spectral_index() -> f64 {
    5.0 / 3.0
}

impl Default for TurbulenceSection {
    fn default() -> Self {
        Self {
            enabled: false,
            amplitude: default_turb_amplitude(),
            correlation_time: default_turb_tau(),
            k_min: default_turb_k_min(),
            k_max: default_turb_k_max(),
            spectral_index: default_turb_spectral_index(),
        }
    }
}

/// Plasma de dos fluidos: temperaturas de electrones e iones separadas (Phase 149).
///
/// El acoplamiento Coulomb transfiere calor entre electrones e iones:
/// `dT_e/dt = −ν_ei (T_e − T_i)` con `ν_ei ∝ n_e / T_e^{3/2}`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TwoFluidSection {
    /// Activa el plasma de dos fluidos (default: `false`).
    #[serde(default)]
    pub enabled: bool,
    /// Coeficiente de acoplamiento Coulomb `ν_ei` en unidades internas (default: `1.0`).
    #[serde(default = "default_nu_ei_coeff")]
    pub nu_ei_coeff: f64,
    /// Temperatura electrónica inicial en Kelvin (default: igual a T_i).
    /// Si `0.0`, se inicializa igual a T_i al arranque.
    #[serde(default)]
    pub t_e_init_k: f64,
}

fn default_nu_ei_coeff() -> f64 {
    1.0
}

impl Default for TwoFluidSection {
    fn default() -> Self {
        Self {
            enabled: false,
            nu_ei_coeff: default_nu_ei_coeff(),
            t_e_init_k: 0.0,
        }
    }
}

// ── SIDM (Phase 157) ─────────────────────────────────────────────────────────

/// Configuración SIDM — materia oscura auto-interactuante (Phase 157).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SidmSection {
    /// `true` activa el módulo SIDM en cada paso de tiempo.
    #[serde(default)]
    pub enabled: bool,
    /// Sección eficaz por masa σ/m en unidades internas (default: 1×10⁻⁵).
    #[serde(default = "default_sidm_sigma_m")]
    pub sigma_m: f64,
    /// Velocidad máxima de corte para el scattering (default: 1×10⁶).
    #[serde(default = "default_sidm_v_max")]
    pub v_max: f64,
}

fn default_sidm_sigma_m() -> f64 {
    1.0e-5
}
fn default_sidm_v_max() -> f64 {
    1.0e6
}

impl Default for SidmSection {
    fn default() -> Self {
        Self {
            enabled: false,
            sigma_m: default_sidm_sigma_m(),
            v_max: default_sidm_v_max(),
        }
    }
}

// ── Gravedad modificada f(R) (Phase 158) ─────────────────────────────────────

/// Configuración de gravedad modificada Hu-Sawicki f(R) (Phase 158).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModifiedGravitySection {
    /// `true` activa el módulo post-gravedad normal.
    #[serde(default)]
    pub enabled: bool,
    /// Modelo de gravedad modificada: sólo `"hu_sawicki"` por ahora.
    #[serde(default = "default_mg_model")]
    pub model: String,
    /// Parámetro |f_R0| del modelo Hu-Sawicki (default: 1×10⁻⁴).
    #[serde(default = "default_f_r0")]
    pub f_r0: f64,
    /// Índice n del modelo Hu-Sawicki (default: 1).
    #[serde(default = "default_mg_n")]
    pub n: f64,
}

fn default_mg_model() -> String {
    "hu_sawicki".to_string()
}
fn default_f_r0() -> f64 {
    1.0e-4
}
fn default_mg_n() -> f64 {
    1.0
}

impl Default for ModifiedGravitySection {
    fn default() -> Self {
        Self {
            enabled: false,
            model: default_mg_model(),
            f_r0: default_f_r0(),
            n: default_mg_n(),
        }
    }
}
