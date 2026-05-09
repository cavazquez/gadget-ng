use serde::{Deserialize, Serialize};

// ── InsituAnalysisSection ─────────────────────────────────────────────────────

/// Configuración del análisis in-situ ejecutado durante el loop `stepping` (Phase 63).
///
/// Se activa añadiendo `[insitu_analysis]` al TOML de configuración:
///
/// ```toml
/// [insitu_analysis]
/// enabled   = true
/// interval  = 20       # cada 20 pasos
/// pk_mesh   = 32
/// fof_b     = 0.2
/// fof_min_part = 20
/// xi_bins   = 10       # 0 = desactivado
/// output_dir = "runs/cosmo/insitu"
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InsituAnalysisSection {
    /// Activa el análisis in-situ (default: `false`).
    #[serde(default)]
    pub enabled: bool,
    /// Ejecutar cada N pasos. `0` → desactivado aunque `enabled = true`.
    #[serde(default = "default_insitu_interval")]
    pub interval: u64,
    /// Resolución del grid para P(k) (por lado). Default: 32.
    #[serde(default = "default_pk_mesh")]
    pub pk_mesh: usize,
    /// Parámetro de enlace FoF. Default: 0.2.
    #[serde(default = "default_fof_b")]
    pub fof_b: f64,
    /// Mínimo de partículas para un halo FoF. Default: 20.
    #[serde(default = "default_fof_min_part")]
    pub fof_min_part: usize,
    /// Número de bins para ξ(r). `0` → no calcular ξ(r). Default: 0.
    #[serde(default)]
    pub xi_bins: usize,
    /// Número de bins en μ para P(k,μ) en espacio de redshift. `0` → no calcular. Default: 0.
    #[serde(default)]
    pub pk_rsd_bins: usize,
    /// Número de bins para el bispectrum equilateral B(k). `0` → no calcular. Default: 0.
    #[serde(default)]
    pub bispectrum_bins: usize,
    /// Activar cálculo de assembly bias (correlación spin/concentración vs entorno). Default: false.
    #[serde(default)]
    pub assembly_bias_enabled: bool,
    /// Radio de suavizado para el campo de densidad del entorno (unidades internas). Default: 5.0.
    #[serde(default = "default_ab_smooth_r")]
    pub assembly_bias_smooth_r: f64,
    /// Activar cálculo del perfil de temperatura del IGM T(z). Default: false.
    #[serde(default)]
    pub igm_temp_enabled: bool,
    /// Activar estadísticas de la línea de 21cm (δT_b, P(k)₂₁cm). Default: false.
    #[serde(default)]
    pub cm21_enabled: bool,
    /// Activar cálculo del efecto Sunyaev-Zel'dovich (Compton-y + kSZ). Default: false.
    #[serde(default)]
    pub sz_enabled: bool,
    /// Resolución del mapa SZ (píxeles por lado). Default: 256.
    #[serde(default = "default_sz_n_pixels")]
    pub sz_n_pixels: usize,
    /// Activar análisis del bosque Lyman-α (τ_GP, F(v), P(k)_F). Default: false.
    #[serde(default)]
    pub lya_enabled: bool,
    /// Número de sightlines para el bosque Ly-α. Default: 256.
    #[serde(default = "default_lya_n_sightlines")]
    pub lya_n_sightlines: usize,
    /// Directorio de salida para los archivos `insitu_NNNNNN.json`.
    /// Si es `None` se usa `<out_dir>/insitu/`.
    #[serde(default)]
    pub output_dir: Option<std::path::PathBuf>,
}

fn default_insitu_interval() -> u64 {
    0
}
fn default_pk_mesh() -> usize {
    32
}
fn default_fof_b() -> f64 {
    0.2
}
fn default_fof_min_part() -> usize {
    20
}
fn default_ab_smooth_r() -> f64 {
    5.0
}
fn default_sz_n_pixels() -> usize {
    256
}
fn default_lya_n_sightlines() -> usize {
    256
}

impl Default for InsituAnalysisSection {
    fn default() -> Self {
        Self {
            enabled: false,
            interval: default_insitu_interval(),
            pk_mesh: default_pk_mesh(),
            fof_b: default_fof_b(),
            fof_min_part: default_fof_min_part(),
            xi_bins: 0,
            pk_rsd_bins: 0,
            bispectrum_bins: 0,
            assembly_bias_enabled: false,
            assembly_bias_smooth_r: default_ab_smooth_r(),
            igm_temp_enabled: false,
            cm21_enabled: false,
            sz_enabled: false,
            sz_n_pixels: default_sz_n_pixels(),
            lya_enabled: false,
            lya_n_sightlines: default_lya_n_sightlines(),
            output_dir: None,
        }
    }
}

// ── DecompositionConfig ───────────────────────────────────────────────────────

/// Configuración de la descomposición de dominio SFC y balanceo de carga.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecompositionConfig {
    /// Si `true`, los cutpoints de la SFC se calculan por **prefix-sum de costes**
    /// de árbol en lugar de por conteo uniforme de partículas.
    ///
    /// Requiere que el solver sea Barnes-Hut (o compatible con `accelerations_with_costs`).
    /// En solvers que no devuelven costes, este flag se ignora silenciosamente.
    ///
    /// Default: `false` (compatible con comportamiento anterior).
    #[serde(default)]
    pub cost_weighted: bool,

    /// Factor de suavizado exponencial (EMA) para los costes por partícula entre pasos.
    /// `costs_new = alpha * costs_step + (1 - alpha) * costs_prev`.
    ///
    /// Valores típicos: 0.2–0.5. Default: `0.3`.
    #[serde(default = "default_ema_alpha")]
    pub ema_alpha: f64,
}

fn default_ema_alpha() -> f64 {
    0.3
}

impl Default for DecompositionConfig {
    fn default() -> Self {
        Self {
            cost_weighted: false,
            ema_alpha: default_ema_alpha(),
        }
    }
}
