use serde::{Deserialize, Serialize};

/// Criterio de asignación del paso individual en block timesteps.
///
/// - `acceleration` (default) → `dt_i = η · sqrt(ε / |a_i|)` (criterio de Aarseth básico,
///   solo magnitud de aceleración). Retrocompatible con el comportamiento previo.
/// - `jerk` → `dt_i = η · sqrt(|a_i| / |ȧ_i|)` donde el jerk se aproxima como
///   `ȧ ≈ (a_i − a_prev) / dt_prev` mediante diferencia finita sobre el último paso
///   individual de la partícula. Más próximo al criterio de GADGET-2/4.
///   Si el jerk es cero o dt_prev ≤ 0, se degrada automáticamente al criterio `acceleration`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum TimestepCriterion {
    /// `dt_i = η · sqrt(ε / |a_i|)` (default, retrocompatible).
    #[default]
    Acceleration,
    /// `dt_i = η · sqrt(|a_i| / |ȧ_i|)` con jerk por diferencia finita.
    Jerk,
}

/// Parámetros de pasos temporales (opcional; retrocompatible: `hierarchical = false`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimestepSection {
    /// `false` (default) → paso global uniforme `dt` para todas las partículas.
    /// `true` → block timesteps al estilo GADGET-4: cada partícula elige su propio
    /// paso como potencia de 2 de `dt_base`, según el criterio de Aarseth.
    #[serde(default)]
    pub hierarchical: bool,
    /// Parámetro adimensional de Aarseth: `dt_i = eta * sqrt(eps / |a_i|)`.
    /// Valores típicos: 0.01–0.05. Por defecto 0.025.
    #[serde(default = "default_eta")]
    pub eta: f64,
    /// Número máximo de niveles de subdivisión (potencias de 2).
    /// Nivel `k` → paso `dt_base / 2^k`. Por defecto 6 (64 sub-pasos por paso base).
    #[serde(default = "default_max_level")]
    pub max_level: u32,
    /// Criterio de asignación del paso individual por partícula.
    /// Ver [`TimestepCriterion`]. Default: `acceleration`.
    #[serde(default)]
    pub criterion: TimestepCriterion,
    /// Paso mínimo absoluto (override del mínimo implícito `dt_base / 2^max_level`).
    /// `None` (default) → usar el mínimo implícito del nivel.
    #[serde(default)]
    pub dt_min: Option<f64>,
    /// Paso máximo absoluto (override del máximo implícito `dt_base`).
    /// `None` (default) → usar `dt_base` como máximo.
    #[serde(default)]
    pub dt_max: Option<f64>,
    /// Cota cosmológica del timestep por partícula: `dt_i ≤ κ_h · a / H(a)`.
    /// Solo se aplica en el path jerárquico con cosmología activa.
    /// `None` (default) → sin cota cosmológica en el rebinning jerárquico.
    /// Valor típico: 0.02–0.05.
    #[serde(default)]
    pub kappa_h: Option<f64>,
}

fn default_eta() -> f64 {
    0.025
}

fn default_max_level() -> u32 {
    6
}

impl Default for TimestepSection {
    fn default() -> Self {
        Self {
            hierarchical: false,
            eta: default_eta(),
            max_level: default_max_level(),
            criterion: TimestepCriterion::default(),
            dt_min: None,
            dt_max: None,
            kappa_h: None,
        }
    }
}
