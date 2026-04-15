use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunConfig {
    pub simulation: SimulationSection,
    pub initial_conditions: InitialConditionsSection,
    #[serde(default)]
    pub output: OutputSection,
    #[serde(default)]
    pub gravity: GravitySection,
    #[serde(default)]
    pub performance: PerformanceSection,
    #[serde(default)]
    pub timestep: TimestepSection,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationSection {
    pub dt: f64,
    pub num_steps: u64,
    pub softening: f64,
    #[serde(default = "default_g")]
    pub gravitational_constant: f64,
    pub particle_count: usize,
    pub box_size: f64,
    pub seed: u64,
}

fn default_g() -> f64 {
    1.0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InitialConditionsSection {
    pub kind: IcKind,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IcKind {
    Lattice,
    TwoBody {
        mass1: f64,
        mass2: f64,
        separation: f64,
    },
}

/// Parámetros del solver de gravedad (opcional en TOML; valores por defecto retrocompatibles).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GravitySection {
    #[serde(default = "default_solver_kind")]
    pub solver: SolverKind,
    /// Criterio Barnes–Hut `s/d < theta` (solo `barnes_hut`). Con `theta = 0` no se usa MAC (equivale a recorrido exhaustivo).
    #[serde(default = "default_theta")]
    pub theta: f64,
    /// Número de celdas por lado del grid PM (`pm`, `tree_pm`). El grid total es `pm_grid_size³`.
    /// Potencia de 2 recomendada para eficiencia FFT.
    #[serde(default = "default_pm_grid_size")]
    pub pm_grid_size: usize,
    /// Radio de splitting Gaussiano para el solver `tree_pm` (mismas unidades que posiciones).
    /// Si es ≤ 0 se calcula automáticamente como `2.5 × (box_size / pm_grid_size)`.
    #[serde(default = "default_r_split")]
    pub r_split: f64,
}

fn default_solver_kind() -> SolverKind {
    SolverKind::Direct
}

fn default_theta() -> f64 {
    0.5
}

fn default_pm_grid_size() -> usize {
    64
}

fn default_r_split() -> f64 {
    0.0
}

impl Default for GravitySection {
    fn default() -> Self {
        Self {
            solver: default_solver_kind(),
            theta: default_theta(),
            pm_grid_size: default_pm_grid_size(),
            r_split: default_r_split(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SolverKind {
    Direct,
    BarnesHut,
    /// Particle-Mesh (PM): FFT periódico 3D. Configurar también `pm_grid_size`.
    Pm,
    /// TreePM: Barnes-Hut (corto alcance, kernel erfc) + PM filtrado (largo alcance, kernel erf).
    /// Configurar `pm_grid_size` y opcionalmente `r_split`.
    TreePm,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SnapshotFormat {
    Jsonl,
    Hdf5,
    Bincode,
}

fn default_snapshot_format() -> SnapshotFormat {
    SnapshotFormat::Jsonl
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputSection {
    #[serde(default = "default_snapshot_format")]
    pub snapshot_format: SnapshotFormat,
}

impl Default for OutputSection {
    fn default() -> Self {
        Self {
            snapshot_format: default_snapshot_format(),
        }
    }
}

/// Parámetros de rendimiento (opcional; retrocompatible: defaults = serial determinista).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSection {
    /// `true` (default) → bucles seriales, paridad serial/MPI garantizada.
    /// `false` → Rayon activo (requiere build con `--features simd`); el orden de suma
    /// puede diferir → no se garantiza paridad bit-a-bit con el modo serial.
    #[serde(default = "default_deterministic")]
    pub deterministic: bool,
    /// Número de hilos Rayon. `None` → detecta automáticamente (número de CPUs lógicas).
    #[serde(default)]
    pub num_threads: Option<usize>,
}

fn default_deterministic() -> bool {
    true
}

impl Default for PerformanceSection {
    fn default() -> Self {
        Self {
            deterministic: default_deterministic(),
            num_threads: None,
        }
    }
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
        }
    }
}

impl RunConfig {
    pub fn softening_squared(&self) -> f64 {
        let e = self.simulation.softening;
        e * e
    }
}
