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
    #[serde(default)]
    pub cosmology: CosmologySection,
    /// Sistema de unidades físicas (opcional; `enabled = false` por defecto).
    #[serde(default)]
    pub units: UnitsSection,
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
    /// MessagePack binario compacto (puro Rust, interoperable con Python/R/Julia).
    /// Requiere feature `msgpack` en `gadget-ng-io`.
    Msgpack,
    /// NetCDF-4 (HDF5 backend). Estándar en astrofísica/geofísica;
    /// legible directamente con `xarray`, `netCDF4`, Julia `NCDatasets`.
    /// Requiere feature `netcdf` en `gadget-ng-io` y `libnetcdf` en el sistema.
    Netcdf,
}

fn default_snapshot_format() -> SnapshotFormat {
    SnapshotFormat::Jsonl
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputSection {
    #[serde(default = "default_snapshot_format")]
    pub snapshot_format: SnapshotFormat,
    /// Guardar checkpoint cada N pasos (0 = desactivado).
    ///
    /// El checkpoint se escribe en `<out_dir>/checkpoint/` y permite reanudar
    /// la simulación con `gadget-ng stepping --resume <out_dir>`.
    #[serde(default)]
    pub checkpoint_interval: u64,
}

impl Default for OutputSection {
    fn default() -> Self {
        Self {
            snapshot_format:     default_snapshot_format(),
            checkpoint_interval: 0,
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
    /// `true` → intentar usar el solver GPU wgpu (requiere `--features gpu`).
    /// Si no hay GPU disponible en el host, se cae automáticamente al solver CPU.
    /// Con `false` (default) siempre se usa CPU.
    #[serde(default)]
    pub use_gpu: bool,

    /// `true` → árbol de Barnes-Hut distribuido: cada rango construye un árbol local
    /// a partir de sus partículas más los halos de los rangos vecinos (izquierdo y derecho
    /// en el eje x). La comunicación es punto-a-punto (`exchange_halos_by_x`), no
    /// Allgather global; escala a N > memoria de un nodo.
    ///
    /// Requiere `SolverKind::Tree` (o `TreePm`). Con `false` (default) se usa el
    /// Allgather global clásico.
    #[serde(default)]
    pub use_distributed_tree: bool,

    /// `true` → usar curva de Peano-Hilbert (Morton Z-order 3D) para la partición de
    /// dominio en lugar de slabs 1D en x.
    ///
    /// Requiere también `use_distributed_tree = true`. Con `false` (default)
    /// se usa la descomposición slab 1D original (retrocompatible).
    ///
    /// El balanceo dinámico se activa automáticamente: la descomposición SFC
    /// se recalcula cada `sfc_rebalance_interval` pasos.
    #[serde(default)]
    pub use_sfc: bool,

    /// Cada cuántos pasos se recalcula la partición SFC para balanceo dinámico.
    /// 0 = recalcular en todos los pasos (máximo balanceo, máximo overhead).
    /// Default: 10.
    #[serde(default = "default_sfc_rebalance")]
    pub sfc_rebalance_interval: u64,

    /// Factor de anchura de halo: `halo_width = halo_factor × slab_width`.
    /// Valores típicos: 0.5–1.0. Halos más anchos aumentan la precisión en bordes
    /// de dominio a costa de mayor comunicación y memoria local.
    #[serde(default = "default_halo_factor")]
    pub halo_factor: f64,
}

fn default_deterministic() -> bool {
    true
}

fn default_halo_factor() -> f64 {
    0.5
}

fn default_sfc_rebalance() -> u64 {
    10
}

impl Default for PerformanceSection {
    fn default() -> Self {
        Self {
            deterministic:           default_deterministic(),
            num_threads:             None,
            use_gpu:                 false,
            use_distributed_tree:    false,
            halo_factor:             default_halo_factor(),
            use_sfc:                 false,
            sfc_rebalance_interval:  default_sfc_rebalance(),
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

/// Parámetros cosmológicos (opcional; retrocompatible: `enabled = false`).
///
/// Activa la integración del factor de escala `a(t)` junto a las partículas.
/// Con `enabled = false` (default) el motor usa `dt` plano para drift y kick,
/// sin ninguna corrección cosmológica.
///
/// ## Unidades
///
/// `h0` es H₀ en **unidades internas de tiempo** (1/t_sim). Para simulaciones
/// cosmológicas en unidades naturales (L=Mpc/h, M=10¹⁰ M☉/h, V=km/s) el valor
/// habitual es `h0 ≈ 0.1` (≈ H₀ en unidades de km/s/kpc).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CosmologySection {
    /// `false` (default) → integración Newtoniana plana (sin factor de escala).
    /// `true` → integrar Friedmann y usar factores drift/kick cosmológicos.
    #[serde(default)]
    pub enabled: bool,
    /// Fracción de densidad de materia (sin dimensiones). Default: 0.3.
    #[serde(default = "default_omega_m")]
    pub omega_m: f64,
    /// Fracción de energía oscura (sin dimensiones). Default: 0.7.
    #[serde(default = "default_omega_lambda")]
    pub omega_lambda: f64,
    /// H₀ en unidades internas (1/t_sim). Default: 0.1.
    #[serde(default = "default_h0")]
    pub h0: f64,
    /// Factor de escala inicial. Default: 1.0 (z=0).
    /// Para simulaciones de alta redshift, p. ej. z=49 → `a_init = 0.02`.
    #[serde(default = "default_a_init")]
    pub a_init: f64,
}

fn default_omega_m() -> f64 {
    0.3
}

fn default_omega_lambda() -> f64 {
    0.7
}

fn default_h0() -> f64 {
    0.1
}

fn default_a_init() -> f64 {
    1.0
}

impl Default for CosmologySection {
    fn default() -> Self {
        Self {
            enabled: false,
            omega_m: default_omega_m(),
            omega_lambda: default_omega_lambda(),
            h0: default_h0(),
            a_init: default_a_init(),
        }
    }
}

// ── Sistema de unidades físicas ───────────────────────────────────────────────

/// G en kpc Msun⁻¹ (km/s)² (NIST 2018 redondeado a 5 cifras).
pub const G_KPC_MSUN_KMPS: f64 = 4.3009e-6;

/// Sistema de unidades físicas (opcional; retrocompatible: `enabled = false`).
///
/// Cuando `enabled = true`, la constante gravitacional interna se calcula
/// automáticamente como
///
/// ```text
/// G_int = G_kpc × mass_in_msun / length_in_kpc / velocity_in_km_s²
/// ```
///
/// donde `G_kpc = 4.3009 × 10⁻⁶ kpc Msun⁻¹ (km/s)²`.
///
/// # Ejemplo TOML
/// ```toml
/// [units]
/// enabled        = true
/// length_in_kpc  = 1.0       # 1 u.l. = 1 kpc
/// mass_in_msun   = 1.0e10    # 1 u.m. = 10¹⁰ M☉  (unidades GADGET clásicas)
/// velocity_in_km_s = 1.0     # 1 u.v. = 1 km/s
/// # G_int calculado = 4.3009e-6 × 1e10 / 1 / 1 = 4.3009e4
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnitsSection {
    /// `true` → calcular G internamente a partir de las escalas de unidad.
    /// `false` (default) → usar `simulation.gravitational_constant` sin cambios.
    #[serde(default)]
    pub enabled: bool,
    /// 1 unidad interna de longitud equivale a este número de kpc.
    #[serde(default = "default_unit_one")]
    pub length_in_kpc: f64,
    /// 1 unidad interna de masa equivale a este número de masas solares.
    #[serde(default = "default_unit_one")]
    pub mass_in_msun: f64,
    /// 1 unidad interna de velocidad equivale a este número de km/s.
    #[serde(default = "default_unit_one")]
    pub velocity_in_km_s: f64,
}

fn default_unit_one() -> f64 {
    1.0
}

impl Default for UnitsSection {
    fn default() -> Self {
        Self {
            enabled:          false,
            length_in_kpc:    1.0,
            mass_in_msun:     1.0,
            velocity_in_km_s: 1.0,
        }
    }
}

impl UnitsSection {
    /// G en unidades internas calculado a partir de las escalas.
    pub fn compute_g(&self) -> f64 {
        G_KPC_MSUN_KMPS * self.mass_in_msun
            / self.length_in_kpc
            / (self.velocity_in_km_s * self.velocity_in_km_s)
    }

    /// Unidad de tiempo interna expresada en Gyr
    /// (1 kpc / (km/s) = 0.97779 Gyr).
    pub fn time_unit_in_gyr(&self) -> f64 {
        0.97779 * self.length_in_kpc / self.velocity_in_km_s
    }

    /// Hubble time en unidades internas dado H₀ en km/s/Mpc.
    pub fn hubble_time(&self, h0_km_s_mpc: f64) -> f64 {
        // 1 Mpc = 1000 kpc; t_H = 1/H₀
        // H₀ en unidades internas = h0_km_s_mpc × (velocity_in_km_s / (1000 × length_in_kpc))
        let h0_int = h0_km_s_mpc * self.velocity_in_km_s / (1000.0 * self.length_in_kpc);
        1.0 / h0_int
    }
}

// ── RunConfig ─────────────────────────────────────────────────────────────────

impl RunConfig {
    pub fn softening_squared(&self) -> f64 {
        let e = self.simulation.softening;
        e * e
    }

    /// Constante gravitacional efectiva: si `units.enabled = true` se calcula desde
    /// las escalas de unidad; si no, se devuelve `simulation.gravitational_constant`.
    pub fn effective_g(&self) -> f64 {
        if self.units.enabled {
            self.units.compute_g()
        } else {
            self.simulation.gravitational_constant
        }
    }
}
