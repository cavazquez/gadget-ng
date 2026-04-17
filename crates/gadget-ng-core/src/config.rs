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
    /// Integrador temporal usado por el motor.
    ///
    /// - `leapfrog` (default): leapfrog KDK clásico, orden 2, 2 force evals/step.
    /// - `yoshida4`: composición simpléctica de Yoshida (1990), orden 4, 4
    ///   force evals/step. No compatible con `[timestep] hierarchical = true`.
    #[serde(default)]
    pub integrator: IntegratorKind,
}

/// Selección del integrador temporal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum IntegratorKind {
    /// Leapfrog KDK (default), 2º orden, 2 force evals/step.
    #[default]
    Leapfrog,
    /// Yoshida composición simpléctica 4º orden, 4 force evals/step.
    Yoshida4,
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
    /// Retícula cúbica regular con perturbación aleatoria pequeña.
    /// Requiere `particle_count = n³`.
    Lattice,
    /// Sistema de 2 cuerpos en órbita circular.
    TwoBody {
        mass1: f64,
        mass2: f64,
        separation: f64,
    },
    /// Esfera de Plummer con posiciones muestreadas de la CDF de masa
    /// y velocidades Gaussianas escaladas para el equilibrio virial.
    ///
    /// ```toml
    /// [initial_conditions]
    /// kind = { plummer = { a = 1.0 } }
    /// ```
    Plummer {
        /// Radio de escala de Plummer `a` (en unidades internas).
        #[serde(default = "default_plummer_a")]
        a: f64,
    },
    /// Esfera sólida uniforme con partículas en reposo (v = 0).
    ///
    /// Benchmark clásico de colapso gravitacional frío (cold collapse):
    /// la esfera colapsa libremente y virializa al cabo de ~3 tiempos de caída libre.
    ///
    /// ```toml
    /// [initial_conditions]
    /// kind = { uniform_sphere = { r = 1.0 } }
    /// ```
    UniformSphere {
        /// Radio de la esfera sólida (en unidades internas).
        #[serde(default = "default_sphere_r")]
        r: f64,
    },
}

fn default_plummer_a() -> f64 {
    1.0
}

fn default_sphere_r() -> f64 {
    1.0
}

/// Parámetros del solver de gravedad (opcional en TOML; valores por defecto retrocompatibles).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GravitySection {
    #[serde(default = "default_solver_kind")]
    pub solver: SolverKind,
    /// Criterio Barnes–Hut `s/d < theta` (solo `barnes_hut`). Con `theta = 0` no se usa MAC (equivale a recorrido exhaustivo).
    #[serde(default = "default_theta")]
    pub theta: f64,
    /// Orden de la expansión multipolar para Barnes–Hut (solo `barnes_hut`):
    /// - `1` → monopolo únicamente
    /// - `2` → monopolo + cuadrupolo
    /// - `3` → monopolo + cuadrupolo + octupolo (default, máxima precisión)
    ///
    /// Útil para benchmarks de ablación que cuantifican la contribución de cada término.
    #[serde(default = "default_multipole_order")]
    pub multipole_order: u8,
    /// Criterio de apertura del árbol Barnes–Hut (solo `barnes_hut`):
    /// - `"geometric"` (default) → abre el nodo cuando `s/d ≥ theta` (criterio clásico)
    /// - `"relative"` → abre cuando el error de truncamiento estimado supera `err_tol_force_acc`
    ///   (equivalente a `TypeOfOpeningCriterion=1` de GADGET-4)
    #[serde(default = "default_opening_criterion")]
    pub opening_criterion: OpeningCriterion,
    /// Tolerancia de error de fuerza para el criterio de apertura relativo.
    /// GADGET-4 usa `ErrTolForceAcc ≈ 0.0025`. Solo se usa cuando `opening_criterion = "relative"`.
    #[serde(default = "default_err_tol_force_acc")]
    pub err_tol_force_acc: f64,
    /// Si `true`, aplica el mismo softening Plummer en los términos cuadrupolar y octupolar
    /// (reemplaza `r²` por `r² + ε²` en los denominadores, coherente con el monopolo).
    ///
    /// La inconsistencia de softening (monopolo suavizado, quad/oct bare) es la causa principal
    /// del empeoramiento de precisión en distribuciones concentradas con criterio geométrico.
    ///
    /// `false` (default) → comportamiento clásico/retrocompatible.
    /// `true` → corrección física necesaria para sistemas con `r_núcleo ~ ε`.
    #[serde(default)]
    pub softened_multipoles: bool,
    /// Softening aplicado al **estimador del MAC relativo** (no al cálculo de fuerza).
    ///
    /// - `"bare"` (default) → el estimador usa `|Q|_F / d⁵` (retrocompatible).
    /// - `"consistent"` → usa `|Q|_F / (d² + ε²)^{5/2}`, coherente con el monopolo
    ///   suavizado. Evita sobre-estimar el error de truncamiento cuando `d ~ ε`
    ///   y reduce la apertura espuria de nodos en el núcleo.
    ///
    /// Solo surte efecto cuando `opening_criterion = "relative"`.
    #[serde(default)]
    pub mac_softening: MacSoftening,
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

fn default_multipole_order() -> u8 {
    3
}

fn default_opening_criterion() -> OpeningCriterion {
    OpeningCriterion::Geometric
}

fn default_err_tol_force_acc() -> f64 {
    0.005
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
            multipole_order: default_multipole_order(),
            opening_criterion: default_opening_criterion(),
            err_tol_force_acc: default_err_tol_force_acc(),
            softened_multipoles: false,
            mac_softening: MacSoftening::default(),
            pm_grid_size: default_pm_grid_size(),
            r_split: default_r_split(),
        }
    }
}

/// Softening del estimador del MAC relativo.
///
/// Controla si el término multipolar que entra en el estimador de error usa
/// el denominador bare `d⁵` o el denominador softened-consistent `(d² + ε²)^{5/2}`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum MacSoftening {
    /// `|Q|_F / d⁵` (retrocompatible, por defecto).
    #[default]
    Bare,
    /// `|Q|_F / (d² + ε²)^{5/2}` (coherente con el monopolo softened).
    Consistent,
}

/// Criterio de apertura del árbol Barnes–Hut.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum OpeningCriterion {
    /// Criterio geométrico clásico: abre si `s/d ≥ theta`.
    #[default]
    Geometric,
    /// Criterio relativo (GADGET-4 `TypeOfOpeningCriterion=1`): abre si el error de
    /// truncamiento estimado supera `err_tol_force_acc`.
    Relative,
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
    #[serde(default)]
    pub checkpoint_interval: u64,
    /// Guardar snapshot de partículas cada N pasos en `<out_dir>/frames/snap_{step:06}/`
    /// (0 = desactivado).  Útil para generar animaciones cuadro a cuadro.
    #[serde(default)]
    pub snapshot_interval: u64,
}

impl Default for OutputSection {
    fn default() -> Self {
        Self {
            snapshot_format: default_snapshot_format(),
            checkpoint_interval: 0,
            snapshot_interval: 0,
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

    /// Si `true`, fuerza el path `Allgather O(N·P)` incluso en modo multirank.
    ///
    /// Por defecto (`false`), en modo multirank con solver BarnesHut sin cosmología
    /// ni integrador jerárquico, el motor usa SFC + LET (comunicación selectiva).
    /// Activar este flag es útil para comparar contra el baseline Allgather o para
    /// validación paper-grade del resultado serial.
    #[serde(default)]
    pub force_allgather_fallback: bool,

    /// `true` (default) → usar alltoallv no-bloqueante (Isend/Irecv) para solapar
    /// la evaluación de fuerzas locales con la comunicación LET.
    /// `false` → alltoallv bloqueante (Fase 8 original); útil para comparación.
    ///
    /// En modo serial el valor no tiene efecto: ambos caminos son equivalentes.
    #[serde(default = "default_let_nonblocking")]
    pub let_nonblocking: bool,

    /// `true` (default) → construir un octree sobre los `RemoteMultipoleNode`
    /// importados (`LetTree`) y aplicar fuerzas remotas en O(N_local log N_let).
    /// `false` → loop plano O(N_local × N_let) (baseline Fase 9).
    ///
    /// Solo tiene efecto cuando `use_sfc = true` y el número de nodos LET
    /// importados supera `let_tree_threshold`.
    #[serde(default = "default_use_let_tree")]
    pub use_let_tree: bool,

    /// Umbral mínimo de nodos LET para activar el `LetTree`.
    /// Si los nodos importados son `≤ let_tree_threshold`, se usa el loop plano
    /// (el árbol no compensa su overhead de construcción con pocos nodos).
    /// Default: 64.
    #[serde(default = "default_let_tree_threshold")]
    pub let_tree_threshold: usize,

    /// Número máximo de `RemoteMultipoleNode`s por hoja del `LetTree`.
    /// Valores menores → árbol más profundo, más llamadas MAC, mayor precisión.
    /// Valores mayores → árbol más plano, menos overhead de build, menor precisión.
    /// Default: 8.
    #[serde(default = "default_let_tree_leaf_max")]
    pub let_tree_leaf_max: usize,

    /// Factor multiplicativo sobre `theta` para la exportación LET.
    ///
    /// Controla qué tan agresivamente se poda el árbol local al exportar nodos LET
    /// hacia cada rank remoto. El `theta` efectivo de exportación es:
    ///
    /// ```text
    /// theta_export = theta * let_theta_export_factor   (si let_theta_export_factor > 0)
    /// theta_export = theta                              (si let_theta_export_factor == 0, default)
    /// ```
    ///
    /// - `0.0` (default): usa el mismo `theta` que el walk. Sin cambio respecto a Fases 9-11.
    /// - `> 1.0`: exporta nodos más gruesos → menos nodos, menos bytes enviados, mayor
    ///   error de truncación en el receptor. Ejemplo: `1.4` con `theta = 0.5` → `theta_export = 0.7`.
    /// - `< 1.0`: más conservador que el walk (más nodos exportados, mayor precisión; útil
    ///   solo para debug o validación).
    ///
    /// Solo tiene efecto cuando `use_sfc = true`.
    #[serde(default)]
    pub let_theta_export_factor: f64,

    /// Curva SFC para domain decomposition.
    /// `"morton"` (default) → Z-order 63 bits. Retrocompatible con Fases 8-12.
    /// `"hilbert"` → Peano-Hilbert 3D (Skilling 2004), mejor localidad espacial.
    /// Solo tiene efecto cuando `use_sfc = true` (path SFC+LET).
    #[serde(default)]
    pub sfc_kind: SfcKind,
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

fn default_let_nonblocking() -> bool {
    true
}

fn default_use_let_tree() -> bool {
    true
}

fn default_let_tree_threshold() -> usize {
    64
}

fn default_let_tree_leaf_max() -> usize {
    8
}

/// Curva SFC (Space-Filling Curve) para domain decomposition.
///
/// Controla qué curva se usa para ordenar partículas y construir los cutpoints
/// de la partición de dominio en el path SFC+LET.
///
/// - `"morton"` (default): Z-order 3D, 21 bits/eje. Retrocompatible con Fases 8-12.
/// - `"hilbert"`: Peano-Hilbert 3D (algoritmo Skilling 2004), misma precisión.
///   Mejor localidad espacial que Morton para distribuciones no uniformes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum SfcKind {
    #[default]
    Morton,
    Hilbert,
}

impl Default for PerformanceSection {
    fn default() -> Self {
        Self {
            deterministic: default_deterministic(),
            num_threads: None,
            use_gpu: false,
            use_distributed_tree: false,
            halo_factor: default_halo_factor(),
            use_sfc: false,
            sfc_rebalance_interval: default_sfc_rebalance(),
            force_allgather_fallback: false,
            let_nonblocking: default_let_nonblocking(),
            use_let_tree: default_use_let_tree(),
            let_tree_threshold: default_let_tree_threshold(),
            let_tree_leaf_max: default_let_tree_leaf_max(),
            let_theta_export_factor: 0.0,
            sfc_kind: SfcKind::Morton,
        }
    }
}

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
            enabled: false,
            length_in_kpc: 1.0,
            mass_in_msun: 1.0,
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
