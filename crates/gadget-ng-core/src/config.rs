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
    /// Configuración de descomposición de dominio (opcional; balanceo por coste de árbol).
    #[serde(default)]
    pub decomposition: DecompositionConfig,
    /// Análisis in-situ durante el loop `stepping` (opcional; desactivado por defecto).
    #[serde(default)]
    pub insitu_analysis: InsituAnalysisSection,
    /// Módulo SPH cosmológico (Phase G2; opcional; desactivado por defecto).
    #[serde(default)]
    pub sph: SphSection,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationSection {
    pub dt: f64,
    pub num_steps: u64,
    pub softening: f64,
    /// Si `true`, `softening` se interpreta como ε_phys constante en unidades físicas.
    /// En cada paso el softening comóvil efectivo es `ε_com = softening / a`, de modo que
    /// la longitud física de suavizado permanece fija mientras el universo se expande.
    /// Por defecto `false` (comportamiento legacy: softening comóvil constante).
    /// Solo tiene efecto cuando `[cosmology] enabled = true`.
    #[serde(default)]
    pub physical_softening: bool,
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
    /// Retícula cúbica con perturbaciones gaussianas de posición y velocidad.
    ///
    /// Diseñada para condiciones iniciales cosmológicas: las partículas se colocan
    /// sobre una cuadrícula regular `⌈N^{1/3}⌉³` y reciben perturbaciones Gaussianas.
    /// Con `velocity_amplitude = 0.0` las partículas están en reposo comóvil (p = 0),
    /// adecuado para simulaciones de alta redshift.
    ///
    /// ```toml
    /// [initial_conditions]
    /// kind = { perturbed_lattice = { amplitude = 0.1, velocity_amplitude = 0.0 } }
    /// ```
    PerturbedLattice {
        /// Amplitud de la perturbación de posición como fracción del espaciado de la retícula.
        /// Ejemplo: `0.1` → perturbación de ±10% del spacing de la celda.
        #[serde(default = "default_perturb_amplitude")]
        amplitude: f64,
        /// Amplitud de las velocidades peculiares iniciales en unidades de `H0 * box_size`.
        /// `0.0` (default) = reposo comóvil completo.
        /// Las velocidades se almacenan como momentum canónico `p = a_init * v_peculiar`.
        #[serde(default)]
        velocity_amplitude: f64,
    },
    /// Condiciones iniciales de Zel'dovich (1LPT) a partir de un campo gaussiano en Fourier.
    ///
    /// Las partículas se desplazan desde una retícula regular usando la aproximación de
    /// Zel'dovich: `x = q + Ψ(q)`, donde `Ψ` es el campo de desplazamiento generado
    /// a partir de un espectro de potencia `P(k) ∝ |k|^spectral_index`.
    ///
    /// Los momenta se establecen para ser consistentes con el crecimiento lineal:
    /// `p = a²·f(a)·H(a)·Ψ`, donde `f(a) ≈ Ω_m(a)^0.55`.
    ///
    /// Requiere `cosmology.enabled = true` para las velocidades físicas.
    /// `particle_count` debe ser igual a `grid_size³`.
    ///
    /// ## Unidades y normalización
    ///
    /// Con `transfer = "power_law"` (default): `amplitude` es la amplitud adimensional
    /// del espectro `P(k) = amplitude² · |n|^spectral_index` en unidades de grid.
    ///
    /// Con `transfer = "eisenstein_hu"` y `sigma8`: la amplitud se calcula para que
    /// `σ(8 Mpc/h) = sigma8`. Requiere `box_size_mpc_h` para la conversión de k.
    ///
    /// ## Reproducibilidad
    ///
    /// El campo se genera de forma determinista a partir de `seed`. En MPI, todos los
    /// rangos generan el campo completo y extraen su rango de `gid`.
    ///
    /// ## Configuración legacy (Fase 26, sigue funcionando sin cambios)
    ///
    /// ```toml
    /// [initial_conditions]
    /// kind = { zeldovich = { seed = 42, grid_size = 32, spectral_index = -2.0, amplitude = 1.0e-4 } }
    /// ```
    ///
    /// ## Configuración con Eisenstein–Hu y σ₈ (Fase 27)
    ///
    /// ```toml
    /// [initial_conditions]
    /// kind = { zeldovich = { seed = 42, grid_size = 32, spectral_index = 0.965,
    ///     transfer = "eisenstein_hu", sigma8 = 0.8,
    ///     omega_b = 0.049, h = 0.674, t_cmb = 2.7255,
    ///     box_size_mpc_h = 100.0 } }
    /// ```
    Zeldovich {
        /// Semilla del generador de números aleatorios (reproducibilidad).
        #[serde(default = "default_zel_seed")]
        seed: u64,
        /// Lado de la retícula: `grid_size³` debe coincidir con `particle_count`.
        grid_size: usize,
        /// Índice espectral primordial `n_s`: `P(k) ∝ k^n_s`.
        /// Planck18: 0.965. Valores de prueba: −2, −1, 0 (Harrison–Zel'dovich = 1).
        #[serde(default = "default_spectral_index")]
        spectral_index: f64,
        /// Amplitud adimensional del espectro (usada cuando `sigma8 = None` y `transfer = PowerLaw`).
        /// Valores menores garantizan régimen lineal: p. ej. `1e-4` da `Ψ_rms/d ≈ 0.01–0.1`.
        #[serde(default = "default_zel_amplitude")]
        amplitude: f64,

        // ── Campos Fase 27 (todos con default para retrocompatibilidad) ──
        /// Tipo de función de transferencia a aplicar al espectro.
        /// `"power_law"` (default) = comportamiento legacy; `"eisenstein_hu"` = EH98 no-wiggle.
        #[serde(default)]
        transfer: TransferKind,
        /// Si `Some(sigma8_target)`, la amplitud se calcula para que `σ(8 Mpc/h) = sigma8_target`.
        /// Sobreescribe `amplitude` cuando se usa con `transfer = "eisenstein_hu"`.
        #[serde(default)]
        sigma8: Option<f64>,
        /// Densidad de bariones Ω_b para la función de transferencia E-H.
        /// Default: 0.049 (Planck18).
        #[serde(default = "default_omega_b")]
        omega_b: f64,
        /// Parámetro de Hubble adimensional h = H₀/(100 km/s/Mpc).
        /// Distinto de `cosmology.h0` (que está en unidades internas de tiempo).
        /// Default: 0.674 (Planck18).
        #[serde(default = "default_h_dimless")]
        h: f64,
        /// Temperatura del CMB en Kelvin. Presente para completitud (no usada en no-wiggle).
        /// Default: 2.7255 K.
        #[serde(default = "default_t_cmb")]
        t_cmb: f64,
        /// Tamaño de la caja en Mpc/h. Requerido cuando `transfer = "eisenstein_hu"`.
        /// No modifica el sistema de unidades interno de gadget-ng; solo se usa para
        /// convertir los modos del grid a k [h/Mpc] para T(k) y σ₈.
        #[serde(default)]
        box_size_mpc_h: Option<f64>,
        /// Si `true`, activa correcciones de segundo orden (2LPT).
        ///
        /// El desplazamiento total es `x = q + Ψ¹ + (D₂/D₁²)·Ψ²`, donde Ψ² se obtiene
        /// resolviendo la ecuación de Poisson de segundo orden en k-space.
        ///
        /// Las velocidades incluyen la contribución de segundo orden:
        /// `p = a²·H·[f₁·Ψ¹ + f₂·(D₂/D₁²)·Ψ²]`
        ///
        /// Default: `false` (comportamiento 1LPT legacy, retrocompatible).
        #[serde(default)]
        use_2lpt: bool,

        // ── Campo Fase 40 (reemplaza rescale_to_a_init de Fase 37) ──
        /// Convención física de normalización de amplitud del campo inicial.
        ///
        /// - **`Legacy` (default)**: `σ₈` se aplica directamente en
        ///   `a_init`. La amplitud del campo queda referida al tiempo inicial.
        ///   Compatible bit-a-bit con Fase 26–28/37 (`rescale_to_a_init=false`).
        /// - **`Z0Sigma8` (Fase 40)**: `σ₈` queda referido a `a=1` (convención
        ///   estándar CAMB/CLASS). Los desplazamientos se reducen por
        ///   `s = D(a_init)/D(1)`; el 2LPT se reduce por `s²` (ya que crece
        ///   como `D²`). Las velocidades heredan el factor porque son
        ///   lineales en Ψ¹ y Ψ². Físicamente consistente con σ₈(z=0).
        ///   Equivalente al viejo `rescale_to_a_init=true`.
        ///
        /// Consultar `docs/reports/2026-04-phase40-physical-ics-normalization.md`
        /// para la derivación matemática, auditoría y recomendación final.
        #[serde(default)]
        normalization_mode: NormalizationMode,
    },
}

/// Convención de normalización de amplitud de las ICs cosmológicas.
///
/// Introducida en Fase 40 para reemplazar el flag experimental
/// `rescale_to_a_init` de Fase 37 por una API explícita y tipada.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum NormalizationMode {
    /// `σ₈` aplicado directamente en `a_init`. Comportamiento histórico
    /// de Fase 26–28. Bit-idéntico al viejo `rescale_to_a_init = false`.
    #[default]
    Legacy,
    /// `σ₈` referido a `a=1` (convención CAMB/CLASS); los desplazamientos
    /// y velocidades se reescalan por `s = D(a_init)/D(1)` (y `s²` para 2LPT).
    /// Equivalente al viejo `rescale_to_a_init = true`.
    Z0Sigma8,
}

/// Tipo de función de transferencia cosmológica para el espectro de potencia inicial.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum TransferKind {
    /// Ley de potencia pura `P(k) ∝ k^n_s` (sin función de transferencia).
    /// Comportamiento legacy de Fase 26. T(k) = 1 para todos los modos.
    #[default]
    PowerLaw,
    /// Eisenstein–Hu 1998, aproximación sin-wiggle (no-wiggle).
    /// Requiere `box_size_mpc_h` para la conversión de k.
    EisensteinHu,
}

fn default_plummer_a() -> f64 {
    1.0
}

fn default_sphere_r() -> f64 {
    1.0
}

fn default_perturb_amplitude() -> f64 {
    0.1
}

fn default_zel_seed() -> u64 {
    42
}

fn default_spectral_index() -> f64 {
    -2.0
}

fn default_zel_amplitude() -> f64 {
    1.0e-4
}

fn default_omega_b() -> f64 {
    0.049 // Planck 2018
}

fn default_h_dimless() -> f64 {
    0.674 // Planck 2018
}

fn default_t_cmb() -> f64 {
    2.7255 // K
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
    /// `true` → usa el path PM distribuido (Fase 19): cada rank deposita su
    /// contribución local al grid, un `allreduce_sum` sobre el grid nm³ reemplaza
    /// el `allgather` O(N·P) de partículas, y todos los ranks resuelven Poisson
    /// de forma independiente (resultado idéntico al ser determinista).
    ///
    /// Solo tiene efecto cuando `cosmology.periodic = true` y `solver = "pm"`.
    /// En `P=1` (serial) el comportamiento es idéntico al path clásico.
    #[serde(default)]
    pub pm_distributed: bool,

    /// Activa el path PM de Fase 20: slab decomposition real en Z con FFT
    /// distribuida mediante alltoall transposes.
    ///
    /// Requisitos: `cosmology.periodic = true`, `solver = "pm"`,
    /// `pm_grid_size % n_ranks == 0`.
    ///
    /// Para `P = 1` el resultado es bit-a-bit idéntico al solver serial.
    /// Cada alltoall transfiere O(nm³/P) datos por rank (P× menos que `pm_distributed`).
    #[serde(default)]
    pub pm_slab: bool,

    /// Activa el path TreePM distribuido mínimo viable (Fase 21).
    ///
    /// Combina:
    /// - **Largo alcance**: PM slab distribuido (Fase 20) con filtro Gaussiano.
    /// - **Corto alcance**: árbol local + halos de partículas en z, con `minimum_image`
    ///   periódico y kernel `erfc(r / (√2·r_s))`.
    ///
    /// Requisitos: `solver = "tree_pm"`, `cosmology.periodic = true`,
    /// `pm_grid_size % n_ranks == 0`.
    ///
    /// **Limitación documentada**: el halo de corto alcance es 1D en z. Las interacciones
    /// que cruzan fronteras x,y entre slabs no están cubiertas. Para un TreePM completo
    /// tipo GADGET se requeriría halo volumétrico SFC 3D.
    ///
    /// Para `P = 1`, el resultado es físicamente equivalente al path serial con allgather.
    #[serde(default)]
    pub treepm_slab: bool,

    /// Activa el halo volumétrico 3D periódico para el árbol SR (Fase 22).
    ///
    /// Requiere `treepm_slab = true`. En lugar del halo 1D-z, calcula el AABB real
    /// de las partículas de cada rank y usa `min_dist2_to_aabb_3d_periodic` para
    /// decidir qué partículas enviar.
    ///
    /// **Para Z-slab uniforme**: produce el mismo conjunto de halos que el 1D-z,
    /// con overhead mínimo. **Para descomposición en octantes o SFC**: cubre
    /// interacciones diagonales periódicas que el halo 1D-z omitiría.
    ///
    /// Corrección del bug de `exchange_halos_sfc`: usa coordenadas con wrap
    /// periódico explícito en vez de coordenadas absolutas.
    #[serde(default)]
    pub treepm_halo_3d: bool,

    /// Activa el dominio 3D/SFC para el árbol de corto alcance (Fase 23).
    ///
    /// Desacopla el SR del slab-z del PM: las partículas se distribuyen por
    /// `SfcDecomposition` (Morton/Hilbert) para el cálculo del árbol SR, mientras
    /// el PM largo alcance sigue usando slab-z sin cambios.
    ///
    /// ## Arquitectura dual
    ///
    /// - **SR domain**: SFC (Morton/Hilbert). `exchange_halos_3d_periodic` es el
    ///   mecanismo activo para cubrir interacciones de corto alcance.
    /// - **PM domain**: z-slab (sin cambios respecto a Fase 20/21).
    /// - **Sincronización PM↔SR**: por cada evaluación de fuerza, el PM clona las
    ///   partículas, migra a z-slab, computa fuerzas y retorna al dominio SFC.
    ///
    /// Requiere `solver = "tree_pm"`, `cosmology.periodic = true`,
    /// `pm_grid_size % n_ranks == 0`.
    ///
    /// Implica `treepm_halo_3d = true` (halo 3D periódico es necesario para SR-SFC).
    #[serde(default)]
    pub treepm_sr_sfc: bool,

    /// Fase 24: scatter/gather PM mínimo entre dominio SFC y slabs.
    ///
    /// Reemplaza la sincronización PM↔SR de Fase 23 (`clone → migrate → PM → back-migrate
    /// → HashMap`) por un protocolo scatter/gather explícito que envía solo los datos
    /// mínimos necesarios:
    ///
    /// - **Scatter**: `(global_id, position, mass)` → 40 bytes/partícula
    /// - **Gather**: `(global_id, acc_pm)` → 32 bytes/partícula
    ///
    /// Total round-trip: ~72 bytes/partícula vs ~176 bytes del path de Fase 23.
    ///
    /// Las partículas verdaderas permanecen en el dominio SFC sin ningún clone.
    /// El PM slab actúa como servicio de campo: recibe contribuciones de densidad
    /// y devuelve aceleraciones PM, sin poseer partículas.
    ///
    /// Solo activo si `treepm_sr_sfc = true`. Requiere `solver = "tree_pm"`.
    #[serde(default)]
    pub treepm_pm_scatter_gather: bool,
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
            pm_distributed: false,
            pm_slab: false,
            treepm_slab: false,
            treepm_halo_3d: false,
            treepm_sr_sfc: false,
            treepm_pm_scatter_gather: false,
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

    /// `true` → intentar usar el solver PM CUDA (requiere `--features cuda` y nvcc+cuFFT).
    /// Si el dispositivo CUDA no está disponible en el host o el crate se compiló sin
    /// toolchain CUDA, se cae automáticamente al solver CPU sin error fatal.
    /// Solo tiene efecto cuando `[gravity] solver = "Pm"`.
    /// Con `false` (default) siempre se usa el solver PM CPU.
    #[serde(default)]
    pub use_gpu_cuda: bool,

    /// `true` → intentar usar el solver PM HIP/ROCm (requiere `--features hip` y hipcc+rocFFT).
    /// Misma semántica de degradación elegante que `use_gpu_cuda`.
    /// `use_gpu_cuda` tiene precedencia si ambos están en `true`.
    /// Solo tiene efecto cuando `[gravity] solver = "Pm"`.
    #[serde(default)]
    pub use_gpu_hip: bool,

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

    /// Umbral de desbalance de carga para forzar un rebalanceo inmediato.
    ///
    /// Si `max(walk_ns) / min(walk_ns) > rebalance_imbalance_threshold`,
    /// se fuerza un rebalanceo SFC en el siguiente paso, independientemente
    /// de `sfc_rebalance_interval`.
    ///
    /// - `0.0` (default): criterio por coste desactivado; sólo se rebalancea
    ///   cada `sfc_rebalance_interval` pasos.
    /// - Valores típicos: `1.3` (30 % de desbalance relativo), `1.5`, `2.0`.
    /// - Solo tiene efecto cuando `use_sfc = true`.
    #[serde(default)]
    pub rebalance_imbalance_threshold: f64,
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
            use_gpu_cuda: false,
            use_gpu_hip: false,
            rebalance_imbalance_threshold: 0.0,
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
    /// Condiciones de contorno periódicas (Fase 18).
    ///
    /// `false` (default) → caja no periódica. Las fuerzas usan árbol Barnes-Hut o
    ///   distancias euclídeas sin imagen mínima; las posiciones no se envuelven.
    ///
    /// `true` → caja periódica. Requiere `gravity.solver = "pm"` o `"tree_pm"`:
    ///   el solver PM usa CIC+FFT periódica para calcular las fuerzas correctamente.
    ///   Las posiciones se envuelven a `[0, box_size)` tras cada paso de drift.
    ///   Las fuerzas de árbol (BarnesHut) NO son periódicas; usar PM o TreePM.
    #[serde(default)]
    pub periodic: bool,
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
    /// Si `true`, el motor calcula automáticamente G a partir de `omega_m` y `h0`
    /// usando la condición de Friedmann `G = 3·Ω_m·H₀²/(8π)` (ρ̄_m = 1).
    ///
    /// Requiere `enabled = true`. Cuando está activo, el campo
    /// `simulation.gravitational_constant` se ignora para el modo cosmológico
    /// y se emite un `info!` con el valor calculado. Si el campo
    /// `simulation.gravitational_constant` difiere del G auto en más de 1 %,
    /// también se emite un `warn!` de inconsistencia.
    ///
    /// Default: `false` (retrocompatible — se usa `simulation.gravitational_constant`).
    ///
    /// # Ejemplo TOML
    /// ```toml
    /// [cosmology]
    /// enabled  = true
    /// auto_g   = true
    /// omega_m  = 0.315
    /// omega_lambda = 0.685
    /// h0       = 0.1
    /// a_init   = 0.02
    /// ```
    #[serde(default)]
    pub auto_g: bool,
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
            periodic: false,
            omega_m: default_omega_m(),
            omega_lambda: default_omega_lambda(),
            h0: default_h0(),
            a_init: default_a_init(),
            auto_g: false,
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

// ── SphSection ───────────────────────────────────────────────────────────────

/// Tipo de enfriamiento radiativo para partículas de gas.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum CoolingKind {
    /// Sin enfriamiento (default).
    #[default]
    None,
    /// Enfriamiento atómico H+He: `Λ(T) = Λ₀ · T^β` con floor en `t_floor_k`.
    AtomicHHe,
}

/// Configuración del módulo SPH cosmológico (Phase G2).
///
/// Añadir `[sph]` al TOML:
///
/// ```toml
/// [sph]
/// enabled       = true
/// gamma         = 1.6667
/// alpha_visc    = 1.0
/// n_neigh       = 32
/// cooling       = "atomic_h_he"
/// t_floor_k     = 1e4
/// gas_fraction  = 0.1
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SphSection {
    /// Activa el módulo SPH (default: `false`).
    #[serde(default)]
    pub enabled: bool,
    /// Índice adiabático γ (default: 5/3).
    #[serde(default = "default_gamma")]
    pub gamma: f64,
    /// Parámetro de viscosidad artificial α (default: 1.0).
    #[serde(default = "default_alpha_visc")]
    pub alpha_visc: f64,
    /// Número objetivo de vecinos SPH (default: 32).
    #[serde(default = "default_n_neigh")]
    pub n_neigh: usize,
    /// Tipo de enfriamiento radiativo.
    #[serde(default)]
    pub cooling: CoolingKind,
    /// Temperatura de floor en Kelvin (default: 10⁴ K).
    #[serde(default = "default_t_floor_k")]
    pub t_floor_k: f64,
    /// Fracción de partículas inicializadas como gas (default: 0.0).
    #[serde(default)]
    pub gas_fraction: f64,
}

fn default_gamma() -> f64 { 5.0 / 3.0 }
fn default_alpha_visc() -> f64 { 1.0 }
fn default_n_neigh() -> usize { 32 }
fn default_t_floor_k() -> f64 { 1e4 }

impl Default for SphSection {
    fn default() -> Self {
        Self {
            enabled: false,
            gamma: default_gamma(),
            alpha_visc: default_alpha_visc(),
            n_neigh: default_n_neigh(),
            cooling: CoolingKind::None,
            t_floor_k: default_t_floor_k(),
            gas_fraction: 0.0,
        }
    }
}

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
    /// Directorio de salida para los archivos `insitu_NNNNNN.json`.
    /// Si es `None` se usa `<out_dir>/insitu/`.
    #[serde(default)]
    pub output_dir: Option<std::path::PathBuf>,
}

fn default_insitu_interval() -> u64 { 0 }
fn default_pk_mesh() -> usize { 32 }
fn default_fof_b() -> f64 { 0.2 }
fn default_fof_min_part() -> usize { 20 }

impl Default for InsituAnalysisSection {
    fn default() -> Self {
        Self {
            enabled: false,
            interval: default_insitu_interval(),
            pk_mesh: default_pk_mesh(),
            fof_b: default_fof_b(),
            fof_min_part: default_fof_min_part(),
            xi_bins: 0,
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

// ── RunConfig ─────────────────────────────────────────────────────────────────

impl RunConfig {
    pub fn softening_squared(&self) -> f64 {
        let e = self.simulation.softening;
        e * e
    }

    /// Constante gravitacional efectiva en el modo de integración actual.
    ///
    /// Prioridad (de mayor a menor):
    ///
    /// 1. `units.enabled = true` → `G_int = G_KPC_MSUN_KMPS × mass/length/v²`
    /// 2. `cosmology.enabled = true && cosmology.auto_g = true`
    ///    → `G = 3·Ω_m·H₀²/(8π)` (condición de Friedmann para ρ̄_m=1)
    /// 3. Fallback → `simulation.gravitational_constant`
    ///
    /// Para diagnosticar inconsistencias usa `cosmo_g_diagnostic`.
    pub fn effective_g(&self) -> f64 {
        if self.units.enabled {
            self.units.compute_g()
        } else if self.cosmology.enabled && self.cosmology.auto_g {
            crate::cosmology::g_code_consistent(self.cosmology.omega_m, self.cosmology.h0)
        } else {
            self.simulation.gravitational_constant
        }
    }

    /// Diagnóstico de consistencia cosmológica de G.
    ///
    /// Devuelve `Some((g_consistent, error_relativo))` cuando `cosmology.enabled = true`
    /// y se puede calcular G auto-consistente a partir de `omega_m` y `h0`.
    /// El error relativo mide cuánto difiere `effective_g()` del valor Friedmann-consistente.
    ///
    /// Devuelve `None` si la cosmología está desactivada.
    pub fn cosmo_g_diagnostic(&self) -> Option<(f64, f64)> {
        if !self.cosmology.enabled {
            return None;
        }
        let g_consistent =
            crate::cosmology::g_code_consistent(self.cosmology.omega_m, self.cosmology.h0);
        let g_used = self.effective_g();
        let rel_err = if g_consistent > 0.0 {
            (g_used - g_consistent).abs() / g_consistent
        } else {
            f64::INFINITY
        };
        Some((g_consistent, rel_err))
    }
}
