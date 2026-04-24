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
    /// Módulo SPH cosmológico (Phase 66; opcional; desactivado por defecto).
    #[serde(default)]
    pub sph: SphSection,
    /// Solver de transferencia radiativa M1 (Phase 81; opcional; desactivado por defecto).
    #[serde(default)]
    pub rt: RtSection,
    /// Reionización del Universo: fuentes UV puntuales (Phase 89; opcional).
    #[serde(default)]
    pub reionization: ReionizationSection,
    /// Magnetohidrodinámica ideal (Phase 126; opcional; desactivado por defecto).
    #[serde(default)]
    pub mhd: MhdSection,
    /// Forzado de turbulencia MHD Ornstein-Uhlenbeck (Phase 140; opcional; desactivado).
    #[serde(default)]
    pub turbulence: TurbulenceSection,
    /// Plasma de dos fluidos T_e ≠ T_i (Phase 149; opcional; desactivado).
    #[serde(default)]
    pub two_fluid: TwoFluidSection,
    /// Materia oscura auto-interactuante SIDM (Phase 157; opcional; desactivado).
    #[serde(default)]
    pub sidm: SidmSection,
    /// Gravedad modificada f(R) con screening chameleon (Phase 158; opcional; desactivado).
    #[serde(default)]
    pub modified_gravity: ModifiedGravitySection,
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
    #[serde(default)]
    pub auto_g: bool,
    /// Parámetro CPL w₀ para energía oscura dinámica (Phase 155).
    /// `w(a) = w0 + wa*(1-a)`. Default: -1.0 (ΛCDM).
    #[serde(default = "default_w0")]
    pub w0: f64,
    /// Parámetro CPL wₐ para energía oscura dinámica (Phase 155).
    /// Default: 0.0 (ΛCDM).
    #[serde(default)]
    pub wa: f64,
    /// Suma de masas de neutrinos en eV (Phase 156).
    /// `Ω_ν = m_ν / (93.14 eV × h²)`. Default: 0.0 (sin neutrinos).
    #[serde(default)]
    pub m_nu_ev: f64,
}

fn default_omega_m() -> f64 { 0.3 }
fn default_omega_lambda() -> f64 { 0.7 }
fn default_h0() -> f64 { 0.1 }
fn default_a_init() -> f64 { 1.0 }
fn default_w0() -> f64 { -1.0 }

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
            w0: default_w0(),
            wa: 0.0,
            m_nu_ev: 0.0,
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
    /// Enfriamiento con contribución metálica (Phase 111).
    ///
    /// `Λ(T, Z) = Λ_HHe(T) + (Z/Z_sun) × Λ_metal(T)` — Sutherland & Dopita (1993).
    MetalCooling,
    /// Enfriamiento tabulado con interpolación bilineal (Phase 119).
    ///
    /// Tabla interna 7×20 derivada de Sutherland & Dopita (1993): 7 bins en Z/Z_sun
    /// y 20 bins en log10(T) de 4.0 a 8.5. Interpolación bilineal en (Z, log T).
    MetalTabular,
}

/// Configuración del módulo SPH cosmológico (Phase 66).
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
    /// Configuración del feedback estelar (Phase 78).
    #[serde(default)]
    pub feedback: FeedbackSection,
    /// Configuración del feedback AGN (Phase 96).
    #[serde(default)]
    pub agn: AgnSection,
    /// Configuración del enriquecimiento químico metálico (Phase 109).
    #[serde(default)]
    pub enrichment: EnrichmentSection,
    /// Configuración del modelo ISM multifase (Phase 114).
    #[serde(default)]
    pub ism: IsmSection,
    /// Configuración del módulo de rayos cósmicos (Phase 117).
    #[serde(default)]
    pub cr: CrSection,
    /// Configuración de la conducción térmica del ICM (Phase 121).
    #[serde(default)]
    pub conduction: ConductionSection,
    /// Configuración del gas molecular H₂ (Phase 122).
    #[serde(default)]
    pub molecular: MolecularSection,
    /// Configuración del polvo intersticial (Phase 130).
    #[serde(default)]
    pub dust: DustSection,
    /// Factor de supresión del cooling por campo magnético (Phase 134).
    ///
    /// `Λ_eff = Λ(T) / (1 + mag_suppress_cooling / β)`.
    /// `0.0` = sin supresión magnética (default).
    #[serde(default)]
    pub mag_suppress_cooling: f64,
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
            feedback: FeedbackSection::default(),
            agn: AgnSection::default(),
            enrichment: EnrichmentSection::default(),
            ism: IsmSection::default(),
            cr: CrSection::default(),
            conduction: ConductionSection::default(),
            molecular: MolecularSection::default(),
            dust: DustSection::default(),
            mag_suppress_cooling: 0.0,
        }
    }
}

/// Configuración del feedback estelar por supernovas (Phase 78).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackSection {
    /// Activa el feedback estelar (default: `false`).
    #[serde(default)]
    pub enabled: bool,
    /// Velocidad del kick de SN en km/s (default: 350 km/s).
    #[serde(default = "default_v_kick")]
    pub v_kick_km_s: f64,
    /// Eficiencia de energía SN ε_SN: fracción de E_SN=10⁵¹ erg transferida (default: 0.1).
    #[serde(default = "default_eps_sn")]
    pub eps_sn: f64,
    /// Densidad umbral para SFR en unidades internas (default: 0.1).
    #[serde(default = "default_rho_sf")]
    pub rho_sf: f64,
    /// SFR mínima para activar el feedback (default: 1e-4 en unidades internas).
    #[serde(default = "default_sfr_min")]
    pub sfr_min: f64,
    /// Vientos galácticos (Phase 108). Default: desactivado.
    #[serde(default)]
    pub wind: WindParams,
    /// Fracción de masa que se convierte en estrella al hacer spawning (Phase 112).
    /// Default: 0.5 (el gas padre retiene el 50% de su masa).
    #[serde(default = "default_m_star_fraction")]
    pub m_star_fraction: f64,
    /// Masa mínima de gas para que la partícula no se elimine tras el spawning (Phase 112).
    /// Partículas con masa < m_gas_min se marcan para eliminación.
    #[serde(default = "default_m_gas_min")]
    pub m_gas_min: f64,
    /// Normalización de la DTD de SN Ia: A_Ia [SN / Gyr / M_sun] (Phase 113).
    /// Basado en Maoz & Mannucci (2012): ~2 × 10⁻³ SN/Gyr/M_sun.
    #[serde(default = "default_a_ia")]
    pub a_ia: f64,
    /// Tiempo mínimo para SN Ia tras la formación estelar [Gyr] (Phase 113).
    /// Default: 0.1 Gyr.
    #[serde(default = "default_t_ia_min_gyr")]
    pub t_ia_min_gyr: f64,
    /// Energía inyectada por SN Ia en unidades internas (Phase 113).
    /// E_Ia ≈ 1e51 erg × 1.3 en unidades gadget-ng.
    #[serde(default = "default_e_ia_code")]
    pub e_ia_code: f64,
    /// Activa el feedback mecánico de vientos estelares pre-SN (Phase 115).
    /// Modela vientos de estrellas OB y Wolf-Rayet (~10-30 Myr antes de SN II).
    #[serde(default)]
    pub stellar_wind_enabled: bool,
    /// Velocidad terminal del viento estelar [km/s] (Phase 115). Default: 2000 km/s.
    #[serde(default = "default_v_stellar_wind")]
    pub v_stellar_wind_km_s: f64,
    /// Factor de carga másica del viento estelar η_w = Ṁ_wind / SFR (Phase 115). Default: 0.1.
    #[serde(default = "default_eta_stellar_wind")]
    pub eta_stellar_wind: f64,
}

/// Parámetros del modelo de vientos galácticos (Phase 108).
///
/// Basado en la prescripción de Springel & Hernquist (2003) MNRAS 339, 289.
///
/// La velocidad del viento es `v_wind_km_s` y el factor de carga de masa `η`
/// determina cuánta masa se eyecta por unidad de masa estelar formada.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindParams {
    /// Activa los vientos galácticos (default: `false`).
    #[serde(default)]
    pub enabled: bool,
    /// Velocidad terminal del viento en km/s (default: 480 km/s ≈ 2× v_SN).
    #[serde(default = "default_v_wind")]
    pub v_wind_km_s: f64,
    /// Factor de carga de masa η = Ṁ_wind / SFR (default: 2.0).
    #[serde(default = "default_mass_loading")]
    pub mass_loading: f64,
    /// Tiempo de desacoplamiento hidrológico en Myr (default: 0.0 = sin desacoplamiento).
    #[serde(default)]
    pub t_decoupling_myr: f64,
}

fn default_v_wind() -> f64 { 480.0 }
fn default_mass_loading() -> f64 { 2.0 }

impl Default for WindParams {
    fn default() -> Self {
        Self {
            enabled: false,
            v_wind_km_s: default_v_wind(),
            mass_loading: default_mass_loading(),
            t_decoupling_myr: 0.0,
        }
    }
}

fn default_v_kick() -> f64 { 350.0 }
fn default_eps_sn() -> f64 { 0.1 }
fn default_rho_sf() -> f64 { 0.1 }
fn default_sfr_min() -> f64 { 1e-4 }
fn default_m_star_fraction() -> f64 { 0.5 }
fn default_m_gas_min() -> f64 { 0.01 }
fn default_a_ia() -> f64 { 2e-3 }
fn default_t_ia_min_gyr() -> f64 { 0.1 }
fn default_e_ia_code() -> f64 { 1.54e-3 * 1.3 } // E_Ia ≈ 1.3 × E_SN
fn default_v_stellar_wind() -> f64 { 2000.0 }
fn default_eta_stellar_wind() -> f64 { 0.1 }

impl Default for FeedbackSection {
    fn default() -> Self {
        Self {
            enabled: false,
            v_kick_km_s: default_v_kick(),
            eps_sn: default_eps_sn(),
            rho_sf: default_rho_sf(),
            sfr_min: default_sfr_min(),
            wind: WindParams::default(),
            m_star_fraction: default_m_star_fraction(),
            m_gas_min: default_m_gas_min(),
            a_ia: default_a_ia(),
            t_ia_min_gyr: default_t_ia_min_gyr(),
            e_ia_code: default_e_ia_code(),
            stellar_wind_enabled: false,
            v_stellar_wind_km_s: default_v_stellar_wind(),
            eta_stellar_wind: default_eta_stellar_wind(),
        }
    }
}

/// Configuración del enriquecimiento químico metálico (Phase 109).
///
/// Controla los yields de SN II y AGB usados en `apply_enrichment`.
///
/// ```toml
/// [sph.enrichment]
/// enabled    = true
/// yield_snii = 0.02   # fracción de masa en metales eyectada por SN II
/// yield_agb  = 0.04   # fracción de masa en metales eyectada por AGB
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnrichmentSection {
    /// Activa el enriquecimiento químico (default: `false`).
    #[serde(default)]
    pub enabled: bool,
    /// Yield metálico de SN II: fracción de masa en Z eyectada (default: 0.02).
    #[serde(default = "default_yield_snii")]
    pub yield_snii: f64,
    /// Yield metálico de AGB: fracción de masa en Z eyectada (default: 0.04).
    #[serde(default = "default_yield_agb")]
    pub yield_agb: f64,
}

fn default_yield_snii() -> f64 { 0.02 }
fn default_yield_agb() -> f64 { 0.04 }

impl Default for EnrichmentSection {
    fn default() -> Self {
        Self {
            enabled: false,
            yield_snii: default_yield_snii(),
            yield_agb: default_yield_agb(),
        }
    }
}

/// Configuración del modelo ISM multifase fría-caliente (Phase 114).
///
/// Basado en el modelo de presión efectiva de Springel & Hernquist (2003).
///
/// ```toml
/// [sph.ism]
/// enabled = true
/// q_star  = 2.5
/// f_cold  = 0.5
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IsmSection {
    /// Activa el modelo ISM multifase (default: `false`).
    #[serde(default)]
    pub enabled: bool,
    /// Parámetro de escala de presión efectiva q* (default: 2.5).
    /// Controla la rigidez del ISM frío: P_eff = (γ-1) ρ (u + q_star × u_cold).
    #[serde(default = "default_q_star")]
    pub q_star: f64,
    /// Fracción fría inicial del ISM (default: 0.5).
    #[serde(default = "default_f_cold")]
    pub f_cold: f64,
}

fn default_q_star() -> f64 { 2.5 }
fn default_f_cold() -> f64 { 0.5 }

impl Default for IsmSection {
    fn default() -> Self {
        Self {
            enabled: false,
            q_star: default_q_star(),
            f_cold: default_f_cold(),
        }
    }
}

/// Configuración del gas molecular simple HI → H₂ (Phase 122).
///
/// ```toml
/// [sph.molecular]
/// enabled          = true
/// rho_h2_threshold = 100.0
/// sfr_h2_boost     = 2.0
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MolecularSection {
    /// Activa el módulo de gas molecular (default: `false`).
    #[serde(default)]
    pub enabled: bool,
    /// Densidad umbral de formación de H₂ en unidades internas (default: `100.0`).
    #[serde(default = "default_rho_h2_threshold")]
    pub rho_h2_threshold: f64,
    /// Factor multiplicativo de SFR en gas molecular (default: `2.0`).
    #[serde(default = "default_sfr_h2_boost")]
    pub sfr_h2_boost: f64,
}

fn default_rho_h2_threshold() -> f64 { 100.0 }
fn default_sfr_h2_boost() -> f64 { 2.0 }

impl Default for MolecularSection {
    fn default() -> Self {
        Self {
            enabled: false,
            rho_h2_threshold: default_rho_h2_threshold(),
            sfr_h2_boost: default_sfr_h2_boost(),
        }
    }
}

/// Configuración de la conducción térmica del ICM (Phase 121).
///
/// Modelo de Spitzer (1962) con factor de supresión ψ por campo B / turbulencia.
///
/// ```toml
/// [sph.conduction]
/// enabled        = true
/// kappa_spitzer  = 1e-4
/// psi_suppression = 0.1
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConductionSection {
    /// Activa la conducción térmica (default: `false`).
    #[serde(default)]
    pub enabled: bool,
    /// Conductividad Spitzer en unidades internas (default: `1e-4`).
    #[serde(default = "default_kappa_spitzer")]
    pub kappa_spitzer: f64,
    /// Factor de supresión ψ ∈ [0,1] por campo magnético o turbulencia (default: `0.1`).
    #[serde(default = "default_psi_suppression")]
    pub psi_suppression: f64,
    /// Activa conducción anisótropa ∥B (Phase 133). Si `false` usa Spitzer isótropo.
    #[serde(default)]
    pub anisotropic: bool,
    /// Conductividad paralela a B (Phase 133). Default: igual a `kappa_spitzer`.
    #[serde(default = "default_kappa_par")]
    pub kappa_par: f64,
    /// Conductividad perpendicular a B (Phase 133). Default: `1e-6` (muy suprimida).
    #[serde(default = "default_kappa_perp")]
    pub kappa_perp: f64,
}

fn default_kappa_spitzer() -> f64 { 1e-4 }
fn default_psi_suppression() -> f64 { 0.1 }
fn default_kappa_par() -> f64 { 1e-4 }
fn default_kappa_perp() -> f64 { 1e-6 }

impl Default for ConductionSection {
    fn default() -> Self {
        Self {
            enabled: false,
            kappa_spitzer: default_kappa_spitzer(),
            psi_suppression: default_psi_suppression(),
            anisotropic: false,
            kappa_par: default_kappa_par(),
            kappa_perp: default_kappa_perp(),
        }
    }
}

/// Configuración del módulo de rayos cósmicos (Phase 117).
///
/// ```toml
/// [sph.cr]
/// enabled     = true
/// cr_fraction = 0.1
/// kappa_cr    = 3e-3
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrSection {
    /// Activa el módulo de rayos cósmicos (default: `false`).
    #[serde(default)]
    pub enabled: bool,
    /// Fracción de energía de SN II que va a CRs (default: 0.1).
    #[serde(default = "default_cr_fraction")]
    pub cr_fraction: f64,
    /// Coeficiente de difusión isótropa κ_CR [unidades internas] (default: 3e-3).
    #[serde(default = "default_kappa_cr")]
    pub kappa_cr: f64,
    /// Factor de supresión de difusión CR por campo magnético (Phase 129).
    ///
    /// `f_suppress = 1 / (1 + b_cr_suppress × |B|²)`.
    /// `1.0` → supresión moderada; `0.0` → sin supresión (recupera comportamiento antiguo).
    #[serde(default = "default_b_cr_suppress")]
    pub b_cr_suppress: f64,
}

fn default_cr_fraction() -> f64 { 0.1 }
fn default_kappa_cr() -> f64 { 3e-3 }
fn default_b_cr_suppress() -> f64 { 1.0 }

impl Default for CrSection {
    fn default() -> Self {
        Self {
            enabled: false,
            cr_fraction: default_cr_fraction(),
            kappa_cr: default_kappa_cr(),
            b_cr_suppress: default_b_cr_suppress(),
        }
    }
}

/// Configuración del feedback de Agujeros Negros Supermasivos (AGN) (Phase 96).
///
/// Configura el modelo de acreción Bondi-Hoyle y feedback térmico/cinético.
///
/// ```toml
/// [sph.agn]
/// enabled      = true
/// eps_feedback = 0.05
/// m_seed       = 1e5
/// v_kick_agn   = 500.0
/// r_influence  = 1.0
/// n_agn_bh     = 1
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgnSection {
    /// Activa el feedback AGN (default: `false`).
    #[serde(default)]
    pub enabled: bool,
    /// Eficiencia radiativa del feedback ε_feedback (default: 0.05).
    #[serde(default = "default_eps_feedback")]
    pub eps_feedback: f64,
    /// Masa semilla del agujero negro [M_sol/h] (default: 1e5).
    #[serde(default = "default_m_seed")]
    pub m_seed: f64,
    /// Velocidad de kick cinético AGN [km/s] (default: 500.0; 0 = solo térmico).
    #[serde(default = "default_v_kick_agn")]
    pub v_kick_agn: f64,
    /// Radio de influencia máximo para depositar energía (default: 1.0).
    #[serde(default = "default_r_influence")]
    pub r_influence: f64,
    /// Número de BH semilla a colocar en los N halos FoF más masivos (default: 1).
    /// Si no hay halos identificados, se coloca uno en el centro de la caja.
    #[serde(default = "default_n_agn_bh")]
    pub n_agn_bh: usize,
    /// Umbral de Eddington para bifurcar modo quasar ↔ radio (Phase 116).
    /// `Ṁ / Ṁ_Edd < f_edd_threshold` → modo radio (jets mecánicos).
    /// Default: 0.01 (1% de la tasa de Eddington).
    #[serde(default = "default_f_edd_threshold")]
    pub f_edd_threshold: f64,
    /// Radio de la burbuja de jets mecánicos AGN [unidades internas] (Phase 116).
    /// Default: 2.0.
    #[serde(default = "default_r_bubble")]
    pub r_bubble: f64,
    /// Eficiencia del modo radio `ε_radio` (Phase 116).
    /// Fracción de la energía de acreción que va a kicks mecánicos. Default: 0.2.
    #[serde(default = "default_eps_radio")]
    pub eps_radio: f64,
}

fn default_eps_feedback() -> f64 { 0.05 }
fn default_m_seed() -> f64 { 1e5 }
fn default_v_kick_agn() -> f64 { 500.0 }
fn default_r_influence() -> f64 { 1.0 }
fn default_n_agn_bh() -> usize { 1 }
fn default_f_edd_threshold() -> f64 { 0.01 }
fn default_r_bubble() -> f64 { 2.0 }
fn default_eps_radio() -> f64 { 0.2 }

impl Default for AgnSection {
    fn default() -> Self {
        Self {
            enabled: false,
            eps_feedback: default_eps_feedback(),
            m_seed: default_m_seed(),
            v_kick_agn: default_v_kick_agn(),
            r_influence: default_r_influence(),
            n_agn_bh: default_n_agn_bh(),
            f_edd_threshold: default_f_edd_threshold(),
            r_bubble: default_r_bubble(),
            eps_radio: default_eps_radio(),
        }
    }
}

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

fn default_c_red_factor() -> f64 { 100.0 }
fn default_kappa_abs() -> f64 { 1.0 }
fn default_rt_mesh() -> usize { 32 }
fn default_rt_substeps() -> usize { 5 }

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

fn default_uv_luminosity() -> f64 { 1.0 }
fn default_z_reion_start() -> f64 { 12.0 }
fn default_z_reion_end() -> f64 { 6.0 }

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
    /// Directorio de salida para los archivos `insitu_NNNNNN.json`.
    /// Si es `None` se usa `<out_dir>/insitu/`.
    #[serde(default)]
    pub output_dir: Option<std::path::PathBuf>,
}

fn default_insitu_interval() -> u64 { 0 }
fn default_pk_mesh() -> usize { 32 }
fn default_fof_b() -> f64 { 0.2 }
fn default_fof_min_part() -> usize { 20 }
fn default_ab_smooth_r() -> f64 { 5.0 }

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

    /// Verifica la combinación softening/cosmología y retorna advertencias si las hay.
    ///
    /// - `physical_softening = true` sin `cosmology.enabled = true` no tiene efecto.
    /// - `physical_softening = false` con `cosmology.enabled = true` usa softening comóvil
    ///   constante (comportamiento legacy).
    pub fn softening_warnings(&self) -> Vec<&'static str> {
        let mut warnings = Vec::new();
        if self.simulation.physical_softening && !self.cosmology.enabled {
            warnings.push(
                "physical_softening = true no tiene efecto sin cosmology.enabled = true \
                 (el softening comóvil fijo ya es constante en simulaciones newtonianas)"
            );
        }
        warnings
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
}

/// Configuración del módulo de polvo intersticial básico (Phase 130).
///
/// Modelo simplificado de acreción D/G por metalicidad y destrucción por sputtering.
///
/// ```toml
/// [sph.dust]
/// enabled     = true
/// d_to_g_max  = 0.01
/// t_destroy_k = 1e6
/// tau_grow    = 1.0
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DustSection {
    /// Activa el módulo de polvo (default: `false`).
    #[serde(default)]
    pub enabled: bool,
    /// D/G máximo solar (default: `0.01`).
    #[serde(default = "default_d_to_g_max")]
    pub d_to_g_max: f64,
    /// Temperatura de destrucción por sputtering en K (default: `1e6`).
    #[serde(default = "default_t_destroy_k")]
    pub t_destroy_k: f64,
    /// Tiempo de crecimiento por accreción en unidades internas (default: `1.0`).
    #[serde(default = "default_tau_grow")]
    pub tau_grow: f64,
    /// Opacidad del polvo en UV: `κ_dust [cm²/g]` (Phase 137). Default: `1000.0`.
    #[serde(default = "default_kappa_dust_uv")]
    pub kappa_dust_uv: f64,
}

fn default_d_to_g_max() -> f64 { 0.01 }
fn default_t_destroy_k() -> f64 { 1e6 }
fn default_tau_grow() -> f64 { 1.0 }
fn default_kappa_dust_uv() -> f64 { 1000.0 }

impl Default for DustSection {
    fn default() -> Self {
        Self {
            enabled: false,
            d_to_g_max: default_d_to_g_max(),
            t_destroy_k: default_t_destroy_k(),
            tau_grow: default_tau_grow(),
            kappa_dust_uv: default_kappa_dust_uv(),
        }
    }
}

fn default_mhd_c_h() -> f64 { 1.0 }
fn default_mhd_c_r() -> f64 { 0.5 }
fn default_cfl_mhd() -> f64 { 0.3 }
fn default_alpha_b() -> f64 { 0.5 }
fn default_beta_freeze() -> f64 { 100.0 }
fn default_v_rel_threshold() -> f64 { 0.1 }
fn default_f_reconnection() -> f64 { 0.01 }
fn default_v_jet() -> f64 { 0.3 }
fn default_n_jet_halos() -> usize { 1 }

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
        }
    }
}

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

fn default_turb_amplitude() -> f64 { 1e-3 }
fn default_turb_tau() -> f64 { 1.0 }
fn default_turb_k_min() -> f64 { 1.0 }
fn default_turb_k_max() -> f64 { 4.0 }
fn default_turb_spectral_index() -> f64 { 5.0 / 3.0 }

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

fn default_nu_ei_coeff() -> f64 { 1.0 }

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

fn default_sidm_sigma_m() -> f64 { 1.0e-5 }
fn default_sidm_v_max() -> f64 { 1.0e6 }

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

fn default_mg_model() -> String { "hu_sawicki".to_string() }
fn default_f_r0() -> f64 { 1.0e-4 }
fn default_mg_n() -> f64 { 1.0 }

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
