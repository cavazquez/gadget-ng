use serde::{Deserialize, Serialize};

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
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum TransferKind {
    /// Ley de potencia pura `P(k) ∝ k^n_s` (sin función de transferencia).
    /// Comportamiento legacy de Fase 26. T(k) = 1 para todos los modos.
    #[default]
    PowerLaw,
    /// Eisenstein–Hu 1998, aproximación sin-wiggle (no-wiggle).
    /// Requiere `box_size_mpc_h` para la conversión de k.
    EisensteinHu,
    /// Transfer function tabulada (p.ej. salida CLASS/CAMB).
    ///
    /// Se espera archivo de dos columnas numéricas por fila:
    /// `k[h/Mpc]  T(k)`.
    ///
    /// También se aceptan líneas con columnas extra (se usan las dos primeras),
    /// comentarios (`# ...`) y filas vacías.
    Tabulated { path: String },
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
