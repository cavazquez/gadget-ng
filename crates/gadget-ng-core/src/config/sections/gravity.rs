use serde::{Deserialize, Serialize};

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
    /// - `3` → monopolo + cuadrupolo + octupolo (default)
    /// - `4` → incluye hexadecapolo (l = 4, 15 componentes STF compactas)
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
