use serde::{Deserialize, Serialize};

/// Configuración opt-in para kernels CUDA smoke/parity por módulo.
///
/// Cada flag habilita el path CUDA para un módulo de física específico cuando
/// `[performance] use_gpu_cuda = true`. Sin el flag, el módulo usa CPU
/// independientemente de `use_gpu_cuda`. Los flags solo tienen efecto cuando
/// se compila con `--features cuda` y hay un dispositivo NVIDIA disponible.
///
/// **Paridad real (✅):** `use_gpu_cuda` por sí solo activa gravedad directa y PM.
/// **Paridad smoke/parity (⚠️):** estos flags opt-in activan kernels que aún no
/// tienen validación 1:1 contra CPU en hardware real. Si el kernel CUDA falla
/// (dispositivo no disponible, error de ejecución), se cae al path CPU sin error fatal.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AcceleratorsSection {
    /// Activar kernels CUDA SPH (densidad Wendland, Balsara, fuerzas clásicas/Gadget-2).
    /// Requiere `[performance] use_gpu_cuda = true` y `--features cuda`.
    #[serde(default)]
    pub cuda_sph: bool,

    /// Activar kernels CUDA MHD (inducción, resistividad, fuerzas magnéticas,
    /// limpieza Dedner, flux-freeze, Braginskii, reconexión/dinamo, ambipolar,
    /// two-fluid, estadísticas B).
    /// Requiere `[performance] use_gpu_cuda = true` y `--features cuda`.
    #[serde(default)]
    pub cuda_mhd: bool,

    /// Activar kernel CUDA cooling (H/He, metales, UVB).
    /// Requiere `[performance] use_gpu_cuda = true` y `--features cuda`.
    #[serde(default)]
    pub cuda_cooling: bool,

    /// Activar kernel CUDA dust (crecimiento/sputtering/radiation pressure).
    /// Requiere `[performance] use_gpu_cuda = true` y `--features cuda`.
    #[serde(default)]
    pub cuda_dust: bool,

    /// Activar kernel CUDA H₂ molecular (HI→H₂ con dust shielding).
    /// Requiere `[performance] use_gpu_cuda = true` y `--features cuda`.
    #[serde(default)]
    pub cuda_h2: bool,

    /// Activar kernels CUDA RT (diagnósticos M1, foto-calentamiento).
    /// Requiere `[performance] use_gpu_cuda = true` y `--features cuda`.
    #[serde(default)]
    pub cuda_rt: bool,

    /// Activar kernels CUDA árbol local (walk monopolo Barnes–Hut) y SIDM.
    /// Requiere `[performance] use_gpu_cuda = true` y `--features cuda`.
    #[serde(default)]
    pub cuda_tree: bool,

    /// Activar kernels CUDA RT chemistry: tasas de fotoionización Γ_HI por partícula
    /// (NGP lookup), cooling Bremsstrahlung+Lyα, red química stiff de 12 especies
    /// (subciclo implícito adaptativo), estadísticos de reionización y campo 21cm.
    /// Solo tiene efecto cuando `[reionization] enabled = true`.
    /// Requiere `[performance] use_gpu_cuda = true` y `--features cuda`.
    #[serde(default)]
    pub cuda_rt_chem: bool,

    /// Activar kernels CUDA MHD CR streaming + backreaction (O(N²) por partícula).
    /// Solo tiene efecto cuando `[sph.cr] streaming_coefficient > 0`
    /// y `[mhd] enabled = true`.
    /// Requiere `[performance] use_gpu_cuda = true` y `--features cuda`.
    #[serde(default)]
    pub cuda_cr: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum SnapshotFormat {
    #[default]
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
    /// `false` → Rayon activo (requiere build con `--features simd`): solver global
    /// Allgather y recorridos locales del árbol (slab/SFC, LET, jerárquico). El orden
    /// de suma puede diferir → no se garantiza paridad bit-a-bit con el modo serial.
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

    /// `true` → intentar Barnes–Hut monopolar en **wgpu** (árbol en CPU, walk en GPU).
    /// Requiere `--features gpu` y `[gravity] multipole_order = 1` con criterio
    /// geométrico. No reemplaza SFC+LET (fuerzas locales+remotas siguen en CPU).
    /// Con `false` (default) se usa el walk CPU.
    #[serde(default)]
    pub use_gpu_barnes_hut: bool,

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

    /// `true` → con `solver = "tree_pm"`, intentar TreePM híbrido (PM CUDA filtrado + SR wgpu).
    /// Requiere `--features gpu,cuda`. Si falla, TreePM en CPU.
    #[serde(default)]
    pub use_gpu_treepm: bool,

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
            use_gpu_barnes_hut: false,
            use_gpu_cuda: false,
            use_gpu_hip: false,
            use_gpu_treepm: false,
            rebalance_imbalance_threshold: 0.0,
        }
    }
}
