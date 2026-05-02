//! HPC timing aggregates and per-step diagnostic structs for the simulation engine.

// ── Timing report ─────────────────────────────────────────────────────────────

/// Estadísticas detalladas HPC por evaluación de fuerza en el path SFC+LET.
///
/// Se serializa como campo `hpc_stats` en cada línea de `diagnostics.jsonl`.
/// Los tiempos son nanosegundos acumulados; se acumulan a lo largo de todas
/// las evaluaciones de fuerza dentro del paso (1 para leapfrog, 3 para Yoshida4).
#[derive(serde::Serialize, Default, Clone)]
pub(crate) struct HpcStepStats {
    /// Tiempo de construcción del octree local (ns).
    pub tree_build_ns: u64,
    /// Tiempo de exportación de nodos LET hacia todos los rangos remotos (ns).
    pub let_export_ns: u64,
    /// Tiempo de empaquetado de nodos LET — `pack_let_nodes` (ns).
    pub let_pack_ns: u64,
    /// Tiempo del allgather de AABBs (ns).
    pub aabb_allgather_ns: u64,
    /// Tiempo total del alltoallv LET (ns).
    /// - Path no-bloqueante: incluye el trabajo de overlap; `wait_ns ≈ let_alltoallv_ns - walk_local_ns`.
    /// - Path bloqueante: tiempo de espera puro de la colectiva.
    pub let_alltoallv_ns: u64,
    /// Tiempo del walk local del árbol (ns).
    /// En el path no-bloqueante, se solapa con `let_alltoallv_ns`.
    pub walk_local_ns: u64,
    /// Tiempo de aplicación de las fuerzas de nodos LET remotos (loop plano) (ns).
    pub apply_let_ns: u64,
    /// Nodos LET exportados a todos los rangos remotos en este paso.
    pub let_nodes_exported: usize,
    /// Nodos LET importados de todos los rangos remotos en este paso.
    pub let_nodes_imported: usize,
    /// Bytes enviados en alltoallv LET.
    pub bytes_sent: usize,
    /// Bytes recibidos en alltoallv LET.
    pub bytes_recv: usize,
    /// Tiempo de construcción del `LetTree` (ns). 0 si el path plano está activo.
    pub let_tree_build_ns: u64,
    /// Tiempo del walk del `LetTree` (N_local recorridos) (ns). 0 si el path plano está activo.
    pub let_tree_walk_ns: u64,
    /// Número de nodos en el `LetTree` construido. 0 si el path plano está activo.
    pub let_tree_nodes: usize,
    /// `true` si el walk del `LetTree` se ejecutó con Rayon (feature `simd`).
    pub let_tree_parallel: bool,
    /// Máximo de nodos LET exportados a un único rank remoto en este paso.
    pub max_let_nodes_per_rank: usize,
    /// Número total de nodos del árbol local (para calcular prune ratio).
    pub local_tree_nodes: usize,
    /// Tiempo de rebalanceo SFC (recompute cutpoints + bbox global) (ns).
    pub domain_rebalance_ns: u64,
    /// Tiempo de migración de partículas (exchange_domain_sfc) (ns).
    pub domain_migration_ns: u64,
    /// Partículas locales tras migración (count al inicio del paso).
    pub local_particle_count: usize,
    /// Tiempo en LetTree::apply_leaf total (ns) — profiling P14.
    pub apply_leaf_ns: u64,
    /// Total de RMNs procesados en hojas del LetTree — profiling P14.
    pub apply_leaf_rmn_count: u64,
    /// Número de llamadas a apply_leaf — profiling P14.
    pub apply_leaf_calls: u64,
    /// Tiempo en evaluaciones de nodos LET con RmnSoa (SoA path) (ns) — P14.
    pub rmn_soa_pack_ns: u64,
    /// Tiempo en accel_from_let_soa (flat LET SoA path) (ns) — P14.
    pub accel_from_let_soa_ns: u64,
    /// Número de llamadas a apply_leaf_soa_4xi (Fase 16).
    pub apply_leaf_tile_calls: u64,
    /// Suma de tile_size en cada llamada tileada (partículas-i procesadas en modo 4xi).
    pub apply_leaf_tile_i_count: u64,

    // ── Diagnósticos TreePM distribuido (Fase 21) ─────────────────────────────
    /// Partículas halo intercambiadas para corto alcance TreePM (suma de todos los pasos).
    pub short_range_halo_particles: usize,
    /// Bytes de partículas halo comunicados para corto alcance TreePM (suma de pasos).
    pub short_range_halo_bytes: usize,
    /// Tiempo acumulado en el árbol de corto alcance TreePM (ns).
    pub tree_short_ns: u64,
    /// Tiempo acumulado en el PM de largo alcance TreePM (ns).
    pub pm_long_ns: u64,
    /// Tiempo total acumulado en el pipeline TreePM distribuido (ns).
    pub treepm_total_ns: u64,
    /// Identificador del path activo: `"treepm_serial"` | `"treepm_allgather"` |
    /// `"treepm_slab_1d"` | `"treepm_slab_3d"`.
    pub path_active: String,

    // ── Diagnósticos halo 3D periódico (Fase 22) ──────────────────────────────
    /// Partículas halo recibidas por `exchange_halos_3d_periodic` en este paso.
    pub halo_3d_particles: usize,
    /// Bytes totales recibidos por `exchange_halos_3d_periodic` en este paso.
    pub halo_3d_bytes: usize,
    /// Tiempo acumulado en `exchange_halos_3d_periodic` (ns).
    pub halo_3d_ns: u64,

    // ── Diagnósticos dominio 3D/SFC para SR (Fase 23) ─────────────────────────
    /// Partículas locales en el dominio SFC SR al inicio de la evaluación de fuerza.
    pub sr_domain_particle_count: usize,
    /// Partículas halo vecinas recibidas para el SR (via `exchange_halos_3d_periodic`).
    pub sr_halo_3d_neighbors: usize,
    /// Tiempo acumulado en sincronización PM↔SR (clone + migraciones z↔SFC + lookup) (ns).
    pub sr_sync_ns: u64,

    // ── Diagnósticos scatter/gather PM (Fase 24) ──────────────────────────────
    /// Partículas enviadas al scatter PM (= partículas locales SFC) por paso.
    pub pm_scatter_particles: usize,
    /// Bytes totales enviados en el scatter PM (5 × f64 × scatter_particles).
    pub pm_scatter_bytes: usize,
    /// Tiempo del scatter alltoallv PM (ns).
    pub pm_scatter_ns: u64,
    /// Partículas recibidas en el gather PM por paso.
    pub pm_gather_particles: usize,
    /// Bytes totales recibidos en el gather PM (4 × f64 × gather_particles).
    pub pm_gather_bytes: usize,
    /// Tiempo del gather alltoallv PM (ns).
    pub pm_gather_ns: u64,
}

/// Resumen HPC agregado incluido en `timings.json`.
#[derive(serde::Serialize)]
pub(crate) struct HpcTimingsAggregate {
    pub mean_tree_build_s: f64,
    pub mean_let_export_s: f64,
    pub mean_let_pack_s: f64,
    pub mean_aabb_allgather_s: f64,
    pub mean_let_alltoallv_s: f64,
    pub mean_walk_local_s: f64,
    pub mean_apply_let_s: f64,
    pub mean_let_nodes_exported: f64,
    pub mean_let_nodes_imported: f64,
    pub mean_bytes_sent: f64,
    pub mean_bytes_recv: f64,
    /// Fracción del tiempo total de paso gastada esperando MPI (alltoallv − walk_local).
    pub wait_fraction: f64,
    /// Tiempo medio de construcción del `LetTree` por paso (s). 0 si path plano activo.
    pub mean_let_tree_build_s: f64,
    /// Tiempo medio del walk del `LetTree` por paso (s). 0 si path plano activo.
    pub mean_let_tree_walk_s: f64,
    /// Media de nodos en el `LetTree` por paso. 0 si path plano activo.
    pub mean_let_tree_nodes: f64,
    /// `true` si el walk del `LetTree` usó Rayon (feature `simd`).
    pub let_tree_parallel: bool,
    /// Media del máximo de nodos LET por rank remoto.
    pub mean_max_let_nodes_per_rank: f64,
    /// Media del número de nodos del árbol local.
    pub mean_local_tree_nodes: f64,
    /// Ratio de poda: `let_nodes_exported / (local_tree_nodes * (P-1))`.
    /// Mide qué fracción del árbol se exporta en promedio por rank remoto.
    pub mean_export_prune_ratio: f64,
    /// Tiempo medio de rebalanceo SFC (recompute cutpoints) por paso (s).
    pub mean_domain_rebalance_s: f64,
    /// Tiempo medio de migración de partículas por paso (s).
    pub mean_domain_migration_s: f64,
    /// Media de partículas locales por paso.
    pub mean_local_particle_count: f64,
    /// Ratio de imbalance de partículas: max_count / min_count entre ranks.
    /// Calculado al final de la simulación con el último paso.
    pub particle_imbalance_ratio: f64,
    /// Curva SFC usada: "morton" o "hilbert".
    pub sfc_kind: String,
    /// Tiempo medio en LetTree::apply_leaf por paso (s) — profiling P14.
    pub mean_apply_leaf_s: f64,
    /// Media de RMNs procesados en hojas por paso — profiling P14.
    pub mean_apply_leaf_rmn_count: f64,
    /// Media de llamadas a apply_leaf por paso — profiling P14.
    pub mean_apply_leaf_calls: f64,
    /// Tiempo medio de empaquetado AoS→SoA (RmnSoa::from_slice) por paso (s) — P14.
    pub mean_rmn_soa_pack_s: f64,
    /// Tiempo medio en accel_from_let_soa (flat LET SoA) por paso (s) — P14.
    pub mean_accel_from_let_soa_s: f64,
    /// Si el path SoA está activo (feature simd).
    pub soa_simd_active: bool,
    /// Media de llamadas a apply_leaf_soa_4xi (Fase 16) por paso.
    pub mean_apply_leaf_tile_calls: f64,
    /// Media de partículas-i procesadas en modo 4xi por paso.
    pub mean_apply_leaf_tile_i_count: f64,
    /// Ratio de utilización de tiles: tile_i_count / (tile_calls * 4). Ideal = 1.0.
    pub tile_utilization_ratio: f64,

    // ── TreePM distribuido (Fase 21) ──────────────────────────────────────────
    /// Media de partículas halo de corto alcance intercambiadas por paso.
    pub mean_short_range_halo_particles: f64,
    /// Media de bytes de halo de corto alcance por paso.
    pub mean_short_range_halo_bytes: f64,
    /// Tiempo medio del árbol de corto alcance TreePM por paso (s).
    pub mean_tree_short_s: f64,
    /// Tiempo medio del PM de largo alcance TreePM por paso (s).
    pub mean_pm_long_s: f64,
    /// Tiempo medio total del pipeline TreePM distribuido por paso (s).
    pub mean_treepm_total_s: f64,
    /// Fracción del tiempo TreePM gastada en el árbol de corto alcance.
    pub tree_fraction: f64,
    /// Fracción del tiempo TreePM gastada en el PM de largo alcance.
    pub pm_fraction: f64,
    /// Path activo del TreePM: `"treepm_serial"` | `"treepm_allgather"` |
    /// `"treepm_slab_1d"` | `"treepm_slab_3d"`.
    pub path_active: String,

    // ── Halo 3D periódico (Fase 22) ───────────────────────────────────────────
    /// Media de partículas halo 3D por paso (halo volumétrico periódico).
    pub mean_halo_3d_particles: f64,
    /// Media de bytes de halo 3D por paso.
    pub mean_halo_3d_bytes: f64,
    /// Tiempo medio de `exchange_halos_3d_periodic` por paso (s).
    pub mean_halo_3d_s: f64,

    // ── Dominio 3D/SFC SR (Fase 23) ───────────────────────────────────────────
    /// Media de partículas en dominio SFC SR por paso.
    pub mean_sr_domain_particle_count: f64,
    /// Media de partículas halo SR vecinas (via halo 3D periódico) por paso.
    pub mean_sr_halo_3d_neighbors: f64,
    /// Tiempo medio de sincronización PM↔SR por paso (s).
    pub mean_sr_sync_s: f64,
    /// Fracción del tiempo TreePM gastada en sincronización PM↔SR.
    pub sr_sync_fraction: f64,

    // ── Scatter/Gather PM (Fase 24) ───────────────────────────────────────────
    /// Media de partículas enviadas en el scatter PM por paso.
    pub mean_pm_scatter_particles: f64,
    /// Media de bytes enviados en el scatter PM por paso.
    pub mean_pm_scatter_bytes: f64,
    /// Tiempo medio del scatter PM alltoallv por paso (s).
    pub mean_pm_scatter_s: f64,
    /// Media de partículas recibidas en el gather PM por paso.
    pub mean_pm_gather_particles: f64,
    /// Media de bytes recibidos en el gather PM por paso.
    pub mean_pm_gather_bytes: f64,
    /// Tiempo medio del gather PM alltoallv por paso (s).
    pub mean_pm_gather_s: f64,
    /// Fracción del tiempo TreePM gastada en scatter+gather PM (Fase 24 vs Fase 23 sr_sync).
    pub pm_sync_fraction: f64,
}

/// Resumen HPC agregado para el path TreePM SR-SFC (Fases 23/24).
///
/// Escrito como campo `"treepm_hpc"` en `timings.json` al final del run.
/// Solo se emite si el path `use_treepm_sr_sfc` estuvo activo.
#[derive(serde::Serialize)]
pub(crate) struct TreePmAggregate {
    /// Tiempo medio de scatter alltoallv PM (o clone+migrate) por paso (segundos).
    pub mean_scatter_s: f64,
    /// Tiempo medio de gather alltoallv PM por paso (segundos). 0 en Fase 23.
    pub mean_gather_s: f64,
    /// Tiempo medio de resolución PM (FFT + interpolación) por paso (segundos).
    pub mean_pm_solve_s: f64,
    /// Tiempo medio de intercambio de halos SR 3D por paso (segundos).
    pub mean_sr_halo_s: f64,
    /// Tiempo medio del árbol corto alcance por paso (segundos).
    pub mean_tree_sr_s: f64,
    /// Media de partículas enviadas en scatter PM (o clonadas) por paso.
    pub mean_scatter_particles: f64,
    /// Media de bytes enviados en scatter por paso.
    pub mean_scatter_bytes: f64,
    /// Media de bytes recibidos en gather por paso.
    pub mean_gather_bytes: f64,
    /// Fracción del tiempo TreePM gastada en scatter+gather (sincronización PM↔SR).
    /// `(scatter_ns + gather_ns) / (scatter_ns + gather_ns + pm_solve_ns + sr_halo_ns + tree_sr_ns)`
    pub pm_sync_fraction: f64,
    /// Tiempo medio total del path TreePM SR-SFC por paso (segundos).
    pub mean_treepm_total_s: f64,
    /// Path activo: `"scatter_gather"` (Fase 24) o `"clone_migrate"` (Fase 23).
    pub path_active: &'static str,
}

/// Resumen de tiempos por fase, escrito en `<out>/timings.json` al final del run.
///
/// Permite medir el desglose entre comunicación MPI, cálculo de fuerzas e integración
/// sin necesidad de herramientas externas de profiling.
#[derive(serde::Serialize)]
pub(crate) struct TimingsReport {
    /// Número de pasos ejecutados.
    pub steps: u64,
    /// Número de partículas totales.
    pub total_particles: usize,
    /// Tiempo de pared total del loop de integración (segundos).
    pub total_wall_s: f64,
    /// Tiempo acumulado en comunicación MPI / allgather (segundos).
    pub total_comm_s: f64,
    /// Tiempo acumulado en cálculo de fuerzas gravitatorias (segundos).
    pub total_gravity_s: f64,
    /// Tiempo acumulado en kicks+drifts de integración (segundos).
    pub total_integration_s: f64,
    /// Tiempo de pared medio por paso (segundos).
    pub mean_step_wall_s: f64,
    /// Tiempo medio de comunicación por paso (segundos).
    pub mean_comm_s: f64,
    /// Tiempo medio de fuerza gravitatoria por paso (segundos).
    pub mean_gravity_s: f64,
    /// Fracción del tiempo total gastada en comunicación.
    pub comm_fraction: f64,
    /// Fracción del tiempo total gastada en fuerzas.
    pub gravity_fraction: f64,
    /// Resumen detallado HPC (solo para path SFC+LET).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hpc: Option<HpcTimingsAggregate>,
    /// Resumen HPC del path TreePM SR-SFC (Fases 23/24). Solo se emite si estuvo activo.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub treepm_hpc: Option<TreePmAggregate>,
}
/// Agregados locales O(N) usados en los diagnósticos por paso.
#[derive(Clone, Copy, Default)]
pub(crate) struct LocalMoments {
    /// Momento lineal total: Σ mᵢ vᵢ.
    pub p: [f64; 3],
    /// Momento angular total respecto al origen: Σ mᵢ (rᵢ × vᵢ).
    pub l: [f64; 3],
    /// Σ mᵢ rᵢ (para el centro de masa).
    pub mass_weighted_pos: [f64; 3],
    /// Σ mᵢ.
    pub mass: f64,
}
/// Diagnóstico por llamada a compute_acc en el path TreePM SR-SFC (Fases 23/24).
///
/// Se acumula sobre todas las sub-llamadas de un paso (1 para leapfrog, 3 para Yoshida4)
/// y se serializa como campo `"treepm"` en cada línea de `diagnostics.jsonl`.
#[derive(serde::Serialize, Default, Clone, Copy)]
pub(crate) struct TreePmStepDiag {
    /// Tiempo total de scatter alltoallv PM (Fase 24) o clone+migrate (Fase 23), en ns.
    pub scatter_ns: u64,
    /// Tiempo total de gather alltoallv PM (Fase 24), en ns. 0 para Fase 23.
    pub gather_ns: u64,
    /// Tiempo total de resolución PM (FFT + interpolación) en el slab, en ns.
    pub pm_solve_ns: u64,
    /// Tiempo total de intercambio de halos SR 3D periódico, en ns.
    pub sr_halo_ns: u64,
    /// Tiempo total del árbol corto alcance (SFC domain), en ns.
    pub tree_sr_ns: u64,
    /// Número de partículas enviadas en scatter PM (Fase 24) o clonadas (Fase 23).
    pub scatter_particles: usize,
    /// Bytes enviados en scatter PM por este rank (Fase 24). 0 para Fase 23.
    pub scatter_bytes: usize,
    /// Bytes recibidos en gather PM por este rank (Fase 24). 0 para Fase 23.
    pub gather_bytes: usize,
    /// Path activo: `"sg"` (Fase 24 scatter/gather) o `"clone"` (Fase 23 clone+migrate).
    pub path: &'static str,
}

impl TreePmStepDiag {
    /// Acumula otro `TreePmStepDiag` (suma campos numéricos; preserva `path` del `other`).
    pub(crate) fn add(self, other: Self) -> Self {
        Self {
            scatter_ns: self.scatter_ns + other.scatter_ns,
            gather_ns: self.gather_ns + other.gather_ns,
            pm_solve_ns: self.pm_solve_ns + other.pm_solve_ns,
            sr_halo_ns: self.sr_halo_ns + other.sr_halo_ns,
            tree_sr_ns: self.tree_sr_ns + other.tree_sr_ns,
            scatter_particles: self.scatter_particles + other.scatter_particles,
            scatter_bytes: self.scatter_bytes + other.scatter_bytes,
            gather_bytes: self.gather_bytes + other.gather_bytes,
            path: other.path,
        }
    }
}

/// Datos de diagnóstico cosmológico opcionales para cada paso.
pub(crate) struct CosmoDiag {
    /// Factor de escala actual `a`.
    pub a: f64,
    /// Redshift `z = 1/a - 1`.
    pub z: f64,
    /// Velocidad peculiar RMS `sqrt(⟨|p/a|²⟩)` en unidades internas.
    pub v_rms: f64,
    /// Contraste de densidad RMS `sqrt(⟨(δρ/ρ̄)²⟩)` sobre malla 16³.
    pub delta_rms: f64,
    /// Parámetro de Hubble H(a) en unidades internas.
    pub hubble: f64,
    /// Diagnóstico TreePM SR-SFC por paso (Fases 23/24). `None` si no está activo.
    pub treepm: Option<TreePmStepDiag>,
}
