use crate::config_load;
use crate::error::CliError;
#[cfg(feature = "simd")]
use gadget_ng_core::RayonDirectGravity;
use gadget_ng_core::{
    build_particles_for_gid_range, cosmology::CosmologyParams, density_contrast_rms,
    gravity_coupling_qksl, hubble_param, peculiar_vrms, wrap_position, DirectGravity,
    GravitySolver, IntegratorKind, OpeningCriterion, Particle, RunConfig, SfcKind, SolverKind,
    Vec3,
};
use gadget_ng_integrators::{
    hierarchical_kdk_step, leapfrog_cosmo_kdk_step, leapfrog_kdk_step, yoshida4_cosmo_kdk_step,
    yoshida4_kdk_step, CosmoFactors, HierarchicalState, StepStats, YOSHIDA4_W0, YOSHIDA4_W1,
};
use gadget_ng_io::SnapshotReader;
use gadget_ng_io::{
    write_snapshot_formatted, JsonlReader, JsonlWriter, Provenance, SnapshotEnv, SnapshotUnits,
    SnapshotWriter,
};
use gadget_ng_parallel::{gid_block_range, ParallelRuntime, SfcDecomposition, SlabDecomposition};
use gadget_ng_pm::distributed as pm_dist;
use gadget_ng_pm::slab_fft::SlabLayout;
use gadget_ng_pm::slab_pm;
use gadget_ng_pm::{solve_forces_pencil2d, PencilLayout2D, PmSolver};
#[cfg(feature = "simd")]
use gadget_ng_tree::RayonBarnesHutGravity;
use gadget_ng_tree::{
    accel_from_let, pack_let_nodes, unpack_let_nodes, walk_stats_begin, walk_stats_end,
    BarnesHutGravity, Octree,
};
use gadget_ng_treepm::{distributed as treepm_dist, TreePmSolver};
use std::fs;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::process::Command;
use std::time::Instant;

// ── Timing report ─────────────────────────────────────────────────────────────

/// Estadísticas detalladas HPC por evaluación de fuerza en el path SFC+LET.
///
/// Se serializa como campo `hpc_stats` en cada línea de `diagnostics.jsonl`.
/// Los tiempos son nanosegundos acumulados; se acumulan a lo largo de todas
/// las evaluaciones de fuerza dentro del paso (1 para leapfrog, 3 para Yoshida4).
#[derive(serde::Serialize, Default, Clone)]
struct HpcStepStats {
    /// Tiempo de construcción del octree local (ns).
    tree_build_ns: u64,
    /// Tiempo de exportación de nodos LET hacia todos los rangos remotos (ns).
    let_export_ns: u64,
    /// Tiempo de empaquetado de nodos LET — `pack_let_nodes` (ns).
    let_pack_ns: u64,
    /// Tiempo del allgather de AABBs (ns).
    aabb_allgather_ns: u64,
    /// Tiempo total del alltoallv LET (ns).
    /// - Path no-bloqueante: incluye el trabajo de overlap; `wait_ns ≈ let_alltoallv_ns - walk_local_ns`.
    /// - Path bloqueante: tiempo de espera puro de la colectiva.
    let_alltoallv_ns: u64,
    /// Tiempo del walk local del árbol (ns).
    /// En el path no-bloqueante, se solapa con `let_alltoallv_ns`.
    walk_local_ns: u64,
    /// Tiempo de aplicación de las fuerzas de nodos LET remotos (loop plano) (ns).
    apply_let_ns: u64,
    /// Nodos LET exportados a todos los rangos remotos en este paso.
    let_nodes_exported: usize,
    /// Nodos LET importados de todos los rangos remotos en este paso.
    let_nodes_imported: usize,
    /// Bytes enviados en alltoallv LET.
    bytes_sent: usize,
    /// Bytes recibidos en alltoallv LET.
    bytes_recv: usize,
    /// Tiempo de construcción del `LetTree` (ns). 0 si el path plano está activo.
    let_tree_build_ns: u64,
    /// Tiempo del walk del `LetTree` (N_local recorridos) (ns). 0 si el path plano está activo.
    let_tree_walk_ns: u64,
    /// Número de nodos en el `LetTree` construido. 0 si el path plano está activo.
    let_tree_nodes: usize,
    /// `true` si el walk del `LetTree` se ejecutó con Rayon (feature `simd`).
    let_tree_parallel: bool,
    /// Máximo de nodos LET exportados a un único rank remoto en este paso.
    max_let_nodes_per_rank: usize,
    /// Número total de nodos del árbol local (para calcular prune ratio).
    local_tree_nodes: usize,
    /// Tiempo de rebalanceo SFC (recompute cutpoints + bbox global) (ns).
    domain_rebalance_ns: u64,
    /// Tiempo de migración de partículas (exchange_domain_sfc) (ns).
    domain_migration_ns: u64,
    /// Partículas locales tras migración (count al inicio del paso).
    local_particle_count: usize,
    /// Tiempo en LetTree::apply_leaf total (ns) — profiling P14.
    apply_leaf_ns: u64,
    /// Total de RMNs procesados en hojas del LetTree — profiling P14.
    apply_leaf_rmn_count: u64,
    /// Número de llamadas a apply_leaf — profiling P14.
    apply_leaf_calls: u64,
    /// Tiempo en evaluaciones de nodos LET con RmnSoa (SoA path) (ns) — P14.
    rmn_soa_pack_ns: u64,
    /// Tiempo en accel_from_let_soa (flat LET SoA path) (ns) — P14.
    accel_from_let_soa_ns: u64,
    /// Número de llamadas a apply_leaf_soa_4xi (Fase 16).
    apply_leaf_tile_calls: u64,
    /// Suma de tile_size en cada llamada tileada (partículas-i procesadas en modo 4xi).
    apply_leaf_tile_i_count: u64,

    // ── Diagnósticos TreePM distribuido (Fase 21) ─────────────────────────────
    /// Partículas halo intercambiadas para corto alcance TreePM (suma de todos los pasos).
    short_range_halo_particles: usize,
    /// Bytes de partículas halo comunicados para corto alcance TreePM (suma de pasos).
    short_range_halo_bytes: usize,
    /// Tiempo acumulado en el árbol de corto alcance TreePM (ns).
    tree_short_ns: u64,
    /// Tiempo acumulado en el PM de largo alcance TreePM (ns).
    pm_long_ns: u64,
    /// Tiempo total acumulado en el pipeline TreePM distribuido (ns).
    treepm_total_ns: u64,
    /// Identificador del path activo: `"treepm_serial"` | `"treepm_allgather"` |
    /// `"treepm_slab_1d"` | `"treepm_slab_3d"`.
    path_active: String,

    // ── Diagnósticos halo 3D periódico (Fase 22) ──────────────────────────────
    /// Partículas halo recibidas por `exchange_halos_3d_periodic` en este paso.
    halo_3d_particles: usize,
    /// Bytes totales recibidos por `exchange_halos_3d_periodic` en este paso.
    halo_3d_bytes: usize,
    /// Tiempo acumulado en `exchange_halos_3d_periodic` (ns).
    halo_3d_ns: u64,

    // ── Diagnósticos dominio 3D/SFC para SR (Fase 23) ─────────────────────────
    /// Partículas locales en el dominio SFC SR al inicio de la evaluación de fuerza.
    sr_domain_particle_count: usize,
    /// Partículas halo vecinas recibidas para el SR (via `exchange_halos_3d_periodic`).
    sr_halo_3d_neighbors: usize,
    /// Tiempo acumulado en sincronización PM↔SR (clone + migraciones z↔SFC + lookup) (ns).
    sr_sync_ns: u64,

    // ── Diagnósticos scatter/gather PM (Fase 24) ──────────────────────────────
    /// Partículas enviadas al scatter PM (= partículas locales SFC) por paso.
    pm_scatter_particles: usize,
    /// Bytes totales enviados en el scatter PM (5 × f64 × scatter_particles).
    pm_scatter_bytes: usize,
    /// Tiempo del scatter alltoallv PM (ns).
    pm_scatter_ns: u64,
    /// Partículas recibidas en el gather PM por paso.
    pm_gather_particles: usize,
    /// Bytes totales recibidos en el gather PM (4 × f64 × gather_particles).
    pm_gather_bytes: usize,
    /// Tiempo del gather alltoallv PM (ns).
    pm_gather_ns: u64,
}

/// Resumen HPC agregado incluido en `timings.json`.
#[derive(serde::Serialize)]
struct HpcTimingsAggregate {
    mean_tree_build_s: f64,
    mean_let_export_s: f64,
    mean_let_pack_s: f64,
    mean_aabb_allgather_s: f64,
    mean_let_alltoallv_s: f64,
    mean_walk_local_s: f64,
    mean_apply_let_s: f64,
    mean_let_nodes_exported: f64,
    mean_let_nodes_imported: f64,
    mean_bytes_sent: f64,
    mean_bytes_recv: f64,
    /// Fracción del tiempo total de paso gastada esperando MPI (alltoallv − walk_local).
    wait_fraction: f64,
    /// Tiempo medio de construcción del `LetTree` por paso (s). 0 si path plano activo.
    mean_let_tree_build_s: f64,
    /// Tiempo medio del walk del `LetTree` por paso (s). 0 si path plano activo.
    mean_let_tree_walk_s: f64,
    /// Media de nodos en el `LetTree` por paso. 0 si path plano activo.
    mean_let_tree_nodes: f64,
    /// `true` si el walk del `LetTree` usó Rayon (feature `simd`).
    let_tree_parallel: bool,
    /// Media del máximo de nodos LET por rank remoto.
    mean_max_let_nodes_per_rank: f64,
    /// Media del número de nodos del árbol local.
    mean_local_tree_nodes: f64,
    /// Ratio de poda: `let_nodes_exported / (local_tree_nodes * (P-1))`.
    /// Mide qué fracción del árbol se exporta en promedio por rank remoto.
    mean_export_prune_ratio: f64,
    /// Tiempo medio de rebalanceo SFC (recompute cutpoints) por paso (s).
    mean_domain_rebalance_s: f64,
    /// Tiempo medio de migración de partículas por paso (s).
    mean_domain_migration_s: f64,
    /// Media de partículas locales por paso.
    mean_local_particle_count: f64,
    /// Ratio de imbalance de partículas: max_count / min_count entre ranks.
    /// Calculado al final de la simulación con el último paso.
    particle_imbalance_ratio: f64,
    /// Curva SFC usada: "morton" o "hilbert".
    sfc_kind: String,
    /// Tiempo medio en LetTree::apply_leaf por paso (s) — profiling P14.
    mean_apply_leaf_s: f64,
    /// Media de RMNs procesados en hojas por paso — profiling P14.
    mean_apply_leaf_rmn_count: f64,
    /// Media de llamadas a apply_leaf por paso — profiling P14.
    mean_apply_leaf_calls: f64,
    /// Tiempo medio de empaquetado AoS→SoA (RmnSoa::from_slice) por paso (s) — P14.
    mean_rmn_soa_pack_s: f64,
    /// Tiempo medio en accel_from_let_soa (flat LET SoA) por paso (s) — P14.
    mean_accel_from_let_soa_s: f64,
    /// Si el path SoA está activo (feature simd).
    soa_simd_active: bool,
    /// Media de llamadas a apply_leaf_soa_4xi (Fase 16) por paso.
    mean_apply_leaf_tile_calls: f64,
    /// Media de partículas-i procesadas en modo 4xi por paso.
    mean_apply_leaf_tile_i_count: f64,
    /// Ratio de utilización de tiles: tile_i_count / (tile_calls * 4). Ideal = 1.0.
    tile_utilization_ratio: f64,

    // ── TreePM distribuido (Fase 21) ──────────────────────────────────────────
    /// Media de partículas halo de corto alcance intercambiadas por paso.
    mean_short_range_halo_particles: f64,
    /// Media de bytes de halo de corto alcance por paso.
    mean_short_range_halo_bytes: f64,
    /// Tiempo medio del árbol de corto alcance TreePM por paso (s).
    mean_tree_short_s: f64,
    /// Tiempo medio del PM de largo alcance TreePM por paso (s).
    mean_pm_long_s: f64,
    /// Tiempo medio total del pipeline TreePM distribuido por paso (s).
    mean_treepm_total_s: f64,
    /// Fracción del tiempo TreePM gastada en el árbol de corto alcance.
    tree_fraction: f64,
    /// Fracción del tiempo TreePM gastada en el PM de largo alcance.
    pm_fraction: f64,
    /// Path activo del TreePM: `"treepm_serial"` | `"treepm_allgather"` |
    /// `"treepm_slab_1d"` | `"treepm_slab_3d"`.
    path_active: String,

    // ── Halo 3D periódico (Fase 22) ───────────────────────────────────────────
    /// Media de partículas halo 3D por paso (halo volumétrico periódico).
    mean_halo_3d_particles: f64,
    /// Media de bytes de halo 3D por paso.
    mean_halo_3d_bytes: f64,
    /// Tiempo medio de `exchange_halos_3d_periodic` por paso (s).
    mean_halo_3d_s: f64,

    // ── Dominio 3D/SFC SR (Fase 23) ───────────────────────────────────────────
    /// Media de partículas en dominio SFC SR por paso.
    mean_sr_domain_particle_count: f64,
    /// Media de partículas halo SR vecinas (via halo 3D periódico) por paso.
    mean_sr_halo_3d_neighbors: f64,
    /// Tiempo medio de sincronización PM↔SR por paso (s).
    mean_sr_sync_s: f64,
    /// Fracción del tiempo TreePM gastada en sincronización PM↔SR.
    sr_sync_fraction: f64,

    // ── Scatter/Gather PM (Fase 24) ───────────────────────────────────────────
    /// Media de partículas enviadas en el scatter PM por paso.
    mean_pm_scatter_particles: f64,
    /// Media de bytes enviados en el scatter PM por paso.
    mean_pm_scatter_bytes: f64,
    /// Tiempo medio del scatter PM alltoallv por paso (s).
    mean_pm_scatter_s: f64,
    /// Media de partículas recibidas en el gather PM por paso.
    mean_pm_gather_particles: f64,
    /// Media de bytes recibidos en el gather PM por paso.
    mean_pm_gather_bytes: f64,
    /// Tiempo medio del gather PM alltoallv por paso (s).
    mean_pm_gather_s: f64,
    /// Fracción del tiempo TreePM gastada en scatter+gather PM (Fase 24 vs Fase 23 sr_sync).
    pm_sync_fraction: f64,
}

/// Resumen HPC agregado para el path TreePM SR-SFC (Fases 23/24).
///
/// Escrito como campo `"treepm_hpc"` en `timings.json` al final del run.
/// Solo se emite si el path `use_treepm_sr_sfc` estuvo activo.
#[derive(serde::Serialize)]
struct TreePmAggregate {
    /// Tiempo medio de scatter alltoallv PM (o clone+migrate) por paso (segundos).
    mean_scatter_s: f64,
    /// Tiempo medio de gather alltoallv PM por paso (segundos). 0 en Fase 23.
    mean_gather_s: f64,
    /// Tiempo medio de resolución PM (FFT + interpolación) por paso (segundos).
    mean_pm_solve_s: f64,
    /// Tiempo medio de intercambio de halos SR 3D por paso (segundos).
    mean_sr_halo_s: f64,
    /// Tiempo medio del árbol corto alcance por paso (segundos).
    mean_tree_sr_s: f64,
    /// Media de partículas enviadas en scatter PM (o clonadas) por paso.
    mean_scatter_particles: f64,
    /// Media de bytes enviados en scatter por paso.
    mean_scatter_bytes: f64,
    /// Media de bytes recibidos en gather por paso.
    mean_gather_bytes: f64,
    /// Fracción del tiempo TreePM gastada en scatter+gather (sincronización PM↔SR).
    /// `(scatter_ns + gather_ns) / (scatter_ns + gather_ns + pm_solve_ns + sr_halo_ns + tree_sr_ns)`
    pm_sync_fraction: f64,
    /// Tiempo medio total del path TreePM SR-SFC por paso (segundos).
    mean_treepm_total_s: f64,
    /// Path activo: `"scatter_gather"` (Fase 24) o `"clone_migrate"` (Fase 23).
    path_active: &'static str,
}

/// Resumen de tiempos por fase, escrito en `<out>/timings.json` al final del run.
///
/// Permite medir el desglose entre comunicación MPI, cálculo de fuerzas e integración
/// sin necesidad de herramientas externas de profiling.
#[derive(serde::Serialize)]
struct TimingsReport {
    /// Número de pasos ejecutados.
    steps: u64,
    /// Número de partículas totales.
    total_particles: usize,
    /// Tiempo de pared total del loop de integración (segundos).
    total_wall_s: f64,
    /// Tiempo acumulado en comunicación MPI / allgather (segundos).
    total_comm_s: f64,
    /// Tiempo acumulado en cálculo de fuerzas gravitatorias (segundos).
    total_gravity_s: f64,
    /// Tiempo acumulado en kicks+drifts de integración (segundos).
    total_integration_s: f64,
    /// Tiempo de pared medio por paso (segundos).
    mean_step_wall_s: f64,
    /// Tiempo medio de comunicación por paso (segundos).
    mean_comm_s: f64,
    /// Tiempo medio de fuerza gravitatoria por paso (segundos).
    mean_gravity_s: f64,
    /// Fracción del tiempo total gastada en comunicación.
    comm_fraction: f64,
    /// Fracción del tiempo total gastada en fuerzas.
    gravity_fraction: f64,
    /// Resumen detallado HPC (solo para path SFC+LET).
    #[serde(skip_serializing_if = "Option::is_none")]
    hpc: Option<HpcTimingsAggregate>,
    /// Resumen HPC del path TreePM SR-SFC (Fases 23/24). Solo se emite si estuvo activo.
    #[serde(skip_serializing_if = "Option::is_none")]
    treepm_hpc: Option<TreePmAggregate>,
}

// ── Checkpoint ────────────────────────────────────────────────────────────────

#[derive(serde::Serialize, serde::Deserialize)]
struct CheckpointMeta {
    schema_version: u32,
    /// Último paso completado (el siguiente paso a ejecutar es `completed_step + 1`).
    completed_step: u64,
    /// Factor de escala al final de `completed_step` (1.0 si no hay cosmología).
    a_current: f64,
    /// Hash SHA-256 del TOML canónico, para detectar cambios de config al reanudar.
    config_hash: String,
    /// Número total de partículas (verificación).
    total_particles: usize,
    /// `true` si también se guardó `hierarchical_state.json`.
    has_hierarchical_state: bool,
    /// Informativo: el `SfcDecomposition` no se serializa; se reconstruye al reanudar
    /// desde las posiciones restauradas.  Siempre `false` en el archivo.
    #[serde(default)]
    sfc_state_saved: bool,
    /// `true` si también se guardó `agn_bhs.json` (Phase 106).
    #[serde(default)]
    has_agn_state: bool,
    /// `true` si también se guardó `chem_states.json` (Phase 106).
    #[serde(default)]
    has_chem_state: bool,
}

/// Guarda estado de checkpoint en `<out_dir>/checkpoint/`.
///
/// Solo rank 0 escribe; el directorio se sobreescribe en cada checkpoint
/// (siempre representa el último paso completado).
///
/// Phase 106: incluye estado AGN (`agn_bhs.json`) y química (`chem_states.json`).
#[allow(clippy::too_many_arguments)]
fn save_checkpoint<R: ParallelRuntime + ?Sized>(
    rt: &R,
    completed_step: u64,
    a_current: f64,
    local: &[Particle],
    total: usize,
    h_state: Option<&HierarchicalState>,
    out_dir: &Path,
    cfg_hash: &str,
    agn_bhs: &[gadget_ng_sph::BlackHole],
    chem_states: &[gadget_ng_rt::ChemState],
) -> Result<(), CliError> {
    let ck_dir = out_dir.join("checkpoint");
    // Recopilar todas las partículas en rank 0 y escribir.
    if let Some(all) = rt.root_gather_particles(local, total) {
        fs::create_dir_all(&ck_dir).map_err(|e| CliError::io(&ck_dir, e))?;
        // Partículas en JSONL (siempre, independientemente del formato de snapshot).
        let dummy_prov = Provenance::new("checkpoint", None, "release", vec![], vec![], cfg_hash);
        let env = SnapshotEnv::default();
        JsonlWriter.write(&ck_dir, &all, &dummy_prov, &env)?;
        // Guardar estado jerárquico si existe.
        if let Some(hs) = h_state {
            hs.save(&ck_dir).map_err(|e| CliError::io(&ck_dir, e))?;
        }
        // Phase 106: guardar estado AGN si hay agujeros negros activos.
        let has_agn = !agn_bhs.is_empty();
        if has_agn {
            let agn_path = ck_dir.join("agn_bhs.json");
            fs::write(&agn_path, serde_json::to_string_pretty(agn_bhs)?)
                .map_err(|e| CliError::io(&agn_path, e))?;
        }
        // Phase 106: guardar estados de química si están activos.
        let has_chem = !chem_states.is_empty();
        if has_chem {
            let chem_path = ck_dir.join("chem_states.json");
            fs::write(&chem_path, serde_json::to_string_pretty(chem_states)?)
                .map_err(|e| CliError::io(&chem_path, e))?;
        }
        // meta.json del checkpoint (diferente al meta.json del snapshot).
        let meta = CheckpointMeta {
            schema_version: 1,
            completed_step,
            a_current,
            config_hash: cfg_hash.to_owned(),
            total_particles: total,
            has_hierarchical_state: h_state.is_some(),
            sfc_state_saved: false,
            has_agn_state: has_agn,
            has_chem_state: has_chem,
        };
        let meta_path = ck_dir.join("checkpoint.json");
        fs::write(&meta_path, serde_json::to_string_pretty(&meta)?)
            .map_err(|e| CliError::io(&meta_path, e))?;
    }
    rt.barrier();
    Ok(())
}

/// Carga el estado de checkpoint desde `<resume_dir>/checkpoint/`.
///
/// Devuelve `(partículas_locales, completed_step, a_current, h_state_opt,
///           agn_bhs_opt, chem_states_opt)`.
///
/// Phase 106: incluye estado AGN y química si fueron guardados.
fn load_checkpoint<R: ParallelRuntime + ?Sized>(
    rt: &R,
    resume_dir: &Path,
    lo: usize,
    hi: usize,
    cfg_hash: &str,
) -> Result<(
    Vec<Particle>,
    u64,
    f64,
    Option<HierarchicalState>,
    Option<Vec<gadget_ng_sph::BlackHole>>,
    Option<Vec<gadget_ng_rt::ChemState>>,
), CliError> {
    let ck_dir = resume_dir.join("checkpoint");
    let meta_path = ck_dir.join("checkpoint.json");
    let meta_str = fs::read_to_string(&meta_path).map_err(|e| CliError::io(&meta_path, e))?;
    let meta: CheckpointMeta = serde_json::from_str(&meta_str)?;
    if meta.config_hash != cfg_hash {
        rt.root_eprintln(&format!(
            "[gadget-ng] ADVERTENCIA: el hash del config ha cambiado \
             desde que se guardó el checkpoint (esperado {}, actual {}). \
             Los resultados pueden diferir.",
            meta.config_hash, cfg_hash
        ));
    }
    // Leer todas las partículas y filtrar las que corresponden a este rango.
    let data = JsonlReader.read(&ck_dir)?;
    let local: Vec<Particle> = data
        .particles
        .into_iter()
        .filter(|p| p.global_id >= lo && p.global_id < hi)
        .collect();
    // Estado jerárquico (opcional).
    let h_state = if meta.has_hierarchical_state {
        Some(HierarchicalState::load(&ck_dir).map_err(|e| CliError::io(&ck_dir, e))?)
    } else {
        None
    };
    // Phase 106: cargar estado AGN si fue guardado.
    let agn_bhs = if meta.has_agn_state {
        let agn_path = ck_dir.join("agn_bhs.json");
        let s = fs::read_to_string(&agn_path).map_err(|e| CliError::io(&agn_path, e))?;
        let bhs: Vec<gadget_ng_sph::BlackHole> = serde_json::from_str(&s)?;
        Some(bhs)
    } else {
        None
    };
    // Phase 106: cargar estados de química si fueron guardados.
    let chem_states = if meta.has_chem_state {
        let chem_path = ck_dir.join("chem_states.json");
        let s = fs::read_to_string(&chem_path).map_err(|e| CliError::io(&chem_path, e))?;
        let cs: Vec<gadget_ng_rt::ChemState> = serde_json::from_str(&s)?;
        Some(cs)
    } else {
        None
    };
    Ok((local, meta.completed_step, meta.a_current, h_state, agn_bhs, chem_states))
}

pub fn cmd_config_print(cfg_path: &Path) -> Result<(), CliError> {
    let cfg = config_load::load_run_config(cfg_path)?;
    config_load::print_resolved_config(&cfg)?;
    let hash = config_load::config_canonical_hash(&cfg)?;
    println!("canonical_toml_sha256={hash}");
    Ok(())
}

fn try_git_commit() -> Option<String> {
    let out = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()?;
    if out.status.success() {
        String::from_utf8(out.stdout)
            .ok()
            .map(|s| s.trim().to_string())
    } else {
        None
    }
}

/// Decide si rebalancear el dominio SFC en este paso.
///
/// Retorna `true` si se cumple cualquiera de:
/// 1. `cost_pending` — el último paso midió un desbalance de carga que supera
///    `rebalance_imbalance_threshold` (criterio dinámico).
/// 2. El intervalo de rebalanceo fijo se cumple: `(step - start) % interval == 0`
///    (cuando `interval > 0`).
///
/// # Parámetros
/// - `step`: paso actual (comenzando en `start_step`).
/// - `start_step`: primer paso de la corrida (puede ser > 0 después de un restart).
/// - `interval`: `cfg.performance.sfc_rebalance_interval`.
/// - `cost_pending`: bandera levantada cuando `max_ns/min_ns > threshold`.
#[inline]
fn should_rebalance(step: u64, start_step: u64, interval: u64, cost_pending: bool) -> bool {
    if cost_pending {
        return true;
    }
    if interval == 0 {
        return true;
    }
    (step - start_step) % interval == 0
}

fn kinetic_local(parts: &[Particle]) -> f64 {
    parts
        .iter()
        .map(|p| 0.5 * p.mass * p.velocity.dot(p.velocity))
        .sum()
}

/// Agregados locales O(N) usados en los diagnósticos por paso.
#[derive(Clone, Copy, Default)]
struct LocalMoments {
    /// Momento lineal total: Σ mᵢ vᵢ.
    p: [f64; 3],
    /// Momento angular total respecto al origen: Σ mᵢ (rᵢ × vᵢ).
    l: [f64; 3],
    /// Σ mᵢ rᵢ (para el centro de masa).
    mass_weighted_pos: [f64; 3],
    /// Σ mᵢ.
    mass: f64,
}

fn local_moments(parts: &[Particle]) -> LocalMoments {
    let mut m = LocalMoments::default();
    for p in parts {
        let w = p.mass;
        let r = &p.position;
        let v = &p.velocity;
        m.mass += w;
        m.p[0] += w * v.x;
        m.p[1] += w * v.y;
        m.p[2] += w * v.z;
        m.l[0] += w * (r.y * v.z - r.z * v.y);
        m.l[1] += w * (r.z * v.x - r.x * v.z);
        m.l[2] += w * (r.x * v.y - r.y * v.x);
        m.mass_weighted_pos[0] += w * r.x;
        m.mass_weighted_pos[1] += w * r.y;
        m.mass_weighted_pos[2] += w * r.z;
    }
    m
}

/// Diagnóstico por llamada a compute_acc en el path TreePM SR-SFC (Fases 23/24).
///
/// Se acumula sobre todas las sub-llamadas de un paso (1 para leapfrog, 3 para Yoshida4)
/// y se serializa como campo `"treepm"` en cada línea de `diagnostics.jsonl`.
#[derive(serde::Serialize, Default, Clone, Copy)]
struct TreePmStepDiag {
    /// Tiempo total de scatter alltoallv PM (Fase 24) o clone+migrate (Fase 23), en ns.
    scatter_ns: u64,
    /// Tiempo total de gather alltoallv PM (Fase 24), en ns. 0 para Fase 23.
    gather_ns: u64,
    /// Tiempo total de resolución PM (FFT + interpolación) en el slab, en ns.
    pm_solve_ns: u64,
    /// Tiempo total de intercambio de halos SR 3D periódico, en ns.
    sr_halo_ns: u64,
    /// Tiempo total del árbol corto alcance (SFC domain), en ns.
    tree_sr_ns: u64,
    /// Número de partículas enviadas en scatter PM (Fase 24) o clonadas (Fase 23).
    scatter_particles: usize,
    /// Bytes enviados en scatter PM por este rank (Fase 24). 0 para Fase 23.
    scatter_bytes: usize,
    /// Bytes recibidos en gather PM por este rank (Fase 24). 0 para Fase 23.
    gather_bytes: usize,
    /// Path activo: `"sg"` (Fase 24 scatter/gather) o `"clone"` (Fase 23 clone+migrate).
    path: &'static str,
}

impl TreePmStepDiag {
    /// Acumula otro `TreePmStepDiag` (suma campos numéricos; preserva `path` del `other`).
    fn add(self, other: Self) -> Self {
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
struct CosmoDiag {
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

#[allow(clippy::too_many_arguments)]
fn write_diagnostic_line<R: ParallelRuntime + ?Sized>(
    rt: &R,
    step: u64,
    local: &[Particle],
    diag_path: &Path,
    diag_file: &mut Option<File>,
    step_stats: Option<&StepStats>,
    hpc_stats: Option<&HpcStepStats>,
    cosmo_diag: Option<&CosmoDiag>,
) -> Result<(), CliError> {
    let ke_loc = kinetic_local(local);
    let ke = rt.allreduce_sum_f64(ke_loc);
    // Agregados O(N): p, L, COM. Usan 10 allreduces; coste despreciable frente al paso.
    let lm = local_moments(local);
    let px = rt.allreduce_sum_f64(lm.p[0]);
    let py = rt.allreduce_sum_f64(lm.p[1]);
    let pz = rt.allreduce_sum_f64(lm.p[2]);
    let lx = rt.allreduce_sum_f64(lm.l[0]);
    let ly = rt.allreduce_sum_f64(lm.l[1]);
    let lz = rt.allreduce_sum_f64(lm.l[2]);
    let mrx = rt.allreduce_sum_f64(lm.mass_weighted_pos[0]);
    let mry = rt.allreduce_sum_f64(lm.mass_weighted_pos[1]);
    let mrz = rt.allreduce_sum_f64(lm.mass_weighted_pos[2]);
    let mtot = rt.allreduce_sum_f64(lm.mass);
    let com = if mtot > 0.0 {
        [mrx / mtot, mry / mtot, mrz / mtot]
    } else {
        [0.0, 0.0, 0.0]
    };
    if let Some(ref mut f) = diag_file {
        let mut obj = serde_json::json!({
            "step": step,
            "kinetic_energy": ke,
            "momentum": [px, py, pz],
            "angular_momentum": [lx, ly, lz],
            "com": com,
            "mass_total": mtot,
        });
        // Si se proveen estadísticas del paso jerárquico, añadirlas como campos opcionales.
        if let Some(ss) = step_stats {
            let map = obj.as_object_mut().unwrap();
            map.insert(
                "level_histogram".into(),
                serde_json::Value::Array(
                    ss.level_histogram
                        .iter()
                        .map(|&v| serde_json::Value::Number(v.into()))
                        .collect(),
                ),
            );
            map.insert("active_total".into(), ss.active_total.into());
            map.insert("force_evals".into(), ss.force_evals.into());
            map.insert("dt_min_effective".into(), ss.dt_min_effective.into());
            map.insert("dt_max_effective".into(), ss.dt_max_effective.into());
        }
        if let Some(hs) = hpc_stats {
            let map = obj.as_object_mut().unwrap();
            map.insert(
                "hpc_stats".into(),
                serde_json::to_value(hs).unwrap_or(serde_json::Value::Null),
            );
        }
        // Campos cosmológicos opcionales.
        if let Some(cd) = cosmo_diag {
            let map = obj.as_object_mut().unwrap();
            map.insert("a".into(), cd.a.into());
            map.insert("z".into(), cd.z.into());
            map.insert("v_rms".into(), cd.v_rms.into());
            map.insert("delta_rms".into(), cd.delta_rms.into());
            map.insert("hubble".into(), cd.hubble.into());
            // Diagnóstico TreePM SR-SFC (Fases 23/24) si estuvo activo en este paso.
            if let Some(td) = cd.treepm {
                map.insert(
                    "treepm".into(),
                    serde_json::to_value(td).unwrap_or(serde_json::Value::Null),
                );
            }
        }
        let line = obj.to_string();
        writeln!(f, "{line}").map_err(|e| CliError::io(diag_path, e))?;
    }
    rt.barrier();
    Ok(())
}

/// Calcula aceleraciones para `parts` (partículas locales) usando un árbol construido
/// a partir de `parts` + `halos` (partículas de rangos vecinos).
///
/// - `parts[0..n_local]`  → partículas propias; sus índices locales (0..n_local) sirven
///   para la auto-exclusión en el árbol.
/// - `halos` → partículas recibidas del halo; se incluyen en el árbol pero no se
///   computan sus aceleraciones.
fn compute_forces_local_tree(
    parts: &[Particle],
    halos: &[Particle],
    theta: f64,
    g: f64,
    eps2: f64,
    out: &mut [Vec3],
) {
    debug_assert_eq!(parts.len(), out.len());
    if parts.is_empty() {
        return;
    }
    let all_pos: Vec<Vec3> = parts
        .iter()
        .chain(halos.iter())
        .map(|p| p.position)
        .collect();
    let all_mass: Vec<f64> = parts.iter().chain(halos.iter()).map(|p| p.mass).collect();
    let tree = Octree::build(&all_pos, &all_mass);
    for (li, acc_out) in out.iter_mut().enumerate() {
        *acc_out = tree.walk_accel(parts[li].position, li, g, eps2, theta, &all_pos, &all_mass);
    }
}

/// Variante de `compute_forces_local_tree` que además devuelve el coste de interacción
/// (nodos abiertos del walk) por partícula local. Se usa para el balanceo SFC ponderado.
fn compute_forces_local_tree_with_costs(
    parts: &[Particle],
    halos: &[Particle],
    theta: f64,
    g: f64,
    eps2: f64,
    out: &mut [Vec3],
    costs: &mut Vec<u64>,
) {
    debug_assert_eq!(parts.len(), out.len());
    costs.clear();
    if parts.is_empty() {
        return;
    }
    let all_pos: Vec<Vec3> = parts
        .iter()
        .chain(halos.iter())
        .map(|p| p.position)
        .collect();
    let all_mass: Vec<f64> = parts.iter().chain(halos.iter()).map(|p| p.mass).collect();
    let tree = Octree::build(&all_pos, &all_mass);
    for (li, acc_out) in out.iter_mut().enumerate() {
        walk_stats_begin();
        *acc_out = tree.walk_accel(parts[li].position, li, g, eps2, theta, &all_pos, &all_mass);
        let stats = walk_stats_end();
        costs.push(stats.opened_nodes);
    }
}

/// Calcula aceleraciones solo para las partículas activas en `active_local`,
/// usando árbol construido con `parts` + `halos` del rank vecino.
///
/// Variante jerárquica de `compute_forces_local_tree`: al integrador de block
/// timesteps solo le interesan las fuerzas de las partículas activas en este
/// subpaso; las inactivas usan el predictor de Störmer.
///
/// - `parts[active_local[j]]` → `acc[j]` (tamaño de `acc` = `active_local.len()`).
/// - Índice de auto-exclusión: se pasa `active_local[j]` al walk para evitar la
///   auto-interacción con la partícula evaluada dentro del árbol local.
fn compute_forces_hierarchical_let(
    parts: &[Particle],
    halos: &[Particle],
    active_local: &[usize],
    theta: f64,
    g: f64,
    eps2: f64,
    acc: &mut [Vec3],
) {
    debug_assert_eq!(acc.len(), active_local.len());
    if parts.is_empty() || active_local.is_empty() {
        return;
    }
    let all_pos: Vec<Vec3> = parts
        .iter()
        .chain(halos.iter())
        .map(|p| p.position)
        .collect();
    let all_mass: Vec<f64> = parts.iter().chain(halos.iter()).map(|p| p.mass).collect();
    let tree = Octree::build(&all_pos, &all_mass);
    for (j, &li) in active_local.iter().enumerate() {
        acc[j] = tree.walk_accel(parts[li].position, li, g, eps2, theta, &all_pos, &all_mass);
    }
}

/// Calcula aceleraciones para `parts` usando árbol local + nodos LET remotos.
///
/// 1. Construye un árbol con solo las partículas locales.
/// 2. Para cada partícula, aplica la fuerza del árbol local (con auto-exclusión).
/// 3. Suma la contribución de los nodos multipolares remotos (`remote_let_bufs`,
///    buffers wire en `f64` empaquetados con [`pack_let_nodes`]).
///
/// Esta función implementa el kernel SFC+LET de Fase 8.
fn compute_forces_sfc_let(
    parts: &[Particle],
    remote_let_bufs: &[Vec<f64>],
    theta: f64,
    g: f64,
    eps2: f64,
    out: &mut [Vec3],
) {
    debug_assert_eq!(parts.len(), out.len());
    if parts.is_empty() {
        return;
    }
    let all_pos: Vec<Vec3> = parts.iter().map(|p| p.position).collect();
    let all_mass: Vec<f64> = parts.iter().map(|p| p.mass).collect();
    let tree = Octree::build(&all_pos, &all_mass);

    // Desempaquetar nodos LET remotos (verificar múltiplo de RMN_FLOATS antes).
    let mut remote_nodes = Vec::new();
    for buf in remote_let_bufs {
        if !buf.is_empty() {
            remote_nodes.extend(unpack_let_nodes(buf));
        }
    }

    for (li, acc_out) in out.iter_mut().enumerate() {
        let a_local = tree.walk_accel(parts[li].position, li, g, eps2, theta, &all_pos, &all_mass);
        let a_remote = accel_from_let(parts[li].position, &remote_nodes, g, eps2);
        *acc_out = a_local + a_remote;
    }
}

fn make_solver(cfg: &RunConfig) -> Box<dyn GravitySolver> {
    // Solver GPU wgpu — activado con `[performance] use_gpu = true` en el TOML.
    // Requiere compilar con `--features gpu`. Si no hay GPU disponible en el host
    // (headless, CI), `try_new()` devuelve None y se continúa con el solver CPU.
    #[cfg(feature = "gpu")]
    if cfg.performance.use_gpu {
        if let Some(gpu) = gadget_ng_core::GpuDirectGravity::try_new() {
            eprintln!("[gadget-ng] GPU wgpu activado (gravedad directa f32).");
            return Box::new(gpu);
        }
        eprintln!("[gadget-ng] ADVERTENCIA: use_gpu=true pero no hay GPU disponible; usando CPU.");
    }

    // Solver PM CUDA — activado con `[performance] use_gpu_cuda = true`.
    // Requiere `--features cuda`. Si nvcc no estaba disponible en build time o no hay
    // dispositivo CUDA, `try_new()` devuelve None y se continúa con el solver CPU.
    #[cfg(feature = "cuda")]
    if cfg.performance.use_gpu_cuda && cfg.gravity.solver == SolverKind::Pm {
        if let Some(solver) =
            gadget_ng_cuda::CudaPmSolver::try_new(cfg.gravity.pm_grid_size, cfg.simulation.box_size)
        {
            eprintln!(
                "[gadget-ng] PM CUDA activado (grilla {}³).",
                solver.grid_size()
            );
            return Box::new(solver);
        }
        eprintln!(
            "[gadget-ng] ADVERTENCIA: use_gpu_cuda=true pero CUDA no disponible; usando CPU PM."
        );
    }

    // Solver PM HIP — activado con `[performance] use_gpu_hip = true`.
    // `use_gpu_cuda` tiene precedencia si ambos están en true.
    // Requiere `--features hip`. Si hipcc no estaba disponible en build time o no hay
    // dispositivo ROCm, `try_new()` devuelve None y se continúa con el solver CPU.
    #[cfg(feature = "hip")]
    if cfg.performance.use_gpu_hip && cfg.gravity.solver == SolverKind::Pm {
        if let Some(solver) =
            gadget_ng_hip::HipPmSolver::try_new(cfg.gravity.pm_grid_size, cfg.simulation.box_size)
        {
            eprintln!(
                "[gadget-ng] PM HIP/ROCm activado (grilla {}³).",
                solver.grid_size()
            );
            return Box::new(solver);
        }
        eprintln!(
            "[gadget-ng] ADVERTENCIA: use_gpu_hip=true pero HIP/ROCm no disponible; usando CPU PM."
        );
    }

    // Los solvers PM y TreePM no usan Rayon; se enrutan antes del bloque SIMD.
    if cfg.gravity.solver == SolverKind::Pm {
        return Box::new(PmSolver {
            grid_size: cfg.gravity.pm_grid_size,
            box_size: cfg.simulation.box_size,
        });
    }
    if cfg.gravity.solver == SolverKind::TreePm {
        return Box::new(TreePmSolver {
            grid_size: cfg.gravity.pm_grid_size,
            box_size: cfg.simulation.box_size,
            r_split: cfg.gravity.r_split,
        });
    }

    #[cfg(feature = "simd")]
    if !cfg.performance.deterministic {
        if let Some(n) = cfg.performance.num_threads {
            // Intentar configurar el pool global de Rayon; si ya está inicializado se ignora.
            let _ = rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .build_global();
        }
        return match cfg.gravity.solver {
            SolverKind::Direct => Box::new(RayonDirectGravity),
            SolverKind::BarnesHut => Box::new(RayonBarnesHutGravity {
                theta: cfg.gravity.theta,
                multipole_order: cfg.gravity.multipole_order,
                use_relative_criterion: cfg.gravity.opening_criterion == OpeningCriterion::Relative,
                err_tol_force_acc: cfg.gravity.err_tol_force_acc,
                softened_multipoles: cfg.gravity.softened_multipoles,
                mac_softening: cfg.gravity.mac_softening,
            }),
            SolverKind::Pm | SolverKind::TreePm => unreachable!("handled above"),
        };
    }
    // Modo serial (default): determinismo garantizado.
    match cfg.gravity.solver {
        SolverKind::Direct => Box::new(DirectGravity),
        SolverKind::BarnesHut => Box::new(BarnesHutGravity {
            theta: cfg.gravity.theta,
            multipole_order: cfg.gravity.multipole_order,
            use_relative_criterion: cfg.gravity.opening_criterion == OpeningCriterion::Relative,
            err_tol_force_acc: cfg.gravity.err_tol_force_acc,
            softened_multipoles: cfg.gravity.softened_multipoles,
            mac_softening: cfg.gravity.mac_softening,
        }),
        SolverKind::Pm | SolverKind::TreePm => unreachable!("handled above"),
    }
}

/// Ejecuta el integrador leapfrog KDK.
///
/// `resume_from`: `Some(dir)` → reanudar desde el checkpoint guardado en `<dir>/checkpoint/`.
pub fn run_stepping<R: ParallelRuntime + ?Sized>(
    rt: &R,
    cfg: &RunConfig,
    out_dir: &Path,
    write_final_snapshot: bool,
    resume_from: Option<&Path>,
) -> Result<(), CliError> {
    let total = cfg.simulation.particle_count;
    let (lo, hi) = gid_block_range(total, rt.rank(), rt.size());

    let g = cfg.effective_g();

    // ── Diagnóstico de consistencia cosmológica de G (Phase 51) ──────────────
    if let Some((g_consistent, rel_err)) = cfg.cosmo_g_diagnostic() {
        if cfg.cosmology.auto_g {
            // auto_g activo: G fue calculado por g_code_consistent, informar.
            rt.root_eprintln(&format!(
                "[gadget-ng] cosmology.auto_g=true → G auto-consistente: {g:.4e} \
                 (3·Ω_m·H₀²/8π, condición de Friedmann satisfecha)"
            ));
        } else if rel_err > 0.01 {
            // G manual está >1% fuera de la condición de Friedmann.
            rt.root_eprintln(&format!(
                "[gadget-ng] ADVERTENCIA: G ({g:.4e}) inconsistente con cosmología \
                 ({:.1}% fuera de G_consistente={g_consistent:.4e}). \
                 Usa [cosmology] auto_g = true para corregir automáticamente.",
                rel_err * 100.0
            ));
        }
        // Si rel_err ≤ 1% y auto_g = false: G manual ya es consistente, no se avisa.
    }

    let eps2_base = cfg.softening_squared();
    let physical_softening = cfg.simulation.physical_softening && cfg.cosmology.enabled;
    // Calcula eps2 efectivo para un dado factor de escala.
    // Con physical_softening=true: ε_com(a) = softening/a → eps2 = (softening/a)².
    // Con physical_softening=false: eps2 = softening² constante (comportamiento legacy).
    let eps2_at = |a: f64| -> f64 {
        if physical_softening {
            let eps_com = cfg.simulation.softening / a;
            eps_com * eps_com
        } else {
            eps2_base
        }
    };
    // eps2 general: constante para paths no-cosmológicos. En los loops cosmológicos
    // se recalcula por paso con `eps2_at(a_current)`.
    let eps2 = eps2_base;
    let theta = cfg.gravity.theta;
    let solver = make_solver(cfg);
    let dt = cfg.simulation.dt;
    let checkpoint_interval = cfg.output.checkpoint_interval;
    let snapshot_interval = cfg.output.snapshot_interval;

    // Hash canónico de la config (para checkpoint).
    let cfg_hash = config_load::config_canonical_hash(cfg).unwrap_or_else(|_| "unknown".to_owned());

    // ── Inicialización de estado ──────────────────────────────────────────────
    // Si `resume_from` está presente, cargamos el checkpoint;
    // si no, construimos las condiciones iniciales desde la config.
    // Phase 106: almacenar estado AGN y química cargados desde checkpoint, si aplica.
    let mut resume_agn_bhs: Option<Vec<gadget_ng_sph::BlackHole>> = None;
    let mut resume_chem_states: Option<Vec<gadget_ng_rt::ChemState>> = None;

    let (mut local, start_step, mut a_current, mut h_state_resume) =
        if let Some(resume_dir) = resume_from {
            rt.root_eprintln(&format!(
                "[gadget-ng] Reanudando desde checkpoint en {:?}",
                resume_dir.join("checkpoint")
            ));
            let (p, completed, a, hs, agn_opt, chem_opt) =
                load_checkpoint(rt, resume_dir, lo, hi, &cfg_hash)?;
            resume_agn_bhs = agn_opt;
            resume_chem_states = chem_opt;
            (p, completed + 1, a, hs)
        } else {
            let p = build_particles_for_gid_range(cfg, lo, hi)?;
            let a0 = if cfg.cosmology.enabled {
                cfg.cosmology.a_init
            } else {
                1.0
            };
            (p, 1u64, a0, None)
        };

    let mut scratch = vec![Vec3::zero(); local.len()];
    let mut global_pos: Vec<Vec3> = Vec::new();
    let mut global_mass: Vec<f64> = Vec::new();

    // Árbol distribuido: solo para BarnesHut sin integrador jerárquico ni cosmología.
    let is_barnes_hut_eligible = cfg.gravity.solver == SolverKind::BarnesHut
        && !cfg.timestep.hierarchical
        && !cfg.cosmology.enabled;

    // Phase 56: jerárquico + SFC (halos de partículas) en multirank.
    // Reemplaza allgatherv O(N·P) por exchange_halos_sfc O(N_halo) dentro del closure de fuerzas.
    // Solo disponible para BarnesHut, sin cosmología (cosmología jerárquica: fase futura).
    // Con `force_allgather_fallback = true` o en rank-1 se usa el path allgather legacy.
    let use_hierarchical_let = cfg.gravity.solver == SolverKind::BarnesHut
        && cfg.timestep.hierarchical
        && !cfg.cosmology.enabled
        && rt.size() > 1
        && !cfg.performance.force_allgather_fallback;

    // SFC+LET es el path por defecto para multirank + BarnesHut.
    // Se desactiva solo con `force_allgather_fallback = true` (legacy) o tamaño 1.
    let use_sfc_let =
        is_barnes_hut_eligible && rt.size() > 1 && !cfg.performance.force_allgather_fallback;

    // Path slab legacy: use_distributed_tree + !use_sfc (retrocompatible).
    let use_dtree = !use_sfc_let
        && cfg.performance.use_distributed_tree
        && is_barnes_hut_eligible
        && !cfg.performance.use_sfc;
    // Path SFC legacy (halos de partículas): use_distributed_tree + use_sfc.
    let use_sfc = !use_sfc_let
        && cfg.performance.use_distributed_tree
        && is_barnes_hut_eligible
        && cfg.performance.use_sfc;

    let sfc_rebalance = cfg.performance.sfc_rebalance_interval;

    // Fase 17b: SFC+LET cosmológico.
    // Activado cuando: BarnesHut + cosmología + aperiódico + multirank + sin force_allgather_fallback.
    // No toca `is_barnes_hut_eligible` original (que exige `!cosmology.enabled` para el
    // path newtoniano SFC+LET). La cosmología distribuida usa su propio flag y branch.
    //
    // Fase 18: se desactiva cuando `periodic = true`: las fuerzas de árbol no usan
    // minimum_image; la cosmología periódica requiere PM o TreePM (allgather path).
    let use_sfc_let_cosmo = cfg.gravity.solver == SolverKind::BarnesHut
        && cfg.cosmology.enabled
        && !cfg.cosmology.periodic
        && !cfg.timestep.hierarchical
        && rt.size() > 1
        && !cfg.performance.force_allgather_fallback;

    // Validación de consistencia Fase 18: periodicidad requiere PM o TreePM.
    // BarnesHut + periodic es una combinación no soportada (fuerzas no periódicas).
    if cfg.cosmology.enabled && cfg.cosmology.periodic {
        if cfg.gravity.solver != SolverKind::Pm && cfg.gravity.solver != SolverKind::TreePm {
            return Err(CliError::InvalidConfig(
                "cosmology.periodic = true requiere gravity.solver = \"pm\" o \"tree_pm\".\n\
                 El solver BarnesHut no implementa fuerzas periódicas (minimum_image en árbol).\n\
                 Usa solver = \"pm\" para caja periódica con Fase 18."
                    .into(),
            ));
        }
        rt.root_eprintln(
            "[gadget-ng] Cosmología PERIÓDICA activada: PM + G/a + wrap_position (Fase 18).",
        );
        if cfg.gravity.treepm_slab && cfg.gravity.solver == SolverKind::TreePm {
            let r_s_log = if cfg.gravity.r_split > 0.0 {
                cfg.gravity.r_split
            } else {
                2.5 * cfg.simulation.box_size / cfg.gravity.pm_grid_size as f64
            };
            rt.root_eprintln(&format!(
                "[gadget-ng] TREEPM SLAB DISTRIBUIDO (Fase 21): PM largo alcance slab + \
                 árbol corto alcance periódico. r_split={:.4} r_cut={:.4}",
                r_s_log,
                5.0 * r_s_log
            ));
        } else if cfg.gravity.pm_slab && cfg.gravity.solver == SolverKind::Pm {
            let nm = cfg.gravity.pm_grid_size;
            let p = rt.size() as usize;
            if p > nm {
                // P > nm → slab 1D imposible; validar que pencil 2D sea factible.
                let (py, pz) = PencilLayout2D::factorize(nm, p);
                if py * pz != p || !nm.is_multiple_of(py) || !nm.is_multiple_of(pz) {
                    return Err(CliError::InvalidConfig(format!(
                        "pencil_2d (Fase 46): no existe factorización válida \
                         para nm={nm} y P={p}. Se requiere P ≤ nm² con nm%py==0 y nm%pz==0."
                    )));
                }
                rt.root_eprintln(&format!(
                    "[gadget-ng] PM PENCIL 2D (Fase 46): P={p} > nm={nm}; \
                     grilla {py}×{pz}, escala hasta P≤nm²={}.",
                    nm * nm
                ));
            } else {
                if !nm.is_multiple_of(p) {
                    return Err(CliError::InvalidConfig(format!(
                        "pm_slab requiere pm_grid_size ({nm}) % n_ranks ({p}) == 0"
                    )));
                }
                rt.root_eprintln(
                    "[gadget-ng] PM SLAB (Fase 20): FFT distribuida alltoall O(nm³/P) + slab decomposition.",
                );
            }
        } else if cfg.gravity.pm_distributed && cfg.gravity.solver == SolverKind::Pm {
            rt.root_eprintln(
                "[gadget-ng] PM DISTRIBUIDO (Fase 19): allreduce O(nm³) reemplaza allgather O(N·P).",
            );
        }
    }

    if use_sfc_let_cosmo {
        rt.root_eprintln(
            "[gadget-ng] SFC+LET COSMOLÓGICO activado: G/a scaling + LET distribuido (Fase 17b).",
        );
    } else if use_sfc_let {
        rt.root_eprintln(
            "[gadget-ng] SFC+LET activado: árbol distribuido con Locally Essential Trees (Fase 8).",
        );
    } else if use_sfc {
        rt.root_eprintln(
            "[gadget-ng] Árbol distribuido SFC (Morton Z-order 3D, halos de partículas).",
        );
    } else if use_dtree {
        rt.root_eprintln("[gadget-ng] Árbol distribuido activo (halos punto-a-punto en x).");
    } else if cfg.performance.force_allgather_fallback && rt.size() > 1 {
        rt.root_eprintln(
            "[gadget-ng] ADVERTENCIA: force_allgather_fallback=true → comunicación O(N·P).",
        );
    }

    // Estado cosmológico: factor de escala `a` y parámetros (si está habilitado).
    let cosmo_state: Option<(CosmologyParams, f64)> = if cfg.cosmology.enabled {
        let params = CosmologyParams::new(
            cfg.cosmology.omega_m,
            cfg.cosmology.omega_lambda,
            cfg.cosmology.h0,
        );
        Some((params, cfg.cosmology.a_init))
    } else {
        None
    };
    // `a_current` ya fue inicializado arriba (desde checkpoint o desde a_init).

    fs::create_dir_all(out_dir).map_err(|e| CliError::io(out_dir, e))?;
    let prov = provenance_for_run(cfg)?;

    let diag_path = out_dir.join("diagnostics.jsonl");
    let mut diag_file = if rt.rank() == 0 {
        Some(fs::File::create(&diag_path).map_err(|e| CliError::io(&diag_path, e))?)
    } else {
        None
    };

    write_diagnostic_line(rt, 0, &local, &diag_path, &mut diag_file, None, None, None)?;

    // `h_state_opt` se mantiene vivo tras el bucle para poder guardarlo con el snapshot.
    let mut h_state_opt: Option<HierarchicalState> = None;

    // ── Campo de radiación M1 (Phase 82b) — debe declararse ANTES de las macros ──
    // Se inicializa antes del loop si rt.enabled; de lo contrario permanece None.
    let mut rt_field_opt: Option<gadget_ng_rt::RadiationField> = if cfg.rt.enabled {
        let n = cfg.rt.rt_mesh;
        let dx = if n > 0 { cfg.simulation.box_size / n as f64 } else { 1.0 };
        Some(gadget_ng_rt::RadiationField::uniform(n, n, n, dx, 0.0))
    } else {
        None
    };

    // ── Estados de química SPH acoplados a EoR (Phase 95) ──────────────────────────
    // Vector paralelo a `local`: un ChemState por cada partícula de gas.
    // Phase 106: si hay estado de química en checkpoint, restaurarlo; en caso
    // contrario inicializar en estado neutro (o dejar vacío si EoR desactivado).
    let mut sph_chem_states: Vec<gadget_ng_rt::ChemState> = if cfg.reionization.enabled {
        if let Some(restored) = resume_chem_states.take() {
            restored
        } else {
            local
                .iter()
                .map(|_| gadget_ng_rt::ChemState::neutral())
                .collect()
        }
    } else {
        Vec::new()
    };

    // Macro local para guardar checkpoint cuando toca.
    // Phase 106: $agn_bhs y $chem son pasados explícitamente para evitar problemas
    // de scope en las distintas rutas del motor (jerárquico, TreePM, PM, etc.).
    macro_rules! maybe_checkpoint {
        ($step:expr, $hs:expr, $agn_bhs:expr, $chem:expr) => {
            if checkpoint_interval > 0 && $step % checkpoint_interval == 0 {
                save_checkpoint(
                    rt, $step, a_current, &local, total, $hs, out_dir, &cfg_hash,
                    $agn_bhs, $chem,
                )?;
            }
        };
        // Variante sin estado AGN/chem para rutas que no tienen SPH.
        ($step:expr, $hs:expr) => {
            maybe_checkpoint!($step, $hs, &[], &[]);
        };
    }

    // Macro local para guardar frame de snapshot intermedio.
    macro_rules! maybe_snap_frame {
        ($step:expr) => {
            if snapshot_interval > 0 && $step % snapshot_interval == 0 {
                if let Some(all_parts) = rt.root_gather_particles(&local, total) {
                    let frame_dir = out_dir.join("frames").join(format!("snap_{:06}", $step));
                    fs::create_dir_all(&frame_dir).map_err(|e| CliError::io(&frame_dir, e))?;
                    let t = $step as f64 * cfg.simulation.dt;
                    let z = if cfg.cosmology.enabled {
                        1.0 / a_current - 1.0
                    } else {
                        0.0
                    };
                    let env = snapshot_env_for(cfg, t, z);
                    write_snapshot_formatted(
                        cfg.output.snapshot_format,
                        &frame_dir,
                        &all_parts,
                        &prov,
                        &env,
                    )?;
                }
            }
        };
    }

    // ── Centros de halos FoF para AGN (Phase 100) ───────────────────────────────────
    // Actualizados por maybe_insitu! y consumidos por maybe_agn!.
    let mut halo_centers: Vec<gadget_ng_core::Vec3> = Vec::new();

    // Macro local para análisis in-situ (Phase 63 + química Phase 95 + AGN FoF Phase 100).
    // Actualiza halo_centers con los centros de los N halos más masivos para maybe_agn!.
    macro_rules! maybe_insitu {
        ($step:expr) => {
            let (_insitu_ran, _insitu_fx) = crate::insitu::maybe_run_insitu(
                &local,
                &cfg.insitu_analysis,
                cfg.simulation.box_size,
                a_current,
                $step,
                out_dir,
                if sph_chem_states.is_empty() { None } else { Some(&sph_chem_states) },
            );
            if _insitu_ran && !_insitu_fx.halo_centers.is_empty() {
                halo_centers = _insitu_fx.halo_centers;
            }
        };
    }

    // Macro local para el paso SPH cosmológico (Phase 66, fix Phase 82a).
    // Solo actúa si `cfg.sph.enabled` es true; de lo contrario es un no-op.
    // Construye los CosmoFactors desde cosmo_state si está activo, o usa factores planos.
    // $sph_step: índice del paso actual (u64), usado para la semilla de feedback.
    macro_rules! maybe_sph {
        ($sph_step:expr) => {
            if cfg.sph.enabled {
                let cf_sph = match &cosmo_state {
                    Some((cp, _)) => {
                        let (d, k, k2) = cp.drift_kick_factors(a_current, cfg.simulation.dt);
                        CosmoFactors { drift: d, kick_half: k, kick_half2: k2 }
                    }
                    None => CosmoFactors::flat(cfg.simulation.dt),
                };
                let gamma = cfg.sph.gamma;
                let alpha = cfg.sph.alpha_visc;
                let n_neigh = cfg.sph.n_neigh as f64;
                gadget_ng_sph::sph_cosmo_kdk_step(
                    &mut local,
                    cf_sph,
                    gamma,
                    alpha,
                    n_neigh,
                    |_parts| {},  // gravedad ya fue calculada por el solver principal
                );
                if cfg.sph.cooling != gadget_ng_core::CoolingKind::None {
                    gadget_ng_sph::apply_cooling(&mut local, &cfg.sph, cfg.simulation.dt);
                }
                // Phase 78: Feedback estelar estocástico
                if cfg.sph.feedback.enabled {
                    let sfr = gadget_ng_sph::compute_sfr(&local, &cfg.sph.feedback);
                    // Semilla única por paso y rank.
                    let mut fb_seed = ($sph_step as u64)
                        .wrapping_mul(2654435761)
                        .wrapping_add(rt.rank() as u64);
                    gadget_ng_sph::apply_sn_feedback(
                        &mut local,
                        &sfr,
                        &cfg.sph.feedback,
                        cfg.simulation.dt,
                        &mut fb_seed,
                    );
                }
            }
        };
    }

    // Macro local para el solver de transferencia radiativa M1 (Phase 82b).
    // Solo actúa si `cfg.rt.enabled` es true; de lo contrario es un no-op.
    macro_rules! maybe_rt {
        () => {
            if cfg.rt.enabled {
                if let Some(ref mut rf) = rt_field_opt {
                    let m1p = gadget_ng_rt::M1Params {
                        c_red_factor: cfg.rt.c_red_factor,
                        kappa_abs: cfg.rt.kappa_abs,
                        kappa_scat: 0.0,
                        substeps: cfg.rt.substeps,
                    };
                    gadget_ng_rt::m1_update(rf, cfg.simulation.dt, &m1p);
                    gadget_ng_rt::radiation_gas_coupling_step(
                        &mut local,
                        rf,
                        &m1p,
                        cfg.simulation.dt,
                        cfg.simulation.box_size,
                    );
                }
            }
        };
    }


    // Agujeros negros activos durante la simulación (Phase 96 + FoF Phase 100).
    // Phase 106: si hay estado AGN en checkpoint, restaurarlo.
    let mut agn_bhs: Vec<gadget_ng_sph::BlackHole> =
        resume_agn_bhs.take().unwrap_or_default();

    // Macro local para feedback AGN (Phase 96 + halos FoF Phase 100).
    // Coloca BH seeds en los N centros de halos más masivos (Phase 100).
    // Fallback al centro de la caja si no hay halos identificados aún.
    macro_rules! maybe_agn {
        ($sph_step_agn:expr) => {
            if cfg.sph.agn.enabled {
                let agn_params = gadget_ng_sph::AgnParams {
                    eps_feedback: cfg.sph.agn.eps_feedback,
                    m_seed: cfg.sph.agn.m_seed,
                    v_kick_agn: cfg.sph.agn.v_kick_agn,
                    r_influence: cfg.sph.agn.r_influence,
                };
                let n_bh = cfg.sph.agn.n_agn_bh.max(1);
                if !halo_centers.is_empty() {
                    // Sincronizar BHs con centros de halos FoF más masivos
                    let n_new = halo_centers.len().min(n_bh);
                    if agn_bhs.len() != n_new {
                        agn_bhs.resize_with(n_new, || gadget_ng_sph::BlackHole::new(
                            gadget_ng_core::Vec3::zero(), agn_params.m_seed,
                        ));
                    }
                    for (bh, &pos) in agn_bhs.iter_mut().zip(halo_centers.iter()) {
                        bh.pos = pos;
                    }
                } else if agn_bhs.is_empty() {
                    // Fallback: semilla en el centro de la caja hasta que haya halos
                    let center = cfg.simulation.box_size * 0.5;
                    agn_bhs.push(gadget_ng_sph::BlackHole::new(
                        gadget_ng_core::Vec3::new(center, center, center),
                        agn_params.m_seed,
                    ));
                }
                gadget_ng_sph::grow_black_holes(&mut agn_bhs, &local, &agn_params, cfg.simulation.dt);
                gadget_ng_sph::apply_agn_feedback(&mut local, &agn_bhs, &agn_params, cfg.simulation.dt);
                let _ = $sph_step_agn;
            }
        };
    }

    // Macro local para EoR completo z=6-12 (Phase 95).
    // Integra reionización con fuentes UV y química SPH acoplada (Phase 95 v2).
    // Usa sph_chem_states (un ChemState por partícula, sincronizado con local[]).
    macro_rules! maybe_reionization {
        ($a_cur:expr) => {
            if cfg.reionization.enabled {
                let _a_cur: f64 = $a_cur;
                let _z_eor = if _a_cur > 0.0 { 1.0 / _a_cur - 1.0 } else { f64::INFINITY };
                if _z_eor >= cfg.reionization.z_end && _z_eor <= cfg.reionization.z_start {
                    if let Some(ref mut rf) = rt_field_opt {
                        let m1p = gadget_ng_rt::M1Params {
                            c_red_factor: cfg.rt.c_red_factor,
                            kappa_abs: cfg.rt.kappa_abs,
                            kappa_scat: 0.0,
                            substeps: cfg.rt.substeps,
                        };
                        // Sincronizar longitud: si local creció (ej. partículas cargadas)
                        // extender sph_chem_states con estados neutros
                        if sph_chem_states.len() < local.len() {
                            let extra = local.len() - sph_chem_states.len();
                            sph_chem_states.extend(
                                std::iter::repeat(gadget_ng_rt::ChemState::neutral()).take(extra)
                            );
                        } else if sph_chem_states.len() > local.len() {
                            sph_chem_states.truncate(local.len());
                        }

                        // Fuentes UV: posiciones uniformes en la caja
                        let n_src = cfg.reionization.n_sources.max(1);
                        let lum = cfg.reionization.uv_luminosity;
                        let bsz = cfg.simulation.box_size;
                        let sources: Vec<gadget_ng_rt::UvSource> = (0..n_src)
                            .map(|i| {
                                let frac = (i as f64 + 0.5) / n_src as f64;
                                gadget_ng_rt::UvSource {
                                    pos: gadget_ng_core::Vec3::new(
                                        frac * bsz, frac * bsz, frac * bsz,
                                    ),
                                    luminosity: lum,
                                }
                            })
                            .collect();

                        // Paso de reionización con química acoplada real
                        let _reion_state = gadget_ng_rt::reionization_step(
                            rf,
                            &mut sph_chem_states,
                            &sources,
                            &m1p,
                            cfg.simulation.dt,
                            bsz,
                            _z_eor,
                        );
                    }
                }
            }
        };
    }

    // ── Acumuladores de tiempos por fase ─────────────────────────────────────
    let mut acc_comm_ns: u64 = 0;
    let mut acc_gravity_ns: u64 = 0;
    let mut acc_step_ns: u64 = 0;
    let mut steps_run: u64 = 0;
    let wall_loop_start = Instant::now();
    // Resumen HPC detallado; solo se puebla en el path SFC+LET.
    let mut hpc_aggregate_opt: Option<HpcTimingsAggregate> = None;
    // Resumen HPC del path TreePM SR-SFC; se puebla al final si use_treepm_sr_sfc estuvo activo.
    let mut treepm_hpc_opt: Option<TreePmAggregate> = None;
    // Acumulador TreePM SR-SFC (Fases 23/24); se puebla si use_treepm_sr_sfc está activo.
    let mut acc_tpm = TreePmStepDiag::default();
    let mut tpm_step_count: u64 = 0;

    let integrator_kind = cfg.simulation.integrator;
    if cfg.timestep.hierarchical && integrator_kind != IntegratorKind::Leapfrog {
        return Err(CliError::InvalidConfig(
            "Yoshida4 no está implementado con block timesteps (timestep.hierarchical = true); \
             usa integrator = leapfrog o desactiva hierarchical"
                .into(),
        ));
    }

    if cfg.timestep.hierarchical {
        let eta = cfg.timestep.eta;
        let max_level = cfg.timestep.max_level;
        let criterion = cfg.timestep.criterion;
        let kappa_h_hier = cfg.timestep.kappa_h;

        // Acoplamiento gravitacional cosmológico: G_eff = G · a³ (convención QKSL/GADGET-4).
        // Se usa para la aceleración inicial y para cada evaluación de fuerzas en el paso.
        // Con cosmología desactivada, g_hier = g (comportamiento newtoniano).
        let g_hier_init = if cosmo_state.is_some() {
            gravity_coupling_qksl(g, a_current)
        } else {
            g
        };

        // ── Phase 56: infraestructura SFC para el path jerárquico+LET ─────────
        // Solo se inicializa cuando use_hierarchical_let = true.
        let sfc_kind_hier = cfg.performance.sfc_kind;
        let halo_factor_hier = cfg.performance.halo_factor;
        let mut sfc_decomp_hier = if use_hierarchical_let {
            use gadget_ng_parallel::sfc::global_bbox;
            let (gxlo, gxhi, gylo, gyhi, gzlo, gzhi) = global_bbox(rt, &local);
            let all_pos_init: Vec<Vec3> = local.iter().map(|p| p.position).collect();
            Some(SfcDecomposition::build_with_bbox_and_kind(
                &all_pos_init,
                gxlo,
                gxhi,
                gylo,
                gyhi,
                gzlo,
                gzhi,
                rt.size(),
                sfc_kind_hier,
            ))
        } else {
            None
        };

        // Reutilizar HierarchicalState del checkpoint, o crear uno nuevo.
        let mut h_state = h_state_resume.take().unwrap_or_else(|| {
            let mut hs = HierarchicalState::new(local.len());
            // Aceleraciones iniciales: usa halos SFC si disponible, allgather si no.
            if let Some(ref sfc_d) = sfc_decomp_hier {
                let hw = sfc_d.halo_width(halo_factor_hier);
                let halos_init = rt.exchange_halos_sfc(&local, sfc_d, hw);
                let all_idx: Vec<usize> = (0..local.len()).collect();
                compute_forces_hierarchical_let(
                    &local,
                    &halos_init,
                    &all_idx,
                    theta,
                    g_hier_init,
                    eps2,
                    &mut scratch,
                );
            } else {
                rt.allgatherv_state(&local, total, &mut global_pos, &mut global_mass);
                let init_idx: Vec<usize> = local.iter().map(|p| p.global_id).collect();
                solver.accelerations_for_indices(
                    &global_pos,
                    &global_mass,
                    eps2,
                    g_hier_init,
                    &init_idx,
                    &mut scratch,
                );
            }
            for (p, &a) in local.iter_mut().zip(scratch.iter()) {
                p.acceleration = a;
            }
            hs.init_from_accels(&local, eps2, dt, eta, max_level, criterion);
            hs
        });

        // Seguimiento de desbalance de costo para el path jerárquico+LET.
        let mut cost_rebalance_hier = false;
        let imbalance_threshold_hier = cfg.performance.rebalance_imbalance_threshold;

        for step in start_step..=cfg.simulation.num_steps {
            let step_start = Instant::now();
            let mut this_comm: u64 = 0;
            let mut this_grav: u64 = 0;

            // Acoplamiento G·a³ fijo al inicio del paso (mismo que el path leapfrog cosmo).
            // El valor de 'a' se actualiza dentro de hierarchical_kdk_step al final del paso.
            let g_step = if cosmo_state.is_some() {
                gravity_coupling_qksl(g, a_current)
            } else {
                g
            };
            // Softening físico: recalcular ε_com = ε_phys/a en cada paso.
            let eps2 = eps2_at(a_current);

            // ── Phase 56: rebalanceo SFC + migración de dominio (base-steps) ──
            // La migración se ejecuta una vez por base-step; dentro de
            // hierarchical_kdk_step los 2^max_level subpasos finos usan el
            // snapshot de la SFC tomado al inicio del paso.
            if let Some(ref mut sfc_d) = sfc_decomp_hier {
                use gadget_ng_parallel::sfc::global_bbox;
                let do_rebalance =
                    should_rebalance(step, start_step, sfc_rebalance, cost_rebalance_hier);
                cost_rebalance_hier = false;
                if do_rebalance {
                    let t_rb = Instant::now();
                    let (gxlo, gxhi, gylo, gyhi, gzlo, gzhi) = global_bbox(rt, &local);
                    let pos_loc: Vec<Vec3> = local.iter().map(|p| p.position).collect();
                    *sfc_d = SfcDecomposition::build_with_bbox_and_kind(
                        &pos_loc,
                        gxlo,
                        gxhi,
                        gylo,
                        gyhi,
                        gzlo,
                        gzhi,
                        rt.size(),
                        sfc_kind_hier,
                    );
                    this_comm += t_rb.elapsed().as_nanos() as u64;
                }
                let t_domain = Instant::now();
                rt.exchange_domain_sfc(&mut local, sfc_d);
                this_comm += t_domain.elapsed().as_nanos() as u64;
                scratch.resize(local.len(), Vec3::zero());
            }
            // Snapshot inmutable de la SFC para el closure de fuerzas (todos los subpasos finos).
            let sfc_snap_hier = sfc_decomp_hier.clone();

            let cosmo_arg = cosmo_state
                .as_ref()
                .map(|(params, _)| (params, &mut a_current));
            let step_stats = hierarchical_kdk_step(
                &mut local,
                &mut h_state,
                dt,
                eps2,
                eta,
                max_level,
                criterion,
                cosmo_arg,
                kappa_h_hier,
                |parts, active_local, acc| {
                    if let Some(ref sfc_snap) = sfc_snap_hier {
                        // Path Phase 56: halos SFC + árbol local, solo activos.
                        let hw = sfc_snap.halo_width(halo_factor_hier);
                        let t0 = Instant::now();
                        let halos = rt.exchange_halos_sfc(parts, sfc_snap, hw);
                        this_comm += t0.elapsed().as_nanos() as u64;
                        let t1 = Instant::now();
                        compute_forces_hierarchical_let(
                            parts,
                            &halos,
                            active_local,
                            theta,
                            g_step,
                            eps2,
                            acc,
                        );
                        this_grav += t1.elapsed().as_nanos() as u64;
                    } else {
                        // Path allgather legacy: envía todos los datos, evalúa para activos.
                        let t0 = Instant::now();
                        rt.allgatherv_state(parts, total, &mut global_pos, &mut global_mass);
                        this_comm += t0.elapsed().as_nanos() as u64;
                        let global_idx: Vec<usize> =
                            active_local.iter().map(|&li| parts[li].global_id).collect();
                        let t1 = Instant::now();
                        solver.accelerations_for_indices(
                            &global_pos,
                            &global_mass,
                            eps2,
                            g_step,
                            &global_idx,
                            acc,
                        );
                        this_grav += t1.elapsed().as_nanos() as u64;
                    }
                },
            );
            // ── Detección de desbalance de costo (path jerárquico+LET) ─────────
            if rt.size() > 1 && this_grav > 0 && imbalance_threshold_hier > 1.0 {
                let wl_max = rt.allreduce_max_f64(this_grav as f64);
                let wl_min = rt.allreduce_min_f64(this_grav as f64).max(1.0);
                if wl_max / wl_min > imbalance_threshold_hier {
                    cost_rebalance_hier = true;
                }
            }

            acc_step_ns += step_start.elapsed().as_nanos() as u64;
            acc_comm_ns += this_comm;
            acc_gravity_ns += this_grav;
            steps_run += 1;

            // Diagnóstico cosmológico: a_current ya fue actualizado por hierarchical_kdk_step.
            let hier_cosmo_diag = cosmo_state.as_ref().map(|(cp, _)| CosmoDiag {
                a: a_current,
                z: 1.0 / a_current - 1.0,
                v_rms: peculiar_vrms(&local, a_current),
                delta_rms: density_contrast_rms(&local, cfg.simulation.box_size, 16),
                hubble: hubble_param(*cp, a_current),
                treepm: None,
            });
            write_diagnostic_line(
                rt,
                step,
                &local,
                &diag_path,
                &mut diag_file,
                Some(&step_stats),
                None,
                hier_cosmo_diag.as_ref(),
            )?;
            maybe_checkpoint!(step, Some(&h_state));
            maybe_snap_frame!(step);
            maybe_insitu!(step);
            maybe_sph!(step);
            maybe_agn!(step);
            maybe_rt!();
            maybe_reionization!(a_current);
        }
        h_state_opt = Some(h_state);
    } else if use_sfc_let_cosmo {
        // ── SFC+LET + Cosmología: Fase 17b ─────────────────────────────────────────
        //
        // Integra la física comóvil (momentum canónico p = a² dx_c/dt) con el backend
        // SFC+LET distribuido, aplicando la corrección G/a en cada evaluación de fuerza.
        //
        // Diferencias clave respecto al path allgather cosmológico (Fase 17a):
        //   • Comunicación O(log N) via LET en lugar de O(N) via allgather global.
        //   • Descomposición SFC con rebalanceo dinámico por partícula.
        //   • Path bloqueante (sin overlap compute/comm) para máxima correctitud.
        //
        // Restricciones de esta fase:
        //   • Caja no periódica (igual que Fase 17a).
        //   • Soporta Leapfrog y Yoshida4; g_cosmo = g / a_inicio_paso para todos los
        //     sub-pasos (correcto a primer orden en dt, consistente con Fase 17a).
        use gadget_ng_parallel::sfc::global_bbox;

        let (cosmo_params, _) = cosmo_state.unwrap();
        let sfc_kind = cfg.performance.sfc_kind;
        let (gxlo, gxhi, gylo, gyhi, gzlo, gzhi) = global_bbox(rt, &local);
        let all_pos_init: Vec<Vec3> = local.iter().map(|p| p.position).collect();
        let mut sfc_decomp = SfcDecomposition::build_with_bbox_and_kind(
            &all_pos_init,
            gxlo,
            gxhi,
            gylo,
            gyhi,
            gzlo,
            gzhi,
            rt.size(),
            sfc_kind,
        );
        let size = rt.size() as usize;
        let my_rank = rt.rank() as usize;
        let f_export = cfg.performance.let_theta_export_factor;
        let theta_export = if f_export > 0.0 {
            theta * f_export
        } else {
            theta
        };
        // Seguimiento de desbalance de costo para el path cosmológico SFC+LET.
        let mut cost_rebalance_cosmo = false;
        let imbalance_threshold_cosmo = cfg.performance.rebalance_imbalance_threshold;

        for step in start_step..=cfg.simulation.num_steps {
            let step_start = Instant::now();
            let mut this_comm: u64 = 0;
            let mut this_grav: u64 = 0;

            // ── Corrección comóvil (Phase 45): `g · a³` al inicio del paso ───────
            // Con la convención QKSL (`drift = ∫dt/a²`, `kick = ∫dt/a`), la fuerza
            // que debe recibir el solver para que `dp/dt = −∇Φ_pec` es `g · a³`.
            // El histórico `g/a` metía un factor `a⁴` de error. Ver Phase 45.
            let g_cosmo = gravity_coupling_qksl(g, a_current);
            // Softening físico: recalcular ε_com = ε_phys/a en cada paso.
            let eps2 = eps2_at(a_current);

            // ── Rebalanceo SFC ────────────────────────────────────────────────────
            let do_rebalance =
                should_rebalance(step, start_step, sfc_rebalance, cost_rebalance_cosmo);
            cost_rebalance_cosmo = false;
            if do_rebalance {
                let t_rb = Instant::now();
                let (gxlo, gxhi, gylo, gyhi, gzlo, gzhi) = global_bbox(rt, &local);
                let pos_loc: Vec<Vec3> = local.iter().map(|p| p.position).collect();
                sfc_decomp = SfcDecomposition::build_with_bbox_and_kind(
                    &pos_loc,
                    gxlo,
                    gxhi,
                    gylo,
                    gyhi,
                    gzlo,
                    gzhi,
                    rt.size(),
                    sfc_kind,
                );
                this_comm += t_rb.elapsed().as_nanos() as u64;
            }

            // ── Migración de dominio SFC ──────────────────────────────────────────
            let t_domain = Instant::now();
            rt.exchange_domain_sfc(&mut local, &sfc_decomp);
            this_comm += t_domain.elapsed().as_nanos() as u64;
            scratch.resize(local.len(), Vec3::zero());

            // ── Cierre de evaluación de fuerza con G/a (bloqueante) ───────────────
            //
            // Se captura `g_cosmo` fijo para este paso, consistente con el integrador:
            // tanto Leapfrog como Yoshida4 cosmo usan g/a del inicio del paso.
            let mut force_cosmo = |parts: &[Particle], acc: &mut [Vec3]| {
                // 1. AABB allgather — para saber a qué rangos enviar nodos LET.
                let my_aabb: Vec<f64> = if parts.is_empty() {
                    vec![
                        f64::INFINITY,
                        f64::NEG_INFINITY,
                        f64::INFINITY,
                        f64::NEG_INFINITY,
                        f64::INFINITY,
                        f64::NEG_INFINITY,
                    ]
                } else {
                    let xlo = parts
                        .iter()
                        .map(|p| p.position.x)
                        .fold(f64::INFINITY, f64::min);
                    let xhi = parts
                        .iter()
                        .map(|p| p.position.x)
                        .fold(f64::NEG_INFINITY, f64::max);
                    let ylo = parts
                        .iter()
                        .map(|p| p.position.y)
                        .fold(f64::INFINITY, f64::min);
                    let yhi = parts
                        .iter()
                        .map(|p| p.position.y)
                        .fold(f64::NEG_INFINITY, f64::max);
                    let zlo = parts
                        .iter()
                        .map(|p| p.position.z)
                        .fold(f64::INFINITY, f64::min);
                    let zhi = parts
                        .iter()
                        .map(|p| p.position.z)
                        .fold(f64::NEG_INFINITY, f64::max);
                    vec![xlo, xhi, ylo, yhi, zlo, zhi]
                };
                let t_aabb = Instant::now();
                let all_aabbs = rt.allgather_f64(&my_aabb);
                this_comm += t_aabb.elapsed().as_nanos() as u64;

                // 2. Árbol local con partículas propias de este rango.
                let t_build = Instant::now();
                let all_pos_l: Vec<Vec3> = parts.iter().map(|p| p.position).collect();
                let all_mass_l: Vec<f64> = parts.iter().map(|p| p.mass).collect();
                let tree = Octree::build(&all_pos_l, &all_mass_l);
                this_grav += t_build.elapsed().as_nanos() as u64;

                // 3. Exportar y empaquetar nodos LET hacia cada rango remoto.
                let t_exp = Instant::now();
                let mut sends: Vec<Vec<f64>> = (0..size).map(|_| Vec::new()).collect();
                for r in 0..size {
                    if r == my_rank {
                        continue;
                    }
                    let ra = &all_aabbs[r];
                    if ra.len() < 6 {
                        continue;
                    }
                    let target_aabb = [ra[0], ra[1], ra[2], ra[3], ra[4], ra[5]];
                    let let_nodes = tree.export_let(target_aabb, theta_export);
                    if !let_nodes.is_empty() {
                        sends[r] = pack_let_nodes(&let_nodes);
                    }
                }
                this_grav += t_exp.elapsed().as_nanos() as u64;

                // 4. Intercambio bloqueante de nodos LET.
                let t_comm2 = Instant::now();
                let received = rt.alltoallv_f64(&sends);
                this_comm += t_comm2.elapsed().as_nanos() as u64;

                // 5. Calcular fuerzas locales + remotas usando g_cosmo = G/a.
                //    `compute_forces_sfc_let` acepta `g` como parámetro explícito,
                //    así que pasamos `g_cosmo` sin modificar la función auxiliar.
                let t_grav = Instant::now();
                compute_forces_sfc_let(parts, &received, theta, g_cosmo, eps2, acc);
                this_grav += t_grav.elapsed().as_nanos() as u64;
            };

            // ── Integrador cosmológico KDK ────────────────────────────────────────
            match integrator_kind {
                IntegratorKind::Leapfrog => {
                    let (drift, kick_half, kick_half2) =
                        cosmo_params.drift_kick_factors(a_current, dt);
                    let cf = CosmoFactors {
                        drift,
                        kick_half,
                        kick_half2,
                    };
                    a_current = cosmo_params.advance_a(a_current, dt);
                    leapfrog_cosmo_kdk_step(&mut local, cf, &mut scratch, |parts, acc| {
                        force_cosmo(parts, acc);
                    });
                }
                IntegratorKind::Yoshida4 => {
                    // g_cosmo = g / a_inicio_paso, aplicado a los 3 sub-pasos.
                    // Los CosmoFactors se pre-calculan avanzando a_current por sub-paso.
                    let sub_dts = [YOSHIDA4_W1 * dt, YOSHIDA4_W0 * dt, YOSHIDA4_W1 * dt];
                    let mut cfs = [CosmoFactors::flat(0.0); 3];
                    for (i, &sub_dt) in sub_dts.iter().enumerate() {
                        let (drift, kick_half, kick_half2) =
                            cosmo_params.drift_kick_factors(a_current, sub_dt);
                        cfs[i] = CosmoFactors {
                            drift,
                            kick_half,
                            kick_half2,
                        };
                        a_current = cosmo_params.advance_a(a_current, sub_dt);
                    }
                    yoshida4_cosmo_kdk_step(&mut local, cfs, &mut scratch, |parts, acc| {
                        force_cosmo(parts, acc);
                    });
                }
            }

            // ── Diagnósticos cosmológicos (p17b-diag) ────────────────────────────
            // v_rms: suma local de |p/a|² → allreduce → sqrt(sum/total)
            // Esto es correcto en MPI: cada rango contribuye su segmento de partículas.
            let sum_v2_local: f64 = local
                .iter()
                .map(|p| {
                    let v = p.velocity * (1.0 / a_current);
                    v.dot(v)
                })
                .sum();
            let sum_v2 = rt.allreduce_sum_f64(sum_v2_local);
            let v_rms_dist = if total > 0 {
                (sum_v2 / total as f64).sqrt()
            } else {
                0.0
            };
            // delta_rms: aproximación local (cada rango ve su subconjunto de partículas).
            // En MPI, este valor es una estimación local; el reporte lo nota explícitamente.
            let delta_rms_local = density_contrast_rms(&local, cfg.simulation.box_size, 16);

            let cd = CosmoDiag {
                a: a_current,
                z: 1.0 / a_current - 1.0,
                v_rms: v_rms_dist,
                delta_rms: delta_rms_local,
                hubble: hubble_param(cosmo_params, a_current),
                treepm: None,
            };
            acc_step_ns += step_start.elapsed().as_nanos() as u64;
            acc_comm_ns += this_comm;
            // ── Detección de desbalance de costo (path cosmológico SFC+LET) ─────
            if rt.size() > 1 && this_grav > 0 && imbalance_threshold_cosmo > 1.0 {
                let wl_max = rt.allreduce_max_f64(this_grav as f64);
                let wl_min = rt.allreduce_min_f64(this_grav as f64).max(1.0);
                if wl_max / wl_min > imbalance_threshold_cosmo {
                    cost_rebalance_cosmo = true;
                }
            }

            acc_gravity_ns += this_grav;
            steps_run += 1;
            write_diagnostic_line(
                rt,
                step,
                &local,
                &diag_path,
                &mut diag_file,
                None,
                None,
                Some(&cd),
            )?;
            maybe_checkpoint!(step, None);
            maybe_snap_frame!(step);
            maybe_insitu!(step);
            maybe_sph!(step);
            maybe_agn!(step);
            maybe_rt!();
            maybe_reionization!(a_current);
        }
    } else if let Some((ref cosmo_params, _)) = cosmo_state {
        // Leapfrog / Yoshida4 cosmológico: factores drift/kick calculados por paso.
        //
        // Fase 18: si `cosmology.periodic = true` (y solver es PM/TreePM), tras el
        // integrador se envuelven las posiciones a [0, box_size)³ con `wrap_position`.
        // El solver PM ya usa `rem_euclid` internamente en CIC, pero el wrap explícito
        // garantiza correctitud de diagnósticos y snapshots.
        let cosmo_periodic = cfg.cosmology.periodic;
        let box_size = cfg.simulation.box_size;

        let pm_nm = cfg.gravity.pm_grid_size;

        // Fase 19: PM distribuido. Activo cuando periodic=true, solver=PM y pm_distributed=true.
        // En este path el allgather O(N·P) de partículas se reemplaza por un allreduce O(nm³)
        // del grid de densidad, lo que elimina la dependencia de comunicación en N.
        let use_pm_dist = cfg.gravity.pm_distributed
            && !cfg.gravity.pm_slab  // pm_slab tiene prioridad
            && cfg.cosmology.periodic
            && cfg.gravity.solver == SolverKind::Pm;

        // Fase 20: PM slab distribuido. FFT distribuida real mediante alltoall transposes.
        // Cada rank posee nz_local = nm/P planos Z del grid; la FFT Z se distribuye.
        // Solo activo cuando P ≤ nm; si P > nm la FFT slab 1D no tiene suficientes planos Z.
        let use_pm_slab = cfg.gravity.pm_slab
            && cfg.cosmology.periodic
            && cfg.gravity.solver == SolverKind::Pm
            && (rt.size() as usize) <= pm_nm;

        // Fase 46: PM pencil 2D. Activado automáticamente cuando pm_slab=true pero P > nm.
        // La descomposición pencil 2D usa una malla Py × Pz de procesos (P ≤ nm²),
        // eliminando la restricción P ≤ nm del slab 1D.
        let use_pm_pencil2d = cfg.gravity.pm_slab
            && cfg.cosmology.periodic
            && cfg.gravity.solver == SolverKind::Pm
            && (rt.size() as usize) > pm_nm;

        // Fase 21: TreePM slab distribuido.
        // PM largo alcance: slab FFT con filtro Gaussiano (como Fase 20).
        // Árbol corto alcance: árbol local + halos periódicos en z + minimum_image.
        let use_treepm_slab = cfg.gravity.treepm_slab
            && cfg.cosmology.periodic
            && cfg.gravity.solver == SolverKind::TreePm;

        // Fase 22: halo volumétrico 3D periódico para SR.
        // Requiere treepm_slab=true. Usa AABBs reales + minimum_image 3D en vez de halo 1D-z.
        let use_treepm_3d_halo = cfg.gravity.treepm_halo_3d && use_treepm_slab;

        // Fase 23: dominio 3D/SFC para el árbol SR (desacoplado del slab-z PM).
        // Requiere treepm_slab=true. El SR usa SfcDecomposition; el PM sigue en z-slab.
        // Implica use_treepm_3d_halo (el halo 3D es necesario para SR-SFC correcto).
        let use_treepm_sr_sfc = cfg.gravity.treepm_sr_sfc && use_treepm_slab;

        // Fase 24: scatter/gather PM mínimo entre dominio SFC y slabs PM.
        // Reemplaza el clone+migrate de Fase 23 por un alltoallv de datos mínimos.
        // Solo activo si use_treepm_sr_sfc está habilitado.
        let use_treepm_pm_scatter_gather =
            cfg.gravity.treepm_pm_scatter_gather && use_treepm_sr_sfc;

        // Precomputar límites de slab Z para Fase 20 (solo si pm_slab activo).
        let slab_layout_opt: Option<SlabLayout> = if use_pm_slab {
            Some(SlabLayout::new(
                pm_nm,
                rt.rank() as usize,
                rt.size() as usize,
            ))
        } else {
            None
        };

        // Fase 46: PencilLayout2D para PM pencil 2D (P > nm).
        let pencil_layout_opt: Option<PencilLayout2D> = if use_pm_pencil2d {
            let p = rt.size() as usize;
            let (py, pz) = PencilLayout2D::factorize(pm_nm, p);
            if py * pz != p || !pm_nm.is_multiple_of(py) || !pm_nm.is_multiple_of(pz) {
                return Err(CliError::InvalidConfig(format!(
                    "pencil_2d: no existe factorización válida para nm={pm_nm} y P={p}. \
                     Se requiere P ≤ nm² con nm % py == 0 y nm % pz == 0."
                )));
            }
            Some(PencilLayout2D::new(pm_nm, rt.rank() as usize, py, pz))
        } else {
            None
        };

        // Precomputar SlabLayout para Fase 21 TreePM distribuido.
        let treepm_slab_layout_opt: Option<SlabLayout> = if use_treepm_slab {
            let nm = pm_nm;
            let p = rt.size() as usize;
            if !nm.is_multiple_of(p) {
                return Err(CliError::InvalidConfig(format!(
                    "treepm_slab requiere pm_grid_size ({nm}) % n_ranks ({p}) == 0"
                )));
            }
            Some(SlabLayout::new(nm, rt.rank() as usize, p))
        } else {
            None
        };

        // Radio de splitting efectivo para TreePM slab.
        let treepm_r_split = if use_treepm_slab {
            let r_s = cfg.gravity.r_split;
            if r_s > 0.0 {
                r_s
            } else {
                2.5 * box_size / pm_nm as f64
            }
        } else {
            0.0
        };
        let treepm_r_cut = 5.0 * treepm_r_split;

        // Fase 23: SfcDecomposition para el dominio SR.
        // Se inicializa con la bbox global y se rebalanceará cada sfc_rebalance_interval pasos.
        let sr_sfc_kind = cfg.performance.sfc_kind;
        let sr_sfc_rebalance = cfg.performance.sfc_rebalance_interval;
        let mut sr_sfc_decomp_opt: Option<gadget_ng_parallel::SfcDecomposition> =
            if use_treepm_sr_sfc && rt.size() > 1 {
                use gadget_ng_parallel::sfc::global_bbox;
                let (gxlo, gxhi, gylo, gyhi, gzlo, gzhi) = global_bbox(rt, &local);
                let pos_loc: Vec<Vec3> = local.iter().map(|p| p.position).collect();
                Some(
                    gadget_ng_parallel::SfcDecomposition::build_with_bbox_and_kind(
                        &pos_loc,
                        gxlo,
                        gxhi,
                        gylo,
                        gyhi,
                        gzlo,
                        gzhi,
                        rt.size(),
                        sr_sfc_kind,
                    ),
                )
            } else {
                None
            };

        for step in start_step..=cfg.simulation.num_steps {
            let step_start = Instant::now();
            let mut this_comm: u64 = 0;
            let mut this_grav: u64 = 0;

            // Celda de acumulación para diagnósticos TreePM del paso actual.
            // Se resetea cada paso; el closure compute_acc acumula en ella (puede llamarse
            // múltiples veces por paso en Yoshida4). Cell<T> permite interior mutability sin &mut.
            let tpm_diag_cell = std::cell::Cell::new(TreePmStepDiag::default());

            // Fase 20: migrar partículas a su slab Z al inicio de cada paso.
            // Necesario para que deposit_slab_extended reciba las partículas correctas.
            if use_pm_slab && rt.size() > 1 {
                if let Some(ref layout) = slab_layout_opt {
                    let z_lo = layout.z_lo_idx as f64 * box_size / pm_nm as f64;
                    let z_hi = (layout.z_lo_idx + layout.nz_local) as f64 * box_size / pm_nm as f64;
                    rt.exchange_domain_by_z(&mut local, z_lo, z_hi);
                }
            }

            // Fase 21: migrar partículas a su slab Z para TreePM distribuido.
            // Fase 23 (use_treepm_sr_sfc): este path usa SFC en su lugar (ver abajo).
            if use_treepm_slab && !use_treepm_sr_sfc && rt.size() > 1 {
                if let Some(ref layout) = treepm_slab_layout_opt {
                    let z_lo = layout.z_lo_idx as f64 * box_size / pm_nm as f64;
                    let z_hi = (layout.z_lo_idx + layout.nz_local) as f64 * box_size / pm_nm as f64;
                    rt.exchange_domain_by_z(&mut local, z_lo, z_hi);
                }
            }

            // Fase 23: rebalanceo y migración SFC para el dominio SR.
            // Las partículas viven en SFC domain; el PM las clonará temporalmente a z-slab.
            if use_treepm_sr_sfc && rt.size() > 1 {
                use gadget_ng_parallel::sfc::global_bbox;
                let do_rebalance =
                    sr_sfc_rebalance == 0 || (step - start_step) % sr_sfc_rebalance.max(1) == 0;
                if do_rebalance {
                    let (gxlo, gxhi, gylo, gyhi, gzlo, gzhi) = global_bbox(rt, &local);
                    let pos_loc: Vec<Vec3> = local.iter().map(|p| p.position).collect();
                    sr_sfc_decomp_opt = Some(
                        gadget_ng_parallel::SfcDecomposition::build_with_bbox_and_kind(
                            &pos_loc,
                            gxlo,
                            gxhi,
                            gylo,
                            gyhi,
                            gzlo,
                            gzhi,
                            rt.size(),
                            sr_sfc_kind,
                        ),
                    );
                }
                if let Some(ref sfc_d) = sr_sfc_decomp_opt {
                    rt.exchange_domain_sfc(&mut local, sfc_d);
                }
            }

            // Redimensionar scratch tras posible migración SFC (Fase 23: exchange_domain_sfc
            // puede cambiar local.len(); leapfrog_cosmo_kdk_step requiere scratch.len() == local.len()).
            scratch.resize(local.len(), Vec3::zero());

            // Corrección de fuerza comóvil (Phase 45): `g_cosmo = g · a³`.
            // Con la convención canónica QKSL (`p = a²·ẋ_c`, `drift = ∫dt/a²`,
            // `kick = ∫dt/a`), pasar `g · a³` al solver hace que `dp/dt = −∇Φ_pec`.
            // El histórico `g/a` producía fuerzas ~6·10⁶× más grandes de lo
            // correcto a `a = 0.02`, haciendo que `v_rms` explotara.
            let g_cosmo = gravity_coupling_qksl(g, a_current);
            // Fix Phase 101: softening físico — recalcular ε_com = softening/a por paso.
            let eps2 = eps2_at(a_current);
            let mut compute_acc = |parts: &[Particle],
                                   acc: &mut [Vec3],
                                   this_comm: &mut u64,
                                   this_grav: &mut u64| {
                if use_treepm_sr_sfc {
                    // ─ Fase 23: TreePM SR desacoplado del slab-z (dominio 3D/SFC) ────
                    //
                    // Arquitectura dual:
                    //   • SR tree:  dominio SFC real (Morton/Hilbert). Halos 3D periódicos.
                    //   • PM LR:    slab-z sin cambios (Fase 20/21). Sincronización explícita.
                    //
                    // F_total = F_lr (PM slab + erf) + F_sr (árbol erfc, dominio SFC)
                    let layout = treepm_slab_layout_opt.as_ref().unwrap();
                    let r_s = treepm_r_split;
                    let r_cut = treepm_r_cut;
                    let _t_tpm = Instant::now();

                    // ── 1. SR: halo volumétrico 3D periódico (path activo principal) ─
                    let t_sr_halo = Instant::now();
                    let sr_halos = rt.exchange_halos_3d_periodic(parts, box_size, r_cut);
                    let sr_halo_comm_ns = t_sr_halo.elapsed().as_nanos() as u64;
                    *this_comm += sr_halo_comm_ns;

                    // ── 2. SR: árbol corto alcance sobre dominio SFC + halos 3D ──────
                    let t_sr_tree = Instant::now();
                    let sr_params = treepm_dist::SfcShortRangeParams {
                        local_particles: parts,
                        halo_particles: &sr_halos,
                        eps2,
                        g: g_cosmo,
                        r_split: r_s,
                        box_size,
                    };
                    let mut acc_sr = vec![Vec3::zero(); parts.len()];
                    treepm_dist::short_range_accels_sfc(&sr_params, &mut acc_sr);
                    let tree_sr_ns = t_sr_tree.elapsed().as_nanos() as u64;
                    *this_grav += tree_sr_ns;

                    // ── 3. PM LR: scatter/gather (Fase 24) o clone+migrate (Fase 23) ──
                    let t_pm_sync = Instant::now();

                    let (acc_lr, sg_stats_opt): (Vec<Vec3>, Option<treepm_dist::PmScatterStats>) =
                        if use_treepm_pm_scatter_gather {
                            // ── Fase 24: scatter/gather PM mínimo ───────────────────────
                            //
                            // Las partículas permanecen en SFC domain.
                            // Se envía solo (gid, pos, mass) al slab PM destino.
                            // El slab PM devuelve (gid, acc_pm) al source rank.
                            // Sin clone, sin migración de Particle completo.
                            let (acc_pm, sg_stats) = treepm_dist::pm_scatter_gather_accels(
                                parts, layout, g_cosmo, r_s, box_size, rt,
                            );
                            *this_comm += sg_stats.scatter_ns + sg_stats.gather_ns;
                            *this_grav += sg_stats.pm_solve_ns;
                            (acc_pm, Some(sg_stats))
                        } else {
                            // ── Fase 23: clone → z-slab → PM → back-SFC → HashMap ───────
                            //
                            // Conservado como fallback para comparación con Fase 24.

                            // 3a. Clonar partículas locales y migrar a z-slab PM.
                            let mut pm_parts = parts.to_vec();
                            let pm_z_lo = layout.z_lo_idx as f64 * box_size / pm_nm as f64;
                            let pm_z_hi = (layout.z_lo_idx + layout.nz_local) as f64 * box_size
                                / pm_nm as f64;
                            rt.exchange_domain_by_z(&mut pm_parts, pm_z_lo, pm_z_hi);
                            *this_comm += t_pm_sync.elapsed().as_nanos() as u64;

                            // 3b. PM pipeline sobre las partículas en z-slab.
                            let t_pm = Instant::now();
                            let pm_pos: Vec<Vec3> = pm_parts.iter().map(|p| p.position).collect();
                            let pm_mass: Vec<f64> = pm_parts.iter().map(|p| p.mass).collect();
                            let mut density_ext =
                                slab_pm::deposit_slab_extended(&pm_pos, &pm_mass, layout, box_size);
                            slab_pm::exchange_density_halos_z(&mut density_ext, layout, rt);
                            *this_comm += t_pm.elapsed().as_nanos() as u64;

                            let t_pm2 = Instant::now();
                            let mut forces = slab_pm::forces_from_slab(
                                &density_ext,
                                layout,
                                g_cosmo,
                                box_size,
                                Some(r_s),
                                rt,
                            );
                            slab_pm::exchange_force_halos_z(&mut forces, layout, rt);
                            *this_comm += t_pm2.elapsed().as_nanos() as u64;

                            let t_interp = Instant::now();
                            let acc_lr_pm =
                                slab_pm::interpolate_slab_local(&pm_pos, &forces, layout, box_size);
                            *this_grav += t_interp.elapsed().as_nanos() as u64;

                            // 3c. Embeber acc_lr en pm_parts.acceleration.
                            for (p, a) in pm_parts.iter_mut().zip(acc_lr_pm.iter()) {
                                p.acceleration = *a;
                            }

                            // 3d. Retornar pm_parts al dominio SFC.
                            let t_back = Instant::now();
                            if let Some(ref sfc_d) = sr_sfc_decomp_opt {
                                rt.exchange_domain_sfc(&mut pm_parts, sfc_d);
                            }
                            *this_comm += t_back.elapsed().as_nanos() as u64;

                            // 3e. Lookup de acc_lr por global_id.
                            let lr_map: std::collections::HashMap<usize, Vec3> = pm_parts
                                .iter()
                                .map(|p| (p.global_id, p.acceleration))
                                .collect();
                            let acc_lr: Vec<Vec3> = parts
                                .iter()
                                .map(|p| lr_map.get(&p.global_id).copied().unwrap_or(Vec3::zero()))
                                .collect();
                            (acc_lr, None)
                        };

                    let sr_sync_elapsed = t_pm_sync.elapsed().as_nanos() as u64;

                    // ── 4. Suma: F_total = F_lr + F_sr ──────────────────────────────
                    for (k, a) in acc.iter_mut().enumerate() {
                        *a = acc_lr[k] + acc_sr[k];
                    }

                    let _ = sr_sync_elapsed;
                    // ── Diagnóstico TreePM SR-SFC: propagar stats al scope exterior ──
                    // Se construye un TreePmStepDiag con los valores medidos en esta
                    // sub-llamada y se acumula en tpm_diag_cell (interior mutability).
                    let new_diag = TreePmStepDiag {
                        scatter_ns: sg_stats_opt.map_or(0, |s| s.scatter_ns),
                        gather_ns: sg_stats_opt.map_or(0, |s| s.gather_ns),
                        pm_solve_ns: sg_stats_opt.map_or(0, |s| s.pm_solve_ns),
                        sr_halo_ns: sr_halo_comm_ns,
                        tree_sr_ns,
                        scatter_particles: sg_stats_opt
                            .map_or(parts.len(), |s| s.scatter_particles),
                        scatter_bytes: sg_stats_opt.map_or(0, |s| s.scatter_bytes),
                        gather_bytes: sg_stats_opt.map_or(0, |s| s.gather_bytes),
                        path: if use_treepm_pm_scatter_gather {
                            "sg"
                        } else {
                            "clone"
                        },
                    };
                    tpm_diag_cell.set(tpm_diag_cell.get().add(new_diag));
                } else if use_treepm_slab {
                    // ─ Fase 21/22: TreePM slab distribuido ───────────────────────────
                    //
                    // F_total = F_lr (PM slab + filtro Gaussiano) + F_sr (árbol erfc + minimum_image)
                    //
                    // Fase 21: halo 1D en z.
                    // Fase 22 (use_treepm_3d_halo): halo volumétrico 3D periódico.
                    let layout = treepm_slab_layout_opt.as_ref().unwrap();
                    let r_s = treepm_r_split;
                    let r_cut = treepm_r_cut;
                    let t_tpm = Instant::now();

                    // ── 1. Halos de corto alcance ────────────────────────────────────
                    let t_comm_sr = Instant::now();
                    let sr_halos = if use_treepm_3d_halo {
                        // Fase 22: halo volumétrico 3D periódico (fix del bug de exchange_halos_sfc).
                        rt.exchange_halos_3d_periodic(parts, box_size, r_cut)
                    } else {
                        // Fase 21: halo 1D-z periódico.
                        let z_lo = layout.z_lo_idx as f64 * box_size / pm_nm as f64;
                        let z_hi =
                            (layout.z_lo_idx + layout.nz_local) as f64 * box_size / pm_nm as f64;
                        rt.exchange_halos_by_z_periodic(parts, z_lo, z_hi, r_cut)
                    };
                    let halo_comm_ns = t_comm_sr.elapsed().as_nanos() as u64;
                    *this_comm += halo_comm_ns;

                    // ── 2. PM largo alcance (slab FFT con filtro Gaussiano) ──────────
                    let t_pm = Instant::now();
                    let local_pos: Vec<Vec3> = parts.iter().map(|p| p.position).collect();
                    let local_mass: Vec<f64> = parts.iter().map(|p| p.mass).collect();

                    let mut density_ext =
                        slab_pm::deposit_slab_extended(&local_pos, &local_mass, layout, box_size);
                    slab_pm::exchange_density_halos_z(&mut density_ext, layout, rt);
                    *this_comm += t_pm.elapsed().as_nanos() as u64;

                    let t_pm2 = Instant::now();
                    let mut forces = slab_pm::forces_from_slab(
                        &density_ext,
                        layout,
                        g_cosmo,
                        box_size,
                        Some(r_s),
                        rt,
                    );
                    slab_pm::exchange_force_halos_z(&mut forces, layout, rt);
                    *this_comm += t_pm2.elapsed().as_nanos() as u64;

                    let t_interp = Instant::now();
                    let acc_lr =
                        slab_pm::interpolate_slab_local(&local_pos, &forces, layout, box_size);
                    let pm_ns = t_interp.elapsed().as_nanos() as u64;
                    *this_grav += pm_ns;

                    // ── 3. Árbol corto alcance (minimum_image periódico) ─────────────
                    let t_sr = Instant::now();
                    let sr_params = treepm_dist::SlabShortRangeParams {
                        local_particles: parts,
                        halo_particles: &sr_halos,
                        eps2,
                        g: g_cosmo,
                        r_split: r_s,
                        box_size,
                    };
                    let mut acc_sr = vec![Vec3::zero(); parts.len()];
                    treepm_dist::short_range_accels_slab(&sr_params, &mut acc_sr);
                    let tree_ns = t_sr.elapsed().as_nanos() as u64;
                    *this_grav += tree_ns;

                    // ── 4. Suma: F_total = F_lr + F_sr ──────────────────────────────
                    for (k, a) in acc.iter_mut().enumerate() {
                        *a = acc_lr[k] + acc_sr[k];
                    }

                    let tpm_total_ns = t_tpm.elapsed().as_nanos() as u64;
                    // Estadísticas para diagnostics.jsonl (conservadas para evitar dead_code).
                    let _sr_halo_n = sr_halos.len();
                    let _sr_halo_b =
                        sr_halos.len() * std::mem::size_of::<gadget_ng_core::Particle>();
                    let _halo_ns_cap = halo_comm_ns;
                    let _ = tpm_total_ns;
                } else if use_pm_slab {
                    // ─ Fase 20: PM slab distribuido ──────────────────────────────────
                    let layout = slab_layout_opt.as_ref().unwrap();
                    let t0 = Instant::now();

                    let local_pos: Vec<Vec3> = parts.iter().map(|p| p.position).collect();
                    let local_mass: Vec<f64> = parts.iter().map(|p| p.mass).collect();

                    // 1. Depósito CIC en slab extendido (con ghost right).
                    let mut density_ext =
                        slab_pm::deposit_slab_extended(&local_pos, &local_mass, layout, box_size);
                    // 2. Intercambio de halos de densidad Z (ring periódico).
                    slab_pm::exchange_density_halos_z(&mut density_ext, layout, rt);
                    *this_comm += t0.elapsed().as_nanos() as u64;

                    // 3. Solve de Poisson distribuido (alltoall transposes).
                    let t1 = Instant::now();
                    let mut forces = slab_pm::forces_from_slab(
                        &density_ext,
                        layout,
                        g_cosmo,
                        box_size,
                        None,
                        rt,
                    );
                    // 4. Intercambio de halos de fuerza Z (para CIC correcto en bordes).
                    slab_pm::exchange_force_halos_z(&mut forces, layout, rt);
                    *this_comm += t1.elapsed().as_nanos() as u64;

                    // 5. Interpolación CIC local.
                    let t2 = Instant::now();
                    let accels =
                        slab_pm::interpolate_slab_local(&local_pos, &forces, layout, box_size);
                    for (a, v) in acc.iter_mut().zip(accels.iter()) {
                        *a = *v;
                    }
                    *this_grav += t2.elapsed().as_nanos() as u64;
                } else if use_pm_pencil2d {
                    // ─ Fase 46: PM pencil 2D (P > nm) ────────────────────────────────
                    // Usa descomposición pencil 2D (Py × Pz = P) para la FFT distribuida.
                    // Permite escalar a P ≤ nm² ranks (vs P ≤ nm del slab 1D).
                    //
                    // Pipeline:
                    //   1. Depósito CIC local → allreduce global nm³
                    //   2. Extraer slab 2D local [ny_local × nz_local × nm]
                    //   3. solve_forces_pencil2d → fuerzas slab 2D local
                    //   4. Allgather fuerzas → reconstruir grid global nm³
                    //   5. Interpolación CIC local
                    let pencil_layout = pencil_layout_opt.as_ref().unwrap();
                    let nm = pm_nm;
                    let pz = pencil_layout.pz;
                    let ny_local = pencil_layout.ny_local;
                    let nz_local = pencil_layout.nz_local;
                    let iy_lo = pencil_layout.rank_y * ny_local;
                    let iz_lo = pencil_layout.rank_z * nz_local;
                    let n_ranks = pencil_layout.n_ranks;

                    // 1. Depósito CIC + allreduce global.
                    let t0 = Instant::now();
                    let local_pos: Vec<Vec3> = parts.iter().map(|p| p.position).collect();
                    let local_mass: Vec<f64> = parts.iter().map(|p| p.mass).collect();
                    let mut density = pm_dist::deposit_local(&local_pos, &local_mass, box_size, nm);
                    rt.allreduce_sum_f64_slice(&mut density);
                    *this_comm += t0.elapsed().as_nanos() as u64;

                    // 2. Extraer slab 2D del rank actual: density_2d[iy_local][iz_local][ix].
                    // CIC usa density[iz * nm² + iy * nm + ix].
                    let mut density_2d = vec![0.0f64; ny_local * nz_local * nm];
                    for iy_local_i in 0..ny_local {
                        let iy = iy_lo + iy_local_i;
                        for iz_local_i in 0..nz_local {
                            let iz = iz_lo + iz_local_i;
                            for ix in 0..nm {
                                density_2d[iy_local_i * nz_local * nm + iz_local_i * nm + ix] =
                                    density[iz * nm * nm + iy * nm + ix];
                            }
                        }
                    }
                    drop(density);

                    // 3. Solve pencil FFT: alltoalls dentro de subcomunicadores Y/Z.
                    let t1 = Instant::now();
                    let [fx_2d, fy_2d, fz_2d] = solve_forces_pencil2d(
                        &density_2d,
                        pencil_layout,
                        g_cosmo,
                        box_size,
                        None,
                        rt,
                    );
                    *this_grav += t1.elapsed().as_nanos() as u64;

                    // 4. Allgather slabs de fuerza → reconstruir grids nm³ globales.
                    let t2 = Instant::now();
                    let all_fx = rt.allgather_f64(&fx_2d);
                    let all_fy = rt.allgather_f64(&fy_2d);
                    let all_fz = rt.allgather_f64(&fz_2d);
                    *this_comm += t2.elapsed().as_nanos() as u64;

                    let mut fx_global = vec![0.0f64; nm * nm * nm];
                    let mut fy_global = vec![0.0f64; nm * nm * nm];
                    let mut fz_global = vec![0.0f64; nm * nm * nm];
                    for r in 0..n_ranks {
                        let ry_r = r / pz;
                        let rz_r = r % pz;
                        let iy_r_lo = ry_r * ny_local;
                        let iz_r_lo = rz_r * nz_local;
                        for iy_l in 0..ny_local {
                            let iy = iy_r_lo + iy_l;
                            for iz_l in 0..nz_local {
                                let iz = iz_r_lo + iz_l;
                                for ix in 0..nm {
                                    let src = iy_l * nz_local * nm + iz_l * nm + ix;
                                    let dst = iz * nm * nm + iy * nm + ix;
                                    fx_global[dst] = all_fx[r][src];
                                    fy_global[dst] = all_fy[r][src];
                                    fz_global[dst] = all_fz[r][src];
                                }
                            }
                        }
                    }

                    // 5. Interpolación CIC local desde el grid global.
                    let t3 = Instant::now();
                    let accels = pm_dist::interpolate_local(
                        &local_pos, &fx_global, &fy_global, &fz_global, nm, box_size,
                    );
                    for (a, v) in acc.iter_mut().zip(accels.iter()) {
                        *a = *v;
                    }
                    *this_grav += t3.elapsed().as_nanos() as u64;
                } else if use_pm_dist {
                    // ─ Fase 19: PM distribuido ────────────────────────────────────────
                    // 1. Depósito CIC local (O(N/P) por rank).
                    let t0 = Instant::now();
                    let local_pos: Vec<Vec3> = parts.iter().map(|p| p.position).collect();
                    let local_mass: Vec<f64> = parts.iter().map(|p| p.mass).collect();
                    let mut density =
                        pm_dist::deposit_local(&local_pos, &local_mass, box_size, pm_nm);
                    // 2. allreduce_sum del grid de densidad (O(nm³), no O(N·P)).
                    rt.allreduce_sum_f64_slice(&mut density);
                    *this_comm += t0.elapsed().as_nanos() as u64;
                    // 3. Solve de Poisson (determinista, idéntico en todos los ranks).
                    let t1 = Instant::now();
                    let [fx, fy, fz] =
                        pm_dist::forces_from_global_density(&density, g_cosmo, pm_nm, box_size);
                    // 4. Interpolación CIC local (O(N/P) por rank).
                    let accels =
                        pm_dist::interpolate_local(&local_pos, &fx, &fy, &fz, pm_nm, box_size);
                    for (a, v) in acc.iter_mut().zip(accels.iter()) {
                        *a = *v;
                    }
                    *this_grav += t1.elapsed().as_nanos() as u64;
                } else {
                    // ─ Path clásico: allgather de partículas (Fase 18 y anteriores) ──
                    let t0 = Instant::now();
                    rt.allgatherv_state(parts, total, &mut global_pos, &mut global_mass);
                    *this_comm += t0.elapsed().as_nanos() as u64;
                    let idx: Vec<usize> = parts.iter().map(|p| p.global_id).collect();
                    let t1 = Instant::now();
                    solver.accelerations_for_indices(
                        &global_pos,
                        &global_mass,
                        eps2,
                        g_cosmo,
                        &idx,
                        acc,
                    );
                    *this_grav += t1.elapsed().as_nanos() as u64;
                }
            };
            match integrator_kind {
                IntegratorKind::Leapfrog => {
                    let (drift, kick_half, kick_half2) =
                        cosmo_params.drift_kick_factors(a_current, dt);
                    let cf = CosmoFactors {
                        drift,
                        kick_half,
                        kick_half2,
                    };
                    a_current = cosmo_params.advance_a(a_current, dt);
                    leapfrog_cosmo_kdk_step(&mut local, cf, &mut scratch, |parts, acc| {
                        compute_acc(parts, acc, &mut this_comm, &mut this_grav);
                    });
                }
                IntegratorKind::Yoshida4 => {
                    let sub_dts = [YOSHIDA4_W1 * dt, YOSHIDA4_W0 * dt, YOSHIDA4_W1 * dt];
                    let mut cfs = [CosmoFactors::flat(0.0); 3];
                    for (i, &sub_dt) in sub_dts.iter().enumerate() {
                        let (drift, kick_half, kick_half2) =
                            cosmo_params.drift_kick_factors(a_current, sub_dt);
                        cfs[i] = CosmoFactors {
                            drift,
                            kick_half,
                            kick_half2,
                        };
                        a_current = cosmo_params.advance_a(a_current, sub_dt);
                    }
                    yoshida4_cosmo_kdk_step(&mut local, cfs, &mut scratch, |parts, acc| {
                        compute_acc(parts, acc, &mut this_comm, &mut this_grav);
                    });
                }
            }

            // Fase 18: envolver posiciones periódicamente tras cada paso de drift.
            // El wrap garantiza que todas las posiciones queden en [0, box_size)³,
            // lo que es necesario para diagnósticos correctos y consistencia física.
            if cosmo_periodic {
                for p in local.iter_mut() {
                    p.position = wrap_position(p.position, box_size);
                }
            }

            acc_step_ns += step_start.elapsed().as_nanos() as u64;
            acc_comm_ns += this_comm;
            acc_gravity_ns += this_grav;
            steps_run += 1;

            // Leer diagnóstico TreePM acumulado del paso (suma de todas las sub-llamadas).
            let step_tpm = tpm_diag_cell.get();
            if use_treepm_sr_sfc {
                acc_tpm = acc_tpm.add(step_tpm);
                tpm_step_count += 1;
            }

            let cd = CosmoDiag {
                a: a_current,
                z: 1.0 / a_current - 1.0,
                v_rms: peculiar_vrms(&local, a_current),
                delta_rms: density_contrast_rms(&local, cfg.simulation.box_size, 16),
                hubble: hubble_param(*cosmo_params, a_current),
                treepm: if use_treepm_sr_sfc {
                    Some(step_tpm)
                } else {
                    None
                },
            };
            write_diagnostic_line(
                rt,
                step,
                &local,
                &diag_path,
                &mut diag_file,
                None,
                None,
                Some(&cd),
            )?;
            maybe_checkpoint!(step, None);
            maybe_snap_frame!(step);
            maybe_insitu!(step);
            maybe_sph!(step);
            maybe_agn!(step);
            maybe_rt!();
            maybe_reionization!(a_current);
        }
    } else if use_sfc_let {
        // ── SFC + LET: Fase 9 — overlap compute/comm + Rayon + HpcStepStats ──
        //
        // Mejoras sobre Fase 8:
        //   • `alltoallv_f64_overlap`: el walk local se solapa con la comm LET.
        //   • Rayon (`#[cfg(feature = "simd")]`): walk paralelo intra-rango.
        //   • HpcStepStats: desglose de tiempos por fase escrito en diagnostics.jsonl.
        //   • Corrección de atribución: build/export/pack ahora son `this_grav` (no comm).
        //   • Rebalanceo dinámico basado en costo: `allreduce max/min walk_local_ns`.
        use gadget_ng_parallel::sfc::global_bbox;

        let sfc_kind = cfg.performance.sfc_kind;
        let (gxlo, gxhi, gylo, gyhi, gzlo, gzhi) = global_bbox(rt, &local);
        let all_pos: Vec<Vec3> = local.iter().map(|p| p.position).collect();
        let mut sfc_decomp = SfcDecomposition::build_with_bbox_and_kind(
            &all_pos,
            gxlo,
            gxhi,
            gylo,
            gyhi,
            gzlo,
            gzhi,
            rt.size(),
            sfc_kind,
        );
        let size = rt.size() as usize;
        let my_rank = rt.rank() as usize;
        let use_overlap = cfg.performance.let_nonblocking;

        // Acumuladores HPC agregados para TimingsReport.
        let mut acc_hpc = HpcStepStats::default();
        // Umbral para rebalanceo por costo: si max/min walk_local_ns > 1.3 → rebalanceo inmediato.
        let mut cost_rebalance_pending = false;
        // Seguimiento de particle imbalance: max/min a través de allreduce por paso.
        let mut acc_max_local: f64 = 0.0;
        let mut acc_min_local: f64 = f64::MAX;

        for step in start_step..=cfg.simulation.num_steps {
            let step_start = Instant::now();
            let mut this_comm: u64 = 0;
            let mut this_grav: u64 = 0;

            // ── Rebalanceo SFC (por intervalo o por desequilibrio de costo) ────
            let do_rebalance =
                should_rebalance(step, start_step, sfc_rebalance, cost_rebalance_pending);
            cost_rebalance_pending = false;
            if do_rebalance {
                let t_rb = Instant::now();
                let (gxlo, gxhi, gylo, gyhi, gzlo, gzhi) = global_bbox(rt, &local);
                let pos_loc: Vec<Vec3> = local.iter().map(|p| p.position).collect();
                sfc_decomp = SfcDecomposition::build_with_bbox_and_kind(
                    &pos_loc,
                    gxlo,
                    gxhi,
                    gylo,
                    gyhi,
                    gzlo,
                    gzhi,
                    rt.size(),
                    sfc_kind,
                );
                let rb_ns = t_rb.elapsed().as_nanos() as u64;
                this_comm += rb_ns;
                acc_hpc.domain_rebalance_ns += rb_ns;
            }

            // ── Migración de partículas ──────────────────────────────────────
            let t_domain = Instant::now();
            rt.exchange_domain_sfc(&mut local, &sfc_decomp);
            let migration_ns = t_domain.elapsed().as_nanos() as u64;
            this_comm += migration_ns;
            acc_hpc.domain_migration_ns += migration_ns;
            acc_hpc.local_particle_count += local.len();
            // Acumulamos max/min de partículas locales para calcular imbalance.
            {
                let n_loc = local.len() as f64;
                let max_n = rt.allreduce_max_f64(n_loc);
                let min_n = rt.allreduce_min_f64(n_loc);
                acc_max_local += max_n;
                acc_min_local = acc_min_local.min(min_n);
            }
            scratch.resize(local.len(), Vec3::zero());

            // ── Función de evaluación de fuerza SFC+LET ─────────────────────
            let sfc_snap = sfc_decomp.clone();
            let mut hpc = HpcStepStats::default();

            {
                let mut force_eval = |parts: &[Particle], acc: &mut [Vec3]| {
                    // 1. Allgather de AABBs (puro MPI → this_comm).
                    let my_aabb: Vec<f64> = if parts.is_empty() {
                        vec![
                            f64::INFINITY,
                            f64::NEG_INFINITY,
                            f64::INFINITY,
                            f64::NEG_INFINITY,
                            f64::INFINITY,
                            f64::NEG_INFINITY,
                        ]
                    } else {
                        let xlo = parts
                            .iter()
                            .map(|p| p.position.x)
                            .fold(f64::INFINITY, f64::min);
                        let xhi = parts
                            .iter()
                            .map(|p| p.position.x)
                            .fold(f64::NEG_INFINITY, f64::max);
                        let ylo = parts
                            .iter()
                            .map(|p| p.position.y)
                            .fold(f64::INFINITY, f64::min);
                        let yhi = parts
                            .iter()
                            .map(|p| p.position.y)
                            .fold(f64::NEG_INFINITY, f64::max);
                        let zlo = parts
                            .iter()
                            .map(|p| p.position.z)
                            .fold(f64::INFINITY, f64::min);
                        let zhi = parts
                            .iter()
                            .map(|p| p.position.z)
                            .fold(f64::NEG_INFINITY, f64::max);
                        vec![xlo, xhi, ylo, yhi, zlo, zhi]
                    };
                    let t_aabb = Instant::now();
                    let all_aabbs = rt.allgather_f64(&my_aabb);
                    let aabb_ns = t_aabb.elapsed().as_nanos() as u64;
                    hpc.aabb_allgather_ns += aabb_ns;
                    this_comm += aabb_ns;

                    // 2. Construir árbol local (cómputo → this_grav).
                    let all_pos_l: Vec<Vec3> = parts.iter().map(|p| p.position).collect();
                    let all_mass_l: Vec<f64> = parts.iter().map(|p| p.mass).collect();
                    let t_build = Instant::now();
                    let tree = Octree::build(&all_pos_l, &all_mass_l);
                    let build_ns = t_build.elapsed().as_nanos() as u64;
                    hpc.tree_build_ns += build_ns;
                    this_grav += build_ns;

                    // 3. Exportar y empaquetar nodos LET (cómputo → this_grav).
                    // Calcular theta_export: factor=0.0 significa "usa theta" (retrocompat.).
                    let f_export = cfg.performance.let_theta_export_factor;
                    let theta_export = if f_export > 0.0 {
                        theta * f_export
                    } else {
                        theta
                    };

                    let mut sends: Vec<Vec<f64>> = (0..size).map(|_| Vec::new()).collect();
                    let mut total_let_exported = 0usize;
                    let mut total_bytes_sent = 0usize;
                    let mut max_let_per_rank = 0usize;
                    let mut export_ns_eval = 0u64;
                    let mut pack_ns_eval = 0u64;
                    for r in 0..size {
                        if r == my_rank {
                            continue;
                        }
                        let ra = &all_aabbs[r];
                        if ra.len() < 6 {
                            continue;
                        }
                        let target_aabb = [ra[0], ra[1], ra[2], ra[3], ra[4], ra[5]];
                        let t_exp = Instant::now();
                        let let_nodes = tree.export_let(target_aabb, theta_export);
                        export_ns_eval += t_exp.elapsed().as_nanos() as u64;
                        let n_exp = let_nodes.len();
                        if n_exp > 0 {
                            total_let_exported += n_exp;
                            if n_exp > max_let_per_rank {
                                max_let_per_rank = n_exp;
                            }
                            let t_pack = Instant::now();
                            sends[r] = pack_let_nodes(&let_nodes);
                            pack_ns_eval += t_pack.elapsed().as_nanos() as u64;
                            total_bytes_sent += sends[r].len() * std::mem::size_of::<f64>();
                        }
                    }
                    hpc.let_export_ns += export_ns_eval;
                    hpc.let_pack_ns += pack_ns_eval;
                    this_grav += export_ns_eval + pack_ns_eval;
                    hpc.let_nodes_exported += total_let_exported;
                    hpc.bytes_sent += total_bytes_sent;
                    hpc.max_let_nodes_per_rank += max_let_per_rank;
                    hpc.local_tree_nodes += tree.node_count();

                    // 4. Alltoallv LET + walk local (overlap o bloqueante).
                    if use_overlap {
                        // ── Path no-bloqueante: walk solapa con comm ──────────
                        let mut local_accels: Vec<Vec3> = vec![Vec3::zero(); parts.len()];
                        let mut walk_ns_inner = 0u64;

                        let t_comm_total = Instant::now();
                        let received = {
                            let mut do_walk = || {
                                let t_w = Instant::now();
                                #[cfg(feature = "simd")]
                                {
                                    use rayon::prelude::*;
                                    local_accels.par_iter_mut().enumerate().for_each(|(li, a)| {
                                        *a = tree.walk_accel(
                                            parts[li].position,
                                            li,
                                            g,
                                            eps2,
                                            theta,
                                            &all_pos_l,
                                            &all_mass_l,
                                        );
                                    });
                                }
                                #[cfg(not(feature = "simd"))]
                                {
                                    for (li, a) in local_accels.iter_mut().enumerate() {
                                        *a = tree.walk_accel(
                                            parts[li].position,
                                            li,
                                            g,
                                            eps2,
                                            theta,
                                            &all_pos_l,
                                            &all_mass_l,
                                        );
                                    }
                                }
                                walk_ns_inner = t_w.elapsed().as_nanos() as u64;
                            };
                            rt.alltoallv_f64_overlap(sends, &mut do_walk)
                        };
                        let total_overlap_ns = t_comm_total.elapsed().as_nanos() as u64;
                        hpc.let_alltoallv_ns += total_overlap_ns;
                        hpc.walk_local_ns += walk_ns_inner;
                        // Tiempo de espera MPI puro ≈ total_overlap - walk
                        let wait_ns = total_overlap_ns.saturating_sub(walk_ns_inner);
                        this_comm += wait_ns;
                        this_grav += walk_ns_inner;

                        // 5. Desempaquetar nodos LET remotos.
                        let mut remote_nodes = Vec::new();
                        let mut total_bytes_recv = 0usize;
                        for buf in &received {
                            if !buf.is_empty() {
                                total_bytes_recv += buf.len() * std::mem::size_of::<f64>();
                                remote_nodes.extend(unpack_let_nodes(buf));
                            }
                        }
                        hpc.let_nodes_imported += remote_nodes.len();
                        hpc.bytes_recv += total_bytes_recv;

                        // 6. Aplicar fuerzas LET remotas: LetTree o loop plano.
                        let use_lt = cfg.performance.use_let_tree
                            && remote_nodes.len() > cfg.performance.let_tree_threshold;
                        let t_apply = Instant::now();
                        if use_lt {
                            let t_ltb = Instant::now();
                            let let_tree = gadget_ng_tree::LetTree::build_with_leaf_max(
                                &remote_nodes,
                                cfg.performance.let_tree_leaf_max,
                            );
                            let ltb_ns = t_ltb.elapsed().as_nanos() as u64;
                            hpc.let_tree_build_ns += ltb_ns;
                            hpc.let_tree_nodes += let_tree.node_count();
                            this_grav += ltb_ns;

                            gadget_ng_tree::let_tree_prof_begin();
                            let t_ltw = Instant::now();
                            #[cfg(feature = "simd")]
                            {
                                use gadget_ng_core::Vec3;
                                use rayon::prelude::*;
                                // Fase 16: loop tileado 4×N_i via par_chunks_mut(4).
                                // Cada tile procesa 4 partículas simultáneamente con walk_accel_4xi.
                                acc.par_chunks_mut(4)
                                    .zip(local_accels.par_chunks(4))
                                    .zip(parts.par_chunks(4))
                                    .for_each(|((acc_tile, local_tile), parts_tile)| {
                                        let tile_size = parts_tile.len();
                                        let mut pos = [Vec3::zero(); 4];
                                        for k in 0..tile_size {
                                            pos[k] = parts_tile[k].position;
                                        }
                                        let result =
                                            let_tree.walk_accel_4xi(pos, tile_size, g, eps2, theta);
                                        for k in 0..tile_size {
                                            acc_tile[k] = local_tile[k] + result[k];
                                        }
                                    });
                                hpc.let_tree_parallel = true;
                            }
                            #[cfg(not(feature = "simd"))]
                            {
                                for (li, a_out) in acc.iter_mut().enumerate() {
                                    let a_remote =
                                        let_tree.walk_accel(parts[li].position, g, eps2, theta);
                                    *a_out = local_accels[li] + a_remote;
                                }
                            }
                            let ltw_ns = t_ltw.elapsed().as_nanos() as u64;
                            hpc.let_tree_walk_ns += ltw_ns;
                            let (leaf_calls, leaf_rmn) = gadget_ng_tree::let_tree_prof_end();
                            hpc.apply_leaf_calls += leaf_calls;
                            hpc.apply_leaf_rmn_count += leaf_rmn;
                            let (tile_calls, tile_i) = gadget_ng_tree::let_tree_tile_prof_read();
                            hpc.apply_leaf_tile_calls += tile_calls;
                            hpc.apply_leaf_tile_i_count += tile_i;
                            this_grav += ltw_ns;
                        } else {
                            // Path flat LET: con feature "simd" usa RmnSoa para kernel fusionado.
                            #[cfg(feature = "simd")]
                            {
                                let t_pack = Instant::now();
                                let soa = gadget_ng_tree::RmnSoa::from_slice(&remote_nodes);
                                hpc.rmn_soa_pack_ns += t_pack.elapsed().as_nanos() as u64;

                                let t_soa = Instant::now();
                                use rayon::prelude::*;
                                acc.par_iter_mut().enumerate().for_each(|(li, a_out)| {
                                    *a_out = local_accels[li]
                                        + gadget_ng_tree::accel_from_let_soa(
                                            parts[li].position,
                                            &soa,
                                            g,
                                            eps2,
                                        );
                                });
                                hpc.accel_from_let_soa_ns += t_soa.elapsed().as_nanos() as u64;
                            }
                            #[cfg(not(feature = "simd"))]
                            {
                                for (li, a_out) in acc.iter_mut().enumerate() {
                                    let a_remote =
                                        accel_from_let(parts[li].position, &remote_nodes, g, eps2);
                                    *a_out = local_accels[li] + a_remote;
                                }
                            }
                        }
                        let apply_ns = t_apply.elapsed().as_nanos() as u64;
                        hpc.apply_let_ns += apply_ns;
                        this_grav += if use_lt { 0 } else { apply_ns };
                    } else {
                        // ── Path bloqueante (Fase 8 original) ────────────────
                        let t_comm2 = Instant::now();
                        let received = rt.alltoallv_f64(&sends);
                        let comm2_ns = t_comm2.elapsed().as_nanos() as u64;
                        hpc.let_alltoallv_ns += comm2_ns;
                        this_comm += comm2_ns;

                        let t_grav = Instant::now();
                        compute_forces_sfc_let(parts, &received, theta, g, eps2, acc);
                        let grav_ns = t_grav.elapsed().as_nanos() as u64;
                        hpc.walk_local_ns += grav_ns;
                        this_grav += grav_ns;

                        // Contabilizar nodos importados para stats.
                        let mut total_bytes_recv = 0usize;
                        let mut total_imported = 0usize;
                        for buf in &received {
                            if !buf.is_empty() {
                                total_bytes_recv += buf.len() * std::mem::size_of::<f64>();
                                total_imported += buf.len() / gadget_ng_tree::RMN_FLOATS;
                            }
                        }
                        hpc.let_nodes_imported += total_imported;
                        hpc.bytes_recv += total_bytes_recv;
                    }

                    let _ = sfc_snap.n_ranks(); // evitar unused warning
                };

                match integrator_kind {
                    IntegratorKind::Leapfrog => {
                        leapfrog_kdk_step(&mut local, dt, &mut scratch, &mut force_eval);
                    }
                    IntegratorKind::Yoshida4 => {
                        yoshida4_kdk_step(&mut local, dt, &mut scratch, &mut force_eval);
                    }
                }
            } // force_eval dropped → borrows de hpc liberados

            // ── Rebalanceo dinámico por costo ───────────────────────────────
            // Si max/min walk_local > rebalance_imbalance_threshold, forzar
            // rebalanceo en el próximo paso.
            let imbalance_threshold = cfg.performance.rebalance_imbalance_threshold;
            if rt.size() > 1 && hpc.walk_local_ns > 0 && imbalance_threshold > 1.0 {
                let wl = hpc.walk_local_ns as f64;
                let wl_max = rt.allreduce_max_f64(wl);
                let wl_min = rt.allreduce_min_f64(wl).max(1.0);
                if wl_max / wl_min > imbalance_threshold {
                    cost_rebalance_pending = true;
                }
            }

            // ── Actualizar acumuladores ─────────────────────────────────────
            acc_step_ns += step_start.elapsed().as_nanos() as u64;
            acc_comm_ns += this_comm;
            acc_gravity_ns += this_grav;
            acc_hpc.tree_build_ns += hpc.tree_build_ns;
            acc_hpc.let_export_ns += hpc.let_export_ns;
            acc_hpc.let_pack_ns += hpc.let_pack_ns;
            acc_hpc.aabb_allgather_ns += hpc.aabb_allgather_ns;
            acc_hpc.let_alltoallv_ns += hpc.let_alltoallv_ns;
            acc_hpc.walk_local_ns += hpc.walk_local_ns;
            acc_hpc.apply_let_ns += hpc.apply_let_ns;
            acc_hpc.let_nodes_exported += hpc.let_nodes_exported;
            acc_hpc.let_nodes_imported += hpc.let_nodes_imported;
            acc_hpc.bytes_sent += hpc.bytes_sent;
            acc_hpc.bytes_recv += hpc.bytes_recv;
            acc_hpc.let_tree_build_ns += hpc.let_tree_build_ns;
            acc_hpc.let_tree_walk_ns += hpc.let_tree_walk_ns;
            acc_hpc.let_tree_nodes += hpc.let_tree_nodes;
            acc_hpc.let_tree_parallel |= hpc.let_tree_parallel;
            acc_hpc.max_let_nodes_per_rank += hpc.max_let_nodes_per_rank;
            acc_hpc.local_tree_nodes += hpc.local_tree_nodes;
            acc_hpc.apply_leaf_ns += hpc.apply_leaf_ns;
            acc_hpc.apply_leaf_rmn_count += hpc.apply_leaf_rmn_count;
            acc_hpc.apply_leaf_calls += hpc.apply_leaf_calls;
            acc_hpc.rmn_soa_pack_ns += hpc.rmn_soa_pack_ns;
            acc_hpc.accel_from_let_soa_ns += hpc.accel_from_let_soa_ns;
            acc_hpc.apply_leaf_tile_calls += hpc.apply_leaf_tile_calls;
            acc_hpc.apply_leaf_tile_i_count += hpc.apply_leaf_tile_i_count;
            steps_run += 1;

            write_diagnostic_line(
                rt,
                step,
                &local,
                &diag_path,
                &mut diag_file,
                None,
                Some(&hpc),
                None,
            )?;
            maybe_checkpoint!(step, None);
            maybe_snap_frame!(step);
            maybe_insitu!(step);
            maybe_sph!(step);
            maybe_agn!(step);
            maybe_rt!();
            maybe_reionization!(a_current);
        }

        // Construir resumen HPC que se pasará al bloque de timings.json genérico.
        if steps_run > 0 {
            let n = steps_run as f64;
            let ns2s = 1e-9_f64;
            let total_step_s = acc_step_ns as f64 * ns2s;
            let wait_total_s = (acc_hpc
                .let_alltoallv_ns
                .saturating_sub(acc_hpc.walk_local_ns)) as f64
                * ns2s;
            hpc_aggregate_opt = Some(HpcTimingsAggregate {
                mean_tree_build_s: acc_hpc.tree_build_ns as f64 * ns2s / n,
                mean_let_export_s: acc_hpc.let_export_ns as f64 * ns2s / n,
                mean_let_pack_s: acc_hpc.let_pack_ns as f64 * ns2s / n,
                mean_aabb_allgather_s: acc_hpc.aabb_allgather_ns as f64 * ns2s / n,
                mean_let_alltoallv_s: acc_hpc.let_alltoallv_ns as f64 * ns2s / n,
                mean_walk_local_s: acc_hpc.walk_local_ns as f64 * ns2s / n,
                mean_apply_let_s: acc_hpc.apply_let_ns as f64 * ns2s / n,
                mean_let_nodes_exported: acc_hpc.let_nodes_exported as f64 / n,
                mean_let_nodes_imported: acc_hpc.let_nodes_imported as f64 / n,
                mean_bytes_sent: acc_hpc.bytes_sent as f64 / n,
                mean_bytes_recv: acc_hpc.bytes_recv as f64 / n,
                wait_fraction: if total_step_s > 0.0 {
                    wait_total_s / total_step_s
                } else {
                    0.0
                },
                mean_let_tree_build_s: acc_hpc.let_tree_build_ns as f64 * ns2s / n,
                mean_let_tree_walk_s: acc_hpc.let_tree_walk_ns as f64 * ns2s / n,
                mean_let_tree_nodes: acc_hpc.let_tree_nodes as f64 / n,
                let_tree_parallel: acc_hpc.let_tree_parallel,
                mean_max_let_nodes_per_rank: acc_hpc.max_let_nodes_per_rank as f64 / n,
                mean_local_tree_nodes: acc_hpc.local_tree_nodes as f64 / n,
                mean_export_prune_ratio: {
                    let denom = acc_hpc.local_tree_nodes as f64 * (size as f64 - 1.0).max(1.0);
                    if denom > 0.0 {
                        acc_hpc.let_nodes_exported as f64 / denom
                    } else {
                        0.0
                    }
                },
                mean_domain_rebalance_s: acc_hpc.domain_rebalance_ns as f64 * ns2s / n,
                mean_domain_migration_s: acc_hpc.domain_migration_ns as f64 * ns2s / n,
                mean_local_particle_count: acc_hpc.local_particle_count as f64 / n,
                particle_imbalance_ratio: {
                    let mean_max = acc_max_local / n;
                    let mean_min = acc_min_local.max(1.0);
                    mean_max / mean_min
                },
                sfc_kind: match cfg.performance.sfc_kind {
                    SfcKind::Morton => "morton".to_string(),
                    SfcKind::Hilbert => "hilbert".to_string(),
                },
                mean_apply_leaf_s: acc_hpc.apply_leaf_ns as f64 * ns2s / n,
                mean_apply_leaf_rmn_count: acc_hpc.apply_leaf_rmn_count as f64 / n,
                mean_apply_leaf_calls: acc_hpc.apply_leaf_calls as f64 / n,
                mean_rmn_soa_pack_s: acc_hpc.rmn_soa_pack_ns as f64 * ns2s / n,
                mean_accel_from_let_soa_s: acc_hpc.accel_from_let_soa_ns as f64 * ns2s / n,
                #[cfg(feature = "simd")]
                soa_simd_active: true,
                #[cfg(not(feature = "simd"))]
                soa_simd_active: false,
                mean_apply_leaf_tile_calls: acc_hpc.apply_leaf_tile_calls as f64 / n,
                mean_apply_leaf_tile_i_count: acc_hpc.apply_leaf_tile_i_count as f64 / n,
                tile_utilization_ratio: {
                    let calls = acc_hpc.apply_leaf_tile_calls as f64;
                    if calls > 0.0 {
                        acc_hpc.apply_leaf_tile_i_count as f64 / (calls * 4.0)
                    } else {
                        0.0
                    }
                },
                // Campos TreePM: cero para el path SFC+LET.
                mean_short_range_halo_particles: 0.0,
                mean_short_range_halo_bytes: 0.0,
                mean_tree_short_s: 0.0,
                mean_pm_long_s: 0.0,
                mean_treepm_total_s: 0.0,
                tree_fraction: 0.0,
                pm_fraction: 0.0,
                path_active: "sfc_let".to_string(),
                // Campos halo 3D: cero para path SFC+LET.
                mean_halo_3d_particles: 0.0,
                mean_halo_3d_bytes: 0.0,
                mean_halo_3d_s: 0.0,
                // Campos Fase 23: cero para path SFC+LET.
                mean_sr_domain_particle_count: 0.0,
                mean_sr_halo_3d_neighbors: 0.0,
                mean_sr_sync_s: 0.0,
                sr_sync_fraction: 0.0,
                // Campos Fase 24: cero para path SFC+LET.
                mean_pm_scatter_particles: 0.0,
                mean_pm_scatter_bytes: 0.0,
                mean_pm_scatter_s: 0.0,
                mean_pm_gather_particles: 0.0,
                mean_pm_gather_bytes: 0.0,
                mean_pm_gather_s: 0.0,
                pm_sync_fraction: 0.0,
            });
        }
    } else if use_sfc {
        // ── Árbol distribuido SFC legacy: Morton Z-order 3D, halos de partículas
        use gadget_ng_parallel::sfc::global_bbox;

        let sfc_kind_legacy = cfg.performance.sfc_kind;
        let cost_weighted = cfg.decomposition.cost_weighted;
        let ema_alpha = cfg.decomposition.ema_alpha.clamp(0.0, 1.0);

        let (gxlo, gxhi, gylo, gyhi, gzlo, gzhi) = global_bbox(rt, &local);
        let all_pos: Vec<Vec3> = local.iter().map(|p| p.position).collect();
        let mut sfc_decomp = SfcDecomposition::build_with_bbox_and_kind(
            &all_pos,
            gxlo,
            gxhi,
            gylo,
            gyhi,
            gzlo,
            gzhi,
            rt.size(),
            sfc_kind_legacy,
        );

        // Costes EMA por partícula (indexados por posición local). Inicializados a 1.0
        // (coste uniforme) antes de recibir la primera medición real.
        let mut local_particle_costs: Vec<f64> = vec![1.0; local.len()];
        // Buffer de costes crudos de cada llamada de fuerza.
        let mut raw_costs: Vec<u64> = Vec::new();

        for step in start_step..=cfg.simulation.num_steps {
            let step_start = Instant::now();
            let mut this_comm: u64 = 0;
            let mut this_grav: u64 = 0;
            let do_rebalance =
                sfc_rebalance == 0 || (step - start_step) % sfc_rebalance.max(1) == 0;
            if do_rebalance {
                let t_rb = Instant::now();
                let (gxlo, gxhi, gylo, gyhi, gzlo, gzhi) = global_bbox(rt, &local);
                let all_pos_loc: Vec<Vec3> = local.iter().map(|p| p.position).collect();
                if cost_weighted && all_pos_loc.len() == local_particle_costs.len() {
                    // Balanceo ponderado por coste de árbol acumulado (EMA).
                    sfc_decomp = SfcDecomposition::build_weighted(
                        &all_pos_loc,
                        &local_particle_costs,
                        gxlo,
                        gxhi,
                        gylo,
                        gyhi,
                        gzlo,
                        gzhi,
                        rt.size(),
                        sfc_kind_legacy,
                    );
                } else {
                    sfc_decomp = SfcDecomposition::build_with_bbox_and_kind(
                        &all_pos_loc,
                        gxlo,
                        gxhi,
                        gylo,
                        gyhi,
                        gzlo,
                        gzhi,
                        rt.size(),
                        sfc_kind_legacy,
                    );
                }
                this_comm += t_rb.elapsed().as_nanos() as u64;
            }

            let hw = sfc_decomp.halo_width(cfg.performance.halo_factor);
            let t_domain = Instant::now();
            rt.exchange_domain_sfc(&mut local, &sfc_decomp);
            this_comm += t_domain.elapsed().as_nanos() as u64;
            // Ajustar costes locales tras la migración (nuevas partículas reciben coste 1.0).
            if local.len() != local_particle_costs.len() {
                local_particle_costs.resize(local.len(), 1.0);
            }
            scratch.resize(local.len(), Vec3::zero());

            let sfc_snap = sfc_decomp.clone();
            // Flag para decidir si recoger costes en esta evaluación.
            let collect_costs = cost_weighted;
            let mut force_eval = |parts: &[Particle], acc: &mut [Vec3]| {
                let t0 = Instant::now();
                let halos = rt.exchange_halos_sfc(parts, &sfc_snap, hw);
                this_comm += t0.elapsed().as_nanos() as u64;
                let t1 = Instant::now();
                if collect_costs {
                    compute_forces_local_tree_with_costs(
                        parts,
                        &halos,
                        theta,
                        g,
                        eps2,
                        acc,
                        &mut raw_costs,
                    );
                } else {
                    compute_forces_local_tree(parts, &halos, theta, g, eps2, acc);
                }
                this_grav += t1.elapsed().as_nanos() as u64;
            };
            match integrator_kind {
                IntegratorKind::Leapfrog => {
                    leapfrog_kdk_step(&mut local, dt, &mut scratch, &mut force_eval);
                }
                IntegratorKind::Yoshida4 => {
                    yoshida4_kdk_step(&mut local, dt, &mut scratch, &mut force_eval);
                }
            }

            // Actualizar costes EMA con los valores del paso recién completado.
            if cost_weighted && raw_costs.len() == local_particle_costs.len() {
                for (ema, &raw) in local_particle_costs.iter_mut().zip(raw_costs.iter()) {
                    let new_cost = (raw as f64).max(1.0);
                    *ema = ema_alpha * new_cost + (1.0 - ema_alpha) * *ema;
                }
            }

            acc_step_ns += step_start.elapsed().as_nanos() as u64;
            acc_comm_ns += this_comm;
            acc_gravity_ns += this_grav;
            steps_run += 1;
            write_diagnostic_line(
                rt,
                step,
                &local,
                &diag_path,
                &mut diag_file,
                None,
                None,
                None,
            )?;
            maybe_checkpoint!(step, None);
            maybe_snap_frame!(step);
            maybe_insitu!(step);
            maybe_sph!(step);
            maybe_agn!(step);
            maybe_rt!();
            maybe_reionization!(a_current);
        }
    } else if use_dtree {
        // ── Árbol distribuido slab 1D: halos punto-a-punto en x ──────────────
        for step in start_step..=cfg.simulation.num_steps {
            let step_start = Instant::now();
            let mut this_comm: u64 = 0;
            let mut this_grav: u64 = 0;
            let x_lo_loc = local
                .iter()
                .map(|p| p.position.x)
                .fold(f64::INFINITY, f64::min);
            let x_hi_loc = local
                .iter()
                .map(|p| p.position.x)
                .fold(f64::NEG_INFINITY, f64::max);
            let t_allreduce = Instant::now();
            let x_lo = rt.allreduce_min_f64(x_lo_loc);
            let x_hi = rt.allreduce_max_f64(x_hi_loc);
            this_comm += t_allreduce.elapsed().as_nanos() as u64;
            let decomp = SlabDecomposition::new(x_lo, x_hi, rt.size());
            let (my_x_lo, my_x_hi) = decomp.bounds(rt.rank());
            let hw = decomp.halo_width(cfg.performance.halo_factor);

            let t_domain = Instant::now();
            rt.exchange_domain_by_x(&mut local, my_x_lo, my_x_hi);
            this_comm += t_domain.elapsed().as_nanos() as u64;
            scratch.resize(local.len(), Vec3::zero());

            let mut force_eval = |parts: &[Particle], acc: &mut [Vec3]| {
                let t0 = Instant::now();
                let halos = rt.exchange_halos_by_x(parts, my_x_lo, my_x_hi, hw);
                this_comm += t0.elapsed().as_nanos() as u64;
                let t1 = Instant::now();
                compute_forces_local_tree(parts, &halos, theta, g, eps2, acc);
                this_grav += t1.elapsed().as_nanos() as u64;
            };
            match integrator_kind {
                IntegratorKind::Leapfrog => {
                    leapfrog_kdk_step(&mut local, dt, &mut scratch, &mut force_eval);
                }
                IntegratorKind::Yoshida4 => {
                    yoshida4_kdk_step(&mut local, dt, &mut scratch, &mut force_eval);
                }
            }
            acc_step_ns += step_start.elapsed().as_nanos() as u64;
            acc_comm_ns += this_comm;
            acc_gravity_ns += this_grav;
            steps_run += 1;
            write_diagnostic_line(
                rt,
                step,
                &local,
                &diag_path,
                &mut diag_file,
                None,
                None,
                None,
            )?;
            maybe_checkpoint!(step, None);
            maybe_snap_frame!(step);
            maybe_insitu!(step);
            maybe_sph!(step);
            maybe_agn!(step);
            maybe_rt!();
            maybe_reionization!(a_current);
        }
    } else {
        // ── Leapfrog clásico: Allgather global ────────────────────────────────
        for step in start_step..=cfg.simulation.num_steps {
            let step_start = Instant::now();
            let mut this_comm: u64 = 0;
            let mut this_grav: u64 = 0;
            let mut force_eval = |parts: &[Particle], acc: &mut [Vec3]| {
                let t0 = Instant::now();
                rt.allgatherv_state(parts, total, &mut global_pos, &mut global_mass);
                this_comm += t0.elapsed().as_nanos() as u64;
                let idx: Vec<usize> = parts.iter().map(|p| p.global_id).collect();
                let t1 = Instant::now();
                solver.accelerations_for_indices(&global_pos, &global_mass, eps2, g, &idx, acc);
                this_grav += t1.elapsed().as_nanos() as u64;
            };
            match integrator_kind {
                IntegratorKind::Leapfrog => {
                    leapfrog_kdk_step(&mut local, dt, &mut scratch, &mut force_eval);
                }
                IntegratorKind::Yoshida4 => {
                    yoshida4_kdk_step(&mut local, dt, &mut scratch, &mut force_eval);
                }
            }
            acc_step_ns += step_start.elapsed().as_nanos() as u64;
            acc_comm_ns += this_comm;
            acc_gravity_ns += this_grav;
            steps_run += 1;
            write_diagnostic_line(
                rt,
                step,
                &local,
                &diag_path,
                &mut diag_file,
                None,
                None,
                None,
            )?;
            maybe_checkpoint!(step, None);
            maybe_snap_frame!(step);
            maybe_insitu!(step);
            maybe_sph!(step);
            maybe_agn!(step);
            maybe_rt!();
            maybe_reionization!(a_current);
        }
    }

    // ── Calcular TreePmAggregate si el path SR-SFC estuvo activo ─────────────
    if tpm_step_count > 0 {
        let n = tpm_step_count as f64;
        let ns2s = 1e-9_f64;
        let total_sync = (acc_tpm.scatter_ns + acc_tpm.gather_ns) as f64;
        let total_treepm = (acc_tpm.scatter_ns
            + acc_tpm.gather_ns
            + acc_tpm.pm_solve_ns
            + acc_tpm.sr_halo_ns
            + acc_tpm.tree_sr_ns) as f64;
        treepm_hpc_opt = Some(TreePmAggregate {
            mean_scatter_s: acc_tpm.scatter_ns as f64 * ns2s / n,
            mean_gather_s: acc_tpm.gather_ns as f64 * ns2s / n,
            mean_pm_solve_s: acc_tpm.pm_solve_ns as f64 * ns2s / n,
            mean_sr_halo_s: acc_tpm.sr_halo_ns as f64 * ns2s / n,
            mean_tree_sr_s: acc_tpm.tree_sr_ns as f64 * ns2s / n,
            mean_scatter_particles: acc_tpm.scatter_particles as f64 / n,
            mean_scatter_bytes: acc_tpm.scatter_bytes as f64 / n,
            mean_gather_bytes: acc_tpm.gather_bytes as f64 / n,
            pm_sync_fraction: if total_treepm > 0.0 {
                total_sync / total_treepm
            } else {
                0.0
            },
            mean_treepm_total_s: total_treepm * ns2s / n,
            path_active: acc_tpm.path,
        });
    }

    // ── Escribir timings.json ─────────────────────────────────────────────────
    if rt.rank() == 0 && steps_run > 0 {
        let total_wall_s = wall_loop_start.elapsed().as_secs_f64();
        let total_comm_s = acc_comm_ns as f64 * 1e-9;
        let total_gravity_s = acc_gravity_ns as f64 * 1e-9;
        let total_step_s = acc_step_ns as f64 * 1e-9;
        let total_integration_s = (total_step_s - total_comm_s - total_gravity_s).max(0.0);
        let report = TimingsReport {
            steps: steps_run,
            total_particles: total,
            total_wall_s,
            total_comm_s,
            total_gravity_s,
            total_integration_s,
            mean_step_wall_s: total_step_s / steps_run as f64,
            mean_comm_s: total_comm_s / steps_run as f64,
            mean_gravity_s: total_gravity_s / steps_run as f64,
            comm_fraction: if total_step_s > 0.0 {
                total_comm_s / total_step_s
            } else {
                0.0
            },
            gravity_fraction: if total_step_s > 0.0 {
                total_gravity_s / total_step_s
            } else {
                0.0
            },
            hpc: hpc_aggregate_opt,
            treepm_hpc: treepm_hpc_opt,
        };
        let timings_path = out_dir.join("timings.json");
        if let Ok(f) = fs::File::create(&timings_path) {
            let _ = serde_json::to_writer_pretty(f, &report);
        }
    }

    if write_final_snapshot {
        if let Some(parts) = rt.root_gather_particles(&local, total) {
            let snap_dir = out_dir.join("snapshot_final");
            fs::create_dir_all(&snap_dir).map_err(|e| CliError::io(&snap_dir, e))?;
            let time_final = cfg.simulation.num_steps as f64 * cfg.simulation.dt;
            // Con cosmología, redshift = 1/a - 1; sin cosmología, z = 0.
            let redshift = if cfg.cosmology.enabled {
                1.0 / a_current - 1.0
            } else {
                0.0
            };
            let env = snapshot_env_for(cfg, time_final, redshift);
            write_snapshot_formatted(cfg.output.snapshot_format, &snap_dir, &parts, &prov, &env)?;
            // Guardar el estado jerárquico junto al snapshot si aplica.
            if cfg.timestep.hierarchical {
                if let Some(ref h_state) = h_state_opt {
                    h_state
                        .save(&snap_dir)
                        .map_err(|e| CliError::io(&snap_dir, e))?;
                }
            }
        }
    }
    Ok(())
}

fn provenance_for_run(cfg: &RunConfig) -> Result<Provenance, CliError> {
    let cfg_hash = config_load::config_canonical_hash(cfg)?;
    Ok(Provenance::new(
        env!("CARGO_PKG_VERSION"),
        try_git_commit(),
        if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        }
        .to_string(),
        enabled_features_list(),
        std::env::args().collect(),
        cfg_hash,
    ))
}

/// Construye el bloque de unidades para `SnapshotEnv`, si el config usa unidades físicas.
fn snapshot_units_for(cfg: &RunConfig) -> Option<SnapshotUnits> {
    if cfg.units.enabled {
        Some(SnapshotUnits {
            length_in_kpc: cfg.units.length_in_kpc,
            mass_in_msun: cfg.units.mass_in_msun,
            velocity_in_km_s: cfg.units.velocity_in_km_s,
            time_in_gyr: cfg.units.time_unit_in_gyr(),
            g_internal: cfg.units.compute_g(),
        })
    } else {
        None
    }
}

fn snapshot_env_for(cfg: &RunConfig, time: f64, redshift: f64) -> SnapshotEnv {
    // h_dimless es h = H₀/(100 km/s/Mpc). cfg.cosmology.h0 está en unidades internas
    // (1/t_sim), así que no es equivalente. Se usa como mejor aproximación disponible;
    // para runs con unidades físicas el valor correcto es el `h` de las ICs Zeldovich.
    let (omega_m, omega_lambda, h_dimless) = if cfg.cosmology.enabled {
        (
            cfg.cosmology.omega_m,
            cfg.cosmology.omega_lambda,
            cfg.cosmology.h0,
        )
    } else {
        (0.0, 0.0, 1.0)
    };
    SnapshotEnv {
        time,
        redshift,
        box_size: cfg.simulation.box_size,
        units: snapshot_units_for(cfg),
        omega_m,
        omega_lambda,
        h_dimless,
    }
}

fn enabled_features_list() -> Vec<String> {
    let mut f = Vec::new();
    if cfg!(feature = "mpi") {
        f.push("mpi".into());
    }
    if cfg!(feature = "bincode") {
        f.push("bincode".into());
    }
    if cfg!(feature = "hdf5") {
        f.push("hdf5".into());
    }
    if cfg!(feature = "gpu") {
        f.push("gpu".into());
    }
    if cfg!(feature = "simd") {
        f.push("simd".into());
    }
    f
}

pub fn run_snapshot<R: ParallelRuntime + ?Sized>(
    rt: &R,
    cfg: &RunConfig,
    out_dir: &Path,
) -> Result<(), CliError> {
    let total = cfg.simulation.particle_count;
    let (lo, hi) = gid_block_range(total, rt.rank(), rt.size());
    let local = build_particles_for_gid_range(cfg, lo, hi)?;
    let prov = provenance_for_run(cfg)?;
    if let Some(parts) = rt.root_gather_particles(&local, total) {
        fs::create_dir_all(out_dir).map_err(|e| CliError::io(out_dir, e))?;
        let env = snapshot_env_for(cfg, 0.0, 0.0);
        write_snapshot_formatted(cfg.output.snapshot_format, out_dir, &parts, &prov, &env)?;
    }
    Ok(())
}

// ── Visualize ─────────────────────────────────────────────────────────────────

/// Lee un snapshot JSONL y renderiza las partículas a PNG.
pub fn run_visualize(
    snapshot_dir: &Path,
    out_png: &Path,
    width: u32,
    height: u32,
    projection: &str,
    color: &str,
) -> Result<(), CliError> {
    use gadget_ng_core::{SnapshotFormat, Vec3};
    use gadget_ng_vis::{ColorMode, Projection, Renderer, RendererConfig};

    let data = gadget_ng_io::read_snapshot_formatted(SnapshotFormat::Jsonl, snapshot_dir)
        .map_err(CliError::Snapshot)?;
    let box_size = data.box_size;
    let n = data.particles.len();

    if n == 0 {
        eprintln!("Advertencia: snapshot vacío en {:?}", snapshot_dir);
        return Ok(());
    }

    let positions: Vec<Vec3> = data.particles.iter().map(|p| p.position).collect();
    let velocities: Vec<Vec3> = data.particles.iter().map(|p| p.velocity).collect();

    let proj = match projection {
        "xz" => Projection::XZ,
        "yz" => Projection::YZ,
        _ => Projection::XY,
    };
    let cmode = match color {
        "white" => ColorMode::White,
        _ => ColorMode::Velocity,
    };

    let cfg = RendererConfig {
        width,
        height,
        projection: proj,
        color_mode: cmode,
        box_size,
    };
    let mut renderer = Renderer::new(cfg);
    renderer.render_frame(&positions, &velocities);

    if let Some(parent) = out_png.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).map_err(|e| CliError::io(parent, e))?;
        }
    }
    renderer
        .save_frame(out_png)
        .map_err(|e| CliError::io(out_png, std::io::Error::other(e.to_string())))?;

    println!(
        "Visualización: {n} partículas → {:?} ({}×{} px, proj={projection}, color={color})",
        out_png, width, height
    );
    Ok(())
}

// ── Analyse ───────────────────────────────────────────────────────────────────

/// Lee un snapshot JSONL y ejecuta análisis FoF + P(k).
pub fn run_analyse(
    snapshot_dir: &Path,
    out_dir: &Path,
    linking_length: f64,
    min_particles: usize,
    pk_mesh: usize,
) -> Result<(), CliError> {
    use gadget_ng_analysis::catalog::{write_halo_catalog, write_power_spectrum};
    use gadget_ng_analysis::AnalysisParams;
    use gadget_ng_core::SnapshotFormat;

    let data = gadget_ng_io::read_snapshot_formatted(SnapshotFormat::Jsonl, snapshot_dir)
        .map_err(CliError::Snapshot)?;
    let box_size = data.box_size;
    let n = data.particles.len();

    if n == 0 {
        eprintln!("Advertencia: snapshot vacío en {:?}", snapshot_dir);
        return Ok(());
    }

    // Separación media entre partículas → longitud de enlace física.
    let rho_bg = (n as f64) / (box_size * box_size * box_size);
    let l_mean = rho_bg.cbrt().recip();
    let b = linking_length * l_mean;

    let params = AnalysisParams {
        box_size,
        b,
        min_particles,
        rho_crit: rho_bg,
        pk_mesh,
    };

    let result = gadget_ng_analysis::analyse(&data.particles, &params);

    fs::create_dir_all(out_dir).map_err(|e| CliError::io(out_dir, e))?;
    write_halo_catalog(out_dir, &result.halos).map_err(|e| CliError::io(out_dir, e))?;
    write_power_spectrum(out_dir, &result.power_spectrum).map_err(|e| CliError::io(out_dir, e))?;

    println!(
        "Análisis: {n} partículas, {} halos, {} bins P(k) → {:?}",
        result.halos.len(),
        result.power_spectrum.len(),
        out_dir
    );
    Ok(())
}
