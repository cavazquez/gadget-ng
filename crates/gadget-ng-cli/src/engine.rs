use crate::config_load;
use crate::error::CliError;
#[cfg(feature = "simd")]
use gadget_ng_core::RayonDirectGravity;
use gadget_ng_core::{
    build_particles_for_gid_range, cosmology::CosmologyParams, DirectGravity, GravitySolver,
    IntegratorKind, OpeningCriterion, Particle, RunConfig, SolverKind, Vec3,
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
use gadget_ng_pm::PmSolver;
#[cfg(feature = "simd")]
use gadget_ng_tree::RayonBarnesHutGravity;
use gadget_ng_tree::{accel_from_let, pack_let_nodes, unpack_let_nodes, BarnesHutGravity, Octree};
use gadget_ng_treepm::TreePmSolver;
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
}

/// Guarda estado de checkpoint en `<out_dir>/checkpoint/`.
///
/// Solo rank 0 escribe; el directorio se sobreescribe en cada checkpoint
/// (siempre representa el último paso completado).
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
        // meta.json del checkpoint (diferente al meta.json del snapshot).
        let meta = CheckpointMeta {
            schema_version: 1,
            completed_step,
            a_current,
            config_hash: cfg_hash.to_owned(),
            total_particles: total,
            has_hierarchical_state: h_state.is_some(),
        };
        let meta_path = ck_dir.join("checkpoint.json");
        fs::write(&meta_path, serde_json::to_string_pretty(&meta)?)
            .map_err(|e| CliError::io(&meta_path, e))?;
    }
    rt.barrier();
    Ok(())
}

/// Carga el estado de checkpoint desde `<resume_dir>/checkpoint/`.
/// Devuelve `(partículas_locales, completed_step, a_current, h_state_opt)`.
fn load_checkpoint<R: ParallelRuntime + ?Sized>(
    rt: &R,
    resume_dir: &Path,
    lo: usize,
    hi: usize,
    cfg_hash: &str,
) -> Result<(Vec<Particle>, u64, f64, Option<HierarchicalState>), CliError> {
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
    Ok((local, meta.completed_step, meta.a_current, h_state))
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

fn write_diagnostic_line<R: ParallelRuntime + ?Sized>(
    rt: &R,
    step: u64,
    local: &[Particle],
    diag_path: &Path,
    diag_file: &mut Option<File>,
    step_stats: Option<&StepStats>,
    hpc_stats: Option<&HpcStepStats>,
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
        let a_local =
            tree.walk_accel(parts[li].position, li, g, eps2, theta, &all_pos, &all_mass);
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
    let eps2 = cfg.softening_squared();
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
    let (mut local, start_step, mut a_current, mut h_state_resume) =
        if let Some(resume_dir) = resume_from {
            rt.root_eprintln(&format!(
                "[gadget-ng] Reanudando desde checkpoint en {:?}",
                resume_dir.join("checkpoint")
            ));
            let (p, completed, a, hs) = load_checkpoint(rt, resume_dir, lo, hi, &cfg_hash)?;
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

    // SFC+LET es el path por defecto para multirank + BarnesHut.
    // Se desactiva solo con `force_allgather_fallback = true` (legacy) o tamaño 1.
    let use_sfc_let = is_barnes_hut_eligible
        && rt.size() > 1
        && !cfg.performance.force_allgather_fallback;

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

    if use_sfc_let {
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

    write_diagnostic_line(rt, 0, &local, &diag_path, &mut diag_file, None, None)?;

    // `h_state_opt` se mantiene vivo tras el bucle para poder guardarlo con el snapshot.
    let mut h_state_opt: Option<HierarchicalState> = None;

    // Macro local para guardar checkpoint cuando toca.
    macro_rules! maybe_checkpoint {
        ($step:expr, $hs:expr) => {
            if checkpoint_interval > 0 && $step % checkpoint_interval == 0 {
                save_checkpoint(rt, $step, a_current, &local, total, $hs, out_dir, &cfg_hash)?;
            }
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
                    let env = SnapshotEnv {
                        time: t,
                        redshift: z,
                        box_size: cfg.simulation.box_size,
                        units: snapshot_units_for(cfg),
                    };
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

    // ── Acumuladores de tiempos por fase ─────────────────────────────────────
    let mut acc_comm_ns: u64 = 0;
    let mut acc_gravity_ns: u64 = 0;
    let mut acc_step_ns: u64 = 0;
    let mut steps_run: u64 = 0;
    let wall_loop_start = Instant::now();
    // Resumen HPC detallado; solo se puebla en el path SFC+LET.
    let mut hpc_aggregate_opt: Option<HpcTimingsAggregate> = None;

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
        // Reutilizar HierarchicalState del checkpoint, o crear uno nuevo.
        let mut h_state = h_state_resume.take().unwrap_or_else(|| {
            let mut hs = HierarchicalState::new(local.len());
            // Aceleraciones iniciales para el primer kick.
            rt.allgatherv_state(&local, total, &mut global_pos, &mut global_mass);
            let init_idx: Vec<usize> = local.iter().map(|p| p.global_id).collect();
            solver.accelerations_for_indices(
                &global_pos,
                &global_mass,
                eps2,
                g,
                &init_idx,
                &mut scratch,
            );
            for (p, &a) in local.iter_mut().zip(scratch.iter()) {
                p.acceleration = a;
            }
            hs.init_from_accels(&local, eps2, dt, eta, max_level, criterion);
            hs
        });

        for step in start_step..=cfg.simulation.num_steps {
            let step_start = Instant::now();
            let mut this_comm: u64 = 0;
            let mut this_grav: u64 = 0;
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
                |parts, active_local, acc| {
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
                        g,
                        &global_idx,
                        acc,
                    );
                    this_grav += t1.elapsed().as_nanos() as u64;
                },
            );
            acc_step_ns += step_start.elapsed().as_nanos() as u64;
            acc_comm_ns += this_comm;
            acc_gravity_ns += this_grav;
            steps_run += 1;
            write_diagnostic_line(rt, step, &local, &diag_path, &mut diag_file, Some(&step_stats), None)?;
            maybe_checkpoint!(step, Some(&h_state));
            maybe_snap_frame!(step);
        }
        h_state_opt = Some(h_state);
    } else if let Some((ref cosmo_params, _)) = cosmo_state {
        // Leapfrog / Yoshida4 cosmológico: factores drift/kick calculados por paso.
        for step in start_step..=cfg.simulation.num_steps {
            let step_start = Instant::now();
            let mut this_comm: u64 = 0;
            let mut this_grav: u64 = 0;
            let mut compute_acc =
                |parts: &[Particle], acc: &mut [Vec3], this_comm: &mut u64, this_grav: &mut u64| {
                    let t0 = Instant::now();
                    rt.allgatherv_state(parts, total, &mut global_pos, &mut global_mass);
                    *this_comm += t0.elapsed().as_nanos() as u64;
                    let idx: Vec<usize> = parts.iter().map(|p| p.global_id).collect();
                    let t1 = Instant::now();
                    solver.accelerations_for_indices(
                        &global_pos,
                        &global_mass,
                        eps2,
                        g,
                        &idx,
                        acc,
                    );
                    *this_grav += t1.elapsed().as_nanos() as u64;
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
            acc_step_ns += step_start.elapsed().as_nanos() as u64;
            acc_comm_ns += this_comm;
            acc_gravity_ns += this_grav;
            steps_run += 1;
            write_diagnostic_line(rt, step, &local, &diag_path, &mut diag_file, None, None)?;
            maybe_checkpoint!(step, None);
            maybe_snap_frame!(step);
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

        let (gxlo, gxhi, gylo, gyhi, gzlo, gzhi) = global_bbox(rt, &local);
        let all_pos: Vec<Vec3> = local.iter().map(|p| p.position).collect();
        let mut sfc_decomp = SfcDecomposition::build_with_bbox(
            &all_pos, gxlo, gxhi, gylo, gyhi, gzlo, gzhi, rt.size(),
        );
        let size = rt.size() as usize;
        let my_rank = rt.rank() as usize;
        let use_overlap = cfg.performance.let_nonblocking;

        // Acumuladores HPC agregados para TimingsReport.
        let mut acc_hpc = HpcStepStats::default();
        // Umbral para rebalanceo por costo: si max/min walk_local_ns > 1.3 → rebalanceo inmediato.
        let mut cost_rebalance_pending = false;

        for step in start_step..=cfg.simulation.num_steps {
            let step_start = Instant::now();
            let mut this_comm: u64 = 0;
            let mut this_grav: u64 = 0;

            // ── Rebalanceo SFC (por intervalo o por desequilibrio de costo) ────
            let do_rebalance = cost_rebalance_pending
                || sfc_rebalance == 0
                || (step - start_step) % sfc_rebalance.max(1) == 0;
            cost_rebalance_pending = false;
            if do_rebalance {
                let t_rb = Instant::now();
                let (gxlo, gxhi, gylo, gyhi, gzlo, gzhi) = global_bbox(rt, &local);
                let pos_loc: Vec<Vec3> = local.iter().map(|p| p.position).collect();
                sfc_decomp = SfcDecomposition::build_with_bbox(
                    &pos_loc, gxlo, gxhi, gylo, gyhi, gzlo, gzhi, rt.size(),
                );
                this_comm += t_rb.elapsed().as_nanos() as u64;
            }

            // ── Migración de partículas ──────────────────────────────────────
            let t_domain = Instant::now();
            rt.exchange_domain_sfc(&mut local, &sfc_decomp);
            this_comm += t_domain.elapsed().as_nanos() as u64;
            scratch.resize(local.len(), Vec3::zero());

            // ── Función de evaluación de fuerza SFC+LET ─────────────────────
            let sfc_snap = sfc_decomp.clone();
            let mut hpc = HpcStepStats::default();

            {
                let mut force_eval = |parts: &[Particle], acc: &mut [Vec3]| {
                    // 1. Allgather de AABBs (puro MPI → this_comm).
                    let my_aabb: Vec<f64> = if parts.is_empty() {
                        vec![
                            f64::INFINITY, f64::NEG_INFINITY,
                            f64::INFINITY, f64::NEG_INFINITY,
                            f64::INFINITY, f64::NEG_INFINITY,
                        ]
                    } else {
                        let xlo = parts.iter().map(|p| p.position.x).fold(f64::INFINITY, f64::min);
                        let xhi = parts.iter().map(|p| p.position.x).fold(f64::NEG_INFINITY, f64::max);
                        let ylo = parts.iter().map(|p| p.position.y).fold(f64::INFINITY, f64::min);
                        let yhi = parts.iter().map(|p| p.position.y).fold(f64::NEG_INFINITY, f64::max);
                        let zlo = parts.iter().map(|p| p.position.z).fold(f64::INFINITY, f64::min);
                        let zhi = parts.iter().map(|p| p.position.z).fold(f64::NEG_INFINITY, f64::max);
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
                    let theta_export = if f_export > 0.0 { theta * f_export } else { theta };

                    let mut sends: Vec<Vec<f64>> = (0..size).map(|_| Vec::new()).collect();
                    let mut total_let_exported = 0usize;
                    let mut total_bytes_sent = 0usize;
                    let mut max_let_per_rank = 0usize;
                    let mut export_ns_eval = 0u64;
                    let mut pack_ns_eval = 0u64;
                    for r in 0..size {
                        if r == my_rank { continue; }
                        let ra = &all_aabbs[r];
                        if ra.len() < 6 { continue; }
                        let target_aabb = [ra[0], ra[1], ra[2], ra[3], ra[4], ra[5]];
                        let t_exp = Instant::now();
                        let let_nodes = tree.export_let(target_aabb, theta_export);
                        export_ns_eval += t_exp.elapsed().as_nanos() as u64;
                        let n_exp = let_nodes.len();
                        if n_exp > 0 {
                            total_let_exported += n_exp;
                            if n_exp > max_let_per_rank { max_let_per_rank = n_exp; }
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
                                    local_accels
                                        .par_iter_mut()
                                        .enumerate()
                                        .for_each(|(li, a)| {
                                            *a = tree.walk_accel(
                                                parts[li].position,
                                                li,
                                                g, eps2, theta,
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
                                            g, eps2, theta,
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

                            let t_ltw = Instant::now();
                            #[cfg(feature = "simd")]
                            {
                                use rayon::prelude::*;
                                acc.par_iter_mut().enumerate().for_each(|(li, a_out)| {
                                    *a_out = local_accels[li]
                                        + let_tree.walk_accel(parts[li].position, g, eps2, theta);
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
                            this_grav += ltw_ns;
                        } else {
                            for (li, a_out) in acc.iter_mut().enumerate() {
                                let a_remote =
                                    accel_from_let(parts[li].position, &remote_nodes, g, eps2);
                                *a_out = local_accels[li] + a_remote;
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
                                total_imported +=
                                    buf.len() / gadget_ng_tree::RMN_FLOATS;
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
            // Si max/min walk_local > 1.3, forzar rebalanceo en el próximo paso.
            if rt.size() > 1 && hpc.walk_local_ns > 0 {
                let wl = hpc.walk_local_ns as f64;
                let wl_max = rt.allreduce_max_f64(wl);
                let wl_min = rt.allreduce_min_f64(wl).max(1.0);
                if wl_max / wl_min > 1.3 {
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
            steps_run += 1;

            write_diagnostic_line(
                rt, step, &local, &diag_path, &mut diag_file, None, Some(&hpc),
            )?;
            maybe_checkpoint!(step, None);
            maybe_snap_frame!(step);
        }

        // Construir resumen HPC que se pasará al bloque de timings.json genérico.
        if steps_run > 0 {
            let n = steps_run as f64;
            let ns2s = 1e-9_f64;
            let total_step_s = acc_step_ns as f64 * ns2s;
            let wait_total_s = (acc_hpc.let_alltoallv_ns.saturating_sub(acc_hpc.walk_local_ns))
                as f64
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
            });
        }
    } else if use_sfc {
        // ── Árbol distribuido SFC legacy: Morton Z-order 3D, halos de partículas
        use gadget_ng_parallel::sfc::global_bbox;

        let (gxlo, gxhi, gylo, gyhi, gzlo, gzhi) = global_bbox(rt, &local);
        let all_pos: Vec<Vec3> = local.iter().map(|p| p.position).collect();
        let mut sfc_decomp = SfcDecomposition::build_with_bbox(
            &all_pos, gxlo, gxhi, gylo, gyhi, gzlo, gzhi, rt.size(),
        );

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
                sfc_decomp = SfcDecomposition::build_with_bbox(
                    &all_pos_loc, gxlo, gxhi, gylo, gyhi, gzlo, gzhi, rt.size(),
                );
                this_comm += t_rb.elapsed().as_nanos() as u64;
            }

            let hw = sfc_decomp.halo_width(cfg.performance.halo_factor);
            let t_domain = Instant::now();
            rt.exchange_domain_sfc(&mut local, &sfc_decomp);
            this_comm += t_domain.elapsed().as_nanos() as u64;
            scratch.resize(local.len(), Vec3::zero());

            let sfc_snap = sfc_decomp.clone();
            let mut force_eval = |parts: &[Particle], acc: &mut [Vec3]| {
                let t0 = Instant::now();
                let halos = rt.exchange_halos_sfc(parts, &sfc_snap, hw);
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
            write_diagnostic_line(rt, step, &local, &diag_path, &mut diag_file, None, None)?;
            maybe_checkpoint!(step, None);
            maybe_snap_frame!(step);
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
            write_diagnostic_line(rt, step, &local, &diag_path, &mut diag_file, None, None)?;
            maybe_checkpoint!(step, None);
            maybe_snap_frame!(step);
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
            write_diagnostic_line(rt, step, &local, &diag_path, &mut diag_file, None, None)?;
            maybe_checkpoint!(step, None);
            maybe_snap_frame!(step);
        }
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
            let env = SnapshotEnv {
                time: time_final,
                redshift,
                box_size: cfg.simulation.box_size,
                units: snapshot_units_for(cfg),
            };
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
        let env = SnapshotEnv {
            time: 0.0,
            redshift: 0.0,
            box_size: cfg.simulation.box_size,
            units: snapshot_units_for(cfg),
        };
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
