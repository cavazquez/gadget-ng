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
use gadget_ng_tree::{BarnesHutGravity, Octree};
use gadget_ng_treepm::TreePmSolver;
use std::fs;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::process::Command;
use std::time::Instant;

// ── Timing report ─────────────────────────────────────────────────────────────

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
    // Combinar posiciones y masas: [locales, halos]
    let all_pos: Vec<Vec3> = parts
        .iter()
        .chain(halos.iter())
        .map(|p| p.position)
        .collect();
    let all_mass: Vec<f64> = parts.iter().chain(halos.iter()).map(|p| p.mass).collect();
    let tree = Octree::build(&all_pos, &all_mass);
    // Solo se computan fuerzas para los índices locales (0..parts.len()).
    // El índice LOCAL `li` coincide con `particle_idx` en las hojas del árbol local.
    for (li, acc_out) in out.iter_mut().enumerate() {
        *acc_out = tree.walk_accel(parts[li].position, li, g, eps2, theta, &all_pos, &all_mass);
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
    let use_dtree = cfg.performance.use_distributed_tree
        && cfg.gravity.solver == SolverKind::BarnesHut
        && !cfg.timestep.hierarchical
        && !cfg.cosmology.enabled;
    // SFC: requiere use_distributed_tree + use_sfc.
    let use_sfc = use_dtree && cfg.performance.use_sfc;
    let sfc_rebalance = cfg.performance.sfc_rebalance_interval;
    if use_sfc {
        rt.root_eprintln(
            "[gadget-ng] Árbol distribuido SFC (Morton Z-order 3D, balanceo dinámico).",
        );
    } else if use_dtree {
        rt.root_eprintln("[gadget-ng] Árbol distribuido activo (halos punto-a-punto en x).");
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

    write_diagnostic_line(rt, 0, &local, &diag_path, &mut diag_file, None)?;

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
            write_diagnostic_line(rt, step, &local, &diag_path, &mut diag_file, Some(&step_stats))?;
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
            write_diagnostic_line(rt, step, &local, &diag_path, &mut diag_file, None)?;
            maybe_checkpoint!(step, None);
            maybe_snap_frame!(step);
        }
    } else if use_sfc {
        // ── Árbol distribuido SFC: Morton Z-order 3D, balanceo dinámico ───────
        let box_size = cfg.simulation.box_size;
        let all_pos: Vec<Vec3> = local.iter().map(|p| p.position).collect();
        let mut sfc_decomp = SfcDecomposition::build(&all_pos, box_size, rt.size());

        for step in start_step..=cfg.simulation.num_steps {
            let step_start = Instant::now();
            let mut this_comm: u64 = 0;
            let mut this_grav: u64 = 0;
            let do_rebalance =
                sfc_rebalance == 0 || (step - start_step) % sfc_rebalance.max(1) == 0;
            if do_rebalance {
                let all_pos_loc: Vec<Vec3> = local.iter().map(|p| p.position).collect();
                sfc_decomp = SfcDecomposition::build(&all_pos_loc, box_size, rt.size());
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
            write_diagnostic_line(rt, step, &local, &diag_path, &mut diag_file, None)?;
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
            write_diagnostic_line(rt, step, &local, &diag_path, &mut diag_file, None)?;
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
            write_diagnostic_line(rt, step, &local, &diag_path, &mut diag_file, None)?;
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
