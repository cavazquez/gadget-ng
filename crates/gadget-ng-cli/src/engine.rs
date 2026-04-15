use crate::config_load;
use crate::error::CliError;
#[cfg(feature = "simd")]
use gadget_ng_core::RayonDirectGravity;
use gadget_ng_core::{
    build_particles_for_gid_range, DirectGravity, GravitySolver, Particle, RunConfig, SolverKind,
    Vec3,
};
use gadget_ng_integrators::{hierarchical_kdk_step, leapfrog_kdk_step, HierarchicalState};
use gadget_ng_io::{write_snapshot_formatted, Provenance, SnapshotEnv};
use gadget_ng_parallel::{gid_block_range, ParallelRuntime};
use gadget_ng_pm::PmSolver;
use gadget_ng_treepm::TreePmSolver;
use gadget_ng_tree::BarnesHutGravity;
#[cfg(feature = "simd")]
use gadget_ng_tree::RayonBarnesHutGravity;
use std::fs;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::process::Command;

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

fn write_diagnostic_line<R: ParallelRuntime + ?Sized>(
    rt: &R,
    step: u64,
    local: &[Particle],
    diag_path: &Path,
    diag_file: &mut Option<File>,
) -> Result<(), CliError> {
    let ke_loc = kinetic_local(local);
    let ke = rt.allreduce_sum_f64(ke_loc);
    if let Some(ref mut f) = diag_file {
        let line = serde_json::json!({"step": step, "kinetic_energy": ke}).to_string();
        writeln!(f, "{line}").map_err(|e| CliError::io(diag_path, e))?;
    }
    rt.barrier();
    Ok(())
}

fn make_solver(cfg: &RunConfig) -> Box<dyn GravitySolver> {
    // TODO(gpu): cuando `cfg.performance.use_gpu` esté disponible en el schema
    // TOML, descomentar el bloque siguiente para enrutar al kernel GPU:
    //
    // #[cfg(feature = "gpu")]
    // if cfg.performance.use_gpu {
    //     return Box::new(gadget_ng_core::GpuDirectGravity);
    // }

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
            }),
            SolverKind::Pm | SolverKind::TreePm => unreachable!("handled above"),
        };
    }
    // Modo serial (default): determinismo garantizado.
    match cfg.gravity.solver {
        SolverKind::Direct => Box::new(DirectGravity),
        SolverKind::BarnesHut => Box::new(BarnesHutGravity {
            theta: cfg.gravity.theta,
        }),
        SolverKind::Pm | SolverKind::TreePm => unreachable!("handled above"),
    }
}

pub fn run_stepping<R: ParallelRuntime + ?Sized>(
    rt: &R,
    cfg: &RunConfig,
    out_dir: &Path,
    write_final_snapshot: bool,
) -> Result<(), CliError> {
    let total = cfg.simulation.particle_count;
    let (lo, hi) = gid_block_range(total, rt.rank(), rt.size());
    let mut local = build_particles_for_gid_range(cfg, lo, hi)?;
    let mut scratch = vec![Vec3::zero(); local.len()];
    let mut global_pos: Vec<Vec3> = Vec::new();
    let mut global_mass: Vec<f64> = Vec::new();
    let g = cfg.simulation.gravitational_constant;
    let eps2 = cfg.softening_squared();
    let solver = make_solver(cfg);
    let dt = cfg.simulation.dt;

    fs::create_dir_all(out_dir).map_err(|e| CliError::io(out_dir, e))?;
    let prov = provenance_for_run(cfg)?;

    let diag_path = out_dir.join("diagnostics.jsonl");
    let mut diag_file = if rt.rank() == 0 {
        Some(fs::File::create(&diag_path).map_err(|e| CliError::io(&diag_path, e))?)
    } else {
        None
    };

    write_diagnostic_line(rt, 0, &local, &diag_path, &mut diag_file)?;

    if cfg.timestep.hierarchical {
        let eta = cfg.timestep.eta;
        let max_level = cfg.timestep.max_level;
        let mut h_state = HierarchicalState::new(local.len());
        // Calcular aceleraciones iniciales para que el primer START kick sea correcto.
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
        h_state.init_from_accels(&local, eps2, dt, eta, max_level);

        for step in 1..=cfg.simulation.num_steps {
            hierarchical_kdk_step(
                &mut local,
                &mut h_state,
                dt,
                eps2,
                eta,
                max_level,
                |parts, active_local, acc| {
                    rt.allgatherv_state(parts, total, &mut global_pos, &mut global_mass);
                    // Convertir índices locales a global_id para el solver.
                    let global_idx: Vec<usize> =
                        active_local.iter().map(|&li| parts[li].global_id).collect();
                    solver.accelerations_for_indices(
                        &global_pos,
                        &global_mass,
                        eps2,
                        g,
                        &global_idx,
                        acc,
                    );
                },
            );
            write_diagnostic_line(rt, step, &local, &diag_path, &mut diag_file)?;
        }
    } else {
        for step in 1..=cfg.simulation.num_steps {
            leapfrog_kdk_step(&mut local, dt, &mut scratch, |parts, acc| {
                rt.allgatherv_state(parts, total, &mut global_pos, &mut global_mass);
                let idx: Vec<usize> = parts.iter().map(|p| p.global_id).collect();
                solver.accelerations_for_indices(&global_pos, &global_mass, eps2, g, &idx, acc);
            });
            write_diagnostic_line(rt, step, &local, &diag_path, &mut diag_file)?;
        }
    }

    if write_final_snapshot {
        if let Some(parts) = rt.root_gather_particles(&local, total) {
            let snap_dir = out_dir.join("snapshot_final");
            let env = SnapshotEnv {
                time: cfg.simulation.num_steps as f64 * cfg.simulation.dt,
                redshift: 0.0,
                box_size: cfg.simulation.box_size,
            };
            write_snapshot_formatted(cfg.output.snapshot_format, &snap_dir, &parts, &prov, &env)?;
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
        };
        write_snapshot_formatted(cfg.output.snapshot_format, out_dir, &parts, &prov, &env)?;
    }
    Ok(())
}
