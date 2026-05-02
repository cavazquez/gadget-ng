//! Local Barnes–Hut / LET kernels and gravity solver construction (`make_solver`).

#[cfg(feature = "simd")]
use gadget_ng_core::RayonDirectGravity;
use gadget_ng_core::{
    DirectGravity, GravitySolver, OpeningCriterion, Particle, RunConfig, SolverKind, Vec3,
};
use gadget_ng_pm::PmSolver;
#[cfg(feature = "simd")]
use gadget_ng_tree::RayonBarnesHutGravity;
use gadget_ng_tree::{
    BarnesHutGravity, Octree, accel_from_let, unpack_let_nodes, walk_stats_begin, walk_stats_end,
};
use gadget_ng_treepm::TreePmSolver;

/// Calcula aceleraciones para `parts` (partículas locales) usando un árbol construido
/// a partir de `parts` + `halos` (partículas de rangos vecinos).
///
/// - `parts[0..n_local]`  → partículas propias; sus índices locales (0..n_local) sirven
///   para la auto-exclusión en el árbol.
/// - `halos` → partículas recibidas del halo; se incluyen en el árbol pero no se
///   computan sus aceleraciones.
pub(crate) fn compute_forces_local_tree(
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
pub(crate) fn compute_forces_local_tree_with_costs(
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
pub(crate) fn compute_forces_hierarchical_let(
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
pub(crate) fn compute_forces_sfc_let(
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

pub(crate) fn make_solver(cfg: &RunConfig) -> Box<dyn GravitySolver> {
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
