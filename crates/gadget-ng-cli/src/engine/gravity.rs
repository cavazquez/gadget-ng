//! Local Barnes–Hut / LET kernels and gravity solver construction (`make_solver`).

#[cfg(feature = "simd")]
use gadget_ng_core::RayonDirectGravity;
use gadget_ng_core::{
    DirectGravity, GravitySolver, MacSoftening, OpeningCriterion, Particle, RunConfig, SolverKind,
    Vec3,
};
use gadget_ng_pm::PmSolver;
#[cfg(feature = "simd")]
use gadget_ng_tree::RayonBarnesHutGravity;
use gadget_ng_tree::{
    BarnesHutGravity, Octree, accel_from_let, unpack_let_nodes, walk_stats_begin, walk_stats_end,
};
use gadget_ng_treepm::TreePmSolver;

/// Parámetros del walk multipolar local (mismos que [`BarnesHutGravity`] / `[gravity]` en TOML).
#[derive(Clone, Copy)]
pub(crate) struct LocalBhWalkParams {
    pub theta: f64,
    pub multipole_order: u8,
    pub use_relative_criterion: bool,
    pub err_tol_force_acc: f64,
    pub softened_multipoles: bool,
    pub mac_softening: MacSoftening,
}

pub(crate) fn local_bh_walk_params(cfg: &RunConfig) -> LocalBhWalkParams {
    LocalBhWalkParams {
        theta: cfg.gravity.theta,
        multipole_order: cfg.gravity.multipole_order,
        use_relative_criterion: cfg.gravity.opening_criterion == OpeningCriterion::Relative,
        err_tol_force_acc: cfg.gravity.err_tol_force_acc,
        softened_multipoles: cfg.gravity.softened_multipoles,
        mac_softening: cfg.gravity.mac_softening,
    }
}

/// Rayon en el recorrido del árbol local (MPI / SFC): mismo criterio que [`make_solver`]
/// (`feature simd` + `[performance] deterministic = false` + solver BH).
#[cfg(feature = "simd")]
pub(crate) fn local_bh_use_rayon(cfg: &RunConfig) -> bool {
    cfg.gravity.solver == SolverKind::BarnesHut && !cfg.performance.deterministic
}

#[cfg(not(feature = "simd"))]
pub(crate) fn local_bh_use_rayon(_cfg: &RunConfig) -> bool {
    false
}

#[inline]
fn walk_accel_local(
    tree: &Octree,
    pos_i: Vec3,
    gi: usize,
    g: f64,
    eps2: f64,
    positions: &[Vec3],
    masses: &[f64],
    bh: LocalBhWalkParams,
) -> Vec3 {
    tree.walk_accel_multipole(
        pos_i,
        gi,
        g,
        eps2,
        bh.theta,
        positions,
        masses,
        bh.multipole_order,
        bh.use_relative_criterion,
        bh.err_tol_force_acc,
        bh.softened_multipoles,
        bh.mac_softening,
    )
}

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
    g: f64,
    eps2: f64,
    out: &mut [Vec3],
    bh: LocalBhWalkParams,
    parallel_rayon: bool,
) {
    debug_assert_eq!(parts.len(), out.len());
    #[cfg(not(feature = "simd"))]
    let _ = parallel_rayon;
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
    #[cfg(feature = "simd")]
    if parallel_rayon {
        use rayon::prelude::*;
        out.par_iter_mut().enumerate().for_each(|(li, acc_out)| {
            *acc_out = walk_accel_local(
                &tree,
                parts[li].position,
                li,
                g,
                eps2,
                &all_pos,
                &all_mass,
                bh,
            );
        });
        return;
    }
    for (li, acc_out) in out.iter_mut().enumerate() {
        *acc_out = walk_accel_local(
            &tree,
            parts[li].position,
            li,
            g,
            eps2,
            &all_pos,
            &all_mass,
            bh,
        );
    }
}

/// Variante de `compute_forces_local_tree` que además devuelve el coste de interacción
/// (nodos abiertos del walk) por partícula local. Se usa para el balanceo SFC ponderado.
pub(crate) fn compute_forces_local_tree_with_costs(
    parts: &[Particle],
    halos: &[Particle],
    g: f64,
    eps2: f64,
    out: &mut [Vec3],
    costs: &mut Vec<u64>,
    bh: LocalBhWalkParams,
    parallel_rayon: bool,
) {
    debug_assert_eq!(parts.len(), out.len());
    #[cfg(not(feature = "simd"))]
    let _ = parallel_rayon;
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
    #[cfg(feature = "simd")]
    if parallel_rayon {
        use rayon::prelude::*;
        costs.resize(parts.len(), 0);
        out.par_iter_mut()
            .zip(costs.par_iter_mut())
            .enumerate()
            .for_each(|(li, (acc_out, cost_slot))| {
                walk_stats_begin();
                *acc_out = walk_accel_local(
                    &tree,
                    parts[li].position,
                    li,
                    g,
                    eps2,
                    &all_pos,
                    &all_mass,
                    bh,
                );
                *cost_slot = walk_stats_end().opened_nodes;
            });
        return;
    }
    for (li, acc_out) in out.iter_mut().enumerate() {
        walk_stats_begin();
        *acc_out = walk_accel_local(
            &tree,
            parts[li].position,
            li,
            g,
            eps2,
            &all_pos,
            &all_mass,
            bh,
        );
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
    g: f64,
    eps2: f64,
    acc: &mut [Vec3],
    bh: LocalBhWalkParams,
    parallel_rayon: bool,
) {
    debug_assert_eq!(acc.len(), active_local.len());
    #[cfg(not(feature = "simd"))]
    let _ = parallel_rayon;
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
    #[cfg(feature = "simd")]
    if parallel_rayon {
        use rayon::prelude::*;
        acc.par_iter_mut()
            .zip(active_local.par_iter())
            .for_each(|(acc_j, &li)| {
                *acc_j = walk_accel_local(
                    &tree,
                    parts[li].position,
                    li,
                    g,
                    eps2,
                    &all_pos,
                    &all_mass,
                    bh,
                );
            });
        return;
    }
    for (j, &li) in active_local.iter().enumerate() {
        acc[j] = walk_accel_local(
            &tree,
            parts[li].position,
            li,
            g,
            eps2,
            &all_pos,
            &all_mass,
            bh,
        );
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
    g: f64,
    eps2: f64,
    out: &mut [Vec3],
    bh: LocalBhWalkParams,
    parallel_rayon: bool,
) {
    debug_assert_eq!(parts.len(), out.len());
    #[cfg(not(feature = "simd"))]
    let _ = parallel_rayon;
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

    #[cfg(feature = "simd")]
    if parallel_rayon {
        use rayon::prelude::*;
        out.par_iter_mut().enumerate().for_each(|(li, acc_out)| {
            let a_local = walk_accel_local(
                &tree,
                parts[li].position,
                li,
                g,
                eps2,
                &all_pos,
                &all_mass,
                bh,
            );
            let a_remote = accel_from_let(parts[li].position, &remote_nodes, g, eps2);
            *acc_out = a_local + a_remote;
        });
        return;
    }

    for (li, acc_out) in out.iter_mut().enumerate() {
        let a_local = walk_accel_local(
            &tree,
            parts[li].position,
            li,
            g,
            eps2,
            &all_pos,
            &all_mass,
            bh,
        );
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

#[cfg(all(test, feature = "simd"))]
mod local_tree_rayon_tests {
    use super::{
        LocalBhWalkParams, compute_forces_local_tree, local_bh_use_rayon, walk_accel_local,
    };
    use gadget_ng_core::{MacSoftening, Particle, RunConfig, SolverKind, Vec3};

    fn sample_bh_params() -> LocalBhWalkParams {
        LocalBhWalkParams {
            theta: 0.5,
            multipole_order: 3,
            use_relative_criterion: false,
            err_tol_force_acc: 0.005,
            softened_multipoles: false,
            mac_softening: MacSoftening::Bare,
        }
    }

    /// Partículas en retícula 4×4×1 (16) para ejercitar el walk sin depender de ICs TOML.
    fn lattice_16() -> Vec<Particle> {
        (0..16)
            .map(|i| {
                let x = (i % 4) as f64 * 0.2 + 0.01;
                let y = ((i / 4) % 4) as f64 * 0.2 + 0.01;
                let z = 0.5;
                Particle::new(i, 1.0, Vec3::new(x, y, z), Vec3::zero())
            })
            .collect()
    }

    #[test]
    fn compute_forces_local_tree_parallel_matches_serial() {
        let parts = lattice_16();
        let halos: Vec<Particle> = Vec::new();
        let g = 1.0;
        let eps2 = 0.01_f64;
        let bh = sample_bh_params();

        let mut out_ser = vec![Vec3::zero(); parts.len()];
        compute_forces_local_tree(&parts, &halos, g, eps2, &mut out_ser, bh, false);

        let mut out_par = vec![Vec3::zero(); parts.len()];
        compute_forces_local_tree(&parts, &halos, g, eps2, &mut out_par, bh, true);

        let tol = 1.0e-12_f64;
        for (a, b) in out_ser.iter().zip(out_par.iter()) {
            assert!(
                (a.x - b.x).abs() <= tol && (a.y - b.y).abs() <= tol && (a.z - b.z).abs() <= tol,
                "serial vs rayon local tree mismatch: {a:?} vs {b:?}"
            );
        }
    }

    #[test]
    fn local_bh_use_rayon_respects_deterministic_and_solver() {
        let cfg: RunConfig = toml::from_str(
            r#"
[simulation]
dt = 0.01
num_steps = 1
softening = 0.05
particle_count = 8
box_size = 2.0
seed = 0

[initial_conditions]
kind = "lattice"

[gravity]
solver = "barnes_hut"

[performance]
deterministic = false
"#,
        )
        .expect("minimal RunConfig TOML");

        assert!(
            local_bh_use_rayon(&cfg),
            "BarnesHut + deterministic=false should enable local rayon walk"
        );

        let mut cfg_det = cfg.clone();
        cfg_det.performance.deterministic = true;
        assert!(
            !local_bh_use_rayon(&cfg_det),
            "deterministic=true should disable local rayon walk"
        );

        let mut cfg_pm = cfg.clone();
        cfg_pm.gravity.solver = SolverKind::Pm;
        assert!(
            !local_bh_use_rayon(&cfg_pm),
            "PM solver should not enable Barnes-Hut local rayon walk"
        );
    }

    /// Coherencia directa con `Octree::walk_accel_multipole` en un punto (smoke).
    #[test]
    fn walk_accel_local_matches_octree_for_one_particle() {
        let parts = lattice_16();
        let all_pos: Vec<Vec3> = parts.iter().map(|p| p.position).collect();
        let all_mass: Vec<f64> = parts.iter().map(|p| p.mass).collect();
        let tree = gadget_ng_tree::Octree::build(&all_pos, &all_mass);
        let bh = sample_bh_params();
        let g = 1.0;
        let eps2 = 0.01;
        let gi = 3usize;
        let a = walk_accel_local(
            &tree,
            parts[gi].position,
            gi,
            g,
            eps2,
            &all_pos,
            &all_mass,
            bh,
        );
        let b = tree.walk_accel_multipole(
            parts[gi].position,
            gi,
            g,
            eps2,
            bh.theta,
            &all_pos,
            &all_mass,
            bh.multipole_order,
            bh.use_relative_criterion,
            bh.err_tol_force_acc,
            bh.softened_multipoles,
            bh.mac_softening,
        );
        let tol = 1.0e-15_f64;
        assert!((a.x - b.x).abs() < tol);
        assert!((a.y - b.y).abs() < tol);
        assert!((a.z - b.z).abs() < tol);
    }
}
