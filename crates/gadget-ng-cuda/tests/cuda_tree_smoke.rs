use gadget_ng_core::{Particle, Vec3};
use gadget_ng_cuda::CudaTreeSolver;
use gadget_ng_tree::{RemoteMultipoleNode, RmnSoa};

fn cuda_solver_or_skip() -> Option<CudaTreeSolver> {
    match CudaTreeSolver::try_new_checked() {
        Ok(solver) => Some(solver),
        Err(e) => {
            eprintln!("SKIP CudaTreeSolver: {e}");
            None
        }
    }
}

/// Crea nodos LET sintéticos con monopolo + cuadrupolo pequeño, octupolo cero.
fn synthetic_let_nodes(n: usize, scale: f64) -> RmnSoa {
    let mut rmns = Vec::with_capacity(n);
    for i in 0..n {
        let angle = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
        let r = scale;
        rmns.push(RemoteMultipoleNode {
            com: Vec3::new(r * angle.cos(), r * angle.sin(), 0.0),
            mass: 1.0,
            // Pequeño tensor cuadrupolar
            quad: [0.01, 0.005, 0.0, 0.01, 0.0, -0.02],
            oct: [0.0; 7],
            hex: [0.0; 15],
            half_size: scale * 0.1,
        });
    }
    RmnSoa::from_slice(&rmns)
}

#[test]
#[ignore = "Requiere hardware CUDA; ejecutar con `-- --ignored`"]
fn cuda_tree_let_accel_matches_cpu() {
    let Some(cuda) = cuda_solver_or_skip() else {
        return;
    };

    const N_PART: usize = 512;
    const N_NODES: usize = 256;
    const G: f64 = 1.0;
    const EPS2: f64 = 1.0e-4;

    // Partículas en esfera uniforme
    let particles: Vec<Particle> = (0..N_PART)
        .map(|i| {
            let t = 2.0 * std::f64::consts::PI * i as f64 / N_PART as f64;
            let phi = std::f64::consts::PI * i as f64 / N_PART as f64;
            Particle::new(
                i,
                1.0 / N_PART as f64,
                Vec3::new(
                    0.3 * phi.sin() * t.cos(),
                    0.3 * phi.sin() * t.sin(),
                    0.3 * phi.cos(),
                ),
                Vec3::zero(),
            )
        })
        .collect();

    // Nodos LET en anillo exterior — separados para que los multipoles sean pequeños
    let nodes = synthetic_let_nodes(N_NODES, 3.0);

    // Referencia CPU (solo mono+quad+oct, sin hex porque hex=0 en los nodos sintéticos)
    let cpu_acc: Vec<Vec3> = particles
        .iter()
        .map(|p| nodes.accel(p.position, G, EPS2))
        .collect();

    // CUDA
    let gpu_acc = cuda
        .try_tree_walk_let(&particles, &nodes, G, EPS2)
        .expect("cuda_tree_let_accel debe completar sin error");

    assert_eq!(gpu_acc.len(), N_PART);

    // Comparar: f32 vs f64 → tolerancia ~1e-4 relativo
    let mut max_rel = 0.0_f64;
    for (i, (c, g)) in cpu_acc.iter().zip(gpu_acc.iter()).enumerate() {
        let mag = (c.x.powi(2) + c.y.powi(2) + c.z.powi(2)).sqrt();
        if mag < 1.0e-12 {
            continue;
        }
        let diff = ((g.x - c.x).powi(2) + (g.y - c.y).powi(2) + (g.z - c.z).powi(2)).sqrt();
        let rel = diff / mag;
        if rel > max_rel {
            max_rel = rel;
        }
        assert!(
            rel < 1.0e-3,
            "let_acc[{i}]: cpu=({:.4e},{:.4e},{:.4e}) gpu=({:.4e},{:.4e},{:.4e}) rel={:.3e}",
            c.x,
            c.y,
            c.z,
            g.x,
            g.y,
            g.z,
            rel
        );
    }
    eprintln!("cuda_tree_let_accel_matches_cpu: max_rel={max_rel:.3e} (tol=1e-3)");
}

#[test]
#[ignore = "Requiere hardware CUDA; ejecutar con `-- --ignored`"]
fn cuda_tree_walk_monopole_returns_finite_accelerations() {
    let Some(cuda) = cuda_solver_or_skip() else {
        return;
    };
    let particles = vec![
        Particle::new(0, 1.0, Vec3::new(-0.5, 0.0, 0.0), Vec3::zero()),
        Particle::new(1, 2.0, Vec3::new(0.5, 0.0, 0.0), Vec3::zero()),
    ];
    let acc = cuda
        .try_walk_monopole(&particles, 1.0, 1.0e-4)
        .expect("cuda tree walk should complete");
    assert_eq!(acc.len(), particles.len());
    assert!(
        acc.iter()
            .all(|a| a.x.is_finite() && a.y.is_finite() && a.z.is_finite())
    );
}

#[test]
fn cuda_tree_solver_returns_none_without_hardware() {
    let _ = CudaTreeSolver::try_new();
}

// ── AP-20: TreePM short-range GPU vs CPU ─────────────────────────────────────

#[test]
#[ignore = "requiere GPU NVIDIA (GTX 1060 / sm_61)"]
fn cuda_treepm_sr_match_cpu() {
    use gadget_ng_treepm::{ShortRangeParams, short_range_accels};

    let solver = match CudaTreeSolver::try_new_checked() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("SKIP: {e}");
            return;
        }
    };

    let n = 128_usize;
    let g = 1.0_f64;
    let r_split = 0.5_f64;
    let r_cut = 5.0 * r_split;
    let eps2 = 1.0e-4_f64;
    let box_size = 10.0_f64;

    // Partículas en distribución uniforme en el cubo
    let particles: Vec<Particle> = (0..n)
        .map(|k| {
            let t = k as f64 / n as f64;
            Particle::new(
                k,
                1.0,
                Vec3::new(
                    box_size * (0.1 + 0.8 * (t * 7.3).fract()),
                    box_size * (0.1 + 0.8 * (t * 3.7 + 0.5).fract()),
                    box_size * (0.1 + 0.8 * (t * 5.1 + 0.2).fract()),
                ),
                Vec3::zero(),
            )
        })
        .collect();

    let positions: Vec<Vec3> = particles.iter().map(|p| p.position).collect();
    let masses: Vec<f64> = particles.iter().map(|p| p.mass).collect();

    // CPU reference (O(N²) sin árbol: pasamos solo posiciones+masas, all-to-all)
    let indices: Vec<usize> = (0..n).collect();
    let cpu_sr_params = ShortRangeParams {
        positions: &positions,
        masses: &masses,
        eps2,
        g,
        r_split,
        r_cut2: r_cut * r_cut,
    };
    let mut acc_cpu = vec![Vec3::zero(); n];
    short_range_accels(&cpu_sr_params, &indices, &mut acc_cpu);

    // GPU
    let acc_gpu = solver
        .try_short_range(&positions, &masses, r_split, r_cut, eps2, g, Some(box_size))
        .expect("try_short_range falló en GPU");

    assert_eq!(acc_gpu.len(), n);

    // Magnitud relativa max
    let mut max_rel = 0.0_f64;
    let mut counted = 0usize;
    for i in 0..n {
        let mag_cpu = (acc_cpu[i].x * acc_cpu[i].x
            + acc_cpu[i].y * acc_cpu[i].y
            + acc_cpu[i].z * acc_cpu[i].z)
            .sqrt();
        if mag_cpu < 1.0e-12 {
            continue;
        }
        let dx = acc_gpu[i].x - acc_cpu[i].x;
        let dy = acc_gpu[i].y - acc_cpu[i].y;
        let dz = acc_gpu[i].z - acc_cpu[i].z;
        let diff = (dx * dx + dy * dy + dz * dz).sqrt();
        let rel = diff / mag_cpu;
        if rel > max_rel {
            max_rel = rel;
        }
        counted += 1;
    }
    assert!(counted > 0, "ninguna partícula tiene |a_cpu| > eps");
    assert!(
        max_rel < 0.05,
        "TreePM SR GPU vs CPU: max rel = {max_rel:.3} > 5% ({counted} partículas)"
    );
    eprintln!("[cuda_treepm_sr_match_cpu] OK — max_rel = {max_rel:.4} ({counted} partículas)");
}

// ── AP-20: Barnes-Hut GPU walk vs CPU ────────────────────────────────────────

#[test]
#[ignore = "requiere GPU NVIDIA (GTX 1060 / sm_61)"]
fn cuda_bh_walk_match_cpu() {
    use gadget_ng_core::MacSoftening;
    use gadget_ng_tree::Octree;

    let solver = match CudaTreeSolver::try_new_checked() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("SKIP: {e}");
            return;
        }
    };

    let n = 256_usize;
    let g = 1.0_f64;
    let theta = 0.5_f64;
    let eps2 = 1.0e-4_f64;

    // Partículas en distribución quasi-uniforme en [0,10]³
    let particles: Vec<Particle> = (0..n)
        .map(|k| {
            let t = k as f64 / n as f64;
            Particle::new(
                k,
                1.0,
                Vec3::new(
                    10.0 * (0.05 + 0.9 * (t * 7.3).fract()),
                    10.0 * (0.05 + 0.9 * (t * 3.7 + 0.5).fract()),
                    10.0 * (0.05 + 0.9 * (t * 5.1 + 0.2).fract()),
                ),
                Vec3::zero(),
            )
        })
        .collect();

    let positions: Vec<Vec3> = particles.iter().map(|p| p.position).collect();
    let masses: Vec<f64> = particles.iter().map(|p| p.mass).collect();

    // CPU reference: BH walk monopole (order=1)
    let tree = Octree::build(&positions, &masses);
    let acc_cpu: Vec<Vec3> = (0..n)
        .map(|i| {
            tree.walk_accel_multipole(
                positions[i],
                i,
                g,
                eps2,
                theta,
                &positions,
                &masses,
                1, // monopolo
                false,
                false,
                0.0,
                false,
                MacSoftening::Bare,
            )
        })
        .collect();

    // GPU BH walk
    let nodes = tree.export_bh_monopole_gpu_nodes();
    let target_idx: Vec<usize> = (0..n).collect();
    let acc_gpu = solver
        .try_bh_local_walk(&positions, &target_idx, &nodes, tree.root, theta, g, eps2)
        .expect("try_bh_local_walk falló en GPU");

    assert_eq!(acc_gpu.len(), n);

    // Magnitud relativa máxima
    let mut max_rel = 0.0_f64;
    let mut counted = 0usize;
    for i in 0..n {
        let mag = (acc_cpu[i].x * acc_cpu[i].x
            + acc_cpu[i].y * acc_cpu[i].y
            + acc_cpu[i].z * acc_cpu[i].z)
            .sqrt();
        if mag < 1.0e-12 {
            continue;
        }
        let dx = acc_gpu[i].x - acc_cpu[i].x;
        let dy = acc_gpu[i].y - acc_cpu[i].y;
        let dz = acc_gpu[i].z - acc_cpu[i].z;
        let diff = (dx * dx + dy * dy + dz * dz).sqrt();
        let rel = diff / mag;
        if rel > max_rel {
            max_rel = rel;
        }
        counted += 1;
    }
    assert!(counted > n / 2, "pocas partículas válidas para comparar");
    assert!(
        max_rel < 0.01,
        "BH GPU vs CPU: max rel = {max_rel:.4} > 1% ({counted} partículas)"
    );
    eprintln!("[cuda_bh_walk_match_cpu] OK — max_rel = {max_rel:.4} ({counted} partículas)");
}
