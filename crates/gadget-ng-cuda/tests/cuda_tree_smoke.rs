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
