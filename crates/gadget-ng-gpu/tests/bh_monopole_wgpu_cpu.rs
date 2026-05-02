//! Barnes–Hut monopolo en wgpu vs `Octree::walk_accel_multipole` (orden 1, MAC geométrico).

use gadget_ng_core::{MacSoftening, Vec3};
use gadget_ng_gpu::GpuBarnesHutMonopole;
use gadget_ng_tree::Octree;

/// Compara el kernel BH monopolo (árbol en CPU, walk en GPU) con el recorrido CPU
/// en modo monopolar explícito (`multipole_order = 1`, sin MAC relativo).
#[test]
fn bh_monopole_gpu_matches_cpu_order1() {
    let Some(gpu) = GpuBarnesHutMonopole::try_new() else {
        eprintln!("[SKIP] bh_monopole_gpu_matches_cpu_order1: sin adaptador wgpu");
        return;
    };

    let n = 32_usize;
    let positions: Vec<Vec3> = (0..n)
        .map(|i| {
            let t = i as f64 * 0.23;
            Vec3::new(t.sin() * 2.0, t.cos() * 1.2, (t * 0.4).sin() * 0.8)
        })
        .collect();
    let masses: Vec<f64> = (0..n).map(|i| 0.2 + 0.05 * (i as f64 % 5.0)).collect();

    let tree = Octree::build(&positions, &masses);
    let nodes = tree.export_bh_monopole_gpu_nodes();
    let root = tree.root;

    let flat_pos: Vec<f32> = positions
        .iter()
        .flat_map(|p| [p.x as f32, p.y as f32, p.z as f32])
        .collect();
    let flat_mass: Vec<f32> = masses.iter().map(|&m| m as f32).collect();
    let query: Vec<u32> = (0..n as u32).collect();

    let eps2 = 0.01_f32;
    let g = 1.0_f32;
    let theta = 0.45_f32;

    let gpu_acc =
        gpu.compute_accelerations_raw(&flat_pos, &flat_mass, &nodes, root, &query, eps2, g, theta);

    let eps2_d = f64::from(eps2);
    let g_d = f64::from(g);
    let theta_d = f64::from(theta);

    let mut max_rel = 0.0_f32;
    for i in 0..n {
        let a_cpu = tree.walk_accel_multipole(
            positions[i],
            i,
            g_d,
            eps2_d,
            theta_d,
            &positions,
            &masses,
            1,
            false,
            0.0,
            false,
            MacSoftening::Bare,
        );
        let cpu = [a_cpu.x as f32, a_cpu.y as f32, a_cpu.z as f32];
        for c in 0..3 {
            let gu = gpu_acc[3 * i + c];
            let diff = (gu - cpu[c]).abs();
            let mag = cpu[c].abs().max(1e-10);
            let rel = diff / mag;
            if rel > max_rel {
                max_rel = rel;
            }
        }
    }

    println!("BH monopolo wgpu vs CPU walk (order=1): max_rel_err = {max_rel:.4e}");
    assert!(
        max_rel < 5e-3,
        "BH monopolo GPU vs CPU: max_rel_err = {max_rel:.4e} (esperado < 5e-3)"
    );
}
