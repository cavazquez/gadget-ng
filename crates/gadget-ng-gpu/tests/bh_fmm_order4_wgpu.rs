//! Paridad orden multipolar 4: GPU vs [`gadget_ng_tree::Octree::walk_accel_multipole`].

use gadget_ng_core::{MacSoftening, Vec3};
use gadget_ng_gpu::{BhFmmKernelParams, GpuBarnesHutFmm};
use gadget_ng_tree::Octree;

fn skip_no_wgpu() -> bool {
    if GpuBarnesHutFmm::try_new().is_none() {
        eprintln!("[SKIP] bh_fmm_order4: sin wgpu");
        return true;
    }
    false
}

#[test]
fn bh_fmm_order4_gpu_vs_cpu_multipole() {
    if skip_no_wgpu() {
        return;
    }
    let n = 32usize;
    let positions: Vec<Vec3> = (0..n)
        .map(|i| {
            let t = i as f64 * 0.11;
            Vec3::new(t.sin() * 2.0, t.cos() * 1.5, (t * 0.9).sin())
        })
        .collect();
    let masses: Vec<f64> = (0..n).map(|i| 0.3 + (i as f64 * 0.03)).collect();

    let eps2 = 0.02_f64;
    let g = 1.0_f64;
    let theta = 0.45_f64;

    let tree = Octree::build(&positions, &masses);
    let gi = 7usize;
    let acc_cpu = tree.walk_accel_multipole(
        positions[gi],
        gi,
        g,
        eps2,
        theta,
        &positions,
        &masses,
        4,
        false,
        0.002,
        false,
        MacSoftening::Bare,
    );

    let gpu = GpuBarnesHutFmm::try_new().expect("wgpu");
    let nodes = tree.export_bh_fmm_gpu_nodes();
    let positions_f32: Vec<f32> = positions
        .iter()
        .flat_map(|p| [p.x as f32, p.y as f32, p.z as f32])
        .collect();
    let masses_f32: Vec<f32> = masses.iter().map(|&m| m as f32).collect();
    let query = vec![gi as u32];
    let kp = BhFmmKernelParams {
        eps2: eps2 as f32,
        g: g as f32,
        theta: theta as f32,
        err_tol: 0.002_f32,
        multipole_order: 4,
        use_relative_criterion: false,
        softened_multipoles: false,
        mac_softening: 0,
    };
    let raw =
        gpu.compute_accelerations_raw(&positions_f32, &masses_f32, &nodes, tree.root, &query, kp);
    let acc_gpu = Vec3::new(raw[0] as f64, raw[1] as f64, raw[2] as f64);

    let err = (acc_cpu - acc_gpu).norm();
    let scale = acc_cpu.norm().max(1e-12);
    let rel = err / scale;
    println!("BH order4 GPU vs CPU: rel_err = {rel:.4e}");
    assert!(
        rel < 5e-2,
        "orden 4 FMM GPU vs CPU: error relativo {rel:.3e} demasiado alto"
    );
}
