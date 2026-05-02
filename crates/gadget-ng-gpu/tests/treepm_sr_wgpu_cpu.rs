//! Corto alcance TreePM en wgpu vs CPU (`short_range_accels`).

use gadget_ng_core::Vec3;
use gadget_ng_gpu::GpuTreePmShortRange;
use gadget_ng_tree::Octree;
use gadget_ng_treepm::short_range::{ShortRangeParams, short_range_accels};

#[test]
fn treepm_short_gpu_matches_cpu() {
    let Some(gpu) = GpuTreePmShortRange::try_new() else {
        eprintln!("[SKIP] treepm_short_gpu_matches_cpu: sin wgpu");
        return;
    };

    let n = 28usize;
    let positions: Vec<Vec3> = (0..n)
        .map(|i| {
            let t = i as f64 * 0.17;
            Vec3::new(t.sin() * 3.0, t.cos(), (t * 0.5).sin() * 2.0)
        })
        .collect();
    let masses: Vec<f64> = (0..n).map(|i| 0.5 + (i as f64 * 0.02)).collect();

    let eps2 = 0.015_f64;
    let g = 1.0_f64;
    let r_split = 0.08_f64;
    let r_cut = 5.0 * r_split;
    let r_cut2 = r_cut * r_cut;

    let sr = ShortRangeParams {
        positions: &positions,
        masses: &masses,
        eps2,
        g,
        r_split,
        r_cut2,
    };

    let idx: Vec<usize> = (0..n).collect();
    let mut out_cpu = vec![Vec3::zero(); n];
    short_range_accels(&sr, &idx, &mut out_cpu);

    let tree = Octree::build(&positions, &masses);
    let nodes = tree.export_bh_monopole_gpu_nodes();
    let root = tree.root;

    let flat_pos: Vec<f32> = positions
        .iter()
        .flat_map(|p| [p.x as f32, p.y as f32, p.z as f32])
        .collect();
    let flat_mass: Vec<f32> = masses.iter().map(|&m| m as f32).collect();
    let query: Vec<u32> = (0..n as u32).collect();

    let gpu_raw = gpu.compute_accelerations_raw(
        &flat_pos,
        &flat_mass,
        &nodes,
        root,
        &query,
        eps2 as f32,
        g as f32,
        r_split as f32,
        r_cut2 as f32,
    );

    let mut max_rel = 0.0_f64;
    for i in 0..n {
        let gx = gpu_raw[3 * i] as f64;
        let gy = gpu_raw[3 * i + 1] as f64;
        let gz = gpu_raw[3 * i + 2] as f64;
        let d = Vec3::new(gx - out_cpu[i].x, gy - out_cpu[i].y, gz - out_cpu[i].z);
        let err = d.norm();
        let scale = out_cpu[i].norm().max(1e-12);
        max_rel = max_rel.max(err / scale);
    }

    println!("TreePM SR wgpu vs CPU: max_rel_err = {max_rel:.4e}");
    assert!(
        max_rel < 3e-3,
        "TreePM short GPU vs CPU max_rel {max_rel:.4e}"
    );
}
