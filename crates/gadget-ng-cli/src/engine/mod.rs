//! Simulation engine: integration loop, gravity kernels, checkpoints, and CLI helpers.

mod checkpoint;
mod cmds;
mod diagnostics;
mod gravity;
mod provenance;
mod stepping;
mod timings;
#[cfg(all(feature = "gpu", feature = "cuda"))]
mod treepm_gpu_hybrid;

pub use cmds::{cmd_config_print, run_snapshot, run_visualize};
pub use stepping::run_stepping;

/// Renderiza el snapshot final a PNG/PPM si está disponible.
pub fn render_snapshot_visualization(
    out: &std::path::Path,
    _vis_snapshot: u64,
    vis_proj: &str,
    vis_mode: &str,
    vis_format: &str,
) {
    let snap_dir = out.join("snapshot_final");
    if !snap_dir.exists() {
        return;
    }
    use gadget_ng_core::SnapshotFormat;
    use gadget_ng_vis::Projection;
    let data = gadget_ng_io::read_snapshot_formatted(SnapshotFormat::Jsonl, &snap_dir);
    let Ok(data) = data else { return };
    let positions: Vec<gadget_ng_core::Vec3> = data.particles.iter().map(|p| p.position).collect();
    let proj = match vis_proj.to_lowercase().as_str() {
        "xz" => Projection::XZ,
        "yz" => Projection::YZ,
        _ => Projection::XY,
    };
    let pixels = match vis_mode.to_lowercase().as_str() {
        "density" => gadget_ng_vis::render_density_ppm(&positions, data.box_size, 1024, 1024, proj),
        _ => gadget_ng_vis::render_ppm_projection(&positions, data.box_size, 1024, 1024, proj),
    };
    let ext = if vis_format.to_lowercase() == "png" {
        "png"
    } else {
        "ppm"
    };
    let out_path = out.join(format!("snapshot_final.{ext}"));
    let result = if ext == "png" {
        gadget_ng_vis::write_png(&out_path, &pixels, 1024, 1024)
    } else {
        gadget_ng_vis::write_ppm(&out_path, &pixels, 1024, 1024)
    };
    match result {
        Ok(()) => eprintln!("[vis] imagen escrita en {:?}", out_path),
        Err(e) => eprintln!("[vis] Error escribiendo imagen: {e}"),
    }
}

#[cfg(test)]
mod vis_tests {
    use super::render_snapshot_visualization;
    use gadget_ng_core::{Particle, Vec3};
    use gadget_ng_io::{Provenance, write_snapshot};
    use std::fs;
    use std::path::Path;

    fn write_lattice_snapshot(dir: &Path, n: usize) {
        let side = (n as f64).cbrt().round() as usize;
        let particles: Vec<Particle> = (0..n)
            .map(|i| {
                let ix = i % side;
                let iy = (i / side) % side;
                let iz = i / (side * side);
                Particle::new(
                    i,
                    1.0 / n as f64,
                    Vec3::new(
                        (ix as f64 + 0.5) / side as f64,
                        (iy as f64 + 0.5) / side as f64,
                        (iz as f64 + 0.5) / side as f64,
                    ),
                    Vec3::zero(),
                )
            })
            .collect();
        let prov = Provenance::new("test", None, "debug", vec![], vec![], "test");
        write_snapshot(dir, &particles, &prov).expect("write_snapshot");
    }

    #[test]
    fn render_snapshot_visualization_exports_ppm_and_png() {
        let dir = tempfile::tempdir().expect("tempdir");
        let snap = dir.path().join("snapshot_final");
        fs::create_dir_all(&snap).expect("mkdir");
        write_lattice_snapshot(&snap, 8);
        render_snapshot_visualization(dir.path(), 0, "xz", "density", "ppm");
        render_snapshot_visualization(dir.path(), 0, "xy", "points", "png");
        assert!(dir.path().join("snapshot_final.ppm").exists());
        assert!(dir.path().join("snapshot_final.png").exists());
    }

    #[test]
    fn render_snapshot_visualization_skips_missing_dir() {
        let dir = tempfile::tempdir().expect("tempdir");
        render_snapshot_visualization(dir.path(), 0, "xy", "points", "ppm");
        assert!(!dir.path().join("snapshot_final.ppm").exists());
    }
}
