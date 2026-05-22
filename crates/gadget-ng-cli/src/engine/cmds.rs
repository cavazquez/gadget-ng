//! CLI entry points other than the main stepping loop.

use crate::config_load;
use crate::error::CliError;
use gadget_ng_core::{RunConfig, build_particles_for_gid_range};
use gadget_ng_io::write_snapshot_formatted;
use gadget_ng_parallel::{ParallelRuntime, gid_block_range};
use std::fs;
use std::path::Path;

use super::provenance::{provenance_for_run, snapshot_env_for};

pub fn cmd_config_print(cfg_path: &Path) -> Result<(), CliError> {
    let cfg = config_load::load_run_config(cfg_path)?;
    config_load::print_resolved_config(&cfg)?;
    let hash = config_load::config_canonical_hash(&cfg)?;
    println!("canonical_toml_sha256={hash}");
    Ok(())
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
        let env = snapshot_env_for(cfg, 0.0, 0.0);
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

    if let Some(parent) = out_png.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent).map_err(|e| CliError::io(parent, e))?;
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

// ── Analyse (legacy — reemplazado por Analyze) ─────────────────────────────────
// `run_analyse` eliminado; usar `analyze_cmd::run_analyze` en su lugar.

#[cfg(test)]
mod tests {
    use super::*;
    use gadget_ng_core::{Particle, RunConfig, Vec3};
    use gadget_ng_io::{Provenance, write_snapshot};
    use gadget_ng_parallel::SerialRuntime;
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

    fn minimal_lattice_cfg() -> RunConfig {
        toml::from_str(
            r#"
[simulation]
particle_count = 8
box_size = 1.0
dt = 0.01
num_steps = 1
softening = 0.05
seed = 3

[initial_conditions]
kind = "lattice"
"#,
        )
        .expect("toml parse")
    }

    #[test]
    fn cmd_config_print_emits_canonical_hash() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cfg_path = dir.path().join("cfg.toml");
        fs::write(
            &cfg_path,
            r#"
[simulation]
particle_count = 8
box_size = 1.0
dt = 0.01
num_steps = 1
softening = 0.05
seed = 1

[initial_conditions]
kind = "lattice"
"#,
        )
        .expect("write cfg");
        cmd_config_print(&cfg_path).expect("cmd_config_print ok");
    }

    #[test]
    fn run_snapshot_writes_jsonl() {
        let dir = tempfile::tempdir().expect("tempdir");
        let out = dir.path().join("snap");
        let cfg = minimal_lattice_cfg();
        let rt = SerialRuntime;
        run_snapshot(&rt, &cfg, &out).expect("run_snapshot ok");
        assert!(out.join("particles.jsonl").exists());
    }

    #[test]
    fn run_visualize_velocity_and_white_projections() {
        let dir = tempfile::tempdir().expect("tempdir");
        let snap = dir.path().join("snap");
        fs::create_dir_all(&snap).expect("mkdir");
        write_lattice_snapshot(&snap, 8);

        let png_xy = dir.path().join("xy.png");
        run_visualize(&snap, &png_xy, 64, 64, "xy", "velocity").expect("xy velocity");
        assert!(png_xy.metadata().expect("meta").len() > 0);

        for (proj, name) in [("xz", "xz.png"), ("yz", "yz.png")] {
            let png = dir.path().join(name);
            run_visualize(&snap, &png, 48, 48, proj, "white").expect("projection");
            assert!(png.exists());
        }
    }

    #[test]
    fn run_visualize_empty_snapshot_is_ok() {
        let dir = tempfile::tempdir().expect("tempdir");
        let snap = dir.path().join("snap");
        fs::create_dir_all(&snap).expect("mkdir");
        let prov = Provenance::new("test", None, "debug", vec![], vec![], "test");
        write_snapshot(&snap, &[], &prov).expect("empty snapshot");
        let png = dir.path().join("empty.png");
        run_visualize(&snap, &png, 32, 32, "xy", "velocity").expect("empty ok");
        assert!(!png.exists());
    }
}
