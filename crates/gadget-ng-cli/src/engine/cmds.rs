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
