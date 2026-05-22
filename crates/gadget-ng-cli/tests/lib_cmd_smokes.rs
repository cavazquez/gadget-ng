//! Smokes de subcomandos vía API de librería (cobertura de analyze, fisher, mah, config, snapshot, visualize).

use gadget_ng_cli::{
    analyze_cmd::{AnalyzeParams, run_analyze},
    cmd_config_print,
    fisher_cmd::run_fisher,
    mah_cmd::run_mah,
    merge_tree_cmd::run_merge_tree,
    run_snapshot, run_visualize,
};
use gadget_ng_core::{Particle, Vec3};
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

fn write_gas_lattice_snapshot(dir: &Path, n: usize) {
    let side = (n as f64).cbrt().round() as usize;
    let particles: Vec<Particle> = (0..n)
        .map(|i| {
            let ix = i % side;
            let iy = (i / side) % side;
            let iz = i / (side * side);
            Particle::new_gas(
                i,
                1.0 / n as f64,
                Vec3::new(
                    (ix as f64 + 0.5) / side as f64,
                    (iy as f64 + 0.5) / side as f64,
                    (iz as f64 + 0.5) / side as f64,
                ),
                Vec3::zero(),
                100.0,
                0.05,
            )
        })
        .collect();
    let prov = Provenance::new("test", None, "debug", vec![], vec![], "test");
    write_snapshot(dir, &particles, &prov).expect("write_snapshot");
}

#[test]
fn smoke_cmd_config_print_valid_toml() {
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
fn smoke_run_snapshot_lattice() {
    let dir = tempfile::tempdir().expect("tempdir");
    let out = dir.path().join("snap");
    let cfg: gadget_ng_core::RunConfig = toml::from_str(
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
    .expect("toml parse");
    let rt = SerialRuntime;
    run_snapshot(&rt, &cfg, &out).expect("run_snapshot ok");
    assert!(out.join("particles.jsonl").exists());
}

#[test]
fn smoke_run_visualize_from_snapshot() {
    let dir = tempfile::tempdir().expect("tempdir");
    let snap = dir.path().join("snap");
    fs::create_dir_all(&snap).expect("mkdir");
    write_lattice_snapshot(&snap, 8);
    let png = dir.path().join("frame.png");
    run_visualize(&snap, &png, 64, 64, "xy", "velocity").expect("run_visualize ok");
    assert!(png.exists());
    assert!(png.metadata().expect("meta").len() > 0);
}

#[test]
fn smoke_run_visualize_projections_and_white() {
    let dir = tempfile::tempdir().expect("tempdir");
    let snap = dir.path().join("snap");
    fs::create_dir_all(&snap).expect("mkdir");
    write_lattice_snapshot(&snap, 8);
    for (proj, name) in [("xz", "xz.png"), ("yz", "yz.png")] {
        let png = dir.path().join(name);
        run_visualize(&snap, &png, 48, 48, proj, "white").expect("run_visualize ok");
        assert!(png.exists());
        assert!(png.metadata().expect("meta").len() > 0);
    }
}

#[test]
fn smoke_render_snapshot_visualization_exports() {
    use gadget_ng_cli::render_snapshot_visualization;

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
fn smoke_run_analyze_lattice() {
    let dir = tempfile::tempdir().expect("tempdir");
    let snap = dir.path().join("snap");
    fs::create_dir_all(&snap).expect("mkdir");
    write_lattice_snapshot(&snap, 8);
    let out = dir.path().join("results.json");
    let params = AnalyzeParams {
        snapshot_dir: &snap,
        out_path: &out,
        fof_b: 0.2,
        min_particles: 4,
        pk_mesh: 8,
        xi_bins: 4,
        nfw_min_part: 50,
        cosmology: None,
        box_size_mpc_h: Some(1.0),
        subfind: false,
        subfind_min_particles: 50,
        hdf5_catalog: false,
        cm21: false,
        igm_temp: false,
        agn_stats: false,
        eor_state: false,
        luminosity: false,
        xray: false,
        cuda_analysis: false,
    };
    run_analyze(&params).expect("run_analyze ok");
    assert!(out.exists());
    let text = fs::read_to_string(&out).expect("read results");
    assert!(text.contains("halos") || text.contains("power_spectrum"));
}

#[test]
fn smoke_run_analyze_extended_flags_gas() {
    let dir = tempfile::tempdir().expect("tempdir");
    let snap = dir.path().join("snap");
    fs::create_dir_all(&snap).expect("mkdir");
    write_gas_lattice_snapshot(&snap, 8);
    let out = dir.path().join("results.json");
    let params = AnalyzeParams {
        snapshot_dir: &snap,
        out_path: &out,
        fof_b: 0.2,
        min_particles: 4,
        pk_mesh: 8,
        xi_bins: 4,
        nfw_min_part: 50,
        cosmology: Some((0.315, 0.685, 3.0)),
        box_size_mpc_h: Some(1.0),
        subfind: false,
        subfind_min_particles: 50,
        hdf5_catalog: false,
        cm21: true,
        igm_temp: true,
        agn_stats: true,
        eor_state: true,
        luminosity: false,
        xray: true,
        cuda_analysis: false,
    };
    run_analyze(&params).expect("run_analyze extended ok");
    assert!(out.exists());
    let analyze_dir = dir.path().join("analyze");
    assert!(analyze_dir.join("cm21_output.json").exists());
    assert!(analyze_dir.join("igm_temp.json").exists());
    assert!(analyze_dir.join("agn_stats.json").exists());
    assert!(analyze_dir.join("eor_state.json").exists());
    assert!(analyze_dir.join("xray.json").exists());
}

#[test]
fn smoke_run_fisher_writes_json() {
    let dir = tempfile::tempdir().expect("tempdir");
    let out = dir.path().join("fisher.json");
    run_fisher(
        0.315, 0.049, 0.674, 0.965, 0.8111, -1.0, 0.0, 0.06, 0.01, 1.0e9, false, &out,
    )
    .expect("run_fisher ok");
    assert!(out.exists());
    let text = fs::read_to_string(&out).expect("read fisher");
    assert!(text.contains("fisher_matrix") || text.contains("param_names"));
}

#[test]
fn smoke_run_mah_from_minimal_forest() {
    let dir = tempfile::tempdir().expect("tempdir");
    let tree_path = dir.path().join("merger_tree.json");
    fs::write(
        &tree_path,
        r#"{
  "nodes": [
    {
      "snapshot": 0,
      "halo_id": 0,
      "mass_msun_h": 1.0e12,
      "n_particles": 64,
      "x_com": [0.5, 0.5, 0.5],
      "prog_main_id": 1,
      "merger_ids": [],
      "merger_mass_ratio": []
    },
    {
      "snapshot": 1,
      "halo_id": 1,
      "mass_msun_h": 2.0e12,
      "n_particles": 128,
      "x_com": [0.5, 0.5, 0.5],
      "prog_main_id": null,
      "merger_ids": [],
      "merger_mass_ratio": []
    }
  ],
  "roots": [1]
}"#,
    )
    .expect("write tree");
    let out = dir.path().join("mah.json");
    run_mah(&tree_path, &[1.0, 0.0], 1, 1.0, 0.0, &out).expect("run_mah ok");
    assert!(out.exists());
    let text = fs::read_to_string(&out).expect("read mah");
    assert!(text.contains("mah"));
    assert!(text.contains("mcbride_fit"));
}

#[test]
fn smoke_run_merge_tree_single_epoch() {
    let dir = tempfile::tempdir().expect("tempdir");
    let snap = dir.path().join("snap0");
    fs::create_dir_all(&snap).expect("mkdir snap");
    write_lattice_snapshot(&snap, 8);
    let catalog = dir.path().join("halos.jsonl");
    fs::write(
        &catalog,
        r#"{"halo_id":0,"n_particles":8,"mass":8.0,"x_com":0.5,"y_com":0.5,"z_com":0.5,"vx_com":0.0,"vy_com":0.0,"vz_com":0.0,"velocity_dispersion":0.0,"r_vir":0.2}
"#,
    )
    .expect("write catalog");
    let out = dir.path().join("forest.json");
    run_merge_tree(&[snap], &[catalog], &out, 0.1).expect("merge tree ok");
    assert!(out.exists());
}
