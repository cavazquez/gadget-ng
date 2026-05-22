//! Smokes de integración que llaman `gadget_ng_cli::run_stepping` directamente
//! (sin `std::process::Command`). Esto hace que tarpaulin instrumente todas las
//! rutas de código del bucle de integración.
//!
//! Cada smoke usa 2 pasos y pocas partículas para ser rápido en CI.

use gadget_ng_cli::run_stepping;
use gadget_ng_parallel::SerialRuntime;

fn run_toml(toml_str: &str) {
    let dir = tempfile::tempdir().expect("tempdir");
    let cfg: gadget_ng_core::RunConfig = toml::from_str(toml_str).expect("toml parse");
    let rt = SerialRuntime;
    run_stepping(&rt, &cfg, dir.path(), false, None).expect("run_stepping ok");
}

fn assert_insitu_files_exist(dir: &std::path::Path, steps: &[u64]) {
    for step in steps {
        let path = dir.join(format!("insitu_{step:06}.json"));
        assert!(path.exists(), "falta archivo in-situ {path:?}");
        let text = std::fs::read_to_string(&path).expect("read insitu json");
        let json: serde_json::Value = serde_json::from_str(&text).expect("parse insitu json");
        assert!(
            json.get("power_spectrum").is_some(),
            "insitu debe incluir P(k)"
        );
    }
}

/// Configuración base mínima: Direct gravity, 2 pasos, 8 partículas retícula.
fn base_toml(extra: &str) -> String {
    format!(
        r#"
[simulation]
particle_count = 8
box_size       = 1.0
dt             = 0.01
num_steps      = 2
softening      = 0.05
seed           = 1

[initial_conditions]
kind = "lattice"

[output]
checkpoint_interval = 0
snapshot_interval   = 0

{extra}
"#
    )
}

// ── Plummer (Barnes–Hut) ──────────────────────────────────────────────────────

#[test]
fn smoke_stepping_plummer_bh() {
    let dir = tempfile::tempdir().expect("tempdir");
    let cfg: gadget_ng_core::RunConfig = toml::from_str(
        r#"
[simulation]
particle_count = 32
box_size       = 20.0
dt             = 0.01
num_steps      = 2
softening      = 0.1
seed           = 42

[initial_conditions]
kind = { plummer = { a = 1.0 } }

[gravity]
solver = "barnes_hut"
theta  = 0.5

[output]
checkpoint_interval = 0
snapshot_interval   = 0
"#,
    )
    .expect("toml parse");
    let rt = SerialRuntime;
    run_stepping(&rt, &cfg, dir.path(), false, None).expect("plummer bh smoke ok");
}

// ── PM solver ────────────────────────────────────────────────────────────────

#[test]
fn smoke_stepping_pm_lattice() {
    run_toml(&base_toml(
        r#"
[gravity]
solver       = "pm"
pm_grid_size = 8
"#,
    ));
}

// ── TreePM solver ─────────────────────────────────────────────────────────────

#[test]
fn smoke_stepping_treepm_lattice() {
    run_toml(&base_toml(
        r#"
[gravity]
solver       = "tree_pm"
pm_grid_size = 8
"#,
    ));
}

// ── Direct gravity (default) ─────────────────────────────────────────────────

#[test]
fn smoke_stepping_direct_lattice() {
    run_toml(&base_toml(""));
}

// ── Cosmología habilitada ─────────────────────────────────────────────────────

#[test]
fn smoke_stepping_cosmo_pm() {
    run_toml(&base_toml(
        r#"
[gravity]
solver       = "pm"
pm_grid_size = 8

[cosmology]
enabled      = true
omega_m      = 0.3
omega_lambda = 0.7
h0           = 0.7
a_init       = 0.02
a_final      = 0.03
"#,
    ));
}

// ── Checkpoint round-trip (save + resume) ────────────────────────────────────

#[test]
fn smoke_stepping_checkpoint_resume() {
    let dir = tempfile::tempdir().expect("tempdir");
    let out1 = dir.path().join("run1");
    let cfg: gadget_ng_core::RunConfig = toml::from_str(
        r#"
[simulation]
particle_count = 8
box_size       = 1.0
dt             = 0.01
num_steps      = 2
softening      = 0.05
seed           = 7

[initial_conditions]
kind = "lattice"

[output]
checkpoint_interval = 1
snapshot_interval   = 0
"#,
    )
    .expect("toml parse");
    let rt = SerialRuntime;
    run_stepping(&rt, &cfg, &out1, false, None).expect("first run ok");

    // Verificar que el checkpoint fue creado.
    assert!(
        out1.join("checkpoint").exists(),
        "checkpoint dir must exist after first run"
    );

    // Reanudar desde el checkpoint.
    let out2 = dir.path().join("run2");
    run_stepping(&rt, &cfg, &out2, false, Some(&out1)).expect("resume ok");
}

// ── Snapshot final escrito ────────────────────────────────────────────────────

#[test]
fn smoke_stepping_writes_final_snapshot() {
    let dir = tempfile::tempdir().expect("tempdir");
    let cfg: gadget_ng_core::RunConfig = toml::from_str(&base_toml("")).expect("toml parse");
    let rt = SerialRuntime;
    run_stepping(&rt, &cfg, dir.path(), true, None).expect("ok");
    assert!(
        dir.path().join("snapshot_final").exists(),
        "snapshot_final must exist when write_final_snapshot=true"
    );
}

// ── SPH mínimo (gas, 1 paso) ──────────────────────────────────────────────────

#[test]
fn smoke_stepping_sph_minimal() {
    // SPH requiere particle_count = n³ para la retícula. Usamos 8 (2³), gas_fraction = 1.
    let dir = tempfile::tempdir().expect("tempdir");
    let cfg: gadget_ng_core::RunConfig = toml::from_str(
        r#"
[simulation]
particle_count = 8
box_size       = 1.0
dt             = 0.001
num_steps      = 1
softening      = 0.1
seed           = 5

[initial_conditions]
kind = "lattice"

[output]
checkpoint_interval = 0
snapshot_interval   = 0

[sph]
enabled      = true
gas_fraction = 1.0
"#,
    )
    .expect("toml parse");
    let rt = SerialRuntime;
    run_stepping(&rt, &cfg, dir.path(), false, None).expect("sph smoke ok");
}

// ── MHD mínimo (gas + campo B uniforme) ─────────────────────────────────────

#[test]
fn smoke_stepping_mhd_minimal() {
    run_toml(&base_toml(
        r#"
[sph]
enabled      = true
gas_fraction = 1.0

[mhd]
enabled = true
b0_kind = "uniform"
b0_uniform = [0.0, 0.0, 0.01]
"#,
    ));
}

// ── Integrador jerárquico (block timesteps) ───────────────────────────────────

#[test]
fn smoke_stepping_hierarchical_leapfrog() {
    run_toml(&base_toml(
        r#"
[timestep]
hierarchical = true
eta = 0.05
max_level = 2
"#,
    ));
}

// ── run_snapshot vía stepping + snapshot final ────────────────────────────────

#[test]
fn smoke_run_snapshot_via_stepping() {
    let dir = tempfile::tempdir().expect("tempdir");
    let cfg: gadget_ng_core::RunConfig = toml::from_str(
        r#"
[simulation]
particle_count = 8
box_size       = 1.0
dt             = 0.01
num_steps      = 1
softening      = 0.05
seed           = 11

[initial_conditions]
kind = "lattice"

[output]
checkpoint_interval = 0
snapshot_interval   = 0
"#,
    )
    .expect("toml parse");
    let rt = SerialRuntime;
    run_stepping(&rt, &cfg, dir.path(), true, None).expect("stepping with snapshot ok");
    assert!(dir.path().join("snapshot_final").exists());
}

// ── Yoshida-4 integrator ──────────────────────────────────────────────────────

#[test]
fn smoke_stepping_yoshida4() {
    // `integrator` es un campo de `[simulation]`, no una sección anidada.
    let dir = tempfile::tempdir().expect("tempdir");
    let cfg: gadget_ng_core::RunConfig = toml::from_str(
        r#"
[simulation]
particle_count = 8
box_size       = 1.0
dt             = 0.01
num_steps      = 2
softening      = 0.05
seed           = 1
integrator     = "yoshida4"

[initial_conditions]
kind = "lattice"

[output]
checkpoint_interval = 0
snapshot_interval   = 0
"#,
    )
    .expect("toml parse");
    let rt = SerialRuntime;
    run_stepping(&rt, &cfg, dir.path(), false, None).expect("yoshida4 smoke ok");
}

// ── Snapshots intermedios (snapshot_interval > 0) ────────────────────────────

#[test]
fn smoke_stepping_snapshot_interval_writes_frames() {
    let dir = tempfile::tempdir().expect("tempdir");
    let cfg: gadget_ng_core::RunConfig = toml::from_str(
        r#"
[simulation]
particle_count = 8
box_size       = 1.0
dt             = 0.01
num_steps      = 2
softening      = 0.05
seed           = 13

[initial_conditions]
kind = "lattice"

[output]
checkpoint_interval = 0
snapshot_interval   = 1
"#,
    )
    .expect("toml parse");
    let rt = SerialRuntime;
    run_stepping(&rt, &cfg, dir.path(), false, None).expect("snapshot interval smoke ok");
    assert!(
        dir.path().join("frames").join("snap_000001").exists(),
        "snap_000001 must exist when snapshot_interval=1"
    );
    assert!(
        dir.path().join("frames").join("snap_000002").exists(),
        "snap_000002 must exist when snapshot_interval=1"
    );
    assert!(
        dir.path()
            .join("frames")
            .join("snap_000001")
            .join("particles.jsonl")
            .exists()
    );
}

// ── Resume end-to-end (pasos restantes tras checkpoint) ───────────────────────

#[test]
fn smoke_stepping_resume_completes_remaining_steps() {
    use std::fs;

    let dir = tempfile::tempdir().expect("tempdir");
    let out1 = dir.path().join("run1");
    let cfg_partial: gadget_ng_core::RunConfig = toml::from_str(
        r#"
[simulation]
particle_count = 8
box_size       = 1.0
dt             = 0.01
num_steps      = 2
softening      = 0.05
seed           = 17

[initial_conditions]
kind = "lattice"

[output]
checkpoint_interval = 1
snapshot_interval   = 0
"#,
    )
    .expect("toml parse");
    let rt = SerialRuntime;
    run_stepping(&rt, &cfg_partial, &out1, false, None).expect("partial run ok");

    let ck_meta_path = out1.join("checkpoint").join("checkpoint.json");
    let partial_meta: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&ck_meta_path).expect("read checkpoint"))
            .expect("parse checkpoint json");
    assert_eq!(
        partial_meta["completed_step"].as_u64(),
        Some(2),
        "primer tramo debe completar 2 pasos"
    );

    let mut cfg_full = cfg_partial;
    cfg_full.simulation.num_steps = 4;
    let out2 = dir.path().join("run2");
    run_stepping(&rt, &cfg_full, &out2, false, Some(&out1)).expect("resume run ok");

    let final_meta: serde_json::Value = serde_json::from_str(
        &fs::read_to_string(out2.join("checkpoint").join("checkpoint.json"))
            .expect("read final ck"),
    )
    .expect("parse final checkpoint");
    assert_eq!(
        final_meta["completed_step"].as_u64(),
        Some(4),
        "resume debe completar hasta num_steps=4"
    );
}

// ── RT + reionización mínima (SPH + M1) ───────────────────────────────────────

#[test]
fn smoke_stepping_rt_reionization_minimal() {
    run_toml(&base_toml(
        r#"
[sph]
enabled      = true
gas_fraction = 1.0

[rt]
enabled    = true
rt_mesh    = 4
substeps   = 1

[reionization]
enabled       = true
n_sources     = 2
uv_luminosity = 0.1
z_start       = 1.0
z_end         = 0.0
"#,
    ));
}

// ── SIDM mínimo ───────────────────────────────────────────────────────────────

#[test]
fn smoke_stepping_sidm_minimal() {
    run_toml(&base_toml(
        r#"
[sidm]
enabled  = true
sigma_m  = 1.0e-5
v_max    = 1.0e6
"#,
    ));
}

// ── In-situ analysis (Phase 63+) ──────────────────────────────────────────────

#[test]
fn smoke_stepping_insitu_basic() {
    let dir = tempfile::tempdir().expect("tempdir");
    let cfg: gadget_ng_core::RunConfig = toml::from_str(&base_toml(
        r#"
[insitu_analysis]
enabled      = true
interval     = 1
pk_mesh      = 8
fof_min_part = 4
xi_bins      = 4
"#,
    ))
    .expect("toml parse");
    let rt = SerialRuntime;
    run_stepping(&rt, &cfg, dir.path(), false, None).expect("insitu basic ok");
    assert_insitu_files_exist(dir.path(), &[1, 2]);
}

#[test]
fn smoke_stepping_insitu_extended_flags() {
    let dir = tempfile::tempdir().expect("tempdir");
    let cfg: gadget_ng_core::RunConfig = toml::from_str(&base_toml(
        r#"
[sph]
enabled      = true
gas_fraction = 1.0

[rt]
enabled  = true
rt_mesh  = 4
substeps = 1

[reionization]
enabled       = true
n_sources     = 2
uv_luminosity = 0.1
z_start       = 1.0
z_end         = 0.0

[insitu_analysis]
enabled            = true
interval           = 1
pk_mesh            = 8
fof_min_part       = 4
xi_bins            = 4
bispectrum_bins    = 2
igm_temp_enabled   = true
cm21_enabled       = true
sz_enabled         = true
sz_n_pixels        = 8
lya_enabled        = true
lya_n_sightlines   = 8
wl_enabled         = true
wl_n_pixels        = 8
wl_fov_rad         = 0.08
"#,
    ))
    .expect("toml parse");
    let rt = SerialRuntime;
    run_stepping(&rt, &cfg, dir.path(), false, None).expect("insitu extended ok");
    let path = dir.path().join("insitu_000001.json");
    assert!(path.exists());
    let json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).expect("read insitu")).expect("parse");
    assert!(json.get("sz_compton_y").is_some(), "sz debe estar presente");
    assert!(json.get("lya").is_some(), "lya debe estar presente");
    assert!(json.get("wl").is_some(), "wl debe estar presente");
    assert!(
        json.get("bk_equilateral")
            .and_then(|v| v.as_array())
            .is_some_and(|a| !a.is_empty()),
        "bispectrum debe tener bins"
    );
}

#[test]
fn smoke_stepping_modified_gravity_fr() {
    run_toml(&base_toml(
        r#"
[modified_gravity]
enabled = true
f_r0    = 1.0e-4
n       = 1.0
"#,
    ));
}

#[test]
fn smoke_stepping_agn_with_insitu() {
    run_toml(&base_toml(
        r#"
[sph]
enabled      = true
gas_fraction = 0.5

[sph.agn]
enabled = true
n_agn_bh = 1

[insitu_analysis]
enabled      = true
interval     = 1
pk_mesh      = 8
fof_min_part = 4
"#,
    ));
}

#[test]
fn smoke_stepping_dark_matter_wdm_cosmo() {
    run_toml(&base_toml(
        r#"
[gravity]
solver       = "pm"
pm_grid_size = 8

[cosmology]
enabled      = true
omega_m      = 0.3
omega_lambda = 0.7
h0           = 0.7
a_init       = 0.02
a_final      = 0.03

[dark_matter]
enabled   = true
model     = "warm"
m_wdm_kev = 3.0
"#,
    ));
}
