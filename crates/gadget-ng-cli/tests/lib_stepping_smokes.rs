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
