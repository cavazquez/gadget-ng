//! Smoke: `gadget-ng stepping` con Plummer (2 pasos, pocas partículas).

use std::fs;
use std::process::Command;

#[test]
fn stepping_plummer_smoke_exits_ok() {
    let out = tempfile::tempdir().expect("tempdir");
    let manifest = env!("CARGO_MANIFEST_DIR");
    let base = format!("{manifest}/../../examples/plummer_sphere.toml");
    let cfg_src = fs::read_to_string(&base).expect("read plummer example");
    let cfg_fast = cfg_src
        .replace("particle_count = 512", "particle_count = 64")
        .replace("num_steps      = 200", "num_steps      = 2")
        .replace("checkpoint_interval = 100", "checkpoint_interval = 0")
        .replace("snapshot_interval   = 2", "snapshot_interval   = 0");
    let cfg_path = out.path().join("plummer_smoke.toml");
    fs::write(&cfg_path, cfg_fast).expect("write temp config");

    let exe = env!("CARGO_BIN_EXE_gadget-ng");
    let status = Command::new(exe)
        .args([
            "stepping",
            "--config",
            cfg_path.to_str().expect("utf8 config path"),
            "--out",
            out.path().join("run").to_str().expect("utf8 out path"),
        ])
        .status()
        .expect("spawn gadget-ng");
    assert!(status.success(), "plummer stepping smoke should succeed");
}
