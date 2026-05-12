//! Comprueba `--pk-model` en `gadget-ng fisher` y coherencia del JSON de salida.

use std::fs;
use std::process::Command;

#[test]
fn fisher_help_documents_pk_model() {
    let exe = env!("CARGO_BIN_EXE_gadget-ng");
    let out = Command::new(exe)
        .args(["fisher", "--help"])
        .output()
        .expect("spawn gadget-ng fisher --help");
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(
        text.contains("pk-model") && text.contains("linear") && text.contains("nonlinear"),
        "help should list --pk-model values; got:\n{text}"
    );
}

#[test]
fn fisher_rejects_invalid_pk_model() {
    let exe = env!("CARGO_BIN_EXE_gadget-ng");
    let out = Command::new(exe)
        .args([
            "fisher",
            "--pk-model",
            "not-a-mode",
            "--out",
            "/tmp/gadget_ng_fisher_should_not_exist.json",
        ])
        .output()
        .expect("spawn gadget-ng fisher");
    assert!(
        !out.status.success(),
        "invalid pk-model should fail; stdout: {}",
        String::from_utf8_lossy(&out.stdout)
    );
}

#[test]
fn fisher_linear_and_nonlinear_json_use_nonlinear_field() {
    let exe = env!("CARGO_BIN_EXE_gadget-ng");
    let dir = tempfile::tempdir().expect("tempdir");
    let linear_path = dir.path().join("fisher_linear.json");
    let nonlinear_path = dir.path().join("fisher_nonlinear.json");

    let status_linear = Command::new(exe)
        .args([
            "fisher",
            "--pk-model",
            "linear",
            "--out",
            linear_path.to_str().expect("utf8 path"),
        ])
        .status()
        .expect("spawn fisher linear");
    assert!(
        status_linear.success(),
        "fisher --pk-model linear should succeed"
    );

    let status_nl = Command::new(exe)
        .args([
            "fisher",
            "--pk-model",
            "nonlinear",
            "--out",
            nonlinear_path.to_str().expect("utf8 path"),
        ])
        .status()
        .expect("spawn fisher nonlinear");
    assert!(
        status_nl.success(),
        "fisher --pk-model nonlinear should succeed"
    );

    let linear_json: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&linear_path).expect("read linear json"))
            .expect("parse linear json");
    let nl_json: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&nonlinear_path).expect("read nl json"))
            .expect("parse nl json");

    assert_eq!(
        linear_json["config"]["use_nonlinear"].as_bool(),
        Some(false)
    );
    assert_eq!(nl_json["config"]["use_nonlinear"].as_bool(), Some(true));
}
