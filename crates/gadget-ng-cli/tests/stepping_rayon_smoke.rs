//! Ejecuta `gadget-ng stepping` con el ejemplo `nbody_bh_dtree_rayon_smoke.toml`.
//! Requiere `--features simd` (Rayon + kernels locales paralelos).

#[cfg(feature = "simd")]
mod simd_only {
    use std::process::Command;

    #[test]
    fn stepping_bh_dtree_rayon_smoke_exits_ok() {
        let out = tempfile::tempdir().expect("tempdir");
        let manifest = env!("CARGO_MANIFEST_DIR");
        let cfg = format!("{manifest}/../../examples/nbody_bh_dtree_rayon_smoke.toml");
        let exe = env!("CARGO_BIN_EXE_gadget-ng");
        let status = Command::new(exe)
            .args([
                "stepping",
                "--config",
                &cfg,
                "--out",
                out.path().to_str().expect("utf8 out path"),
            ])
            .status()
            .expect("spawn gadget-ng");
        assert!(
            status.success(),
            "stepping with simd + deterministic=false + use_distributed_tree should succeed"
        );
    }
}
