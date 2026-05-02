//! Snapshot provenance metadata (`Provenance`, units, feature tags).

use crate::config_load;
use crate::error::CliError;
use gadget_ng_core::RunConfig;
use gadget_ng_io::{Provenance, SnapshotEnv, SnapshotUnits};
use std::process::Command;

pub(crate) fn try_git_commit() -> Option<String> {
    let out = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()?;
    if out.status.success() {
        String::from_utf8(out.stdout)
            .ok()
            .map(|s| s.trim().to_string())
    } else {
        None
    }
}
pub(crate) fn provenance_for_run(cfg: &RunConfig) -> Result<Provenance, CliError> {
    let cfg_hash = config_load::config_canonical_hash(cfg)?;
    let mut feats = enabled_features_list();
    feats.extend(gravity_kernel_tags(cfg));
    Ok(Provenance::new(
        env!("CARGO_PKG_VERSION"),
        try_git_commit(),
        if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        }
        .to_string(),
        feats,
        std::env::args().collect(),
        cfg_hash,
    ))
}

/// Etiquetas sintéticas para snapshots: intención de backend gravitatorio (no garantiza que wgpu haya inicializado).
#[cfg(any(feature = "gpu", feature = "cuda", feature = "hip"))]
fn gravity_kernel_tags(cfg: &RunConfig) -> Vec<String> {
    use gadget_ng_core::SolverKind;
    let mut v = Vec::new();
    #[cfg(feature = "gpu")]
    {
        if cfg.performance.use_gpu && cfg.gravity.solver == SolverKind::Direct {
            v.push("gravity:intent_wgpu_direct".into());
        }
        if cfg.performance.use_gpu_barnes_hut && cfg.gravity.solver == SolverKind::BarnesHut {
            v.push("gravity:intent_wgpu_bh_fmm".into());
        }
    }
    #[cfg(feature = "cuda")]
    {
        if cfg.performance.use_gpu_cuda && cfg.gravity.solver == SolverKind::Pm {
            v.push("gravity:intent_cuda_pm".into());
        }
    }
    #[cfg(all(feature = "gpu", feature = "cuda"))]
    {
        if cfg.performance.use_gpu_treepm && cfg.gravity.solver == SolverKind::TreePm {
            v.push("gravity:intent_gpu_treepm_hybrid".into());
        }
    }
    #[cfg(feature = "hip")]
    {
        if cfg.performance.use_gpu_hip && cfg.gravity.solver == SolverKind::Pm {
            v.push("gravity:intent_hip_pm".into());
        }
    }
    v
}

#[cfg(not(any(feature = "gpu", feature = "cuda", feature = "hip")))]
fn gravity_kernel_tags(_cfg: &RunConfig) -> Vec<String> {
    Vec::new()
}

/// Construye el bloque de unidades para `SnapshotEnv`, si el config usa unidades físicas.
pub(crate) fn snapshot_units_for(cfg: &RunConfig) -> Option<SnapshotUnits> {
    if cfg.units.enabled {
        Some(SnapshotUnits {
            length_in_kpc: cfg.units.length_in_kpc,
            mass_in_msun: cfg.units.mass_in_msun,
            velocity_in_km_s: cfg.units.velocity_in_km_s,
            time_in_gyr: cfg.units.time_unit_in_gyr(),
            g_internal: cfg.units.compute_g(),
        })
    } else {
        None
    }
}

pub(crate) fn snapshot_env_for(cfg: &RunConfig, time: f64, redshift: f64) -> SnapshotEnv {
    // h_dimless es h = H₀/(100 km/s/Mpc). cfg.cosmology.h0 está en unidades internas
    // (1/t_sim), así que no es equivalente. Se usa como mejor aproximación disponible;
    // para runs con unidades físicas el valor correcto es el `h` de las ICs Zeldovich.
    let (omega_m, omega_lambda, h_dimless) = if cfg.cosmology.enabled {
        (
            cfg.cosmology.omega_m,
            cfg.cosmology.omega_lambda,
            cfg.cosmology.h0,
        )
    } else {
        (0.0, 0.0, 1.0)
    };
    SnapshotEnv {
        time,
        redshift,
        box_size: cfg.simulation.box_size,
        units: snapshot_units_for(cfg),
        omega_m,
        omega_lambda,
        h_dimless,
    }
}

pub(crate) fn enabled_features_list() -> Vec<String> {
    let mut f = Vec::new();
    if cfg!(feature = "mpi") {
        f.push("mpi".into());
    }
    if cfg!(feature = "bincode") {
        f.push("bincode".into());
    }
    if cfg!(feature = "hdf5") {
        f.push("hdf5".into());
    }
    if cfg!(feature = "gpu") {
        f.push("gpu".into());
    }
    if cfg!(feature = "simd") {
        f.push("simd".into());
    }
    f
}
