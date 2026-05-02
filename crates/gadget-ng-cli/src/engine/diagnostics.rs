//! Kinetic/momentum diagnostics and `diagnostics.jsonl` lines.

use crate::error::CliError;
use gadget_ng_core::Particle;
use gadget_ng_integrators::StepStats;
use gadget_ng_parallel::ParallelRuntime;
use std::fs::File;
use std::io::Write;
use std::path::Path;

use super::timings::{CosmoDiag, HpcStepStats, LocalMoments};

pub(crate) fn should_rebalance(
    step: u64,
    start_step: u64,
    interval: u64,
    cost_pending: bool,
) -> bool {
    if cost_pending {
        return true;
    }
    if interval == 0 {
        return true;
    }
    (step - start_step).is_multiple_of(interval)
}

pub(crate) fn kinetic_local(parts: &[Particle]) -> f64 {
    parts
        .iter()
        .map(|p| 0.5 * p.mass * p.velocity.dot(p.velocity))
        .sum()
}

pub(crate) fn local_moments(parts: &[Particle]) -> LocalMoments {
    let mut m = LocalMoments::default();
    for p in parts {
        let w = p.mass;
        let r = &p.position;
        let v = &p.velocity;
        m.mass += w;
        m.p[0] += w * v.x;
        m.p[1] += w * v.y;
        m.p[2] += w * v.z;
        m.l[0] += w * (r.y * v.z - r.z * v.y);
        m.l[1] += w * (r.z * v.x - r.x * v.z);
        m.l[2] += w * (r.x * v.y - r.y * v.x);
        m.mass_weighted_pos[0] += w * r.x;
        m.mass_weighted_pos[1] += w * r.y;
        m.mass_weighted_pos[2] += w * r.z;
    }
    m
}
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_diagnostic_line<R: ParallelRuntime + ?Sized>(
    rt: &R,
    step: u64,
    local: &[Particle],
    diag_path: &Path,
    diag_file: &mut Option<File>,
    step_stats: Option<&StepStats>,
    hpc_stats: Option<&HpcStepStats>,
    cosmo_diag: Option<&CosmoDiag>,
) -> Result<(), CliError> {
    let ke_loc = kinetic_local(local);
    let ke = rt.allreduce_sum_f64(ke_loc);
    // Agregados O(N): p, L, COM. Usan 10 allreduces; coste despreciable frente al paso.
    let lm = local_moments(local);
    let px = rt.allreduce_sum_f64(lm.p[0]);
    let py = rt.allreduce_sum_f64(lm.p[1]);
    let pz = rt.allreduce_sum_f64(lm.p[2]);
    let lx = rt.allreduce_sum_f64(lm.l[0]);
    let ly = rt.allreduce_sum_f64(lm.l[1]);
    let lz = rt.allreduce_sum_f64(lm.l[2]);
    let mrx = rt.allreduce_sum_f64(lm.mass_weighted_pos[0]);
    let mry = rt.allreduce_sum_f64(lm.mass_weighted_pos[1]);
    let mrz = rt.allreduce_sum_f64(lm.mass_weighted_pos[2]);
    let mtot = rt.allreduce_sum_f64(lm.mass);
    let com = if mtot > 0.0 {
        [mrx / mtot, mry / mtot, mrz / mtot]
    } else {
        [0.0, 0.0, 0.0]
    };
    if let Some(f) = diag_file {
        let mut obj = serde_json::json!({
            "step": step,
            "kinetic_energy": ke,
            "momentum": [px, py, pz],
            "angular_momentum": [lx, ly, lz],
            "com": com,
            "mass_total": mtot,
        });
        // Si se proveen estadísticas del paso jerárquico, añadirlas como campos opcionales.
        if let Some(ss) = step_stats {
            let map = obj.as_object_mut().unwrap();
            map.insert(
                "level_histogram".into(),
                serde_json::Value::Array(
                    ss.level_histogram
                        .iter()
                        .map(|&v| serde_json::Value::Number(v.into()))
                        .collect(),
                ),
            );
            map.insert("active_total".into(), ss.active_total.into());
            map.insert("force_evals".into(), ss.force_evals.into());
            map.insert("dt_min_effective".into(), ss.dt_min_effective.into());
            map.insert("dt_max_effective".into(), ss.dt_max_effective.into());
        }
        if let Some(hs) = hpc_stats {
            let map = obj.as_object_mut().unwrap();
            map.insert(
                "hpc_stats".into(),
                serde_json::to_value(hs).unwrap_or(serde_json::Value::Null),
            );
        }
        // Campos cosmológicos opcionales.
        if let Some(cd) = cosmo_diag {
            let map = obj.as_object_mut().unwrap();
            map.insert("a".into(), cd.a.into());
            map.insert("z".into(), cd.z.into());
            map.insert("v_rms".into(), cd.v_rms.into());
            map.insert("delta_rms".into(), cd.delta_rms.into());
            map.insert("hubble".into(), cd.hubble.into());
            // Diagnóstico TreePM SR-SFC (Fases 23/24) si estuvo activo en este paso.
            if let Some(td) = cd.treepm {
                map.insert(
                    "treepm".into(),
                    serde_json::to_value(td).unwrap_or(serde_json::Value::Null),
                );
            }
        }
        let line = obj.to_string();
        writeln!(f, "{line}").map_err(|e| CliError::io(diag_path, e))?;
    }
    rt.barrier();
    Ok(())
}
