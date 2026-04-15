use crate::ParallelRuntime;
use gadget_ng_core::{Particle, Vec3};

#[derive(Debug, Default, Clone, Copy)]
pub struct SerialRuntime;

impl ParallelRuntime for SerialRuntime {
    fn rank(&self) -> i32 {
        0
    }

    fn size(&self) -> i32 {
        1
    }

    fn barrier(&self) {}

    fn root_eprintln(&self, msg: &str) {
        eprintln!("{msg}");
    }

    fn allgatherv_state(
        &self,
        local: &[Particle],
        total_count: usize,
        global_positions: &mut Vec<Vec3>,
        global_masses: &mut Vec<f64>,
    ) {
        global_positions.clear();
        global_masses.clear();
        global_positions.resize(total_count, Vec3::zero());
        global_masses.resize(total_count, 0.0);
        for p in local {
            if p.global_id < total_count {
                global_positions[p.global_id] = p.position;
                global_masses[p.global_id] = p.mass;
            }
        }
    }

    fn root_gather_particles(
        &self,
        local: &[Particle],
        total_count: usize,
    ) -> Option<Vec<Particle>> {
        debug_assert_eq!(
            local.len(),
            total_count,
            "en modo serial se espera el sistema completo en un solo rango"
        );
        Some(local.to_vec())
    }

    fn allreduce_sum_f64(&self, v: f64) -> f64 {
        v
    }
}
