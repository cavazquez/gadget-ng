use crate::ParallelRuntime;
use crate::sfc::SfcDecomposition;
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

    fn allreduce_min_f64(&self, v: f64) -> f64 {
        v
    }

    fn allreduce_max_f64(&self, v: f64) -> f64 {
        v
    }

    fn exchange_domain_by_x(&self, _local: &mut Vec<Particle>, _my_x_lo: f64, _my_x_hi: f64) {
        // Serial: rango único, nunca hay partículas fuera del dominio.
    }

    fn exchange_halos_by_x(
        &self,
        _local: &[Particle],
        _my_x_lo: f64,
        _my_x_hi: f64,
        _halo_width: f64,
    ) -> Vec<Particle> {
        // Serial: no hay vecinos; el árbol local ya tiene todas las partículas.
        Vec::new()
    }

    fn exchange_domain_by_z(&self, _local: &mut Vec<Particle>, _my_z_lo: f64, _my_z_hi: f64) {
        // Serial: rango único, todas las partículas pertenecen al rango 0.
    }

    fn exchange_halos_by_z(
        &self,
        _local: &[Particle],
        _my_z_lo: f64,
        _my_z_hi: f64,
        _halo_width: f64,
    ) -> Vec<Particle> {
        Vec::new()
    }

    fn exchange_halos_by_z_periodic(
        &self,
        _local: &[Particle],
        _my_z_lo: f64,
        _my_z_hi: f64,
        _halo_width: f64,
    ) -> Vec<Particle> {
        // P=1: árbol local ya tiene todas las partículas; no hay vecinos periódicos.
        Vec::new()
    }

    fn exchange_halos_3d_periodic(
        &self,
        _local: &[Particle],
        _box_size: f64,
        _halo_width: f64,
    ) -> Vec<Particle> {
        // P=1: árbol local ya tiene todas las partículas; no hay vecinos.
        Vec::new()
    }

    fn exchange_domain_sfc(&self, _local: &mut Vec<Particle>, _decomp: &SfcDecomposition) {
        // Serial: rango único, todas las partículas pertenecen al rango 0.
    }

    fn exchange_halos_sfc(
        &self,
        _local: &[Particle],
        _decomp: &SfcDecomposition,
        _halo_width: f64,
    ) -> Vec<Particle> {
        Vec::new()
    }

    fn allreduce_sum_f64_slice(&self, _buf: &mut [f64]) {
        // Serial: rango único; el buffer ya contiene la suma global.
    }

    fn allgather_f64(&self, local: &[f64]) -> Vec<Vec<f64>> {
        vec![local.to_vec()]
    }

    fn alltoallv_f64(&self, _sends: &[Vec<f64>]) -> Vec<Vec<f64>> {
        // En serial no hay otros rangos; el llamante envía sends[0] a sí mismo,
        // pero como no hay migración real, devolvemos un slot vacío.
        vec![Vec::new()]
    }

    fn alltoallv_f64_overlap(
        &self,
        _sends: Vec<Vec<f64>>,
        overlap_work: &mut dyn FnMut(),
    ) -> Vec<Vec<f64>> {
        // En serial: no hay comunicación real; ejecutar el trabajo de overlap
        // de forma síncrona y devolver slot vacío.
        overlap_work();
        vec![Vec::new()]
    }

    fn alltoallv_f64_subgroup(&self, sends: &[Vec<f64>], _color: i32) -> Vec<Vec<f64>> {
        // En serial el subgrupo tiene un solo miembro (self).
        // self-comunicación: devolvemos lo que enviamos a nosotros mismos.
        assert_eq!(
            sends.len(),
            1,
            "serial: subgrupo debe tener exactamente 1 miembro"
        );
        vec![sends[0].clone()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ParallelRuntime;

    #[test]
    fn serial_runtime_is_single_rank() {
        let rt = SerialRuntime;
        assert_eq!(rt.rank(), 0);
        assert_eq!(rt.size(), 1);
    }

    #[test]
    fn allgatherv_state_fills_global_arrays() {
        let rt = SerialRuntime;
        let local = vec![
            Particle::new(1, 2.0, Vec3::new(3.0, 0.0, 0.0), Vec3::zero()),
            Particle::new(0, 1.0, Vec3::new(1.0, 0.0, 0.0), Vec3::zero()),
        ];
        let mut pos = Vec::new();
        let mut mass = Vec::new();
        rt.allgatherv_state(&local, 2, &mut pos, &mut mass);
        assert_eq!(pos.len(), 2);
        assert!((mass[0] - 1.0).abs() < 1e-12);
        assert!((pos[1].x - 3.0).abs() < 1e-12);
    }

    #[test]
    fn allgather_f64_wraps_local_slice() {
        let rt = SerialRuntime;
        let out = rt.allgather_f64(&[1.0, 2.0]);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0], vec![1.0, 2.0]);
    }

    #[test]
    fn halo_exchange_returns_empty() {
        let rt = SerialRuntime;
        let local = vec![Particle::new(0, 1.0, Vec3::zero(), Vec3::zero())];
        assert!(rt.exchange_halos_3d_periodic(&local, 1.0, 0.1).is_empty());
    }
}
