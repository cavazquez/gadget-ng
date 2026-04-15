mod decompose;
#[allow(dead_code)]
mod pack;
mod serial;

#[cfg(feature = "mpi")]
mod mpi_rt;

pub use decompose::gid_block_range;
pub use serial::SerialRuntime;

#[cfg(feature = "mpi")]
pub use mpi_rt::MpiRuntime;

use gadget_ng_core::{Particle, Vec3};

/// Entorno de ejecución paralela: mismo contrato para serial y MPI.
pub trait ParallelRuntime {
    fn rank(&self) -> i32;
    fn size(&self) -> i32;
    fn barrier(&self);

    /// Solo el rango 0 escribe a stderr.
    fn root_eprintln(&self, msg: &str);

    /// Reconstruye `positions[global_id]` y `masses[global_id]` para el cálculo de fuerzas.
    fn allgatherv_state(
        &self,
        local: &[Particle],
        total_count: usize,
        global_positions: &mut Vec<Vec3>,
        global_masses: &mut Vec<f64>,
    );

    /// Reúne el estado completo en el rango 0 para I/O (`None` en otros rangos).
    fn root_gather_particles(
        &self,
        local: &[Particle],
        total_count: usize,
    ) -> Option<Vec<Particle>>;

    /// Suma global (`MPI_Allreduce` en MPI).
    fn allreduce_sum_f64(&self, v: f64) -> f64;
}
