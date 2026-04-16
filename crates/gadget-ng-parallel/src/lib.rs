mod decompose;
#[allow(dead_code)]
mod pack;
mod serial;
pub mod domain;

#[cfg(feature = "mpi")]
mod mpi_rt;

pub use decompose::gid_block_range;
pub use domain::SlabDecomposition;
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

    // ── Árbol distribuido ─────────────────────────────────────────────────────

    /// Mínimo global de `v` entre todos los rangos.
    fn allreduce_min_f64(&self, v: f64) -> f64;

    /// Máximo global de `v` entre todos los rangos.
    fn allreduce_max_f64(&self, v: f64) -> f64;

    /// Migra partículas a su rango propietario según la descomposición de slabs.
    ///
    /// Las partículas cuya posición x ya no pertenece al slab `[my_x_lo, my_x_hi)` se
    /// envían al vecino izquierdo o derecho y se eliminan del vector `local`.
    /// Las partículas recibidas de los vecinos se añaden a `local`.
    fn exchange_domain_by_x(
        &self,
        local: &mut Vec<Particle>,
        my_x_lo: f64,
        my_x_hi: f64,
    );

    /// Intercambia partículas de halo con los rangos vecinos (punto-a-punto).
    ///
    /// - Envía al vecino izquierdo  las partículas con `x < my_x_lo + halo_width`.
    /// - Envía al vecino derecho    las partículas con `x > my_x_hi - halo_width`.
    /// - Devuelve las partículas recibidas de ambos vecinos (halos).
    ///
    /// Las partículas halos NO se añaden a `local`; el llamante decide cómo combinarlas.
    fn exchange_halos_by_x(
        &self,
        local: &[Particle],
        my_x_lo: f64,
        my_x_hi: f64,
        halo_width: f64,
    ) -> Vec<Particle>;
}
