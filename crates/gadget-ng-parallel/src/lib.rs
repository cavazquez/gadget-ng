mod decompose;
pub mod domain;
pub mod pack;
mod serial;
pub mod sfc;

#[cfg(feature = "mpi")]
mod mpi_rt;

pub use decompose::gid_block_range;
pub use domain::SlabDecomposition;
pub use serial::SerialRuntime;
pub use sfc::{morton3, SfcDecomposition};

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
    fn exchange_domain_by_x(&self, local: &mut Vec<Particle>, my_x_lo: f64, my_x_hi: f64);

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

    // ── SFC (Peano-Hilbert) ───────────────────────────────────────────────────

    /// Migra partículas usando una descomposición SFC (Morton Z-order).
    ///
    /// Cada partícula se asigna al rango según `decomp.rank_for_pos(p.position)`.
    /// Las partículas del rango incorrecto se envían a sus rangos correctos usando
    /// un `Alltoallv`, no solo a vecinos rank±1.
    ///
    /// En modo serial (1 rango) es un no-op.
    fn exchange_domain_sfc(&self, local: &mut Vec<Particle>, decomp: &sfc::SfcDecomposition);

    /// Intercambia halos SFC usando una AABB 3D real.
    ///
    /// Para cada rank r, expande su AABB local por `halo_width` en las tres
    /// dimensiones y envía partículas propias dentro de esa región expandida.
    /// En modo serial devuelve Vec vacío.
    fn exchange_halos_sfc(
        &self,
        local: &[Particle],
        decomp: &sfc::SfcDecomposition,
        halo_width: f64,
    ) -> Vec<Particle>;

    // ── Primitivas de comunicación genérica ──────────────────────────────────

    /// Allgather de datos `f64`: contribuye `local` desde este rango y recibe
    /// los datos de todos los rangos. Devuelve `result[r]` = datos del rango `r`.
    ///
    /// En serial devuelve `vec![local.to_vec()]`.
    fn allgather_f64(&self, local: &[f64]) -> Vec<Vec<f64>>;

    /// Alltoallv de datos `f64`: `sends[r]` = datos a enviar al rango `r`.
    /// Devuelve `received[r]` = datos recibidos del rango `r`.
    ///
    /// En serial devuelve `vec![vec![]]` (ningún otro rango existe).
    fn alltoallv_f64(&self, sends: &[Vec<f64>]) -> Vec<Vec<f64>>;

    /// Alltoallv no-bloqueante de `f64` con overlap de cómputo.
    ///
    /// Implementa el patrón:
    /// 1. Intercambia conteos (bloqueante, O(P) enteros, coste despreciable).
    /// 2. Emite todos los `Isend` + `Irecv` no-bloqueantes (P2P).
    /// 3. Llama a `overlap_work()` mientras los mensajes están en vuelo.
    /// 4. Espera todas las requests.
    /// 5. Devuelve `received[r]` = datos recibidos del rango `r`.
    ///
    /// `sends` se pasa por valor para garantizar que los buffers viven
    /// durante toda la duración de las requests no-bloqueantes.
    ///
    /// En serial llama a `overlap_work()` inmediatamente y devuelve `vec![vec![]]`.
    fn alltoallv_f64_overlap(
        &self,
        sends: Vec<Vec<f64>>,
        overlap_work: &mut dyn FnMut(),
    ) -> Vec<Vec<f64>>;
}
