mod decompose;
pub mod domain;
pub mod halo3d;
pub mod pack;
mod serial;
pub mod sfc;

#[cfg(feature = "mpi")]
mod mpi_rt;

pub use decompose::gid_block_range;
pub use domain::SlabDecomposition;
pub use halo3d::{
    aabb_to_f64, compute_aabb_3d, f64_to_aabb, is_in_periodic_halo,
    min_dist2_to_aabb_3d_periodic, minimum_image_scalar, Aabb3,
};
pub use serial::SerialRuntime;
pub use sfc::{hilbert3, morton3, SfcDecomposition};

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

    /// Migra partículas a su rango propietario según la descomposición de slabs en **z**.
    ///
    /// Análogo a [`exchange_domain_by_x`] pero usando la coordenada z.
    /// Se usa para la descomposición de dominio del slab PM distribuido (Fase 20).
    fn exchange_domain_by_z(&self, local: &mut Vec<Particle>, my_z_lo: f64, my_z_hi: f64);

    /// Intercambia partículas de halo con los rangos vecinos en el eje z.
    ///
    /// Análogo a [`exchange_halos_by_x`] pero para el eje z.
    fn exchange_halos_by_z(
        &self,
        local: &[Particle],
        my_z_lo: f64,
        my_z_hi: f64,
        halo_width: f64,
    ) -> Vec<Particle>;

    /// Intercambia halos de corto alcance en z con **wrap periódico**.
    ///
    /// Versión periódica de [`exchange_halos_by_z`] para el path TreePM distribuido (Fase 21).
    /// Garantiza que rank 0 reciba halos de rank P-1 y viceversa (borde periódico).
    ///
    /// - Cada rank envía a su vecino izquierdo las partículas con `z < z_lo + halo_width`.
    /// - Cada rank envía a su vecino derecho las partículas con `z > z_hi - halo_width`.
    /// - El vecino izquierdo de rank 0 es rank P-1 (periódico); ídem al revés.
    ///
    /// Las posiciones de las partículas halo **no se modifican** (el llamante aplica
    /// `minimum_image` en el walk del árbol). En modo serial (P=1) retorna Vec vacío
    /// ya que el árbol local ya contiene todas las partículas.
    fn exchange_halos_by_z_periodic(
        &self,
        local: &[Particle],
        my_z_lo: f64,
        my_z_hi: f64,
        halo_width: f64,
    ) -> Vec<Particle>;

    /// Halo volumétrico 3D periódico para el árbol de corto alcance (Fase 22).
    ///
    /// Para cada rank r, calcula el AABB real de sus partículas locales y determina
    /// qué partículas de otros ranks están dentro de `halo_width` de ese AABB,
    /// usando `minimum_image` periódico en las tres dimensiones.
    ///
    /// ## Protocolo
    ///
    /// 1. Calcula el AABB real de `local` (no los límites del slab).
    /// 2. `allgather_f64` de todas las AABBs (6 f64 por rank).
    /// 3. Para cada rank r: envía partículas locales con
    ///    `min_dist2_to_aabb_3d_periodic(p, aabb_r, box_size) < halo_width²`.
    /// 4. `alltoallv_f64` y desempaqueta halos recibidos.
    ///
    /// ## Correctitud periódica
    ///
    /// Cubre todos los casos: bordes en x, y, z e interacciones diagonales periódicas
    /// (x+y, x+z, y+z, x+y+z). Correcto para cualquier descomposición de dominio
    /// (Z-slab, SFC, octantes), a diferencia de `exchange_halos_sfc` que usa
    /// coordenadas absolutas sin wrap.
    ///
    /// ## Equivalencia con halo 1D para Z-slab uniforme
    ///
    /// Para Z-slab con partículas uniformes, las AABBs abarcan [0,L)×[0,L)×[z_lo,z_hi).
    /// Expandidas por r_cut en los tres ejes producen el mismo conjunto de vecinos
    /// que el halo 1D-z. La diferencia aparece con descomposiciones no-Z-slab.
    ///
    /// En modo serial (P=1) retorna `Vec::new()`.
    fn exchange_halos_3d_periodic(
        &self,
        local: &[Particle],
        box_size: f64,
        halo_width: f64,
    ) -> Vec<Particle>;

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

    // ── PM distribuido ────────────────────────────────────────────────────────

    /// Reduce suma elemento a elemento de un array `f64` entre todos los rangos
    /// (`MPI_Allreduce(MPI_SUM)` en MPI; no-op en serial).
    ///
    /// Tras la llamada, `buf[i]` contiene la suma de `buf[i]` de todos los rangos.
    /// Se usa para combinar grids de densidad parciales en el PM distribuido
    /// (Fase 19): cada rank deposita su contribución local y la reducción produce
    /// la densidad global sin necesitar `allgatherv_state` de partículas.
    fn allreduce_sum_f64_slice(&self, buf: &mut [f64]);

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

    // ── Subcomunicadores (Pencil FFT 2D) ─────────────────────────────────────

    /// Alltoall dentro de un subgrupo identificado por `color`.
    ///
    /// Todos los ranks que comparten el mismo `color` forman un subgrupo.
    /// `sends[i]` = datos a enviar al i-ésimo miembro del subgrupo (rank local 0, 1, …).
    /// Devuelve `received[i]` = datos recibidos del i-ésimo miembro.
    ///
    /// Se usa en la FFT pencil 2D para intercambiar datos dentro de filas (Y-group)
    /// o columnas (Z-group) de la malla 2D de procesos, evitando la sobrecarga
    /// de un alltoall con todos P ranks cuando solo se necesita comunicar con Py o Pz.
    ///
    /// En MPI: usa `MPI_Comm_split(color)` para crear un subcomunicador temporal
    /// y realiza un alltoallv dentro de él.
    /// En serial (P=1): el subgrupo tiene un único miembro; devuelve `vec![sends[0].clone()]`.
    fn alltoallv_f64_subgroup(&self, sends: &[Vec<f64>], color: i32) -> Vec<Vec<f64>>;
}
