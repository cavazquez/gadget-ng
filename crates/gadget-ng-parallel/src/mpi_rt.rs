use crate::pack;
use crate::ParallelRuntime;
use gadget_ng_core::{Particle, Vec3};
use mpi::collective::SystemOperation;
use mpi::datatype::PartitionMut;
use mpi::environment::Universe;
use mpi::traits::*;
use mpi::Count;

pub struct MpiRuntime {
    _universe: Universe,
}

impl MpiRuntime {
    /// Debe llamarse una sola vez al inicio del proceso MPI.
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        let universe = mpi::initialize().expect("MPI no inicializado; lanzar con mpiexec/mpirun");
        Self {
            _universe: universe,
        }
    }

    fn world(&self) -> mpi::topology::SimpleCommunicator {
        self._universe.world()
    }
}

impl ParallelRuntime for MpiRuntime {
    fn rank(&self) -> i32 {
        self.world().rank()
    }

    fn size(&self) -> i32 {
        self.world().size()
    }

    fn barrier(&self) {
        self.world().barrier();
    }

    fn root_eprintln(&self, msg: &str) {
        if self.rank() == 0 {
            eprintln!("{msg}");
        }
    }

    fn allgatherv_state(
        &self,
        local: &[Particle],
        total_count: usize,
        global_positions: &mut Vec<Vec3>,
        global_masses: &mut Vec<f64>,
    ) {
        let world = self.world();
        let sendbuf = pack::pack_pm(local);
        let my_elems = sendbuf.len() as Count;
        let mut counts = vec![0 as Count; world.size() as usize];
        world.all_gather_into(&[my_elems][..], &mut counts[..]);

        let mut displs: Vec<Count> = Vec::with_capacity(world.size() as usize);
        let mut acc: Count = 0;
        for &c in &counts {
            displs.push(acc);
            acc += c;
        }
        let total_elems = acc as usize;
        let mut recvbuf = vec![0.0f64; total_elems];
        {
            let mut part = PartitionMut::new(&mut recvbuf[..], counts.clone(), &displs[..]);
            world.all_gather_varcount_into(&sendbuf[..], &mut part);
        }
        // SAFETY: `mpi::Count` es alias de `c_int` (normalmente `i32`), misma representación.
        let counts_i32: &[i32] =
            unsafe { std::slice::from_raw_parts(counts.as_ptr().cast::<i32>(), counts.len()) };
        pack::unpack_pm_flat(
            &recvbuf,
            counts_i32,
            global_positions,
            global_masses,
            total_count,
        );
    }

    fn root_gather_particles(
        &self,
        local: &[Particle],
        total_count: usize,
    ) -> Option<Vec<Particle>> {
        let world = self.world();
        let root_rank = 0;
        let root = world.process_at_rank(root_rank);
        let rank = world.rank();
        let sendbuf = pack::pack_full(local);
        let my_elems = sendbuf.len() as Count;
        let mut counts = vec![0 as Count; world.size() as usize];
        world.all_gather_into(&[my_elems][..], &mut counts[..]);

        if rank == root_rank {
            let mut displs: Vec<Count> = Vec::with_capacity(world.size() as usize);
            let mut acc: Count = 0;
            for &c in &counts {
                displs.push(acc);
                acc += c;
            }
            let mut recvbuf = vec![0.0f64; acc as usize];
            {
                let mut part = PartitionMut::new(&mut recvbuf[..], counts, &displs[..]);
                root.gather_varcount_into_root(&sendbuf[..], &mut part);
            }
            Some(pack::unpack_full_to_particles(&recvbuf, total_count))
        } else {
            root.gather_varcount_into(&sendbuf[..]);
            None
        }
    }

    fn allreduce_sum_f64(&self, v: f64) -> f64 {
        let world = self.world();
        let mut out = 0.0f64;
        world.all_reduce_into(&v, &mut out, SystemOperation::sum());
        out
    }

    fn allreduce_min_f64(&self, v: f64) -> f64 {
        let world = self.world();
        let mut out = 0.0f64;
        world.all_reduce_into(&v, &mut out, SystemOperation::min());
        out
    }

    fn allreduce_max_f64(&self, v: f64) -> f64 {
        let world = self.world();
        let mut out = 0.0f64;
        world.all_reduce_into(&v, &mut out, SystemOperation::max());
        out
    }

    fn exchange_domain_by_x(
        &self,
        local: &mut Vec<Particle>,
        my_x_lo: f64,
        my_x_hi: f64,
    ) {
        let world = self.world();
        let rank  = world.rank();
        let size  = world.size();

        // Separar partículas que deben migrar a la izquierda, derecha, o quedarse.
        let mut stay     = Vec::new();
        let mut go_left  = Vec::new();
        let mut go_right = Vec::new();
        for p in local.drain(..) {
            if p.position.x < my_x_lo && rank > 0 {
                go_left.push(p);
            } else if p.position.x >= my_x_hi && rank < size - 1 {
                go_right.push(p);
            } else {
                stay.push(p);
            }
        }

        // Intercambio punto-a-punto: patrón odd-even para evitar deadlock.
        // Ronda 1: enviar/recibir en dirección derecha.
        // Ronda 2: enviar/recibir en dirección izquierda.
        let left  = if rank > 0        { Some(world.process_at_rank(rank - 1)) } else { None };
        let right = if rank < size - 1 { Some(world.process_at_rank(rank + 1)) } else { None };

        let recv_from_right = point_to_point_exchange(&world, rank, &right, &left,
            &pack::pack_halo(&go_right), &pack::pack_halo(&go_left));

        // Recombinar
        *local = stay;
        local.extend(recv_from_right.0);
        local.extend(recv_from_right.1);
    }

    fn exchange_halos_by_x(
        &self,
        local: &[Particle],
        my_x_lo: f64,
        my_x_hi: f64,
        halo_width: f64,
    ) -> Vec<Particle> {
        let world = self.world();
        let rank  = world.rank();
        let size  = world.size();

        // Partículas que son halo para el vecino izquierdo y derecho.
        let send_left:  Vec<&Particle> = local.iter()
            .filter(|p| p.position.x < my_x_lo + halo_width && rank > 0)
            .collect();
        let send_right: Vec<&Particle> = local.iter()
            .filter(|p| p.position.x > my_x_hi - halo_width && rank < size - 1)
            .collect();

        let left  = if rank > 0        { Some(world.process_at_rank(rank - 1)) } else { None };
        let right = if rank < size - 1 { Some(world.process_at_rank(rank + 1)) } else { None };

        // Serializar (solo referencias → clonar los datos necesarios)
        let buf_to_right: Vec<Particle> = send_right.iter().map(|p| (*p).clone()).collect();
        let buf_to_left:  Vec<Particle> = send_left.iter().map(|p| (*p).clone()).collect();

        let (from_left, from_right) = point_to_point_exchange(
            &world, rank, &right, &left,
            &pack::pack_halo(&buf_to_right),
            &pack::pack_halo(&buf_to_left),
        );
        let mut halos = from_left;
        halos.extend(from_right);
        halos
    }
}

// ── Utilidades de comunicación punto-a-punto ─────────────────────────────────

/// Intercambia datos f64 con los vecinos izquierdo y derecho de forma deadlock-free.
///
/// Usa patrón de dos rondas para evitar deadlock:
/// - Ronda 1 (pares envían derecha, impares reciben derecha): send_data_right ↔ recv_from_left
/// - Ronda 2 (pares envían izquierda, impares reciben izquierda): send_data_left ↔ recv_from_right
///
/// Devuelve `(halos_from_left, halos_from_right)` ya desempaquetados como `Vec<Particle>`.
fn point_to_point_exchange(
    world: &mpi::topology::SimpleCommunicator,
    rank:  i32,
    right: &Option<mpi::topology::Process<'_>>,
    left:  &Option<mpi::topology::Process<'_>>,
    send_right: &[f64],
    send_left:  &[f64],
) -> (Vec<Particle>, Vec<Particle>) {
    let mut from_left:  Vec<f64> = Vec::new();
    let mut from_right: Vec<f64> = Vec::new();

    // ── Ronda 1: pares envían →derecha y reciben ←izquierda ──────────────────
    if rank % 2 == 0 {
        if let Some(r) = right { r.send(send_right); }
        if let Some(l) = left  { let (v, _) = l.receive_vec::<f64>(); from_left = v; }
    } else {
        if let Some(l) = left  { let (v, _) = l.receive_vec::<f64>(); from_left = v; }
        if let Some(r) = right { r.send(send_right); }
    }
    world.barrier();

    // ── Ronda 2: pares envían ←izquierda y reciben →derecha ──────────────────
    if rank % 2 == 0 {
        if let Some(l) = left  { l.send(send_left); }
        if let Some(r) = right { let (v, _) = r.receive_vec::<f64>(); from_right = v; }
    } else {
        if let Some(r) = right { let (v, _) = r.receive_vec::<f64>(); from_right = v; }
        if let Some(l) = left  { l.send(send_left); }
    }
    world.barrier();

    (pack::unpack_halo(&from_left), pack::unpack_halo(&from_right))
}
