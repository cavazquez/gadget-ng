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
}
