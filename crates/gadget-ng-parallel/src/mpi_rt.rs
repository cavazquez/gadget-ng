use crate::pack;
use crate::ParallelRuntime;
use gadget_ng_core::{Particle, Vec3};
use mpi::collective::SystemOperation;
use mpi::datatype::{Partition, PartitionMut};
use mpi::environment::Universe;
use mpi::request::WaitGuard;
use mpi::traits::*;
use mpi::Count;

pub struct MpiRuntime {
    _universe: Universe,
}

/// Tag MPI dedicado a la fase de conteos del intercambio halos SFC (evita mezclar con `f64`).
const TAG_SFC_HALO_COUNT: mpi::Tag = 9001;

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

    /// Fase de datos P2P (Isend/Irecv) una vez conocidos `recv_counts` por rango.
    fn alltoallv_f64_p2p_from_recv_counts(
        &self,
        sends: Vec<Vec<f64>>,
        recv_counts: Vec<Count>,
        overlap_work: &mut dyn FnMut(),
    ) -> Vec<Vec<f64>> {
        let world = self.world();
        let size = world.size() as usize;
        let rank = world.rank() as usize;

        let mut recv_offsets = vec![0usize; size];
        let mut roff = 0usize;
        for r in 0..size {
            recv_offsets[r] = roff;
            roff += recv_counts[r] as usize;
        }

        let mut flat_recv = vec![0.0f64; roff];

        mpi::request::scope(|scope| {
            let mut guards: Vec<WaitGuard<[f64], _>> = Vec::new();

            for r in 0..size {
                if r == rank {
                    continue;
                }
                let rlen = recv_counts[r] as usize;
                if rlen > 0 {
                    let slice: &mut [f64] = unsafe {
                        std::slice::from_raw_parts_mut(
                            flat_recv.as_mut_ptr().add(recv_offsets[r]),
                            rlen,
                        )
                    };
                    let req = world
                        .process_at_rank(r as i32)
                        .immediate_receive_into(scope, slice);
                    guards.push(WaitGuard::from(req));
                }

                if !sends[r].is_empty() {
                    let req = world
                        .process_at_rank(r as i32)
                        .immediate_send(scope, sends[r].as_slice());
                    guards.push(WaitGuard::from(req));
                }
            }

            overlap_work();
            drop(guards);
        });

        let mut result = Vec::with_capacity(size);
        for r in 0..size {
            let n = recv_counts[r] as usize;
            let off = recv_offsets[r];
            result.push(flat_recv[off..off + n].to_vec());
        }
        result
    }

    /// `Alltoallv` lógico con conteos por pareja geométrica dispersa cuando hay rangos
    /// que no pueden intercambiar halos SFC (sin `MPI_Alltoall` de enteros en ese caso).
    fn alltoallv_f64_halos_sfc(
        &self,
        sends: Vec<Vec<f64>>,
        all_aabbs: &[Vec<f64>],
        halo_width: f64,
        overlap_work: &mut dyn FnMut(),
    ) -> Vec<Vec<f64>> {
        let world = self.world();
        let size = world.size() as usize;
        let rank = world.rank() as usize;

        let send_counts: Vec<Count> = sends.iter().map(|v| v.len() as Count).collect();

        let mut pair_active = vec![false; size];
        for j in 0..size {
            if j != rank {
                pair_active[j] =
                    crate::halo3d::halos_sfc_pair_may_exchange(rank, j, all_aabbs, halo_width);
            }
        }
        let inactive_pairs = (0..size).filter(|&j| j != rank && !pair_active[j]).count();

        let recv_counts = if size <= 8 || inactive_pairs == 0 {
            let mut r = vec![0 as Count; size];
            world.all_to_all_into(&send_counts[..], &mut r[..]);
            r
        } else {
            for j in 0..size {
                if j != rank && !pair_active[j] {
                    debug_assert_eq!(
                        send_counts[j], 0,
                        "halos SFC: par geométrico inactivo no debe tener envío pendiente"
                    );
                }
            }
            let mut count_recv = vec![0 as Count; size];
            mpi::request::scope(|scope| {
                let mut guards: Vec<WaitGuard<[Count], _>> = Vec::new();
                let ptr = count_recv.as_mut_ptr();
                for j in 0..size {
                    if j == rank || !pair_active[j] {
                        continue;
                    }
                    // SAFETY: una celda por `j`; índices distintos → slices disjuntos hasta WaitGuard.
                    let buf: &mut [Count] =
                        unsafe { std::slice::from_raw_parts_mut(ptr.add(j), 1) };
                    let req = world
                        .process_at_rank(j as i32)
                        .immediate_receive_into_with_tag(scope, buf, TAG_SFC_HALO_COUNT);
                    guards.push(WaitGuard::from(req));
                }
                let sptr = send_counts.as_ptr();
                for j in 0..size {
                    if j == rank || !pair_active[j] {
                        continue;
                    }
                    // SAFETY: `send_counts` no se muta hasta completar las requests.
                    let buf: &[Count] = unsafe { std::slice::from_raw_parts(sptr.add(j), 1) };
                    let req = world.process_at_rank(j as i32).immediate_send_with_tag(
                        scope,
                        buf,
                        TAG_SFC_HALO_COUNT,
                    );
                    guards.push(WaitGuard::from(req));
                }
                drop(guards);
            });
            count_recv
        };

        self.alltoallv_f64_p2p_from_recv_counts(sends, recv_counts, overlap_work)
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

    fn allreduce_sum_f64_slice(&self, buf: &mut [f64]) {
        let world = self.world();
        let sendbuf = buf.to_vec();
        world.all_reduce_into(&sendbuf[..], buf, SystemOperation::sum());
    }

    fn exchange_domain_by_x(&self, local: &mut Vec<Particle>, my_x_lo: f64, my_x_hi: f64) {
        let world = self.world();
        let rank = world.rank();
        let size = world.size();

        // Separar partículas que deben migrar a la izquierda, derecha, o quedarse.
        let mut stay = Vec::new();
        let mut go_left = Vec::new();
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
        let left = if rank > 0 {
            Some(world.process_at_rank(rank - 1))
        } else {
            None
        };
        let right = if rank < size - 1 {
            Some(world.process_at_rank(rank + 1))
        } else {
            None
        };

        let recv_from_right = point_to_point_exchange(
            &world,
            rank,
            &right,
            &left,
            &pack::pack_halo(&go_right),
            &pack::pack_halo(&go_left),
        );

        // Recombinar
        *local = stay;
        local.extend(recv_from_right.0);
        local.extend(recv_from_right.1);
    }

    fn exchange_domain_by_z(&self, local: &mut Vec<Particle>, my_z_lo: f64, my_z_hi: f64) {
        let world = self.world();
        let rank = world.rank();
        let size = world.size();

        let mut stay = Vec::new();
        let mut go_left = Vec::new();
        let mut go_right = Vec::new();
        for p in local.drain(..) {
            if p.position.z < my_z_lo && rank > 0 {
                go_left.push(p);
            } else if p.position.z >= my_z_hi && rank < size - 1 {
                go_right.push(p);
            } else {
                stay.push(p);
            }
        }

        let left = if rank > 0 {
            Some(world.process_at_rank(rank - 1))
        } else {
            None
        };
        let right = if rank < size - 1 {
            Some(world.process_at_rank(rank + 1))
        } else {
            None
        };

        let recv = point_to_point_exchange(
            &world,
            rank,
            &right,
            &left,
            &pack::pack_halo(&go_right),
            &pack::pack_halo(&go_left),
        );
        *local = stay;
        local.extend(recv.0);
        local.extend(recv.1);
    }

    fn exchange_halos_by_z(
        &self,
        local: &[Particle],
        my_z_lo: f64,
        my_z_hi: f64,
        halo_width: f64,
    ) -> Vec<Particle> {
        let world = self.world();
        let rank = world.rank();
        let size = world.size();

        let buf_left: Vec<Particle> = local
            .iter()
            .filter(|p| p.position.z < my_z_lo + halo_width && rank > 0)
            .cloned()
            .collect();
        let buf_right: Vec<Particle> = local
            .iter()
            .filter(|p| p.position.z > my_z_hi - halo_width && rank < size - 1)
            .cloned()
            .collect();

        let left = if rank > 0 {
            Some(world.process_at_rank(rank - 1))
        } else {
            None
        };
        let right = if rank < size - 1 {
            Some(world.process_at_rank(rank + 1))
        } else {
            None
        };

        let (from_left, from_right) = point_to_point_exchange(
            &world,
            rank,
            &right,
            &left,
            &pack::pack_halo(&buf_right),
            &pack::pack_halo(&buf_left),
        );
        let mut halos = from_left;
        halos.extend(from_right);
        halos
    }

    fn exchange_domain_sfc(
        &self,
        local: &mut Vec<Particle>,
        decomp: &crate::sfc::SfcDecomposition,
    ) {
        let rank = self.world().rank();
        let size = self.world().size() as usize;

        // Particionar en stay + buckets por destino (Alltoallv, sin restricción de vecinos).
        let (stay, leaves) = crate::sfc::partition_local(local, decomp, rank);

        let mut sends: Vec<Vec<f64>> = (0..size).map(|_| Vec::new()).collect();
        for (r, particles) in leaves {
            sends[r as usize] = pack::pack_halo(&particles);
        }

        let received = self.alltoallv_f64(&sends);

        *local = stay;
        for (r, data) in received.iter().enumerate() {
            if r != rank as usize && !data.is_empty() {
                local.extend(pack::unpack_halo(data));
            }
        }
    }

    fn exchange_halos_sfc(
        &self,
        local: &[Particle],
        _decomp: &crate::sfc::SfcDecomposition,
        halo_width: f64,
    ) -> Vec<Particle> {
        let rank = self.world().rank();
        let size = self.world().size() as usize;

        // Paso 1: calcular AABB local ajustada (6 f64).
        let my_aabb = compute_aabb(local);

        // Paso 2: Allgather las AABBs de todos los rangos.
        let all_aabbs = self.allgather_f64(&my_aabb);

        // Paso 3: para cada rank r, expandir su AABB por halo_width y enviar las
        // partículas propias que caen dentro de la AABB expandida.
        let mut sends: Vec<Vec<f64>> = (0..size).map(|_| Vec::new()).collect();
        for r in 0..size {
            if r == rank as usize {
                continue;
            }
            let a = &all_aabbs[r];
            let Some(expanded_r) = crate::halo3d::flat_aabb_expand_components(a, halo_width) else {
                continue;
            };
            // Poda conservadora: solo si la AABB remota es válida y no corta la nuestra
            // tras expandir por halo, podemos omitir el filtrado (no hay partículas que enviar).
            if crate::halo3d::flat_aabb_is_valid(a)
                && !crate::halo3d::flat_aabb_intersects(&my_aabb, &expanded_r)
            {
                continue;
            }
            let [rxlo, rxhi, rylo, ryhi, rzlo, rzhi] = expanded_r;
            let in_halo: Vec<Particle> = local
                .iter()
                .filter(|p| {
                    p.position.x >= rxlo
                        && p.position.x <= rxhi
                        && p.position.y >= rylo
                        && p.position.y <= ryhi
                        && p.position.z >= rzlo
                        && p.position.z <= rzhi
                })
                .cloned()
                .collect();
            if !in_halo.is_empty() {
                sends[r] = pack::pack_halo(&in_halo);
            }
        }

        // Paso 4: conteos + datos (P2P disperso para enteros si hay pares geométricos inactivos).
        let received = self.alltoallv_f64_halos_sfc(sends, &all_aabbs, halo_width, &mut || {});

        let mut halos = Vec::new();
        for (r, data) in received.iter().enumerate() {
            if r != rank as usize && !data.is_empty() {
                halos.extend(pack::unpack_halo(data));
            }
        }
        halos
    }

    fn allgather_f64(&self, local: &[f64]) -> Vec<Vec<f64>> {
        let world = self.world();
        let my_len = local.len() as Count;
        let mut counts = vec![0 as Count; world.size() as usize];
        world.all_gather_into(&[my_len][..], &mut counts[..]);

        let mut displs: Vec<Count> = Vec::with_capacity(world.size() as usize);
        let mut acc: Count = 0;
        for &c in &counts {
            displs.push(acc);
            acc += c;
        }
        let mut recvbuf = vec![0.0f64; acc as usize];
        {
            let mut part = PartitionMut::new(&mut recvbuf[..], counts.clone(), &displs[..]);
            world.all_gather_varcount_into(local, &mut part);
        }

        let mut result = Vec::with_capacity(world.size() as usize);
        let mut off = 0usize;
        for &c in &counts {
            let n = c as usize;
            result.push(recvbuf[off..off + n].to_vec());
            off += n;
        }
        result
    }

    fn alltoallv_f64(&self, sends: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let world = self.world();
        let size = world.size() as usize;
        assert_eq!(
            sends.len(),
            size,
            "alltoallv_f64: sends.len() debe ser igual a world.size()"
        );

        // Paso 1: intercambiar conteos via Alltoall (mensajes fijos de 1 Count por rango).
        let send_counts: Vec<Count> = sends.iter().map(|v| v.len() as Count).collect();
        let mut recv_counts = vec![0 as Count; size];
        world.all_to_all_into(&send_counts[..], &mut recv_counts[..]);

        // Paso 2: construir sendbuf y displacements.
        let mut sendbuf: Vec<f64> = Vec::new();
        let mut send_displs: Vec<Count> = Vec::with_capacity(size);
        let mut sdisp: Count = 0;
        for v in sends.iter() {
            send_displs.push(sdisp);
            sendbuf.extend_from_slice(v);
            sdisp += v.len() as Count;
        }

        // Paso 3: construir recvbuf y displacements.
        let mut recv_displs: Vec<Count> = Vec::with_capacity(size);
        let mut rdisp: Count = 0;
        for &c in &recv_counts {
            recv_displs.push(rdisp);
            rdisp += c;
        }
        let mut recvbuf = vec![0.0f64; rdisp as usize];

        // Paso 4: Alltoallv.
        {
            let send_part = Partition::new(&sendbuf[..], send_counts, &send_displs[..]);
            let mut recv_part =
                PartitionMut::new(&mut recvbuf[..], recv_counts.clone(), &recv_displs[..]);
            world.all_to_all_varcount_into(&send_part, &mut recv_part);
        }

        // Dividir el buffer recibido en subvectores por rango.
        let mut result = Vec::with_capacity(size);
        let mut off = 0usize;
        for &c in &recv_counts {
            let n = c as usize;
            result.push(recvbuf[off..off + n].to_vec());
            off += n;
        }
        result
    }

    fn alltoallv_f64_overlap(
        &self,
        sends: Vec<Vec<f64>>,
        overlap_work: &mut dyn FnMut(),
    ) -> Vec<Vec<f64>> {
        let world = self.world();
        let size = world.size() as usize;
        assert_eq!(
            sends.len(),
            size,
            "alltoallv_f64_overlap: sends.len() debe ser igual a world.size()"
        );

        let send_counts: Vec<Count> = sends.iter().map(|v| v.len() as Count).collect();
        let mut recv_counts = vec![0 as Count; size];
        world.all_to_all_into(&send_counts[..], &mut recv_counts[..]);

        self.alltoallv_f64_p2p_from_recv_counts(sends, recv_counts, overlap_work)
    }

    fn alltoallv_f64_subgroup(&self, sends: &[Vec<f64>], color: i32) -> Vec<Vec<f64>> {
        use mpi::topology::Color;

        let world = self.world();
        let sub_comm = world
            .split_by_color(Color::with_value(color))
            .expect("alltoallv_f64_subgroup: split_by_color devolvió None");

        let size = sub_comm.size() as usize;
        assert_eq!(
            sends.len(),
            size,
            "alltoallv_f64_subgroup: sends.len() ({}) debe ser igual al tamaño del subgrupo ({})",
            sends.len(),
            size
        );

        // Intercambiar conteos via Alltoall dentro del subcomunicador.
        let send_counts: Vec<Count> = sends.iter().map(|v| v.len() as Count).collect();
        let mut recv_counts = vec![0 as Count; size];
        sub_comm.all_to_all_into(&send_counts[..], &mut recv_counts[..]);

        // Construir sendbuf y displacements.
        let mut sendbuf: Vec<f64> = Vec::new();
        let mut send_displs: Vec<Count> = Vec::with_capacity(size);
        let mut sdisp: Count = 0;
        for v in sends.iter() {
            send_displs.push(sdisp);
            sendbuf.extend_from_slice(v);
            sdisp += v.len() as Count;
        }

        // Construir recvbuf y displacements.
        let mut recv_displs: Vec<Count> = Vec::with_capacity(size);
        let mut rdisp: Count = 0;
        for &c in &recv_counts {
            recv_displs.push(rdisp);
            rdisp += c;
        }
        let mut recvbuf = vec![0.0f64; rdisp as usize];

        {
            let send_part = Partition::new(&sendbuf[..], send_counts, &send_displs[..]);
            let mut recv_part =
                PartitionMut::new(&mut recvbuf[..], recv_counts.clone(), &recv_displs[..]);
            sub_comm.all_to_all_varcount_into(&send_part, &mut recv_part);
        }

        // Dividir buffer recibido en subvectores por rango dentro del subgrupo.
        let mut result = Vec::with_capacity(size);
        let mut off = 0usize;
        for &c in &recv_counts {
            let n = c as usize;
            result.push(recvbuf[off..off + n].to_vec());
            off += n;
        }
        result
    }

    fn exchange_halos_by_z_periodic(
        &self,
        local: &[Particle],
        my_z_lo: f64,
        my_z_hi: f64,
        halo_width: f64,
    ) -> Vec<Particle> {
        let world = self.world();
        let rank = world.rank() as usize;
        let size = world.size() as usize;

        if size == 1 {
            return Vec::new();
        }

        // Vecinos en anillo periódico.
        let left_rank = ((rank as i64 - 1).rem_euclid(size as i64)) as usize;
        let right_rank = (rank + 1) % size;

        // Partículas que son halo para el vecino izquierdo (z ∈ [z_lo, z_lo + halo_width)).
        let buf_left: Vec<Particle> = local
            .iter()
            .filter(|p| p.position.z < my_z_lo + halo_width)
            .cloned()
            .collect();

        // Partículas que son halo para el vecino derecho (z ∈ (z_hi - halo_width, z_hi]).
        let buf_right: Vec<Particle> = local
            .iter()
            .filter(|p| p.position.z > my_z_hi - halo_width)
            .cloned()
            .collect();

        // Intercambio via alltoallv (maneja el anillo periódico sin restricción de vecinos lineales).
        let mut sends: Vec<Vec<f64>> = (0..size).map(|_| Vec::new()).collect();
        sends[left_rank] = pack::pack_halo(&buf_left);
        sends[right_rank] = pack::pack_halo(&buf_right);

        let received = self.alltoallv_f64(&sends);

        let mut halos: Vec<Particle> = Vec::new();
        for (r, data) in received.iter().enumerate() {
            if r != rank && !data.is_empty() {
                halos.extend(pack::unpack_halo(data));
            }
        }
        halos
    }

    fn exchange_halos_3d_periodic(
        &self,
        local: &[Particle],
        box_size: f64,
        halo_width: f64,
    ) -> Vec<Particle> {
        let world = self.world();
        let rank = world.rank() as usize;
        let size = world.size() as usize;

        if size == 1 {
            return Vec::new();
        }

        // Paso 1: AABB real de las partículas locales (6 f64).
        let my_aabb = crate::halo3d::compute_aabb_3d(local);
        let my_aabb_data = crate::halo3d::aabb_to_f64(&my_aabb);

        // Paso 2: allgather de todas las AABBs (6 f64 × P).
        let all_aabbs = self.allgather_f64(&my_aabb_data);

        // Paso 3: para cada rank r, determinar qué partículas locales enviar.
        // Criterio: distancia 3D periódica de la partícula al AABB del rank r < halo_width.
        let halo_w2 = halo_width * halo_width;
        let mut sends: Vec<Vec<f64>> = (0..size).map(|_| Vec::new()).collect();

        for r in 0..size {
            if r == rank {
                continue;
            }
            let a = &all_aabbs[r];
            let aabb_r = match crate::halo3d::f64_to_aabb(a) {
                Some(ab) if ab.is_valid() => ab,
                _ => continue,
            };
            let candidates: Vec<Particle> = local
                .iter()
                .filter(|p| {
                    let pos = [p.position.x, p.position.y, p.position.z];
                    crate::halo3d::min_dist2_to_aabb_3d_periodic(pos, &aabb_r, box_size) < halo_w2
                })
                .cloned()
                .collect();
            if !candidates.is_empty() {
                sends[r] = pack::pack_halo(&candidates);
            }
        }

        // Paso 4: alltoallv + desempaquetado.
        let received = self.alltoallv_f64(&sends);

        let mut halos: Vec<Particle> = Vec::new();
        for (r, data) in received.iter().enumerate() {
            if r != rank && !data.is_empty() {
                halos.extend(pack::unpack_halo(data));
            }
        }
        halos
    }

    fn exchange_halos_by_x(
        &self,
        local: &[Particle],
        my_x_lo: f64,
        my_x_hi: f64,
        halo_width: f64,
    ) -> Vec<Particle> {
        let world = self.world();
        let rank = world.rank();
        let size = world.size();

        // Partículas que son halo para el vecino izquierdo y derecho.
        let send_left: Vec<&Particle> = local
            .iter()
            .filter(|p| p.position.x < my_x_lo + halo_width && rank > 0)
            .collect();
        let send_right: Vec<&Particle> = local
            .iter()
            .filter(|p| p.position.x > my_x_hi - halo_width && rank < size - 1)
            .collect();

        let left = if rank > 0 {
            Some(world.process_at_rank(rank - 1))
        } else {
            None
        };
        let right = if rank < size - 1 {
            Some(world.process_at_rank(rank + 1))
        } else {
            None
        };

        // Serializar (solo referencias → clonar los datos necesarios)
        let buf_to_right: Vec<Particle> = send_right.iter().map(|p| (*p).clone()).collect();
        let buf_to_left: Vec<Particle> = send_left.iter().map(|p| (*p).clone()).collect();

        let (from_left, from_right) = point_to_point_exchange(
            &world,
            rank,
            &right,
            &left,
            &pack::pack_halo(&buf_to_right),
            &pack::pack_halo(&buf_to_left),
        );
        let mut halos = from_left;
        halos.extend(from_right);
        halos
    }
}

// ── Utilidades locales ────────────────────────────────────────────────────────

/// Calcula la AABB ajustada de un slice de partículas como `[xlo, xhi, ylo, yhi, zlo, zhi]`.
///
/// Para partículas vacías devuelve infinitos (convención segura para allreduce).
fn compute_aabb(particles: &[Particle]) -> Vec<f64> {
    if particles.is_empty() {
        return vec![
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ];
    }
    let xlo = particles
        .iter()
        .map(|p| p.position.x)
        .fold(f64::INFINITY, f64::min);
    let xhi = particles
        .iter()
        .map(|p| p.position.x)
        .fold(f64::NEG_INFINITY, f64::max);
    let ylo = particles
        .iter()
        .map(|p| p.position.y)
        .fold(f64::INFINITY, f64::min);
    let yhi = particles
        .iter()
        .map(|p| p.position.y)
        .fold(f64::NEG_INFINITY, f64::max);
    let zlo = particles
        .iter()
        .map(|p| p.position.z)
        .fold(f64::INFINITY, f64::min);
    let zhi = particles
        .iter()
        .map(|p| p.position.z)
        .fold(f64::NEG_INFINITY, f64::max);
    vec![xlo, xhi, ylo, yhi, zlo, zhi]
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
    rank: i32,
    right: &Option<mpi::topology::Process<'_>>,
    left: &Option<mpi::topology::Process<'_>>,
    send_right: &[f64],
    send_left: &[f64],
) -> (Vec<Particle>, Vec<Particle>) {
    let mut from_left: Vec<f64> = Vec::new();
    let mut from_right: Vec<f64> = Vec::new();

    // ── Ronda 1: pares envían →derecha y reciben ←izquierda ──────────────────
    if rank % 2 == 0 {
        if let Some(r) = right {
            r.send(send_right);
        }
        if let Some(l) = left {
            let (v, _) = l.receive_vec::<f64>();
            from_left = v;
        }
    } else {
        if let Some(l) = left {
            let (v, _) = l.receive_vec::<f64>();
            from_left = v;
        }
        if let Some(r) = right {
            r.send(send_right);
        }
    }
    world.barrier();

    // ── Ronda 2: pares envían ←izquierda y reciben →derecha ──────────────────
    if rank % 2 == 0 {
        if let Some(l) = left {
            l.send(send_left);
        }
        if let Some(r) = right {
            let (v, _) = r.receive_vec::<f64>();
            from_right = v;
        }
    } else {
        if let Some(r) = right {
            let (v, _) = r.receive_vec::<f64>();
            from_right = v;
        }
        if let Some(l) = left {
            l.send(send_left);
        }
    }
    world.barrier();

    (
        pack::unpack_halo(&from_left),
        pack::unpack_halo(&from_right),
    )
}
