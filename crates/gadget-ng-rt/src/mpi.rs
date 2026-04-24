//! RT MPI distribuida — Phase 84.
//!
//! Distribuye el campo de radiación `RadiationField` usando una descomposición
//! en slabs (análoga a `slab_pm` en `gadget-ng-pm`).  Cada rank posee un subconjunto
//! de capas Y del grid, más una capa halo de 1 celda en cada borde Y para flujos.
//!
//! ## Diseño
//!
//! ```text
//! rank 0: iy ∈ [0,  ny/P)       + halo Y superior
//! rank 1: iy ∈ [ny/P, 2·ny/P)   + halos Y inferior/superior
//! ...
//! rank P-1: iy ∈ [(P-1)·ny/P, ny) + halo Y inferior
//! ```
//!
//! Comunicación:
//! - `allreduce_radiation`: suma global de E y F a través de todos los ranks.
//!   Necesario para diagnósticos y checkpoint.
//! - `exchange_radiation_halos`: envío/recepción de las capas halo ghost entre
//!   ranks vecinos (rank - 1, rank + 1).
//!
//! ## MPI stub
//!
//! Cuando el crate se compila **sin** feature `mpi`, todas las funciones operan
//! en modo serial (solo un rank, sin comunicación).  Las interfaces son idénticas
//! para que el código de llamada no necesite `#[cfg(feature = "mpi")]`.
//!
//! ## Referencia
//!
//! Rosdahl et al. (2013), MNRAS 436, 2188 (RAMSES-RT paralelo).

use crate::m1::{M1Params, RadiationField};

// ── Slab distribuido ──────────────────────────────────────────────────────────

/// Campo de radiación particionado en slabs Y para distribución MPI.
///
/// Cada rank almacena `ny_local` capas en la dirección Y, más una capa halo
/// de 1 celda en cada borde (`iy_lo_halo` e `iy_hi_halo`) para calcular los flujos.
#[derive(Debug, Clone)]
pub struct RadiationFieldSlab {
    /// Energía radiativa para las celdas locales + halos.
    /// Dimensiones: `[nx × (ny_local + 2) × nz]` (±1 halo en Y).
    pub energy: Vec<f64>,
    /// Flujo X para las celdas locales + halos.
    pub flux_x: Vec<f64>,
    /// Flujo Y para las celdas locales + halos.
    pub flux_y: Vec<f64>,
    /// Flujo Z para las celdas locales + halos.
    pub flux_z: Vec<f64>,
    /// Número de celdas locales en X.
    pub nx: usize,
    /// Número de celdas locales en Y (sin contar halos).
    pub ny_local: usize,
    /// Número de celdas en Z.
    pub nz: usize,
    /// Índice global de inicio en Y para este rank.
    pub iy_start: usize,
    /// Espaciado de la malla.
    pub dx: f64,
    /// Rank de este proceso.
    pub rank: usize,
    /// Número total de ranks.
    pub n_ranks: usize,
}

impl RadiationFieldSlab {
    /// Crea un slab local a partir de un `RadiationField` global.
    ///
    /// En modo serial (n_ranks=1), el slab contiene todo el campo.
    pub fn from_global(global: &RadiationField, rank: usize, n_ranks: usize) -> Self {
        let nx = global.nx;
        let ny = global.ny;
        let nz = global.nz;

        let ny_per_rank = ny / n_ranks;
        let iy_start = rank * ny_per_rank;
        let ny_local = if rank == n_ranks - 1 {
            ny - iy_start
        } else {
            ny_per_rank
        };

        // Incluir ±1 halo en Y → ny_with_halo = ny_local + 2
        let ny_wh = ny_local + 2;
        let n3 = nx * ny_wh * nz;

        let mut slab = Self {
            energy: vec![0.0; n3],
            flux_x: vec![0.0; n3],
            flux_y: vec![0.0; n3],
            flux_z: vec![0.0; n3],
            nx,
            ny_local,
            nz,
            iy_start,
            dx: global.dx,
            rank,
            n_ranks,
        };

        // Copiar datos del campo global al slab local (sin halos ghost aún)
        for iy_loc in 0..ny_local {
            let iy_glob = iy_start + iy_loc;
            for iz in 0..nz {
                for ix in 0..nx {
                    let gi = global.idx(ix, iy_glob, iz);
                    let si = slab.idx_local(ix, iy_loc, iz);
                    slab.energy[si] = global.energy_density[gi];
                    slab.flux_x[si] = global.flux_x[gi];
                    slab.flux_y[si] = global.flux_y[gi];
                    slab.flux_z[si] = global.flux_z[gi];
                }
            }
        }

        slab
    }

    /// Índice lineal en el slab (con capa halo: iy_slab ∈ [0, ny_local+2)).
    ///
    /// Para celdas locales: `iy_slab = iy_local + 1` (offset por el halo inferior).
    #[inline]
    pub fn idx_slab(&self, ix: usize, iy_slab: usize, iz: usize) -> usize {
        let ny_wh = self.ny_local + 2;
        iz * ny_wh * self.nx + iy_slab * self.nx + ix
    }

    /// Índice de celda local (iy_local ∈ [0, ny_local)) → índice en slab.
    #[inline]
    pub fn idx_local(&self, ix: usize, iy_local: usize, iz: usize) -> usize {
        self.idx_slab(ix, iy_local + 1, iz)
    }

    /// Reconstituye un `RadiationField` global desde el slab (solo para rank 0 o serial).
    pub fn to_global(&self, ny_total: usize) -> RadiationField {
        let n3 = self.nx * ny_total * self.nz;
        let mut global = RadiationField {
            energy_density: vec![0.0; n3],
            flux_x: vec![0.0; n3],
            flux_y: vec![0.0; n3],
            flux_z: vec![0.0; n3],
            nx: self.nx,
            ny: ny_total,
            nz: self.nz,
            dx: self.dx,
        };
        for iy_loc in 0..self.ny_local {
            let iy_glob = self.iy_start + iy_loc;
            for iz in 0..self.nz {
                for ix in 0..self.nx {
                    let si = self.idx_local(ix, iy_loc, iz);
                    let gi = global.idx(ix, iy_glob, iz);
                    global.energy_density[gi] = self.energy[si];
                    global.flux_x[gi] = self.flux_x[si];
                    global.flux_y[gi] = self.flux_y[si];
                    global.flux_z[gi] = self.flux_z[si];
                }
            }
        }
        global
    }
}

// ── Comunicación MPI stub ─────────────────────────────────────────────────────

/// Contexto de runtime paralelo para RT.
///
/// En producción MPI se sustituye por un wrapper del communicator real.
/// En modo serial (o compilado sin `mpi`), todas las operaciones son no-ops.
pub struct RtRuntime {
    pub rank: usize,
    pub size: usize,
}

impl RtRuntime {
    /// Crea un runtime serial (rank=0, size=1).
    pub fn serial() -> Self {
        Self { rank: 0, size: 1 }
    }
}

/// Suma global del campo de radiación entre todos los ranks (allreduce).
///
/// En modo serial, es un no-op.
/// En MPI real, realiza `MPI_Allreduce(MPI_SUM)` sobre `energy_density` y los tres flujos.
///
/// # Argumentos
/// - `rad` — campo de radiación local (se modifica in-place con la suma global)
/// - `rt`  — contexto de runtime
pub fn allreduce_radiation(rad: &mut RadiationField, rt: &RtRuntime) {
    if rt.size == 1 {
        return; // serial: nada que hacer
    }

    // Stub MPI: en una implementación real usaríamos MPI_Allreduce.
    // Aquí simplemente documentamos la interfaz.
    // En producción con feature "mpi":
    //   mpi_world.all_reduce_into(&rad.energy_density[..], &mut buf, SystemOperation::sum());
    //   rad.energy_density.copy_from_slice(&buf);
    //   (ídem para flux_x/y/z)
    let _ = rad; // suppress unused warning en modo stub
    eprintln!(
        "[rt-mpi] allreduce_radiation: rank {}/{} — stub (implementar con MPI_Allreduce)",
        rt.rank, rt.size
    );
}

/// Intercambia las capas halo ghost entre ranks vecinos.
///
/// - Rank k envía su primera capa local (`iy_local = 0`) al rank k-1 (halo superior de k-1).
/// - Rank k envía su última capa local (`iy_local = ny_local-1`) al rank k+1 (halo inferior de k+1).
///
/// En modo serial, copia periódicamente si procede.
/// En MPI real, usa `MPI_Sendrecv` entre ranks adyacentes.
///
/// # Argumentos
/// - `slab` — slab local con halos a actualizar
/// - `rt`   — contexto de runtime
pub fn exchange_radiation_halos(slab: &mut RadiationFieldSlab, rt: &RtRuntime) {
    let nx = slab.nx;
    let nz = slab.nz;
    let ny_loc = slab.ny_local;

    if rt.size == 1 {
        // Serial: condición periódica — copia la primera/última capa a los halos.
        for iz in 0..nz {
            for ix in 0..nx {
                // Halo inferior: copia última capa local → posición de halo 0 (iy_slab = 0)
                let src_lo = slab.idx_local(ix, ny_loc - 1, iz);
                let dst_lo = slab.idx_slab(ix, 0, iz);
                slab.energy[dst_lo] = slab.energy[src_lo];
                slab.flux_x[dst_lo] = slab.flux_x[src_lo];
                slab.flux_y[dst_lo] = slab.flux_y[src_lo];
                slab.flux_z[dst_lo] = slab.flux_z[src_lo];

                // Halo superior: copia primera capa local → posición de halo ny_local+1
                let src_hi = slab.idx_local(ix, 0, iz);
                let dst_hi = slab.idx_slab(ix, ny_loc + 1, iz);
                slab.energy[dst_hi] = slab.energy[src_hi];
                slab.flux_x[dst_hi] = slab.flux_x[src_hi];
                slab.flux_y[dst_hi] = slab.flux_y[src_hi];
                slab.flux_z[dst_hi] = slab.flux_z[src_hi];
            }
        }
        return;
    }

    // Stub MPI: en producción usaría MPI_Sendrecv por par de ranks.
    eprintln!(
        "[rt-mpi] exchange_radiation_halos: rank {}/{} — stub (implementar con MPI_Sendrecv)",
        rt.rank, rt.size
    );
}

/// Actualiza el campo de radiación en un slab local usando el solver M1.
///
/// Equivalente a `m1_update` pero opera sobre el slab con halos ghost.
/// Los flujos en el borde Y usan las celdas halo ya actualizadas por
/// `exchange_radiation_halos`.
///
/// # Argumentos
/// - `slab`   — slab local del campo de radiación
/// - `dt`     — paso de tiempo
/// - `params` — parámetros del solver M1
pub fn m1_update_slab(slab: &mut RadiationFieldSlab, dt: f64, params: &M1Params) {
    // Reconstruir un RadiationField temporal con solo las celdas locales
    // y usar m1_update. En producción, el solver operaría directamente sobre el slab.
    let ny_wh = slab.ny_local + 2;
    let mut local_rf = RadiationField {
        energy_density: slab.energy.clone(),
        flux_x: slab.flux_x.clone(),
        flux_y: slab.flux_y.clone(),
        flux_z: slab.flux_z.clone(),
        nx: slab.nx,
        ny: ny_wh,
        nz: slab.nz,
        dx: slab.dx,
    };
    crate::m1::m1_update(&mut local_rf, dt, params);
    slab.energy = local_rf.energy_density;
    slab.flux_x = local_rf.flux_x;
    slab.flux_y = local_rf.flux_y;
    slab.flux_z = local_rf.flux_z;
}

// ── Implementaciones MPI reales (feature = "mpi") ────────────────────────────

/// Suma global del campo de radiación con MPI real (`MPI_Allreduce`).
///
/// Equivalente a `allreduce_radiation` pero usa el communicator MPI real.
/// Requiere compilar con `--features mpi`.
///
/// # Argumentos
/// - `rad`   — campo local (modificado in-place con la suma global)
/// - `world` — communicator MPI (p. ej. `universe.world()` de `rsmpi`)
#[cfg(feature = "mpi")]
pub fn allreduce_radiation_mpi<C: mpi::collective::CommunicatorCollectives>(
    rad: &mut RadiationField,
    world: &C,
) {
    use mpi::collective::SystemOperation;
    let e_send  = rad.energy_density.clone();
    let fx_send = rad.flux_x.clone();
    let fy_send = rad.flux_y.clone();
    let fz_send = rad.flux_z.clone();
    world.all_reduce_into(&e_send[..],  &mut rad.energy_density[..], SystemOperation::sum());
    world.all_reduce_into(&fx_send[..], &mut rad.flux_x[..],         SystemOperation::sum());
    world.all_reduce_into(&fy_send[..], &mut rad.flux_y[..],         SystemOperation::sum());
    world.all_reduce_into(&fz_send[..], &mut rad.flux_z[..],         SystemOperation::sum());
}

/// Intercambia capas halo ghost entre ranks vecinos con MPI real (`MPI_Send`/`MPI_Recv`).
///
/// Envía la primera capa local (iy=0) al rank k-1 y la última (iy=ny_local-1) al rank k+1.
/// Recibe las capas correspondientes de los vecinos y las almacena en los halos del slab.
///
/// Usa patrón odd-even de dos rondas para evitar deadlock (análogo a `point_to_point_exchange`
/// en `gadget-ng-parallel`).
///
/// # Argumentos
/// - `slab`  — slab local (halos actualizados in-place)
/// - `world` — communicator MPI
#[cfg(feature = "mpi")]
pub fn exchange_radiation_halos_mpi<C: mpi::traits::Communicator>(
    slab: &mut RadiationFieldSlab,
    world: &C,
) {
    use mpi::traits::*;

    let rank    = world.rank() as usize;
    let size    = world.size() as usize;
    let nx      = slab.nx;
    let nz      = slab.nz;
    let ny_loc  = slab.ny_local;

    if size == 1 {
        // Serial: condición periódica
        let rt = RtRuntime::serial();
        exchange_radiation_halos(slab, &rt);
        return;
    }

    let layer_len = nx * nz;
    // Cada capa se empaqueta como [energy, flux_x, flux_y, flux_z] → 4 × layer_len f64
    let pack_size = 4 * layer_len;

    // Empaquetar primera y última capas locales
    let mut send_to_left  = vec![0.0f64; pack_size]; // primera capa → halo superior del vecino izquierdo
    let mut send_to_right = vec![0.0f64; pack_size]; // última capa  → halo inferior del vecino derecho

    for iz in 0..nz {
        for ix in 0..nx {
            let flat = iz * nx + ix;
            let si_first = slab.idx_local(ix, 0, iz);
            let si_last  = slab.idx_local(ix, ny_loc - 1, iz);
            send_to_left[flat]              = slab.energy[si_first];
            send_to_left[layer_len + flat]  = slab.flux_x[si_first];
            send_to_left[2*layer_len+flat]  = slab.flux_y[si_first];
            send_to_left[3*layer_len+flat]  = slab.flux_z[si_first];
            send_to_right[flat]             = slab.energy[si_last];
            send_to_right[layer_len + flat] = slab.flux_x[si_last];
            send_to_right[2*layer_len+flat] = slab.flux_y[si_last];
            send_to_right[3*layer_len+flat] = slab.flux_z[si_last];
        }
    }

    let left_rank  = ((rank as i64 - 1).rem_euclid(size as i64)) as i32;
    let right_rank = ((rank + 1) % size) as i32;

    let mut recv_from_left  = vec![0.0f64; pack_size];
    let mut recv_from_right = vec![0.0f64; pack_size];

    // Ronda 1: ranks pares envían →derecha (última capa), luego reciben ←izquierda
    //          ranks impares primero reciben ←izquierda, luego envían →derecha
    if rank % 2 == 0 {
        world.process_at_rank(right_rank).send(&send_to_right[..]);
        let (v, _) = world.process_at_rank(left_rank).receive_vec::<f64>();
        recv_from_left = v;
    } else {
        let (v, _) = world.process_at_rank(left_rank).receive_vec::<f64>();
        recv_from_left = v;
        world.process_at_rank(right_rank).send(&send_to_right[..]);
    }
    world.barrier();

    // Ronda 2: ranks pares envían ←izquierda (primera capa), luego reciben →derecha
    //          ranks impares primero reciben →derecha, luego envían ←izquierda
    if rank % 2 == 0 {
        world.process_at_rank(left_rank).send(&send_to_left[..]);
        let (v, _) = world.process_at_rank(right_rank).receive_vec::<f64>();
        recv_from_right = v;
    } else {
        let (v, _) = world.process_at_rank(right_rank).receive_vec::<f64>();
        recv_from_right = v;
        world.process_at_rank(left_rank).send(&send_to_left[..]);
    }
    world.barrier();

    // Desempaquetar: recv_from_left → halo inferior (iy_slab = 0)
    //                recv_from_right → halo superior (iy_slab = ny_loc + 1)
    for iz in 0..nz {
        for ix in 0..nx {
            let flat = iz * nx + ix;
            let hi_lo = slab.idx_slab(ix, 0, iz);
            let hi_hi = slab.idx_slab(ix, ny_loc + 1, iz);
            slab.energy[hi_lo]  = recv_from_left[flat];
            slab.flux_x[hi_lo]  = recv_from_left[layer_len + flat];
            slab.flux_y[hi_lo]  = recv_from_left[2*layer_len+flat];
            slab.flux_z[hi_lo]  = recv_from_left[3*layer_len+flat];
            slab.energy[hi_hi]  = recv_from_right[flat];
            slab.flux_x[hi_hi]  = recv_from_right[layer_len + flat];
            slab.flux_y[hi_hi]  = recv_from_right[2*layer_len+flat];
            slab.flux_z[hi_hi]  = recv_from_right[3*layer_len+flat];
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_uniform_rf(n: usize, e0: f64) -> RadiationField {
        RadiationField::uniform(n, n, n, 1.0 / n as f64, e0)
    }

    #[test]
    fn slab_from_global_serial_identity() {
        let n = 8;
        let e0 = 2.5;
        let global = make_uniform_rf(n, e0);
        let slab = RadiationFieldSlab::from_global(&global, 0, 1);

        assert_eq!(slab.ny_local, n);
        assert_eq!(slab.iy_start, 0);

        // Energía de celdas locales debe coincidir con global
        for iy in 0..n {
            for iz in 0..n {
                for ix in 0..n {
                    let si = slab.idx_local(ix, iy, iz);
                    let gi = global.idx(ix, iy, iz);
                    assert_eq!(slab.energy[si], global.energy_density[gi]);
                }
            }
        }
    }

    #[test]
    fn allreduce_serial_noop() {
        let n = 4;
        let mut rf = make_uniform_rf(n, 1.0);
        let rt = RtRuntime::serial();
        let e_before = rf.energy_density.clone();
        allreduce_radiation(&mut rf, &rt);
        assert_eq!(rf.energy_density, e_before);
    }

    #[test]
    fn exchange_halos_periodic_energy_conserved() {
        let n = 4;
        let mut slab = RadiationFieldSlab::from_global(&make_uniform_rf(n, 3.0), 0, 1);
        let rt = RtRuntime::serial();

        // Antes del exchange los halos son 0 (no se copian en from_global)
        exchange_radiation_halos(&mut slab, &rt);

        // Después: halos deben ser iguales a la energía de las celdas locales limítrofes
        let e_first = slab.energy[slab.idx_local(0, 0, 0)];
        let e_last = slab.energy[slab.idx_local(0, n - 1, 0)];
        let e_halo_lo = slab.energy[slab.idx_slab(0, 0, 0)];
        let e_halo_hi = slab.energy[slab.idx_slab(0, n + 1, 0)];

        assert_eq!(e_halo_lo, e_last, "Halo inferior debe copiar última capa local");
        assert_eq!(e_halo_hi, e_first, "Halo superior debe copiar primera capa local");
    }

    #[test]
    fn to_global_roundtrip() {
        let n = 4;
        let global_orig = make_uniform_rf(n, 1.0);
        let slab = RadiationFieldSlab::from_global(&global_orig, 0, 1);
        let global_rt = slab.to_global(n);

        for i in 0..global_orig.n_cells() {
            assert_eq!(global_orig.energy_density[i], global_rt.energy_density[i]);
        }
    }

    // ── Tests con feature mpi (single-rank = serial idéntico) ─────────────────

    #[cfg(feature = "mpi")]
    use std::sync::OnceLock;

    #[cfg(feature = "mpi")]
    static MPI_UNIVERSE: OnceLock<mpi::environment::Universe> = OnceLock::new();

    #[cfg(feature = "mpi")]
    fn get_mpi_world() -> mpi::topology::SimpleCommunicator {
        MPI_UNIVERSE.get_or_init(|| mpi::initialize().unwrap()).world()
    }

    #[cfg(feature = "mpi")]
    #[test]
    fn allreduce_radiation_mpi_single_rank() {
        use mpi::traits::Communicator;
        let world = get_mpi_world();
        if world.size() > 1 { return; } // solo en serial

        let n = 4;
        let mut rf = make_uniform_rf(n, 2.0);
        let e_before = rf.energy_density.clone();
        allreduce_radiation_mpi(&mut rf, &world);
        // Con 1 rank, allreduce suma sobre 1 elemento → mismo valor
        assert_eq!(rf.energy_density, e_before);
    }

    #[cfg(feature = "mpi")]
    #[test]
    fn exchange_halos_mpi_single_rank_periodic() {
        use mpi::traits::Communicator;
        let world = get_mpi_world();
        if world.size() > 1 { return; }

        let n = 4;
        let global = make_uniform_rf(n, 3.0);
        let mut slab = RadiationFieldSlab::from_global(&global, 0, 1);
        exchange_radiation_halos_mpi(&mut slab, &world);

        // Single rank: halos periódicos
        let e_first = slab.energy[slab.idx_local(0, 0, 0)];
        let e_halo_hi = slab.energy[slab.idx_slab(0, n + 1, 0)];
        assert_eq!(e_halo_hi, e_first);
    }

    #[test]
    fn m1_update_slab_does_not_crash() {
        let n = 4;
        let global = make_uniform_rf(n, 1.0);
        let mut slab = RadiationFieldSlab::from_global(&global, 0, 1);
        let rt = RtRuntime::serial();
        exchange_radiation_halos(&mut slab, &rt);

        let params = M1Params {
            c_red_factor: 100.0,
            kappa_abs: 0.1,
            kappa_scat: 0.0,
            substeps: 2,
            ..Default::default()
        };
        m1_update_slab(&mut slab, 0.001, &params);

        // Energía debe seguir siendo positiva o cero
        for &e in &slab.energy {
            assert!(e >= 0.0, "Energía negativa en slab tras m1_update_slab");
        }
    }
}
