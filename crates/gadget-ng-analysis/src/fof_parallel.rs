//! FoF paralelo MPI usando intercambio de partículas frontera (Phase 61).
//!
//! ## Algoritmo
//!
//! 1. **FoF local**: cada rank ejecuta FoF sobre sus partículas propias.
//! 2. **Intercambio de halos**: las partículas dentro de una franja `b × l̄`
//!    alrededor de las fronteras del dominio SFC se intercambian con los vecinos
//!    vía [`ParallelRuntime::exchange_halos_sfc`].
//! 3. **FoF combinado**: se ejecuta [`find_halos_combined`] sobre el conjunto
//!    local + halos; solo se retienen grupos cuya raíz Union-Find es local.
//! 4. **Resultado**: cada rank emite su porción del catálogo sin duplicar halos.
//!
//! Con `SerialRuntime` (P=1) no hay intercambio y el resultado es idéntico al
//! FoF serial estándar.
//!
//! ## Ejemplo
//!
//! ```rust,no_run
//! use gadget_ng_analysis::fof_parallel::find_halos_parallel;
//! use gadget_ng_parallel::{SerialRuntime, SfcDecomposition};
//! use gadget_ng_core::{Vec3, Particle};
//!
//! let rt = SerialRuntime;
//! let positions = vec![Vec3::new(0.1, 0.1, 0.1), Vec3::new(0.9, 0.9, 0.9)];
//! let decomp = SfcDecomposition::build(&positions, 1.0, 1);
//! let particles: Vec<Particle> = (0..2)
//!     .map(|i| Particle::new(i, 1.0, positions[i], Vec3::zero()))
//!     .collect();
//! let halos = find_halos_parallel(&particles, &rt, &decomp, 1.0, 0.2, 2, 0.0);
//! ```

use gadget_ng_core::Particle;
use gadget_ng_parallel::{ParallelRuntime, sfc::SfcDecomposition};

use crate::fof::{FofHalo, find_halos, find_halos_combined};

/// Ejecuta FoF sobre las partículas locales usando intercambio MPI de frontera.
///
/// # Parámetros
/// - `local`: partículas propias del rank actual.
/// - `runtime`: entorno de ejecución (`SerialRuntime` o `MpiRuntime`).
/// - `decomp`: descomposición SFC activa (define fronteras de dominio).
/// - `box_size`: tamaño de la caja de simulación.
/// - `b`: parámetro de enlace FoF (típico 0.2).
/// - `min_particles`: número mínimo de partículas para un halo.
/// - `rho_crit`: densidad crítica (0 → usa `r_max` para `r_vir`).
///
/// # Retorna
/// Catálogo de halos locales al rank. Con `SerialRuntime` (P=1) el resultado es
/// idéntico al FoF serial estándar.
pub fn find_halos_parallel<R: ParallelRuntime>(
    local: &[Particle],
    runtime: &R,
    decomp: &SfcDecomposition,
    box_size: f64,
    b: f64,
    min_particles: usize,
    rho_crit: f64,
) -> Vec<FofHalo> {
    if runtime.size() == 1 {
        // Camino rápido: sin MPI, equivalente al FoF serial.
        let positions: Vec<_> = local.iter().map(|p| p.position).collect();
        let velocities: Vec<_> = local.iter().map(|p| p.velocity).collect();
        let masses: Vec<_> = local.iter().map(|p| p.mass).collect();
        return find_halos(
            &positions,
            &velocities,
            &masses,
            box_size,
            b,
            min_particles,
            rho_crit,
        );
    }

    // Calcular la longitud de enlace basada en el número total de partículas.
    let n_local_f = local.len() as f64;
    let n_total = runtime.allreduce_sum_f64(n_local_f) as usize;
    let l_mean = (box_size * box_size * box_size / n_total.max(1) as f64).cbrt();
    let ll = b * l_mean;

    // Intercambiar partículas dentro de la franja `ll` alrededor de las fronteras SFC.
    let halo_particles = runtime.exchange_halos_sfc(local, decomp, ll);

    let n_local_count = local.len();
    let mut all_positions = Vec::with_capacity(n_local_count + halo_particles.len());
    let mut all_velocities = Vec::with_capacity(n_local_count + halo_particles.len());
    let mut all_masses = Vec::with_capacity(n_local_count + halo_particles.len());

    for p in local.iter().chain(halo_particles.iter()) {
        all_positions.push(p.position);
        all_velocities.push(p.velocity);
        all_masses.push(p.mass);
    }

    find_halos_combined(
        &all_positions,
        &all_velocities,
        &all_masses,
        n_local_count,
        box_size,
        b,
        min_particles,
        rho_crit,
    )
}
