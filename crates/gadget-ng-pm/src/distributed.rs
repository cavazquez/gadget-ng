//! PM distribuido sin allgather de partículas (Fase 19).
//!
//! Expone las tres fases del pipeline PM como funciones independientes,
//! permitiendo que el motor inserte un `allreduce_sum_f64_slice` entre
//! el depósito local y el solve de Poisson:
//!
//! ```text
//! 1. deposit_local(local_pos, local_mass, box_size, nm)
//!      → local_density[nm³]          (O(N/P) por rank)
//!
//! 2. rt.allreduce_sum_f64_slice(&mut density)
//!      → global_density[nm³]         (O(nm³), independiente de N)
//!
//! 3. forces_from_global_density(density, g, nm, box_size)
//!      → [fx, fy, fz][nm³]           (determinista: mismo resultado en todos los ranks)
//!
//! 4. interpolate_local(local_pos, fx, fy, fz, nm, box_size)
//!      → local_accels                (O(N/P) por rank)
//! ```
//!
//! La comunicación pasa de **O(N·P)** (allgather de partículas) a
//! **O(nm³)** (allreduce del grid de densidad), que es independiente de N.
//!
//! ## Correctitud
//!
//! Como el solve de Poisson es determinista y todos los ranks tienen el mismo
//! `global_density` tras el allreduce, cada rank puede resolver Poisson de
//! forma independiente y obtener el mismo `force_field`. No se necesita
//! comunicación adicional entre el solve y la interpolación.

use crate::{cic, fft_poisson};
use gadget_ng_core::Vec3;

/// Fase 1 — Depósito CIC local.
///
/// Cada rank llama esta función con sus partículas locales. El resultado es
/// la contribución parcial de este rank al grid global de densidad.
/// Tras sumar las contribuciones de todos los ranks con `allreduce_sum_f64_slice`,
/// el grid contendrá la densidad global correcta.
///
/// - `positions` — posiciones de las partículas locales (en `[0, box_size)`).
/// - `masses`    — masas de las partículas locales.
/// - `box_size`  — longitud del cubo periódico.
/// - `nm`        — número de celdas por lado del grid.
///
/// Devuelve `density[nm³]` con unidades masa/celda (contribución local).
pub fn deposit_local(positions: &[Vec3], masses: &[f64], box_size: f64, nm: usize) -> Vec<f64> {
    cic::assign(positions, masses, box_size, nm)
}

/// Fase 3 — Solve de Poisson periódico y derivación de fuerzas.
///
/// Toma la densidad **global** (ya reducida por `allreduce_sum_f64_slice`)
/// y devuelve las tres componentes de la fuerza gravitacional en el grid.
///
/// La física es idéntica al solver PM existente. Esta función es un thin
/// wrapper sobre `fft_poisson::solve_forces` para que el motor pueda
/// intercalar el allreduce sin modificar `PmSolver`.
///
/// - `density`  — grid de densidad global `nm³` (masa/celda).
/// - `g`        — constante gravitacional efectiva (usar `g_cosmo = G/a` en cosmo).
/// - `nm`       — número de celdas por lado.
/// - `box_size` — longitud del cubo periódico.
///
/// Devuelve `[fx_grid, fy_grid, fz_grid]` de longitud `nm³` cada uno.
pub fn forces_from_global_density(
    density: &[f64],
    g: f64,
    nm: usize,
    box_size: f64,
) -> [Vec<f64>; 3] {
    fft_poisson::solve_forces(density, g, nm, box_size)
}

/// Fase 4 — Interpolación CIC de fuerzas del grid a partículas locales.
///
/// Cada rank interpola el `force_field` global (que es idéntico en todos los
/// ranks tras el solve determinista) para sus propias partículas locales.
///
/// - `positions`           — posiciones de las partículas locales.
/// - `fx`, `fy`, `fz`     — grids de fuerza (longitud `nm³` cada uno).
/// - `nm`                  — número de celdas por lado.
/// - `box_size`            — longitud del cubo periódico.
///
/// Devuelve las aceleraciones interpoladas (una por partícula local).
pub fn interpolate_local(
    positions: &[Vec3],
    fx: &[f64],
    fy: &[f64],
    fz: &[f64],
    nm: usize,
    box_size: f64,
) -> Vec<Vec3> {
    cic::interpolate(fx, fy, fz, positions, box_size, nm)
}

#[cfg(test)]
mod tests {
    use super::*;
    use gadget_ng_core::Vec3;

    #[test]
    fn deposit_local_matches_full_assign() {
        let nm = 8usize;
        let box_size = 1.0_f64;
        let pos = vec![
            Vec3::new(0.1, 0.2, 0.3),
            Vec3::new(0.7, 0.5, 0.9),
            Vec3::new(0.4, 0.4, 0.4),
        ];
        let mass = vec![1.0_f64, 2.0, 0.5];

        let via_deposit = deposit_local(&pos, &mass, box_size, nm);
        let via_assign = cic::assign(&pos, &mass, box_size, nm);

        assert_eq!(via_deposit.len(), via_assign.len());
        for (a, b) in via_deposit.iter().zip(via_assign.iter()) {
            assert!((a - b).abs() < 1e-15, "deposit_local != assign: {a} vs {b}");
        }
    }

    #[test]
    fn forces_wrapper_matches_solve_forces() {
        let nm = 8usize;
        let box_size = 1.0_f64;
        let g = 1.0_f64;
        let pos = vec![Vec3::new(0.25, 0.25, 0.25)];
        let mass = vec![1.0_f64];
        let density = cic::assign(&pos, &mass, box_size, nm);

        let via_wrapper = forces_from_global_density(&density, g, nm, box_size);
        let via_direct = fft_poisson::solve_forces(&density, g, nm, box_size);

        for c in 0..3 {
            for (a, b) in via_wrapper[c].iter().zip(via_direct[c].iter()) {
                assert!(
                    (a - b).abs() < 1e-15,
                    "forces_from_global_density != solve_forces: {a} vs {b}"
                );
            }
        }
    }

    #[test]
    fn interpolate_local_wrapper_matches_interpolate() {
        let nm = 8usize;
        let box_size = 1.0_f64;
        let pos = vec![Vec3::new(0.1, 0.5, 0.8)];
        let mass = vec![1.0_f64];
        let density = cic::assign(&pos, &mass, box_size, nm);
        let [fx, fy, fz] = fft_poisson::solve_forces(&density, 1.0, nm, box_size);

        let via_wrapper = interpolate_local(&pos, &fx, &fy, &fz, nm, box_size);
        let via_direct = cic::interpolate(&fx, &fy, &fz, &pos, box_size, nm);

        assert_eq!(via_wrapper.len(), via_direct.len());
        for (a, b) in via_wrapper.iter().zip(via_direct.iter()) {
            assert!((a.x - b.x).abs() < 1e-15);
            assert!((a.y - b.y).abs() < 1e-15);
            assert!((a.z - b.z).abs() < 1e-15);
        }
    }

    #[test]
    fn distributed_pipeline_mass_conservation() {
        // Simula dos ranks: cada uno deposita su mitad de partículas,
        // luego suma los grids (simula allreduce) y verifica conservación.
        let nm = 8usize;
        let box_size = 1.0_f64;
        let pos_r0 = vec![Vec3::new(0.1, 0.2, 0.3), Vec3::new(0.9, 0.9, 0.9)];
        let mass_r0 = vec![1.0_f64, 2.0];
        let pos_r1 = vec![Vec3::new(0.5, 0.5, 0.5), Vec3::new(0.3, 0.7, 0.1)];
        let mass_r1 = vec![3.0_f64, 0.5];

        let mut grid_r0 = deposit_local(&pos_r0, &mass_r0, box_size, nm);
        let grid_r1 = deposit_local(&pos_r1, &mass_r1, box_size, nm);

        // Simula allreduce_sum_f64_slice
        for (a, b) in grid_r0.iter_mut().zip(grid_r1.iter()) {
            *a += b;
        }

        let total_mass_grid: f64 = grid_r0.iter().sum();
        let total_mass_particles: f64 = mass_r0.iter().chain(mass_r1.iter()).sum();
        assert!(
            (total_mass_grid - total_mass_particles).abs() < 1e-12,
            "masa en grid={total_mass_grid}, masa partículas={total_mass_particles}"
        );
    }
}
