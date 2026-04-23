//! `PmSolver`: implementación de `GravitySolver` basada en Particle-Mesh periódico 3D.
use gadget_ng_core::{GravitySolver, Vec3};

use crate::{cic, fft_poisson};

/// Solver de gravedad Particle-Mesh (PM) con condiciones de contorno periódicas.
///
/// Utiliza CIC (Cloud-in-Cell) para asignar masa al grid y FFT para resolver
/// la ecuación de Poisson en k-space. Las fuerzas se interpolan de vuelta a
/// las partículas con el mismo esquema CIC.
///
/// El costo es O(N + NM³ log NM) por evaluación, donde NM = `grid_size`.
/// Para NM ~ N^(1/3) la operación es O(N log N).
///
/// ## Configuración TOML
///
/// ```toml
/// [gravity]
/// solver = "pm"
/// pm_grid_size = 64   # potencia de 2 recomendada
/// ```
#[derive(Debug, Clone)]
pub struct PmSolver {
    /// Número de celdas por lado del grid (NM). El grid total es NM³ celdas.
    /// Se recomienda que sea potencia de 2 para eficiencia FFT.
    pub grid_size: usize,
    /// Longitud del cubo periódico en las mismas unidades que las posiciones.
    pub box_size: f64,
}

impl GravitySolver for PmSolver {
    /// Calcula aceleraciones para el subconjunto `global_indices` usando todas
    /// las posiciones y masas globales para construir el campo de densidad PM.
    ///
    /// El softening `eps2` y la constante gravitacional `g` se usan directamente:
    /// `g` escala el potencial PM; `eps2` no se aplica explícitamente (el PM ya
    /// tiene suavizado natural a la escala de la celda `box_size / grid_size`).
    fn accelerations_for_indices(
        &self,
        global_positions: &[Vec3],
        global_masses: &[f64],
        _eps2: f64,
        g: f64,
        global_indices: &[usize],
        out: &mut [Vec3],
    ) {
        assert_eq!(global_indices.len(), out.len());
        let nm = self.grid_size;

        // ── 1. Asignar toda la masa al grid ───────────────────────────────────
        #[cfg(feature = "rayon")]
        let density = cic::assign_rayon(global_positions, global_masses, self.box_size, nm);
        #[cfg(not(feature = "rayon"))]
        let density = cic::assign(global_positions, global_masses, self.box_size, nm);

        // ── 2. Resolver Poisson → fuerzas en el grid ──────────────────────────
        let [fx_grid, fy_grid, fz_grid] = fft_poisson::solve_forces(&density, g, nm, self.box_size);

        // ── 3. Interpolar fuerzas a las posiciones activas ────────────────────
        let active_pos: Vec<Vec3> = global_indices
            .iter()
            .map(|&i| global_positions[i])
            .collect();
        #[cfg(feature = "rayon")]
        let acc =
            cic::interpolate_rayon(&fx_grid, &fy_grid, &fz_grid, &active_pos, self.box_size, nm);
        #[cfg(not(feature = "rayon"))]
        let acc = cic::interpolate(&fx_grid, &fy_grid, &fz_grid, &active_pos, self.box_size, nm);

        out.copy_from_slice(&acc);
    }
}

unsafe impl Send for PmSolver {}
unsafe impl Sync for PmSolver {}
