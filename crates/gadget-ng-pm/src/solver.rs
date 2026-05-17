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
    /// Suavizado Plummer en k-space (`exp(−k²ε²)`). `None` = sólo resolución de malla.
    pub plummer_eps: Option<f64>,
    /// Parámetros f(R) para una quinta fuerza PM homogénea de largo alcance.
    pub modified_gravity: Option<gadget_ng_core::FRParams>,
    /// Si true, usa screening f(R) espacial en malla.
    pub fr_nonlinear_mesh: bool,
    /// Iteraciones de suavizado para el screening f(R).
    pub fr_mesh_iterations: usize,
    /// Mezcla por iteración para el screening f(R).
    pub fr_screening_smoothing: f64,
    /// Campo de screening f(R) pre-computado externamente (AP-20: GPU via CUDA).
    /// Si `Some`, se usa directamente en `solve_forces_fr_screened_mesh` en lugar de
    /// calcular en CPU. Se consume (toma) en la primera llamada.
    pub screening_override: Option<Vec<f64>>,
}

impl PmSolver {
    #[must_use]
    pub fn new(grid_size: usize, box_size: f64) -> Self {
        Self {
            grid_size,
            box_size,
            plummer_eps: None,
            modified_gravity: None,
            fr_nonlinear_mesh: false,
            fr_mesh_iterations: 4,
            fr_screening_smoothing: 0.5,
            screening_override: None,
        }
    }

    /// Inyecta un campo de screening pre-computado (e.g. desde GPU) para la
    /// próxima llamada a `accelerations_for_indices`. Se consume en esa llamada.
    pub fn set_screening_override(&mut self, screening: Vec<f64>) {
        self.screening_override = Some(screening);
    }
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
        let [fx_grid, fy_grid, fz_grid] = if let Some(params) = &self.modified_gravity {
            if self.fr_nonlinear_mesh {
                fft_poisson::solve_forces_fr_screened_mesh(
                    &density,
                    g,
                    nm,
                    self.box_size,
                    fft_poisson::FrMeshParams {
                        fr: params,
                        iterations: self.fr_mesh_iterations,
                        smoothing: self.fr_screening_smoothing,
                        plummer_eps: self.plummer_eps,
                        // AP-20: screening pre-computado por GPU si disponible.
                        screening_override: self.screening_override.clone(),
                    },
                )
            } else {
                fft_poisson::solve_forces_modified_gravity(
                    &density,
                    g,
                    nm,
                    self.box_size,
                    params,
                    self.plummer_eps,
                )
            }
        } else {
            match self.plummer_eps {
                Some(eps) if eps > 0.0 => {
                    fft_poisson::solve_forces_softened(&density, g, nm, self.box_size, Some(eps))
                }
                _ => fft_poisson::solve_forces(&density, g, nm, self.box_size),
            }
        };

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

// SAFETY: PmSolver contiene solo usize, f64 y Option<f64>, todos Send + Sync.
// Las impls manuales son redundantes (el compilador las derivaría automáticamente)
// pero se mantienen para documentar explícitamente que el tipo es thread-safe.
unsafe impl Send for PmSolver {}
unsafe impl Sync for PmSolver {}
