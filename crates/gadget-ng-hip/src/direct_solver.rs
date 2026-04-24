//! `HipDirectGravity` — solver de gravedad directa N² GPU via HIP/ROCm (Phase 163 / V1).
//!
//! ## Algoritmo
//!
//! Idéntico al de `CudaDirectGravity`, pero usando kernels HIP/ROCm:
//!
//! ```text
//! a_i += G * m_j * (r_j - r_i) / (|r_j - r_i|² + ε²)^(3/2)
//! ```
//!
//! ## Estado actual
//!
//! Stub para Phase 163. El método `compute` emite `todo!()` hasta que el kernel
//! HIP sea implementado. El método `try_new` devuelve `Some(Self)` sólo si
//! `HipPmSolver::is_available()` es verdadero.
//!
//! ## Uso futuro
//!
//! ```rust,ignore
//! if let Some(gpu) = HipDirectGravity::try_new(eps) {
//!     let accels = gpu.compute(&positions, &masses);
//! }
//! ```

use crate::HipPmSolver;

/// Solver de gravedad directa N² via HIP/ROCm.
///
/// Construir con [`HipDirectGravity::try_new`]; devuelve `None` si HIP/ROCm
/// no está disponible en el host.
pub struct HipDirectGravity {
    /// Softening gravitacional ε (en unidades internas). Se pasa al kernel.
    pub eps: f32,
    /// Número de hilos por workgroup HIP (potencia de 2, típico: 256).
    pub workgroup_size: usize,
}

impl HipDirectGravity {
    /// Intenta construir el solver de gravedad directa HIP.
    ///
    /// Devuelve `None` si no hay hardware HIP/ROCm disponible o si el crate fue
    /// compilado sin soporte HIP (`hip_unavailable`).
    ///
    /// # Parámetros
    /// - `eps` — softening gravitacional en unidades internas
    pub fn try_new(eps: f32) -> Option<Self> {
        if HipPmSolver::is_available() {
            Some(Self {
                eps,
                workgroup_size: 256,
            })
        } else {
            None
        }
    }

    /// Calcula aceleraciones gravitacionales directas para N partículas.
    ///
    /// # Parámetros
    /// - `pos` — posiciones `[[x, y, z]; N]` en unidades internas (f32)
    /// - `mass` — masas `[m_0, m_1, ..., m_{N-1}]` en unidades internas (f32)
    ///
    /// # Retorna
    ///
    /// Vector de aceleraciones `[[ax, ay, az]; N]` en unidades internas.
    ///
    /// # Panics
    ///
    /// Siempre panics hasta que el kernel HIP sea implementado.
    #[allow(unused_variables)]
    pub fn compute(&self, pos: &[[f32; 3]], mass: &[f32]) -> Vec<[f32; 3]> {
        todo!(
            "kernel HIP directo N² no implementado aún. \
             Ver docs/validation-plan-hpc.md §V1 para el plan de implementación."
        )
    }

    /// Número de partículas máximo recomendado para este solver.
    pub fn recommended_max_n(&self) -> usize {
        65536
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests unitarios (sin hardware HIP requerido)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn try_new_returns_none_without_hip() {
        let result = HipDirectGravity::try_new(0.01);
        match result {
            None => {
                println!("HIP no disponible (esperado en CI): try_new = None");
            }
            Some(ref solver) => {
                println!("HIP disponible: workgroup_size = {}", solver.workgroup_size);
                assert!(solver.eps > 0.0);
                assert!(solver.workgroup_size > 0);
            }
        }
    }

    #[test]
    fn recommended_max_n_is_positive() {
        let solver = HipDirectGravity { eps: 0.01, workgroup_size: 256 };
        assert!(solver.recommended_max_n() > 0);
    }
}
