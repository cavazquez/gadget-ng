//! `CudaDirectGravity` — solver de gravedad directa N² GPU via CUDA (Phase 163 / V1).
//!
//! ## Algoritmo
//!
//! Para cada par (i,j) el kernel CUDA calcula:
//!
//! ```text
//! a_i += G * m_j * (r_j - r_i) / (|r_j - r_i|² + ε²)^(3/2)
//! ```
//!
//! La implementación usa reducción en tiles (tiling) para maximizar el reuso de datos
//! en shared memory, con complejidad O(N²/P) por SM.
//!
//! ## Estado actual
//!
//! Stub para Phase 163. El método `compute` emite `todo!()` hasta que el kernel
//! CUDA sea implementado. El método `try_new` devuelve `Some(Self)` sólo si
//! `CudaPmSolver::is_available()` es verdadero (es decir, si hay hardware CUDA).
//!
//! ## Uso futuro
//!
//! ```rust,ignore
//! if let Some(gpu) = CudaDirectGravity::try_new(eps) {
//!     let accels = gpu.compute(&positions, &masses);
//!     // accels[i] = [ax, ay, az] para la partícula i
//! }
//! ```

use crate::CudaPmSolver;

/// Solver de gravedad directa N² via CUDA.
///
/// Calcula aceleraciones gravitacionales para todas las partículas usando
/// un kernel CUDA de fuerza bruta O(N²) con tiling en shared memory.
///
/// Construir con [`CudaDirectGravity::try_new`]; devuelve `None` si CUDA
/// no está disponible en el host.
pub struct CudaDirectGravity {
    /// Softening gravitacional ε (en unidades internas). Se pasa al kernel.
    pub eps: f32,
    /// Número de hilos por bloque CUDA (potencia de 2, típico: 256).
    pub block_size: usize,
}

impl CudaDirectGravity {
    /// Intenta construir el solver de gravedad directa CUDA.
    ///
    /// Devuelve `None` si no hay hardware CUDA disponible o si el crate fue
    /// compilado sin soporte CUDA (`cuda_unavailable`).
    ///
    /// # Parámetros
    /// - `eps` — softening gravitacional en unidades internas
    pub fn try_new(eps: f32) -> Option<Self> {
        if CudaPmSolver::is_available() {
            Some(Self {
                eps,
                block_size: 256,
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
    /// Siempre panics hasta que el kernel CUDA sea implementado.
    /// Ver `docs/validation-plan-hpc.md` sección V1 para el plan de implementación.
    #[allow(unused_variables)]
    pub fn compute(&self, pos: &[[f32; 3]], mass: &[f32]) -> Vec<[f32; 3]> {
        todo!(
            "kernel CUDA directo N² no implementado aún. \
             Ver docs/validation-plan-hpc.md §V1 para el plan de implementación."
        )
    }

    /// Número de partículas máximo recomendado para este solver en el hardware disponible.
    ///
    /// Heurística: 1 SM = 2048 hilos activos; con tiling de 256, máximo ~16k partículas
    /// antes de que el rendimiento se sature. Para N > 1M usar PM-GPU.
    pub fn recommended_max_n(&self) -> usize {
        // Stub: retorna un valor conservador.
        65536
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests unitarios (sin hardware CUDA requerido)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn try_new_returns_none_without_cuda() {
        // En CI sin hardware CUDA, try_new debe devolver None.
        // Con hardware, devuelve Some. El test pasa en ambos casos.
        let result = CudaDirectGravity::try_new(0.01);
        match result {
            None => {
                println!("CUDA no disponible (esperado en CI): try_new = None");
            }
            Some(ref solver) => {
                println!("CUDA disponible: block_size = {}", solver.block_size);
                assert!(solver.eps > 0.0);
                assert!(solver.block_size > 0);
            }
        }
        // No falla si CUDA no está disponible.
    }

    #[test]
    fn recommended_max_n_is_positive() {
        // El límite recomendado debe ser un número positivo razonable.
        // No requiere hardware.
        let solver = CudaDirectGravity { eps: 0.01, block_size: 256 };
        assert!(solver.recommended_max_n() > 0);
    }
}
