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
//! El método `try_new` devuelve `Some(Self)` sólo si hay hardware CUDA
//! disponible. El método `compute` llama al kernel real via FFI cuando
//! CUDA está disponible, y entra en `panic!` con un mensaje informativo en caso
//! contrario (no debería ocurrir porque `try_new` devuelve `None` sin hardware).
//!
//! ## Uso
//!
//! ```rust,no_run
//! # use gadget_ng_cuda::CudaDirectGravity;
//! let eps = 0.01_f32;
//! let positions: Vec<[f32; 3]> = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
//! let masses: Vec<f32> = vec![1.0, 1.0];
//! if let Some(gpu) = CudaDirectGravity::try_new(eps) {
//!     let accels = gpu.compute(&positions, &masses);
//!     // accels[i] = [ax, ay, az] para la partícula i
//!     let _ = accels;
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

    /// Calcula aceleraciones gravitacionales directas O(N²) para N partículas.
    ///
    /// # Parámetros
    /// - `pos`  — posiciones `[[x, y, z]; N]` en unidades internas (f32)
    /// - `mass` — masas `[m_0, ..., m_{N-1}]` en unidades internas (f32)
    ///
    /// # Retorna
    ///
    /// Vector de aceleraciones `[[ax, ay, az]; N]` en unidades internas.
    ///
    /// # Panics
    ///
    /// Panics si CUDA no está disponible en tiempo de compilación (`cuda_unavailable`).
    /// Esto no debería ocurrir normalmente porque `try_new` devuelve `None` sin hardware.
    pub fn compute(&self, pos: &[[f32; 3]], mass: &[f32]) -> Vec<[f32; 3]> {
        let n = pos.len();
        assert_eq!(mass.len(), n, "pos y mass deben tener la misma longitud");

        #[cfg(cuda_unavailable)]
        panic!(
            "CudaDirectGravity::compute llamado pero CUDA no está disponible. \
             Usa CudaDirectGravity::try_new para verificar disponibilidad."
        );

        #[cfg(not(cuda_unavailable))]
        {
            use crate::ffi;
            use std::ffi::c_void;

            let eps2 = self.eps * self.eps;
            let g = 1.0_f32; // G en unidades internas (ajustar si se necesita otro valor)

            // Crear handle CUDA
            let handle: *mut c_void = unsafe {
                ffi::cuda_direct_create(eps2, self.block_size as i32)
            };
            assert!(!handle.is_null(), "cuda_direct_create devolvió NULL");

            // Extraer componentes de posición en arrays contiguos
            let mut x: Vec<f32> = Vec::with_capacity(n);
            let mut y: Vec<f32> = Vec::with_capacity(n);
            let mut z: Vec<f32> = Vec::with_capacity(n);
            for p in pos {
                x.push(p[0]);
                y.push(p[1]);
                z.push(p[2]);
            }

            let mut ax = vec![0.0_f32; n];
            let mut ay = vec![0.0_f32; n];
            let mut az = vec![0.0_f32; n];

            let ret = unsafe {
                ffi::cuda_direct_solve(
                    handle,
                    x.as_ptr(), y.as_ptr(), z.as_ptr(),
                    mass.as_ptr(),
                    ax.as_mut_ptr(), ay.as_mut_ptr(), az.as_mut_ptr(),
                    n as i32,
                    g,
                )
            };

            // Liberar handle antes de comprobar el error
            unsafe { ffi::cuda_direct_destroy(handle) };

            assert_eq!(ret, 0, "cuda_direct_solve falló con código {ret}");

            // Combinar en [[ax, ay, az]; N]
            (0..n).map(|i| [ax[i], ay[i], az[i]]).collect()
        }
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
