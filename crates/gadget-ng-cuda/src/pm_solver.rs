//! `CudaPmSolver` — wrapper Rust sobre el solver PM CUDA.

#![allow(clippy::needless_return)]

use std::ffi::c_void;

use gadget_ng_core::gravity::GravitySolver;
use gadget_ng_core::vec3::Vec3;

// ── CudaPmSolver ──────────────────────────────────────────────────────────────

/// Solver PM GPU implementado con CUDA (cuFFT).
///
/// Construir con [`CudaPmSolver::try_new`]; devuelve `None` si CUDA no está
/// disponible en el host o si el crate se compiló sin nvcc (toolchain ausente).
pub struct CudaPmSolver {
    #[cfg(not(cuda_unavailable))]
    handle: *mut c_void,
    #[cfg(cuda_unavailable)]
    _phantom: (),
    grid_size: usize,
}

// SAFETY: el handle CUDA es propiedad exclusiva de este struct; no se comparte.
// Las operaciones CUDA sobre un device son internamente thread-safe para streams
// independientes; aquí usamos el stream por defecto, por lo que no enviamos el
// handle a varios hilos simultáneamente (lo garantiza el motor de simulación).
unsafe impl Send for CudaPmSolver {}
unsafe impl Sync for CudaPmSolver {}

impl CudaPmSolver {
    /// `true` si el crate se compiló con soporte CUDA y hay un dispositivo disponible.
    pub fn is_available() -> bool {
        #[cfg(cuda_unavailable)]
        return false;

        #[cfg(not(cuda_unavailable))]
        {
            // Intento de crear un handle de prueba con grilla mínima.
            let h = unsafe { crate::ffi::cuda_pm_create(8, 1.0) };
            if h.is_null() {
                return false;
            }
            unsafe { crate::ffi::cuda_pm_destroy(h) };
            true
        }
    }

    /// Intenta construir el solver PM CUDA.
    ///
    /// # Parámetros
    /// - `grid_size` — lado de la grilla PM (e.g. 64, 128, 256)
    /// - `box_size`  — tamaño de la caja periódica en las mismas unidades que las posiciones
    ///
    /// Devuelve `None` si CUDA no está disponible o si la inicialización falla.
    pub fn try_new(grid_size: usize, box_size: f64) -> Option<Self> {
        #[cfg(cuda_unavailable)]
        {
            let _ = (grid_size, box_size);
            return None;
        }

        #[cfg(not(cuda_unavailable))]
        {
            let handle = unsafe { crate::ffi::cuda_pm_create(grid_size as i32, box_size as f32) };
            if handle.is_null() {
                return None;
            }
            Some(Self { handle, grid_size })
        }
    }

    /// Tamaño de grilla con el que fue construido el solver.
    pub fn grid_size(&self) -> usize {
        self.grid_size
    }
}

impl Drop for CudaPmSolver {
    fn drop(&mut self) {
        #[cfg(not(cuda_unavailable))]
        unsafe {
            crate::ffi::cuda_pm_destroy(self.handle);
        }
    }
}

// ── GravitySolver ─────────────────────────────────────────────────────────────

impl GravitySolver for CudaPmSolver {
    /// Calcula aceleraciones PM GPU para las partículas en `global_indices`.
    ///
    /// El solver PM asigna TODAS las partículas a la grilla de densidad,
    /// resuelve la ecuación de Poisson en k-space con cuFFT y luego interpola
    /// la fuerza solo para las partículas solicitadas.
    ///
    /// Conversión de precisión: f64 → f32 antes de enviar al device (GPU), y
    /// f32 → f64 en los resultados. El error relativo introducido es O(1e-7).
    fn accelerations_for_indices(
        &self,
        global_positions: &[Vec3],
        global_masses: &[f64],
        eps2: f64,
        g: f64,
        global_indices: &[usize],
        out: &mut [Vec3],
    ) {
        assert_eq!(global_indices.len(), out.len());
        if global_indices.is_empty() {
            return;
        }

        #[cfg(cuda_unavailable)]
        {
            // Nunca debería llegar aquí si try_new devuelve None correctamente.
            let _ = (global_positions, global_masses, eps2, g, global_indices);
            for v in out.iter_mut() {
                *v = Vec3::zero();
            }
            return;
        }

        #[cfg(not(cuda_unavailable))]
        {
            let n = global_positions.len();

            // Convertir posiciones y masas a f32 SoA.
            let mut xs: Vec<f32> = Vec::with_capacity(n);
            let mut ys: Vec<f32> = Vec::with_capacity(n);
            let mut zs: Vec<f32> = Vec::with_capacity(n);
            let mut masses: Vec<f32> = Vec::with_capacity(n);
            for (p, &m) in global_positions.iter().zip(global_masses.iter()) {
                xs.push(p.x as f32);
                ys.push(p.y as f32);
                zs.push(p.z as f32);
                masses.push(m as f32);
            }

            // Buffers de salida para TODAS las partículas.
            let mut ax: Vec<f32> = vec![0.0f32; n];
            let mut ay: Vec<f32> = vec![0.0f32; n];
            let mut az: Vec<f32> = vec![0.0f32; n];

            let ret = unsafe {
                crate::ffi::cuda_pm_solve(
                    self.handle,
                    xs.as_ptr(),
                    ys.as_ptr(),
                    zs.as_ptr(),
                    masses.as_ptr(),
                    ax.as_mut_ptr(),
                    ay.as_mut_ptr(),
                    az.as_mut_ptr(),
                    n as i32,
                    eps2 as f32,
                    g as f32,
                )
            };
            if ret != 0 {
                // En caso de error CUDA, dejar aceleraciones en cero.
                eprintln!("[CudaPmSolver] cuda_pm_solve error code {ret}");
                for v in out.iter_mut() {
                    *v = Vec3::zero();
                }
                return;
            }

            // Extraer solo los índices solicitados.
            for (j, &gi) in global_indices.iter().enumerate() {
                out[j] = Vec3::new(ax[gi] as f64, ay[gi] as f64, az[gi] as f64);
            }
        }
    }
}

// Referencia al tipo para silenciar el warning de c_void cuando cuda_unavailable.
#[cfg(cuda_unavailable)]
const _: () = {
    let _ = std::mem::size_of::<c_void>();
};
