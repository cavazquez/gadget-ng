//! `CudaDirectGravity` — solver de gravedad directa N² GPU via CUDA con buffers persistentes.
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
//! El handle CUDA y los buffers device se retienen entre llamadas, eliminando
//! `cuda_direct_create`/`cuda_direct_destroy` y `cudaMalloc`/`cudaFree` por paso.

use crate::pool::CudaPool;
use crate::{CudaExecutionError, CudaPmSolver, CudaUnavailable};

/// Solver de gravedad directa N² via CUDA con handle y buffers persistentes.
///
/// El handle CUDA se crea una vez en [`CudaDirectGravity::try_new`] y se libera
/// en `Drop`. Los buffers device se reutilizan entre pasos vía [`CudaPool`].
#[non_exhaustive]
pub struct CudaDirectGravity {
    /// Softening gravitacional ε (en unidades internas). Se pasa al kernel.
    pub eps: f32,
    /// Número de hilos por bloque CUDA (potencia de 2, típico: 256).
    pub block_size: usize,
    #[cfg(not(cuda_unavailable))]
    handle: *mut std::ffi::c_void,
    #[cfg(not(cuda_unavailable))]
    pool: CudaPool,
    #[cfg(cuda_unavailable)]
    _phantom: (),
}

// SAFETY: el handle CUDA es propiedad exclusiva de este struct; no se comparte.
unsafe impl Send for CudaDirectGravity {}
unsafe impl Sync for CudaDirectGravity {}

impl std::fmt::Debug for CudaDirectGravity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaDirectGravity")
            .field("eps", &self.eps)
            .field("block_size", &self.block_size)
            .finish()
    }
}

impl CudaDirectGravity {
    /// Intenta construir el solver de gravedad directa CUDA con handle persistente.
    pub fn try_new(eps: f32) -> Option<Self> {
        Self::try_new_checked(eps).ok()
    }

    /// Variante fallible de [`Self::try_new`] que conserva el motivo de indisponibilidad.
    pub fn try_new_checked(eps: f32) -> Result<Self, CudaUnavailable> {
        #[cfg(cuda_unavailable)]
        {
            let _ = eps;
            return Err(CudaUnavailable {
                availability: CudaPmSolver::availability(),
            });
        }

        #[cfg(not(cuda_unavailable))]
        {
            use crate::ffi;

            let handle = unsafe { ffi::cuda_direct_create(eps * eps, 256) };
            if handle.is_null() {
                return Err(CudaUnavailable {
                    availability: CudaPmSolver::availability(),
                });
            }
            let pool = CudaPool::try_new_with_capacity(0).map_err(|_| CudaUnavailable {
                availability: CudaPmSolver::availability(),
            })?;
            Ok(Self {
                eps,
                block_size: 256,
                handle,
                pool,
            })
        }
    }

    /// Calcula aceleraciones gravitacionales directas O(N²) para N partículas.
    ///
    /// # Panics
    ///
    /// Panics si CUDA no está disponible en tiempo de compilación (`cuda_unavailable`).
    pub fn compute(&self, pos: &[[f32; 3]], mass: &[f32]) -> Vec<[f32; 3]> {
        match self.try_compute(pos, mass) {
            Ok(accels) => accels,
            Err(err) => panic!("{err}"),
        }
    }

    /// Calcula aceleraciones directas y devuelve un error explícito si CUDA falla.
    pub fn try_compute(
        &self,
        pos: &[[f32; 3]],
        mass: &[f32],
    ) -> Result<Vec<[f32; 3]>, CudaExecutionError> {
        let n = pos.len();
        assert_eq!(mass.len(), n, "pos y mass deben tener la misma longitud");

        #[cfg(cuda_unavailable)]
        {
            let _ = (pos, mass);
            return Err(CudaExecutionError::Unavailable(CudaUnavailable {
                availability: CudaPmSolver::availability(),
            }));
        }

        #[cfg(not(cuda_unavailable))]
        {
            use crate::ffi;

            self.pool.ensure_capacity(n)?;
            self.pool.reset();

            let mut x: Vec<f32> = Vec::with_capacity(n);
            let mut y: Vec<f32> = Vec::with_capacity(n);
            let mut z: Vec<f32> = Vec::with_capacity(n);
            for p in pos {
                x.push(p[0]);
                y.push(p[1]);
                z.push(p[2]);
            }

            // SAFETY: handle is non-NULL (verified in try_new). Pool slots are freshly reset.
            unsafe {
                let d_x = self.pool.upload_f32(0, &x);
                let d_y = self.pool.upload_f32(1, &y);
                let d_z = self.pool.upload_f32(2, &z);
                let d_mass = self.pool.upload_f32(3, mass);
                let d_ax = self.pool.alloc_f32(4, n);
                let d_ay = self.pool.alloc_f32(5, n);
                let d_az = self.pool.alloc_f32(6, n);

                let g = 1.0_f32;
                let ret = ffi::cuda_direct_solve(
                    self.handle,
                    d_x,
                    d_y,
                    d_z,
                    d_mass,
                    d_ax,
                    d_ay,
                    d_az,
                    n as i32,
                    g,
                );
                if ret != 0 {
                    return Err(CudaExecutionError::KernelFailed {
                        kernel: "cuda_direct_solve",
                        code: ret,
                    });
                }

                let mut ax = vec![0.0_f32; n];
                let mut ay = vec![0.0_f32; n];
                let mut az = vec![0.0_f32; n];
                self.pool.download_f32(&mut ax, d_ax)?;
                self.pool.download_f32(&mut ay, d_ay)?;
                self.pool.download_f32(&mut az, d_az)?;

                Ok((0..n).map(|i| [ax[i], ay[i], az[i]]).collect())
            }
        }
    }

    /// Número de partículas máximo recomendado para este solver en el hardware disponible.
    pub fn recommended_max_n(&self) -> usize {
        65536
    }
}

impl Drop for CudaDirectGravity {
    fn drop(&mut self) {
        #[cfg(not(cuda_unavailable))]
        unsafe {
            crate::ffi::cuda_direct_destroy(self.handle);
        }
    }
}

// ── GravitySolver bridge ────────────────────────────────────────────────────

impl gadget_ng_core::gravity::GravitySolver for CudaDirectGravity {
    fn accelerations_for_indices(
        &self,
        global_positions: &[gadget_ng_core::Vec3],
        global_masses: &[f64],
        _eps2: f64,
        _g: f64,
        global_indices: &[usize],
        out: &mut [gadget_ng_core::Vec3],
    ) {
        assert_eq!(global_indices.len(), out.len());
        if global_indices.is_empty() {
            return;
        }
        let pos_f32: Vec<[f32; 3]> = global_positions
            .iter()
            .map(|p| [p.x as f32, p.y as f32, p.z as f32])
            .collect();
        let mass_f32: Vec<f32> = global_masses.iter().map(|&m| m as f32).collect();
        let acc = self.compute(&pos_f32, &mass_f32);
        for (k, &gi) in global_indices.iter().enumerate() {
            let [ax, ay, az] = acc[gi];
            out[k] = gadget_ng_core::Vec3::new(ax as f64, ay as f64, az as f64);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn try_new_returns_none_without_cuda() {
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
    }

    #[test]
    fn recommended_max_n_is_positive() {
        let solver = CudaDirectGravity {
            eps: 0.01,
            block_size: 256,
            #[cfg(not(cuda_unavailable))]
            handle: std::ptr::null_mut(),
            #[cfg(not(cuda_unavailable))]
            pool: unsafe { std::mem::zeroed() },
            #[cfg(cuda_unavailable)]
            _phantom: (),
        };
        assert!(solver.recommended_max_n() > 0);
    }
}

#[cfg(cuda_unavailable)]
const _: () = {
    let _ = std::mem::size_of::<CudaDirectGravity>();
};
