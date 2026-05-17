//! `CudaPmSolver` — wrapper Rust sobre el solver PM CUDA.

#![allow(clippy::needless_return)]

use std::ffi::c_void;

use crate::availability::{self, CudaAvailability, CudaExecutionError, CudaUnavailable};
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
    /// Si es > 0, aplica el mismo filtro Gaussiano en *k* que `fft_poisson::solve_forces_filtered`.
    /// Si es ≤ 0, PM sin filtro (compatible con el comportamiento anterior).
    #[cfg_attr(cuda_unavailable, allow(dead_code))]
    r_split: f32,
}

// SAFETY: el handle CUDA es propiedad exclusiva de este struct; no se comparte.
// Las operaciones CUDA sobre un device son internamente thread-safe para streams
// independientes; aquí usamos el stream por defecto, por lo que no enviamos el
// handle a varios hilos simultáneamente (lo garantiza el motor de simulación).
unsafe impl Send for CudaPmSolver {}
unsafe impl Sync for CudaPmSolver {}

impl CudaPmSolver {
    /// Estado detallado de CUDA, separando compilación y runtime.
    pub fn availability() -> CudaAvailability {
        #[cfg(cuda_unavailable)]
        {
            return availability::build_availability();
        }

        #[cfg(not(cuda_unavailable))]
        {
            let mut status = availability::build_availability();
            // SAFETY: llamada FFI de prueba con grilla mínima; el handle se destruye si existe.
            let h = unsafe { crate::ffi::cuda_pm_create(8, 1.0) };
            if h.is_null() {
                status.runtime_available = false;
                status.reason = "toolchain CUDA disponible, pero no se pudo crear handle runtime";
                return status;
            }
            // SAFETY: h fue creado por cuda_pm_create y no se usará después.
            unsafe { crate::ffi::cuda_pm_destroy(h) };
            status.runtime_available = true;
            status
        }
    }

    /// `true` si el crate se compiló con soporte CUDA y hay un dispositivo disponible.
    pub fn is_available() -> bool {
        Self::availability().is_available()
    }

    /// Intenta construir el solver PM CUDA.
    ///
    /// # Parámetros
    /// - `grid_size` — lado de la grilla PM (e.g. 64, 128, 256)
    /// - `box_size`  — tamaño de la caja periódica en las mismas unidades que las posiciones
    ///
    /// Devuelve `None` si CUDA no está disponible o si la inicialización falla.
    pub fn try_new(grid_size: usize, box_size: f64) -> Option<Self> {
        Self::try_new_checked(grid_size, box_size).ok()
    }

    /// Variante fallible de [`Self::try_new`] que conserva el motivo de indisponibilidad.
    pub fn try_new_checked(grid_size: usize, box_size: f64) -> Result<Self, CudaUnavailable> {
        Self::try_new_with_r_split_checked(grid_size, box_size, 0.0)
    }

    /// Como [`Self::try_new`], pero con filtro Gaussiano TreePM (`r_split` > 0).
    pub fn try_new_with_r_split(grid_size: usize, box_size: f64, r_split: f64) -> Option<Self> {
        Self::try_new_with_r_split_checked(grid_size, box_size, r_split).ok()
    }

    /// Variante fallible de [`Self::try_new_with_r_split`].
    pub fn try_new_with_r_split_checked(
        grid_size: usize,
        box_size: f64,
        r_split: f64,
    ) -> Result<Self, CudaUnavailable> {
        #[cfg(cuda_unavailable)]
        {
            let _ = (grid_size, box_size, r_split);
            return Err(CudaUnavailable {
                availability: Self::availability(),
            });
        }

        #[cfg(not(cuda_unavailable))]
        {
            // SAFETY: FFI call compilada desde C++ CUDA con convenciones ABI compatibles.
            // grid_size y box_size son valores escalares válidos. El handle se verifica
            // no-NULL inmediatamente después.
            let handle = unsafe { crate::ffi::cuda_pm_create(grid_size as i32, box_size as f32) };
            if handle.is_null() {
                return Err(CudaUnavailable {
                    availability: Self::availability(),
                });
            }
            Ok(Self {
                handle,
                grid_size,
                r_split: r_split as f32,
            })
        }
    }

    /// Radio de splitting Gaussiano configurado (0 = sin filtro).
    pub fn r_split(&self) -> f32 {
        #[cfg(cuda_unavailable)]
        {
            return 0.0;
        }
        #[cfg(not(cuda_unavailable))]
        {
            self.r_split
        }
    }

    /// Tamaño de grilla con el que fue construido el solver.
    pub fn grid_size(&self) -> usize {
        self.grid_size
    }

    /// Calcula aceleraciones PM CUDA y devuelve un error explícito si el kernel falla.
    ///
    /// Esta es la API preferida para código que puede decidir cómo reaccionar ante
    /// errores de runtime GPU. La implementación de [`GravitySolver`] conserva la
    /// firma histórica y convierte el error en `panic!` para evitar continuar una
    /// simulación con aceleraciones inválidas.
    pub fn try_accelerations_for_indices(
        &self,
        global_positions: &[Vec3],
        global_masses: &[f64],
        eps2: f64,
        g: f64,
        global_indices: &[usize],
        out: &mut [Vec3],
    ) -> Result<(), CudaExecutionError> {
        assert_eq!(global_indices.len(), out.len());
        if global_indices.is_empty() {
            return Ok(());
        }

        #[cfg(cuda_unavailable)]
        {
            let _ = (
                global_positions,
                global_masses,
                eps2,
                g,
                global_indices,
                out,
            );
            return Err(CudaUnavailable {
                availability: Self::availability(),
            }
            .into());
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

            // SAFETY: handle es no-NULL (verificado en try_new). Todos los punteros
            // provienen de Vec<f32> válidos con capacidad ≥ n. Las longitudes coinciden
            // con n. La llamada se ejecuta en el device CUDA ya inicializado.
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
                    self.r_split,
                )
            };
            if ret != 0 {
                return Err(CudaExecutionError::KernelFailed {
                    kernel: "cuda_pm_solve",
                    code: ret,
                });
            }

            // Extraer solo los índices solicitados.
            for (j, &gi) in global_indices.iter().enumerate() {
                out[j] = Vec3::new(ax[gi] as f64, ay[gi] as f64, az[gi] as f64);
            }
            Ok(())
        }
    }

    /// Calcula el campo de screening chameleon f(R) en GPU (AP-20).
    ///
    /// Replica `gadget_ng_pm::fft_poisson::fr_screening_field`: por cada celda
    /// computa `S = min(1, |f_R / f_R0|)` y luego suaviza con `iterations`
    /// pasos Jacobi con factor de mezcla `smoothing`. Devuelve un `Vec<f64>`
    /// de longitud `nm³`.
    pub fn try_fr_screening_field(
        &self,
        density: &[f64],
        nm: usize,
        f_r0: f64,
        n_fr: f64,
        smoothing: f64,
        iterations: usize,
    ) -> Result<Vec<f64>, CudaExecutionError> {
        let nm3 = nm * nm * nm;
        assert_eq!(density.len(), nm3, "density.len() debe ser nm³");

        #[cfg(cuda_unavailable)]
        {
            let _ = (density, nm, f_r0, n_fr, smoothing, iterations);
            return Err(CudaExecutionError::Unavailable(CudaUnavailable {
                availability: CudaPmSolver::availability(),
            }));
        }

        #[cfg(not(cuda_unavailable))]
        {
            let den_f32: Vec<f32> = density.iter().map(|&v| v as f32).collect();
            let mut screen_out = vec![0.0_f32; nm3];

            // SAFETY: slices son válidos, nm3 == density.len().
            let code = unsafe {
                crate::ffi::cuda_fr_screening_field(
                    den_f32.as_ptr(),
                    screen_out.as_mut_ptr(),
                    nm as i32,
                    f_r0 as f32,
                    n_fr as f32,
                    smoothing as f32,
                    iterations as i32,
                )
            };
            check_kernel("cuda_fr_screening_field", code)?;
            Ok(screen_out.iter().map(|&v| v as f64).collect())
        }
    }
}

impl Drop for CudaPmSolver {
    fn drop(&mut self) {
        // SAFETY: self.handle es no-NULL (verificado en try_new). cuda_pm_destroy
        // libera recursos GPU y se llama exactamente una vez en Drop.
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
        if let Err(err) = self.try_accelerations_for_indices(
            global_positions,
            global_masses,
            eps2,
            g,
            global_indices,
            out,
        ) {
            panic!("CudaPmSolver::accelerations_for_indices falló: {err}");
        }
    }
}

fn check_kernel(kernel: &'static str, code: i32) -> Result<(), CudaExecutionError> {
    if code == 0 {
        Ok(())
    } else {
        Err(CudaExecutionError::KernelFailed { kernel, code })
    }
}

// Referencia al tipo para silenciar el warning de c_void cuando cuda_unavailable.
#[cfg(cuda_unavailable)]
const _: () = {
    let _ = std::mem::size_of::<c_void>();
};
