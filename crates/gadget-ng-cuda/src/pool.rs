//! `CudaPool` — envoltorio Rust del pool de buffers device reutilizables.
//!
//! Los kernels CUDA stateless (SPH, MHD, Tree, RT, Cooling, Dust, Molecular)
//! hacen `cudaMalloc`/`cudaFree` en *cada* llamada. `CudaPool` pre-aloca
//! buffers device y los reutiliza entre pasos de simulación, redimensionando
//! solo cuando el número de partículas excede la capacidad actual.

use std::ffi::c_void;

use crate::availability::{self, CudaAvailability, CudaUnavailable};

/// Pool de buffers device CUDA reutilizables entre pasos de simulación.
///
/// Se construye con [`CudaPool::try_new`] y se libera automáticamente en `Drop`.
/// Los buffers persisten entre llamadas; solo se redimensionan cuando `n` excede
/// la capacidad actual (doblamiento).
pub struct CudaPool {
    #[cfg(not(cuda_unavailable))]
    handle: *mut c_void,
    #[cfg(cuda_unavailable)]
    _phantom: (),
}

// SAFETY: el handle CUDA es propiedad exclusiva de este struct; las operaciones
// CUDA sobre un device son internamente thread-safe para streams independientes.
unsafe impl Send for CudaPool {}
unsafe impl Sync for CudaPool {}

impl CudaPool {
    /// Devuelve el estado de disponibilidad de CUDA.
    pub fn availability() -> CudaAvailability {
        #[cfg(cuda_unavailable)]
        {
            availability::build_availability()
        }

        #[cfg(not(cuda_unavailable))]
        {
            availability::build_availability()
        }
    }

    /// `true` si CUDA está compilado y disponible.
    pub fn is_available() -> bool {
        Self::availability().is_available()
    }

    /// Crea un pool vacío con capacidad inicial 0 (sin pre-asignar).
    pub fn try_new() -> Result<Self, CudaUnavailable> {
        Self::try_new_with_capacity(0)
    }

    /// Crea un pool con capacidad inicial `initial_n` partículas.
    pub fn try_new_with_capacity(initial_n: usize) -> Result<Self, CudaUnavailable> {
        #[cfg(cuda_unavailable)]
        {
            let _ = initial_n;
            return Err(CudaUnavailable {
                availability: Self::availability(),
            });
        }

        #[cfg(not(cuda_unavailable))]
        {
            // SAFETY: FFI call con valor escalar válido.
            let handle = unsafe { crate::ffi::cuda_pool_create(initial_n as i32) };
            if handle.is_null() {
                return Err(CudaUnavailable {
                    availability: Self::availability(),
                });
            }
            Ok(Self { handle })
        }
    }

    /// Asegura capacidad para `n` partículas. Redimensiona si es necesario.
    ///
    /// Devuelve error si la re-asignación falla.
    pub fn ensure_capacity(&self, n: usize) -> Result<(), crate::availability::CudaExecutionError> {
        #[cfg(cuda_unavailable)]
        {
            let _ = n;
            return Err(crate::availability::CudaExecutionError::Unavailable(
                CudaUnavailable {
                    availability: Self::availability(),
                },
            ));
        }

        #[cfg(not(cuda_unavailable))]
        {
            let ret = unsafe { crate::ffi::cuda_pool_ensure(self.handle, n as i32) };
            if ret != 0 {
                return Err(crate::availability::CudaExecutionError::KernelFailed {
                    kernel: "cuda_pool_ensure",
                    code: ret,
                });
            }
            Ok(())
        }
    }

    /// Resetea el pool para la próxima serie de uploads.
    /// Los slots existentes se pueden reescribir sin liberar memoria.
    pub fn reset(&self) {
        #[cfg(not(cuda_unavailable))]
        unsafe {
            crate::ffi::cuda_pool_reset(self.handle);
        }
        #[cfg(cuda_unavailable)]
        {}
    }

    /// Sube datos f32 al device en el slot `slot_index`.
    /// Devuelve el puntero device (válidomientras no se destruya el pool o se llame `ensure_capacity` con_growth).
    ///
    /// # Safety
    /// El puntero devuelto es válido mientras el pool no sea destruido ni redimensionado
    /// por encima de la capacidad actual.
    #[cfg(not(cuda_unavailable))]
    pub unsafe fn upload_f32(&self, slot: i32, data: &[f32]) -> *mut f32 {
        crate::ffi::cuda_pool_upload_f32(self.handle, slot, data.as_ptr(), data.len() as i32)
    }

    /// Sube datos u8 al device en el slot `slot_index`.
    #[cfg(not(cuda_unavailable))]
    pub unsafe fn upload_u8(&self, slot: i32, data: &[u8]) -> *mut u8 {
        crate::ffi::cuda_pool_upload_u8(self.handle, slot, data.as_ptr(), data.len() as i32)
    }

    /// Aloca un buffer f32 de salida (cero-inicializado) en el slot `slot_index`.
    #[cfg(not(cuda_unavailable))]
    pub unsafe fn alloc_f32(&self, slot: i32, n: usize) -> *mut f32 {
        crate::ffi::cuda_pool_alloc_f32(self.handle, slot, n as i32)
    }

    /// Descarga datos f32 del device al host.
    #[cfg(not(cuda_unavailable))]
    pub unsafe fn download_f32(
        &self,
        dst: &mut [f32],
        src: *const f32,
    ) -> Result<(), crate::availability::CudaExecutionError> {
        let ret = crate::ffi::cuda_pool_download_f32(
            self.handle,
            dst.as_mut_ptr(),
            src,
            dst.len() as i32,
        );
        if ret != 0 {
            Err(crate::availability::CudaExecutionError::KernelFailed {
                kernel: "cuda_pool_download_f32",
                code: ret,
            })
        } else {
            Ok(())
        }
    }

    /// Capacidad actual del pool en partículas.
    #[cfg(not(cuda_unavailable))]
    pub fn capacity(&self) -> usize {
        unsafe { crate::ffi::cuda_pool_capacity(self.handle) as usize }
    }

    /// Número de slots actualmente alojados.
    #[cfg(not(cuda_unavailable))]
    pub fn num_slots(&self) -> usize {
        unsafe { crate::ffi::cuda_pool_num_slots(self.handle) as usize }
    }

    /// Handle crudo (solo disponible cuando CUDA está compilado).
    #[cfg(not(cuda_unavailable))]
    pub fn as_ptr(&self) -> *mut c_void {
        self.handle
    }
}

impl Drop for CudaPool {
    fn drop(&mut self) {
        #[cfg(not(cuda_unavailable))]
        unsafe {
            crate::ffi::cuda_pool_destroy(self.handle);
        }
    }
}

#[cfg(cuda_unavailable)]
const _: () = {
    let _ = std::mem::size_of::<CudaPool>();
};
