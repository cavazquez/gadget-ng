//! Diagnóstico de disponibilidad CUDA.

use std::error::Error;
use std::fmt;

/// Estado de disponibilidad del backend CUDA.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CudaAvailability {
    /// `true` si `build.rs` logró compilar y enlazar los kernels CUDA.
    pub compiled: bool,
    /// `true` si además se pudo crear un contexto/handle CUDA en runtime.
    pub runtime_available: bool,
    /// Motivo humano-legible para el estado actual.
    pub reason: &'static str,
}

impl CudaAvailability {
    /// `true` sólo cuando el backend está compilado y usable en runtime.
    pub fn is_available(&self) -> bool {
        self.compiled && self.runtime_available
    }
}

impl fmt::Display for CudaAvailability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CUDA compiled={}, runtime_available={}, reason={}",
            self.compiled, self.runtime_available, self.reason
        )
    }
}

/// Error devuelto por APIs fallibles cuando CUDA no puede usarse.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CudaUnavailable {
    /// Estado capturado en el momento de la consulta.
    pub availability: CudaAvailability,
}

impl fmt::Display for CudaUnavailable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CUDA no disponible: {}", self.availability)
    }
}

impl Error for CudaUnavailable {}

/// Error devuelto por ejecuciones CUDA que fallan tras crear el solver.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CudaExecutionError {
    /// CUDA no está disponible.
    Unavailable(CudaUnavailable),
    /// La inicialización del handle devolvió `NULL`.
    CreateFailed(&'static str),
    /// El kernel devolvió un código de error distinto de cero.
    KernelFailed { kernel: &'static str, code: i32 },
}

impl fmt::Display for CudaExecutionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unavailable(err) => write!(f, "{err}"),
            Self::CreateFailed(what) => write!(f, "falló la creación de {what} CUDA"),
            Self::KernelFailed { kernel, code } => {
                write!(f, "kernel CUDA {kernel} falló con código {code}")
            }
        }
    }
}

impl Error for CudaExecutionError {}

impl From<CudaUnavailable> for CudaExecutionError {
    fn from(value: CudaUnavailable) -> Self {
        Self::Unavailable(value)
    }
}

/// Diagnóstico barato de compilación. No toca el runtime CUDA.
pub fn build_availability() -> CudaAvailability {
    let compiled = option_env!("GADGET_NG_CUDA_COMPILED") == Some("1");
    let reason = option_env!("GADGET_NG_CUDA_BUILD_REASON").unwrap_or("estado CUDA desconocido");
    CudaAvailability {
        compiled,
        runtime_available: false,
        reason,
    }
}
