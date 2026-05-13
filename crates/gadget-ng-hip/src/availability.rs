//! Diagnóstico de disponibilidad HIP/ROCm.

use std::error::Error;
use std::fmt;

/// Estado de disponibilidad del backend HIP.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HipAvailability {
    /// `true` si `build.rs` logró compilar y enlazar los kernels HIP.
    pub compiled: bool,
    /// `true` si además se pudo crear un contexto/handle HIP en runtime.
    pub runtime_available: bool,
    /// Motivo humano-legible para el estado actual.
    pub reason: &'static str,
}

impl HipAvailability {
    /// `true` sólo cuando el backend está compilado y usable en runtime.
    pub fn is_available(&self) -> bool {
        self.compiled && self.runtime_available
    }
}

impl fmt::Display for HipAvailability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "HIP compiled={}, runtime_available={}, reason={}",
            self.compiled, self.runtime_available, self.reason
        )
    }
}

/// Error devuelto por APIs fallibles cuando HIP no puede usarse.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HipUnavailable {
    /// Estado capturado en el momento de la consulta.
    pub availability: HipAvailability,
}

impl fmt::Display for HipUnavailable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HIP no disponible: {}", self.availability)
    }
}

impl Error for HipUnavailable {}

/// Error devuelto por ejecuciones HIP que fallan tras crear el solver.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum HipExecutionError {
    /// HIP no está disponible.
    Unavailable(HipUnavailable),
    /// La inicialización del handle devolvió `NULL`.
    CreateFailed(&'static str),
    /// El kernel devolvió un código de error distinto de cero.
    KernelFailed { kernel: &'static str, code: i32 },
}

impl fmt::Display for HipExecutionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unavailable(err) => write!(f, "{err}"),
            Self::CreateFailed(what) => write!(f, "falló la creación de {what} HIP"),
            Self::KernelFailed { kernel, code } => {
                write!(f, "kernel HIP {kernel} falló con código {code}")
            }
        }
    }
}

impl Error for HipExecutionError {}

impl From<HipUnavailable> for HipExecutionError {
    fn from(value: HipUnavailable) -> Self {
        Self::Unavailable(value)
    }
}

/// Diagnóstico barato de compilación. No toca el runtime HIP.
pub fn build_availability() -> HipAvailability {
    let compiled = option_env!("GADGET_NG_HIP_COMPILED") == Some("1");
    let reason = option_env!("GADGET_NG_HIP_BUILD_REASON").unwrap_or("estado HIP desconocido");
    HipAvailability {
        compiled,
        runtime_available: false,
        reason,
    }
}
