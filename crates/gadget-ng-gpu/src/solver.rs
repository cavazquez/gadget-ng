//! Solver GPU stub — punto de extensión para futuros kernels wgpu / CUDA / HIP.
//!
//! `GpuDirectGravity` es un tipo marcador sin implementación real.
//! La implementación del trait `GravitySolver` (definido en `gadget-ng-core`) se
//! añade en `gadget_ng_core::gpu_bridge` bajo `feature = "gpu"`, evitando la
//! dependencia circular core ↔ gpu.

/// Solver de gravedad directa GPU (placeholder).
///
/// # TODO(gpu)
/// - Añadir `wgpu` / `cust` (CUDA) / `hip-sys` (HIP/ROCm) como dep opcional.
/// - Añadir `GpuDirectGravity::new() -> Result<Self, GpuError>` que inicialice
///   el contexto del device.
/// - Empaquetar posiciones/masas en `GpuParticlesSoA` y subirlos al device.
/// - Lanzar el kernel de Plummer suavizado.
/// - Descargar aceleraciones de vuelta al host en `out`.
#[derive(Debug, Default, Clone, Copy)]
pub struct GpuDirectGravity;
