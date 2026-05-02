//! `gadget-ng-gpu` — kernels gravitatorios en **wgpu** (WGSL).
//!
//! - [`GpuDirectGravity`] — gravedad directa O(N²).
//! - [`GpuBarnesHutMonopole`] — Barnes–Hut monopolo (camino compacto).
//! - [`GpuBarnesHutFmm`] — Barnes–Hut FMM órdenes 1–3 (quad/oct en WGSL).
//!
//! PM y direct en CUDA/HIP: `gadget-ng-cuda` / `gadget-ng-hip`.
//!
//! ## Diseño sin dependencias circulares
//!
//! Este crate no depende de `gadget-ng-core`. Los tipos aquí definidos usan sólo
//! primitivas de Rust (`f64`, `usize`, `Vec<_>`). Las integraciones con
//! `GravitySolver` y `Particle` viven en `gadget_ng_core::gpu_bridge`
//! (compilado bajo `feature = "gpu"` en `gadget-ng-core`).
//!
//! ## Política de datos (SoA)
//!
//! [`GpuParticlesSoA`] agrupa 8 arrays planos contiguos. Cada uno mapea
//! directamente a un buffer de device sin re-empaquetado:
//!
//! ```text
//! xs:     [x0, x1, x2, ...]
//! ys:     [y0, y1, y2, ...]
//! zs:     [z0, z1, z2, ...]
//! vxs:    [vx0, vx1, ...]
//! vys:    [vy0, vy1, ...]
//! vzs:    [vz0, vz1, ...]
//! masses: [m0, m1, m2, ...]
//! ids:    [id0, id1, id2, ...]
//! ```
//!
//! ## Integración
//!
//! [`GpuDirectGravity`] compila y ejecuta el shader anterior; la impl del trait
//! `GravitySolver` para el motor está en `gadget_ng_core::gpu_bridge` con
//! `feature = "gpu"`.
pub mod bh_fmm;
pub mod bh_monopole;
pub mod soa;
pub mod solver;
pub mod treepm_short_wgsl;

pub use bh_fmm::{BhFmmKernelParams, GpuBarnesHutFmm};
pub use bh_monopole::GpuBarnesHutMonopole;
pub use soa::GpuParticlesSoA;
pub use solver::GpuDirectGravity;
pub use treepm_short_wgsl::GpuTreePmShortRange;
