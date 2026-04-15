//! `gadget-ng-gpu` — placeholder para futuros kernels GPU.
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
//! ## Punto de extensión
//!
//! [`GpuDirectGravity`] es hoy un tipo marcador. La impl `GravitySolver` está en
//! `gadget_ng_core::gpu_bridge` y llama a `unimplemented!`. Para los kernels
//! reales, añadir las dependencias GPU en `Cargo.toml` y reemplazar esa impl.
pub mod soa;
pub mod solver;

pub use soa::GpuParticlesSoA;
pub use solver::GpuDirectGravity;
