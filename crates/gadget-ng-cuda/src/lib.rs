//! `gadget-ng-cuda` — solver PM GPU via CUDA (nvcc + cuFFT).
//!
//! # Buffers persistentes (AP-02)
//!
//! Todos los solvers CUDA (SPH, MHD, Tree, RT, Cooling, Dust, Molecular, Direct)
//! retienen un [`CudaPool`] de buffers device entre pasos de simulación. Esto elimina
//! `cudaMalloc`/`cudaFree` por invocación, reduciendo latencia en ~50-100 µs por
//! alloc. Los buffers se redimensionan solo cuando el número de partículas excede
//! la capacidad actual (doblamiento automático).
//!
//! Versión mínima de CUDA Toolkit: 8.0 (`sm_60` Pascal / GTX 10xx).

#![allow(unused_imports)]
#![allow(
    clippy::needless_return,
    clippy::unnecessary_lazy_evaluations,
    clippy::manual_non_exhaustive
)]
//!
//! # Cadena de compilación
//!
//! Este crate implementa una segunda cadena de compilación completa:
//!
//! ```text
//! cuda/pm_gravity.cu  ──nvcc──►  pm_gravity.o  ──ar──►  libpm_cuda.a
//!                                                              │
//!                                              cargo:rustc-link-lib=static=pm_cuda
//! ```
//!
//! La detección y compilación la realiza `build.rs`. Si `nvcc` o `cuFFT` no están
//! disponibles (CI, máquina sin CUDA), el build.rs emite `cargo:rustc-cfg=cuda_unavailable`
//! y el crate compila con stubs que devuelven `None`/`Err`.
//!
//! # Uso
//!
//! ```toml
//! # [performance]
//! # use_gpu_cuda = true
//! ```
//!
//! ```bash
//! cargo build --features cuda -p gadget-ng-cli
//! ```
//!
//! # Algoritmo PM GPU
//!
//! 1. **CIC assign** — asignar masas a grilla N³ (Cloud-In-Cell, atomicAdd en device)
//! 2. **FFT forward R→C** — cuFFT 3D real-to-complex
//! 3. **Poisson + diferenciación espectral** — Φ(k) = −4πG·ρ(k)/k²; F_α(k) = −ik_α Φ(k)
//! 4. **FFT inverse 3×** — cuFFT 3D complex-to-real para cada componente de fuerza
//! 5. **CIC interp** — interpolar fuerza en posiciones de partículas (trilineal)

pub mod availability;
pub mod cooling_solver;
pub mod direct_solver;
pub mod dust_solver;
pub mod ffi;
pub mod mhd_solver;
pub mod molecular_solver;
pub mod pm_solver;
pub mod pool;
pub mod rt_solver;
pub mod sph_solver;
pub mod tree_solver;

pub use availability::{CudaAvailability, CudaExecutionError, CudaUnavailable};
pub use cooling_solver::CudaCoolingSolver;
pub use direct_solver::CudaDirectGravity;
pub use dust_solver::CudaDustSolver;
pub use mhd_solver::CudaMhdSolver;
pub use molecular_solver::CudaMolecularSolver;
pub use pm_solver::CudaPmSolver;
pub use pool::CudaPool;
pub use rt_solver::CudaRtSolver;
pub use sph_solver::CudaSphSolver;
pub use tree_solver::CudaTreeSolver;
