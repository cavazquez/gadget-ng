//! `gadget-ng-cuda` — solver PM GPU via CUDA (nvcc + cuFFT).
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

pub mod direct_solver;
pub mod ffi;
pub mod pm_solver;

pub use direct_solver::CudaDirectGravity;
pub use pm_solver::CudaPmSolver;
