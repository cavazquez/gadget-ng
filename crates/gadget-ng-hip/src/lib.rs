//! `gadget-ng-hip` — solver PM GPU via HIP/ROCm (hipcc + rocFFT).
//!
//! # Cadena de compilación
//!
//! ```text
//! hip/pm_gravity.hip  ──hipcc──►  pm_gravity.o  ──ar──►  libpm_hip.a
//!                                                               │
//!                                               cargo:rustc-link-lib=static=pm_hip
//! ```
//!
//! La detección y compilación la realiza `build.rs`. Si `hipcc` o `rocFFT` no están
//! disponibles (CI, máquina sin ROCm), el build.rs emite `cargo:rustc-cfg=hip_unavailable`
//! y el crate compila con stubs que devuelven `None`/`Err`.
//!
//! # Uso
//!
//! ```toml
//! # [performance]
//! # use_gpu_hip = true
//! ```
//!
//! ```bash
//! cargo build --features hip -p gadget-ng-cli
//! ```
//!
//! # Algoritmo PM GPU (idéntico al de CUDA, con rocFFT en lugar de cuFFT)
//!
//! 1. **CIC assign**  — asignar masas a grilla N³ (Cloud-In-Cell, atomicAdd)
//! 2. **FFT forward** — rocFFT 3D real-to-complex
//! 3. **Poisson**     — Φ(k) = −4πG·ρ(k)/k²; F_α(k) = −ik_α·Φ(k)
//! 4. **FFT inverse** — 3× rocFFT 3D complex-to-real
//! 5. **CIC interp**  — interpolar fuerza en posiciones de partículas

pub mod direct_solver;
pub mod ffi;
pub mod pm_solver;

pub use direct_solver::HipDirectGravity;
pub use pm_solver::HipPmSolver;
