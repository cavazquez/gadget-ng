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
// CI sin nvcc: métodos devuelven Err antes de llamar kernels; helpers quedan sin usar.
#![cfg_attr(cuda_unavailable, allow(dead_code, unused_variables))]
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

pub mod analysis_solver;
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

pub use analysis_solver::CudaAnalysisSolver;
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

#[cfg(test)]
mod integration_tests {
    use super::*;

    macro_rules! assert_solver_constructible {
        ($name:expr, $try:expr) => {
            match $try {
                Ok(_) => {}
                Err(err) => assert!(
                    !err.availability.is_available(),
                    "{name}: falló con CUDA compilado: {err}",
                    name = $name
                ),
            }
        };
    }

    #[test]
    fn pm_solver_try_new_checked() {
        assert_solver_constructible!(
            "CudaPmSolver",
            CudaPmSolver::try_new_checked(16, 1.0)
        );
    }

    #[test]
    fn sph_solver_try_new_checked() {
        assert_solver_constructible!("CudaSphSolver", CudaSphSolver::try_new_checked());
    }

    #[test]
    fn mhd_solver_try_new_checked() {
        assert_solver_constructible!("CudaMhdSolver", CudaMhdSolver::try_new_checked());
    }

    #[test]
    fn tree_solver_try_new_checked() {
        assert_solver_constructible!("CudaTreeSolver", CudaTreeSolver::try_new_checked());
    }

    #[test]
    fn rt_solver_try_new_checked() {
        assert_solver_constructible!("CudaRtSolver", CudaRtSolver::try_new_checked());
    }

    #[test]
    fn cooling_solver_try_new_checked() {
        assert_solver_constructible!("CudaCoolingSolver", CudaCoolingSolver::try_new_checked());
    }

    #[test]
    fn dust_solver_try_new_checked() {
        assert_solver_constructible!("CudaDustSolver", CudaDustSolver::try_new_checked());
    }

    #[test]
    fn molecular_solver_try_new_checked() {
        assert_solver_constructible!(
            "CudaMolecularSolver",
            CudaMolecularSolver::try_new_checked()
        );
    }

    #[test]
    fn analysis_solver_try_new_checked() {
        assert_solver_constructible!(
            "CudaAnalysisSolver",
            CudaAnalysisSolver::try_new_checked()
        );
    }
}
