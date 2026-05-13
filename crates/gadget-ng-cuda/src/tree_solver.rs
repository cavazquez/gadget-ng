//! CUDA tree/SIDM kernels used by the Phase 200 parity track.

use crate::{CudaExecutionError, CudaPmSolver, CudaUnavailable};
#[cfg(not(cuda_unavailable))]
use gadget_ng_core::ParticleType;
use gadget_ng_core::{Particle, Vec3};

/// Solver wrapper for CUDA tree-adjacent kernels.
#[derive(Debug, Clone, Copy)]
pub struct CudaTreeSolver;

impl CudaTreeSolver {
    pub fn try_new() -> Option<Self> {
        Self::try_new_checked().ok()
    }

    pub fn try_new_checked() -> Result<Self, CudaUnavailable> {
        if CudaPmSolver::is_available() {
            Ok(Self)
        } else {
            Err(CudaUnavailable {
                availability: CudaPmSolver::availability(),
            })
        }
    }

    pub fn try_walk_monopole(
        &self,
        particles: &[Particle],
        g: f64,
        eps2: f64,
    ) -> Result<Vec<Vec3>, CudaExecutionError> {
        let n = particles.len();
        if n == 0 {
            return Ok(Vec::new());
        }

        #[cfg(cuda_unavailable)]
        {
            let _ = (particles, g, eps2);
            Err(CudaUnavailable {
                availability: CudaPmSolver::availability(),
            }
            .into())
        }

        #[cfg(not(cuda_unavailable))]
        {
            let x: Vec<f32> = particles.iter().map(|p| p.position.x as f32).collect();
            let y: Vec<f32> = particles.iter().map(|p| p.position.y as f32).collect();
            let z: Vec<f32> = particles.iter().map(|p| p.position.z as f32).collect();
            let mass: Vec<f32> = particles.iter().map(|p| p.mass as f32).collect();
            let mut ax = vec![0.0_f32; n];
            let mut ay = vec![0.0_f32; n];
            let mut az = vec![0.0_f32; n];
            let code = unsafe {
                crate::ffi::cuda_tree_walk_monopole(
                    x.as_ptr(),
                    y.as_ptr(),
                    z.as_ptr(),
                    mass.as_ptr(),
                    ax.as_mut_ptr(),
                    ay.as_mut_ptr(),
                    az.as_mut_ptr(),
                    n as i32,
                    g as f32,
                    eps2 as f32,
                )
            };
            check_kernel("cuda_tree_walk_monopole", code)?;
            Ok((0..n)
                .map(|i| Vec3::new(ax[i] as f64, ay[i] as f64, az[i] as f64))
                .collect())
        }
    }

    pub fn try_sidm_scatter(
        &self,
        particles: &mut [Particle],
        dt: f64,
        sigma_over_m: f64,
        h: f64,
    ) -> Result<(), CudaExecutionError> {
        let n = particles.len();
        if n == 0 {
            return Ok(());
        }

        #[cfg(cuda_unavailable)]
        {
            let _ = (particles, dt, sigma_over_m, h);
            Err(CudaUnavailable {
                availability: CudaPmSolver::availability(),
            }
            .into())
        }

        #[cfg(not(cuda_unavailable))]
        {
            let ptype: Vec<u8> = particles
                .iter()
                .map(|p| match p.ptype {
                    ParticleType::DarkMatter => 0,
                    ParticleType::Gas => 1,
                    ParticleType::Star => 2,
                })
                .collect();
            let x: Vec<f32> = particles.iter().map(|p| p.position.x as f32).collect();
            let y: Vec<f32> = particles.iter().map(|p| p.position.y as f32).collect();
            let z: Vec<f32> = particles.iter().map(|p| p.position.z as f32).collect();
            let vx: Vec<f32> = particles.iter().map(|p| p.velocity.x as f32).collect();
            let vy: Vec<f32> = particles.iter().map(|p| p.velocity.y as f32).collect();
            let vz: Vec<f32> = particles.iter().map(|p| p.velocity.z as f32).collect();
            let mass: Vec<f32> = particles.iter().map(|p| p.mass as f32).collect();
            let mut ovx = vec![0.0_f32; n];
            let mut ovy = vec![0.0_f32; n];
            let mut ovz = vec![0.0_f32; n];
            let code = unsafe {
                crate::ffi::cuda_tree_sidm_scatter(
                    ptype.as_ptr(),
                    x.as_ptr(),
                    y.as_ptr(),
                    z.as_ptr(),
                    vx.as_ptr(),
                    vy.as_ptr(),
                    vz.as_ptr(),
                    mass.as_ptr(),
                    ovx.as_mut_ptr(),
                    ovy.as_mut_ptr(),
                    ovz.as_mut_ptr(),
                    n as i32,
                    dt as f32,
                    sigma_over_m as f32,
                    h as f32,
                )
            };
            check_kernel("cuda_tree_sidm_scatter", code)?;
            for (i, p) in particles.iter_mut().enumerate() {
                p.velocity = Vec3::new(ovx[i] as f64, ovy[i] as f64, ovz[i] as f64);
            }
            Ok(())
        }
    }
}

#[cfg(not(cuda_unavailable))]
fn check_kernel(kernel: &'static str, code: i32) -> Result<(), CudaExecutionError> {
    if code == 0 {
        Ok(())
    } else {
        Err(CudaExecutionError::KernelFailed { kernel, code })
    }
}
