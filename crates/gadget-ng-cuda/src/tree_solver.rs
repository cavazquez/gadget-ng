//! CUDA tree/SIDM kernels used by the Phase 200 parity track.
//!
//! Versión optimizada: `CudaTreeSolver` retiene un [`CudaPool`] de buffers
//! device entre pasos, eliminando `cudaMalloc`/`cudaFree` por invocación.
//! Los buffers se redimensionan solo cuando el número de partículas crece.

use crate::pool::CudaPool;
use crate::{CudaExecutionError, CudaPmSolver, CudaUnavailable};
#[cfg(not(cuda_unavailable))]
use gadget_ng_core::ParticleType;
use gadget_ng_core::{Particle, Vec3};

/// Solver CUDA para kernels tree/SIDM con buffers device persistentes.
pub struct CudaTreeSolver {
    #[cfg(not(cuda_unavailable))]
    pool: CudaPool,
    #[cfg(cuda_unavailable)]
    _phantom: (),
}

impl std::fmt::Debug for CudaTreeSolver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaTreeSolver").finish()
    }
}

impl Clone for CudaTreeSolver {
    fn clone(&self) -> Self {
        Self::try_new_checked().unwrap_or_else(|_| {
            #[cfg(not(cuda_unavailable))]
            {
                panic!("CudaTreeSolver clone failed: CUDA not available");
            }
            #[cfg(cuda_unavailable)]
            {
                Self { _phantom: () }
            }
        })
    }
}

impl CudaTreeSolver {
    pub fn try_new() -> Option<Self> {
        Self::try_new_checked().ok()
    }

    pub fn try_new_checked() -> Result<Self, CudaUnavailable> {
        if !CudaPmSolver::is_available() {
            return Err(CudaUnavailable {
                availability: CudaPmSolver::availability(),
            });
        }

        #[cfg(cuda_unavailable)]
        {
            return Err(CudaUnavailable {
                availability: CudaPmSolver::availability(),
            });
        }

        #[cfg(not(cuda_unavailable))]
        {
            let pool = CudaPool::try_new_with_capacity(0).map_err(|_| CudaUnavailable {
                availability: CudaPmSolver::availability(),
            })?;
            Ok(Self { pool })
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
            return Err(CudaExecutionError::Unavailable(CudaUnavailable {
                availability: CudaPmSolver::availability(),
            }));
        }

        #[cfg(not(cuda_unavailable))]
        {
            self.pool.ensure_capacity(n)?;
            self.pool.reset();

            let x: Vec<f32> = particles.iter().map(|p| p.position.x as f32).collect();
            let y: Vec<f32> = particles.iter().map(|p| p.position.y as f32).collect();
            let z: Vec<f32> = particles.iter().map(|p| p.position.z as f32).collect();
            let mass: Vec<f32> = particles.iter().map(|p| p.mass as f32).collect();

            // SAFETY: pool handle is valid, slots are freshly reset; all slices have length n.
            unsafe {
                let d_x = self.pool.upload_f32(0, &x);
                let d_y = self.pool.upload_f32(1, &y);
                let d_z = self.pool.upload_f32(2, &z);
                let d_mass = self.pool.upload_f32(3, &mass);
                let d_ax = self.pool.alloc_f32(4, n);
                let d_ay = self.pool.alloc_f32(5, n);
                let d_az = self.pool.alloc_f32(6, n);

                let code = crate::ffi::cuda_tree_walk_monopole(
                    d_x,
                    d_y,
                    d_z,
                    d_mass,
                    d_ax,
                    d_ay,
                    d_az,
                    n as i32,
                    g as f32,
                    eps2 as f32,
                );
                check_kernel("cuda_tree_walk_monopole", code)?;

                let mut ax = vec![0.0_f32; n];
                let mut ay = vec![0.0_f32; n];
                let mut az = vec![0.0_f32; n];
                self.pool.download_f32(&mut ax, d_ax)?;
                self.pool.download_f32(&mut ay, d_ay)?;
                self.pool.download_f32(&mut az, d_az)?;

                Ok((0..n)
                    .map(|i| Vec3::new(ax[i] as f64, ay[i] as f64, az[i] as f64))
                    .collect())
            }
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
            return Err(CudaExecutionError::Unavailable(CudaUnavailable {
                availability: CudaPmSolver::availability(),
            }));
        }

        #[cfg(not(cuda_unavailable))]
        {
            self.pool.ensure_capacity(n)?;
            self.pool.reset();

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

            // SAFETY: pool handle is valid, slots are freshly reset; all slices have length n.
            unsafe {
                let d_ptype = self.pool.upload_u8(0, &ptype);
                let d_x = self.pool.upload_f32(1, &x);
                let d_y = self.pool.upload_f32(2, &y);
                let d_z = self.pool.upload_f32(3, &z);
                let d_vx = self.pool.upload_f32(4, &vx);
                let d_vy = self.pool.upload_f32(5, &vy);
                let d_vz = self.pool.upload_f32(6, &vz);
                let d_mass = self.pool.upload_f32(7, &mass);
                let d_ovx = self.pool.alloc_f32(8, n);
                let d_ovy = self.pool.alloc_f32(9, n);
                let d_ovz = self.pool.alloc_f32(10, n);

                let code = crate::ffi::cuda_tree_sidm_scatter(
                    d_ptype,
                    d_x,
                    d_y,
                    d_z,
                    d_vx,
                    d_vy,
                    d_vz,
                    d_mass,
                    d_ovx,
                    d_ovy,
                    d_ovz,
                    n as i32,
                    dt as f32,
                    sigma_over_m as f32,
                    h as f32,
                );
                check_kernel("cuda_tree_sidm_scatter", code)?;

                let mut ovx = vec![0.0_f32; n];
                let mut ovy = vec![0.0_f32; n];
                let mut ovz = vec![0.0_f32; n];
                self.pool.download_f32(&mut ovx, d_ovx)?;
                self.pool.download_f32(&mut ovy, d_ovy)?;
                self.pool.download_f32(&mut ovz, d_ovz)?;

                for (i, p) in particles.iter_mut().enumerate() {
                    p.velocity = Vec3::new(ovx[i] as f64, ovy[i] as f64, ovz[i] as f64);
                }
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

#[cfg(cuda_unavailable)]
const _: () = {
    let _ = std::mem::size_of::<CudaTreeSolver>();
};
