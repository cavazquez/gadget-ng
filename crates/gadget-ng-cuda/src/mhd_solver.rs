//! Kernels MHD simples via CUDA.

use crate::{CudaExecutionError, CudaPmSolver, CudaUnavailable};
use gadget_ng_core::Particle;
#[cfg(not(cuda_unavailable))]
use gadget_ng_core::ParticleType;
#[cfg(not(cuda_unavailable))]
use gadget_ng_core::Vec3;
use gadget_ng_mhd::stats::BFieldStats;

/// Solver CUDA para kernels MHD locales y reducciones simples.
#[derive(Debug, Clone, Copy)]
pub struct CudaMhdSolver;

#[derive(Debug)]
#[cfg(not(cuda_unavailable))]
struct MhdSoa {
    ptype: Vec<u8>,
    mass: Vec<f32>,
    internal_energy: Vec<f32>,
    h: Vec<f32>,
    bx: Vec<f32>,
    by: Vec<f32>,
    bz: Vec<f32>,
}

impl CudaMhdSolver {
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

    pub fn try_apply_flux_freeze(
        &self,
        particles: &mut [Particle],
        gamma: f64,
        beta_freeze: f64,
        rho_ref: f64,
    ) -> Result<(), CudaExecutionError> {
        let n = particles.len();
        if n == 0 {
            return Ok(());
        }

        #[cfg(cuda_unavailable)]
        {
            let _ = (particles, gamma, beta_freeze, rho_ref);
            Err(CudaUnavailable {
                availability: CudaPmSolver::availability(),
            }
            .into())
        }

        #[cfg(not(cuda_unavailable))]
        {
            let soa = MhdSoa::from_particles(particles);
            let mut bx = vec![0.0_f32; n];
            let mut by = vec![0.0_f32; n];
            let mut bz = vec![0.0_f32; n];
            let code = unsafe {
                crate::ffi::cuda_mhd_flux_freeze(
                    soa.ptype.as_ptr(),
                    soa.mass.as_ptr(),
                    soa.internal_energy.as_ptr(),
                    soa.h.as_ptr(),
                    soa.bx.as_ptr(),
                    soa.by.as_ptr(),
                    soa.bz.as_ptr(),
                    bx.as_mut_ptr(),
                    by.as_mut_ptr(),
                    bz.as_mut_ptr(),
                    n as i32,
                    gamma as f32,
                    beta_freeze as f32,
                    rho_ref as f32,
                )
            };
            check_kernel("cuda_mhd_flux_freeze", code)?;
            for (i, p) in particles.iter_mut().enumerate() {
                p.b_field = Vec3::new(bx[i] as f64, by[i] as f64, bz[i] as f64);
            }
            Ok(())
        }
    }

    pub fn try_mean_gas_density(&self, particles: &[Particle]) -> Result<f64, CudaExecutionError> {
        let n = particles.len();
        if n == 0 {
            return Ok(1.0);
        }

        #[cfg(cuda_unavailable)]
        {
            let _ = particles;
            Err(CudaUnavailable {
                availability: CudaPmSolver::availability(),
            }
            .into())
        }

        #[cfg(not(cuda_unavailable))]
        {
            let soa = MhdSoa::from_particles(particles);
            let mut rho = vec![0.0_f32; n];
            let mut count = vec![0.0_f32; n];
            let code = unsafe {
                crate::ffi::cuda_mhd_density_contrib(
                    soa.ptype.as_ptr(),
                    soa.mass.as_ptr(),
                    soa.h.as_ptr(),
                    rho.as_mut_ptr(),
                    count.as_mut_ptr(),
                    n as i32,
                )
            };
            check_kernel("cuda_mhd_density_contrib", code)?;
            let rho_sum: f64 = rho.iter().map(|&v| v as f64).sum();
            let n_gas: f64 = count.iter().map(|&v| v as f64).sum();
            Ok(if n_gas == 0.0 { 1.0 } else { rho_sum / n_gas })
        }
    }

    pub fn try_b_field_stats(
        &self,
        particles: &[Particle],
    ) -> Result<Option<BFieldStats>, CudaExecutionError> {
        let n = particles.len();
        if n == 0 {
            return Ok(None);
        }

        #[cfg(cuda_unavailable)]
        {
            let _ = particles;
            Err(CudaUnavailable {
                availability: CudaPmSolver::availability(),
            }
            .into())
        }

        #[cfg(not(cuda_unavailable))]
        {
            let soa = MhdSoa::from_particles(particles);
            let mut m = vec![0.0_f32; n];
            let mut mb = vec![0.0_f32; n];
            let mut mb2 = vec![0.0_f32; n];
            let mut bmag = vec![0.0_f32; n];
            let mut emag = vec![0.0_f32; n];
            let mut count = vec![0.0_f32; n];
            let code = unsafe {
                crate::ffi::cuda_mhd_b_stats_contrib(
                    soa.ptype.as_ptr(),
                    soa.mass.as_ptr(),
                    soa.bx.as_ptr(),
                    soa.by.as_ptr(),
                    soa.bz.as_ptr(),
                    m.as_mut_ptr(),
                    mb.as_mut_ptr(),
                    mb2.as_mut_ptr(),
                    bmag.as_mut_ptr(),
                    emag.as_mut_ptr(),
                    count.as_mut_ptr(),
                    n as i32,
                )
            };
            check_kernel("cuda_mhd_b_stats_contrib", code)?;
            let m_total: f64 = m.iter().map(|&v| v as f64).sum();
            let mb_sum: f64 = mb.iter().map(|&v| v as f64).sum();
            let mb2_sum: f64 = mb2.iter().map(|&v| v as f64).sum();
            let b_max = bmag.iter().map(|&v| v as f64).fold(0.0_f64, f64::max);
            let e_mag: f64 = emag.iter().map(|&v| v as f64).sum();
            let n_gas = count.iter().filter(|&&v| v > 0.0).count();
            if n_gas == 0 || m_total <= 0.0 {
                return Ok(None);
            }
            Ok(Some(BFieldStats {
                b_mean: mb_sum / m_total,
                b_rms: (mb2_sum / m_total).sqrt(),
                b_max,
                e_mag,
                n_gas,
            }))
        }
    }
}

#[cfg(not(cuda_unavailable))]
impl MhdSoa {
    fn from_particles(particles: &[Particle]) -> Self {
        let n = particles.len();
        let mut ptype = Vec::with_capacity(n);
        let mut mass = Vec::with_capacity(n);
        let mut internal_energy = Vec::with_capacity(n);
        let mut h = Vec::with_capacity(n);
        let mut bx = Vec::with_capacity(n);
        let mut by = Vec::with_capacity(n);
        let mut bz = Vec::with_capacity(n);

        for p in particles {
            ptype.push(match p.ptype {
                ParticleType::DarkMatter => 0,
                ParticleType::Gas => 1,
                ParticleType::Star => 2,
            });
            mass.push(p.mass as f32);
            internal_energy.push(p.internal_energy as f32);
            h.push(p.smoothing_length as f32);
            bx.push(p.b_field.x as f32);
            by.push(p.b_field.y as f32);
            bz.push(p.b_field.z as f32);
        }

        Self {
            ptype,
            mass,
            internal_energy,
            h,
            bx,
            by,
            bz,
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
