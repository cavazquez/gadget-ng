//! Kernels RT locales via CUDA.

use crate::{CudaExecutionError, CudaPmSolver, CudaUnavailable};
use gadget_ng_core::Particle;
#[cfg(not(cuda_unavailable))]
use gadget_ng_core::ParticleType;
#[cfg(not(cuda_unavailable))]
use gadget_ng_rt::m1::C_KMS;
use gadget_ng_rt::m1::{M1Params, RadiationField};

/// Solver CUDA para reducciones/campos RT locales.
#[derive(Debug, Clone, Copy)]
pub struct CudaRtSolver;

impl CudaRtSolver {
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

    /// Calcula energía total, xi por celda y tasa de fotoionización en una pasada CUDA.
    pub fn try_field_diagnostics(
        &self,
        rad: &RadiationField,
        params: &M1Params,
        dv: f64,
    ) -> Result<(f64, Vec<f64>, Vec<f64>), CudaExecutionError> {
        let n = rad.n_cells();
        if n == 0 {
            return Ok((0.0, Vec::new(), Vec::new()));
        }

        #[cfg(cuda_unavailable)]
        {
            let _ = (rad, params, dv);
            Err(CudaUnavailable {
                availability: CudaPmSolver::availability(),
            }
            .into())
        }

        #[cfg(not(cuda_unavailable))]
        {
            let energy: Vec<f32> = rad.energy_density.iter().map(|&v| v as f32).collect();
            let flux_x: Vec<f32> = rad.flux_x.iter().map(|&v| v as f32).collect();
            let flux_y: Vec<f32> = rad.flux_y.iter().map(|&v| v as f32).collect();
            let flux_z: Vec<f32> = rad.flux_z.iter().map(|&v| v as f32).collect();
            let mut energy_contrib = vec![0.0_f32; n];
            let mut xi = vec![0.0_f32; n];
            let mut gamma = vec![0.0_f32; n];
            let c_red_code = C_KMS / params.c_red_factor;
            let c_red_cgs = C_KMS * 1.0e5 / params.c_red_factor;
            let code = unsafe {
                crate::ffi::cuda_rt_energy_xi_photoion(
                    energy.as_ptr(),
                    flux_x.as_ptr(),
                    flux_y.as_ptr(),
                    flux_z.as_ptr(),
                    energy_contrib.as_mut_ptr(),
                    xi.as_mut_ptr(),
                    gamma.as_mut_ptr(),
                    n as i32,
                    dv as f32,
                    c_red_code as f32,
                    c_red_cgs as f32,
                )
            };
            check_kernel("cuda_rt_energy_xi_photoion", code)?;
            Ok((
                energy_contrib.iter().map(|&v| v as f64).sum(),
                xi.into_iter().map(f64::from).collect(),
                gamma.into_iter().map(f64::from).collect(),
            ))
        }
    }

    /// Aplica fotoheating gas-partícula usando tasas `gamma_hi` por celda.
    pub fn try_apply_photoheating(
        &self,
        particles: &mut [Particle],
        rad: &RadiationField,
        gamma_hi: &[f64],
        dt: f64,
        box_size: f64,
    ) -> Result<(), CudaExecutionError> {
        let n = particles.len();
        if n == 0 {
            return Ok(());
        }
        assert_eq!(
            gamma_hi.len(),
            rad.n_cells(),
            "gamma_hi debe cubrir la grilla RT"
        );

        #[cfg(cuda_unavailable)]
        {
            let _ = (particles, rad, gamma_hi, dt, box_size);
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
            let px: Vec<f32> = particles.iter().map(|p| p.position.x as f32).collect();
            let py: Vec<f32> = particles.iter().map(|p| p.position.y as f32).collect();
            let pz: Vec<f32> = particles.iter().map(|p| p.position.z as f32).collect();
            let u_in: Vec<f32> = particles.iter().map(|p| p.internal_energy as f32).collect();
            let gamma: Vec<f32> = gamma_hi.iter().map(|&v| v as f32).collect();
            let mut u_out = vec![0.0_f32; n];
            let code = unsafe {
                crate::ffi::cuda_rt_photoheating(
                    ptype.as_ptr(),
                    px.as_ptr(),
                    py.as_ptr(),
                    pz.as_ptr(),
                    u_in.as_ptr(),
                    gamma.as_ptr(),
                    u_out.as_mut_ptr(),
                    n as i32,
                    rad.nx as i32,
                    rad.ny as i32,
                    rad.nz as i32,
                    box_size as f32,
                    dt as f32,
                )
            };
            check_kernel("cuda_rt_photoheating", code)?;
            for (p, &u) in particles.iter_mut().zip(&u_out) {
                p.internal_energy = u as f64;
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
