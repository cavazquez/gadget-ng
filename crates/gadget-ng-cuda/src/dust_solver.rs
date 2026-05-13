//! Kernels de evolución de polvo via CUDA.
//!
//! Replica la física de `gadget_ng_sph::dust::update_dust` y `apply_dust_radiation_pressure_kick`.

use crate::{CudaExecutionError, CudaPmSolver, CudaUnavailable};
use gadget_ng_core::{DustSection, Particle};
#[cfg(not(cuda_unavailable))]
use gadget_ng_core::{ParticleType, Vec3};

/// Solver CUDA para evolución de polvo D/G.
#[derive(Debug, Clone, Copy)]
pub struct CudaDustSolver;

#[derive(Debug)]
#[cfg(not(cuda_unavailable))]
struct DustSoa {
    ptype: Vec<u8>,
    mass: Vec<f32>,
    smoothing_length: Vec<f32>,
    internal_energy: Vec<f32>,
    dust_to_gas: Vec<f32>,
    metallicity: Vec<f32>,
}

impl CudaDustSolver {
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

    /// Actualiza `dust_to_gas` (acreción + sputtering).
    pub fn try_update_dust(
        &self,
        particles: &mut [Particle],
        cfg: &DustSection,
        gamma: f64,
        dt: f64,
    ) -> Result<(), CudaExecutionError> {
        let n = particles.len();
        if n == 0 || !cfg.enabled {
            return Ok(());
        }

        #[cfg(cuda_unavailable)]
        {
            let _ = (particles, cfg, gamma, dt);
            Err(CudaUnavailable {
                availability: CudaPmSolver::availability(),
            }
            .into())
        }

        #[cfg(not(cuda_unavailable))]
        {
            let soa = DustSoa::from_particles(particles);
            let code = unsafe {
                crate::ffi::cuda_dust_update(
                    soa.ptype.as_ptr(),
                    soa.mass.as_ptr(),
                    soa.smoothing_length.as_ptr(),
                    soa.internal_energy.as_ptr(),
                    soa.dust_to_gas.as_ptr() as *mut f32,
                    soa.metallicity.as_ptr(),
                    n as i32,
                    gamma as f32,
                    dt as f32,
                    cfg.d_to_g_max as f32,
                    cfg.tau_grow as f32,
                    cfg.t_destroy_k as f32,
                )
            };
            check_kernel("cuda_dust_update", code)?;
            for (i, p) in particles.iter_mut().enumerate() {
                p.dust_to_gas = soa.dust_to_gas[i] as f64;
            }
            Ok(())
        }
    }

    /// Aplica impulso de presión de radiación sobre polvo.
    pub fn try_apply_radiation_pressure(
        &self,
        particles: &mut [Particle],
        cfg: &DustSection,
        z_reference: f64,
        dt: f64,
        box_size: f64,
    ) -> Result<(), CudaExecutionError> {
        let n = particles.len();
        if n == 0 || !cfg.enabled || !cfg.radiation_pressure_enabled {
            return Ok(());
        }

        #[cfg(cuda_unavailable)]
        {
            let _ = (particles, cfg, z_reference, dt, box_size);
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
            let mass: Vec<f32> = particles.iter().map(|p| p.mass as f32).collect();
            let h: Vec<f32> = particles
                .iter()
                .map(|p| p.smoothing_length as f32)
                .collect();
            let dust_to_gas: Vec<f32> = particles.iter().map(|p| p.dust_to_gas as f32).collect();
            let mut vx: Vec<f32> = particles.iter().map(|p| p.velocity.x as f32).collect();
            let mut vy: Vec<f32> = particles.iter().map(|p| p.velocity.y as f32).collect();
            let mut vz: Vec<f32> = particles.iter().map(|p| p.velocity.z as f32).collect();
            let mut pos_z: Vec<f32> = particles.iter().map(|p| p.position.z as f32).collect();

            let code = unsafe {
                crate::ffi::cuda_dust_radiation_pressure(
                    ptype.as_ptr(),
                    mass.as_ptr(),
                    h.as_ptr(),
                    dust_to_gas.as_ptr(),
                    vx.as_mut_ptr(),
                    vy.as_mut_ptr(),
                    vz.as_mut_ptr(),
                    pos_z.as_mut_ptr(),
                    n as i32,
                    dt as f32,
                    z_reference as f32,
                    cfg.radiation_pressure_kappa as f32,
                    cfg.radiation_pressure_j_uv as f32,
                    box_size as f32,
                )
            };
            check_kernel("cuda_dust_radiation_pressure", code)?;
            for (i, p) in particles.iter_mut().enumerate() {
                p.velocity = Vec3::new(vx[i] as f64, vy[i] as f64, vz[i] as f64);
            }
            Ok(())
        }
    }
}

#[cfg(not(cuda_unavailable))]
impl DustSoa {
    fn from_particles(particles: &[Particle]) -> Self {
        let n = particles.len();
        let mut ptype = Vec::with_capacity(n);
        let mut mass = Vec::with_capacity(n);
        let mut smoothing_length = Vec::with_capacity(n);
        let mut internal_energy = Vec::with_capacity(n);
        let mut dust_to_gas = Vec::with_capacity(n);
        let mut metallicity = Vec::with_capacity(n);

        for p in particles {
            ptype.push(match p.ptype {
                ParticleType::DarkMatter => 0,
                ParticleType::Gas => 1,
                ParticleType::Star => 2,
            });
            mass.push(p.mass as f32);
            smoothing_length.push(p.smoothing_length as f32);
            internal_energy.push(p.internal_energy as f32);
            dust_to_gas.push(p.dust_to_gas as f32);
            metallicity.push(p.metallicity as f32);
        }

        Self {
            ptype,
            mass,
            smoothing_length,
            internal_energy,
            dust_to_gas,
            metallicity,
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
