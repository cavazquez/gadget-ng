//! Kernels de evolución de polvo via CUDA con buffers persistentes.
//!
//! Versión optimizada: `CudaDustSolver` retiene un [`CudaPool`] de buffers
//! device entre pasos, eliminando `cudaMalloc`/`cudaFree` por invocación.
//! Los buffers se redimensionan solo cuando el número de partículas crece.
//!
//! Replica la física de `gadget_ng_sph::dust::update_dust` y `apply_dust_radiation_pressure_kick`.

use crate::pool::CudaPool;
use crate::{CudaExecutionError, CudaPmSolver, CudaUnavailable};
use gadget_ng_core::{DustSection, Particle};
#[cfg(not(cuda_unavailable))]
use gadget_ng_core::{ParticleType, Vec3};

/// Solver CUDA para evolución de polvo D/G con buffers device persistentes.
pub struct CudaDustSolver {
    #[cfg(not(cuda_unavailable))]
    pool: CudaPool,
    #[cfg(cuda_unavailable)]
    _phantom: (),
}

impl std::fmt::Debug for CudaDustSolver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaDustSolver").finish()
    }
}

impl Clone for CudaDustSolver {
    fn clone(&self) -> Self {
        Self::try_new_checked().unwrap_or_else(|_| {
            #[cfg(not(cuda_unavailable))]
            {
                panic!("CudaDustSolver clone failed: CUDA not available");
            }
            #[cfg(cuda_unavailable)]
            {
                Self { _phantom: () }
            }
        })
    }
}

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
            return Err(CudaExecutionError::Unavailable(CudaUnavailable {
                availability: CudaPmSolver::availability(),
            }));
        }

        #[cfg(not(cuda_unavailable))]
        {
            self.pool.ensure_capacity(n)?;
            self.pool.reset();

            let soa = DustSoa::from_particles(particles);

            // SAFETY: pool handle is valid, slots are freshly reset; all slices have length n.
            unsafe {
                let d_ptype = self.pool.upload_u8(0, &soa.ptype);
                let d_mass = self.pool.upload_f32(1, &soa.mass);
                let d_smoothing_length = self.pool.upload_f32(2, &soa.smoothing_length);
                let d_internal_energy = self.pool.upload_f32(3, &soa.internal_energy);
                let d_dust_to_gas = self.pool.upload_f32(4, &soa.dust_to_gas);
                let d_metallicity = self.pool.upload_f32(5, &soa.metallicity);

                let code = crate::ffi::cuda_dust_update(
                    d_ptype,
                    d_mass,
                    d_smoothing_length,
                    d_internal_energy,
                    d_dust_to_gas,
                    d_metallicity,
                    n as i32,
                    gamma as f32,
                    dt as f32,
                    cfg.d_to_g_max as f32,
                    cfg.tau_grow as f32,
                    cfg.t_destroy_k as f32,
                );
                check_kernel("cuda_dust_update", code)?;

                let mut dust_to_gas_out = vec![0.0_f32; n];
                self.pool
                    .download_f32(&mut dust_to_gas_out, d_dust_to_gas)?;

                for (i, p) in particles.iter_mut().enumerate() {
                    p.dust_to_gas = dust_to_gas_out[i] as f64;
                }
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
            let mass: Vec<f32> = particles.iter().map(|p| p.mass as f32).collect();
            let h: Vec<f32> = particles
                .iter()
                .map(|p| p.smoothing_length as f32)
                .collect();
            let dust_to_gas: Vec<f32> = particles.iter().map(|p| p.dust_to_gas as f32).collect();
            let vx: Vec<f32> = particles.iter().map(|p| p.velocity.x as f32).collect();
            let vy: Vec<f32> = particles.iter().map(|p| p.velocity.y as f32).collect();
            let vz: Vec<f32> = particles.iter().map(|p| p.velocity.z as f32).collect();
            let pos_z: Vec<f32> = particles.iter().map(|p| p.position.z as f32).collect();

            // SAFETY: pool handle is valid, slots are freshly reset; all slices have length n.
            unsafe {
                let d_ptype = self.pool.upload_u8(0, &ptype);
                let d_mass = self.pool.upload_f32(1, &mass);
                let d_h = self.pool.upload_f32(2, &h);
                let d_dust_to_gas = self.pool.upload_f32(3, &dust_to_gas);
                let d_vx = self.pool.upload_f32(4, &vx);
                let d_vy = self.pool.upload_f32(5, &vy);
                let d_vz = self.pool.upload_f32(6, &vz);
                let d_pos_z = self.pool.upload_f32(7, &pos_z);

                let code = crate::ffi::cuda_dust_radiation_pressure(
                    d_ptype,
                    d_mass,
                    d_h,
                    d_dust_to_gas,
                    d_vx,
                    d_vy,
                    d_vz,
                    d_pos_z,
                    n as i32,
                    dt as f32,
                    z_reference as f32,
                    cfg.radiation_pressure_kappa as f32,
                    cfg.radiation_pressure_j_uv as f32,
                    box_size as f32,
                );
                check_kernel("cuda_dust_radiation_pressure", code)?;

                let mut vx_out = vec![0.0_f32; n];
                let mut vy_out = vec![0.0_f32; n];
                let mut vz_out = vec![0.0_f32; n];
                self.pool.download_f32(&mut vx_out, d_vx)?;
                self.pool.download_f32(&mut vy_out, d_vy)?;
                self.pool.download_f32(&mut vz_out, d_vz)?;

                for (i, p) in particles.iter_mut().enumerate() {
                    p.velocity = Vec3::new(vx_out[i] as f64, vy_out[i] as f64, vz_out[i] as f64);
                }
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

#[cfg(cuda_unavailable)]
const _: () = {
    let _ = std::mem::size_of::<CudaDustSolver>();
};
