//! Kernels de enfriamiento radiativo via CUDA.
//!
//! Replica la física de `gadget_ng_sph::cooling`:
//! AtomicHHe, MetalCooling, MetalTabular, UvBackground + supresión MHD.

use crate::{CudaExecutionError, CudaPmSolver, CudaUnavailable};
use gadget_ng_core::{CoolingKind, Particle, SphSection};
#[cfg(not(cuda_unavailable))]
use gadget_ng_core::{ParticleType, UvBackgroundModel};

/// Solver CUDA para cooling radiativo.
#[derive(Debug, Clone, Copy)]
pub struct CudaCoolingSolver;

#[derive(Debug)]
#[cfg(not(cuda_unavailable))]
struct CoolSoa {
    ptype: Vec<u8>,
    mass: Vec<f32>,
    smoothing_length: Vec<f32>,
    internal_energy: Vec<f32>,
    metallicity: Vec<f32>,
    bx: Vec<f32>,
    by: Vec<f32>,
    bz: Vec<f32>,
}

impl CudaCoolingSolver {
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

    /// Aplica cooling a partículas de gas. Soporta f_mag=0 (sin MHD) y f_mag>0 (supresión por β).
    pub fn try_apply_cooling(
        &self,
        particles: &mut [Particle],
        cfg: &SphSection,
        dt: f64,
        redshift: f64,
        f_mag: f64,
    ) -> Result<(), CudaExecutionError> {
        let n = particles.len();
        if n == 0 || cfg.cooling == CoolingKind::None {
            return Ok(());
        }

        #[cfg(cuda_unavailable)]
        {
            let _ = (particles, cfg, dt, redshift, f_mag);
            Err(CudaUnavailable {
                availability: CudaPmSolver::availability(),
            }
            .into())
        }

        #[cfg(not(cuda_unavailable))]
        {
            let soa = CoolSoa::from_particles(particles);
            let cooling_kind: i32 = match cfg.cooling {
                CoolingKind::None => return Ok(()),
                CoolingKind::AtomicHHe => 1,
                CoolingKind::MetalCooling => 2,
                CoolingKind::MetalTabular => 3,
                CoolingKind::UvBackground => 4,
            };
            let uv_model: i32 = match cfg.uv_background_model {
                UvBackgroundModel::None => 0,
                UvBackgroundModel::Hm2012 => 1,
            };
            let code = unsafe {
                crate::ffi::cuda_cooling_apply(
                    soa.ptype.as_ptr(),
                    soa.mass.as_ptr(),
                    soa.smoothing_length.as_ptr(),
                    soa.internal_energy.as_ptr() as *mut f32,
                    soa.metallicity.as_ptr(),
                    soa.bx.as_ptr(),
                    soa.by.as_ptr(),
                    soa.bz.as_ptr(),
                    n as i32,
                    dt as f32,
                    cfg.gamma as f32,
                    cfg.t_floor_k as f32,
                    redshift as f32,
                    cooling_kind,
                    f_mag as f32,
                    cfg.reionization_redshift as f32,
                    uv_model,
                    cfg.self_shielding_nh_cm3 as f32,
                )
            };
            check_kernel("cuda_cooling_apply", code)?;
            for (i, p) in particles.iter_mut().enumerate() {
                p.internal_energy = soa.internal_energy[i] as f64;
            }
            Ok(())
        }
    }
}

#[cfg(not(cuda_unavailable))]
impl CoolSoa {
    fn from_particles(particles: &[Particle]) -> Self {
        let n = particles.len();
        let mut ptype = Vec::with_capacity(n);
        let mut mass = Vec::with_capacity(n);
        let mut smoothing_length = Vec::with_capacity(n);
        let mut internal_energy = Vec::with_capacity(n);
        let mut metallicity = Vec::with_capacity(n);
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
            smoothing_length.push(p.smoothing_length as f32);
            internal_energy.push(p.internal_energy as f32);
            metallicity.push(p.metallicity as f32);
            bx.push(p.b_field.x as f32);
            by.push(p.b_field.y as f32);
            bz.push(p.b_field.z as f32);
        }

        Self {
            ptype,
            mass,
            smoothing_length,
            internal_energy,
            metallicity,
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
