//! Kernels de enfriamiento radiativo via CUDA con buffers persistentes.
//!
//! Versión optimizada: `CudaCoolingSolver` retiene un [`CudaPool`] de buffers
//! device entre pasos, eliminando `cudaMalloc`/`cudaFree` por invocación.
//! Los buffers se redimensionan solo cuando el número de partículas crece.
//!
//! Replica la física de `gadget_ng_sph::cooling`:
//! AtomicHHe, MetalCooling, MetalTabular, UvBackground + supresión MHD.

use crate::pool::CudaPool;
use crate::{CudaExecutionError, CudaPmSolver, CudaUnavailable};
use gadget_ng_core::{CoolingKind, Particle, SphSection};
#[cfg(not(cuda_unavailable))]
use gadget_ng_core::{ParticleType, UvBackgroundModel};

/// Solver CUDA para cooling radiativo con buffers device persistentes.
pub struct CudaCoolingSolver {
    #[cfg(not(cuda_unavailable))]
    pool: CudaPool,
    #[cfg(cuda_unavailable)]
    _phantom: (),
}

impl std::fmt::Debug for CudaCoolingSolver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaCoolingSolver").finish()
    }
}

impl Clone for CudaCoolingSolver {
    fn clone(&self) -> Self {
        Self::try_new_checked().unwrap_or_else(|_| {
            #[cfg(not(cuda_unavailable))]
            {
                panic!("CudaCoolingSolver clone failed: CUDA not available");
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
            return Err(CudaExecutionError::Unavailable(CudaUnavailable {
                availability: CudaPmSolver::availability(),
            }));
        }

        #[cfg(not(cuda_unavailable))]
        {
            self.pool.ensure_capacity(n)?;
            self.pool.reset();

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

            // SAFETY: pool handle is valid, slots are freshly reset; all slices have length n.
            unsafe {
                let d_ptype = self.pool.upload_u8(0, &soa.ptype);
                let d_mass = self.pool.upload_f32(1, &soa.mass);
                let d_smoothing_length = self.pool.upload_f32(2, &soa.smoothing_length);
                let d_internal_energy = self.pool.upload_f32(3, &soa.internal_energy);
                let d_metallicity = self.pool.upload_f32(4, &soa.metallicity);
                let d_bx = self.pool.upload_f32(5, &soa.bx);
                let d_by = self.pool.upload_f32(6, &soa.by);
                let d_bz = self.pool.upload_f32(7, &soa.bz);

                let code = crate::ffi::cuda_cooling_apply(
                    d_ptype,
                    d_mass,
                    d_smoothing_length,
                    d_internal_energy,
                    d_metallicity,
                    d_bx,
                    d_by,
                    d_bz,
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
                );
                check_kernel("cuda_cooling_apply", code)?;

                let mut internal_energy_out = vec![0.0_f32; n];
                self.pool
                    .download_f32(&mut internal_energy_out, d_internal_energy)?;

                for (i, p) in particles.iter_mut().enumerate() {
                    p.internal_energy = internal_energy_out[i] as f64;
                }
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

#[cfg(cuda_unavailable)]
const _: () = {
    let _ = std::mem::size_of::<CudaCoolingSolver>();
};
