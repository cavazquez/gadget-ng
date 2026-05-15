//! Kernel de gas molecular HI→H₂ via CUDA con buffers persistentes.
//!
//! Versión optimizada: `CudaMolecularSolver` retiene un [`CudaPool`] de buffers
//! device entre pasos, eliminando `cudaMalloc`/`cudaFree` por invocación.
//! Los buffers se redimensionan solo cuando el número de partículas crece.
//!
//! Replica la física de `gadget_ng_sph::molecular_gas::update_h2_fraction_with_dust`.

use crate::pool::CudaPool;
use crate::{CudaExecutionError, CudaPmSolver, CudaUnavailable};
#[cfg(not(cuda_unavailable))]
use gadget_ng_core::ParticleType;
use gadget_ng_core::{DustSection, MolecularSection, Particle};

/// Solver CUDA para evolución de fracción H₂ con buffers device persistentes.
pub struct CudaMolecularSolver {
    #[cfg(not(cuda_unavailable))]
    pool: CudaPool,
    #[cfg(cuda_unavailable)]
    _phantom: (),
}

impl std::fmt::Debug for CudaMolecularSolver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaMolecularSolver").finish()
    }
}

impl Clone for CudaMolecularSolver {
    fn clone(&self) -> Self {
        Self::try_new_checked().unwrap_or_else(|_| {
            #[cfg(not(cuda_unavailable))]
            {
                panic!("CudaMolecularSolver clone failed: CUDA not available");
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
struct MolSoa {
    ptype: Vec<u8>,
    mass: Vec<f32>,
    smoothing_length: Vec<f32>,
    h2_fraction: Vec<f32>,
    dust_to_gas: Vec<f32>,
}

impl CudaMolecularSolver {
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

    /// Actualiza `h2_fraction` con o sin shielding de polvo.
    pub fn try_update_h2(
        &self,
        particles: &mut [Particle],
        mol_cfg: &MolecularSection,
        dust_cfg: Option<&DustSection>,
        dt: f64,
    ) -> Result<(), CudaExecutionError> {
        let n = particles.len();
        if n == 0 || !mol_cfg.enabled {
            return Ok(());
        }

        #[cfg(cuda_unavailable)]
        {
            let _ = (particles, mol_cfg, dust_cfg, dt);
            return Err(CudaExecutionError::Unavailable(CudaUnavailable {
                availability: CudaPmSolver::availability(),
            }));
        }

        #[cfg(not(cuda_unavailable))]
        {
            self.pool.ensure_capacity(n)?;
            self.pool.reset();

            let soa = MolSoa::from_particles(particles);
            let t_dissoc = 10.0_f32;
            let dust_enabled = dust_cfg.map_or(0, |d| if d.enabled { 1 } else { 0 });
            let (boost, k_dust, k_sil, k_gra, sil_f, gra_f, model) = if let Some(d) = dust_cfg {
                (
                    d.h2_shielding_boost as f32,
                    d.kappa_dust_uv as f32,
                    d.kappa_silicate_uv as f32,
                    d.kappa_graphite_uv as f32,
                    d.silicate_fraction as f32,
                    d.graphite_fraction as f32,
                    match d.species_model {
                        gadget_ng_core::DustSpeciesModel::Single => 0,
                        gadget_ng_core::DustSpeciesModel::SilicateGraphite => 1,
                    },
                )
            } else {
                (0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32, 1.0_f32, 0.0_f32, 0)
            };

            // SAFETY: pool handle is valid, slots are freshly reset; all slices have length n.
            unsafe {
                let d_ptype = self.pool.upload_u8(0, &soa.ptype);
                let d_mass = self.pool.upload_f32(1, &soa.mass);
                let d_smoothing_length = self.pool.upload_f32(2, &soa.smoothing_length);
                let d_h2_fraction = self.pool.upload_f32(3, &soa.h2_fraction);
                let d_dust_to_gas = self.pool.upload_f32(4, &soa.dust_to_gas);

                let code = crate::ffi::cuda_h2_update(
                    d_ptype,
                    d_mass,
                    d_smoothing_length,
                    d_h2_fraction,
                    d_dust_to_gas,
                    n as i32,
                    dt as f32,
                    mol_cfg.rho_h2_threshold as f32,
                    t_dissoc,
                    dust_enabled,
                    boost,
                    k_dust,
                    k_sil,
                    k_gra,
                    sil_f,
                    gra_f,
                    model,
                );
                check_kernel("cuda_h2_update", code)?;

                let mut h2_out = vec![0.0_f32; n];
                self.pool.download_f32(&mut h2_out, d_h2_fraction)?;

                for (i, p) in particles.iter_mut().enumerate() {
                    p.h2_fraction = h2_out[i] as f64;
                }
            }
            Ok(())
        }
    }
}

#[cfg(not(cuda_unavailable))]
impl MolSoa {
    fn from_particles(particles: &[Particle]) -> Self {
        let n = particles.len();
        let mut ptype = Vec::with_capacity(n);
        let mut mass = Vec::with_capacity(n);
        let mut smoothing_length = Vec::with_capacity(n);
        let mut h2_fraction = Vec::with_capacity(n);
        let mut dust_to_gas = Vec::with_capacity(n);

        for p in particles {
            ptype.push(match p.ptype {
                ParticleType::DarkMatter => 0,
                ParticleType::Gas => 1,
                ParticleType::Star => 2,
            });
            mass.push(p.mass as f32);
            smoothing_length.push(p.smoothing_length as f32);
            h2_fraction.push(p.h2_fraction as f32);
            dust_to_gas.push(p.dust_to_gas as f32);
        }

        Self {
            ptype,
            mass,
            smoothing_length,
            h2_fraction,
            dust_to_gas,
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
    let _ = std::mem::size_of::<CudaMolecularSolver>();
};
