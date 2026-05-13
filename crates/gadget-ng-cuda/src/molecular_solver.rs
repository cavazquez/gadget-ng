//! Kernel de gas molecular HI→H₂ via CUDA.
//!
//! Replica la física de `gadget_ng_sph::molecular_gas::update_h2_fraction_with_dust`.

use crate::{CudaExecutionError, CudaPmSolver, CudaUnavailable};
#[cfg(not(cuda_unavailable))]
use gadget_ng_core::ParticleType;
use gadget_ng_core::{DustSection, MolecularSection, Particle};

/// Solver CUDA para evolución de fracción H₂.
#[derive(Debug, Clone, Copy)]
pub struct CudaMolecularSolver;

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
        if CudaPmSolver::is_available() {
            Ok(Self)
        } else {
            Err(CudaUnavailable {
                availability: CudaPmSolver::availability(),
            })
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
            Err(CudaUnavailable {
                availability: CudaPmSolver::availability(),
            }
            .into())
        }

        #[cfg(not(cuda_unavailable))]
        {
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
            let code = unsafe {
                crate::ffi::cuda_h2_update(
                    soa.ptype.as_ptr(),
                    soa.mass.as_ptr(),
                    soa.smoothing_length.as_ptr(),
                    soa.h2_fraction.as_ptr() as *mut f32,
                    soa.dust_to_gas.as_ptr(),
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
                )
            };
            check_kernel("cuda_h2_update", code)?;
            for (i, p) in particles.iter_mut().enumerate() {
                p.h2_fraction = soa.h2_fraction[i] as f64;
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
