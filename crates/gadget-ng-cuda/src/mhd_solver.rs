//! Kernels MHD simples via CUDA.

use crate::{CudaExecutionError, CudaPmSolver, CudaUnavailable};
use gadget_ng_core::Particle;
#[cfg(not(cuda_unavailable))]
use gadget_ng_core::ParticleType;
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
    x: Vec<f32>,
    y: Vec<f32>,
    z: Vec<f32>,
    vx: Vec<f32>,
    vy: Vec<f32>,
    vz: Vec<f32>,
    rho: Vec<f32>,
    psi: Vec<f32>,
    cr: Vec<f32>,
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

    pub fn try_induction_resistivity(
        &self,
        particles: &mut [Particle],
        dt: f64,
        resistivity: f64,
        periodic_box: f64,
    ) -> Result<(), CudaExecutionError> {
        let n = particles.len();
        if n == 0 {
            return Ok(());
        }

        #[cfg(cuda_unavailable)]
        {
            let _ = (particles, dt, resistivity, periodic_box);
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
                crate::ffi::cuda_mhd_induction_resistivity(
                    soa.ptype.as_ptr(),
                    soa.x.as_ptr(),
                    soa.y.as_ptr(),
                    soa.z.as_ptr(),
                    soa.vx.as_ptr(),
                    soa.vy.as_ptr(),
                    soa.vz.as_ptr(),
                    soa.mass.as_ptr(),
                    soa.rho.as_ptr(),
                    soa.h.as_ptr(),
                    soa.bx.as_ptr(),
                    soa.by.as_ptr(),
                    soa.bz.as_ptr(),
                    bx.as_mut_ptr(),
                    by.as_mut_ptr(),
                    bz.as_mut_ptr(),
                    n as i32,
                    dt as f32,
                    resistivity as f32,
                    periodic_box as f32,
                )
            };
            check_kernel("cuda_mhd_induction_resistivity", code)?;
            for (i, p) in particles.iter_mut().enumerate() {
                p.b_field = Vec3::new(bx[i] as f64, by[i] as f64, bz[i] as f64);
            }
            Ok(())
        }
    }

    pub fn try_magnetic_forces(
        &self,
        particles: &[Particle],
        mu0: f64,
        periodic_box: f64,
    ) -> Result<Vec<Vec3>, CudaExecutionError> {
        let n = particles.len();
        if n == 0 {
            return Ok(Vec::new());
        }

        #[cfg(cuda_unavailable)]
        {
            let _ = (particles, mu0, periodic_box);
            Err(CudaUnavailable {
                availability: CudaPmSolver::availability(),
            }
            .into())
        }

        #[cfg(not(cuda_unavailable))]
        {
            let soa = MhdSoa::from_particles(particles);
            let mut ax = vec![0.0_f32; n];
            let mut ay = vec![0.0_f32; n];
            let mut az = vec![0.0_f32; n];
            let code = unsafe {
                crate::ffi::cuda_mhd_magnetic_forces(
                    soa.ptype.as_ptr(),
                    soa.x.as_ptr(),
                    soa.y.as_ptr(),
                    soa.z.as_ptr(),
                    soa.mass.as_ptr(),
                    soa.rho.as_ptr(),
                    soa.h.as_ptr(),
                    soa.bx.as_ptr(),
                    soa.by.as_ptr(),
                    soa.bz.as_ptr(),
                    ax.as_mut_ptr(),
                    ay.as_mut_ptr(),
                    az.as_mut_ptr(),
                    n as i32,
                    mu0 as f32,
                    periodic_box as f32,
                )
            };
            check_kernel("cuda_mhd_magnetic_forces", code)?;
            Ok((0..n)
                .map(|i| Vec3::new(ax[i] as f64, ay[i] as f64, az[i] as f64))
                .collect())
        }
    }

    pub fn try_dedner_cleaning(
        &self,
        particles: &mut [Particle],
        div_b: &[f32],
        dt: f64,
        ch: f64,
        cr: f64,
    ) -> Result<(), CudaExecutionError> {
        let n = particles.len();
        if n == 0 {
            return Ok(());
        }
        if div_b.len() != n {
            return Err(CudaExecutionError::KernelFailed {
                kernel: "cuda_mhd_dedner_cleaning",
                code: -2,
            });
        }

        #[cfg(cuda_unavailable)]
        {
            let _ = (particles, div_b, dt, ch, cr);
            Err(CudaUnavailable {
                availability: CudaPmSolver::availability(),
            }
            .into())
        }

        #[cfg(not(cuda_unavailable))]
        {
            let soa = MhdSoa::from_particles(particles);
            let mut psi = vec![0.0_f32; n];
            let mut bx = vec![0.0_f32; n];
            let mut by = vec![0.0_f32; n];
            let mut bz = vec![0.0_f32; n];
            let code = unsafe {
                crate::ffi::cuda_mhd_dedner_cleaning(
                    soa.ptype.as_ptr(),
                    div_b.as_ptr(),
                    soa.psi.as_ptr(),
                    soa.bx.as_ptr(),
                    soa.by.as_ptr(),
                    soa.bz.as_ptr(),
                    psi.as_mut_ptr(),
                    bx.as_mut_ptr(),
                    by.as_mut_ptr(),
                    bz.as_mut_ptr(),
                    n as i32,
                    dt as f32,
                    ch as f32,
                    cr as f32,
                )
            };
            check_kernel("cuda_mhd_dedner_cleaning", code)?;
            for (i, p) in particles.iter_mut().enumerate() {
                p.psi_div = psi[i] as f64;
                p.b_field = Vec3::new(bx[i] as f64, by[i] as f64, bz[i] as f64);
            }
            Ok(())
        }
    }

    pub fn try_scalar_diffusion(
        &self,
        particles: &[Particle],
        scalar: &[f32],
        dt: f64,
        kappa_par: f64,
        kappa_perp: f64,
    ) -> Result<Vec<f32>, CudaExecutionError> {
        let n = particles.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        if scalar.len() != n {
            return Err(CudaExecutionError::KernelFailed {
                kernel: "cuda_mhd_scalar_diffusion",
                code: -2,
            });
        }

        #[cfg(cuda_unavailable)]
        {
            let _ = (particles, scalar, dt, kappa_par, kappa_perp);
            Err(CudaUnavailable {
                availability: CudaPmSolver::availability(),
            }
            .into())
        }

        #[cfg(not(cuda_unavailable))]
        {
            let soa = MhdSoa::from_particles(particles);
            let mut out = vec![0.0_f32; n];
            let code = unsafe {
                crate::ffi::cuda_mhd_scalar_diffusion(
                    soa.ptype.as_ptr(),
                    scalar.as_ptr(),
                    soa.bx.as_ptr(),
                    soa.by.as_ptr(),
                    soa.bz.as_ptr(),
                    out.as_mut_ptr(),
                    n as i32,
                    dt as f32,
                    kappa_par as f32,
                    kappa_perp as f32,
                )
            };
            check_kernel("cuda_mhd_scalar_diffusion", code)?;
            Ok(out)
        }
    }

    pub fn try_braginskii_viscosity(
        &self,
        particles: &mut [Particle],
        dt: f64,
        eta: f64,
    ) -> Result<(), CudaExecutionError> {
        let n = particles.len();
        if n == 0 {
            return Ok(());
        }

        #[cfg(cuda_unavailable)]
        {
            let _ = (particles, dt, eta);
            Err(CudaUnavailable {
                availability: CudaPmSolver::availability(),
            }
            .into())
        }

        #[cfg(not(cuda_unavailable))]
        {
            let soa = MhdSoa::from_particles(particles);
            let mut vx = vec![0.0_f32; n];
            let mut vy = vec![0.0_f32; n];
            let mut vz = vec![0.0_f32; n];
            let code = unsafe {
                crate::ffi::cuda_mhd_braginskii_viscosity(
                    soa.ptype.as_ptr(),
                    soa.vx.as_ptr(),
                    soa.vy.as_ptr(),
                    soa.vz.as_ptr(),
                    soa.bx.as_ptr(),
                    soa.by.as_ptr(),
                    soa.bz.as_ptr(),
                    vx.as_mut_ptr(),
                    vy.as_mut_ptr(),
                    vz.as_mut_ptr(),
                    n as i32,
                    dt as f32,
                    eta as f32,
                )
            };
            check_kernel("cuda_mhd_braginskii_viscosity", code)?;
            for (i, p) in particles.iter_mut().enumerate() {
                p.velocity = Vec3::new(vx[i] as f64, vy[i] as f64, vz[i] as f64);
            }
            Ok(())
        }
    }

    pub fn try_reconnection_streaming_dynamo(
        &self,
        particles: &mut [Particle],
        dt: f64,
        stream_coeff: f64,
        reconnection_frac: f64,
        dynamo_alpha: f64,
    ) -> Result<(), CudaExecutionError> {
        let n = particles.len();
        if n == 0 {
            return Ok(());
        }

        #[cfg(cuda_unavailable)]
        {
            let _ = (particles, dt, stream_coeff, reconnection_frac, dynamo_alpha);
            Err(CudaUnavailable {
                availability: CudaPmSolver::availability(),
            }
            .into())
        }

        #[cfg(not(cuda_unavailable))]
        {
            let soa = MhdSoa::from_particles(particles);
            let mut cr = vec![0.0_f32; n];
            let mut bx = vec![0.0_f32; n];
            let mut by = vec![0.0_f32; n];
            let mut bz = vec![0.0_f32; n];
            let mut u = vec![0.0_f32; n];
            let code = unsafe {
                crate::ffi::cuda_mhd_reconnection_streaming_dynamo(
                    soa.ptype.as_ptr(),
                    soa.cr.as_ptr(),
                    soa.bx.as_ptr(),
                    soa.by.as_ptr(),
                    soa.bz.as_ptr(),
                    soa.internal_energy.as_ptr(),
                    cr.as_mut_ptr(),
                    bx.as_mut_ptr(),
                    by.as_mut_ptr(),
                    bz.as_mut_ptr(),
                    u.as_mut_ptr(),
                    n as i32,
                    dt as f32,
                    stream_coeff as f32,
                    reconnection_frac as f32,
                    dynamo_alpha as f32,
                )
            };
            check_kernel("cuda_mhd_reconnection_streaming_dynamo", code)?;
            for (i, p) in particles.iter_mut().enumerate() {
                p.cr_energy = cr[i] as f64;
                p.internal_energy = u[i] as f64;
                p.b_field = Vec3::new(bx[i] as f64, by[i] as f64, bz[i] as f64);
            }
            Ok(())
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
        let mut x = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        let mut z = Vec::with_capacity(n);
        let mut vx = Vec::with_capacity(n);
        let mut vy = Vec::with_capacity(n);
        let mut vz = Vec::with_capacity(n);
        let mut rho = Vec::with_capacity(n);
        let mut psi = Vec::with_capacity(n);
        let mut cr = Vec::with_capacity(n);
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
            let h_i = p.smoothing_length.max(1.0e-12);
            h.push(h_i as f32);
            x.push(p.position.x as f32);
            y.push(p.position.y as f32);
            z.push(p.position.z as f32);
            vx.push(p.velocity.x as f32);
            vy.push(p.velocity.y as f32);
            vz.push(p.velocity.z as f32);
            rho.push((p.mass / (h_i * h_i * h_i)) as f32);
            psi.push(p.psi_div as f32);
            cr.push(p.cr_energy as f32);
            bx.push(p.b_field.x as f32);
            by.push(p.b_field.y as f32);
            bz.push(p.b_field.z as f32);
        }

        Self {
            ptype,
            mass,
            internal_energy,
            h,
            x,
            y,
            z,
            vx,
            vy,
            vz,
            rho,
            psi,
            cr,
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
