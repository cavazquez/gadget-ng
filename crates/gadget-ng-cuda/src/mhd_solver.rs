//! Kernels MHD simples via CUDA con buffers persistentes.
//!
//! Versión optimizada: `CudaMhdSolver` retiene un [`CudaPool`] de buffers
//! device entre pasos, eliminando `cudaMalloc`/`cudaFree` por invocación.
//! Los buffers se redimensionan solo cuando el número de partículas crece.

use crate::pool::CudaPool;
use crate::{CudaExecutionError, CudaPmSolver, CudaUnavailable};
use gadget_ng_core::Particle;
#[cfg(not(cuda_unavailable))]
use gadget_ng_core::ParticleType;
use gadget_ng_core::Vec3;
use gadget_ng_mhd::stats::BFieldStats;

/// Solver CUDA para kernels MHD locales y reducciones simples con buffers device persistentes.
///
/// Se construye con [`CudaMhdSolver::try_new`] y se mantiene vivo durante
/// toda la simulación. Los buffers device se reutilizan entre pasos de
/// tiempo; solo se redimensionan cuando el número de partículas excede
/// la capacidad actual.
pub struct CudaMhdSolver {
    #[cfg(not(cuda_unavailable))]
    pool: CudaPool,
    #[cfg(cuda_unavailable)]
    _phantom: (),
}

impl std::fmt::Debug for CudaMhdSolver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaMhdSolver").finish()
    }
}

impl Clone for CudaMhdSolver {
    fn clone(&self) -> Self {
        Self::try_new_checked().unwrap_or_else(|_| {
            #[cfg(not(cuda_unavailable))]
            {
                panic!("CudaMhdSolver clone failed: CUDA not available");
            }
            #[cfg(cuda_unavailable)]
            {
                Self { _phantom: () }
            }
        })
    }
}

impl CudaMhdSolver {
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
            return Err(CudaExecutionError::Unavailable(CudaUnavailable {
                availability: CudaPmSolver::availability(),
            }));
        }

        #[cfg(not(cuda_unavailable))]
        {
            self.pool.ensure_capacity(n)?;
            self.pool.reset();

            let soa = MhdSoa::from_particles(particles);

            // SAFETY: pool handle is valid, slots are freshly reset; all slices have length n.
            unsafe {
                let d_ptype = self.pool.upload_u8(0, &soa.ptype);
                let d_mass = self.pool.upload_f32(1, &soa.mass);
                let d_internal_energy = self.pool.upload_f32(2, &soa.internal_energy);
                let d_h = self.pool.upload_f32(3, &soa.h);
                let d_bx_in = self.pool.upload_f32(4, &soa.bx);
                let d_by_in = self.pool.upload_f32(5, &soa.by);
                let d_bz_in = self.pool.upload_f32(6, &soa.bz);
                let d_bx_out = self.pool.alloc_f32(7, n);
                let d_by_out = self.pool.alloc_f32(8, n);
                let d_bz_out = self.pool.alloc_f32(9, n);

                let code = crate::ffi::cuda_mhd_flux_freeze(
                    d_ptype,
                    d_mass,
                    d_internal_energy,
                    d_h,
                    d_bx_in,
                    d_by_in,
                    d_bz_in,
                    d_bx_out,
                    d_by_out,
                    d_bz_out,
                    n as i32,
                    gamma as f32,
                    beta_freeze as f32,
                    rho_ref as f32,
                );
                check_kernel("cuda_mhd_flux_freeze", code)?;

                let mut bx = vec![0.0_f32; n];
                let mut by = vec![0.0_f32; n];
                let mut bz = vec![0.0_f32; n];
                self.pool.download_f32(&mut bx, d_bx_out)?;
                self.pool.download_f32(&mut by, d_by_out)?;
                self.pool.download_f32(&mut bz, d_bz_out)?;

                for (i, p) in particles.iter_mut().enumerate() {
                    p.b_field = Vec3::new(bx[i] as f64, by[i] as f64, bz[i] as f64);
                }
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
            return Err(CudaExecutionError::Unavailable(CudaUnavailable {
                availability: CudaPmSolver::availability(),
            }));
        }

        #[cfg(not(cuda_unavailable))]
        {
            self.pool.ensure_capacity(n)?;
            self.pool.reset();

            let soa = MhdSoa::from_particles(particles);

            unsafe {
                let d_ptype = self.pool.upload_u8(0, &soa.ptype);
                let d_mass = self.pool.upload_f32(1, &soa.mass);
                let d_h = self.pool.upload_f32(2, &soa.h);
                let d_rho = self.pool.alloc_f32(3, n);
                let d_count = self.pool.alloc_f32(4, n);

                let code = crate::ffi::cuda_mhd_density_contrib(
                    d_ptype, d_mass, d_h, d_rho, d_count, n as i32,
                );
                check_kernel("cuda_mhd_density_contrib", code)?;

                let mut rho = vec![0.0_f32; n];
                let mut count = vec![0.0_f32; n];
                self.pool.download_f32(&mut rho, d_rho)?;
                self.pool.download_f32(&mut count, d_count)?;

                let rho_sum: f64 = rho.iter().map(|&v| v as f64).sum();
                let n_gas: f64 = count.iter().map(|&v| v as f64).sum();
                Ok(if n_gas == 0.0 { 1.0 } else { rho_sum / n_gas })
            }
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
            return Err(CudaExecutionError::Unavailable(CudaUnavailable {
                availability: CudaPmSolver::availability(),
            }));
        }

        #[cfg(not(cuda_unavailable))]
        {
            self.pool.ensure_capacity(n)?;
            self.pool.reset();

            let soa = MhdSoa::from_particles(particles);

            unsafe {
                let d_ptype = self.pool.upload_u8(0, &soa.ptype);
                let d_mass = self.pool.upload_f32(1, &soa.mass);
                let d_bx = self.pool.upload_f32(2, &soa.bx);
                let d_by = self.pool.upload_f32(3, &soa.by);
                let d_bz = self.pool.upload_f32(4, &soa.bz);
                let d_m = self.pool.alloc_f32(5, n);
                let d_mb = self.pool.alloc_f32(6, n);
                let d_mb2 = self.pool.alloc_f32(7, n);
                let d_bmag = self.pool.alloc_f32(8, n);
                let d_emag = self.pool.alloc_f32(9, n);
                let d_count = self.pool.alloc_f32(10, n);

                let code = crate::ffi::cuda_mhd_b_stats_contrib(
                    d_ptype, d_mass, d_bx, d_by, d_bz, d_m, d_mb, d_mb2, d_bmag, d_emag, d_count,
                    n as i32,
                );
                check_kernel("cuda_mhd_b_stats_contrib", code)?;

                let mut m = vec![0.0_f32; n];
                let mut mb = vec![0.0_f32; n];
                let mut mb2 = vec![0.0_f32; n];
                let mut bmag = vec![0.0_f32; n];
                let mut emag = vec![0.0_f32; n];
                let mut count = vec![0.0_f32; n];
                self.pool.download_f32(&mut m, d_m)?;
                self.pool.download_f32(&mut mb, d_mb)?;
                self.pool.download_f32(&mut mb2, d_mb2)?;
                self.pool.download_f32(&mut bmag, d_bmag)?;
                self.pool.download_f32(&mut emag, d_emag)?;
                self.pool.download_f32(&mut count, d_count)?;

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
            return Err(CudaExecutionError::Unavailable(CudaUnavailable {
                availability: CudaPmSolver::availability(),
            }));
        }

        #[cfg(not(cuda_unavailable))]
        {
            self.pool.ensure_capacity(n)?;
            self.pool.reset();

            let soa = MhdSoa::from_particles(particles);

            unsafe {
                let d_ptype = self.pool.upload_u8(0, &soa.ptype);
                let d_x = self.pool.upload_f32(1, &soa.x);
                let d_y = self.pool.upload_f32(2, &soa.y);
                let d_z = self.pool.upload_f32(3, &soa.z);
                let d_vx = self.pool.upload_f32(4, &soa.vx);
                let d_vy = self.pool.upload_f32(5, &soa.vy);
                let d_vz = self.pool.upload_f32(6, &soa.vz);
                let d_mass = self.pool.upload_f32(7, &soa.mass);
                let d_rho = self.pool.upload_f32(8, &soa.rho);
                let d_h = self.pool.upload_f32(9, &soa.h);
                let d_bx_in = self.pool.upload_f32(10, &soa.bx);
                let d_by_in = self.pool.upload_f32(11, &soa.by);
                let d_bz_in = self.pool.upload_f32(12, &soa.bz);
                let d_bx_out = self.pool.alloc_f32(13, n);
                let d_by_out = self.pool.alloc_f32(14, n);
                let d_bz_out = self.pool.alloc_f32(15, n);

                let code = crate::ffi::cuda_mhd_induction_resistivity(
                    d_ptype,
                    d_x,
                    d_y,
                    d_z,
                    d_vx,
                    d_vy,
                    d_vz,
                    d_mass,
                    d_rho,
                    d_h,
                    d_bx_in,
                    d_by_in,
                    d_bz_in,
                    d_bx_out,
                    d_by_out,
                    d_bz_out,
                    n as i32,
                    dt as f32,
                    resistivity as f32,
                    periodic_box as f32,
                );
                check_kernel("cuda_mhd_induction_resistivity", code)?;

                let mut bx = vec![0.0_f32; n];
                let mut by = vec![0.0_f32; n];
                let mut bz = vec![0.0_f32; n];
                self.pool.download_f32(&mut bx, d_bx_out)?;
                self.pool.download_f32(&mut by, d_by_out)?;
                self.pool.download_f32(&mut bz, d_bz_out)?;

                for (i, p) in particles.iter_mut().enumerate() {
                    p.b_field = Vec3::new(bx[i] as f64, by[i] as f64, bz[i] as f64);
                }
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
            return Err(CudaExecutionError::Unavailable(CudaUnavailable {
                availability: CudaPmSolver::availability(),
            }));
        }

        #[cfg(not(cuda_unavailable))]
        {
            self.pool.ensure_capacity(n)?;
            self.pool.reset();

            let soa = MhdSoa::from_particles(particles);

            unsafe {
                let d_ptype = self.pool.upload_u8(0, &soa.ptype);
                let d_x = self.pool.upload_f32(1, &soa.x);
                let d_y = self.pool.upload_f32(2, &soa.y);
                let d_z = self.pool.upload_f32(3, &soa.z);
                let d_mass = self.pool.upload_f32(4, &soa.mass);
                let d_rho = self.pool.upload_f32(5, &soa.rho);
                let d_h = self.pool.upload_f32(6, &soa.h);
                let d_bx = self.pool.upload_f32(7, &soa.bx);
                let d_by = self.pool.upload_f32(8, &soa.by);
                let d_bz = self.pool.upload_f32(9, &soa.bz);
                let d_ax = self.pool.alloc_f32(10, n);
                let d_ay = self.pool.alloc_f32(11, n);
                let d_az = self.pool.alloc_f32(12, n);

                let code = crate::ffi::cuda_mhd_magnetic_forces(
                    d_ptype,
                    d_x,
                    d_y,
                    d_z,
                    d_mass,
                    d_rho,
                    d_h,
                    d_bx,
                    d_by,
                    d_bz,
                    d_ax,
                    d_ay,
                    d_az,
                    n as i32,
                    mu0 as f32,
                    periodic_box as f32,
                );
                check_kernel("cuda_mhd_magnetic_forces", code)?;

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
            return Err(CudaExecutionError::Unavailable(CudaUnavailable {
                availability: CudaPmSolver::availability(),
            }));
        }

        #[cfg(not(cuda_unavailable))]
        {
            self.pool.ensure_capacity(n)?;
            self.pool.reset();

            let soa = MhdSoa::from_particles(particles);

            unsafe {
                let d_ptype = self.pool.upload_u8(0, &soa.ptype);
                let d_div_b = self.pool.upload_f32(1, div_b);
                let d_psi = self.pool.upload_f32(2, &soa.psi);
                let d_bx_in = self.pool.upload_f32(3, &soa.bx);
                let d_by_in = self.pool.upload_f32(4, &soa.by);
                let d_bz_in = self.pool.upload_f32(5, &soa.bz);
                let d_psi_out = self.pool.alloc_f32(6, n);
                let d_bx_out = self.pool.alloc_f32(7, n);
                let d_by_out = self.pool.alloc_f32(8, n);
                let d_bz_out = self.pool.alloc_f32(9, n);

                let code = crate::ffi::cuda_mhd_dedner_cleaning(
                    d_ptype, d_div_b, d_psi, d_bx_in, d_by_in, d_bz_in, d_psi_out, d_bx_out,
                    d_by_out, d_bz_out, n as i32, dt as f32, ch as f32, cr as f32,
                );
                check_kernel("cuda_mhd_dedner_cleaning", code)?;

                let mut psi = vec![0.0_f32; n];
                let mut bx = vec![0.0_f32; n];
                let mut by = vec![0.0_f32; n];
                let mut bz = vec![0.0_f32; n];
                self.pool.download_f32(&mut psi, d_psi_out)?;
                self.pool.download_f32(&mut bx, d_bx_out)?;
                self.pool.download_f32(&mut by, d_by_out)?;
                self.pool.download_f32(&mut bz, d_bz_out)?;

                for (i, p) in particles.iter_mut().enumerate() {
                    p.psi_div = psi[i] as f64;
                    p.b_field = Vec3::new(bx[i] as f64, by[i] as f64, bz[i] as f64);
                }
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
            return Err(CudaExecutionError::Unavailable(CudaUnavailable {
                availability: CudaPmSolver::availability(),
            }));
        }

        #[cfg(not(cuda_unavailable))]
        {
            self.pool.ensure_capacity(n)?;
            self.pool.reset();

            let soa = MhdSoa::from_particles(particles);

            unsafe {
                let d_ptype = self.pool.upload_u8(0, &soa.ptype);
                let d_scalar = self.pool.upload_f32(1, scalar);
                let d_bx = self.pool.upload_f32(2, &soa.bx);
                let d_by = self.pool.upload_f32(3, &soa.by);
                let d_bz = self.pool.upload_f32(4, &soa.bz);
                let d_out = self.pool.alloc_f32(5, n);

                let code = crate::ffi::cuda_mhd_scalar_diffusion(
                    d_ptype,
                    d_scalar,
                    d_bx,
                    d_by,
                    d_bz,
                    d_out,
                    n as i32,
                    dt as f32,
                    kappa_par as f32,
                    kappa_perp as f32,
                );
                check_kernel("cuda_mhd_scalar_diffusion", code)?;

                let mut out = vec![0.0_f32; n];
                self.pool.download_f32(&mut out, d_out)?;

                Ok(out)
            }
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
            return Err(CudaExecutionError::Unavailable(CudaUnavailable {
                availability: CudaPmSolver::availability(),
            }));
        }

        #[cfg(not(cuda_unavailable))]
        {
            self.pool.ensure_capacity(n)?;
            self.pool.reset();

            let soa = MhdSoa::from_particles(particles);

            unsafe {
                let d_ptype = self.pool.upload_u8(0, &soa.ptype);
                let d_vx = self.pool.upload_f32(1, &soa.vx);
                let d_vy = self.pool.upload_f32(2, &soa.vy);
                let d_vz = self.pool.upload_f32(3, &soa.vz);
                let d_bx = self.pool.upload_f32(4, &soa.bx);
                let d_by = self.pool.upload_f32(5, &soa.by);
                let d_bz = self.pool.upload_f32(6, &soa.bz);
                let d_vx_out = self.pool.alloc_f32(7, n);
                let d_vy_out = self.pool.alloc_f32(8, n);
                let d_vz_out = self.pool.alloc_f32(9, n);

                let code = crate::ffi::cuda_mhd_braginskii_viscosity(
                    d_ptype, d_vx, d_vy, d_vz, d_bx, d_by, d_bz, d_vx_out, d_vy_out, d_vz_out,
                    n as i32, dt as f32, eta as f32,
                );
                check_kernel("cuda_mhd_braginskii_viscosity", code)?;

                let mut vx = vec![0.0_f32; n];
                let mut vy = vec![0.0_f32; n];
                let mut vz = vec![0.0_f32; n];
                self.pool.download_f32(&mut vx, d_vx_out)?;
                self.pool.download_f32(&mut vy, d_vy_out)?;
                self.pool.download_f32(&mut vz, d_vz_out)?;

                for (i, p) in particles.iter_mut().enumerate() {
                    p.velocity = Vec3::new(vx[i] as f64, vy[i] as f64, vz[i] as f64);
                }
            }
            Ok(())
        }
    }

    /// CR streaming O(N²): actualiza `cr_energy` con pérdidas compresional + streaming.
    ///
    /// Replica `gadget_ng_mhd::streaming::streaming_crk` en GPU con aritmética f32.
    /// La tolerancia en smoke tests es 5% (divergencia f32/f64 en la suma SPH de div_v).
    pub fn try_cr_streaming(
        &self,
        particles: &mut [Particle],
        dt: f64,
        streaming_coeff: f64,
        periodic_box: Option<f64>,
    ) -> Result<(), CudaExecutionError> {
        let n = particles.len();
        if n == 0 {
            return Ok(());
        }

        #[cfg(cuda_unavailable)]
        {
            let _ = (particles, dt, streaming_coeff, periodic_box);
            return Err(CudaExecutionError::Unavailable(CudaUnavailable {
                availability: CudaPmSolver::availability(),
            }));
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
            let vx: Vec<f32> = particles.iter().map(|p| p.velocity.x as f32).collect();
            let vy: Vec<f32> = particles.iter().map(|p| p.velocity.y as f32).collect();
            let vz: Vec<f32> = particles.iter().map(|p| p.velocity.z as f32).collect();
            let mass: Vec<f32> = particles.iter().map(|p| p.mass as f32).collect();
            let h_sml: Vec<f32> = particles
                .iter()
                .map(|p| p.smoothing_length.max(1e-10) as f32)
                .collect();
            let cr_in: Vec<f32> = particles.iter().map(|p| p.cr_energy as f32).collect();
            let bx: Vec<f32> = particles.iter().map(|p| p.b_field.x as f32).collect();
            let by: Vec<f32> = particles.iter().map(|p| p.b_field.y as f32).collect();
            let bz: Vec<f32> = particles.iter().map(|p| p.b_field.z as f32).collect();
            let mut cr_out = vec![0.0_f32; n];
            let pbox = periodic_box.unwrap_or(0.0) as f32;

            // SAFETY: todos los slices son válidos y de longitud n.
            let code = unsafe {
                crate::ffi::cuda_mhd_cr_streaming(
                    ptype.as_ptr(),
                    px.as_ptr(), py.as_ptr(), pz.as_ptr(),
                    vx.as_ptr(), vy.as_ptr(), vz.as_ptr(),
                    mass.as_ptr(), h_sml.as_ptr(),
                    cr_in.as_ptr(),
                    bx.as_ptr(), by.as_ptr(), bz.as_ptr(),
                    cr_out.as_mut_ptr(),
                    n as i32,
                    dt as f32,
                    streaming_coeff as f32,
                    pbox,
                )
            };
            check_kernel("cuda_mhd_cr_streaming", code)?;
            for (p, &cr) in particles.iter_mut().zip(&cr_out) {
                p.cr_energy = cr as f64;
            }
            Ok(())
        }
    }

    /// CR backreaction O(N²): aceleración gas desde gradiente de presión CR.
    ///
    /// Replica `gadget_ng_mhd::streaming::cr_pressure_backreaction` en GPU.
    /// Devuelve el vector de aceleraciones por partícula.
    pub fn try_cr_backreaction(
        &self,
        particles: &[Particle],
        periodic_box: Option<f64>,
    ) -> Result<Vec<Vec3>, CudaExecutionError> {
        let n = particles.len();
        if n == 0 {
            return Ok(Vec::new());
        }

        #[cfg(cuda_unavailable)]
        {
            let _ = (particles, periodic_box);
            return Err(CudaExecutionError::Unavailable(CudaUnavailable {
                availability: CudaPmSolver::availability(),
            }));
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
            let mass: Vec<f32> = particles.iter().map(|p| p.mass as f32).collect();
            let h_sml: Vec<f32> = particles
                .iter()
                .map(|p| p.smoothing_length.max(1e-10) as f32)
                .collect();
            let cr: Vec<f32> = particles.iter().map(|p| p.cr_energy as f32).collect();
            let mut ax = vec![0.0_f32; n];
            let mut ay = vec![0.0_f32; n];
            let mut az = vec![0.0_f32; n];
            let pbox = periodic_box.unwrap_or(0.0) as f32;

            // SAFETY: todos los slices son válidos y de longitud n.
            let code = unsafe {
                crate::ffi::cuda_mhd_cr_backreaction(
                    ptype.as_ptr(),
                    px.as_ptr(), py.as_ptr(), pz.as_ptr(),
                    mass.as_ptr(), h_sml.as_ptr(),
                    cr.as_ptr(),
                    ax.as_mut_ptr(), ay.as_mut_ptr(), az.as_mut_ptr(),
                    n as i32,
                    pbox,
                )
            };
            check_kernel("cuda_mhd_cr_backreaction", code)?;
            let result = ax
                .iter()
                .zip(ay.iter())
                .zip(az.iter())
                .map(|((&x, &y), &z)| Vec3 {
                    x: x as f64,
                    y: y as f64,
                    z: z as f64,
                })
                .collect();
            Ok(result)
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
            return Err(CudaExecutionError::Unavailable(CudaUnavailable {
                availability: CudaPmSolver::availability(),
            }));
        }

        #[cfg(not(cuda_unavailable))]
        {
            self.pool.ensure_capacity(n)?;
            self.pool.reset();

            let soa = MhdSoa::from_particles(particles);

            unsafe {
                let d_ptype = self.pool.upload_u8(0, &soa.ptype);
                let d_cr = self.pool.upload_f32(1, &soa.cr);
                let d_bx = self.pool.upload_f32(2, &soa.bx);
                let d_by = self.pool.upload_f32(3, &soa.by);
                let d_bz = self.pool.upload_f32(4, &soa.bz);
                let d_u = self.pool.upload_f32(5, &soa.internal_energy);
                let d_cr_out = self.pool.alloc_f32(6, n);
                let d_bx_out = self.pool.alloc_f32(7, n);
                let d_by_out = self.pool.alloc_f32(8, n);
                let d_bz_out = self.pool.alloc_f32(9, n);
                let d_u_out = self.pool.alloc_f32(10, n);

                let code = crate::ffi::cuda_mhd_reconnection_streaming_dynamo(
                    d_ptype,
                    d_cr,
                    d_bx,
                    d_by,
                    d_bz,
                    d_u,
                    d_cr_out,
                    d_bx_out,
                    d_by_out,
                    d_bz_out,
                    d_u_out,
                    n as i32,
                    dt as f32,
                    stream_coeff as f32,
                    reconnection_frac as f32,
                    dynamo_alpha as f32,
                );
                check_kernel("cuda_mhd_reconnection_streaming_dynamo", code)?;

                let mut cr = vec![0.0_f32; n];
                let mut bx = vec![0.0_f32; n];
                let mut by = vec![0.0_f32; n];
                let mut bz = vec![0.0_f32; n];
                let mut u = vec![0.0_f32; n];
                self.pool.download_f32(&mut cr, d_cr_out)?;
                self.pool.download_f32(&mut bx, d_bx_out)?;
                self.pool.download_f32(&mut by, d_by_out)?;
                self.pool.download_f32(&mut bz, d_bz_out)?;
                self.pool.download_f32(&mut u, d_u_out)?;

                for (i, p) in particles.iter_mut().enumerate() {
                    p.cr_energy = cr[i] as f64;
                    p.internal_energy = u[i] as f64;
                    p.b_field = Vec3::new(bx[i] as f64, by[i] as f64, bz[i] as f64);
                }
            }
            Ok(())
        }
    }
}

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

#[cfg(cuda_unavailable)]
const _: () = {
    let _ = std::mem::size_of::<CudaMhdSolver>();
};
