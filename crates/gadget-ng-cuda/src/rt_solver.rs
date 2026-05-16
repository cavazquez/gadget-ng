//! Kernels RT locales via CUDA con buffers persistentes.
//!
//! Versión optimizada: `CudaRtSolver` retiene un [`CudaPool`] de buffers
//! device entre pasos, eliminando `cudaMalloc`/`cudaFree` por invocación.
//! Los buffers se redimensionan solo cuando el número de partículas crece.

use crate::pool::CudaPool;
use crate::{CudaExecutionError, CudaPmSolver, CudaUnavailable};
use gadget_ng_core::Particle;
#[cfg(not(cuda_unavailable))]
use gadget_ng_core::ParticleType;
#[cfg(not(cuda_unavailable))]
use gadget_ng_rt::m1::C_KMS;
use gadget_ng_rt::m1::{M1Params, RadiationField};

/// Solver CUDA para reducciones/campos RT locales con buffers device persistentes.
pub struct CudaRtSolver {
    #[cfg(not(cuda_unavailable))]
    pool: CudaPool,
    #[cfg(cuda_unavailable)]
    _phantom: (),
}

impl std::fmt::Debug for CudaRtSolver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaRtSolver").finish()
    }
}

impl Clone for CudaRtSolver {
    fn clone(&self) -> Self {
        Self::try_new_checked().unwrap_or_else(|_| {
            #[cfg(not(cuda_unavailable))]
            {
                panic!("CudaRtSolver clone failed: CUDA not available");
            }
            #[cfg(cuda_unavailable)]
            {
                Self { _phantom: () }
            }
        })
    }
}

impl CudaRtSolver {
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
            return Err(CudaExecutionError::Unavailable(CudaUnavailable {
                availability: CudaPmSolver::availability(),
            }));
        }

        #[cfg(not(cuda_unavailable))]
        {
            self.pool.ensure_capacity(n)?;
            self.pool.reset();

            let energy: Vec<f32> = rad.energy_density.iter().map(|&v| v as f32).collect();
            let flux_x: Vec<f32> = rad.flux_x.iter().map(|&v| v as f32).collect();
            let flux_y: Vec<f32> = rad.flux_y.iter().map(|&v| v as f32).collect();
            let flux_z: Vec<f32> = rad.flux_z.iter().map(|&v| v as f32).collect();

            // SAFETY: pool handle is valid, slots are freshly reset; all slices have length n.
            unsafe {
                let d_energy = self.pool.upload_f32(0, &energy);
                let d_flux_x = self.pool.upload_f32(1, &flux_x);
                let d_flux_y = self.pool.upload_f32(2, &flux_y);
                let d_flux_z = self.pool.upload_f32(3, &flux_z);
                let d_energy_contrib = self.pool.alloc_f32(4, n);
                let d_xi = self.pool.alloc_f32(5, n);
                let d_gamma = self.pool.alloc_f32(6, n);

                let c_red_code = C_KMS / params.c_red_factor;
                let c_red_cgs = C_KMS * 1.0e5 / params.c_red_factor;
                let code = crate::ffi::cuda_rt_energy_xi_photoion(
                    d_energy,
                    d_flux_x,
                    d_flux_y,
                    d_flux_z,
                    d_energy_contrib,
                    d_xi,
                    d_gamma,
                    n as i32,
                    dv as f32,
                    c_red_code as f32,
                    c_red_cgs as f32,
                );
                check_kernel("cuda_rt_energy_xi_photoion", code)?;

                let mut energy_contrib = vec![0.0_f32; n];
                let mut xi = vec![0.0_f32; n];
                let mut gamma = vec![0.0_f32; n];
                self.pool
                    .download_f32(&mut energy_contrib, d_energy_contrib)?;
                self.pool.download_f32(&mut xi, d_xi)?;
                self.pool.download_f32(&mut gamma, d_gamma)?;

                Ok((
                    energy_contrib.iter().map(|&v| v as f64).sum(),
                    xi.into_iter().map(f64::from).collect(),
                    gamma.into_iter().map(f64::from).collect(),
                ))
            }
        }
    }

    /// Ejecuta `n_substeps` pasos del solver M1 HLL completo en GPU.
    ///
    /// Cada sub-paso replica exactamente el algoritmo de [`gadget_ng_rt::m1::m1_update`]
    /// con aritmética f32. El número de sub-pasos debe ser pre-calculado por el caller
    /// (igual que en la versión CPU).
    pub fn try_m1_advection(
        &self,
        rad: &mut RadiationField,
        dt: f64,
        params: &M1Params,
    ) -> Result<(), CudaExecutionError> {
        let n = rad.n_cells();
        if n == 0 {
            return Ok(());
        }

        #[cfg(cuda_unavailable)]
        {
            let _ = (rad, dt, params);
            return Err(CudaExecutionError::Unavailable(CudaUnavailable {
                availability: CudaPmSolver::availability(),
            }));
        }

        #[cfg(not(cuda_unavailable))]
        {
            let c_red = C_KMS / params.c_red_factor;
            let dt_cfl = rad.dx / c_red * 0.5;
            let n_sub = ((dt / dt_cfl).ceil() as usize)
                .max(1)
                .max(params.substeps);
            let dt_sub = dt / n_sub as f64;
            let kappa = (params.kappa_abs + params.kappa_scat) as f32;

            // Convertir a f32 para la GPU.
            let mut e: Vec<f32> = rad.energy_density.iter().map(|&v| v as f32).collect();
            let mut fx: Vec<f32> = rad.flux_x.iter().map(|&v| v as f32).collect();
            let mut fy: Vec<f32> = rad.flux_y.iter().map(|&v| v as f32).collect();
            let mut fz: Vec<f32> = rad.flux_z.iter().map(|&v| v as f32).collect();

            for _ in 0..n_sub {
                // SAFETY: los arrays f32 son válidos; el kernel escribe en buffers temporales
                // device y copia de vuelta a los mismos punteros host.
                let code = unsafe {
                    crate::ffi::cuda_rt_m1_substep(
                        e.as_mut_ptr(),
                        fx.as_mut_ptr(),
                        fy.as_mut_ptr(),
                        fz.as_mut_ptr(),
                        rad.nx as i32,
                        rad.ny as i32,
                        rad.nz as i32,
                        rad.dx as f32,
                        dt_sub as f32,
                        c_red as f32,
                        kappa,
                    )
                };
                check_kernel("cuda_rt_m1_substep", code)?;
            }

            for i in 0..n {
                rad.energy_density[i] = e[i] as f64;
                rad.flux_x[i] = fx[i] as f64;
                rad.flux_y[i] = fy[i] as f64;
                rad.flux_z[i] = fz[i] as f64;
            }
            Ok(())
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
            let px: Vec<f32> = particles.iter().map(|p| p.position.x as f32).collect();
            let py: Vec<f32> = particles.iter().map(|p| p.position.y as f32).collect();
            let pz: Vec<f32> = particles.iter().map(|p| p.position.z as f32).collect();
            let u_in: Vec<f32> = particles.iter().map(|p| p.internal_energy as f32).collect();
            let gamma: Vec<f32> = gamma_hi.iter().map(|&v| v as f32).collect();

            // SAFETY: pool handle is valid, slots are freshly reset; all slices have length n.
            unsafe {
                let d_ptype = self.pool.upload_u8(0, &ptype);
                let d_px = self.pool.upload_f32(1, &px);
                let d_py = self.pool.upload_f32(2, &py);
                let d_pz = self.pool.upload_f32(3, &pz);
                let d_u_in = self.pool.upload_f32(4, &u_in);
                let d_gamma = self.pool.upload_f32(5, &gamma);
                let d_u_out = self.pool.alloc_f32(6, n);

                let code = crate::ffi::cuda_rt_photoheating(
                    d_ptype,
                    d_px,
                    d_py,
                    d_pz,
                    d_u_in,
                    d_gamma,
                    d_u_out,
                    n as i32,
                    rad.nx as i32,
                    rad.ny as i32,
                    rad.nz as i32,
                    box_size as f32,
                    dt as f32,
                );
                check_kernel("cuda_rt_photoheating", code)?;

                let mut u_out = vec![0.0_f32; n];
                self.pool.download_f32(&mut u_out, d_u_out)?;

                for (p, &u) in particles.iter_mut().zip(&u_out) {
                    p.internal_energy = u as f64;
                }
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

#[cfg(cuda_unavailable)]
const _: () = {
    let _ = std::mem::size_of::<CudaRtSolver>();
};
