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
            let n_sub = ((dt / dt_cfl).ceil() as usize).max(1).max(params.substeps);
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

    /// Calcula Γ_HI por partícula (NGP lookup) usando el campo RT.
    pub fn try_chemistry_rates(
        &self,
        particles: &[Particle],
        rad: &RadiationField,
        params: &M1Params,
        box_size: f64,
    ) -> Result<Vec<f64>, CudaExecutionError> {
        let n = particles.len();
        if n == 0 {
            return Ok(Vec::new());
        }

        #[cfg(cuda_unavailable)]
        {
            let _ = (particles, rad, params, box_size);
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
            let energy: Vec<f32> = rad.energy_density.iter().map(|&v| v as f32).collect();
            let n_cells = rad.n_cells();
            let c_red_cgs = (C_KMS * 1.0e5 / params.c_red_factor) as f32;
            let mut gamma_out = vec![0.0_f32; n];

            // SAFETY: todos los slices son válidos y tienen longitud correcta.
            let code = unsafe {
                crate::ffi::cuda_rt_chemistry_rates(
                    ptype.as_ptr(),
                    px.as_ptr(),
                    py.as_ptr(),
                    pz.as_ptr(),
                    energy.as_ptr(),
                    gamma_out.as_mut_ptr(),
                    n as i32,
                    n_cells as i32,
                    rad.nx as i32,
                    rad.ny as i32,
                    rad.nz as i32,
                    box_size as f32,
                    c_red_cgs,
                )
            };
            check_kernel("cuda_rt_chemistry_rates", code)?;
            Ok(gamma_out.into_iter().map(f64::from).collect())
        }
    }

    /// Aplica cooling_rate_approx a las energías internas de las partículas gas.
    pub fn try_apply_cooling(
        &self,
        particles: &mut [Particle],
        x_e: &[f64],
        params: &gadget_ng_rt::chemistry::ChemParams,
        dt: f64,
    ) -> Result<(), CudaExecutionError> {
        let n = particles.len();
        if n == 0 {
            return Ok(());
        }

        #[cfg(cuda_unavailable)]
        {
            let _ = (particles, x_e, params, dt);
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
            let mut u: Vec<f32> = particles.iter().map(|p| p.internal_energy as f32).collect();
            let xe_f32: Vec<f32> = x_e.iter().map(|&v| v as f32).collect();

            // SAFETY: slices válidos; u es modificado in-place en el kernel.
            let code = unsafe {
                crate::ffi::cuda_rt_cooling_apply(
                    ptype.as_ptr(),
                    u.as_mut_ptr(),
                    xe_f32.as_ptr(),
                    n as i32,
                    params.gamma as f32,
                    params.n_h_ref as f32,
                    dt as f32,
                )
            };
            check_kernel("cuda_rt_cooling_apply", code)?;
            for (p, &ui) in particles.iter_mut().zip(&u) {
                p.internal_energy = ui as f64;
            }
            Ok(())
        }
    }

    /// Solver químico stiff (subciclo implícito) en GPU sobre todas las partículas.
    pub fn try_apply_chemistry(
        &self,
        chem_states: &mut [gadget_ng_rt::chemistry::ChemState],
        gamma_hi: &[f64],
        temperature: &[f64],
        particle_types: &[gadget_ng_core::ParticleType],
        dt: f64,
        n_h_ref: f64,
    ) -> Result<(), CudaExecutionError> {
        let n = chem_states.len();
        if n == 0 {
            return Ok(());
        }

        #[cfg(cuda_unavailable)]
        {
            let _ = (
                chem_states,
                gamma_hi,
                temperature,
                particle_types,
                dt,
                n_h_ref,
            );
            return Err(CudaExecutionError::Unavailable(CudaUnavailable {
                availability: CudaPmSolver::availability(),
            }));
        }

        #[cfg(not(cuda_unavailable))]
        {
            let ptype: Vec<u8> = particle_types
                .iter()
                .map(|t| match t {
                    ParticleType::DarkMatter => 0,
                    ParticleType::Gas => 1,
                    ParticleType::Star => 2,
                })
                .collect();
            // Extraer los 12 campos de ChemState en arrays f32
            macro_rules! field_f32 {
                ($field:ident) => {
                    chem_states
                        .iter()
                        .map(|s| s.$field as f32)
                        .collect::<Vec<f32>>()
                };
            }
            let mut x_hi = field_f32!(x_hi);
            let mut x_hii = field_f32!(x_hii);
            let mut x_hei = field_f32!(x_hei);
            let mut x_heii = field_f32!(x_heii);
            let mut x_heiii = field_f32!(x_heiii);
            let mut x_e = field_f32!(x_e);
            let mut x_hm = field_f32!(x_hm);
            let mut x_h2 = field_f32!(x_h2);
            let mut x_h2p = field_f32!(x_h2p);
            let mut x_d = field_f32!(x_d);
            let mut x_dp = field_f32!(x_dp);
            let mut x_hd = field_f32!(x_hd);
            let ghi_f32: Vec<f32> = gamma_hi.iter().map(|&v| v as f32).collect();
            let temp_f32: Vec<f32> = temperature.iter().map(|&v| v as f32).collect();

            // SAFETY: todos los arrays son válidos y de longitud n; el kernel opera in-place.
            let code = unsafe {
                crate::ffi::cuda_rt_chemistry_stiff(
                    ptype.as_ptr(),
                    x_hi.as_mut_ptr(),
                    x_hii.as_mut_ptr(),
                    x_hei.as_mut_ptr(),
                    x_heii.as_mut_ptr(),
                    x_heiii.as_mut_ptr(),
                    x_e.as_mut_ptr(),
                    x_hm.as_mut_ptr(),
                    x_h2.as_mut_ptr(),
                    x_h2p.as_mut_ptr(),
                    x_d.as_mut_ptr(),
                    x_dp.as_mut_ptr(),
                    x_hd.as_mut_ptr(),
                    ghi_f32.as_ptr(),
                    temp_f32.as_ptr(),
                    n as i32,
                    dt as f32,
                    n_h_ref as f32,
                )
            };
            check_kernel("cuda_rt_chemistry_stiff", code)?;

            // Copiar de vuelta al slice de ChemState
            for (i, s) in chem_states.iter_mut().enumerate() {
                s.x_hi = x_hi[i] as f64;
                s.x_hii = x_hii[i] as f64;
                s.x_hei = x_hei[i] as f64;
                s.x_heii = x_heii[i] as f64;
                s.x_heiii = x_heiii[i] as f64;
                s.x_e = x_e[i] as f64;
                s.x_hm = x_hm[i] as f64;
                s.x_h2 = x_h2[i] as f64;
                s.x_h2p = x_h2p[i] as f64;
                s.x_d = x_d[i] as f64;
                s.x_dp = x_dp[i] as f64;
                s.x_hd = x_hd[i] as f64;
            }
            Ok(())
        }
    }

    /// Reducción paralela sobre x_hii: media, sigma, fracción ionizada.
    pub fn try_reionization_stats(
        &self,
        chem_states: &[gadget_ng_rt::chemistry::ChemState],
        z: f64,
        n_sources: usize,
    ) -> Result<gadget_ng_rt::reionization::ReionizationState, CudaExecutionError> {
        let n = chem_states.len();

        #[cfg(cuda_unavailable)]
        {
            let _ = (chem_states, z, n_sources);
            return Err(CudaExecutionError::Unavailable(CudaUnavailable {
                availability: CudaPmSolver::availability(),
            }));
        }

        #[cfg(not(cuda_unavailable))]
        {
            if n == 0 {
                return Ok(gadget_ng_rt::reionization::ReionizationState {
                    x_hii_mean: 0.0,
                    x_hii_sigma: 0.0,
                    ionized_volume_fraction: 0.0,
                    z,
                    n_sources,
                });
            }
            let xhii_f32: Vec<f32> = chem_states.iter().map(|s| s.x_hii as f32).collect();
            let mut sum_xhii = 0.0_f64;
            let mut sum_sq = 0.0_f64;
            let mut ionized_count = 0_i32;

            // SAFETY: slices válidos; salidas son escalares.
            let code = unsafe {
                crate::ffi::cuda_rt_reionization_stats(
                    xhii_f32.as_ptr(),
                    n as i32,
                    &mut sum_xhii,
                    &mut sum_sq,
                    &mut ionized_count,
                )
            };
            check_kernel("cuda_rt_reionization_stats", code)?;

            let mean = sum_xhii / n as f64;
            let var = (sum_sq / n as f64 - mean * mean).max(0.0);
            let ivf = ionized_count as f64 / n as f64;
            Ok(gadget_ng_rt::reionization::ReionizationState {
                x_hii_mean: mean,
                x_hii_sigma: var.sqrt(),
                ionized_volume_fraction: ivf,
                z,
                n_sources,
            })
        }
    }

    /// Calcula el campo de temperatura de brillo 21cm por partícula.
    pub fn try_cm21_field(
        &self,
        chem_states: &[gadget_ng_rt::chemistry::ChemState],
        overdensity: &[f64],
        z: f64,
    ) -> Result<Vec<f64>, CudaExecutionError> {
        let n = chem_states.len();
        if n == 0 {
            return Ok(Vec::new());
        }

        #[cfg(cuda_unavailable)]
        {
            let _ = (chem_states, overdensity, z);
            return Err(CudaExecutionError::Unavailable(CudaUnavailable {
                availability: CudaPmSolver::availability(),
            }));
        }

        #[cfg(not(cuda_unavailable))]
        {
            let xhii_f32: Vec<f32> = chem_states.iter().map(|s| s.x_hii as f32).collect();
            let od_f32: Vec<f32> = overdensity.iter().map(|&v| v as f32).collect();
            let mut dtb = vec![0.0_f32; n];

            // SAFETY: slices válidos y de longitud n.
            let code = unsafe {
                crate::ffi::cuda_rt_cm21_field(
                    xhii_f32.as_ptr(),
                    od_f32.as_ptr(),
                    z as f32,
                    dtb.as_mut_ptr(),
                    n as i32,
                )
            };
            check_kernel("cuda_rt_cm21_field", code)?;
            Ok(dtb.into_iter().map(f64::from).collect())
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

    /// Calcula estadísticas de temperatura del IGM (media y sigma) vía reducción GPU.
    ///
    /// Filtra partículas IGM con densidad `rho_sph < delta_max × mean_density`.
    /// La mediana se aproxima como la media (el kernel no ordena).
    pub fn try_igm_temp_profile(
        &self,
        particles: &[Particle],
        chem_states: &[gadget_ng_rt::chemistry::ChemState],
        mean_density: f64,
        z: f64,
        params: &gadget_ng_rt::IgmTempParams,
    ) -> Result<gadget_ng_rt::IgmTempBin, CudaExecutionError> {
        let n = particles.len();
        if n == 0 {
            return Ok(gadget_ng_rt::IgmTempBin {
                z,
                ..Default::default()
            });
        }

        #[cfg(cuda_unavailable)]
        {
            let _ = (particles, chem_states, mean_density, z, params);
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
            let u: Vec<f32> = particles.iter().map(|p| p.internal_energy as f32).collect();
            let h: Vec<f32> = particles
                .iter()
                .map(|p| p.smoothing_length as f32)
                .collect();
            let mass: Vec<f32> = particles.iter().map(|p| p.mass as f32).collect();
            // Campos ChemState — misma fórmula que CPU (free_h = x_hi + x_hii + x_e).
            let x_hi: Vec<f32> = chem_states.iter().map(|s| s.x_hi as f32).collect();
            let x_hii: Vec<f32> = chem_states.iter().map(|s| s.x_hii as f32).collect();
            let x_e: Vec<f32> = chem_states.iter().map(|s| s.x_e as f32).collect();
            let x_d: Vec<f32> = chem_states.iter().map(|s| s.x_d as f32).collect();
            let x_hei: Vec<f32> = chem_states.iter().map(|s| s.x_hei as f32).collect();
            let x_heii: Vec<f32> = chem_states.iter().map(|s| s.x_heii as f32).collect();
            let x_heiii: Vec<f32> = chem_states.iter().map(|s| s.x_heiii as f32).collect();

            let mut t_mean = 0.0_f64;
            let mut t_sigma = 0.0_f64;
            let mut n_igm = 0_i32;
            // Buffer para temperaturas filtradas (máximo n elementos).
            let mut temps_buf = vec![0.0_f32; n];

            // SAFETY: todos los slices son válidos y tienen longitud n.
            let code = unsafe {
                crate::ffi::cuda_rt_igm_temp_full(
                    ptype.as_ptr(),
                    u.as_ptr(),
                    h.as_ptr(),
                    mass.as_ptr(),
                    x_hi.as_ptr(),
                    x_hii.as_ptr(),
                    x_e.as_ptr(),
                    x_d.as_ptr(),
                    x_hei.as_ptr(),
                    x_heii.as_ptr(),
                    x_heiii.as_ptr(),
                    n as i32,
                    params.gamma as f32,
                    params.delta_max as f32,
                    mean_density as f32,
                    &mut t_mean,
                    &mut t_sigma,
                    &mut n_igm,
                    temps_buf.as_mut_ptr(),
                )
            };
            check_kernel("cuda_rt_igm_temp_full", code)?;

            // Calcular percentiles ordenando el array compacto en host.
            let (t_median, t_p16, t_p84) = if n_igm > 0 {
                let mut sorted: Vec<f32> = temps_buf[..n_igm as usize].to_vec();
                sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let m = sorted.len();
                let percentile = |p: f64| -> f64 {
                    let idx = ((p / 100.0) * (m - 1) as f64).round() as usize;
                    sorted[idx.min(m - 1)] as f64
                };
                (percentile(50.0), percentile(16.0), percentile(84.0))
            } else {
                (t_mean, 0.0, 0.0)
            };

            Ok(gadget_ng_rt::IgmTempBin {
                z,
                t_mean,
                t_median,
                t_sigma,
                t_p16,
                t_p84,
                n_particles: n_igm as usize,
            })
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
