//! Kernels SPH O(N²) via CUDA con buffers persistentes.
//!
//! Versión optimizada: `CudaSphSolver` retiene un [`CudaPool`] de buffers
//! device entre pasos, eliminando `cudaMalloc`/`cudaFree` por invocación.
//! Los buffers se redimensionan solo cuando el número de partículas crece.

use crate::pool::CudaPool;
use crate::{CudaExecutionError, CudaPmSolver, CudaUnavailable};
use gadget_ng_core::Vec3;
use gadget_ng_core::{Particle, ParticleType};
use gadget_ng_sph::particle::SphParticle;

/// Solver CUDA para kernels SPH locales O(N²) con buffers device persistentes.
///
/// Se construye con [`CudaSphSolver::try_new`] y se mantiene vivo durante
/// toda la simulación. Los buffers device se reutilizan entre pasos de
/// tiempo; solo se redimensionan cuando el número de partículas excede
/// la capacidad actual.
pub struct CudaSphSolver {
    #[cfg(not(cuda_unavailable))]
    pool: CudaPool,
    #[cfg(cuda_unavailable)]
    _phantom: (),
}

impl std::fmt::Debug for CudaSphSolver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaSphSolver").finish()
    }
}

impl Clone for CudaSphSolver {
    fn clone(&self) -> Self {
        Self::try_new_checked().unwrap_or_else(|_| {
            #[cfg(not(cuda_unavailable))]
            {
                panic!("CudaSphSolver clone failed: CUDA not available");
            }
            #[cfg(cuda_unavailable)]
            {
                Self { _phantom: () }
            }
        })
    }
}

impl CudaSphSolver {
    /// Intenta crear el solver SPH CUDA con buffers persistentes.
    pub fn try_new() -> Option<Self> {
        Self::try_new_checked().ok()
    }

    /// Variante fallible de [`Self::try_new`] que conserva el diagnóstico.
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

    /// Asegura que los buffers del pool tengan capacidad para `n` partículas.
    pub fn ensure_capacity(&self, n: usize) -> Result<(), CudaExecutionError> {
        #[cfg(cuda_unavailable)]
        {
            let _ = n;
            return Err(CudaExecutionError::Unavailable(CudaUnavailable {
                availability: CudaPmSolver::availability(),
            }));
        }

        #[cfg(not(cuda_unavailable))]
        {
            self.pool.ensure_capacity(n)
        }
    }

    /// Calcula `h_sml`, `rho`, `pressure` y `entropy`.
    pub fn try_compute_density(
        &self,
        particles: &mut [SphParticle],
        periodic_box: Option<f64>,
    ) -> Result<(), CudaExecutionError> {
        let n = particles.len();
        if n == 0 {
            return Ok(());
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
            self.pool.ensure_capacity(n)?;
            self.pool.reset();

            let soa = SphSoa::from_particles(particles);

            // SAFETY: pool handle is valid, slots are freshly reset; all slices have length n.
            unsafe {
                let d_x = self.pool.upload_f32(0, &soa.x);
                let d_y = self.pool.upload_f32(1, &soa.y);
                let d_z = self.pool.upload_f32(2, &soa.z);
                let d_mass = self.pool.upload_f32(3, &soa.mass);
                let d_is_gas = self.pool.upload_u8(4, &soa.is_gas);
                let d_u = self.pool.upload_f32(5, &soa.u);
                let d_h = self.pool.upload_f32(6, &soa.h);
                let d_h_out = self.pool.alloc_f32(7, n);
                let d_rho = self.pool.alloc_f32(8, n);
                let d_pressure = self.pool.alloc_f32(9, n);
                let d_entropy = self.pool.alloc_f32(10, n);

                let code = crate::ffi::cuda_sph_density(
                    d_x,
                    d_y,
                    d_z,
                    d_mass,
                    d_is_gas,
                    d_u,
                    d_h,
                    d_h_out,
                    d_rho,
                    d_pressure,
                    d_entropy,
                    n as i32,
                    periodic_box_f32(periodic_box),
                );
                check_kernel("cuda_sph_density", code)?;

                let mut h_out = vec![0.0_f32; n];
                let mut rho_out = vec![0.0_f32; n];
                let mut pressure_out = vec![0.0_f32; n];
                let mut entropy_out = vec![0.0_f32; n];
                self.pool.download_f32(&mut h_out, d_h_out)?;
                self.pool.download_f32(&mut rho_out, d_rho)?;
                self.pool.download_f32(&mut pressure_out, d_pressure)?;
                self.pool.download_f32(&mut entropy_out, d_entropy)?;

                for (i, p) in particles.iter_mut().enumerate() {
                    if let Some(gas) = p.gas.as_mut() {
                        gas.h_sml = h_out[i] as f64;
                        gas.rho = rho_out[i] as f64;
                        gas.pressure = pressure_out[i] as f64;
                        gas.entropy = entropy_out[i] as f64;
                    }
                }
            }
            Ok(())
        }
    }

    /// Calcula el factor de Balsara para cada partícula de gas.
    pub fn try_compute_balsara(
        &self,
        particles: &mut [SphParticle],
        periodic_box: Option<f64>,
    ) -> Result<(), CudaExecutionError> {
        let n = particles.len();
        if n == 0 {
            return Ok(());
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
            self.pool.ensure_capacity(n)?;
            self.pool.reset();

            let soa = SphSoa::from_particles(particles);

            unsafe {
                let d_x = self.pool.upload_f32(0, &soa.x);
                let d_y = self.pool.upload_f32(1, &soa.y);
                let d_z = self.pool.upload_f32(2, &soa.z);
                let d_vx = self.pool.upload_f32(3, &soa.vx);
                let d_vy = self.pool.upload_f32(4, &soa.vy);
                let d_vz = self.pool.upload_f32(5, &soa.vz);
                let d_mass = self.pool.upload_f32(6, &soa.mass);
                let d_is_gas = self.pool.upload_u8(7, &soa.is_gas);
                let d_rho = self.pool.upload_f32(8, &soa.rho);
                let d_pressure = self.pool.upload_f32(9, &soa.pressure);
                let d_h = self.pool.upload_f32(10, &soa.h);
                let d_balsara = self.pool.alloc_f32(11, n);

                let code = crate::ffi::cuda_sph_balsara(
                    d_x,
                    d_y,
                    d_z,
                    d_vx,
                    d_vy,
                    d_vz,
                    d_mass,
                    d_is_gas,
                    d_rho,
                    d_pressure,
                    d_h,
                    d_balsara,
                    n as i32,
                    periodic_box_f32(periodic_box),
                );
                check_kernel("cuda_sph_balsara", code)?;

                let mut balsara_out = vec![1.0_f32; n];
                self.pool.download_f32(&mut balsara_out, d_balsara)?;

                for (i, p) in particles.iter_mut().enumerate() {
                    if let Some(gas) = p.gas.as_mut() {
                        gas.balsara = balsara_out[i] as f64;
                    }
                }
            }
            Ok(())
        }
    }

    /// Calcula `acc_sph` y `du_dt` usando la formulación clásica.
    pub fn try_compute_forces(
        &self,
        particles: &mut [SphParticle],
        periodic_box: Option<f64>,
    ) -> Result<(), CudaExecutionError> {
        let n = particles.len();
        if n == 0 {
            return Ok(());
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
            self.pool.ensure_capacity(n)?;
            self.pool.reset();

            let soa = SphSoa::from_particles(particles);

            unsafe {
                let d_x = self.pool.upload_f32(0, &soa.x);
                let d_y = self.pool.upload_f32(1, &soa.y);
                let d_z = self.pool.upload_f32(2, &soa.z);
                let d_vx = self.pool.upload_f32(3, &soa.vx);
                let d_vy = self.pool.upload_f32(4, &soa.vy);
                let d_vz = self.pool.upload_f32(5, &soa.vz);
                let d_mass = self.pool.upload_f32(6, &soa.mass);
                let d_is_gas = self.pool.upload_u8(7, &soa.is_gas);
                let d_rho = self.pool.upload_f32(8, &soa.rho);
                let d_pressure = self.pool.upload_f32(9, &soa.pressure);
                let d_h = self.pool.upload_f32(10, &soa.h);
                let d_ax = self.pool.alloc_f32(11, n);
                let d_ay = self.pool.alloc_f32(12, n);
                let d_az = self.pool.alloc_f32(13, n);
                let d_du_dt = self.pool.alloc_f32(14, n);

                let code = crate::ffi::cuda_sph_forces(
                    d_x,
                    d_y,
                    d_z,
                    d_vx,
                    d_vy,
                    d_vz,
                    d_mass,
                    d_is_gas,
                    d_rho,
                    d_pressure,
                    d_h,
                    d_ax,
                    d_ay,
                    d_az,
                    d_du_dt,
                    n as i32,
                    periodic_box_f32(periodic_box),
                );
                check_kernel("cuda_sph_forces", code)?;

                let mut ax = vec![0.0_f32; n];
                let mut ay = vec![0.0_f32; n];
                let mut az = vec![0.0_f32; n];
                let mut du_dt = vec![0.0_f32; n];
                self.pool.download_f32(&mut ax, d_ax)?;
                self.pool.download_f32(&mut ay, d_ay)?;
                self.pool.download_f32(&mut az, d_az)?;
                self.pool.download_f32(&mut du_dt, d_du_dt)?;

                for (i, p) in particles.iter_mut().enumerate() {
                    if let Some(gas) = p.gas.as_mut() {
                        gas.acc_sph = Vec3::new(ax[i] as f64, ay[i] as f64, az[i] as f64);
                        gas.du_dt = du_dt[i] as f64;
                    }
                }
            }
            Ok(())
        }
    }

    /// Calcula fuerzas SPH Gadget-2 con limitador de Balsara.
    pub fn try_compute_gadget2_forces(
        &self,
        particles: &mut [SphParticle],
        periodic_box: Option<f64>,
    ) -> Result<(), CudaExecutionError> {
        let n = particles.len();
        if n == 0 {
            return Ok(());
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
            self.pool.ensure_capacity(n)?;
            self.pool.reset();

            let soa = SphSoa::from_particles(particles);

            unsafe {
                let d_x = self.pool.upload_f32(0, &soa.x);
                let d_y = self.pool.upload_f32(1, &soa.y);
                let d_z = self.pool.upload_f32(2, &soa.z);
                let d_vx = self.pool.upload_f32(3, &soa.vx);
                let d_vy = self.pool.upload_f32(4, &soa.vy);
                let d_vz = self.pool.upload_f32(5, &soa.vz);
                let d_mass = self.pool.upload_f32(6, &soa.mass);
                let d_is_gas = self.pool.upload_u8(7, &soa.is_gas);
                let d_rho = self.pool.upload_f32(8, &soa.rho);
                let d_pressure = self.pool.upload_f32(9, &soa.pressure);
                let d_h = self.pool.upload_f32(10, &soa.h);
                let d_balsara = self.pool.upload_f32(11, &soa.balsara);

                let d_ax = self.pool.alloc_f32(12, n);
                let d_ay = self.pool.alloc_f32(13, n);
                let d_az = self.pool.alloc_f32(14, n);
                let d_da_dt = self.pool.alloc_f32(15, n);
                let d_du_dt = self.pool.alloc_f32(16, n);
                let d_max_vsig = self.pool.alloc_f32(17, n);

                let code = crate::ffi::cuda_sph_gadget2_forces(
                    d_x,
                    d_y,
                    d_z,
                    d_vx,
                    d_vy,
                    d_vz,
                    d_mass,
                    d_is_gas,
                    d_rho,
                    d_pressure,
                    d_h,
                    d_balsara,
                    d_ax,
                    d_ay,
                    d_az,
                    d_da_dt,
                    d_du_dt,
                    d_max_vsig,
                    n as i32,
                    periodic_box_f32(periodic_box),
                );
                check_kernel("cuda_sph_gadget2_forces", code)?;

                let mut ax = vec![0.0_f32; n];
                let mut ay = vec![0.0_f32; n];
                let mut az = vec![0.0_f32; n];
                let mut da_dt = vec![0.0_f32; n];
                let mut du_dt = vec![0.0_f32; n];
                let mut max_vsig = vec![0.0_f32; n];
                self.pool.download_f32(&mut ax, d_ax)?;
                self.pool.download_f32(&mut ay, d_ay)?;
                self.pool.download_f32(&mut az, d_az)?;
                self.pool.download_f32(&mut da_dt, d_da_dt)?;
                self.pool.download_f32(&mut du_dt, d_du_dt)?;
                self.pool.download_f32(&mut max_vsig, d_max_vsig)?;

                for (i, p) in particles.iter_mut().enumerate() {
                    if let Some(gas) = p.gas.as_mut() {
                        gas.acc_sph = Vec3::new(ax[i] as f64, ay[i] as f64, az[i] as f64);
                        gas.da_dt = da_dt[i] as f64;
                        gas.du_dt = du_dt[i] as f64;
                        gas.max_vsig = max_vsig[i] as f64;
                    }
                }
            }
            Ok(())
        }
    }

    /// Pipeline persistente densidad+Balsara+fuerzas SPH sobre `gadget_ng_core::Particle`.
    ///
    /// Permite cablear el solver CUDA directamente en el integrador KDK sin necesidad de
    /// convertir a `SphParticle`. El SoA se sube **una sola vez** al device; los buffers
    /// intermedios (rho, pressure, h_out, balsara) permanecen en device entre kernels.
    /// Los pasos son:
    /// 1. CUDA density  → slots 10–13 (h_out, rho, pressure, entropy)
    /// 2. CUDA Balsara  → slot 14 (balsara), usando d_rho/d_pressure/d_h del device
    /// 3. CUDA gadget2_forces → slots 15–20 (ax,ay,az,da_dt,du_dt,max_vsig), usando
    ///    d_rho/d_pressure/d_h/d_balsara del device
    ///
    /// Devuelve `(rho[i], acc_sph[i], du_dt[i])` para cada partícula (cero para DM).
    /// También escribe `smoothing_length` actualizado de vuelta en cada partícula gas.
    ///
    /// Nota: el kernel CUDA usa `γ = 5/3` fijo; coincide con el default del motor.
    ///
    /// # Mapa de slots del pool (21 slots en total, sin reset entre kernels)
    ///
    /// | Slot | Campo      | Uso                              |
    /// |------|------------|----------------------------------|
    /// | 0–2  | x, y, z    | upload; density+balsara+forces   |
    /// | 3–5  | vx,vy,vz   | upload; balsara+forces           |
    /// | 6    | mass       | upload; los tres kernels         |
    /// | 7    | is_gas     | upload (u8); los tres kernels    |
    /// | 8    | u_arr      | upload; density                  |
    /// | 9    | h_in       | upload; density                  |
    /// | 10   | h_out      | alloc; density→balsara+forces    |
    /// | 11   | rho        | alloc; density→balsara+forces    |
    /// | 12   | pressure   | alloc; density→balsara+forces    |
    /// | 13   | entropy    | alloc; density (descartado)      |
    /// | 14   | balsara    | alloc; balsara→gadget2_forces    |
    /// | 15–17| ax,ay,az   | alloc; gadget2_forces (salida)   |
    /// | 18   | da_dt      | alloc; gadget2_forces (descartado)|
    /// | 19   | du_dt      | alloc; gadget2_forces (salida)   |
    /// | 20   | max_vsig   | alloc; gadget2_forces (descartado)|
    #[allow(clippy::type_complexity)]
    pub fn try_sph_density_and_forces_core(
        &self,
        particles: &mut [Particle],
        periodic_box: Option<f64>,
    ) -> Result<(Vec<f64>, Vec<Vec3>, Vec<f64>), CudaExecutionError> {
        let n = particles.len();
        if n == 0 {
            return Ok((vec![], vec![], vec![]));
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
            let pbox_f32 = periodic_box_f32(periodic_box);

            // ── SoA desde Particle core (una sola vez) ────────────────────────
            let mut x = Vec::with_capacity(n);
            let mut y = Vec::with_capacity(n);
            let mut z = Vec::with_capacity(n);
            let mut vx = Vec::with_capacity(n);
            let mut vy = Vec::with_capacity(n);
            let mut vz = Vec::with_capacity(n);
            let mut mass = Vec::with_capacity(n);
            let mut is_gas = Vec::with_capacity(n);
            let mut u_arr = Vec::with_capacity(n);
            let mut h_arr = Vec::with_capacity(n);

            for p in particles.iter() {
                x.push(p.position.x as f32);
                y.push(p.position.y as f32);
                z.push(p.position.z as f32);
                vx.push(p.velocity.x as f32);
                vy.push(p.velocity.y as f32);
                vz.push(p.velocity.z as f32);
                mass.push(p.mass as f32);
                let gas = p.ptype == ParticleType::Gas;
                is_gas.push(if gas { 1u8 } else { 0u8 });
                u_arr.push(if gas { p.internal_energy as f32 } else { 0.0 });
                h_arr.push(if gas && p.smoothing_length > 0.0 {
                    p.smoothing_length as f32
                } else {
                    1.0
                });
            }

            // ── Preparar pool (un solo reset; slots 0-20 persisten en device) ─
            self.pool.ensure_capacity(n)?;
            self.pool.reset();

            let mut h_out_host = vec![0.0_f32; n];
            let mut rho_host = vec![0.0_f32; n];
            let mut ax_out = vec![0.0_f32; n];
            let mut ay_out = vec![0.0_f32; n];
            let mut az_out = vec![0.0_f32; n];
            let mut du_dt_out = vec![0.0_f32; n];

            // SAFETY: pool handle válido, 21 slots únicos asignados, longitudes n.
            unsafe {
                // Slots 0-9: inputs subidos una sola vez.
                let d_x = self.pool.upload_f32(0, &x);
                let d_y = self.pool.upload_f32(1, &y);
                let d_z = self.pool.upload_f32(2, &z);
                let d_vx = self.pool.upload_f32(3, &vx);
                let d_vy = self.pool.upload_f32(4, &vy);
                let d_vz = self.pool.upload_f32(5, &vz);
                let d_mass = self.pool.upload_f32(6, &mass);
                let d_is_gas = self.pool.upload_u8(7, &is_gas);
                let d_u = self.pool.upload_f32(8, &u_arr);
                let d_h_in = self.pool.upload_f32(9, &h_arr);

                // Slots 10-13: salidas de density (permanecen en device).
                let d_h_out = self.pool.alloc_f32(10, n);
                let d_rho = self.pool.alloc_f32(11, n);
                let d_pressure = self.pool.alloc_f32(12, n);
                let d_entropy = self.pool.alloc_f32(13, n);

                // ── Kernel 1: densidad ────────────────────────────────────────
                let code = crate::ffi::cuda_sph_density(
                    d_x, d_y, d_z, d_mass, d_is_gas, d_u, d_h_in, d_h_out, d_rho, d_pressure,
                    d_entropy, n as i32, pbox_f32,
                );
                check_kernel("cuda_sph_density", code)?;

                // Slot 14: salida de Balsara (permanece en device para gadget2_forces).
                let d_balsara = self.pool.alloc_f32(14, n);

                // ── Kernel 2: Balsara (lee d_rho/d_pressure/d_h_out del device) ─
                let code = crate::ffi::cuda_sph_balsara(
                    d_x, d_y, d_z, d_vx, d_vy, d_vz, d_mass, d_is_gas, d_rho, d_pressure, d_h_out,
                    d_balsara, n as i32, pbox_f32,
                );
                check_kernel("cuda_sph_balsara", code)?;

                // Slots 15-20: salidas de gadget2_forces.
                let d_ax = self.pool.alloc_f32(15, n);
                let d_ay = self.pool.alloc_f32(16, n);
                let d_az = self.pool.alloc_f32(17, n);
                let d_da_dt = self.pool.alloc_f32(18, n);
                let d_du_dt = self.pool.alloc_f32(19, n);
                let d_max_vsig = self.pool.alloc_f32(20, n);

                // ── Kernel 3: fuerzas Gadget-2 con Balsara ────────────────────
                let code = crate::ffi::cuda_sph_gadget2_forces(
                    d_x, d_y, d_z, d_vx, d_vy, d_vz, d_mass, d_is_gas, d_rho, d_pressure, d_h_out,
                    d_balsara, d_ax, d_ay, d_az, d_da_dt, d_du_dt, d_max_vsig, n as i32, pbox_f32,
                );
                check_kernel("cuda_sph_gadget2_forces", code)?;

                // ── Descarga final: solo los campos necesarios ────────────────
                self.pool.download_f32(&mut h_out_host, d_h_out)?;
                self.pool.download_f32(&mut rho_host, d_rho)?;
                self.pool.download_f32(&mut ax_out, d_ax)?;
                self.pool.download_f32(&mut ay_out, d_ay)?;
                self.pool.download_f32(&mut az_out, d_az)?;
                self.pool.download_f32(&mut du_dt_out, d_du_dt)?;
            }

            // Escribir h_sml actualizado de vuelta a partículas (diferido al final).
            for (i, p) in particles.iter_mut().enumerate() {
                if p.ptype == ParticleType::Gas {
                    p.smoothing_length = h_out_host[i] as f64;
                }
            }

            // ── Ensamblar resultado ───────────────────────────────────────────
            let rho_f64: Vec<f64> = rho_host.iter().map(|&v| v as f64).collect();
            let acc_sph: Vec<Vec3> = (0..n)
                .map(|i| Vec3::new(ax_out[i] as f64, ay_out[i] as f64, az_out[i] as f64))
                .collect();
            let du_dt_f64: Vec<f64> = du_dt_out.iter().map(|&v| v as f64).collect();

            Ok((rho_f64, acc_sph, du_dt_f64))
        }
    }
}

#[cfg(not(cuda_unavailable))]
struct SphSoa {
    x: Vec<f32>,
    y: Vec<f32>,
    z: Vec<f32>,
    vx: Vec<f32>,
    vy: Vec<f32>,
    vz: Vec<f32>,
    mass: Vec<f32>,
    is_gas: Vec<u8>,
    u: Vec<f32>,
    h: Vec<f32>,
    rho: Vec<f32>,
    pressure: Vec<f32>,
    balsara: Vec<f32>,
}

#[cfg(not(cuda_unavailable))]
impl SphSoa {
    fn from_particles(particles: &[SphParticle]) -> Self {
        let n = particles.len();
        let mut x = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        let mut z = Vec::with_capacity(n);
        let mut vx = Vec::with_capacity(n);
        let mut vy = Vec::with_capacity(n);
        let mut vz = Vec::with_capacity(n);
        let mut mass = Vec::with_capacity(n);
        let mut is_gas = Vec::with_capacity(n);
        let mut u = Vec::with_capacity(n);
        let mut h = Vec::with_capacity(n);
        let mut rho = Vec::with_capacity(n);
        let mut pressure = Vec::with_capacity(n);
        let mut balsara = Vec::with_capacity(n);

        for p in particles {
            x.push(p.position.x as f32);
            y.push(p.position.y as f32);
            z.push(p.position.z as f32);
            vx.push(p.velocity.x as f32);
            vy.push(p.velocity.y as f32);
            vz.push(p.velocity.z as f32);
            mass.push(p.mass as f32);
            if let Some(gas) = p.gas.as_ref() {
                is_gas.push(1);
                u.push(gas.u as f32);
                h.push(gas.h_sml as f32);
                rho.push(gas.rho as f32);
                pressure.push(gas.pressure as f32);
                balsara.push(gas.balsara as f32);
            } else {
                is_gas.push(0);
                u.push(0.0);
                h.push(1.0);
                rho.push(0.0);
                pressure.push(0.0);
                balsara.push(1.0);
            }
        }

        Self {
            x,
            y,
            z,
            vx,
            vy,
            vz,
            mass,
            is_gas,
            u,
            h,
            rho,
            pressure,
            balsara,
        }
    }
}

#[cfg(not(cuda_unavailable))]
fn periodic_box_f32(periodic_box: Option<f64>) -> f32 {
    periodic_box.map_or(-1.0, |box_size| box_size as f32)
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
    let _ = std::mem::size_of::<CudaSphSolver>();
};
