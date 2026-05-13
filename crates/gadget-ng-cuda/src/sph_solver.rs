//! Kernels SPH O(N²) via CUDA.
//!
//! Esta primera versión expone wrappers fallibles para densidad, fuerzas SPH
//! clásicas, limitador de Balsara y fuerzas Gadget-2. Los kernels usan f32 en
//! device y escriben los resultados de vuelta sobre [`gadget_ng_sph::SphParticle`].

use crate::{CudaExecutionError, CudaPmSolver, CudaUnavailable};
#[cfg(not(cuda_unavailable))]
use gadget_ng_core::Vec3;
use gadget_ng_sph::particle::SphParticle;

/// Solver CUDA para kernels SPH locales O(N²).
#[derive(Debug, Clone, Copy)]
pub struct CudaSphSolver;

#[derive(Debug)]
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

impl CudaSphSolver {
    /// Intenta crear el solver SPH CUDA.
    pub fn try_new() -> Option<Self> {
        Self::try_new_checked().ok()
    }

    /// Variante fallible de [`Self::try_new`] que conserva el diagnóstico.
    pub fn try_new_checked() -> Result<Self, CudaUnavailable> {
        if CudaPmSolver::is_available() {
            Ok(Self)
        } else {
            Err(CudaUnavailable {
                availability: CudaPmSolver::availability(),
            })
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
            Err(CudaUnavailable {
                availability: CudaPmSolver::availability(),
            }
            .into())
        }

        #[cfg(not(cuda_unavailable))]
        {
            let soa = SphSoa::from_particles(particles);
            let mut h_out = vec![0.0_f32; n];
            let mut rho_out = vec![0.0_f32; n];
            let mut pressure_out = vec![0.0_f32; n];
            let mut entropy_out = vec![0.0_f32; n];
            let code = unsafe {
                crate::ffi::cuda_sph_density(
                    soa.x.as_ptr(),
                    soa.y.as_ptr(),
                    soa.z.as_ptr(),
                    soa.mass.as_ptr(),
                    soa.is_gas.as_ptr(),
                    soa.u.as_ptr(),
                    soa.h.as_ptr(),
                    h_out.as_mut_ptr(),
                    rho_out.as_mut_ptr(),
                    pressure_out.as_mut_ptr(),
                    entropy_out.as_mut_ptr(),
                    n as i32,
                    periodic_box_f32(periodic_box),
                )
            };
            check_kernel("cuda_sph_density", code)?;
            for (i, p) in particles.iter_mut().enumerate() {
                if let Some(gas) = p.gas.as_mut() {
                    gas.h_sml = h_out[i] as f64;
                    gas.rho = rho_out[i] as f64;
                    gas.pressure = pressure_out[i] as f64;
                    gas.entropy = entropy_out[i] as f64;
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
            Err(CudaUnavailable {
                availability: CudaPmSolver::availability(),
            }
            .into())
        }

        #[cfg(not(cuda_unavailable))]
        {
            let soa = SphSoa::from_particles(particles);
            let mut balsara_out = vec![1.0_f32; n];
            let code = unsafe {
                crate::ffi::cuda_sph_balsara(
                    soa.x.as_ptr(),
                    soa.y.as_ptr(),
                    soa.z.as_ptr(),
                    soa.vx.as_ptr(),
                    soa.vy.as_ptr(),
                    soa.vz.as_ptr(),
                    soa.mass.as_ptr(),
                    soa.is_gas.as_ptr(),
                    soa.rho.as_ptr(),
                    soa.pressure.as_ptr(),
                    soa.h.as_ptr(),
                    balsara_out.as_mut_ptr(),
                    n as i32,
                    periodic_box_f32(periodic_box),
                )
            };
            check_kernel("cuda_sph_balsara", code)?;
            for (i, p) in particles.iter_mut().enumerate() {
                if let Some(gas) = p.gas.as_mut() {
                    gas.balsara = balsara_out[i] as f64;
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
            Err(CudaUnavailable {
                availability: CudaPmSolver::availability(),
            }
            .into())
        }

        #[cfg(not(cuda_unavailable))]
        {
            let soa = SphSoa::from_particles(particles);
            let mut ax = vec![0.0_f32; n];
            let mut ay = vec![0.0_f32; n];
            let mut az = vec![0.0_f32; n];
            let mut du_dt = vec![0.0_f32; n];
            let code = unsafe {
                crate::ffi::cuda_sph_forces(
                    soa.x.as_ptr(),
                    soa.y.as_ptr(),
                    soa.z.as_ptr(),
                    soa.vx.as_ptr(),
                    soa.vy.as_ptr(),
                    soa.vz.as_ptr(),
                    soa.mass.as_ptr(),
                    soa.is_gas.as_ptr(),
                    soa.rho.as_ptr(),
                    soa.pressure.as_ptr(),
                    soa.h.as_ptr(),
                    ax.as_mut_ptr(),
                    ay.as_mut_ptr(),
                    az.as_mut_ptr(),
                    du_dt.as_mut_ptr(),
                    n as i32,
                    periodic_box_f32(periodic_box),
                )
            };
            check_kernel("cuda_sph_forces", code)?;
            for (i, p) in particles.iter_mut().enumerate() {
                if let Some(gas) = p.gas.as_mut() {
                    gas.acc_sph = Vec3::new(ax[i] as f64, ay[i] as f64, az[i] as f64);
                    gas.du_dt = du_dt[i] as f64;
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
            Err(CudaUnavailable {
                availability: CudaPmSolver::availability(),
            }
            .into())
        }

        #[cfg(not(cuda_unavailable))]
        {
            let soa = SphSoa::from_particles(particles);
            let mut ax = vec![0.0_f32; n];
            let mut ay = vec![0.0_f32; n];
            let mut az = vec![0.0_f32; n];
            let mut da_dt = vec![0.0_f32; n];
            let mut du_dt = vec![0.0_f32; n];
            let mut max_vsig = vec![0.0_f32; n];
            let code = unsafe {
                crate::ffi::cuda_sph_gadget2_forces(
                    soa.x.as_ptr(),
                    soa.y.as_ptr(),
                    soa.z.as_ptr(),
                    soa.vx.as_ptr(),
                    soa.vy.as_ptr(),
                    soa.vz.as_ptr(),
                    soa.mass.as_ptr(),
                    soa.is_gas.as_ptr(),
                    soa.rho.as_ptr(),
                    soa.pressure.as_ptr(),
                    soa.h.as_ptr(),
                    soa.balsara.as_ptr(),
                    ax.as_mut_ptr(),
                    ay.as_mut_ptr(),
                    az.as_mut_ptr(),
                    da_dt.as_mut_ptr(),
                    du_dt.as_mut_ptr(),
                    max_vsig.as_mut_ptr(),
                    n as i32,
                    periodic_box_f32(periodic_box),
                )
            };
            check_kernel("cuda_sph_gadget2_forces", code)?;
            for (i, p) in particles.iter_mut().enumerate() {
                if let Some(gas) = p.gas.as_mut() {
                    gas.acc_sph = Vec3::new(ax[i] as f64, ay[i] as f64, az[i] as f64);
                    gas.da_dt = da_dt[i] as f64;
                    gas.du_dt = du_dt[i] as f64;
                    gas.max_vsig = max_vsig[i] as f64;
                }
            }
            Ok(())
        }
    }
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
