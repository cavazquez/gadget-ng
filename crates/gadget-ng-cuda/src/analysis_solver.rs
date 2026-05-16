//! Kernels CUDA para análisis in-situ: spin de halo, luminosidad galáctica y X-ray.
//!
//! Cada función es una reducción paralela en GPU con acumulación en shared memory
//! y atomics dobles para la salida global.

use crate::{CudaExecutionError, CudaPmSolver, CudaUnavailable};
use gadget_ng_core::{Particle, ParticleType, Vec3};

/// Solver CUDA para métricas de análisis in-situ.
pub struct CudaAnalysisSolver {
    #[cfg(cuda_unavailable)]
    _phantom: (),
    #[cfg(not(cuda_unavailable))]
    _marker: (),
}

impl std::fmt::Debug for CudaAnalysisSolver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaAnalysisSolver").finish()
    }
}

impl CudaAnalysisSolver {
    /// Crea el solver; devuelve `Err` si CUDA no está disponible.
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
            Ok(Self { _marker: () })
        }
    }

    /// Calcula el momento angular total L = Σ m_i × (r_i - r_com) × (v_i - v_com).
    ///
    /// # Parámetros
    /// - `positions`, `velocities`, `masses` — arrays de partículas del halo.
    /// - `pos_com`, `vel_com` — centro de masa y velocidad del centro de masa.
    ///
    /// # Retorna
    /// Vector L = (Lx, Ly, Lz) o error si CUDA falla.
    pub fn try_halo_spin(
        &self,
        positions: &[Vec3],
        velocities: &[Vec3],
        masses: &[f64],
        pos_com: [f64; 3],
        vel_com: [f64; 3],
    ) -> Result<[f64; 3], CudaExecutionError> {
        let n = positions.len();
        if n == 0 {
            return Ok([0.0; 3]);
        }

        #[cfg(cuda_unavailable)]
        {
            let _ = (positions, velocities, masses, pos_com, vel_com);
            return Err(CudaExecutionError::Unavailable(CudaUnavailable {
                availability: CudaPmSolver::availability(),
            }));
        }

        #[cfg(not(cuda_unavailable))]
        {
            let x: Vec<f32> = positions.iter().map(|p| p.x as f32).collect();
            let y: Vec<f32> = positions.iter().map(|p| p.y as f32).collect();
            let z: Vec<f32> = positions.iter().map(|p| p.z as f32).collect();
            let vx: Vec<f32> = velocities.iter().map(|p| p.x as f32).collect();
            let vy: Vec<f32> = velocities.iter().map(|p| p.y as f32).collect();
            let vz: Vec<f32> = velocities.iter().map(|p| p.z as f32).collect();
            let m: Vec<f32> = masses.iter().map(|&v| v as f32).collect();

            let mut lx = 0.0_f64;
            let mut ly = 0.0_f64;
            let mut lz = 0.0_f64;

            // SAFETY: arrays are valid f32 slices; output pointers are on the stack.
            let code = unsafe {
                crate::ffi::cuda_analysis_halo_spin(
                    x.as_ptr(),
                    y.as_ptr(),
                    z.as_ptr(),
                    vx.as_ptr(),
                    vy.as_ptr(),
                    vz.as_ptr(),
                    m.as_ptr(),
                    n as i32,
                    pos_com[0] as f32,
                    pos_com[1] as f32,
                    pos_com[2] as f32,
                    vel_com[0] as f32,
                    vel_com[1] as f32,
                    vel_com[2] as f32,
                    &mut lx,
                    &mut ly,
                    &mut lz,
                )
            };
            check_kernel("cuda_analysis_halo_spin", code)?;
            Ok([lx, ly, lz])
        }
    }

    /// Calcula luminosidad estelar total + colores (B-V, g-r) para un conjunto de partículas.
    ///
    /// Solo las partículas `Star` contribuyen. Modelo BC03-lite analítico.
    ///
    /// # Retorna
    /// `(l_total, bv, gr, n_stars)` o error.
    pub fn try_galaxy_luminosity(
        &self,
        particles: &[Particle],
    ) -> Result<(f64, f64, f64, usize), CudaExecutionError> {
        let n = particles.len();
        if n == 0 {
            return Ok((0.0, 0.0, 0.0, 0));
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
            let ptype: Vec<u8> = particles
                .iter()
                .map(|p| match p.ptype {
                    ParticleType::DarkMatter => 0,
                    ParticleType::Gas => 1,
                    ParticleType::Star => 2,
                })
                .collect();
            let mass: Vec<f32> = particles.iter().map(|p| p.mass as f32).collect();
            let age: Vec<f32> = particles.iter().map(|p| p.stellar_age as f32).collect();
            let z: Vec<f32> = particles.iter().map(|p| p.metallicity as f32).collect();

            let mut l_total = 0.0_f64;
            let mut bv_w = 0.0_f64;
            let mut gr_w = 0.0_f64;
            let mut n_stars = 0_i32;

            // SAFETY: arrays are valid f32 slices; output pointers are on the stack.
            let code = unsafe {
                crate::ffi::cuda_analysis_luminosity(
                    ptype.as_ptr(),
                    mass.as_ptr(),
                    age.as_ptr(),
                    z.as_ptr(),
                    n as i32,
                    &mut l_total,
                    &mut bv_w,
                    &mut gr_w,
                    &mut n_stars,
                )
            };
            check_kernel("cuda_analysis_luminosity", code)?;

            let (bv, gr) = if l_total > 0.0 {
                (bv_w / l_total, gr_w / l_total)
            } else {
                (0.0, 0.0)
            };
            Ok((l_total, bv, gr, n_stars as usize))
        }
    }

    /// Calcula la luminosidad total de rayos X (bremsstrahlung) para un conjunto de partículas.
    ///
    /// Solo las partículas de gas contribuyen.
    pub fn try_xray_luminosity(
        &self,
        particles: &[Particle],
        gamma: f64,
    ) -> Result<f64, CudaExecutionError> {
        let n = particles.len();
        if n == 0 {
            return Ok(0.0);
        }

        #[cfg(cuda_unavailable)]
        {
            let _ = (particles, gamma);
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
            let mass: Vec<f32> = particles.iter().map(|p| p.mass as f32).collect();
            let h: Vec<f32> = particles
                .iter()
                .map(|p| p.smoothing_length as f32)
                .collect();
            let u: Vec<f32> = particles.iter().map(|p| p.internal_energy as f32).collect();

            let mut lx = 0.0_f64;

            // SAFETY: arrays are valid f32 slices; output pointer is on the stack.
            let code = unsafe {
                crate::ffi::cuda_analysis_xray(
                    ptype.as_ptr(),
                    mass.as_ptr(),
                    h.as_ptr(),
                    u.as_ptr(),
                    n as i32,
                    gamma as f32,
                    &mut lx,
                )
            };
            check_kernel("cuda_analysis_xray", code)?;
            Ok(lx)
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
fn check_kernel(_kernel: &'static str, code: i32) -> Result<(), CudaExecutionError> {
    if code == 0 {
        Ok(())
    } else {
        Err(CudaExecutionError::KernelFailed {
            kernel: "unavailable",
            code,
        })
    }
}
