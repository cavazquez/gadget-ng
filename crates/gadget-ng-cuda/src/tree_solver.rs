//! CUDA tree/SIDM kernels used by the Phase 200 parity track.
//!
//! Versión optimizada: `CudaTreeSolver` retiene un [`CudaPool`] de buffers
//! device entre pasos, eliminando `cudaMalloc`/`cudaFree` por invocación.
//! Los buffers se redimensionan solo cuando el número de partículas crece.

use crate::pool::CudaPool;
use crate::{CudaExecutionError, CudaPmSolver, CudaUnavailable};
#[cfg(not(cuda_unavailable))]
use gadget_ng_core::ParticleType;
use gadget_ng_core::{BhMonopoleGpuNode, Particle, Vec3};
use gadget_ng_tree::RmnSoa;

/// Solver CUDA para kernels tree/SIDM con buffers device persistentes.
pub struct CudaTreeSolver {
    #[cfg(not(cuda_unavailable))]
    pool: CudaPool,
    #[cfg(cuda_unavailable)]
    _phantom: (),
}

impl std::fmt::Debug for CudaTreeSolver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaTreeSolver").finish()
    }
}

impl Clone for CudaTreeSolver {
    fn clone(&self) -> Self {
        Self::try_new_checked().unwrap_or_else(|_| {
            #[cfg(not(cuda_unavailable))]
            {
                panic!("CudaTreeSolver clone failed: CUDA not available");
            }
            #[cfg(cuda_unavailable)]
            {
                Self { _phantom: () }
            }
        })
    }
}

impl CudaTreeSolver {
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

    pub fn try_walk_monopole(
        &self,
        particles: &[Particle],
        g: f64,
        eps2: f64,
    ) -> Result<Vec<Vec3>, CudaExecutionError> {
        let n = particles.len();
        if n == 0 {
            return Ok(Vec::new());
        }

        #[cfg(cuda_unavailable)]
        {
            let _ = (particles, g, eps2);
            return Err(CudaExecutionError::Unavailable(CudaUnavailable {
                availability: CudaPmSolver::availability(),
            }));
        }

        #[cfg(not(cuda_unavailable))]
        {
            self.pool.ensure_capacity(n)?;
            self.pool.reset();

            let x: Vec<f32> = particles.iter().map(|p| p.position.x as f32).collect();
            let y: Vec<f32> = particles.iter().map(|p| p.position.y as f32).collect();
            let z: Vec<f32> = particles.iter().map(|p| p.position.z as f32).collect();
            let mass: Vec<f32> = particles.iter().map(|p| p.mass as f32).collect();

            // SAFETY: pool handle is valid, slots are freshly reset; all slices have length n.
            unsafe {
                let d_x = self.pool.upload_f32(0, &x);
                let d_y = self.pool.upload_f32(1, &y);
                let d_z = self.pool.upload_f32(2, &z);
                let d_mass = self.pool.upload_f32(3, &mass);
                let d_ax = self.pool.alloc_f32(4, n);
                let d_ay = self.pool.alloc_f32(5, n);
                let d_az = self.pool.alloc_f32(6, n);

                let code = crate::ffi::cuda_tree_walk_monopole(
                    d_x,
                    d_y,
                    d_z,
                    d_mass,
                    d_ax,
                    d_ay,
                    d_az,
                    n as i32,
                    g as f32,
                    eps2 as f32,
                );
                check_kernel("cuda_tree_walk_monopole", code)?;

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

    /// Aceleración gravitacional monopolo+cuadrupolo+octupolo a partir de nodos LET
    /// pre-seleccionados (ya con MAC aplicado en CPU).  Hexadecapolo excluido.
    ///
    /// # Parámetros
    /// - `particles` — partículas de las que se calcula la aceleración
    /// - `nodes`     — nodos LET en formato SoA (`RmnSoa`)
    /// - `g`         — constante gravitatoria
    /// - `eps2`      — suavizado Plummer al cuadrado
    pub fn try_tree_walk_let(
        &self,
        particles: &[Particle],
        nodes: &RmnSoa,
        g: f64,
        eps2: f64,
    ) -> Result<Vec<Vec3>, CudaExecutionError> {
        let np = particles.len();
        let nn = nodes.len;
        if np == 0 || nn == 0 {
            return Ok(vec![Vec3::zero(); np]);
        }

        #[cfg(cuda_unavailable)]
        {
            let _ = (particles, nodes, g, eps2);
            return Err(CudaExecutionError::Unavailable(CudaUnavailable {
                availability: CudaPmSolver::availability(),
            }));
        }

        #[cfg(not(cuda_unavailable))]
        {
            // Downcast partículas a f32
            let px: Vec<f32> = particles.iter().map(|p| p.position.x as f32).collect();
            let py: Vec<f32> = particles.iter().map(|p| p.position.y as f32).collect();
            let pz: Vec<f32> = particles.iter().map(|p| p.position.z as f32).collect();

            // Downcast nodos LET a f32
            let cx: Vec<f32> = nodes.cx.iter().map(|&v| v as f32).collect();
            let cy: Vec<f32> = nodes.cy.iter().map(|&v| v as f32).collect();
            let cz: Vec<f32> = nodes.cz.iter().map(|&v| v as f32).collect();
            let nm: Vec<f32> = nodes.mass.iter().map(|&v| v as f32).collect();
            let q: [Vec<f32>; 6] =
                std::array::from_fn(|k| nodes.quad[k].iter().map(|&v| v as f32).collect());
            let o: [Vec<f32>; 7] =
                std::array::from_fn(|k| nodes.oct[k].iter().map(|&v| v as f32).collect());

            let mut ax_out = vec![0.0_f32; np];
            let mut ay_out = vec![0.0_f32; np];
            let mut az_out = vec![0.0_f32; np];

            // SAFETY: los slices tienen longitud correcta (np / nn) verificada arriba.
            let code = unsafe {
                crate::ffi::cuda_tree_let_accel(
                    px.as_ptr(),
                    py.as_ptr(),
                    pz.as_ptr(),
                    np as i32,
                    cx.as_ptr(),
                    cy.as_ptr(),
                    cz.as_ptr(),
                    nm.as_ptr(),
                    q[0].as_ptr(),
                    q[1].as_ptr(),
                    q[2].as_ptr(),
                    q[3].as_ptr(),
                    q[4].as_ptr(),
                    q[5].as_ptr(),
                    o[0].as_ptr(),
                    o[1].as_ptr(),
                    o[2].as_ptr(),
                    o[3].as_ptr(),
                    o[4].as_ptr(),
                    o[5].as_ptr(),
                    o[6].as_ptr(),
                    nn as i32,
                    g as f32,
                    eps2 as f32,
                    ax_out.as_mut_ptr(),
                    ay_out.as_mut_ptr(),
                    az_out.as_mut_ptr(),
                )
            };
            check_kernel("cuda_tree_let_accel", code)?;

            Ok((0..np)
                .map(|i| Vec3::new(ax_out[i] as f64, ay_out[i] as f64, az_out[i] as f64))
                .collect())
        }
    }

    pub fn try_sidm_scatter(
        &self,
        particles: &mut [Particle],
        dt: f64,
        sigma_over_m: f64,
        h: f64,
    ) -> Result<(), CudaExecutionError> {
        let n = particles.len();
        if n == 0 {
            return Ok(());
        }

        #[cfg(cuda_unavailable)]
        {
            let _ = (particles, dt, sigma_over_m, h);
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
            let x: Vec<f32> = particles.iter().map(|p| p.position.x as f32).collect();
            let y: Vec<f32> = particles.iter().map(|p| p.position.y as f32).collect();
            let z: Vec<f32> = particles.iter().map(|p| p.position.z as f32).collect();
            let vx: Vec<f32> = particles.iter().map(|p| p.velocity.x as f32).collect();
            let vy: Vec<f32> = particles.iter().map(|p| p.velocity.y as f32).collect();
            let vz: Vec<f32> = particles.iter().map(|p| p.velocity.z as f32).collect();
            let mass: Vec<f32> = particles.iter().map(|p| p.mass as f32).collect();

            // SAFETY: pool handle is valid, slots are freshly reset; all slices have length n.
            unsafe {
                let d_ptype = self.pool.upload_u8(0, &ptype);
                let d_x = self.pool.upload_f32(1, &x);
                let d_y = self.pool.upload_f32(2, &y);
                let d_z = self.pool.upload_f32(3, &z);
                let d_vx = self.pool.upload_f32(4, &vx);
                let d_vy = self.pool.upload_f32(5, &vy);
                let d_vz = self.pool.upload_f32(6, &vz);
                let d_mass = self.pool.upload_f32(7, &mass);
                let d_ovx = self.pool.alloc_f32(8, n);
                let d_ovy = self.pool.alloc_f32(9, n);
                let d_ovz = self.pool.alloc_f32(10, n);

                let code = crate::ffi::cuda_tree_sidm_scatter(
                    d_ptype,
                    d_x,
                    d_y,
                    d_z,
                    d_vx,
                    d_vy,
                    d_vz,
                    d_mass,
                    d_ovx,
                    d_ovy,
                    d_ovz,
                    n as i32,
                    dt as f32,
                    sigma_over_m as f32,
                    h as f32,
                );
                check_kernel("cuda_tree_sidm_scatter", code)?;

                let mut ovx = vec![0.0_f32; n];
                let mut ovy = vec![0.0_f32; n];
                let mut ovz = vec![0.0_f32; n];
                self.pool.download_f32(&mut ovx, d_ovx)?;
                self.pool.download_f32(&mut ovy, d_ovy)?;
                self.pool.download_f32(&mut ovz, d_ovz)?;

                for (i, p) in particles.iter_mut().enumerate() {
                    p.velocity = Vec3::new(ovx[i] as f64, ovy[i] as f64, ovz[i] as f64);
                }
            }
            Ok(())
        }
    }

    /// Walk Barnes-Hut monopolo en GPU con pila por hilo (AP-20).
    ///
    /// Los nodos (`nodes`) se obtienen vía `Octree::export_bh_monopole_gpu_nodes()`.
    /// El criterio MAC `θ²` controla la apertura; `theta = 0.5` es el valor estándar.
    /// La fuerza es Plummer con suavizado `eps2`. No aplica periodicidad.
    #[expect(clippy::too_many_arguments)]
    pub fn try_bh_local_walk(
        &self,
        positions: &[Vec3],
        target_indices: &[usize],
        nodes: &[BhMonopoleGpuNode],
        root: u32,
        theta: f64,
        g: f64,
        eps2: f64,
    ) -> Result<Vec<Vec3>, CudaExecutionError> {
        let n = target_indices.len();
        if n == 0 || nodes.is_empty() {
            return Ok(vec![Vec3::zero(); n]);
        }

        #[cfg(cuda_unavailable)]
        {
            let _ = (positions, target_indices, nodes, root, theta, g, eps2);
            return Err(CudaExecutionError::Unavailable(crate::CudaUnavailable {
                availability: CudaPmSolver::availability(),
            }));
        }

        #[cfg(not(cuda_unavailable))]
        {
            let qx: Vec<f32> = target_indices
                .iter()
                .map(|&i| positions[i].x as f32)
                .collect();
            let qy: Vec<f32> = target_indices
                .iter()
                .map(|&i| positions[i].y as f32)
                .collect();
            let qz: Vec<f32> = target_indices
                .iter()
                .map(|&i| positions[i].z as f32)
                .collect();
            let tidx: Vec<u32> = target_indices.iter().map(|&i| i as u32).collect();
            let mut ax = vec![0.0_f32; n];
            let mut ay = vec![0.0_f32; n];
            let mut az = vec![0.0_f32; n];
            let theta2 = (theta * theta) as f32;
            let n_nodes = nodes.len() as i32;

            // SAFETY: BhMonopoleGpuNode es repr(C); el puntero es válido.
            let code = unsafe {
                crate::ffi::cuda_bh_walk_monopole(
                    nodes.as_ptr() as *const std::ffi::c_void,
                    n_nodes,
                    root,
                    qx.as_ptr(),
                    qy.as_ptr(),
                    qz.as_ptr(),
                    tidx.as_ptr(),
                    ax.as_mut_ptr(),
                    ay.as_mut_ptr(),
                    az.as_mut_ptr(),
                    n as i32,
                    theta2,
                    g as f32,
                    eps2 as f32,
                )
            };
            check_kernel("cuda_bh_walk_monopole", code)?;
            Ok(ax
                .iter()
                .zip(ay.iter())
                .zip(az.iter())
                .map(|((&axi, &ayi), &azi)| Vec3::new(axi as f64, ayi as f64, azi as f64))
                .collect())
        }
    }

    /// Calcula aceleraciones TreePM de corto alcance en GPU O(N²) con erfc y mínima imagen.
    ///
    /// Replica `gadget_ng_treepm::short_range_accels_sfc` (O(N²) sin árbol) usando
    /// el mismo polinomio erfc de Abramowitz & Stegun §7.1.26. Tolerancia f32 vs f64 < 5%.
    #[expect(clippy::too_many_arguments)]
    pub fn try_short_range(
        &self,
        positions: &[gadget_ng_core::Vec3],
        masses: &[f64],
        r_split: f64,
        r_cut: f64,
        eps2: f64,
        g: f64,
        box_size: Option<f64>,
    ) -> Result<Vec<gadget_ng_core::Vec3>, CudaExecutionError> {
        let n = positions.len();
        assert_eq!(masses.len(), n);
        if n == 0 {
            return Ok(vec![]);
        }

        #[cfg(cuda_unavailable)]
        {
            let _ = (positions, masses, r_split, r_cut, eps2, g, box_size);
            return Err(CudaExecutionError::Unavailable(crate::CudaUnavailable {
                availability: CudaPmSolver::availability(),
            }));
        }

        #[cfg(not(cuda_unavailable))]
        {
            let x: Vec<f32> = positions.iter().map(|p| p.x as f32).collect();
            let y: Vec<f32> = positions.iter().map(|p| p.y as f32).collect();
            let z: Vec<f32> = positions.iter().map(|p| p.z as f32).collect();
            let mass: Vec<f32> = masses.iter().map(|&m| m as f32).collect();
            let mut ax = vec![0.0_f32; n];
            let mut ay = vec![0.0_f32; n];
            let mut az = vec![0.0_f32; n];
            let r_cut2 = (r_cut * r_cut) as f32;
            let bs = box_size.unwrap_or(0.0) as f32;

            // SAFETY: todos los slices tienen longitud n.
            let code = unsafe {
                crate::ffi::cuda_treepm_short_range(
                    x.as_ptr(),
                    y.as_ptr(),
                    z.as_ptr(),
                    mass.as_ptr(),
                    ax.as_mut_ptr(),
                    ay.as_mut_ptr(),
                    az.as_mut_ptr(),
                    n as i32,
                    r_split as f32,
                    r_cut2,
                    eps2 as f32,
                    g as f32,
                    bs,
                )
            };
            check_kernel("cuda_treepm_short_range", code)?;
            Ok(ax
                .iter()
                .zip(ay.iter())
                .zip(az.iter())
                .map(|((&axi, &ayi), &azi)| {
                    gadget_ng_core::Vec3::new(axi as f64, ayi as f64, azi as f64)
                })
                .collect())
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
    let _ = std::mem::size_of::<CudaTreeSolver>();
};
