//! `HipPmSolver` — wrapper Rust sobre el solver PM HIP/ROCm.

#![allow(clippy::needless_return)]

use std::ffi::c_void;

use gadget_ng_core::gravity::GravitySolver;
use gadget_ng_core::vec3::Vec3;

// ── HipPmSolver ───────────────────────────────────────────────────────────────

/// Solver PM GPU implementado con HIP/ROCm (rocFFT).
///
/// Construir con [`HipPmSolver::try_new`]; devuelve `None` si HIP no está
/// disponible en el host o si el crate se compiló sin hipcc (toolchain ausente).
pub struct HipPmSolver {
    #[cfg(not(hip_unavailable))]
    handle: *mut c_void,
    #[cfg(hip_unavailable)]
    _phantom: (),
    grid_size: usize,
    #[cfg_attr(hip_unavailable, allow(dead_code))]
    r_split: f32,
}

// SAFETY: el handle HIP es propiedad exclusiva de este struct; no se comparte.
unsafe impl Send for HipPmSolver {}
unsafe impl Sync for HipPmSolver {}

impl HipPmSolver {
    /// `true` si el crate se compiló con soporte HIP y hay un dispositivo disponible.
    pub fn is_available() -> bool {
        #[cfg(hip_unavailable)]
        return false;

        #[cfg(not(hip_unavailable))]
        {
            let h = unsafe { crate::ffi::hip_pm_create(8, 1.0) };
            if h.is_null() {
                return false;
            }
            unsafe { crate::ffi::hip_pm_destroy(h) };
            true
        }
    }

    /// Intenta construir el solver PM HIP.
    ///
    /// # Parámetros
    /// - `grid_size` — lado de la grilla PM (e.g. 64, 128, 256)
    /// - `box_size`  — tamaño de la caja periódica en las mismas unidades que las posiciones
    ///
    /// Devuelve `None` si HIP no está disponible o si la inicialización falla.
    pub fn try_new(grid_size: usize, box_size: f64) -> Option<Self> {
        Self::try_new_with_r_split(grid_size, box_size, 0.0)
    }

    /// Igual que [`Self::try_new`] con filtro Gaussiano TreePM opcional.
    pub fn try_new_with_r_split(grid_size: usize, box_size: f64, r_split: f64) -> Option<Self> {
        #[cfg(hip_unavailable)]
        {
            let _ = (grid_size, box_size, r_split);
            return None;
        }

        #[cfg(not(hip_unavailable))]
        {
            let handle = unsafe { crate::ffi::hip_pm_create(grid_size as i32, box_size as f32) };
            if handle.is_null() {
                return None;
            }
            Some(Self {
                handle,
                grid_size,
                r_split: r_split as f32,
            })
        }
    }

    pub fn r_split(&self) -> f32 {
        #[cfg(hip_unavailable)]
        {
            return 0.0;
        }
        #[cfg(not(hip_unavailable))]
        {
            self.r_split
        }
    }

    /// Tamaño de grilla con el que fue construido el solver.
    pub fn grid_size(&self) -> usize {
        self.grid_size
    }
}

impl Drop for HipPmSolver {
    fn drop(&mut self) {
        #[cfg(not(hip_unavailable))]
        unsafe {
            crate::ffi::hip_pm_destroy(self.handle);
        }
    }
}

// ── GravitySolver ─────────────────────────────────────────────────────────────

impl GravitySolver for HipPmSolver {
    /// Calcula aceleraciones PM GPU para las partículas en `global_indices`.
    ///
    /// Idéntica lógica que `CudaPmSolver::accelerations_for_indices`; la diferencia
    /// está en las llamadas FFI subyacentes (HIP vs CUDA).
    fn accelerations_for_indices(
        &self,
        global_positions: &[Vec3],
        global_masses: &[f64],
        eps2: f64,
        g: f64,
        global_indices: &[usize],
        out: &mut [Vec3],
    ) {
        assert_eq!(global_indices.len(), out.len());
        if global_indices.is_empty() {
            return;
        }

        #[cfg(hip_unavailable)]
        {
            let _ = (global_positions, global_masses, eps2, g, global_indices);
            for v in out.iter_mut() {
                *v = Vec3::zero();
            }
            return;
        }

        #[cfg(not(hip_unavailable))]
        {
            let n = global_positions.len();

            let mut xs: Vec<f32> = Vec::with_capacity(n);
            let mut ys: Vec<f32> = Vec::with_capacity(n);
            let mut zs: Vec<f32> = Vec::with_capacity(n);
            let mut masses: Vec<f32> = Vec::with_capacity(n);
            for (p, &m) in global_positions.iter().zip(global_masses.iter()) {
                xs.push(p.x as f32);
                ys.push(p.y as f32);
                zs.push(p.z as f32);
                masses.push(m as f32);
            }

            let mut ax: Vec<f32> = vec![0.0f32; n];
            let mut ay: Vec<f32> = vec![0.0f32; n];
            let mut az: Vec<f32> = vec![0.0f32; n];

            let ret = unsafe {
                crate::ffi::hip_pm_solve(
                    self.handle,
                    xs.as_ptr(),
                    ys.as_ptr(),
                    zs.as_ptr(),
                    masses.as_ptr(),
                    ax.as_mut_ptr(),
                    ay.as_mut_ptr(),
                    az.as_mut_ptr(),
                    n as i32,
                    eps2 as f32,
                    g as f32,
                    self.r_split,
                )
            };
            if ret != 0 {
                eprintln!("[HipPmSolver] hip_pm_solve error code {ret}");
                for v in out.iter_mut() {
                    *v = Vec3::zero();
                }
                return;
            }

            for (j, &gi) in global_indices.iter().enumerate() {
                out[j] = Vec3::new(ax[gi] as f64, ay[gi] as f64, az[gi] as f64);
            }
        }
    }
}

// Referencia al tipo para silenciar el warning de c_void cuando hip_unavailable.
#[cfg(hip_unavailable)]
const _: () = {
    let _ = std::mem::size_of::<c_void>();
};
