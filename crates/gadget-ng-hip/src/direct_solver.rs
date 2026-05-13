//! `HipDirectGravity` — solver de gravedad directa N² GPU via HIP/ROCm (Phase 163 / V1).
//!
//! ## Algoritmo
//!
//! Idéntico al de `CudaDirectGravity`, pero usando kernels HIP/ROCm:
//!
//! ```text
//! a_i += G * m_j * (r_j - r_i) / (|r_j - r_i|² + ε²)^(3/2)
//! ```
//!
//! El método `try_new` devuelve `Some(Self)` sólo si hay hardware HIP/ROCm
//! disponible. El método `compute` llama al kernel real via FFI cuando
//! HIP está disponible.
//!
//! ## Uso
//!
//! ```rust,ignore
//! if let Some(gpu) = HipDirectGravity::try_new(eps) {
//!     let accels = gpu.compute(&positions, &masses);
//! }
//! ```

use crate::{HipExecutionError, HipPmSolver, HipUnavailable};

/// Solver de gravedad directa N² via HIP/ROCm.
///
/// Construir con [`HipDirectGravity::try_new`]; devuelve `None` si HIP/ROCm
/// no está disponible en el host.
pub struct HipDirectGravity {
    /// Softening gravitacional ε (en unidades internas). Se pasa al kernel.
    pub eps: f32,
    /// Número de hilos por workgroup HIP (potencia de 2, típico: 256).
    pub workgroup_size: usize,
}

impl HipDirectGravity {
    /// Intenta construir el solver de gravedad directa HIP.
    ///
    /// Devuelve `None` si no hay hardware HIP/ROCm disponible o si el crate fue
    /// compilado sin soporte HIP (`hip_unavailable`).
    ///
    /// # Parámetros
    /// - `eps` — softening gravitacional en unidades internas
    pub fn try_new(eps: f32) -> Option<Self> {
        Self::try_new_checked(eps).ok()
    }

    /// Variante fallible de [`Self::try_new`] que conserva el motivo de indisponibilidad.
    pub fn try_new_checked(eps: f32) -> Result<Self, HipUnavailable> {
        if HipPmSolver::is_available() {
            Ok(Self {
                eps,
                workgroup_size: 256,
            })
        } else {
            Err(HipUnavailable {
                availability: HipPmSolver::availability(),
            })
        }
    }

    /// Calcula aceleraciones gravitacionales directas O(N²) para N partículas.
    ///
    /// # Parámetros
    /// - `pos`  — posiciones `[[x, y, z]; N]` en unidades internas (f32)
    /// - `mass` — masas `[m_0, ..., m_{N-1}]` en unidades internas (f32)
    ///
    /// # Retorna
    ///
    /// Vector de aceleraciones `[[ax, ay, az]; N]` en unidades internas.
    ///
    /// # Panics
    ///
    /// Panics si HIP no está disponible en tiempo de compilación (`hip_unavailable`).
    pub fn compute(&self, pos: &[[f32; 3]], mass: &[f32]) -> Vec<[f32; 3]> {
        match self.try_compute(pos, mass) {
            Ok(accels) => accels,
            Err(err) => panic!("{err}"),
        }
    }

    /// Calcula aceleraciones directas y devuelve un error explícito si HIP falla.
    pub fn try_compute(
        &self,
        pos: &[[f32; 3]],
        mass: &[f32],
    ) -> Result<Vec<[f32; 3]>, HipExecutionError> {
        let n = pos.len();
        assert_eq!(mass.len(), n, "pos y mass deben tener la misma longitud");

        #[cfg(hip_unavailable)]
        {
            let _ = (pos, mass);
            Err(HipUnavailable {
                availability: HipPmSolver::availability(),
            }
            .into())
        }

        #[cfg(not(hip_unavailable))]
        {
            use crate::ffi;
            use std::ffi::c_void;

            let eps2 = self.eps * self.eps;
            let g = 1.0_f32;

            // SAFETY: hip_direct_create compilada con las mismas convenciones ABI.
            // eps2 y workgroup_size son escalares válidos. Handle se verifica no-NULL.
            let handle: *mut c_void =
                unsafe { ffi::hip_direct_create(eps2, self.workgroup_size as i32) };
            if handle.is_null() {
                return Err(HipExecutionError::CreateFailed("HipDirectGravity"));
            }

            let mut x: Vec<f32> = Vec::with_capacity(n);
            let mut y: Vec<f32> = Vec::with_capacity(n);
            let mut z: Vec<f32> = Vec::with_capacity(n);
            for p in pos {
                x.push(p[0]);
                y.push(p[1]);
                z.push(p[2]);
            }

            let mut ax = vec![0.0_f32; n];
            let mut ay = vec![0.0_f32; n];
            let mut az = vec![0.0_f32; n];

            // SAFETY: handle es no-NULL. Punteros de Vec<f32> válidos con longitud n.
            // Los buffers de salida (ax, ay, az) tienen capacidad n.
            let ret = unsafe {
                ffi::hip_direct_solve(
                    handle,
                    x.as_ptr(),
                    y.as_ptr(),
                    z.as_ptr(),
                    mass.as_ptr(),
                    ax.as_mut_ptr(),
                    ay.as_mut_ptr(),
                    az.as_mut_ptr(),
                    n as i32,
                    g,
                )
            };

            // SAFETY: handle es válido y no se usará después de destroy.
            // hip_direct_destroy libera todos los recursos GPU asociados.
            unsafe { ffi::hip_direct_destroy(handle) };

            if ret != 0 {
                return Err(HipExecutionError::KernelFailed {
                    kernel: "hip_direct_solve",
                    code: ret,
                });
            }

            Ok((0..n).map(|i| [ax[i], ay[i], az[i]]).collect())
        }
    }

    /// Número de partículas máximo recomendado para este solver.
    pub fn recommended_max_n(&self) -> usize {
        65536
    }
}

// ── GravitySolver bridge ────────────────────────────────────────────────────

impl gadget_ng_core::gravity::GravitySolver for HipDirectGravity {
    fn accelerations_for_indices(
        &self,
        global_positions: &[gadget_ng_core::Vec3],
        global_masses: &[f64],
        _eps2: f64,
        _g: f64,
        global_indices: &[usize],
        out: &mut [gadget_ng_core::Vec3],
    ) {
        assert_eq!(global_indices.len(), out.len());
        if global_indices.is_empty() {
            return;
        }
        let pos_f32: Vec<[f32; 3]> = global_positions
            .iter()
            .map(|p| [p.x as f32, p.y as f32, p.z as f32])
            .collect();
        let mass_f32: Vec<f32> = global_masses.iter().map(|&m| m as f32).collect();
        let acc = self.compute(&pos_f32, &mass_f32);
        for (k, &gi) in global_indices.iter().enumerate() {
            let [ax, ay, az] = acc[gi];
            out[k] = gadget_ng_core::Vec3::new(ax as f64, ay as f64, az as f64);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests unitarios (sin hardware HIP requerido)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn try_new_returns_none_without_hip() {
        let result = HipDirectGravity::try_new(0.01);
        match result {
            None => {
                println!("HIP no disponible (esperado en CI): try_new = None");
            }
            Some(ref solver) => {
                println!("HIP disponible: workgroup_size = {}", solver.workgroup_size);
                assert!(solver.eps > 0.0);
                assert!(solver.workgroup_size > 0);
            }
        }
    }

    #[test]
    fn recommended_max_n_is_positive() {
        let solver = HipDirectGravity {
            eps: 0.01,
            workgroup_size: 256,
        };
        assert!(solver.recommended_max_n() > 0);
    }
}
