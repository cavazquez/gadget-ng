//! Bindings FFI a las funciones C expuestas por `hip/pm_gravity.hip`.
//!
//! Solo se compila cuando HIP está disponible (es decir, cuando NO se emite
//! el cfg `hip_unavailable` desde build.rs).

#[cfg(not(hip_unavailable))]
extern "C" {
    /// Crea un handle PM HIP con grilla `grid_size³` y caja periódica `box_size`.
    ///
    /// Devuelve `NULL` si la inicialización falla (sin dispositivo HIP/ROCm, …).
    pub fn hip_pm_create(grid_size: i32, box_size: f32) -> *mut std::ffi::c_void;

    /// Libera todos los recursos device y host asociados al handle.
    pub fn hip_pm_destroy(h: *mut std::ffi::c_void);

    /// Ejecuta el solver PM completo (CIC → rocFFT → Poisson → irocFFT → CIC interp).
    ///
    /// Misma semántica que `cuda_pm_solve`; diferencia interna: usa rocFFT en lugar
    /// de cuFFT y `hipMalloc`/`hipMemcpy` en lugar de `cudaMalloc`/`cudaMemcpy`.
    ///
    /// @return `0` si OK, código de error HIP en caso contrario.
    pub fn hip_pm_solve(
        h: *mut std::ffi::c_void,
        x: *const f32,
        y: *const f32,
        z: *const f32,
        mass: *const f32,
        ax: *mut f32,
        ay: *mut f32,
        az: *mut f32,
        n: i32,
        eps2: f32,
        g: f32,
    ) -> i32;
}
