//! Bindings FFI a las funciones C expuestas por `cuda/pm_gravity.cu`.
//!
//! Solo se compila cuando CUDA está disponible (es decir, cuando NO se emite
//! el cfg `cuda_unavailable` desde build.rs).

#[cfg(not(cuda_unavailable))]
extern "C" {
    /// Crea un handle PM CUDA con grilla `grid_size³` y caja periódica `box_size`.
    ///
    /// Devuelve `NULL` si la inicialización falla (sin dispositivo CUDA, sin VRAM, …).
    pub fn cuda_pm_create(grid_size: i32, box_size: f32) -> *mut std::ffi::c_void;

    /// Libera todos los recursos device y host asociados al handle.
    pub fn cuda_pm_destroy(h: *mut std::ffi::c_void);

    /// Ejecuta el solver PM completo (CIC → FFT → Poisson → iFFT → CIC interp).
    ///
    /// # Parámetros
    /// - `h`            — handle creado con `cuda_pm_create`
    /// - `x,y,z`        — posiciones de las `n` partículas (en unidades de `box_size`)
    /// - `mass`         — masas de las `n` partículas
    /// - `ax,ay,az`     — buffers de salida (aceleraciones); deben tener longitud `n`
    /// - `n`            — número de partículas
    /// - `eps2`         — suavizado Plummer² (no usado en PM puro; reservado)
    /// - `g`            — constante gravitatoria
    ///
    /// # Retorna
    /// `0` si OK, código de error CUDA en caso contrario.
    pub fn cuda_pm_solve(
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
