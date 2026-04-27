//! Bindings FFI a las funciones C expuestas por `cuda/pm_gravity.cu` y
//! `cuda/direct_gravity.cu`.
//!
//! Solo se compilan cuando CUDA está disponible (es decir, cuando NO se emite
//! el cfg `cuda_unavailable` desde build.rs).

#[cfg(not(cuda_unavailable))]
unsafe extern "C" {
    // ── Solver PM (CIC + FFT + Poisson) ──────────────────────────────────────

    /// Crea un handle PM CUDA con grilla `grid_size³` y caja periódica `box_size`.
    ///
    /// Devuelve `NULL` si la inicialización falla (sin dispositivo CUDA, sin VRAM, …).
    pub fn cuda_pm_create(grid_size: i32, box_size: f32) -> *mut std::ffi::c_void;

    /// Libera todos los recursos device y host asociados al handle PM.
    pub fn cuda_pm_destroy(h: *mut std::ffi::c_void);

    /// Ejecuta el solver PM completo (CIC → FFT → Poisson → iFFT → CIC interp).
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

    // ── Solver directo N² (gravedad directa con shared-memory tiling) ─────────

    /// Crea un handle para el solver de gravedad directa N².
    ///
    /// - `eps2`       — suavizado Plummer al cuadrado.
    /// - `block_size` — hilos por bloque (potencia de 2; 0 → usa el default compilado).
    ///
    /// Devuelve `NULL` si no hay dispositivo CUDA disponible.
    pub fn cuda_direct_create(eps2: f32, block_size: i32) -> *mut std::ffi::c_void;

    /// Libera todos los recursos asociados al handle directo.
    pub fn cuda_direct_destroy(h: *mut std::ffi::c_void);

    /// Calcula aceleraciones gravitacionales directas O(N²) para `n` partículas.
    ///
    /// # Parámetros
    /// - `h`        — handle creado con `cuda_direct_create`
    /// - `x,y,z`   — posiciones (longitud `n`)
    /// - `mass`     — masas (longitud `n`)
    /// - `ax,ay,az` — aceleraciones de salida (longitud `n`)
    /// - `n`        — número de partículas
    /// - `g`        — constante gravitatoria
    ///
    /// # Retorna
    /// `0` si OK, código de error CUDA en caso contrario.
    pub fn cuda_direct_solve(
        h: *mut std::ffi::c_void,
        x: *const f32,
        y: *const f32,
        z: *const f32,
        mass: *const f32,
        ax: *mut f32,
        ay: *mut f32,
        az: *mut f32,
        n: i32,
        g: f32,
    ) -> i32;
}
