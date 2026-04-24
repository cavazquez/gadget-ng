//! Bindings FFI a las funciones C expuestas por `hip/pm_gravity.hip` y
//! `hip/direct_gravity.hip`.
//!
//! Solo se compilan cuando HIP está disponible (es decir, cuando NO se emite
//! el cfg `hip_unavailable` desde build.rs).

#[cfg(not(hip_unavailable))]
extern "C" {
    // ── Solver PM (CIC + rocFFT + Poisson) ───────────────────────────────────

    /// Crea un handle PM HIP con grilla `grid_size³` y caja periódica `box_size`.
    ///
    /// Devuelve `NULL` si la inicialización falla (sin dispositivo HIP/ROCm, …).
    pub fn hip_pm_create(grid_size: i32, box_size: f32) -> *mut std::ffi::c_void;

    /// Libera todos los recursos device y host asociados al handle PM.
    pub fn hip_pm_destroy(h: *mut std::ffi::c_void);

    /// Ejecuta el solver PM completo (CIC → rocFFT → Poisson → irocFFT → CIC interp).
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

    // ── Solver directo N² (LDS tiling) ───────────────────────────────────────

    /// Crea un handle para el solver de gravedad directa N² HIP.
    ///
    /// - `eps2`       — suavizado Plummer al cuadrado.
    /// - `block_size` — hilos por workgroup (potencia de 2; 0 → usa el default compilado).
    ///
    /// Devuelve `NULL` si no hay dispositivo HIP/ROCm disponible.
    pub fn hip_direct_create(eps2: f32, block_size: i32) -> *mut std::ffi::c_void;

    /// Libera todos los recursos asociados al handle directo HIP.
    pub fn hip_direct_destroy(h: *mut std::ffi::c_void);

    /// Calcula aceleraciones gravitacionales directas O(N²) para `n` partículas.
    ///
    /// # Retorna
    /// `0` si OK, código de error HIP en caso contrario.
    pub fn hip_direct_solve(
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
