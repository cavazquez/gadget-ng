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
        r_split: f32,
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

    // ── Kernels SPH O(N²) ───────────────────────────────────────────────────

    pub fn cuda_sph_density(
        x: *const f32,
        y: *const f32,
        z: *const f32,
        mass: *const f32,
        is_gas: *const u8,
        u: *const f32,
        h_in: *const f32,
        h_out: *mut f32,
        rho_out: *mut f32,
        pressure_out: *mut f32,
        entropy_out: *mut f32,
        n: i32,
        periodic_box: f32,
    ) -> i32;

    pub fn cuda_sph_balsara(
        x: *const f32,
        y: *const f32,
        z: *const f32,
        vx: *const f32,
        vy: *const f32,
        vz: *const f32,
        mass: *const f32,
        is_gas: *const u8,
        rho: *const f32,
        pressure: *const f32,
        h_sml: *const f32,
        balsara_out: *mut f32,
        n: i32,
        periodic_box: f32,
    ) -> i32;

    pub fn cuda_sph_forces(
        x: *const f32,
        y: *const f32,
        z: *const f32,
        vx: *const f32,
        vy: *const f32,
        vz: *const f32,
        mass: *const f32,
        is_gas: *const u8,
        rho: *const f32,
        pressure: *const f32,
        h_sml: *const f32,
        ax_out: *mut f32,
        ay_out: *mut f32,
        az_out: *mut f32,
        du_dt_out: *mut f32,
        n: i32,
        periodic_box: f32,
    ) -> i32;

    pub fn cuda_sph_gadget2_forces(
        x: *const f32,
        y: *const f32,
        z: *const f32,
        vx: *const f32,
        vy: *const f32,
        vz: *const f32,
        mass: *const f32,
        is_gas: *const u8,
        rho: *const f32,
        pressure: *const f32,
        h_sml: *const f32,
        balsara: *const f32,
        ax_out: *mut f32,
        ay_out: *mut f32,
        az_out: *mut f32,
        da_dt_out: *mut f32,
        du_dt_out: *mut f32,
        max_vsig_out: *mut f32,
        n: i32,
        periodic_box: f32,
    ) -> i32;

    // ── Kernels MHD ─────────────────────────────────────────────────────────

    pub fn cuda_mhd_flux_freeze(
        ptype: *const u8,
        mass: *const f32,
        internal_energy: *const f32,
        h_sml: *const f32,
        bx_in: *const f32,
        by_in: *const f32,
        bz_in: *const f32,
        bx_out: *mut f32,
        by_out: *mut f32,
        bz_out: *mut f32,
        n: i32,
        gamma: f32,
        beta_freeze: f32,
        rho_ref: f32,
    ) -> i32;

    pub fn cuda_mhd_density_contrib(
        ptype: *const u8,
        mass: *const f32,
        h_sml: *const f32,
        rho_out: *mut f32,
        count_out: *mut f32,
        n: i32,
    ) -> i32;

    pub fn cuda_mhd_b_stats_contrib(
        ptype: *const u8,
        mass: *const f32,
        bx: *const f32,
        by: *const f32,
        bz: *const f32,
        m_out: *mut f32,
        mb_out: *mut f32,
        mb2_out: *mut f32,
        bmag_out: *mut f32,
        emag_out: *mut f32,
        count_out: *mut f32,
        n: i32,
    ) -> i32;

    pub fn cuda_mhd_induction_resistivity(
        ptype: *const u8,
        x: *const f32,
        y: *const f32,
        z: *const f32,
        vx: *const f32,
        vy: *const f32,
        vz: *const f32,
        mass: *const f32,
        rho: *const f32,
        h_sml: *const f32,
        bx_in: *const f32,
        by_in: *const f32,
        bz_in: *const f32,
        bx_out: *mut f32,
        by_out: *mut f32,
        bz_out: *mut f32,
        n: i32,
        dt: f32,
        resistivity: f32,
        periodic_box: f32,
    ) -> i32;

    pub fn cuda_mhd_magnetic_forces(
        ptype: *const u8,
        x: *const f32,
        y: *const f32,
        z: *const f32,
        mass: *const f32,
        rho: *const f32,
        h_sml: *const f32,
        bx: *const f32,
        by: *const f32,
        bz: *const f32,
        ax_out: *mut f32,
        ay_out: *mut f32,
        az_out: *mut f32,
        n: i32,
        mu0: f32,
        periodic_box: f32,
    ) -> i32;

    pub fn cuda_mhd_dedner_cleaning(
        ptype: *const u8,
        div_b: *const f32,
        psi_in: *const f32,
        bx_in: *const f32,
        by_in: *const f32,
        bz_in: *const f32,
        psi_out: *mut f32,
        bx_out: *mut f32,
        by_out: *mut f32,
        bz_out: *mut f32,
        n: i32,
        dt: f32,
        ch: f32,
        cr: f32,
    ) -> i32;

    pub fn cuda_mhd_scalar_diffusion(
        ptype: *const u8,
        scalar_in: *const f32,
        bx: *const f32,
        by: *const f32,
        bz: *const f32,
        scalar_out: *mut f32,
        n: i32,
        dt: f32,
        kappa_par: f32,
        kappa_perp: f32,
    ) -> i32;

    pub fn cuda_mhd_braginskii_viscosity(
        ptype: *const u8,
        vx_in: *const f32,
        vy_in: *const f32,
        vz_in: *const f32,
        bx: *const f32,
        by: *const f32,
        bz: *const f32,
        vx_out: *mut f32,
        vy_out: *mut f32,
        vz_out: *mut f32,
        n: i32,
        dt: f32,
        eta: f32,
    ) -> i32;

    pub fn cuda_mhd_reconnection_streaming_dynamo(
        ptype: *const u8,
        cr_in: *const f32,
        bx_in: *const f32,
        by_in: *const f32,
        bz_in: *const f32,
        u_in: *const f32,
        cr_out: *mut f32,
        bx_out: *mut f32,
        by_out: *mut f32,
        bz_out: *mut f32,
        u_out: *mut f32,
        n: i32,
        dt: f32,
        stream_coeff: f32,
        reconnection_frac: f32,
        dynamo_alpha: f32,
    ) -> i32;

    // ── Kernels Tree / SIDM ─────────────────────────────────────────────────

    pub fn cuda_tree_walk_monopole(
        x: *const f32,
        y: *const f32,
        z: *const f32,
        mass: *const f32,
        ax_out: *mut f32,
        ay_out: *mut f32,
        az_out: *mut f32,
        n: i32,
        g: f32,
        eps2: f32,
    ) -> i32;

    pub fn cuda_tree_sidm_scatter(
        ptype: *const u8,
        x: *const f32,
        y: *const f32,
        z: *const f32,
        vx_in: *const f32,
        vy_in: *const f32,
        vz_in: *const f32,
        mass: *const f32,
        vx_out: *mut f32,
        vy_out: *mut f32,
        vz_out: *mut f32,
        n: i32,
        dt: f32,
        sigma_over_m: f32,
        h: f32,
    ) -> i32;

    // ── Kernels RT ──────────────────────────────────────────────────────────

    pub fn cuda_rt_energy_xi_photoion(
        energy: *const f32,
        flux_x: *const f32,
        flux_y: *const f32,
        flux_z: *const f32,
        energy_contrib_out: *mut f32,
        xi_out: *mut f32,
        gamma_out: *mut f32,
        n: i32,
        dv: f32,
        c_red_code: f32,
        c_red_cgs: f32,
    ) -> i32;

    pub fn cuda_rt_photoheating(
        ptype: *const u8,
        px: *const f32,
        py: *const f32,
        pz: *const f32,
        internal_energy_in: *const f32,
        gamma_hi: *const f32,
        internal_energy_out: *mut f32,
        n_particles: i32,
        nx: i32,
        ny: i32,
        nz: i32,
        box_size: f32,
        dt: f32,
    ) -> i32;

    // ── Kernels Cooling ─────────────────────────────────────────────────────

    pub fn cuda_cooling_apply(
        ptype: *const u8,
        mass: *const f32,
        smoothing_length: *const f32,
        internal_energy: *mut f32,
        metallicity: *const f32,
        bx: *const f32,
        by: *const f32,
        bz: *const f32,
        n: i32,
        dt: f32,
        gamma: f32,
        t_floor_k: f32,
        redshift: f32,
        cooling_kind: i32,
        f_mag: f32,
        reionization_redshift: f32,
        uv_background_model: i32,
        self_shielding_nh: f32,
    ) -> i32;

    // ── Kernels Dust ────────────────────────────────────────────────────────

    pub fn cuda_dust_update(
        ptype: *const u8,
        mass: *const f32,
        smoothing_length: *const f32,
        internal_energy: *const f32,
        dust_to_gas: *mut f32,
        metallicity: *const f32,
        n: i32,
        gamma: f32,
        dt: f32,
        d_to_g_max: f32,
        tau_grow: f32,
        t_destroy_k: f32,
    ) -> i32;

    pub fn cuda_dust_radiation_pressure(
        ptype: *const u8,
        mass: *const f32,
        smoothing_length: *const f32,
        dust_to_gas: *const f32,
        vx: *mut f32,
        vy: *mut f32,
        vz: *mut f32,
        pos_z: *mut f32,
        n: i32,
        dt: f32,
        z_reference: f32,
        kappa: f32,
        j_uv: f32,
        box_size: f32,
    ) -> i32;

    // ── Kernels Molecular Gas ───────────────────────────────────────────────

    pub fn cuda_h2_update(
        ptype: *const u8,
        mass: *const f32,
        smoothing_length: *const f32,
        h2_fraction: *mut f32,
        dust_to_gas: *const f32,
        n: i32,
        dt: f32,
        rho_h2_threshold: f32,
        t_dissoc: f32,
        dust_enabled: i32,
        h2_shielding_boost: f32,
        kappa_dust_uv: f32,
        kappa_silicate_uv: f32,
        kappa_graphite_uv: f32,
        silicate_fraction: f32,
        graphite_fraction: f32,
        species_model: i32,
    ) -> i32;
}
