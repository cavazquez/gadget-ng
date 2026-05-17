//! Bindings FFI a las funciones C expuestas por los kernels CUDA.
//!
//! Solo se compilan cuando CUDA está disponible (es decir, cuando NO se emite
//! el cfg `cuda_unavailable` desde build.rs).

#[cfg(not(cuda_unavailable))]
unsafe extern "C" {
    // ── Device buffer pool (buffers persistentes entre pasos) ─────────────────

    /// Crea un pool vacío con capacidad inicial `initial_n` (0 = sin pre-asignar).
    pub fn cuda_pool_create(initial_n: i32) -> *mut std::ffi::c_void;

    /// Libera todos los buffers device y el pool.
    pub fn cuda_pool_destroy(pool: *mut std::ffi::c_void);

    /// Asegura capacidad para `n` partículas en todos los slots del pool.
    /// Redimensiona con doble capacidad si es necesario. Devuelve 0 si OK.
    pub fn cuda_pool_ensure(pool: *mut std::ffi::c_void, n: i32) -> i32;

    /// Resetea el pool: marca todos los slots como reutilizables (no libera memoria).
    pub fn cuda_pool_reset(pool: *mut std::ffi::c_void);

    /// Sube datos f32 al device en el slot `slot_index`. Devuelve puntero device.
    pub fn cuda_pool_upload_f32(
        pool: *mut std::ffi::c_void,
        slot_index: i32,
        host_data: *const f32,
        n: i32,
    ) -> *mut f32;

    /// Sube datos u8 al device en el slot `slot_index`. Devuelve puntero device.
    pub fn cuda_pool_upload_u8(
        pool: *mut std::ffi::c_void,
        slot_index: i32,
        host_data: *const u8,
        n: i32,
    ) -> *mut u8;

    /// Aloca un buffer f32 de salida en el pool (cero-inicializado).
    pub fn cuda_pool_alloc_f32(pool: *mut std::ffi::c_void, slot_index: i32, n: i32) -> *mut f32;

    /// Descarga datos f32 del device al host. `n` = número de elementos.
    pub fn cuda_pool_download_f32(
        pool: *mut std::ffi::c_void,
        host_data: *mut f32,
        device_data: *const f32,
        n: i32,
    ) -> i32;

    /// Devuelve la capacidad actual del pool (en partículas).
    pub fn cuda_pool_capacity(pool: *mut std::ffi::c_void) -> i32;

    /// Devuelve el número de slots alojados en el pool.
    pub fn cuda_pool_num_slots(pool: *mut std::ffi::c_void) -> i32;

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

    /// Calcula el campo de screening chameleon f(R) por celda + suavizado Jacobi (AP-20).
    pub fn cuda_fr_screening_field(
        density: *const f32,
        screen_out: *mut f32,
        nm: i32,
        f_r0: f32,
        n_fr: f32,
        smoothing: f32,
        iterations: i32,
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

    /// Aceleración gravitacional de N partículas a partir de N_nodes LET
    /// pre-seleccionados (monopolo + cuadrupolo + octupolo; hexadecapolo excluido).
    ///
    /// # Parámetros
    /// - `px/py/pz`        — posiciones de las partículas (longitud `n_particles`)
    /// - `cx/cy/cz`        — centros de masa de los nodos LET (longitud `n_nodes`)
    /// - `node_mass`       — masas de los nodos (longitud `n_nodes`)
    /// - `q0..q5`          — componentes cuadrupolares qxx,qxy,qxz,qyy,qyz,qzz
    /// - `o0..o6`          — componentes octopolares o_xxx..o_yzz (7 términos STF)
    /// - `ax/ay/az_out`    — aceleraciones de salida (longitud `n_particles`)
    /// - `g`               — constante gravitatoria
    /// - `eps2`            — suavizado Plummer al cuadrado
    pub fn cuda_tree_let_accel(
        px: *const f32,
        py: *const f32,
        pz: *const f32,
        n_particles: i32,
        cx: *const f32,
        cy: *const f32,
        cz: *const f32,
        node_mass: *const f32,
        q0: *const f32,
        q1: *const f32,
        q2: *const f32,
        q3: *const f32,
        q4: *const f32,
        q5: *const f32,
        o0: *const f32,
        o1: *const f32,
        o2: *const f32,
        o3: *const f32,
        o4: *const f32,
        o5: *const f32,
        o6: *const f32,
        n_nodes: i32,
        g: f32,
        eps2: f32,
        ax_out: *mut f32,
        ay_out: *mut f32,
        az_out: *mut f32,
    ) -> i32;

    /// Barnes-Hut local GPU walk monopole con pila por hilo (AP-20).
    pub fn cuda_bh_walk_monopole(
        nodes_raw: *const std::ffi::c_void,
        n_nodes: i32,
        root_idx: u32,
        qx: *const f32,
        qy: *const f32,
        qz: *const f32,
        target_idx: *const u32,
        ax_out: *mut f32,
        ay_out: *mut f32,
        az_out: *mut f32,
        n_targets: i32,
        theta2: f32,
        g: f32,
        eps2: f32,
    ) -> i32;

    /// TreePM short-range O(N²) con erfc y mínima imagen (AP-20).
    pub fn cuda_treepm_short_range(
        x: *const f32,
        y: *const f32,
        z: *const f32,
        mass: *const f32,
        ax_out: *mut f32,
        ay_out: *mut f32,
        az_out: *mut f32,
        n: i32,
        r_split: f32,
        r_cut2: f32,
        eps2: f32,
        g: f32,
        box_size: f32,
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

    /// Un sub-paso del solver M1 HLL completo.
    /// Los arrays host son modificados in-place (entrada y salida).
    pub fn cuda_rt_m1_substep(
        e_host: *mut f32,
        fx_host: *mut f32,
        fy_host: *mut f32,
        fz_host: *mut f32,
        nx: i32,
        ny: i32,
        nz: i32,
        dx: f32,
        dt_sub: f32,
        c_red: f32,
        kappa: f32,
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

    // ── Kernels Analysis ───────────────────────────────────────────────────

    /// Calcula el momento angular L = Σ m_i (r_i - r_com) × (v_i - v_com).
    pub fn cuda_analysis_halo_spin(
        x: *const f32,
        y: *const f32,
        z: *const f32,
        vx: *const f32,
        vy: *const f32,
        vz: *const f32,
        mass: *const f32,
        n: i32,
        cx: f32,
        cy: f32,
        cz: f32,
        vcx: f32,
        vcy: f32,
        vcz: f32,
        lx_out: *mut f64,
        ly_out: *mut f64,
        lz_out: *mut f64,
    ) -> i32;

    /// Calcula luminosidad estelar total + colores (B-V, g-r) vía SSP BC03-lite.
    pub fn cuda_analysis_luminosity(
        ptype: *const u8,
        mass: *const f32,
        age_gyr: *const f32,
        metallicity: *const f32,
        n: i32,
        l_total_out: *mut f64,
        bv_weighted_out: *mut f64,
        gr_weighted_out: *mut f64,
        n_stars_out: *mut i32,
    ) -> i32;

    /// Calcula la luminosidad de rayos X (bremsstrahlung) total.
    pub fn cuda_analysis_xray(
        ptype: *const u8,
        mass: *const f32,
        h_sml: *const f32,
        internal_energy: *const f32,
        n: i32,
        gamma: f32,
        lx_out: *mut f64,
    ) -> i32;

    // ── Kernels RT chemistry rates ────────────────────────────────────────────

    /// Tasa de fotoionización Γ_HI por partícula (NGP lookup en campo E).
    pub fn cuda_rt_chemistry_rates(
        ptype: *const u8,
        px: *const f32,
        py: *const f32,
        pz: *const f32,
        energy_density: *const f32,
        gamma_hi_out: *mut f32,
        n_particles: i32,
        n_cells: i32,
        nx: i32,
        ny: i32,
        nz: i32,
        box_size: f32,
        c_red_cgs: f32,
    ) -> i32;

    /// Aplica cooling_rate_approx a la energía interna de partículas gas.
    pub fn cuda_rt_cooling_apply(
        ptype: *const u8,
        u_inout: *mut f32,
        x_e: *const f32,
        n: i32,
        gamma_eos: f32,
        n_h_ref: f32,
        dt: f32,
    ) -> i32;

    // ── Kernels RT chemistry stiff solver ─────────────────────────────────────

    /// Solver subcíclico implícito de red química (12 especies) por partícula.
    pub fn cuda_rt_chemistry_stiff(
        ptype: *const u8,
        x_hi: *mut f32,
        x_hii: *mut f32,
        x_hei: *mut f32,
        x_heii: *mut f32,
        x_heiii: *mut f32,
        x_e: *mut f32,
        x_hm: *mut f32,
        x_h2: *mut f32,
        x_h2p: *mut f32,
        x_d: *mut f32,
        x_dp: *mut f32,
        x_hd: *mut f32,
        gamma_hi: *const f32,
        temperature: *const f32,
        n: i32,
        dt: f32,
        n_h_ref: f32,
    ) -> i32;

    // ── Kernels RT reionization stats ─────────────────────────────────────────

    /// Reducción paralela: suma x_hii, suma x_hii², cuenta ionizados (>0.5).
    pub fn cuda_rt_reionization_stats(
        x_hii: *const f32,
        n: i32,
        sum_xhii_out: *mut f64,
        sum_sq_out: *mut f64,
        ionized_count_out: *mut i32,
    ) -> i32;

    /// Map: δT_b = 27 × x_HI × overdensity × sqrt((1+z)/10)  \[mK\].
    pub fn cuda_rt_cm21_field(
        x_hii: *const f32,
        overdensity: *const f32,
        z: f32,
        delta_tb_out: *mut f32,
        n: i32,
    ) -> i32;

    // ── Kernel RT IGM temperature (AP-16) ────────────────────────────────────

    /// Reducción: suma_T, suma_T², count para partículas gas IGM.
    /// Usa los campos almacenados en ChemState (misma fórmula que CPU).
    /// Salida: t_mean_out, t_sigma_out, n_igm_out.
    pub fn cuda_rt_igm_temp(
        ptype: *const u8,
        u: *const f32,
        h_sml: *const f32,
        mass: *const f32,
        x_hi: *const f32,
        x_hii: *const f32,
        x_e: *const f32,
        x_d: *const f32,
        x_hei: *const f32,
        x_heii: *const f32,
        x_heiii: *const f32,
        n: i32,
        gamma: f32,
        delta_max: f32,
        mean_density: f32,
        t_mean_out: *mut f64,
        t_sigma_out: *mut f64,
        n_igm_out: *mut i32,
    ) -> i32;

    /// Variante full: además devuelve array compacto de temperaturas IGM para percentiles.
    /// `temps_out` debe apuntar a un buffer de tamaño `n` floats (host).
    pub fn cuda_rt_igm_temp_full(
        ptype: *const u8,
        u: *const f32,
        h_sml: *const f32,
        mass: *const f32,
        x_hi: *const f32,
        x_hii: *const f32,
        x_e: *const f32,
        x_d: *const f32,
        x_hei: *const f32,
        x_heii: *const f32,
        x_heiii: *const f32,
        n: i32,
        gamma: f32,
        delta_max: f32,
        mean_density: f32,
        t_mean_out: *mut f64,
        t_sigma_out: *mut f64,
        n_igm_out: *mut i32,
        temps_out: *mut f32,
    ) -> i32;

    // ── Kernels MHD ambipolar + two-fluid (AP-16) ─────────────────────────────

    /// Hall drift (AP-20): rota B alrededor de v×B con fórmula de Rodrigues; conserva |B|.
    pub fn cuda_mhd_hall_drift(
        ptype: *const u8,
        bx_in: *const f32,
        by_in: *const f32,
        bz_in: *const f32,
        vx: *const f32,
        vy: *const f32,
        vz: *const f32,
        mass: *const f32,
        h_sml: *const f32,
        bx_out: *mut f32,
        by_out: *mut f32,
        bz_out: *mut f32,
        n: i32,
        eta_hall: f32,
        dt: f32,
    ) -> i32;

    /// Difusión ambipolar: amortigua B con rate eta_ad / x_ion; calienta u.
    pub fn cuda_mhd_ambipolar(
        ptype: *const u8,
        bx_in: *const f32,
        by_in: *const f32,
        bz_in: *const f32,
        u_in: *const f32,
        mass: *const f32,
        dust_to_gas: *const f32,
        bx_out: *mut f32,
        by_out: *mut f32,
        bz_out: *mut f32,
        u_out: *mut f32,
        n: i32,
        eta_ad: f32,
        ion_floor: f32,
        dust_coupling: f32,
        heat_eff: f32,
        dt: f32,
    ) -> i32;

    /// Difusión óhmica resistiva (Phase 187): dB/dt = −eta_Ohm B / h².
    pub fn cuda_mhd_ohmic(
        ptype: *const u8,
        bx_in: *const f32,
        by_in: *const f32,
        bz_in: *const f32,
        h_sml: *const f32,
        mass: *const f32,
        bx_out: *mut f32,
        by_out: *mut f32,
        bz_out: *mut f32,
        u_out: *mut f32,
        n: i32,
        eta_ohm: f32,
        heat_eff: f32,
        dt: f32,
    ) -> i32;

    /// Acoplamiento Coulomb e-i: actualiza t_electron hacia T_ion.
    pub fn cuda_mhd_two_fluid(
        ptype: *const u8,
        u_in: *const f32,
        h_sml: *const f32,
        mass: *const f32,
        te_in: *const f32,
        te_out: *mut f32,
        n: i32,
        nu_ei_coeff: f32,
        dt: f32,
    ) -> i32;

    // ── Kernels MHD CR streaming / backreaction ───────────────────────────────

    /// CR streaming O(N²): actualiza cr_energy con pérdidas compresional + streaming.
    pub fn cuda_mhd_cr_streaming(
        ptype: *const u8,
        px: *const f32,
        py: *const f32,
        pz: *const f32,
        vx: *const f32,
        vy: *const f32,
        vz: *const f32,
        mass: *const f32,
        h_sml: *const f32,
        cr_energy_in: *const f32,
        bx: *const f32,
        by: *const f32,
        bz: *const f32,
        cr_energy_out: *mut f32,
        n: i32,
        dt: f32,
        streaming_coeff: f32,
        periodic_box: f32,
    ) -> i32;

    /// CR backreaction O(N²): aceleración gas desde gradiente de presión CR.
    pub fn cuda_mhd_cr_backreaction(
        ptype: *const u8,
        px: *const f32,
        py: *const f32,
        pz: *const f32,
        mass: *const f32,
        h_sml: *const f32,
        cr_energy: *const f32,
        ax_out: *mut f32,
        ay_out: *mut f32,
        az_out: *mut f32,
        n: i32,
        periodic_box: f32,
    ) -> i32;

    // ── Conducción anisótropa / CR diffusion O(N²) (AP-17) ───────────────────

    /// Conducción térmica anisótropa pairwise (Wendland-C6, kappa_par/kappa_perp).
    pub fn cuda_mhd_anisotropic_conduction(
        ptype: *const u8,
        px: *const f32,
        py: *const f32,
        pz: *const f32,
        mass: *const f32,
        h_sml: *const f32,
        u_in: *const f32,
        bx: *const f32,
        by: *const f32,
        bz: *const f32,
        u_out: *mut f32,
        n: i32,
        kappa_par: f32,
        kappa_perp: f32,
        gamma: f32,
        dt: f32,
        periodic_box: f32,
    ) -> i32;

    /// CR diffusion anisótropa pairwise (Wendland-C6, kappa_cr).
    pub fn cuda_mhd_cr_diffusion_anisotropic(
        ptype: *const u8,
        px: *const f32,
        py: *const f32,
        pz: *const f32,
        mass: *const f32,
        h_sml: *const f32,
        cr_energy_in: *const f32,
        bx: *const f32,
        by: *const f32,
        bz: *const f32,
        cr_energy_out: *mut f32,
        n: i32,
        kappa_cr: f32,
        dt: f32,
        periodic_box: f32,
    ) -> i32;
}
