//! Integración cosmológica: factor de escala `a(t)` y factores drift/kick.
//!
//! ## Formulación de momentum canónico (estilo GADGET-4)
//!
//! Se almacena `p = a² dx_c/dt` en `particle.velocity`, donde `x_c` son
//! coordenadas comóviles. En esta formulación:
//!
//! - **Drift** (posición): `Δx_c = p · D`,  `D = ∫_t^{t+dt} dt'/a²(t')`
//! - **Kick** (momentum): `Δp = F · K`,  `K = ∫_t^{t+Δt} dt'/a(t')`
//!
//! No aparece término de arrastre de Hubble explícito: queda absorbido en
//! las variables canónicas.
//!
//! ## Modelo cosmológico
//!
//! ΛCDM plano (sin radiación, sin curvatura):
//! ```text
//! da/dt = a · H₀ · √(Ω_m · a⁻³ + Ω_Λ)
//! ```
//! `H₀` se expresa en **unidades internas de tiempo** (1/t_sim).

/// Parámetros cosmológicos para ΛCDM plano o energía oscura dinámica w(z) CPL.
///
/// `h0` es H₀ en unidades internas de tiempo (1/t_sim), no el parámetro
/// adimensional h₁₀₀ = H₀/(100 km/s/Mpc). El usuario es responsable de
/// la consistencia de unidades con el resto de la simulación.
///
/// ## Energía oscura dinámica (Phase 155)
///
/// El parámetro de ecuación de estado CPL (Chevallier-Polarski-Linder) es:
/// ```text
/// w(a) = w0 + wa × (1 − a)
/// ```
/// Con `w0 = -1, wa = 0` se recupera ΛCDM exactamente.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CosmologyParams {
    /// Fracción de materia: Ω_m (sin dimensiones).
    pub omega_m: f64,
    /// Constante cosmológica: Ω_Λ (sin dimensiones).
    pub omega_lambda: f64,
    /// H₀ en unidades internas de tiempo (1/t_sim).
    pub h0: f64,
    /// Parámetro CPL w₀ (Phase 155). Default: -1.0 (ΛCDM).
    pub w0: f64,
    /// Parámetro CPL wₐ (Phase 155). Default: 0.0 (ΛCDM).
    pub wa: f64,
    /// Ω_ν de neutrinos masivos (Phase 156). Default: 0.0.
    pub omega_nu: f64,
}

/// Retorna el parámetro de ecuación de estado w(a) para el modelo CPL (Phase 155).
///
/// `w(a) = w0 + wa × (1 − a)`
///
/// Para ΛCDM: `w0 = -1, wa = 0 → w(a) = -1`.
#[inline]
pub fn dark_energy_eos(a: f64, w0: f64, wa: f64) -> f64 {
    w0 + wa * (1.0 - a)
}

impl CosmologyParams {
    /// Crea nuevos parámetros cosmológicos ΛCDM (retrocompatible).
    pub fn new(omega_m: f64, omega_lambda: f64, h0: f64) -> Self {
        Self {
            omega_m,
            omega_lambda,
            h0,
            w0: -1.0,
            wa: 0.0,
            omega_nu: 0.0,
        }
    }

    /// Crea parámetros con energía oscura dinámica CPL (Phase 155).
    pub fn new_cpl(omega_m: f64, omega_lambda: f64, h0: f64, w0: f64, wa: f64) -> Self {
        Self { omega_m, omega_lambda, h0, w0, wa, omega_nu: 0.0 }
    }

    /// Crea parámetros con neutrinos masivos (Phase 156).
    pub fn new_with_nu(omega_m: f64, omega_lambda: f64, h0: f64, omega_nu: f64) -> Self {
        Self { omega_m, omega_lambda, h0, w0: -1.0, wa: 0.0, omega_nu }
    }

    /// `da/dt` con energía oscura dinámica w(a) CPL y neutrinos masivos (Phase 155/156).
    ///
    /// Ecuación de Friedmann generalizada:
    /// ```text
    /// H²(a) = H₀² [ Ω_m/a³ + Ω_ν/a³ + Ω_DE(a) ]
    /// Ω_DE(a) = Ω_Λ × a^{-3(1+w0+wa)} × exp(3·wa·(a-1))
    /// ```
    #[inline]
    fn da_dt(self, a: f64) -> f64 {
        let omega_de = if (self.w0 + 1.0).abs() < 1e-10 && self.wa.abs() < 1e-10 {
            // ΛCDM: Ω_DE = Ω_Λ (constante)
            self.omega_lambda
        } else {
            // CPL: Ω_DE(a) = Ω_Λ × a^{-3(1+w0+wa)} × exp(3·wa·(a-1))
            let exp_factor = (3.0 * self.wa * (a - 1.0)).exp();
            let power = -3.0 * (1.0 + self.w0 + self.wa);
            self.omega_lambda * a.powf(power) * exp_factor
        };
        let h_sq = (self.omega_m + self.omega_nu) / (a * a * a) + omega_de;
        a * self.h0 * h_sq.max(0.0).sqrt()
    }

    /// Avanza el factor de escala `a` un intervalo `dt` usando RK4.
    ///
    /// Es el bloque básico de integración; se llama desde `drift_kick_factors`
    /// y `hierarchical_prefixes`.
    pub fn advance_a(self, a: f64, dt: f64) -> f64 {
        let k1 = self.da_dt(a);
        let k2 = self.da_dt(a + 0.5 * dt * k1);
        let k3 = self.da_dt(a + 0.5 * dt * k2);
        let k4 = self.da_dt(a + dt * k3);
        a + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    }

    /// Calcula los factores de drift y kick cosmológicos para un paso `dt`.
    ///
    /// Devuelve `(drift, kick_half, kick_half2)`:
    /// - `drift      = ∫_{t}^{t+dt}   dt'/a²(t')`
    /// - `kick_half  = ∫_{t}^{t+dt/2} dt'/a(t')`
    /// - `kick_half2 = ∫_{t+dt/2}^{t+dt} dt'/a(t')`
    ///
    /// Usa 16 sub-pasos con regla de Simpson por sub-paso (error O(dt⁵) por sub-paso).
    pub fn drift_kick_factors(self, a0: f64, dt: f64) -> (f64, f64, f64) {
        const N_SUB: u32 = 16;
        let sub_dt = dt / N_SUB as f64;
        let mut a = a0;
        let mut drift = 0.0;
        let mut kick_half = 0.0;
        let mut kick_half2 = 0.0;

        for sub in 0..N_SUB {
            let a_left = a;
            let a_mid = self.advance_a(a, sub_dt * 0.5);
            let a_right = self.advance_a(a, sub_dt);

            let d_sub =
                sub_dt / 6.0 * (1.0 / a_left.powi(2) + 4.0 / a_mid.powi(2) + 1.0 / a_right.powi(2));
            drift += d_sub;

            let k_sub = sub_dt / 6.0 * (1.0 / a_left + 4.0 / a_mid + 1.0 / a_right);
            if sub < N_SUB / 2 {
                kick_half += k_sub;
            } else {
                kick_half2 += k_sub;
            }

            a = a_right;
        }

        (drift, kick_half, kick_half2)
    }

    /// Pre-computa factores de drift/kick para el integrador jerárquico.
    ///
    /// Divide cada sub-paso fino `fine_dt` en dos medios sub-pasos (`half_dt = fine_dt/2`).
    /// Construye sumas prefijas sobre los `2·n_fine` medios sub-pasos usando la regla
    /// trapezoidal, lo que permite calcular el factor de kick o drift sobre cualquier
    /// intervalo de medios sub-pasos en O(1).
    ///
    /// Devuelve `(a_arr, kick_prefix, drift_prefix)`, cada uno de longitud `2·n_fine + 1`.
    ///
    /// # Uso en `hierarchical_kdk_step`
    ///
    /// Para finos sub-pasos indexados por `s` (0 …< n_fine):
    /// - **Drift** de sub-paso `s`:
    ///   `drift_prefix[2s+2] − drift_prefix[2s]`
    /// - **Half-kick START** para nivel `lvl` (stride = `2^(max_level−lvl)` pasos finos)
    ///   empezando en sub-paso `s`:
    ///   `kick_prefix[2s + 2^(max_level−lvl)] − kick_prefix[2s]`
    /// - **Half-kick END** para nivel `lvl` terminando en sub-paso `s+1`:
    ///   `kick_prefix[2(s+1)] − kick_prefix[2(s+1) − 2^(max_level−lvl)]`
    pub fn hierarchical_prefixes(
        self,
        a0: f64,
        fine_dt: f64,
        n_fine: usize,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let half_dt = fine_dt * 0.5;
        let n_half = 2 * n_fine;

        let mut a_arr = vec![0.0_f64; n_half + 1];
        a_arr[0] = a0;
        for i in 0..n_half {
            a_arr[i + 1] = self.advance_a(a_arr[i], half_dt);
        }

        let mut kick_prefix = vec![0.0_f64; n_half + 1];
        let mut drift_prefix = vec![0.0_f64; n_half + 1];
        for i in 0..n_half {
            let a_l = a_arr[i];
            let a_r = a_arr[i + 1];
            // Regla trapezoidal dentro de cada medio sub-paso.
            kick_prefix[i + 1] = kick_prefix[i] + 0.5 * half_dt * (1.0 / a_l + 1.0 / a_r);
            drift_prefix[i + 1] =
                drift_prefix[i] + 0.5 * half_dt * (1.0 / a_l.powi(2) + 1.0 / a_r.powi(2));
        }

        (a_arr, kick_prefix, drift_prefix)
    }
}

// ── Diagnósticos cosmológicos ─────────────────────────────────────────────────

use crate::particle::Particle;

/// Velocidad peculiar RMS a partir del momentum canónico almacenado en `velocity`.
///
/// `v_peculiar_rms = sqrt(⟨|p/a|²⟩)` donde `p = a² dx_c/dt` → `v_pec = p/a`.
///
/// Devuelve 0.0 si `particles` está vacío.
pub fn peculiar_vrms(particles: &[Particle], a: f64) -> f64 {
    if particles.is_empty() || a <= 0.0 {
        return 0.0;
    }
    let sum: f64 = particles
        .iter()
        .map(|p| {
            let v = p.velocity * (1.0 / a);
            v.dot(v)
        })
        .sum();
    (sum / particles.len() as f64).sqrt()
}

/// Contraste de densidad RMS sobre una malla cúbica de `n_grid³` celdas.
///
/// Divide `[0, box_size]³` en `n_grid³` celdas y cuenta partículas por celda.
/// La densidad media por celda es `n_particles / n_grid³`.
/// Devuelve `sqrt(⟨(δρ/ρ̄)²⟩)` donde `δρ = ρ_celda - ρ̄`.
///
/// Devuelve 0.0 si `particles` está vacío o `n_grid == 0`.
pub fn density_contrast_rms(particles: &[Particle], box_size: f64, n_grid: usize) -> f64 {
    if particles.is_empty() || n_grid == 0 || box_size <= 0.0 {
        return 0.0;
    }
    let ng = n_grid;
    let inv_cell = ng as f64 / box_size;
    let mut counts = vec![0u32; ng * ng * ng];

    for p in particles {
        // Coordenadas periódicas módulo box_size.
        let xi = (p.position.x.rem_euclid(box_size) * inv_cell) as usize;
        let yi = (p.position.y.rem_euclid(box_size) * inv_cell) as usize;
        let zi = (p.position.z.rem_euclid(box_size) * inv_cell) as usize;
        let xi = xi.min(ng - 1);
        let yi = yi.min(ng - 1);
        let zi = zi.min(ng - 1);
        counts[xi * ng * ng + yi * ng + zi] += 1;
    }

    let mean = particles.len() as f64 / (ng * ng * ng) as f64;
    if mean <= 0.0 {
        return 0.0;
    }
    let sum_sq: f64 = counts
        .iter()
        .map(|&c| {
            let delta = (c as f64 - mean) / mean;
            delta * delta
        })
        .sum();
    (sum_sq / (ng * ng * ng) as f64).sqrt()
}

/// **Phase 45 — Acoplamiento gravitacional en la convención canónica QKSL.**
///
/// Devuelve el valor de `g` que debe pasarse al solver gravitacional
/// (`fft_poisson::solve_forces*`, `GravitySolver::accelerations_for_indices`)
/// para que, bajo la convención del integrador `leapfrog_cosmo_kdk_step`
/// (`drift = ∫dt/a²`, `kick = ∫dt/a`, slot = `p = a²·ẋ_c`), la fuerza
/// efectiva aplicada corresponda a `dp/dt = −∇Φ_pec` (GADGET/QKSL canónico).
///
/// ## Derivación
///
/// - Poisson peculiar comóvil:  `∇²_c Φ_pec = 4π·G·ρ̄·δ·a²`.
/// - Solver retorna `-∇ · Φ̂_solver` con `Φ̂_solver = -4π·g_pass·ρ̂_comov/k²`.
/// - Kick efectivo aplica `Δp = F_solver · ∫dt/a`, es decir `dp/dt = F/a`.
/// - Para que `dp/dt · a = -∇Φ_pec`, se requiere `F_solver = a · (-∇Φ_pec)`
///   y por tanto `g_pass = G · a³`.
///
/// El valor histórico `G/a` (pre-Phase 45) introducía un factor `a⁴`
/// erróneo en la fuerza efectiva: con `a = 0.02` esto sobreestimaba
/// `Δp` por ~6·10⁶, causando que `v_rms` pasara de ~10⁻⁹ a ~10² en
/// decenas de pasos y que `δ_rms` saturase ≈ 1 antes de `a ≈ 0.05`.
///
/// Ver `docs/reports/2026-04-phase45-units-audit.md` para la auditoría
/// empírica y la tabla A/B de convenciones.
#[inline]
pub fn gravity_coupling_qksl(g: f64, a: f64) -> f64 {
    g * a * a * a
}

/// **Phase 49 — Timestep adaptativo cosmológico.**
///
/// Calcula el paso de tiempo máximo compatible con la estabilidad numérica
/// para un integrador KDK cosmológico. Usa dos criterios independientes:
///
/// ## Criterio gravitacional (Quinn et al. 1997)
///
/// ```text
/// dt_grav = η · √(ε / |a_max|)
/// ```
///
/// donde `ε` es el softening, `|a_max|` la aceleración máxima de las
/// partículas y `η ≈ 0.025` la fracción de tolerancia. Este criterio limita
/// el timestep para que ninguna partícula "salte" más de `ε` por unidad de
/// velocidad en un paso.
///
/// ## Criterio de Hubble
///
/// ```text
/// dt_hub = α_H / H(a)
/// ```
///
/// donde `H(a)` es el parámetro de Hubble en unidades internas y
/// `α_H ≈ 0.025` la fracción del tiempo de Hubble. Evita que el factor de
/// escala varíe demasiado en un solo paso.
///
/// ## Resultado
///
/// ```text
/// dt = min(dt_grav, dt_hub, dt_max)
/// ```
///
/// Si `acc_max ≤ 0` (sin fuerzas aún), se omite el criterio gravitacional.
///
/// # Parámetros
///
/// - `params`: parámetros cosmológicos ΛCDM.
/// - `a`: factor de escala actual.
/// - `acc_max`: magnitud máxima de aceleración en unidades de código.
/// - `softening`: longitud de suavizado en unidades de código.
/// - `eta_grav`: fracción gravitacional (típicamente 0.025).
/// - `alpha_h`: fracción del tiempo de Hubble (típicamente 0.025).
/// - `dt_max`: límite superior explícito.
pub fn adaptive_dt_cosmo(
    params: CosmologyParams,
    a: f64,
    acc_max: f64,
    softening: f64,
    eta_grav: f64,
    alpha_h: f64,
    dt_max: f64,
) -> f64 {
    let dt_grav = if acc_max > 0.0 && softening > 0.0 {
        eta_grav * (softening / acc_max).sqrt()
    } else {
        dt_max
    };
    let h_a = hubble_param(params, a);
    let dt_hub = if h_a > 0.0 { alpha_h / h_a } else { dt_max };
    dt_grav.min(dt_hub).min(dt_max)
}

/// **Phase 50 — G auto-consistente para caja unitaria (ρ̄ = 1).**
///
/// Calcula la constante gravitacional `G` (en unidades de código) que satisface
/// la ecuación de Friedmann para ΛCDM en una caja unitaria con densidad media
/// de materia `ρ̄_m = 1`.
///
/// ## Condición de consistencia
///
/// La ecuación de Friedmann (componente de materia) a `a = a_init` es:
///
/// ```text
/// H₀² = (8π·G·ρ̄_m) / 3 · Ω_m     (para ρ̄_total = ρ_crit)
/// ```
///
/// Con `ρ̄_m = 1` (total_mass / box_volume en código):
///
/// ```text
/// G_code = 3 · Ω_m · H₀² / (8π)
/// ```
///
/// Esta condición garantiza que la ecuación del crecimiento lineal (Meszaros):
///
/// ```text
/// δ'' + 2H δ' = 4πG ρ̄_m δ = (3/2) Ω_m H₀² / a³ · δ
/// ```
///
/// produce el factor de crecimiento `D(a)` calculado por `growth_factor_d`.
///
/// ## Diagnóstico de inconsistencia
///
/// Si se usa `G = 1.0` con `H₀ = 0.1` y `Ω_m = 0.315`, el ratio efectivo
/// es `(4πG·ρ̄_m)/H₀² = 4π/0.01 ≈ 1257`, mientras que la física correcta
/// exige `(3/2)·Ω_m ≈ 0.47`. El coupling gravitacional está ~2660× sobreestimado
/// (para evoluciones largas) o ~2660× subrepresentado (cuando se usa G·a³
/// con G=1 muy pequeño relativo a H₀²), lo que impide reproducir D²(a).
///
/// ## Ejemplo de uso
///
/// ```rust,ignore
/// use gadget_ng_core::cosmology::{g_code_consistent, gravity_coupling_qksl};
///
/// let omega_m = 0.315;
/// let h0_code = 0.1;   // en unidades internas (1/t_sim)
/// let g_phys = g_code_consistent(omega_m, h0_code);  // ≈ 3.76e-4
///
/// // En el loop de integración:
/// let g_cosmo = gravity_coupling_qksl(g_phys, a);   // g_phys · a³
/// ```
///
/// # Parámetros
///
/// - `omega_m`: fracción de materia Ω_m (sin dimensiones).
/// - `h0`: H₀ en unidades internas (1/t_sim). Debe ser la **misma** H₀ usada
///   en `CosmologyParams` para que la consistencia sea exacta.
pub fn g_code_consistent(omega_m: f64, h0: f64) -> f64 {
    3.0 * omega_m * h0 * h0 / (8.0 * std::f64::consts::PI)
}

/// Verifica la condición de consistencia cosmológica y devuelve el error relativo.
///
/// Comprueba que `G`, `H₀` y `ρ̄_m` satisfacen la ecuación de Friedmann:
///
/// ```text
/// H₀² = 8π·G·ρ̄_m / 3  ·  1/Ω_m
/// ```
///
/// Es decir, verifica `|G - G_consistente|/G_consistente` donde
/// `G_consistente = 3·Ω_m·H₀²/(8π·ρ̄_m)`.
///
/// ## Uso
///
/// ```rust,ignore
/// let err = cosmo_consistency_error(G, OMEGA_M, H0, rho_bar);
/// assert!(err < 1e-10, "Parámetros inconsistentes: {err:.3e}");
/// ```
///
/// # Retorno
///
/// Error relativo `|G - G_expect|/G_expect` en [0, ∞). Un valor < 1e-10
/// indica consistencia numérica exacta.
pub fn cosmo_consistency_error(g: f64, omega_m: f64, h0: f64, rho_bar: f64) -> f64 {
    if rho_bar <= 0.0 || omega_m <= 0.0 || h0 <= 0.0 {
        return f64::INFINITY;
    }
    let g_expect = 3.0 * omega_m * h0 * h0 / (8.0 * std::f64::consts::PI * rho_bar);
    (g - g_expect).abs() / g_expect
}

/// H(a) = H₀ · √(Ω_m·a⁻³ + Ω_Λ) — parámetro de Hubble en unidades internas.
pub fn hubble_param(params: CosmologyParams, a: f64) -> f64 {
    // Incluye contribución de neutrinos masivos (Phase 156) y w(z) CPL (Phase 155)
    let omega_de = if (params.w0 + 1.0).abs() < 1e-10 && params.wa.abs() < 1e-10 {
        params.omega_lambda
    } else {
        let exp_factor = (3.0 * params.wa * (a - 1.0)).exp();
        let power = -3.0 * (1.0 + params.w0 + params.wa);
        params.omega_lambda * a.powf(power) * exp_factor
    };
    let h_sq = (params.omega_m + params.omega_nu) / (a * a * a) + omega_de;
    params.h0 * h_sq.max(0.0).sqrt()
}

/// Calcula Ω_ν a partir de la suma de masas de neutrinos (Phase 156).
///
/// Usa la relación: `Ω_ν = Σm_ν / (93.14 eV × h²)`.
///
/// # Parámetros
/// - `m_nu_ev`: suma de masas de neutrinos en eV (p. ej. 0.06 para normal hierarchy)
/// - `h100`: parámetro de Hubble adimensional h₁₀₀ = H₀/(100 km/s/Mpc)
///
/// # Retorna
/// Ω_ν (sin dimensiones). Para `m_nu_ev = 0.06, h = 0.674`: Ω_ν ≈ 0.00142.
pub fn omega_nu_from_mass(m_nu_ev: f64, h100: f64) -> f64 {
    if m_nu_ev <= 0.0 { return 0.0; }
    m_nu_ev / (93.14 * h100 * h100)
}

/// Supresión del espectro de potencia por neutrinos masivos (Phase 156).
///
/// Aproximación lineal de Lesgourgues & Pastor (2006):
/// `ΔP/P ≈ -8 × f_ν` para k > k_FS.
///
/// # Parámetros
/// - `f_nu`: fracción de la densidad de materia en neutrinos `f_ν = Ω_ν/Ω_m`
///
/// # Retorna
/// Factor multiplicativo `(1 - 8 × f_ν)` clampado a [0, 1].
pub fn neutrino_suppression(f_nu: f64) -> f64 {
    (1.0 - 8.0 * f_nu).max(0.0).min(1.0)
}

/// Tasa de crecimiento lineal `f(a) = d ln D / d ln a`.
///
/// Usa la aproximación de Linder (2005): `f(a) ≈ Ω_m(a)^0.55`, donde
/// `Ω_m(a) = Ω_m·a⁻³ / (H(a)/H₀)²`.
///
/// Para un universo Einstein–de Sitter (Ω_m=1, Ω_Λ=0): `f=1` exacto.
/// Para ΛCDM estándar: `f ≈ Ω_m(a)^0.55` con error < 1% en z < 2.
///
/// ## Uso en Zel'dovich ICs
///
/// El momentum canónico de la 1LPT es `p = a²·f·H(a)·Ψ`, donde `Ψ` es el
/// campo de desplazamiento a la escala `a`. Esta función devuelve `f` para
/// ese cálculo.
pub fn growth_rate_f(params: CosmologyParams, a: f64) -> f64 {
    if a <= 0.0 {
        return 1.0;
    }
    let h = hubble_param(params, a);
    if h <= 0.0 || params.h0 <= 0.0 {
        return 1.0;
    }
    // Ω_m(a) = Ω_m · a⁻³ / (H(a)/H₀)²
    let h_ratio_sq = (h / params.h0) * (h / params.h0);
    let omega_m_a = params.omega_m / (a * a * a) / h_ratio_sq;
    omega_m_a.max(0.0).powf(0.55)
}

// ── Factor de crecimiento lineal D(a) (Fase 37) ──────────────────────────────

/// Aproximación CPT92 (Carroll–Press–Turner 1992) del modo creciente `D(a)`
/// en cosmología ΛCDM plana:
///
/// ```text
/// D(a) = a · g(a)
/// g(a) = (5/2) · Ω_m(a) /
///        [ Ω_m(a)^(4/7) − Ω_Λ(a) + (1 + Ω_m(a)/2)·(1 + Ω_Λ(a)/70) ]
///
/// con    Ω_m(a) = Ω_m / (Ω_m + Ω_Λ·a³)
///        Ω_Λ(a) = Ω_Λ·a³ / (Ω_m + Ω_Λ·a³)
/// ```
///
/// **Normalización**: la función devuelve `D(a)` sin imponer `D(1)=1`.
/// Para obtener el cociente físico `D(a₁)/D(a₂)` usar [`growth_factor_d_ratio`].
///
/// Para Einstein–de Sitter (Ω_m=1, Ω_Λ=0): `g(a)=1` exacto, `D(a)=a`.
///
/// ## Uso
///
/// En ICs cosmológicas, el factor `s = D(a_init)/D(1)` reescala las
/// amplitudes LPT: `Ψ¹ ← s·Ψ¹`, `Ψ² ← s²·Ψ²`. Esto referencia σ₈ a
/// `a=1` (convención estándar tipo CAMB/CLASS) en lugar de aplicarlo
/// directamente en `a_init`.
pub fn growth_factor_d(params: CosmologyParams, a: f64) -> f64 {
    if a <= 0.0 {
        return 0.0;
    }
    let a3 = a * a * a;
    let denom = params.omega_m + params.omega_lambda * a3;
    if denom <= 0.0 {
        return a;
    }
    let om_a = params.omega_m / denom;
    let ol_a = params.omega_lambda * a3 / denom;
    let bracket = om_a.max(0.0).powf(4.0 / 7.0) - ol_a + (1.0 + om_a / 2.0) * (1.0 + ol_a / 70.0);
    if bracket <= 0.0 {
        return a;
    }
    let g = 2.5 * om_a / bracket;
    a * g
}

/// Cociente del factor de crecimiento lineal `D(a_num)/D(a_den)` en CPT92.
///
/// Uso típico para reescalar ICs cosmológicas a la convención `D(1)=1`:
///
/// ```rust,ignore
/// let s = growth_factor_d_ratio(cosmo, a_init, 1.0);
/// psi1.iter_mut().for_each(|v| *v *= s);
/// psi2.iter_mut().for_each(|v| *v *= s * s);  // 2LPT crece como D²
/// ```
pub fn growth_factor_d_ratio(params: CosmologyParams, a_num: f64, a_den: f64) -> f64 {
    let d_den = growth_factor_d(params, a_den);
    if d_den == 0.0 {
        return 1.0;
    }
    growth_factor_d(params, a_num) / d_den
}

// ── Utilidades periódicas (Fase 18) ──────────────────────────────────────────

/// Diferencia periódica mínima imagen en 1D: resultado en `[-L/2, L/2]`.
///
/// Dado el desplazamiento `dx = x_i - x_j` y la longitud de caja `l`,
/// devuelve el desplazamiento equivalente más corto bajo condiciones periódicas.
///
/// Invariante: `|minimum_image(dx, l)| ≤ l/2`.
#[inline]
pub fn minimum_image(dx: f64, l: f64) -> f64 {
    dx - l * (dx / l).round()
}

/// Envuelve una coordenada escalar a `[0, l)` con condiciones periódicas.
#[inline]
pub fn wrap_coord(x: f64, l: f64) -> f64 {
    x.rem_euclid(l)
}

/// Envuelve la posición de una partícula a `[0, box_size)³` con condiciones periódicas.
///
/// Se debe llamar tras cada paso de drift en simulaciones periódicas para mantener
/// todas las posiciones dentro del cubo de simulación. El PM/CIC trabaja
/// internamente con `rem_euclid`, pero el wrap explícito es necesario para:
/// - diagnósticos correctos (`delta_rms`, snapshots)
/// - evitar deriva numérica acumulada
/// - consistencia con la lógica SFC si se usa en el futuro
#[inline]
pub fn wrap_position(pos: crate::vec3::Vec3, box_size: f64) -> crate::vec3::Vec3 {
    crate::vec3::Vec3::new(
        wrap_coord(pos.x, box_size),
        wrap_coord(pos.y, box_size),
        wrap_coord(pos.z, box_size),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Universo Einstein–de Sitter: Ω_m=1, Ω_Λ=0, a(t) ∝ t^{2/3}.
    ///
    /// Solución analítica desde a(0) = a0:
    /// `a(t) = (a0^{3/2} + 3/2 · H₀ · t)^{2/3}`
    fn eds_a(a0: f64, h0: f64, t: f64) -> f64 {
        (a0.powf(1.5) + 1.5 * h0 * t).powf(2.0 / 3.0)
    }

    fn eds_params(h0: f64) -> CosmologyParams {
        CosmologyParams::new(1.0, 0.0, h0)
    }

    #[test]
    fn advance_a_eds_matches_analytic() {
        let h0 = 1.0_f64;
        let a0 = 1.0_f64;
        let dt = 0.01_f64;
        let p = eds_params(h0);
        let a_num = p.advance_a(a0, dt);
        let a_ana = eds_a(a0, h0, dt);
        let rel = (a_num - a_ana).abs() / a_ana;
        // RK4 → error O(dt⁵) ≈ 1e-10 para dt=0.01
        assert!(
            rel < 1e-8,
            "advance_a EdS: rel_err = {rel:.2e} (esperado < 1e-8)"
        );
    }

    #[test]
    fn advance_a_accumulates_correctly() {
        // Integrar 10 pasos de dt=0.01 debe coincidir con avanzar dt_total=0.1 directamente.
        let p = eds_params(1.0);
        let a0 = 1.0_f64;
        let dt = 0.01_f64;
        let mut a_steps = a0;
        for _ in 0..10 {
            a_steps = p.advance_a(a_steps, dt);
        }
        let a_direct = eds_a(a0, 1.0, 0.1);
        let rel = (a_steps - a_direct).abs() / a_direct;
        assert!(rel < 1e-6, "acumulación de pasos: rel_err = {rel:.2e}");
    }

    /// Solución analítica para el integral drift en EdS:
    /// `∫_0^T dt/a²(t) = (2/H₀) · (1 − a(T)^{−1/2})`
    fn eds_drift_analytic(a0: f64, h0: f64, dt: f64) -> f64 {
        let a_end = eds_a(a0, h0, dt);
        2.0 / h0 * (1.0 - (a0 / a_end).sqrt())
    }

    /// Solución analítica para `∫_0^{T} dt/a(t)` en EdS:
    /// `= (2/H₀) · (a(T)^{1/2} − a0^{1/2})`
    fn eds_kick_integral_analytic(a0: f64, h0: f64, duration: f64) -> f64 {
        let a_end = eds_a(a0, h0, duration);
        2.0 / h0 * (a_end.sqrt() - a0.sqrt())
    }

    #[test]
    fn drift_kick_factors_eds_match_analytic() {
        let h0 = 1.0_f64;
        let a0 = 1.0_f64;
        let dt = 0.05_f64;
        let p = eds_params(h0);

        let (drift, kick_half, kick_half2) = p.drift_kick_factors(a0, dt);

        let drift_ana = eds_drift_analytic(a0, h0, dt);
        // kick_half cubre [0, dt/2]
        let kick_half_ana = eds_kick_integral_analytic(a0, h0, dt * 0.5);
        // kick_half2 cubre [dt/2, dt] — parte desde a(dt/2) durante dt/2
        let a_mid = eds_a(a0, h0, dt * 0.5);
        let kick_half2_ana = eds_kick_integral_analytic(a_mid, h0, dt * 0.5);

        let drift_rel = (drift - drift_ana).abs() / drift_ana;
        let kick_half_rel = (kick_half - kick_half_ana).abs() / kick_half_ana;
        let kick_half2_rel = (kick_half2 - kick_half2_ana).abs() / kick_half2_ana;

        assert!(drift_rel < 1e-7, "drift_factor: rel_err = {drift_rel:.2e}");
        assert!(
            kick_half_rel < 1e-7,
            "kick_half: rel_err = {kick_half_rel:.2e}"
        );
        assert!(
            kick_half2_rel < 1e-7,
            "kick_half2: rel_err = {kick_half2_rel:.2e}"
        );
    }

    #[test]
    fn drift_kick_factors_flat_space_equals_dt() {
        // Sin cosmología (H₀=0): factores deben ser dt y dt/2.
        let p = CosmologyParams::new(0.0, 0.0, 0.0);
        let dt = 0.1_f64;
        // H₀=0 → da/dt=0 → a permanece constante = a0 = 1.0
        let (drift, kick_half, kick_half2) = p.drift_kick_factors(1.0, dt);
        let tol = 1e-12;
        assert!((drift - dt).abs() < tol, "drift != dt cuando H₀=0");
        assert!(
            (kick_half - dt * 0.5).abs() < tol,
            "kick_half != dt/2 cuando H₀=0"
        );
        assert!(
            (kick_half2 - dt * 0.5).abs() < tol,
            "kick_half2 != dt/2 cuando H₀=0"
        );
    }

    #[test]
    fn hierarchical_prefixes_sums_match_drift_kick_factors() {
        // La suma total de los prefijos debe coincidir con drift_kick_factors.
        let p = eds_params(1.0);
        let a0 = 1.0_f64;
        let fine_dt = 0.01_f64;
        let n_fine = 8usize;
        let dt_total = fine_dt * n_fine as f64;

        let (_, kick_prefix, drift_prefix) = p.hierarchical_prefixes(a0, fine_dt, n_fine);
        let n_half = 2 * n_fine;

        let (drift_ref, kick_half_ref, kick_half2_ref) = p.drift_kick_factors(a0, dt_total);

        let drift_total = drift_prefix[n_half];
        let kick_total = kick_prefix[n_half];

        // Los prefijos usan regla trapezoidal; aceptamos error relativo < 0.1 %.
        let drift_rel = (drift_total - drift_ref).abs() / drift_ref;
        let kick_rel = (kick_total - (kick_half_ref + kick_half2_ref)).abs()
            / (kick_half_ref + kick_half2_ref);

        assert!(
            drift_rel < 1e-3,
            "drift total vs drift_kick_factors: {drift_rel:.2e}"
        );
        assert!(
            kick_rel < 1e-3,
            "kick total vs drift_kick_factors: {kick_rel:.2e}"
        );
    }

    // ── Tests Fase 37: growth_factor_d / CPT92 ────────────────────────────

    #[test]
    fn growth_factor_d_eds_equals_a() {
        // En EdS (Ω_m=1, Ω_Λ=0): Ω_m(a)=1, Ω_Λ(a)=0 → g(a) = 2.5/(1 − 0 + 1.5·1) = 1.
        // Entonces D(a) = a · g(a) = a exacto.
        let p = eds_params(1.0);
        for &a in &[0.01, 0.1, 0.5, 1.0, 2.0] {
            let d = growth_factor_d(p, a);
            let rel = (d - a).abs() / a;
            assert!(rel < 1e-12, "EdS D({a})={d} != a (rel={rel:.2e})");
        }
    }

    #[test]
    fn growth_factor_d_ratio_monotonic_lcdm() {
        // En ΛCDM con Ω_m=0.3, D(a) crece monótonamente: D(a_init) < D(1).
        let p = CosmologyParams::new(0.3, 0.7, 1.0);
        for &a_init in &[0.01, 0.02, 0.05, 0.10, 0.5] {
            let s = growth_factor_d_ratio(p, a_init, 1.0);
            assert!(
                (0.0..1.0).contains(&s),
                "s=D({a_init})/D(1)={s} fuera de (0,1)"
            );
            // A mayor a_init, mayor s (monotonía del modo creciente).
            let s_later = growth_factor_d_ratio(p, (a_init * 1.2).min(1.0), 1.0);
            assert!(
                s_later >= s - 1e-12,
                "D no monotónico: s({a_init})={s}, s(1.2·{a_init})={s_later}"
            );
        }
    }

    #[test]
    fn growth_factor_d_planck_at_a002() {
        // ΛCDM Planck-like: Ω_m=0.3, Ω_Λ=0.7, a_init=0.02. Esperamos
        // D(a_init)/D(1) ≈ 0.024 (crecimiento proporcional a a en régimen
        // dominado por materia, con corrección O(1) del prefactor CPT92).
        let p = CosmologyParams::new(0.3, 0.7, 1.0);
        let s = growth_factor_d_ratio(p, 0.02, 1.0);
        // Valor de referencia calculado con la misma fórmula CPT92: ≈ 0.02562
        // (2.5·Ω_m(0.02)/bracket(0.02) ≈ 1.281; 2.5·Ω_m(1)/bracket(1) ≈ 0.78).
        // D(0.02)/D(1) ≈ (0.02·1.281)/(1·0.780) ≈ 0.0328.
        assert!(
            (0.02..0.05).contains(&s),
            "D(0.02)/D(1)={s} fuera del rango esperado [0.02, 0.05]"
        );
    }

    #[test]
    fn growth_factor_d_zero_at_a_zero() {
        let p = CosmologyParams::new(0.3, 0.7, 1.0);
        assert_eq!(growth_factor_d(p, 0.0), 0.0);
    }
}
