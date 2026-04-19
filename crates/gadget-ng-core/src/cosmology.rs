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

/// Parámetros cosmológicos para ΛCDM plano.
///
/// `h0` es H₀ en unidades internas de tiempo (1/t_sim), no el parámetro
/// adimensional h₁₀₀ = H₀/(100 km/s/Mpc). El usuario es responsable de
/// la consistencia de unidades con el resto de la simulación.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CosmologyParams {
    /// Fracción de materia: Ω_m (sin dimensiones).
    pub omega_m: f64,
    /// Constante cosmológica: Ω_Λ (sin dimensiones).
    pub omega_lambda: f64,
    /// H₀ en unidades internas de tiempo (1/t_sim).
    pub h0: f64,
}

impl CosmologyParams {
    /// Crea nuevos parámetros cosmológicos.
    pub fn new(omega_m: f64, omega_lambda: f64, h0: f64) -> Self {
        Self {
            omega_m,
            omega_lambda,
            h0,
        }
    }

    /// `da/dt = a · H₀ · √(Ω_m/a³ + Ω_Λ)`.
    #[inline]
    fn da_dt(self, a: f64) -> f64 {
        let h_sq = self.omega_m / (a * a * a) + self.omega_lambda;
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

/// H(a) = H₀ · √(Ω_m·a⁻³ + Ω_Λ) — parámetro de Hubble en unidades internas.
pub fn hubble_param(params: CosmologyParams, a: f64) -> f64 {
    let h_sq = params.omega_m / (a * a * a) + params.omega_lambda;
    params.h0 * h_sq.max(0.0).sqrt()
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
}
