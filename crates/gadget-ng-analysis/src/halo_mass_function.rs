//! Función de masa de halos: Press-Schechter (1974) y Sheth-Tormen (1999).
//!
//! ## Física
//!
//! La función de masa de halos `dn/d ln M` da el número comoving de halos por
//! unidad de volumen y por intervalo logarítmico de masa:
//!
//! ```text
//! dn/d ln M = (ρ̄_m / M) × |d ln σ⁻¹ / d ln M| × f(σ)
//! ```
//!
//! donde:
//!
//! - `ρ̄_m = Ω_m × ρ_crit,0` es la densidad media de materia comoving.
//! - `σ(M)` es la desviación estándar de las fluctuaciones de densidad suavizadas
//!   en la escala de Lagrange `R(M) = (3M/4πρ̄_m)^{1/3}`.
//! - `f(σ)` es la función de multiplicidad.
//!
//! ## Funciones de multiplicidad
//!
//! **Press-Schechter (1974)**:
//!
//! ```text
//! f_PS(σ) = √(2/π) × (δ_c/σ) × exp(−δ_c²/(2σ²))
//! ```
//!
//! **Sheth-Tormen (1999)** (colapso elipsoidal, mejor ajuste a simulaciones):
//!
//! ```text
//! f_ST(σ) = A × √(2a/π) × (δ_c/σ) × [1 + (σ²/(a δ_c²))^p] × exp(−a δ_c²/(2σ²))
//! ```
//!
//! con `a = 0.707`, `p = 0.3`, `A = 0.3222` (Sheth & Tormen 1999).
//!
//! ## Unidades
//!
//! Todas las masas en **M_sun/h** y longitudes en **Mpc/h** (convención GADGET).
//!
//! ```text
//! ρ̄_m  [M_sun/h / (Mpc/h)³] = Ω_m × 2.775×10¹¹
//! dn/d ln M  [h³/Mpc³]
//! ```
//!
//! ## Ejemplo
//!
//! ```rust,ignore
//! use gadget_ng_analysis::halo_mass_function::{HmfParams, mass_function_table};
//!
//! let params = HmfParams::planck2018();
//! let table = mass_function_table(&params, 1e10, 1e15, 30, 0.0);
//! for bin in &table {
//!     println!("log10(M)={:.2}  dn/dlnM_ST={:.4e}", bin.log10_m, bin.n_st);
//! }
//! ```

use std::f64::consts::PI;

// ── Constantes físicas ────────────────────────────────────────────────────────

/// ρ_crit,0 = 3H₀²/(8πG) en unidades de h² M_sun/Mpc³.
/// En (Mpc/h, M_sun/h): ρ_crit = 2.775×10¹¹ (M_sun/h)/(Mpc/h)³.
pub const RHO_CRIT_H2: f64 = 2.775e11;

/// δ_c: sobredensidad crítica de colapso en aproximación EdS.
/// Para ΛCDM el valor es levemente dependiente de Ω_m pero
/// δ_c ≈ 1.686 es una excelente aproximación para z ≥ 0.
pub const DELTA_C: f64 = 1.686;

// ── Parámetros de la función de multiplicidad Sheth-Tormen ────────────────────

/// Parámetro `a` de Sheth-Tormen (colapso elipsoidal). Valor canónico: 0.707.
pub const ST_A_PARAM: f64 = 0.707;

/// Parámetro `p` de Sheth-Tormen. Valor canónico: 0.3.
pub const ST_P_PARAM: f64 = 0.3;

/// Normalización de Sheth-Tormen: `A = 0.3222` (conserva n_halos = n_total para f(σ)).
pub const ST_A_NORM: f64 = 0.3222;

// ── Parámetros cosmológicos de la HMF ─────────────────────────────────────────

/// Parámetros cosmológicos necesarios para calcular la función de masa.
#[derive(Debug, Clone, Copy)]
pub struct HmfParams {
    /// Fracción de densidad de materia a z=0 (sin dimensiones).
    pub omega_m: f64,
    /// Fracción de densidad de energía oscura a z=0.
    pub omega_lambda: f64,
    /// Parámetro de Hubble adimensional `h = H₀ / (100 km/s/Mpc)`.
    pub h: f64,
    /// Amplitud del espectro de potencias lineal a z=0 (σ₈).
    pub sigma8: f64,
    /// Índice espectral primordial `n_s`.
    pub n_s: f64,
    /// Fracción de bariones `Ω_b`.
    pub omega_b: f64,
    /// Temperatura del CMB en K.
    pub t_cmb: f64,
}

impl HmfParams {
    /// Planck 2018 (TT+TE+EE+lowE, Ω_k=0).
    pub fn planck2018() -> Self {
        Self {
            omega_m: 0.315,
            omega_lambda: 0.685,
            h: 0.674,
            sigma8: 0.811,
            n_s: 0.965,
            omega_b: 0.049,
            t_cmb: 2.7255,
        }
    }

    /// Densidad media de materia comoving en `(M_sun/h) / (Mpc/h)³`.
    #[inline]
    pub fn rho_bar_m(&self) -> f64 {
        self.omega_m * RHO_CRIT_H2
    }

    /// E²(z) = H²(z)/H₀² para ΛCDM plano.
    #[inline]
    pub fn e2(&self, z: f64) -> f64 {
        self.omega_m * (1.0 + z).powi(3) + self.omega_lambda
    }

    /// Factor de crecimiento lineal D(z)/D(0) normalizado.
    ///
    /// Aproximación de Carroll, Press & Turner (1992), precisa al ~1 % para ΛCDM.
    pub fn growth_factor_ratio(&self, z: f64) -> f64 {
        if z <= 0.0 {
            return 1.0;
        }
        let omega_mz = self.omega_m * (1.0 + z).powi(3) / self.e2(z);
        let omega_lz = self.omega_lambda / self.e2(z);
        let g_z = 2.5 * omega_mz
            / (omega_mz.powf(4.0 / 7.0) - omega_lz
                + (1.0 + omega_mz / 2.0) * (1.0 + omega_lz / 70.0));
        let omega_m0 = self.omega_m;
        let omega_l0 = self.omega_lambda;
        let g_0 = 2.5 * omega_m0
            / (omega_m0.powf(4.0 / 7.0) - omega_l0
                + (1.0 + omega_m0 / 2.0) * (1.0 + omega_l0 / 70.0));
        g_z / (g_0 * (1.0 + z))
    }
}

// ── Filtro top-hat ────────────────────────────────────────────────────────────

/// W(x) = 3(sin x − x cos x)/x³ — filtro esférico top-hat en espacio de Fourier.
#[inline]
fn tophat_win(x: f64) -> f64 {
    if x < 1e-4 {
        return 1.0 - x * x / 10.0;
    }
    3.0 * (x.sin() - x * x.cos()) / (x * x * x)
}

// ── σ(M) ──────────────────────────────────────────────────────────────────────

/// σ²(R) = (1/2π²) ∫₀^∞ k² P_lin(k) W²(kR) dk
///
/// Integración trapezoidal en k log-equiespaciado (1200 puntos, k ∈ [1e-5, 1e4]).
fn sigma_sq_r(r_hmpc: f64, p_linear: &dyn Fn(f64) -> f64) -> f64 {
    const N: usize = 1200;
    let log_kmin = 1e-5_f64.ln();
    let log_kmax = 1e4_f64.ln();
    let dlog_k = (log_kmax - log_kmin) / (N as f64 - 1.0);

    let mut integral = 0.0_f64;
    let mut prev = 0.0_f64;
    for i in 0..N {
        let k = (log_kmin + i as f64 * dlog_k).exp();
        let w = tophat_win(k * r_hmpc);
        let curr = k * k * k * p_linear(k) * w * w; // k³ P W² (integrando en ln k)
        if i > 0 {
            integral += 0.5 * (prev + curr) * dlog_k;
        }
        prev = curr;
    }
    integral / (2.0 * PI * PI)
}

/// Derivada `d ln σ / d ln M` calculada por diferencias finitas centrales.
///
/// Usa la relación R ∝ M^{1/3} → `d ln σ / d ln M = (1/3) d ln σ / d ln R`.
fn d_ln_sigma_d_ln_m(r_hmpc: f64, p_linear: &dyn Fn(f64) -> f64) -> f64 {
    // eps = 2%: equilibrio entre error de truncación y cancelación numérica.
    let eps = 0.02_f64;
    let s2p = sigma_sq_r(r_hmpc * (1.0 + eps), p_linear);
    let s2m = sigma_sq_r(r_hmpc * (1.0 - eps), p_linear);
    if s2p <= 0.0 || s2m <= 0.0 {
        return 0.0;
    }
    // d ln σ / d ln R = (1/2) d ln σ² / d ln R
    let dls_dlnr = 0.5 * (s2p.ln() - s2m.ln()) / (2.0 * eps);
    // d ln σ / d ln M = (1/3) d ln σ / d ln R
    dls_dlnr / 3.0
}

// ── Funciones de multiplicidad ─────────────────────────────────────────────────

/// Press-Schechter (1974): `f(σ) = √(2/π) × ν × exp(−ν²/2)` con `ν = δ_c/σ`.
///
/// Satisface ∫ f(σ) d ln σ⁻¹ = 1 (todos los halos en alguna masa).
#[inline]
pub fn multiplicity_ps(sigma: f64) -> f64 {
    let nu = DELTA_C / sigma;
    (2.0 / PI).sqrt() * nu * (-0.5 * nu * nu).exp()
}

/// Sheth-Tormen (1999): colapso elipsoidal, mejor ajuste a simulaciones N-body.
///
/// ```text
/// f_ST(σ) = A √(2a/π) ν [1 + (aν²)^{-p}] exp(−aν²/2)
/// ```
///
/// con `a = 0.707`, `p = 0.3`, `A = 0.3222`, `ν = δ_c/σ`.
#[inline]
pub fn multiplicity_st(sigma: f64) -> f64 {
    let nu = DELTA_C / sigma;
    let a = ST_A_PARAM;
    let p = ST_P_PARAM;
    let a_norm = ST_A_NORM;
    let a_nu2 = a * nu * nu;
    a_norm * (2.0 * a / PI).sqrt() * nu * (1.0 + a_nu2.powf(-p)) * (-0.5 * a_nu2).exp()
}

// ── Resultado de un bin de la HMF ─────────────────────────────────────────────

/// Un bin de la tabla de función de masa de halos.
#[derive(Debug, Clone)]
pub struct HmfBin {
    /// log₁₀(M) con M en M_sun/h.
    pub log10_m: f64,
    /// Masa en M_sun/h.
    pub m_msun_h: f64,
    /// Radio de Lagrange en Mpc/h.
    pub r_hmpc: f64,
    /// σ(M, z): desviación estándar de fluctuaciones de densidad.
    pub sigma: f64,
    /// |d ln σ⁻¹ / d ln M| = −d ln σ / d ln M.
    pub dlns_inv_dlnm: f64,
    /// dn/d ln M Press-Schechter en h³/Mpc³.
    pub n_ps: f64,
    /// dn/d ln M Sheth-Tormen en h³/Mpc³.
    pub n_st: f64,
}

// ── API pública ────────────────────────────────────────────────────────────────

/// Calcula σ(M, z) para una masa dada.
///
/// # Parámetros
///
/// - `m_msun_h`: masa del halo en M_sun/h.
/// - `params`: parámetros cosmológicos.
/// - `z`: redshift objetivo.
///
/// # Retorno
///
/// σ(M, z) = D(z)/D(0) × σ(M, z=0).
pub fn sigma_m(m_msun_h: f64, params: &HmfParams, z: f64) -> f64 {
    use gadget_ng_core::{EisensteinHuParams, amplitude_for_sigma8};

    let eh = EisensteinHuParams {
        omega_m: params.omega_m,
        omega_b: params.omega_b,
        h: params.h,
        t_cmb: params.t_cmb,
    };
    let amp = amplitude_for_sigma8(params.sigma8, params.n_s, &eh);

    // P(k) = amp² × k^n_s × T²(k) — ver ic_zeldovich.rs, línea 275.
    // amplitude_for_sigma8 devuelve A tal que A × sqrt(sigma_sq_unit) = sigma8,
    // lo que implica A² × sigma_sq_unit = sigma8² (varianza correcta).
    let amp2 = amp * amp;
    let p_lin = |k: f64| -> f64 {
        let tk = gadget_ng_core::transfer_eh_nowiggle(k, &eh);
        amp2 * k.powf(params.n_s) * tk * tk
    };

    let r = lagrange_radius(m_msun_h, params.rho_bar_m());
    let sigma_z0 = sigma_sq_r(r, &p_lin).max(0.0).sqrt();
    sigma_z0 * params.growth_factor_ratio(z)
}

/// Radio de Lagrange R(M) en Mpc/h: masa M contiene ρ̄_m en una esfera de radio R.
///
/// `R = (3M / (4π ρ̄_m))^{1/3}`
#[inline]
pub fn lagrange_radius(m_msun_h: f64, rho_bar_m: f64) -> f64 {
    (3.0 * m_msun_h / (4.0 * PI * rho_bar_m)).powf(1.0 / 3.0)
}

/// dn/d ln M para Press-Schechter en h³/Mpc³.
///
/// ```text
/// dn/d ln M = (ρ̄_m / M) × |d ln σ⁻¹ / d ln M| × f_PS(σ)
/// ```
pub fn hmf_press_schechter(m_msun_h: f64, sigma: f64, dlns_inv_dlnm: f64, rho_bar_m: f64) -> f64 {
    let f = multiplicity_ps(sigma);
    (rho_bar_m / m_msun_h) * dlns_inv_dlnm * f
}

/// dn/d ln M para Sheth-Tormen en h³/Mpc³.
pub fn hmf_sheth_tormen(m_msun_h: f64, sigma: f64, dlns_inv_dlnm: f64, rho_bar_m: f64) -> f64 {
    let f = multiplicity_st(sigma);
    (rho_bar_m / m_msun_h) * dlns_inv_dlnm * f
}

/// Genera la tabla completa de la función de masa de halos.
///
/// # Parámetros
///
/// - `params`: parámetros cosmológicos.
/// - `m_min_msun_h`: masa mínima en M_sun/h (e.g. 1e10).
/// - `m_max_msun_h`: masa máxima en M_sun/h (e.g. 1e15).
/// - `n_bins`: número de bins log-equiespaciados en masa.
/// - `z`: redshift.
///
/// # Retorno
///
/// `Vec<HmfBin>` con σ(M), dn/d ln M (PS y ST) para cada bin.
///
/// # Unidades
///
/// - Masas: M_sun/h
/// - dn/d ln M: h³ Mpc⁻³
pub fn mass_function_table(
    params: &HmfParams,
    m_min_msun_h: f64,
    m_max_msun_h: f64,
    n_bins: usize,
    z: f64,
) -> Vec<HmfBin> {
    use gadget_ng_core::{EisensteinHuParams, amplitude_for_sigma8};

    let eh = EisensteinHuParams {
        omega_m: params.omega_m,
        omega_b: params.omega_b,
        h: params.h,
        t_cmb: params.t_cmb,
    };
    let amp = amplitude_for_sigma8(params.sigma8, params.n_s, &eh);
    let d_ratio = params.growth_factor_ratio(z);

    // P_lin(k, z=0) = amp² × k^n_s × T²(k) en (Mpc/h)³
    let amp2 = amp * amp;
    let p_lin = move |k: f64| -> f64 {
        let tk = gadget_ng_core::transfer_eh_nowiggle(k, &eh);
        amp2 * k.powf(params.n_s) * tk * tk
    };

    let rho_bar = params.rho_bar_m();
    let log_mmin = m_min_msun_h.log10();
    let log_mmax = m_max_msun_h.log10();
    let dlog = (log_mmax - log_mmin) / (n_bins as f64 - 1.0).max(1.0);

    (0..n_bins)
        .map(|i| {
            let log10_m = log_mmin + i as f64 * dlog;
            let m = 10.0_f64.powf(log10_m);
            let r = lagrange_radius(m, rho_bar);

            // σ(M, z=0) y derivada
            let sigma_z0 = sigma_sq_r(r, &p_lin).max(0.0).sqrt();
            let sigma = sigma_z0 * d_ratio;

            // d ln σ⁻¹ / d ln M = −d ln σ / d ln M
            let dls_dlnm = d_ln_sigma_d_ln_m(r, &p_lin);
            let dlns_inv = (-dls_dlnm).max(0.0); // debe ser positivo (σ decrece con M)

            let n_ps = if sigma > 1e-10 && dlns_inv > 0.0 {
                hmf_press_schechter(m, sigma, dlns_inv, rho_bar)
            } else {
                0.0
            };
            let n_st = if sigma > 1e-10 && dlns_inv > 0.0 {
                hmf_sheth_tormen(m, sigma, dlns_inv, rho_bar)
            } else {
                0.0
            };

            HmfBin {
                log10_m,
                m_msun_h: m,
                r_hmpc: r,
                sigma,
                dlns_inv_dlnm: dlns_inv,
                n_ps,
                n_st,
            }
        })
        .collect()
}

/// Número total de halos por volumen (integral de dn/d ln M × d ln M).
///
/// Integración trapezoidal sobre la tabla HMF.
pub fn total_halo_density(table: &[HmfBin]) -> (f64, f64) {
    if table.len() < 2 {
        return (0.0, 0.0);
    }
    let mut n_ps = 0.0;
    let mut n_st = 0.0;
    for w in table.windows(2) {
        let dlnm = (w[1].log10_m - w[0].log10_m) * std::f64::consts::LN_10;
        n_ps += 0.5 * (w[0].n_ps + w[1].n_ps) * dlnm;
        n_st += 0.5 * (w[0].n_st + w[1].n_st) * dlnm;
    }
    (n_ps, n_st)
}

// ── Tests unitarios ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn planck() -> HmfParams {
        HmfParams::planck2018()
    }

    /// σ(8 Mpc/h) en M(8) debe ser ≈ σ₈ (consistencia de normalización).
    #[test]
    fn sigma_r8_equals_sigma8() {
        let p = planck();
        let m_r8 = p.rho_bar_m() * (4.0 / 3.0) * PI * 8.0_f64.powi(3);
        let sigma = sigma_m(m_r8, &p, 0.0);
        let rel_err = (sigma / p.sigma8 - 1.0).abs();
        assert!(
            rel_err < 0.02,
            "σ(R=8) = {sigma:.4} vs σ₈ = {} err={:.2}%",
            p.sigma8,
            rel_err * 100.0
        );
    }

    /// σ(M) es monótonamente decreciente con M.
    #[test]
    fn sigma_decreasing_with_mass() {
        let p = planck();
        let masses = [1e10, 1e11, 1e12, 1e13, 1e14, 1e15];
        let sigmas: Vec<f64> = masses.iter().map(|&m| sigma_m(m, &p, 0.0)).collect();
        for i in 1..sigmas.len() {
            assert!(
                sigmas[i] < sigmas[i - 1],
                "σ no monótona: σ[{}]={:.4} σ[{}]={:.4}",
                i,
                sigmas[i],
                i - 1,
                sigmas[i - 1]
            );
        }
    }

    /// σ(M, z) crece con el redshift (menor crecimiento en el pasado).
    #[test]
    fn sigma_decreases_with_redshift() {
        let p = planck();
        let m = 1e13_f64;
        let s_z0 = sigma_m(m, &p, 0.0);
        let s_z1 = sigma_m(m, &p, 1.0);
        let s_z3 = sigma_m(m, &p, 3.0);
        assert!(s_z0 > s_z1, "σ(z=0)={s_z0:.4} debe ser > σ(z=1)={s_z1:.4}");
        assert!(s_z1 > s_z3, "σ(z=1)={s_z1:.4} debe ser > σ(z=3)={s_z3:.4}");
    }

    /// ST siempre da más halos que PS a masas grandes (mejor ajuste a simulaciones).
    #[test]
    fn st_vs_ps_high_mass() {
        let p = planck();
        let table = mass_function_table(&p, 1e13, 1e15, 10, 0.0);
        // A masas grandes (σ ≪ δ_c), ST corrige hacia arriba respecto a PS.
        let high_m = &table[table.len() - 2];
        assert!(
            high_m.n_st >= high_m.n_ps * 0.5,
            "ST no debe ser mucho menor que PS a masas altas"
        );
        println!(
            "[hmf_test] log10M={:.2}  n_PS={:.3e}  n_ST={:.3e}  ratio={:.2}",
            high_m.log10_m,
            high_m.n_ps,
            high_m.n_st,
            high_m.n_st / high_m.n_ps.max(1e-99)
        );
    }

    /// La tabla tiene los campos correctos y coherentes.
    #[test]
    fn table_fields_coherent() {
        let p = planck();
        let table = mass_function_table(&p, 1e10, 1e15, 20, 0.0);
        assert_eq!(table.len(), 20);
        for bin in &table {
            assert!(bin.m_msun_h > 0.0);
            assert!(bin.sigma > 0.0);
            assert!(bin.n_ps >= 0.0);
            assert!(bin.n_st >= 0.0);
            // R(M) debe ser razonable (0.01 – 50 Mpc/h para 10¹⁰–10¹⁵ M_sun/h)
            assert!(
                bin.r_hmpc > 0.01 && bin.r_hmpc < 100.0,
                "R={:.3} fuera de rango para M={:.3e}",
                bin.r_hmpc,
                bin.m_msun_h
            );
        }
    }

    /// n_total está en el orden de magnitud correcto: ~10 h³/Mpc³ para 10¹⁰–10¹⁵.
    #[test]
    fn total_halo_density_reasonable() {
        let p = planck();
        let table = mass_function_table(&p, 1e10, 1e15, 40, 0.0);
        let (n_ps, n_st) = total_halo_density(&table);
        println!("[hmf_test] n_total: PS={n_ps:.4e}  ST={n_st:.4e}  [h³/Mpc³]");
        // Para 10¹⁰–10¹⁵ M_sun/h, la densidad integrada es O(1–100) h³/Mpc³.
        assert!(n_ps > 1e-3 && n_ps < 1e4, "n_PS fuera de rango: {n_ps:.3e}");
        assert!(n_st > 1e-3 && n_st < 1e4, "n_ST fuera de rango: {n_st:.3e}");
    }
}
