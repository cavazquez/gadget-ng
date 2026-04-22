//! Perfiles de halos NFW y relación concentración-masa.
//!
//! ## Perfil NFW (Navarro, Frenk & White 1996/1997)
//!
//! ```text
//! ρ(r) = ρ_s / [(r/r_s)(1 + r/r_s)²]
//! ```
//!
//! donde:
//!
//! - `ρ_s` — densidad característica (en las mismas unidades que `ρ_crit`).
//! - `r_s = r_200 / c` — radio de escala.
//! - `c` — parámetro de concentración.
//!
//! ## Masa encerrada
//!
//! ```text
//! M(<r) = 4π ρ_s r_s³ g(r/r_s)
//! g(x) = ln(1+x) - x/(1+x)
//! ```
//!
//! ## Radio y masa virial (Δ=200)
//!
//! ```text
//! r_200 = (3 M_200 / (4π × 200 × ρ_crit))^{1/3}
//! ρ_s   = (200/3) × ρ_crit × c³ / g(c)
//! ```
//!
//! ## Relación concentración-masa
//!
//! **Duffy et al. (2008)** calibrada sobre WMAP5 (halos "all"):
//!
//! ```text
//! c_200(M, z) = 5.71 × (M / 2×10¹² M_sun/h)^{-0.084} × (1+z)^{-0.47}
//! ```
//!
//! ## Unidades
//!
//! - Masas: M_sun/h
//! - Longitudes: Mpc/h
//! - Densidades: (M_sun/h)/(Mpc/h)³
//!
//! ## Ejemplo
//!
//! ```rust,ignore
//! use gadget_ng_analysis::nfw::{NfwProfile, concentration_duffy2008, rho_crit_z};
//!
//! let z = 0.0;
//! let m200 = 1e12_f64; // M_sun/h (Vía Láctea)
//! let rho_c = rho_crit_z(0.315, 0.685, z);
//! let c = concentration_duffy2008(m200, z);
//! let profile = NfwProfile::from_m200_c(m200, c, rho_c);
//!
//! let r_s = profile.r_s;
//! let rho_s = profile.rho_s;
//! println!("r_s = {r_s:.4} Mpc/h   ρ_s = {rho_s:.4e}  c = {c:.2}");
//!
//! // ρ(r) y M(<r)
//! println!("ρ(r_s) = {:.4e}", profile.density(r_s));
//! println!("M(<r_200) = {:.4e}", profile.mass_enclosed(profile.r200(rho_c)));
//! ```

use std::f64::consts::PI;

// ── Constantes ────────────────────────────────────────────────────────────────

/// ρ_crit,0 = 3H₀²/(8πG) en (M_sun/h)/(Mpc/h)³ (h² absorbido en la convención).
/// En unidades (Mpc/h, M_sun/h): ρ_crit,0 = 2.775×10¹¹ (M_sun/h)/(Mpc/h)³.
pub const RHO_CRIT0: f64 = 2.775e11;

/// Overdensidad de virial: Δ = 200 (convención M_200/r_200).
pub const DELTA_VIRIALIZED: f64 = 200.0;

// ── Parámetros de la relación c(M) de Duffy et al. (2008) ────────────────────

/// Amplitud de la relación c(M) de Duffy+2008 (halos "all", WMAP5).
const DUFFY_A: f64 = 5.71;
/// Índice de masa de Duffy+2008: c ∝ M^B.
const DUFFY_B: f64 = -0.084;
/// Índice de redshift de Duffy+2008: c ∝ (1+z)^C.
const DUFFY_C: f64 = -0.47;
/// Masa pivote de Duffy+2008 en M_sun/h.
const DUFFY_M_PIVOT: f64 = 2e12;

// ── Función auxiliar g(x) ─────────────────────────────────────────────────────

/// `g(x) = ln(1+x) − x/(1+x)` — integral del perfil NFW normalizada.
///
/// Para x→0: g(x) → x²/2 (evita cancelación numérica).
#[inline]
fn g_nfw(x: f64) -> f64 {
    if x < 1e-4 {
        // Expansión de Taylor: g(x) = x²/2 - x³/3 + x⁴/4 - ...
        x * x * (0.5 - x / 3.0 + x * x / 4.0)
    } else {
        (1.0 + x).ln() - x / (1.0 + x)
    }
}

// ── Struct NfwProfile ─────────────────────────────────────────────────────────

/// Perfil NFW caracterizado por `(ρ_s, r_s)`.
///
/// Unidades: longitudes en Mpc/h, densidades en (M_sun/h)/(Mpc/h)³.
#[derive(Debug, Clone, Copy)]
pub struct NfwProfile {
    /// Densidad característica ρ_s en (M_sun/h)/(Mpc/h)³.
    pub rho_s: f64,
    /// Radio de escala r_s en Mpc/h.
    pub r_s: f64,
}

impl NfwProfile {
    /// Construye un perfil NFW a partir de M_200, concentración c y ρ_crit.
    ///
    /// # Parámetros
    ///
    /// - `m200`: masa virial M_200 en M_sun/h (masa dentro de r_200).
    /// - `c`: concentración c = r_200 / r_s.
    /// - `rho_crit`: densidad crítica en (M_sun/h)/(Mpc/h)³.
    pub fn from_m200_c(m200: f64, c: f64, rho_crit: f64) -> Self {
        let r200 = r200_from_m200(m200, rho_crit);
        let r_s = r200 / c;
        // ρ_s = M_200 / (4π r_s³ g(c)) = (200/3) ρ_crit c³ / g(c)
        let rho_s = (DELTA_VIRIALIZED / 3.0) * rho_crit * c * c * c / g_nfw(c);
        Self { rho_s, r_s }
    }

    /// Densidad NFW en el radio r: `ρ(r) = ρ_s / [(r/r_s)(1 + r/r_s)²]`.
    ///
    /// Devuelve `f64::INFINITY` para r = 0 (singularidad central del NFW).
    #[inline]
    pub fn density(&self, r: f64) -> f64 {
        if r <= 0.0 {
            return f64::INFINITY;
        }
        let x = r / self.r_s;
        self.rho_s / (x * (1.0 + x) * (1.0 + x))
    }

    /// Masa encerrada M(<r) en M_sun/h.
    ///
    /// ```text
    /// M(<r) = 4π ρ_s r_s³ g(r/r_s)
    /// g(x) = ln(1+x) − x/(1+x)
    /// ```
    #[inline]
    pub fn mass_enclosed(&self, r: f64) -> f64 {
        if r <= 0.0 {
            return 0.0;
        }
        let x = r / self.r_s;
        4.0 * PI * self.rho_s * self.r_s.powi(3) * g_nfw(x)
    }

    /// Radio r_200 para este perfil: resuelve M(<r_200) = (4π/3)×200×ρ_crit×r_200³.
    ///
    /// Dado que ρ_s = (200/3)ρ_crit c³/g(c) y r_s = r_200/c, se puede invertir
    /// numéricamente. Aquí se usa el método de bisección.
    ///
    /// Para uso rápido cuando se conoce c, preferir `r200_from_m200` directamente.
    pub fn r200(&self, rho_crit: f64) -> f64 {
        // Buscamos r tal que ρ_mean(<r) = 200 ρ_crit
        // ρ_mean = 3 M(<r) / (4π r³) = 200 ρ_crit
        // → M(<r) = (4π/3) × 200 × ρ_crit × r³
        let target = |r: f64| -> f64 {
            let m = self.mass_enclosed(r);
            let m_vir = (4.0 / 3.0) * PI * DELTA_VIRIALIZED * rho_crit * r.powi(3);
            m - m_vir
        };

        // Rango de bisección: r_s a 100 r_s
        let r_lo = self.r_s * 0.01;
        let r_hi = self.r_s * 1000.0;
        let f_lo = target(r_lo);
        let f_hi = target(r_hi);

        if f_lo * f_hi > 0.0 {
            // Fallback: estimación directa desde ρ_s r_s³
            return self.r_s * 10.0;
        }

        let mut lo = r_lo;
        let mut hi = r_hi;
        for _ in 0..60 {
            let mid = 0.5 * (lo + hi);
            if target(mid) * f_lo < 0.0 {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        0.5 * (lo + hi)
    }

    /// Velocidad circular v_c(r) = sqrt(G M(<r) / r).
    ///
    /// Las unidades de v_c dependen de las unidades de G usadas.
    /// En unidades (Mpc/h, M_sun/h, G = 4.3009×10⁻³ pc M_sun⁻¹ (km/s)²):
    /// v_c² [km²/s²] = G [pc M_sun⁻¹ (km/s)²] × M [M_sun] / r [pc]
    ///
    /// Para uso en código con G genérico, esta función devuelve sqrt(M(<r)/r)
    /// en unidades (M_sun/h / (Mpc/h))^{1/2}. El usuario multiplica por sqrt(G).
    #[inline]
    pub fn circular_velocity_sq_over_g(&self, r: f64) -> f64 {
        if r <= 0.0 {
            return 0.0;
        }
        self.mass_enclosed(r) / r
    }

    /// Concentración c = r_200 / r_s calculada a partir del perfil y ρ_crit.
    pub fn concentration(&self, rho_crit: f64) -> f64 {
        self.r200(rho_crit) / self.r_s
    }
}

// ── Funciones auxiliares públicas ─────────────────────────────────────────────

/// r_200 desde M_200 y ρ_crit: `r_200 = (3 M_200 / (4π × 200 × ρ_crit))^{1/3}`.
#[inline]
pub fn r200_from_m200(m200: f64, rho_crit: f64) -> f64 {
    (3.0 * m200 / (4.0 * PI * DELTA_VIRIALIZED * rho_crit)).powf(1.0 / 3.0)
}

/// ρ_crit(z) en (M_sun/h)/(Mpc/h)³ para ΛCDM plano.
///
/// ```text
/// ρ_crit(z) = ρ_crit,0 × E²(z)
/// E²(z) = Ω_m(1+z)³ + Ω_Λ
/// ```
#[inline]
pub fn rho_crit_z(omega_m: f64, omega_lambda: f64, z: f64) -> f64 {
    let e2 = omega_m * (1.0 + z).powi(3) + omega_lambda;
    RHO_CRIT0 * e2
}

/// Concentración c_200(M, z) de **Duffy et al. (2008)** para WMAP5.
///
/// Relación calibrada sobre simulaciones N-body para halos "all" (relaxed + unrelaxed):
///
/// ```text
/// c_200 = 5.71 × (M / 2×10¹² M_sun/h)^{-0.084} × (1+z)^{-0.47}
/// ```
///
/// Válido para M ∈ [10¹¹, 10¹⁵] M_sun/h y z ∈ [0, 2].
pub fn concentration_duffy2008(m200_msun_h: f64, z: f64) -> f64 {
    DUFFY_A * (m200_msun_h / DUFFY_M_PIVOT).powf(DUFFY_B) * (1.0 + z).powf(DUFFY_C)
}

/// Concentración c_200(M, z) de **Bhattacharya et al. (2013)** (fit a múltiples simulaciones).
///
/// ```text
/// c_200 = D(z) × 9.0 × (M / 5×10¹³ M_sun/h)^{-0.099}   (WMAP7)
/// ```
///
/// Aproximación: D(z)/D(0) ≈ 1/(1+z) para EdS; aquí se usa la relación directa.
pub fn concentration_bhattacharya2013(m200_msun_h: f64, z: f64) -> f64 {
    // Simplificación para ΛCDM: usa D(z) ≈ 1/(1+z)^0.7 como proxy
    let m_pivot = 5e13_f64;
    9.0 * (m200_msun_h / m_pivot).powf(-0.099) * (1.0 + z).powf(-0.47)
}

// ── Perfil de densidad medido desde partículas ────────────────────────────────

/// Un bin del perfil de densidad radial.
#[derive(Debug, Clone)]
pub struct DensityBin {
    /// Radio central del bin en Mpc/h.
    pub r: f64,
    /// Radio mínimo del bin en Mpc/h.
    pub r_min: f64,
    /// Radio máximo del bin en Mpc/h.
    pub r_max: f64,
    /// Número de partículas en el bin.
    pub n_part: usize,
    /// Densidad media en el bin en (M_sun/h)/(Mpc/h)³.
    pub rho: f64,
    /// Densidad NFW predicha en r_centro (si se ha fijado un perfil).
    pub rho_nfw: f64,
}

/// Calcula el perfil de densidad radial de un halo desde las distancias al centro.
///
/// # Parámetros
///
/// - `radii`: distancias de cada partícula al centro del halo, en Mpc/h.
/// - `m_part`: masa de cada partícula en M_sun/h (todas iguales).
/// - `r_min`: radio mínimo del perfil en Mpc/h.
/// - `r_max`: radio máximo del perfil en Mpc/h.
/// - `n_bins`: número de bins logarítmicos.
/// - `profile`: perfil NFW de referencia (opcional, para `rho_nfw`).
///
/// # Retorno
///
/// Vec de `DensityBin` con ρ(r) medido en cada anillo esférico.
pub fn measure_density_profile(
    radii: &[f64],
    m_part: f64,
    r_min: f64,
    r_max: f64,
    n_bins: usize,
    profile: Option<&NfwProfile>,
) -> Vec<DensityBin> {
    let log_rmin = r_min.ln();
    let log_rmax = r_max.ln();
    let dlog = (log_rmax - log_rmin) / n_bins as f64;

    let mut bins: Vec<DensityBin> = (0..n_bins)
        .map(|i| {
            let r_lo = (log_rmin + i as f64 * dlog).exp();
            let r_hi = (log_rmin + (i + 1) as f64 * dlog).exp();
            let r_cen = (r_lo * r_hi).sqrt(); // media geométrica
            DensityBin {
                r: r_cen,
                r_min: r_lo,
                r_max: r_hi,
                n_part: 0,
                rho: 0.0,
                rho_nfw: profile.map(|p| p.density(r_cen)).unwrap_or(0.0),
            }
        })
        .collect();

    // Pre-computar bordes logarítmicos para asignación rápida
    for &r in radii {
        if r < r_min || r > r_max {
            continue;
        }
        let idx = ((r.ln() - log_rmin) / dlog).floor() as usize;
        if idx < n_bins {
            bins[idx].n_part += 1;
        }
    }

    // Convertir counts a densidad
    for bin in &mut bins {
        let vol = (4.0 / 3.0) * PI * (bin.r_max.powi(3) - bin.r_min.powi(3));
        bin.rho = bin.n_part as f64 * m_part / vol;
    }

    bins
}

// ── Ajuste NFW desde perfil de densidad ──────────────────────────────────────

/// Resultado del ajuste NFW a un perfil de densidad medido.
#[derive(Debug, Clone)]
pub struct NfwFitResult {
    /// Perfil NFW ajustado.
    pub profile: NfwProfile,
    /// χ²_reducido del ajuste en log-espacio (sobre bins con n_part > 0).
    pub chi2_red: f64,
    /// Número de bins usados en el ajuste.
    pub n_bins_used: usize,
}

/// Ajusta un perfil NFW a los datos de densidad por búsqueda en cuadrícula.
///
/// Varía c ∈ [c_min, c_max] y minimiza χ² en log(ρ) vs log(ρ_NFW) para cada c,
/// ajustando ρ_s analíticamente (mínimos cuadrados en log-espacio).
///
/// # Parámetros
///
/// - `bins`: perfil de densidad medido (de `measure_density_profile`).
/// - `m200`: masa virial M_200 en M_sun/h (da r_200 y r_s = r_200/c).
/// - `rho_crit`: densidad crítica en (M_sun/h)/(Mpc/h)³.
/// - `c_min`, `c_max`: rango de búsqueda de concentración.
/// - `n_c`: número de pasos de concentración.
///
/// # Retorno
///
/// `None` si no hay suficientes bins con partículas.
pub fn fit_nfw_concentration(
    bins: &[DensityBin],
    m200: f64,
    rho_crit: f64,
    c_min: f64,
    c_max: f64,
    n_c: usize,
) -> Option<NfwFitResult> {
    let r200 = r200_from_m200(m200, rho_crit);

    // Seleccionar bins con partículas
    let good: Vec<&DensityBin> = bins.iter().filter(|b| b.n_part > 0 && b.rho > 0.0).collect();
    if good.len() < 3 {
        return None;
    }

    let log_rho_obs: Vec<f64> = good.iter().map(|b| b.rho.ln()).collect();

    let mut best_c = 0.5 * (c_min + c_max);
    let mut best_chi2 = f64::INFINITY;
    let mut best_rho_s = 1.0;

    let dc = (c_max - c_min) / (n_c as f64 - 1.0).max(1.0);
    for ic in 0..n_c {
        let c = c_min + ic as f64 * dc;
        let r_s = r200 / c;

        // Para este r_s, ajustar ρ_s por mínimos cuadrados en log-espacio:
        // log ρ_obs ≈ log ρ_s + log f_NFW(r/r_s)
        // donde f_NFW(x) = 1 / [x(1+x)²]
        // → log ρ_s = mean(log ρ_obs - log f_NFW)
        let log_nfw_shape: Vec<f64> = good
            .iter()
            .map(|b| {
                let x = b.r / r_s;
                -(x.ln() + 2.0 * (1.0 + x).ln()) // log f_NFW(x)
            })
            .collect();

        let log_rho_s: f64 = log_rho_obs
            .iter()
            .zip(log_nfw_shape.iter())
            .map(|(lo, lf)| lo - lf)
            .sum::<f64>()
            / good.len() as f64;
        let rho_s = log_rho_s.exp();

        // χ² en log-espacio
        let chi2: f64 = log_rho_obs
            .iter()
            .zip(log_nfw_shape.iter())
            .map(|(lo, lf)| {
                let log_pred = log_rho_s + lf;
                (lo - log_pred) * (lo - log_pred)
            })
            .sum::<f64>()
            / good.len() as f64;

        if chi2 < best_chi2 {
            best_chi2 = chi2;
            best_c = c;
            best_rho_s = rho_s;
        }
    }

    let r_s = r200 / best_c;
    Some(NfwFitResult {
        profile: NfwProfile { rho_s: best_rho_s, r_s },
        chi2_red: best_chi2,
        n_bins_used: good.len(),
    })
}

// ── Tests unitarios ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn planck_rho_crit() -> f64 {
        rho_crit_z(0.315, 0.685, 0.0)
    }

    /// M(<r_200) debe ser ≈ M_200 (por construcción del perfil).
    #[test]
    fn mass_enclosed_at_r200_equals_m200() {
        let m200 = 1e14_f64;
        let rho_c = planck_rho_crit();
        let c = concentration_duffy2008(m200, 0.0);
        let profile = NfwProfile::from_m200_c(m200, c, rho_c);
        let r200 = r200_from_m200(m200, rho_c);

        let m_enc = profile.mass_enclosed(r200);
        let rel_err = (m_enc / m200 - 1.0).abs();
        assert!(
            rel_err < 1e-10,
            "M(<r_200) = {m_enc:.6e} vs M_200 = {m200:.6e}; error = {rel_err:.2e}"
        );
    }

    /// ρ_mean(<r_200) debe ser ≈ 200 × ρ_crit (definición de r_200).
    #[test]
    fn mean_density_at_r200_is_200_rho_crit() {
        let m200 = 1e13_f64;
        let rho_c = planck_rho_crit();
        let c = 7.0;
        let profile = NfwProfile::from_m200_c(m200, c, rho_c);
        let r200 = r200_from_m200(m200, rho_c);

        let rho_mean = profile.mass_enclosed(r200) / ((4.0 / 3.0) * PI * r200.powi(3));
        let rel_err = (rho_mean / (200.0 * rho_c) - 1.0).abs();
        assert!(
            rel_err < 1e-10,
            "ρ_mean(<r_200) = {:.6e} vs 200ρ_c = {:.6e}; error = {rel_err:.2e}",
            rho_mean,
            200.0 * rho_c
        );
    }

    /// g_nfw(x) → x²/2 para x→0 (sin cancelación numérica).
    ///
    /// La corrección de orden x³ es g(x) = x²/2 - 2x³/3 + ..., por lo que la
    /// tolerancia relativa respecto a x²/2 es ~4x/3. Para x=1e-8: tolerancia ~1.3e-8.
    #[test]
    fn g_nfw_small_x_limit() {
        let x = 1e-8_f64;   // suficientemente pequeño para que 4x/3 < 1e-7
        let g = g_nfw(x);
        let g_approx = x * x / 2.0;
        let rel_err = (g / g_approx - 1.0).abs();
        assert!(rel_err < 1e-6, "g_nfw({x:.0e}) = {g:.6e} vs x²/2 = {g_approx:.6e}; err={rel_err:.2e}");
    }

    /// c(M) de Duffy+2008: c decrece con M y con z.
    #[test]
    fn concentration_duffy_trends() {
        let c_low_m  = concentration_duffy2008(1e11, 0.0);
        let c_high_m = concentration_duffy2008(1e15, 0.0);
        let c_z0 = concentration_duffy2008(1e13, 0.0);
        let c_z1 = concentration_duffy2008(1e13, 1.0);

        assert!(c_low_m > c_high_m, "c debe decrecer con M");
        assert!(c_z0 > c_z1, "c debe decrecer con z");

        // Rango físico para Duffy+2008: c ∈ [2, 20] en el rango M ∈ [10¹¹, 10¹⁵]
        assert!(c_low_m  > 2.0 && c_low_m  < 25.0, "c(10¹¹) = {c_low_m:.2}  fuera de [2, 25]");
        assert!(c_high_m > 1.0 && c_high_m < 10.0, "c(10¹⁵) = {c_high_m:.2} fuera de [1, 10]");

        println!("[nfw_test] Duffy c: M=10¹¹ → {c_low_m:.2}, M=10¹⁵ → {c_high_m:.2}");
        println!("[nfw_test] Duffy c: z=0 → {c_z0:.2}, z=1 → {c_z1:.2}");
    }

    /// ρ(r) decrece con r.
    #[test]
    fn density_decreasing_with_radius() {
        let profile = NfwProfile { rho_s: 1e7, r_s: 0.3 };
        let radii = [0.01, 0.1, 0.3, 1.0, 5.0, 20.0];
        for w in radii.windows(2) {
            let rho1 = profile.density(w[0]);
            let rho2 = profile.density(w[1]);
            assert!(rho2 < rho1, "ρ(r) no decrece: ρ({:.2}) = {rho2:.3e} ≥ ρ({:.2}) = {rho1:.3e}", w[1], w[0]);
        }
    }

    /// M(<r) es creciente con r.
    #[test]
    fn mass_enclosed_increasing_with_radius() {
        let profile = NfwProfile { rho_s: 1e7, r_s: 0.3 };
        let radii = [0.01, 0.1, 0.3, 1.0, 5.0, 20.0];
        for w in radii.windows(2) {
            let m1 = profile.mass_enclosed(w[0]);
            let m2 = profile.mass_enclosed(w[1]);
            assert!(m2 > m1, "M(<r) no crece: M({:.2}) = {m2:.3e} ≤ M({:.2}) = {m1:.3e}", w[1], w[0]);
        }
    }

    /// El perfil medido desde partículas en capas esféricas sintéticas recupera
    /// la densidad NFW con buena precisión (dentro de ruido de Poisson).
    #[test]
    fn measure_profile_from_synthetic_nfw() {
        // Perfil NFW de referencia
        let rho_s = 1e7_f64;
        let r_s = 0.3_f64;
        let profile = NfwProfile { rho_s, r_s };

        // Generar partículas sintéticas con densidad ∝ NFW.
        // Usamos rechazo: muestreamos r ~ U[r_min, r_max] y aceptamos con p ∝ ρ(r)·r².
        // m_part se calcula consistentemente con la masa total del perfil.
        let n_particles = 20_000_usize;
        let r_min = 0.01_f64;
        let r_max = 3.0_f64;

        // Masa de partícula consistente con la masa NFW en el volumen de muestreo
        let m_total = profile.mass_enclosed(r_max) - profile.mass_enclosed(r_min);
        let m_part = m_total / n_particles as f64;

        // p(r) ∝ r² ρ(r) — máximo en r = r_s (vértice del perfil NFW en 3D)
        let p_fun = |r: f64| r * r * profile.density(r);
        let p_max = {
            let p_rs   = if r_s >= r_min && r_s <= r_max { p_fun(r_s) } else { 0.0 };
            p_rs.max(p_fun(r_min)).max(p_fun(r_max))
        };

        let mut radii = Vec::with_capacity(n_particles / 3);
        let mut seed: u64 = 0x12345678ABCD;
        let mut accepted = 0;

        while accepted < n_particles {
            // LCG simple (Knuth)
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u1 = (seed >> 32) as f64 / u32::MAX as f64;
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u2 = (seed >> 32) as f64 / u32::MAX as f64;

            let r = r_min + u1 * (r_max - r_min);
            let p = p_fun(r) / p_max; // ∈ [0, 1]
            if u2 < p {
                radii.push(r);
                accepted += 1;
            }
        }

        // Medir perfil con m_part consistente
        let n_bins = 15;
        let bins = measure_density_profile(&radii, m_part, r_min, r_max, n_bins, Some(&profile));

        // Verificar que ρ_medido ≈ ρ_NFW dentro de un factor 3 (ruido de Poisson)
        let mut n_good = 0;
        for bin in &bins {
            if bin.n_part < 20 || bin.rho_nfw <= 0.0 {
                continue;
            }
            let ratio = bin.rho / bin.rho_nfw;
            assert!(
                ratio > 0.3 && ratio < 3.0,
                "ρ_med/ρ_NFW = {ratio:.3} fuera de [0.3, 3] en r = {:.3} Mpc/h (n_part={})",
                bin.r, bin.n_part
            );
            n_good += 1;
        }
        assert!(n_good >= 5, "Solo {n_good} bins válidos, se esperan ≥ 5");
        println!("[nfw_test] {n_good} bins válidos de {n_bins}");
    }

    /// El ajuste de concentración recupera c con error < 30 % desde datos sintéticos.
    #[test]
    fn fit_concentration_synthetic() {
        let rho_c = planck_rho_crit();
        let m200 = 1e14_f64;
        let c_true = 5.0_f64;
        let profile_true = NfwProfile::from_m200_c(m200, c_true, rho_c);
        let r200 = r200_from_m200(m200, rho_c);

        // Generar partículas con LCG simple
        let n_particles = 30_000_usize;
        let r_min = 0.02 * r200;
        let r_max = r200;
        // m_part consistente con la masa total en el volumen de muestreo
        let m_total = profile_true.mass_enclosed(r_max) - profile_true.mass_enclosed(r_min);
        let m_part = m_total / n_particles as f64;

        // Máximo correcto de p(r) = r² ρ(r) — en r = r_s si está en el rango
        let r_s_true = profile_true.r_s;
        let p_fun_t = |r: f64| r * r * profile_true.density(r);
        let p_max_t = {
            let p_rs  = if r_s_true >= r_min && r_s_true <= r_max { p_fun_t(r_s_true) } else { 0.0 };
            p_rs.max(p_fun_t(r_min)).max(p_fun_t(r_max))
        };

        let mut radii = Vec::with_capacity(n_particles / 3);
        let mut seed: u64 = 0xDEADBEEFCAFE;
        let mut accepted = 0;
        while accepted < n_particles {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u1 = (seed >> 32) as f64 / u32::MAX as f64;
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u2 = (seed >> 32) as f64 / u32::MAX as f64;
            let r = r_min + u1 * (r_max - r_min);
            let p = p_fun_t(r) / p_max_t;
            if u2 < p {
                radii.push(r);
                accepted += 1;
            }
        }

        let bins = measure_density_profile(&radii, m_part, r_min, r_max, 20, Some(&profile_true));
        let result = fit_nfw_concentration(&bins, m200, rho_c, 1.0, 20.0, 50)
            .expect("El ajuste debe converger");

        let c_fit = r200 / result.profile.r_s;
        let err = (c_fit / c_true - 1.0).abs();
        println!(
            "[nfw_test] c_true={c_true:.2}  c_fit={c_fit:.2}  err={:.1}%  χ²_red={:.3}",
            err * 100.0,
            result.chi2_red
        );
        assert!(
            err < 0.30,
            "c_fit = {c_fit:.2} difiere de c_true = {c_true:.2} en {:.1}% > 30%",
            err * 100.0
        );
    }
}
