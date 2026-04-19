//! Funciones de transferencia cosmológica y normalización por σ₈.
//!
//! ## Función de transferencia de Eisenstein–Hu (no-wiggle)
//!
//! Implementa la aproximación sin oscilaciones acústicas bariónicas (BAO) de
//! Eisenstein & Hu (1998, ApJ 496, 605), ecuaciones 29–31.
//!
//! Esta aproximación es apropiada cuando:
//! - La resolución de la caja no puede resolver las oscilaciones BAO (~150 Mpc/h)
//! - Se necesita una función de transferencia realista con corrección bariónica
//! - Se quiere evitar la complejidad de la versión con wiggles
//!
//! ## Normalización por σ₈
//!
//! σ₈ es la RMS de fluctuaciones de densidad suavizadas con un filtro top-hat
//! de radio R = 8 h⁻¹ Mpc:
//!
//! ```text
//! σ²(R) = (1/2π²) ∫ k² P(k) W²(kR) dk
//! W(x)  = 3[sin(x) − x cos(x)] / x³   (top-hat en k-space)
//! ```
//!
//! La normalización se obtiene calculando la amplitud A tal que σ(8) = σ₈_target.
//!
//! ## Unidades
//!
//! | Cantidad | Unidades |
//! |----------|----------|
//! | k en T(k) | h/Mpc |
//! | k en σ²(R) | Mpc⁻¹ (= h/Mpc × h, ver nota abajo) |
//! | R en σ²(R) | Mpc/h |
//! | P(k) | [Mpc/h]³ (convención h³/Mpc³ implícita en la definición de A) |
//!
//! **Nota sobre unidades de k**: la integral de σ₈ se hace en k [Mpc⁻¹] para
//! consistencia con la convención estándar de la literatura. La función de
//! transferencia T(k) recibe k en [h/Mpc], que se convierte internamente.
//! En el generador de ICs, `k_hmpc = 2π |n| h / box_size_mpc_h`.

/// Parámetros cosmológicos para la función de transferencia de Eisenstein–Hu.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EisensteinHuParams {
    /// Densidad total de materia (fracción crítica), sin dimensiones.
    pub omega_m: f64,
    /// Densidad de bariones (fracción crítica), sin dimensiones.
    pub omega_b: f64,
    /// Parámetro de Hubble adimensional: h = H₀ / (100 km/s/Mpc).
    pub h: f64,
    /// Temperatura del CMB en Kelvin (presente para completitud;
    /// la versión no-wiggle no la usa explícitamente pero se documenta).
    pub t_cmb: f64,
}

impl Default for EisensteinHuParams {
    fn default() -> Self {
        // Parámetros Planck 2018 (TT,TE,EE+lowE+lensing)
        Self {
            omega_m: 0.315,
            omega_b: 0.049,
            h: 0.674,
            t_cmb: 2.7255,
        }
    }
}

/// Función de transferencia de Eisenstein–Hu en la aproximación sin-wiggle.
///
/// Implementa las ecuaciones 29–31 de Eisenstein & Hu (1998, ApJ 496, 605).
///
/// # Parámetros
/// - `k_hmpc`: número de onda en unidades de **h/Mpc**.
/// - `p`: parámetros cosmológicos (Ω_m, Ω_b, h).
///
/// # Retorna
/// La función de transferencia T(k) ∈ (0, 1], con T(k→0) = 1.
///
/// # Formulación
///
/// ```text
/// ωm     = Ω_m h²
/// ωb     = Ω_b h²
/// s      = 44.5 ln(9.83/ωm) / sqrt(1 + 10 ωb^{3/4})              [horizonte de sonido, Mpc]
/// α_Γ    = 1 − 0.328 ln(431 ωm)(Ω_b/Ω_m) + 0.38 ln(22.3 ωm)(Ω_b/Ω_m)²
/// Γ(k)   = Ω_m h [α_Γ + (1−α_Γ)/(1 + (0.43 k s)^4)]            [con k en h/Mpc]
/// q(k)   = k h / Γ(k)                                              [k en h/Mpc → k h en Mpc⁻¹]
/// L₀     = ln(2e + 1.8 q)
/// C₀     = 14.2 + 731/(1 + 62.5 q)
/// T(k)   = L₀ / (L₀ + C₀ q²)
/// ```
pub fn transfer_eh_nowiggle(k_hmpc: f64, p: &EisensteinHuParams) -> f64 {
    if k_hmpc <= 0.0 {
        return 1.0;
    }

    let omega_m_h2 = p.omega_m * p.h * p.h;
    let omega_b_h2 = p.omega_b * p.h * p.h;
    let fb = p.omega_b / p.omega_m; // fracción bariónica

    // Horizonte de sonido aproximado [Mpc].
    let s = 44.5 * (9.83_f64 / omega_m_h2).ln()
        / (1.0 + 10.0 * omega_b_h2.powf(0.75)).sqrt();

    // Supresión bariónica del shape parameter.
    let alpha_gamma = 1.0
        - 0.328 * (431.0 * omega_m_h2).ln() * fb
        + 0.38 * (22.3 * omega_m_h2).ln() * fb * fb;

    // Γ_eff dependiente de k (EH98 eq. 31).
    // k_hmpc está en h/Mpc; el producto k_hmpc * s es adimensional
    // (k [h/Mpc] × s [Mpc] = k [h/Mpc] × s [Mpc], ojo: la fórmula original usa
    // k en Mpc⁻¹ = k_hmpc × h, por lo que k_hmpc * s debe ser k_hmpc * h * (s/h)
    // pero en la forma EH98 no-wiggle el argumento es k_hmpc * s con s en Mpc
    // porque Γ ya tiene dimensiones de h/Mpc → k [h/Mpc] × s [Mpc/h] sería 0.43*k*s
    // con s en Mpc/h. Aquí s está en Mpc, así que el argumento es 0.43 * k_hmpc * h * s / h
    // = 0.43 * k_hmpc * s con k en h/Mpc y s en Mpc → tiene dimensiones h.
    // La EH98 usa k en Mpc⁻¹ en la forma (0.43 k s)⁴ con s en Mpc.
    // Con k_hmpc en h/Mpc → k_mpc = k_hmpc * h [Mpc⁻¹].
    let k_mpc = k_hmpc * p.h; // [Mpc⁻¹]
    let ks = 0.43 * k_mpc * s;
    let gamma_eff =
        p.omega_m * p.h * (alpha_gamma + (1.0 - alpha_gamma) / (1.0 + ks.powi(4)));

    // Parámetro adimensional q (EH98 eq. 30).
    // q = k [Mpc⁻¹] / Γ_eff [Mpc⁻¹]; con k_mpc en Mpc⁻¹ y gamma_eff en h/Mpc:
    // gamma_eff tiene unidades de Ω_m·h [h/Mpc] ≡ [Mpc⁻¹·(Mpc/h)·h] = [adim × h × h/Mpc]
    // La fórmula EH98 define q = k/(h·Γ_eff) con k en h/Mpc, Γ_eff adimensional.
    // En su definición original, Γ_eff = Ω_m·h·[...] con Ω_m·h en h/Mpc.
    // Usamos q = k_hmpc / Γ_eff (con k en h/Mpc y Γ_eff en h/Mpc → q adimensional).
    let q = k_hmpc / gamma_eff.max(1e-30);

    // Polinomio de Eisenstein-Hu (EH98 eq. 29).
    let l0 = (std::f64::consts::E * 2.0 + 1.8 * q).ln();
    let c0 = 14.2 + 731.0 / (1.0 + 62.5 * q);
    let t = l0 / (l0 + c0 * q * q);

    t.max(0.0).min(1.0)
}

/// Calcula σ²(R) para el espectro P(k) = k^n_s · T²(k) con amplitud unitaria A=1.
///
/// Integral numérica con la regla del trapecio en espacio logarítmico de k.
///
/// # Parámetros
/// - `r_mpc_h`: radio del filtro top-hat en **Mpc/h**.
/// - `n_s`: índice espectral primordial.
/// - `p`: parámetros cosmológicos para la función de transferencia.
///
/// # Retorna
/// σ²(R, A=1) en unidades de [Mpc/h]³ (la amplitud A² cancela en la normalización).
///
/// # Fórmula
/// ```text
/// σ²(R) = (1/2π²) ∫ k² · k^n_s · T²(k) · W²(kR) dk
/// W(x)  = 3[sin(x) − x cos(x)] / x³
/// ```
/// con k en Mpc⁻¹ y R en Mpc/h, usando k = k_h × h para convertir de h/Mpc.
pub fn sigma_sq_unit(r_mpc_h: f64, n_s: f64, p: &EisensteinHuParams) -> f64 {
    // Rango de integración en k [h/Mpc].
    // k_min debe ser << 1/R, k_max debe ser >> k_eq.
    let k_min_hmpc = 1e-5_f64;
    let k_max_hmpc = 5e2_f64;
    let n_steps = 8192_usize;

    let ln_k_min = k_min_hmpc.ln();
    let ln_k_max = k_max_hmpc.ln();
    let d_ln_k = (ln_k_max - ln_k_min) / n_steps as f64;

    // Integrar en log-k: dk = k · d(ln k)
    // σ²(R) = (1/2π²) ∫ k² P(k) W²(kR) dk
    //       = (1/2π²) ∫ k³ P(k) W²(kR) d(ln k)
    // con P(k) = k^n_s T²(k) (A=1, k en h/Mpc pero la integral es consistente)

    let integrand = |k_hmpc: f64| -> f64 {
        let tk = transfer_eh_nowiggle(k_hmpc, p);
        // W(kR) con k en h/Mpc y R en Mpc/h → kR adimensional
        let x = k_hmpc * r_mpc_h;
        let w = tophat_window(x);
        // k^n_s · T²(k) · W²(kR) · k³ (factor k³ de d(ln k))
        // Usamos k en h/Mpc directamente (la normalización A² absorbe las dimensiones)
        k_hmpc.powf(n_s + 3.0) * tk * tk * w * w
    };

    // Regla del trapecio en log-k.
    let mut sum = 0.0_f64;
    let mut k_prev = k_min_hmpc;
    let mut f_prev = integrand(k_prev);
    for i in 1..=n_steps {
        let k = (ln_k_min + i as f64 * d_ln_k).exp();
        let f = integrand(k);
        sum += 0.5 * (f_prev + f) * d_ln_k;
        k_prev = k;
        f_prev = f;
    }
    let _ = k_prev; // silenciar warning

    sum / (2.0 * std::f64::consts::PI * std::f64::consts::PI)
}

/// Filtro top-hat en k-space: W(x) = 3[sin(x) − x cos(x)] / x³.
///
/// Para x→0: W(x) → 1.
#[inline]
pub fn tophat_window(x: f64) -> f64 {
    if x.abs() < 1e-4 {
        // Expansión de Taylor: 1 - x²/10 + x⁴/280 - ...
        1.0 - x * x / 10.0
    } else {
        3.0 * (x.sin() - x * x.cos()) / (x * x * x)
    }
}

/// Calcula la amplitud A tal que σ(8 Mpc/h) = `sigma8_target`.
///
/// Asume P(k) = A² · k^n_s · T²(k) y devuelve A para que:
/// `σ(8 Mpc/h, A) = sigma8_target`
///
/// # Parámetros
/// - `sigma8_target`: σ₈ objetivo (adimensional).
/// - `n_s`: índice espectral primordial.
/// - `p`: parámetros cosmológicos.
///
/// # Retorna
/// La amplitud A en las mismas unidades que P(k)^{1/2} · k^{-(n_s+3)/2}.
/// Esta amplitud se usa directamente en el generador de ICs para escalar
/// el campo de desplazamiento.
pub fn amplitude_for_sigma8(sigma8_target: f64, n_s: f64, p: &EisensteinHuParams) -> f64 {
    let sigma_sq_unit_val = sigma_sq_unit(8.0, n_s, p);
    if sigma_sq_unit_val <= 0.0 {
        return sigma8_target;
    }
    sigma8_target / sigma_sq_unit_val.sqrt()
}

/// Calcula σ₈ de un conjunto de partículas usando los modos del grid.
///
/// Usa el estimador de P(k) de la grilla directamente: suma sobre modos
/// del grid ponderados por el filtro top-hat, sin depender del estimador
/// post-simulación (que necesita posiciones reales).
///
/// Esta función es interna y se usa en tests.
///
/// # Parámetros
/// - `pk_bins`: vector de (k [h/Mpc], P(k) [Mpc/h]³) en bins esféricos.
/// - `r_mpc_h`: radio del filtro top-hat en Mpc/h (habitualmente 8.0).
///
/// # Retorna
/// σ(R) calculado desde los bins de P(k).
pub fn sigma_from_pk_bins(pk_bins: &[(f64, f64)], r_mpc_h: f64) -> f64 {
    if pk_bins.is_empty() {
        return 0.0;
    }
    // Suma discreta: σ²(R) = (1/2π²) Σ k² P(k) W²(kR) Δk
    // Usando diferencias finitas entre bins para Δk.
    let mut sum = 0.0_f64;
    for i in 0..pk_bins.len() {
        let (k, pk) = pk_bins[i];
        if pk <= 0.0 || k <= 0.0 {
            continue;
        }
        let x = k * r_mpc_h;
        let w = tophat_window(x);
        // Δk desde bin anterior al siguiente (o diferencia simple).
        let dk = if pk_bins.len() > 1 {
            if i == 0 {
                pk_bins[1].0 - pk_bins[0].0
            } else if i == pk_bins.len() - 1 {
                pk_bins[i].0 - pk_bins[i - 1].0
            } else {
                0.5 * (pk_bins[i + 1].0 - pk_bins[i - 1].0)
            }
        } else {
            k * 0.1 // fallback
        };
        sum += k * k * pk * w * w * dk;
    }
    (sum / (2.0 * std::f64::consts::PI * std::f64::consts::PI)).sqrt()
}

// ── Tests unitarios ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn planck18() -> EisensteinHuParams {
        EisensteinHuParams {
            omega_m: 0.315,
            omega_b: 0.049,
            h: 0.674,
            t_cmb: 2.7255,
        }
    }

    /// T(k→0) debe ser ≈ 1 (horizonte cosmológico).
    #[test]
    fn eh_transfer_low_k_is_one() {
        let p = planck18();
        let t = transfer_eh_nowiggle(1e-4, &p);
        assert!(
            (t - 1.0).abs() < 1e-3,
            "T(k=1e-4 h/Mpc) = {:.6} ≠ 1 (tolerancia 1e-3)",
            t
        );
    }

    /// T(k=10 h/Mpc) < T(k=0.1 h/Mpc): supresión en alto k.
    ///
    /// Para Planck18 con EH no-wiggle, la escala de igualdad está en
    /// k_eq ≈ Ω_m h² / 2997 Mpc ≈ 0.015 h/Mpc. Por tanto k=0.1 h/Mpc
    /// ya está dentro del régimen suprimido (T ≈ 0.14), y k=10 h/Mpc
    /// está muy fuertemente suprimido (T ≈ 1e-4).
    #[test]
    fn eh_transfer_high_k_suppressed() {
        let p = planck18();
        let t_low = transfer_eh_nowiggle(0.1, &p);
        let t_high = transfer_eh_nowiggle(10.0, &p);
        assert!(
            t_high < t_low,
            "T(10) = {:.6} ≥ T(0.1) = {:.6} — falta supresión en alto k",
            t_high, t_low
        );
        // T(k=0.1 h/Mpc) con Planck18 EH no-wiggle: ≈ 0.13-0.16.
        // k=0.1 ya está bien por encima de k_eq ≈ 0.015 h/Mpc.
        assert!(t_low > 0.05 && t_low < 0.30,
            "T(0.1 h/Mpc) = {:.4} fuera del rango esperado (0.05, 0.30) para Planck18", t_low);
        // T(k=10 h/Mpc) debe estar muy fuertemente suprimido.
        assert!(t_high < 0.01,
            "T(10 h/Mpc) = {:.6} no está lo suficientemente suprimido (esperado < 0.01)", t_high);
    }

    /// T(k) ∈ (0, 1] para todo k > 0.
    #[test]
    fn eh_transfer_range_valid() {
        let p = planck18();
        for log_k in [-4, -3, -2, -1, 0, 1, 2] {
            let k = 10.0_f64.powi(log_k);
            let t = transfer_eh_nowiggle(k, &p);
            assert!(
                t > 0.0 && t <= 1.0 + 1e-10,
                "T(k={:.0e}) = {:.6} fuera de (0, 1]",
                k, t
            );
        }
    }

    /// σ²(8, A=1) debe ser positiva y finita.
    #[test]
    fn sigma_sq_unit_is_positive_finite() {
        let p = planck18();
        let s2 = sigma_sq_unit(8.0, 0.965, &p);
        assert!(
            s2.is_finite() && s2 > 0.0,
            "σ²(8, A=1) = {:.4e} — debe ser positivo y finito",
            s2
        );
    }

    /// amplitude_for_sigma8(0.8) → el σ₈ del espectro resultante es ≈ 0.8.
    ///
    /// Verificación de autoconsistencia: calculamos A y luego rehacemos la integral.
    #[test]
    fn amplitude_for_sigma8_is_consistent() {
        let p = planck18();
        let sigma8_target = 0.8;
        let n_s = 0.965;
        let a = amplitude_for_sigma8(sigma8_target, n_s, &p);

        // σ²(8) con amplitud A = A² × σ²(8, A=1)
        let s2_unit = sigma_sq_unit(8.0, n_s, &p);
        let sigma8_recovered = a * s2_unit.sqrt();

        let rel_err = (sigma8_recovered - sigma8_target).abs() / sigma8_target;
        assert!(
            rel_err < 1e-3,
            "σ₈ recuperado = {:.6} vs target = {:.6} (error = {:.2e})",
            sigma8_recovered, sigma8_target, rel_err
        );
    }

    /// El filtro top-hat satisface W(0) = 1.
    #[test]
    fn tophat_window_at_zero() {
        let w = tophat_window(0.0);
        assert!((w - 1.0).abs() < 1e-12, "W(0) = {} ≠ 1", w);
    }

    /// El filtro top-hat decrece para x grande.
    #[test]
    fn tophat_window_decays() {
        assert!(tophat_window(1.0) < tophat_window(0.1));
        assert!(tophat_window(10.0) < tophat_window(1.0));
    }

    /// Verificación numérica de T(k) en varios puntos para Planck18.
    ///
    /// Valores de referencia calculados con la fórmula EH98 no-wiggle:
    /// - T(k=1e-3 h/Mpc) ≈ 0.99+ (por encima de k_eq ≈ 0.015 h/Mpc → casi = 1)
    /// - T(k=0.01 h/Mpc) ≈ 0.7–0.9 (cerca del umbral de igualdad)
    /// - T(k=0.1 h/Mpc) ≈ 0.10–0.20 (régimen fuertemente suprimido)
    /// - T(k=1 h/Mpc) ≈ 0.001–0.01 (muy suprimido)
    #[test]
    fn eh_transfer_reference_value() {
        let p = planck18();
        // A k=1e-3 h/Mpc (escala mayor que k_eq), T debe ser muy cercano a 1.
        let t_vlow = transfer_eh_nowiggle(1e-3, &p);
        assert!(t_vlow > 0.90, "T(1e-3 h/Mpc) = {:.4} — esperado > 0.90", t_vlow);

        // A k=0.1 h/Mpc, T ≈ 0.13-0.16 para Planck18.
        let t = transfer_eh_nowiggle(0.1, &p);
        assert!(
            t > 0.05 && t < 0.30,
            "T(k=0.1 h/Mpc, Planck18) = {:.4} — fuera del rango EH98 (0.05, 0.30)",
            t
        );

        // La función es estrictamente decreciente: T(0.01) > T(0.1) > T(1.0).
        let t_mid = transfer_eh_nowiggle(0.01, &p);
        let t_high = transfer_eh_nowiggle(1.0, &p);
        assert!(t_mid > t, "T no decrece de 0.01 a 0.1 h/Mpc: {:.4} vs {:.4}", t_mid, t);
        assert!(t > t_high, "T no decrece de 0.1 a 1.0 h/Mpc: {:.4} vs {:.4}", t, t_high);
    }
}
