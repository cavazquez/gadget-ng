//! Química de no-equilibrio primordial (Phase 86 + red 12 especies).
//!
//! ## Red de química
//!
//! Doce especies: HI, HII, HeI, HeII, HeIII, e⁻, H⁻, H₂, H₂⁺, D, D⁺, HD.
//!
//! Fracciones de ionización (en unidades de la fracción de hidrógeno/helio total):
//! - `x_hi`:   fracción HI  (hidrógeno neutro)
//! - `x_hii`:  fracción HII (hidrógeno ionizado)
//! - `x_hei`:  fracción HeI
//! - `x_heii`: fracción HeII
//! - `x_heiii`:fracción HeIII
//! - `x_e`:    fracción de electrones (número de electrones por átomo de H)
//! - `x_hm`:   fracción H⁻
//! - `x_h2`:   fracción H₂
//! - `x_h2p`:  fracción H₂⁺
//! - `x_d`:    fracción D
//! - `x_dp`:   fracción D⁺
//! - `x_hd`:   fracción HD
//!
//! ## Ecuaciones de red
//!
//! ```text
//! dx_hii/dt  = Γ_HI × x_hi  + β_HI(T) × x_hi × x_e  - α_HII(T) × x_hii × x_e
//! dx_heii/dt = Γ_HeI × x_hei + β_HeI(T) × x_hei × x_e - α_HeII(T) × x_heii × x_e
//!              - β_HeII(T) × x_heii × x_e + α_HeIII(T) × x_heiii × x_e
//! dx_heiii/dt = β_HeII(T) × x_heii × x_e - α_HeIII(T) × x_heiii × x_e
//! dx_hi    = 1 - x_hii                    (conservación de H)
//! dx_hei   = f_He - x_heii - x_heiii      (conservación de He)
//! x_e      = x_hii + x_heii + 2·x_heiii   (neutralidad de carga)
//! ```
//!
//! donde `f_He = 0.0789` es la fracción de helio por número (Y_He/(4·X_H), Planck2018).
//!
//! ## Solver implícito
//!
//! Se usa un solver subcíclico con paso adaptativo (Anninos et al. 1997):
//! - Paso máximo: `dt_chem = min(dt, 0.1 / |max(rates)|)`.
//! - Actualización implícita tipo Euler de primer orden.
//! - Subciclo hasta cubrir el `dt` completo de simulación.
//!
//! ## Tasas de reacción
//!
//! Recombinación (Verner & Ferland 1996):
//! - `α_HII(T)`:   case-B, fit potencial T^-0.7.
//! - `α_HeII(T)`:  caso B, fit exponencial.
//! - `α_HeIII(T)`: similar a HII pero escalado.
//!
//! Ionización colisional (Cen 1992):
//! - `β_HI(T)`:   ∝ T^0.5 × exp(−T_HI/T).
//! - `β_HeI(T)`:  ∝ T^0.5 × exp(−T_HeI/T).
//! - `β_HeII(T)`: ∝ T^0.5 × exp(−T_HeII/T).
//!
//! ## Referencia
//!
//! Anninos et al. (1997), New Astron. 2, 209;
//! Cen (1992), ApJS 78, 341;
//! Verner & Ferland (1996), ApJS 103, 467.

use crate::m1::RadiationField;
use gadget_ng_core::Particle;
#[cfg(feature = "rayon")]
use rayon::prelude::*;

// ── Constantes ────────────────────────────────────────────────────────────────

/// Fracción de helio por número (Y_He / (4 × X_H), Planck 2018).
pub const F_HE: f64 = 0.0789;
/// Abundancia primordial de deuterio por número, D/H.
pub const F_D: f64 = 2.5e-5;
/// Energía de ionización de HI en Kelvin: E_HI / k_B.
const T_HI: f64 = 157_809.1;
/// Energía de ionización de HeI en Kelvin.
const T_HEI: f64 = 285_335.4;
/// Energía de ionización de HeII en Kelvin.
const T_HEII: f64 = 631_515.0;
/// Factor de conversión de energía interna (código → erg/g).
const U_CODE_TO_ERG_G: f64 = 1e10;

// ── Estado químico ────────────────────────────────────────────────────────────

/// Estado de ionización de un elemento de gas (Phase 86).
///
/// Todas las fracciones son adimensionales (número relativo al total de H).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ChemState {
    /// Fracción de HI.
    pub x_hi: f64,
    /// Fracción de HII.
    pub x_hii: f64,
    /// Fracción de HeI.
    pub x_hei: f64,
    /// Fracción de HeII.
    pub x_heii: f64,
    /// Fracción de HeIII.
    pub x_heiii: f64,
    /// Fracción de electrones por átomo de H.
    pub x_e: f64,
    /// Fracción de H⁻ por átomo de H.
    #[serde(default)]
    pub x_hm: f64,
    /// Fracción molecular H₂ por átomo de H.
    #[serde(default)]
    pub x_h2: f64,
    /// Fracción molecular H₂⁺ por átomo de H.
    #[serde(default)]
    pub x_h2p: f64,
    /// Fracción de deuterio neutro D por átomo de H.
    #[serde(default = "default_x_d")]
    pub x_d: f64,
    /// Fracción de deuterio ionizado D⁺ por átomo de H.
    #[serde(default)]
    pub x_dp: f64,
    /// Fracción molecular HD por átomo de H.
    #[serde(default)]
    pub x_hd: f64,
}

fn default_x_d() -> f64 {
    F_D
}

impl ChemState {
    /// Estado inicial completamente neutro.
    pub fn neutral() -> Self {
        Self {
            x_hi: 1.0,
            x_hii: 0.0,
            x_hei: F_HE,
            x_heii: 0.0,
            x_heiii: 0.0,
            x_e: 0.0,
            x_hm: 0.0,
            x_h2: 0.0,
            x_h2p: 0.0,
            x_d: F_D,
            x_dp: 0.0,
            x_hd: 0.0,
        }
    }

    /// Estado completamente ionizado.
    pub fn fully_ionized() -> Self {
        Self {
            x_hi: 0.0,
            x_hii: 1.0,
            x_hei: 0.0,
            x_heii: 0.0,
            x_heiii: F_HE,
            x_e: 1.0 + 2.0 * F_HE + F_D,
            x_hm: 0.0,
            x_h2: 0.0,
            x_h2p: 0.0,
            x_d: 0.0,
            x_dp: F_D,
            x_hd: 0.0,
        }
    }

    /// Aplica conservación de carga y clamp de fracciones a [0, máx].
    pub fn clamp_and_normalize(&mut self) {
        self.x_hm = self.x_hm.clamp(0.0, 1.0);
        self.x_h2 = self.x_h2.clamp(0.0, 0.5);
        self.x_h2p = self.x_h2p.clamp(0.0, 0.5);
        self.x_hd = self.x_hd.clamp(0.0, F_D.min(1.0));

        // D: x_d + x_dp + x_hd = F_D
        self.x_dp = self.x_dp.clamp(0.0, F_D - self.x_hd);
        self.x_d = (F_D - self.x_dp - self.x_hd).max(0.0);

        // H: x_hi + x_hii + x_hm + 2*x_h2 + 2*x_h2p + x_hd = 1
        let molecular_h = self.x_hm + 2.0 * self.x_h2 + 2.0 * self.x_h2p + self.x_hd;
        let atomic_budget = (1.0 - molecular_h).max(0.0);
        self.x_hii = self.x_hii.clamp(0.0, atomic_budget);
        self.x_hi = (atomic_budget - self.x_hii).max(0.0);

        // He: x_hei + x_heii + x_heiii = F_HE
        let he_total = self.x_hei + self.x_heii + self.x_heiii;
        if he_total > 0.0 {
            let scale = F_HE / he_total;
            self.x_hei = (self.x_hei * scale).max(0.0);
            self.x_heii = (self.x_heii * scale).max(0.0);
            self.x_heiii = (self.x_heiii * scale).max(0.0);
        }

        // Neutralidad de carga
        self.x_e = (self.x_hii + self.x_heii + 2.0 * self.x_heiii + self.x_h2p + self.x_dp
            - self.x_hm)
            .max(0.0);
    }

    /// Total de núcleos de hidrógeno en unidades del total de H.
    pub fn hydrogen_nuclei_fraction(&self) -> f64 {
        self.x_hi + self.x_hii + self.x_hm + 2.0 * self.x_h2 + 2.0 * self.x_h2p + self.x_hd
    }

    /// Total de núcleos de deuterio en unidades del total de H.
    pub fn deuterium_nuclei_fraction(&self) -> f64 {
        self.x_d + self.x_dp + self.x_hd
    }

    /// Número de especies químicas explícitas transportadas por [`ChemState`].
    pub const fn species_count() -> usize {
        12
    }

    /// Temperatura del gas dado la energía interna específica `u` [código → K].
    ///
    /// `T = u × (γ-1) × μ × m_p / k_B`
    /// donde μ es el peso molecular medio y el factor de conversión está implícito.
    ///
    /// # Nota de unidades
    /// Para unidades internas de gadget-ng (1 Mpc/h, 10^10 M☉/h, km/s):
    /// `u [km²/s²] → T [K]` ≈ u × 1e10 × (γ-1) × μ × m_p / k_B.
    /// μ ≈ 0.588 (gas totalmente ionizado de H+He al 24% en masa).
    pub fn temperature_from_internal_energy(&self, u_code: f64, gamma: f64) -> f64 {
        const K_B_CGS: f64 = 1.380649e-16; // erg/K
        const M_P_CGS: f64 = 1.672623e-24; // g
        // Peso molecular medio: μ = 4 / (3 + x_hii + x_heii + 2·x_heiii)  (Osterbrock 2006)
        let free_particles_h = self.x_hi + self.x_hii + self.x_hm + self.x_e + self.x_h2p;
        let deuterium_particles = self.x_d + self.x_dp;
        let molecular_particles = self.x_h2 + self.x_hd;
        let helium_particles = self.x_hei + self.x_heii + self.x_heiii;
        let mu = (1.0 + 4.0 * F_HE + 2.0 * F_D)
            / (free_particles_h + deuterium_particles + molecular_particles + helium_particles)
                .max(1e-30);
        let u_cgs = u_code * U_CODE_TO_ERG_G;
        let t = u_cgs * (gamma - 1.0) * mu * M_P_CGS / K_B_CGS;
        t.max(1.0) // temperatura mínima 1 K
    }
}

impl Default for ChemState {
    fn default() -> Self {
        Self::neutral()
    }
}

// ── Tasas de recombinación (Verner & Ferland 1996) ────────────────────────────

/// Tasa de recombinación case-B de HII (cm³/s).
/// Fit: α = 2.753e-14 × (315614/T)^1.5 / (1 + (115188/T)^0.407)^2.242
#[inline]
pub fn alpha_hii(t: f64) -> f64 {
    let t = t.max(1.0);
    let a = 315_614.0 / t;
    2.753e-14 * a.powf(1.5) / (1.0 + (115_188.0 / t).powf(0.407)).powf(2.242)
}

/// Tasa de recombinación case-B de HeII (cm³/s).
/// Fit simplificado: α ≈ 1.26e-14 × (470000/T)^0.75
#[inline]
pub fn alpha_heii(t: f64) -> f64 {
    let t = t.max(1.0);
    1.26e-14 * (470_000.0 / t).powf(0.75)
}

/// Tasa de recombinación case-B de HeIII (cm³/s).
/// Análoga a HII escalada ×4: HeIII + e⁻ → HeII.
#[inline]
pub fn alpha_heiii(t: f64) -> f64 {
    let t = t.max(1.0);
    let a = 1_263_030.0 / t;
    4.0 * 2.753e-14 * a.powf(1.5) / (1.0 + (460_751.0 / t).powf(0.407)).powf(2.242)
}

// ── Tasas de ionización colisional (Cen 1992) ─────────────────────────────────

/// Tasa de ionización colisional de HI (cm³/s).
#[inline]
pub fn beta_hi(t: f64) -> f64 {
    let t = t.max(1.0);
    5.85e-11 * t.sqrt() * (-T_HI / t).exp() / (1.0 + (t / 1e5).sqrt())
}

/// Tasa de ionización colisional de HeI (cm³/s).
#[inline]
pub fn beta_hei(t: f64) -> f64 {
    let t = t.max(1.0);
    2.38e-11 * t.sqrt() * (-T_HEI / t).exp() / (1.0 + (t / 1e5).sqrt())
}

/// Tasa de ionización colisional de HeII (cm³/s).
#[inline]
pub fn beta_heii(t: f64) -> f64 {
    let t = t.max(1.0);
    5.68e-12 * t.sqrt() * (-T_HEII / t).exp() / (1.0 + (t / 1e5).sqrt())
}

/// Formación radiativa de H⁻: H + e⁻ → H⁻ + γ (Abel et al. 1997; fit simplificado).
#[inline]
pub fn k_hm_formation(t: f64) -> f64 {
    let t = t.max(1.0);
    1.4e-18 * t.powf(0.928) * (-t / 16_200.0).exp()
}

/// Asociación asociativa: H⁻ + H → H₂ + e⁻.
#[inline]
pub fn k_h2_from_hm(t: f64) -> f64 {
    let _ = t;
    1.3e-9
}

/// Formación radiativa de H₂⁺: H + H⁺ → H₂⁺ + γ.
#[inline]
pub fn k_h2p_formation(t: f64) -> f64 {
    let t = t.max(1.0);
    1.85e-23 * t.powf(1.8)
}

/// Transferencia de carga: H₂⁺ + H → H₂ + H⁺.
#[inline]
pub fn k_h2_from_h2p(t: f64) -> f64 {
    let _ = t;
    6.4e-10
}

/// Destrucción colisional aproximada de H₂ por H.
#[inline]
pub fn k_h2_collisional_dissociation(t: f64) -> f64 {
    let t = t.max(1.0);
    5.6e-11 * (-52_000.0 / t).exp()
}

/// Charge exchange: D + H⁺ → D⁺ + H.
#[inline]
pub fn k_d_ionization_exchange(t: f64) -> f64 {
    let t = t.max(1.0);
    3.7e-10 * (-43.0 / t).exp()
}

/// Charge exchange inversa: D⁺ + H → D + H⁺.
#[inline]
pub fn k_d_recombination_exchange(t: f64) -> f64 {
    let t = t.max(1.0);
    3.7e-10 * (43.0 / t).exp().min(10.0)
}

/// Formación dominante de HD: D⁺ + H₂ → HD + H⁺.
#[inline]
pub fn k_hd_formation(t: f64) -> f64 {
    let _ = t;
    1.0e-9
}

/// Destrucción aproximada de HD por protones: HD + H⁺ → H₂ + D⁺.
#[inline]
pub fn k_hd_destruction(t: f64) -> f64 {
    let t = t.max(1.0);
    1.0e-9 * (-464.0 / t).exp()
}

// ── Solver implícito subcíclico ───────────────────────────────────────────────

/// Resuelve la red química de no-equilibrio para un paso de tiempo `dt`.
///
/// Usa un solver implícito subcíclico de primer orden (Anninos et al. 1997):
/// - Calcula la escala de tiempo química: `dt_chem = 0.1 / max(|rate|)`.
/// - Itera hasta acumular `dt` completo.
/// - Cada sub-paso aplica actualización implícita linealizada.
///
/// # Argumentos
/// - `state`    — estado de ionización inicial
/// - `gamma_hi` — tasa de fotoionización de HI [1/s], desde el campo de radiación
/// - `gamma_hei`— tasa de fotoionización de HeI [1/s]
/// - `t`        — temperatura del gas [K]
/// - `dt`       — paso de tiempo [s del código]
///
/// # Retorna
/// Estado químico actualizado después de tiempo `dt`.
pub fn solve_chemistry_implicit(
    state: &ChemState,
    gamma_hi: f64,
    gamma_hei: f64,
    t: f64,
    dt: f64,
) -> ChemState {
    let mut st = state.clone();
    let mut t_elapsed = 0.0;

    // Tasas a temperatura fija
    let a_hii = alpha_hii(t);
    let a_heii = alpha_heii(t);
    let a_heiii = alpha_heiii(t);
    let b_hi = beta_hi(t);
    let b_hei = beta_hei(t);
    let b_heii = beta_heii(t);
    let k_hm_form = k_hm_formation(t);
    let k_hm_to_h2 = k_h2_from_hm(t);
    let k_h2p_form = k_h2p_formation(t);
    let k_h2p_to_h2 = k_h2_from_h2p(t);
    let k_h2_diss = k_h2_collisional_dissociation(t);
    let k_d_to_dp = k_d_ionization_exchange(t);
    let k_dp_to_d = k_d_recombination_exchange(t);
    let k_dp_h2_to_hd = k_hd_formation(t);
    let k_hd_to_h2 = k_hd_destruction(t);

    while t_elapsed < dt {
        let xe = st.x_e.max(1e-20);

        // ── Tasas netas para HI/HII ───────────────────────────────────────
        let rate_hii = (gamma_hi + b_hi * xe) * st.x_hi - a_hii * xe * st.x_hii;
        let rate_hei =
            (gamma_hei + b_hei * xe) * st.x_hei - a_heii * xe * st.x_heii - b_heii * xe * st.x_heii
                + a_heiii * xe * st.x_heiii;

        // ── Paso de tiempo de química adaptativo ──────────────────────────
        let h2_form_rate = k_hm_form * st.x_hi * xe + k_h2p_form * st.x_hi * st.x_hii.max(0.0);
        let h2_dest_rate = k_h2_diss * st.x_h2 * st.x_hi.max(0.0);
        let hd_form_rate = k_dp_h2_to_hd * st.x_dp * st.x_h2;
        let hd_dest_rate = k_hd_to_h2 * st.x_hd * st.x_hii.max(0.0);
        let max_rate = rate_hii
            .abs()
            .max(rate_hei.abs())
            .max(h2_form_rate.abs())
            .max(h2_dest_rate.abs())
            .max(hd_form_rate.abs())
            .max(hd_dest_rate.abs())
            .max(1e-30);

        let dt_chem = (0.1 / max_rate).min(dt - t_elapsed).max(1e-20);

        // ── Actualización implícita de primer orden ───────────────────────
        // HII: dx/dt = I_HI - R_HII  →  x_hii_new = (x_hii + dt × I_HI) / (1 + dt × R_HII)
        let i_hi = (gamma_hi + b_hi * xe) * st.x_hi;
        let r_hii_denom = a_hii * xe * dt_chem;
        st.x_hii = (st.x_hii + dt_chem * i_hi) / (1.0 + r_hii_denom);

        // HeII: ionización desde HeI + recombinación desde HeIII
        let i_hei = (gamma_hei + b_hei * xe) * st.x_hei;
        let r_heii = (a_heii + b_heii) * xe;
        st.x_heii =
            (st.x_heii + dt_chem * (i_hei + a_heiii * xe * st.x_heiii)) / (1.0 + dt_chem * r_heii);

        // HeIII: ionización desde HeII
        let i_heii = b_heii * xe * st.x_heii;
        let r_heiii = a_heiii * xe;
        st.x_heiii = (st.x_heiii + dt_chem * i_heii) / (1.0 + dt_chem * r_heiii);

        // Red molecular primordial reducida:
        // H⁻ y H₂⁺ se tratan como intermediarios explícitos con destrucción hacia H₂.
        let hm_create = k_hm_form * st.x_hi * xe;
        let hm_destroy = k_hm_to_h2 * st.x_hi.max(0.0);
        let h2p_create = k_h2p_form * st.x_hi * st.x_hii.max(0.0);
        let h2p_destroy = k_h2p_to_h2 * st.x_hi.max(0.0);
        let hm_next = (st.x_hm + dt_chem * hm_create) / (1.0 + dt_chem * hm_destroy);
        let h2p_next = (st.x_h2p + dt_chem * h2p_create) / (1.0 + dt_chem * h2p_destroy);
        let h2_create = hm_destroy * hm_next + h2p_destroy * h2p_next;
        let h2_destroy = k_h2_diss * st.x_hi.max(0.0);
        st.x_hm = hm_next;
        st.x_h2p = h2p_next;
        st.x_h2 = (st.x_h2 + dt_chem * h2_create) / (1.0 + dt_chem * h2_destroy);

        // Red de deuterio/HD reducida. El deuterio total se conserva en el clamp.
        let dp_create = k_d_to_dp * st.x_d * st.x_hii.max(0.0);
        let dp_destroy = k_dp_to_d * st.x_hi.max(0.0);
        st.x_dp = (st.x_dp + dt_chem * dp_create) / (1.0 + dt_chem * dp_destroy);

        let hd_create = k_dp_h2_to_hd * st.x_dp * st.x_h2;
        let hd_destroy = k_hd_to_h2 * st.x_hii.max(0.0);
        let hd_next = (st.x_hd + dt_chem * hd_create) / (1.0 + dt_chem * hd_destroy);
        let hd_delta = (hd_next - st.x_hd).max(0.0);
        st.x_hd = hd_next;
        st.x_h2 = (st.x_h2 - hd_delta).max(0.0);

        st.clamp_and_normalize();
        t_elapsed += dt_chem;

        // Criterio de convergencia temprana
        if rate_hii.abs() * dt < 1e-6 && rate_hei.abs() * dt < 1e-6 {
            break;
        }
    }

    st
}

// ── Acoplamiento a partículas de gas ─────────────────────────────────────────

/// Parámetros para el módulo de química no-equilibrio.
#[derive(Debug, Clone)]
pub struct ChemParams {
    /// Exponente adiabático γ del gas. Default: 5/3.
    pub gamma: f64,
    /// Número de densidad de hidrógeno de referencia n_H [cm⁻³] para escalar tasas.
    pub n_h_ref: f64,
}

impl Default for ChemParams {
    fn default() -> Self {
        Self {
            gamma: 5.0 / 3.0,
            n_h_ref: 1e-4,
        }
    }
}

/// Actualiza el estado químico y la energía interna de cada partícula de gas.
///
/// Para cada partícula:
/// 1. Obtiene temperatura T desde `internal_energy`.
/// 2. Calcula `gamma_hi` desde el campo de radiación M1.
/// 3. Llama a `solve_chemistry_implicit` para avanzar las fracciones de ionización.
/// 4. Ajusta `internal_energy` por enfriamiento/calentamiento neto.
///
/// Las partículas sin estado químico se inicializan como neutras.
///
/// # Argumentos
/// - `particles` — partículas de gas (se modifican in-place)
/// - `chem_states` — estados de ionización por partícula (longitud = particles.len())
/// - `rad`       — campo de radiación M1 (para Γ_HI)
/// - `params`    — parámetros de química
/// - `dt`        — paso de tiempo [código]
#[cfg(not(feature = "rayon"))]
fn apply_chemistry_impl(
    particles: &mut [Particle],
    chem_states: &mut [ChemState],
    rad: &RadiationField,
    params: &ChemParams,
    dt: f64,
) {
    use crate::m1::M1Params;

    let m1_dummy = M1Params {
        c_red_factor: 100.0,
        kappa_abs: 1.0,
        kappa_scat: 0.0,
        substeps: 1,
        sigma_dust: 0.1,
    };

    assert_eq!(
        particles.len(),
        chem_states.len(),
        "apply_chemistry: particles y chem_states deben tener la misma longitud"
    );

    let box_size = rad.dx * rad.nx as f64;
    let dv = rad.dx.powi(3);

    for (i, p) in particles.iter_mut().enumerate() {
        let st = &mut chem_states[i];

        let u_code = p.internal_energy;
        let t_gas = st.temperature_from_internal_energy(u_code, params.gamma);

        let gamma_hi = photoionization_rate_at_pos(rad, p.position, box_size, &m1_dummy);
        let gamma_hei = 0.0;

        *st = solve_chemistry_implicit(st, gamma_hi, gamma_hei, t_gas, dt);

        let cool_rate = cooling_rate_approx(t_gas, st.x_e, params.n_h_ref)
            + cooling_rate_hd(t_gas, st.x_hd, params.n_h_ref);
        let delta_u = -cool_rate * dt / U_CODE_TO_ERG_G;
        p.internal_energy = (p.internal_energy + delta_u).max(0.0);

        let _ = dv;
    }
}

#[cfg(feature = "rayon")]
fn apply_chemistry_par(
    particles: &mut [Particle],
    chem_states: &mut [ChemState],
    rad: &RadiationField,
    params: &ChemParams,
    dt: f64,
) {
    use crate::m1::M1Params;

    let m1_dummy = M1Params {
        c_red_factor: 100.0,
        kappa_abs: 1.0,
        kappa_scat: 0.0,
        substeps: 1,
        sigma_dust: 0.1,
    };

    assert_eq!(
        particles.len(),
        chem_states.len(),
        "apply_chemistry: particles y chem_states deben tener la misma longitud"
    );

    let box_size = rad.dx * rad.nx as f64;

    particles
        .par_iter_mut()
        .zip(chem_states.par_iter_mut())
        .for_each(|(p, st)| {
            let u_code = p.internal_energy;
            let t_gas = st.temperature_from_internal_energy(u_code, params.gamma);

            let gamma_hi = photoionization_rate_at_pos(rad, p.position, box_size, &m1_dummy);
            let gamma_hei = 0.0;

            *st = solve_chemistry_implicit(st, gamma_hi, gamma_hei, t_gas, dt);

            let cool_rate = cooling_rate_approx(t_gas, st.x_e, params.n_h_ref)
                + cooling_rate_hd(t_gas, st.x_hd, params.n_h_ref);
            let delta_u = -cool_rate * dt / U_CODE_TO_ERG_G;
            p.internal_energy = (p.internal_energy + delta_u).max(0.0);
        });
}

pub fn apply_chemistry(
    particles: &mut [Particle],
    chem_states: &mut [ChemState],
    rad: &RadiationField,
    params: &ChemParams,
    dt: f64,
) {
    #[cfg(feature = "rayon")]
    {
        apply_chemistry_par(particles, chem_states, rad, params, dt);
    }

    #[cfg(not(feature = "rayon"))]
    {
        apply_chemistry_impl(particles, chem_states, rad, params, dt);
    }
}

/// Sección eficaz de HI (cm²) — usada en la tasa de fotoionización.
const SIGMA_HI_CHEM: f64 = 6.3e-18;
/// Energía umbral de ionización de HI (erg).
const H_NU_0_ERG_CHEM: f64 = 2.179e-11;

/// Tasa de fotoionización de HI en la posición de una partícula.
fn photoionization_rate_at_pos(
    rad: &RadiationField,
    pos: gadget_ng_core::Vec3,
    box_size: f64,
    params: &crate::m1::M1Params,
) -> f64 {
    use crate::m1::C_KMS;

    // Índice de celda más cercana
    let ix = ((pos.x / box_size * rad.nx as f64) as usize).min(rad.nx - 1);
    let iy = ((pos.y / box_size * rad.ny as f64) as usize).min(rad.ny - 1);
    let iz = ((pos.z / box_size * rad.nz as f64) as usize).min(rad.nz - 1);

    let idx = rad.idx(ix, iy, iz);
    let e_uv = rad.energy_density[idx].max(0.0);

    // Γ_HI = σ_HI × c_red × E_UV / (h·ν_0)
    let c_red = C_KMS * 1e5 / params.c_red_factor;
    SIGMA_HI_CHEM * c_red * e_uv / H_NU_0_ERG_CHEM
}

/// Tasa de enfriamiento atómico aproximada [erg cm⁻³ s⁻¹ / (n_H²)].
///
/// Combinación de bremsstrahlung + excitación colisional + recombinación.
/// Fit analítico basado en Katz et al. (1996).
#[inline]
pub fn cooling_rate_approx(t: f64, x_e: f64, n_h: f64) -> f64 {
    let t = t.max(1.0);
    // Bremsstrahlung (siempre activo)
    let brems = 1.42e-27 * t.sqrt() * n_h * n_h * x_e;
    // Excitación de Lyα (pico ~1e4 K)
    let lya = 7.5e-19 * (-118_348.0 / t).exp() / (1.0 + (t / 1e5).sqrt()) * n_h * n_h * x_e;
    brems + lya
}

/// Cooling molecular HD aproximado [erg cm⁻³ s⁻¹].
///
/// El dipolo permanente de HD lo vuelve eficiente por debajo de `10^4 K`.
/// Esta forma es deliberadamente suave para tests y runs pequeños: escala como
/// `n_H² x_HD`, tiene activación rotacional cerca de 128 K y se apaga a T muy alta.
#[inline]
pub fn cooling_rate_hd(t: f64, x_hd: f64, n_h: f64) -> f64 {
    let t = t.max(1.0);
    let x_hd = x_hd.max(0.0);
    let n_h = n_h.max(0.0);
    let rotational = (-128.0 / t).exp();
    let low_t_boost = (t / 100.0).powf(2.0) / (1.0 + (t / 1.0e4).powf(2.0));
    1.0e-23 * n_h * n_h * x_hd * rotational * low_t_boost
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neutral_state_conserves_h() {
        let st = ChemState::neutral();
        assert!((st.hydrogen_nuclei_fraction() - 1.0).abs() < 1e-12);
        assert!((st.deuterium_nuclei_fraction() - F_D).abs() < 1e-15);
        assert!((st.x_hei + st.x_heii + st.x_heiii - F_HE).abs() < 1e-12);
    }

    #[test]
    fn chem_state_tracks_twelve_species() {
        assert_eq!(ChemState::species_count(), 12);
        let st = ChemState::neutral();
        assert_eq!(st.x_hm, 0.0);
        assert_eq!(st.x_h2, 0.0);
        assert_eq!(st.x_h2p, 0.0);
        assert_eq!(st.x_d, F_D);
        assert_eq!(st.x_dp, 0.0);
        assert_eq!(st.x_hd, 0.0);
    }

    #[test]
    fn fully_ionized_conserves_charge() {
        let st = ChemState::fully_ionized();
        let x_e_expected = st.x_hii + st.x_heii + 2.0 * st.x_heiii + st.x_dp;
        assert!((st.x_e - x_e_expected).abs() < 1e-12);
    }

    #[test]
    fn recombination_rates_positive() {
        for t in &[1e3_f64, 1e4, 1e5, 1e6] {
            assert!(alpha_hii(*t) > 0.0, "alpha_hii({t}) debe ser positivo");
            assert!(alpha_heii(*t) > 0.0);
            assert!(alpha_heiii(*t) > 0.0);
        }
    }

    #[test]
    fn ionization_rates_positive() {
        for t in &[1e4_f64, 1e5, 1e6] {
            assert!(beta_hi(*t) > 0.0, "beta_hi({t}) debe ser positivo");
            assert!(beta_hei(*t) > 0.0);
            assert!(beta_heii(*t) > 0.0);
        }
    }

    #[test]
    fn molecular_rates_positive() {
        for t in &[1e3_f64, 1e4, 1e5] {
            assert!(k_hm_formation(*t) > 0.0);
            assert!(k_h2_from_hm(*t) > 0.0);
            assert!(k_h2p_formation(*t) > 0.0);
            assert!(k_h2_from_h2p(*t) > 0.0);
            assert!(k_h2_collisional_dissociation(*t) >= 0.0);
            assert!(k_d_ionization_exchange(*t) > 0.0);
            assert!(k_d_recombination_exchange(*t) > 0.0);
            assert!(k_hd_formation(*t) > 0.0);
            assert!(k_hd_destruction(*t) >= 0.0);
        }
    }

    #[test]
    fn alpha_hii_decreases_with_temperature() {
        // Recombinación se enfría a temperaturas altas
        let a_lo = alpha_hii(1e3);
        let a_hi = alpha_hii(1e6);
        assert!(a_lo > a_hi, "α_HII debe disminuir con T");
    }

    #[test]
    fn beta_hi_increases_with_temperature() {
        // Ionización colisional aumenta con T (hasta un máximo)
        let b_lo = beta_hi(1e4);
        let b_hi = beta_hi(1e5);
        assert!(b_hi > b_lo, "β_HI debe aumentar con T (T<T_HI)");
    }

    #[test]
    fn solve_chemistry_neutral_no_photons_stays_neutral() {
        let st = ChemState::neutral();
        let t = 1e3; // temperatura baja: pocas ionizaciones colisionales
        let result = solve_chemistry_implicit(&st, 0.0, 0.0, t, 1e-5);
        // Con T muy baja y sin fotones, debe permanecer casi neutro
        assert!(
            result.x_hi > 0.9,
            "HI debe permanecer neutro: x_hi = {}",
            result.x_hi
        );
    }

    #[test]
    fn solve_chemistry_ionized_recombines() {
        let st = ChemState::fully_ionized();
        let t = 1e4; // temperatura de recombinación eficiente
        // Sin fotones, debe recombinar con el tiempo
        let result = solve_chemistry_implicit(&st, 0.0, 0.0, t, 1e10);
        assert!(
            result.x_hi > st.x_hi,
            "HI debe crecer por recombinación: {} → {}",
            st.x_hi,
            result.x_hi
        );
    }

    #[test]
    fn solve_chemistry_fractions_conserved() {
        let st = ChemState::neutral();
        let result = solve_chemistry_implicit(&st, 1e-13, 0.0, 1e4, 1e6);
        // H conservado: x_hi + x_hii ≈ 1
        let h_total = result.hydrogen_nuclei_fraction();
        assert!((h_total - 1.0).abs() < 0.05, "H no conservado: {h_total}");
        // He conservado: x_hei + x_heii + x_heiii ≈ F_HE
        let he_total = result.x_hei + result.x_heii + result.x_heiii;
        assert!(
            (he_total - F_HE).abs() < 0.01 * F_HE + 1e-10,
            "He no conservado: {he_total}"
        );
        let d_total = result.deuterium_nuclei_fraction();
        assert!((d_total - F_D).abs() < 1e-10, "D no conservado: {d_total}");
    }

    #[test]
    fn solve_chemistry_can_form_trace_h2() {
        let mut st = ChemState::neutral();
        st.x_e = 1e-4;
        let result = solve_chemistry_implicit(&st, 0.0, 0.0, 3e3, 1e12);
        assert!(
            result.x_h2 >= st.x_h2,
            "H2 no debe decrecer desde estado primordial sin fotodisociación"
        );
        assert!((result.hydrogen_nuclei_fraction() - 1.0).abs() < 1e-8);
    }

    #[test]
    fn solve_chemistry_can_form_trace_hd() {
        let mut st = ChemState::neutral();
        st.x_h2 = 1e-3;
        st.x_hi = 1.0 - 2.0 * st.x_h2;
        st.x_hii = 1e-4;
        st.x_dp = F_D * 0.5;
        st.x_d = F_D - st.x_dp;
        st.x_e = st.x_hii + st.x_dp;
        let result = solve_chemistry_implicit(&st, 0.0, 0.0, 200.0, 1e10);
        assert!(
            result.x_hd > st.x_hd,
            "HD debe crecer desde gas con H2 y D+"
        );
        assert!((result.deuterium_nuclei_fraction() - F_D).abs() < 1e-10);
        assert!((result.hydrogen_nuclei_fraction() - 1.0).abs() < 1e-8);
    }

    #[test]
    fn high_uv_field_ionizes_hydrogen() {
        let st = ChemState::neutral();
        let t = 1e4;
        // Campo UV fuerte → ionización rápida
        let result = solve_chemistry_implicit(&st, 1e-10, 0.0, t, 1e12);
        assert!(
            result.x_hii > 0.5,
            "Con UV fuerte HII debe superar 50%: x_hii = {}",
            result.x_hii
        );
    }

    #[test]
    fn temperature_from_internal_energy_reasonable() {
        let st = ChemState::neutral();
        // u = 1.0 (km²/s² × 10^10) → T esperada O(10^4) K (o más baja para gas neutro frío)
        let t = st.temperature_from_internal_energy(1.0, 5.0 / 3.0);
        assert!(t > 10.0, "T debe ser > 10 K: {t}");
        assert!(t < 1e9, "T debe ser < 10^9 K: {t}");
    }

    #[test]
    fn cooling_rate_positive() {
        let rate = cooling_rate_approx(1e4, 0.5, 1e-4);
        assert!(rate >= 0.0, "Tasa de enfriamiento debe ser ≥ 0: {rate}");
    }

    #[test]
    fn hd_cooling_positive_and_temperature_dependent() {
        let cold = cooling_rate_hd(50.0, 1e-3, 1e-6);
        let warm = cooling_rate_hd(200.0, 1e-3, 1e-6);
        assert!(cold >= 0.0);
        assert!(
            warm > cold,
            "HD cooling debe activarse sobre escala rotacional"
        );
    }

    #[test]
    fn clamp_normalize_prevents_negative() {
        let mut st = ChemState {
            x_hi: -0.1,
            x_hii: 1.1,
            x_hei: -0.01,
            x_heii: 0.08,
            x_heiii: 0.005,
            x_e: 0.5,
            x_hm: -0.1,
            x_h2: 2.0,
            x_h2p: -0.2,
            x_d: -1.0,
            x_dp: 1.0,
            x_hd: 1.0,
        };
        st.clamp_and_normalize();
        assert!(st.x_hi >= 0.0, "x_hi negativo tras clamp");
        assert!(st.x_hii >= 0.0 && st.x_hii <= 1.0);
        assert!(st.x_hei >= 0.0);
        assert!(st.x_e >= 0.0);
        assert!((st.deuterium_nuclei_fraction() - F_D).abs() < 1e-12);
    }
}
