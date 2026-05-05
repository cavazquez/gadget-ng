use serde::{Deserialize, Serialize};

use crate::cosmology::NeutrinoHierarchyKind;


/// Parámetros cosmológicos (opcional; retrocompatible: `enabled = false`).
///
/// Activa la integración del factor de escala `a(t)` junto a las partículas.
/// Con `enabled = false` (default) el motor usa `dt` plano para drift y kick,
/// sin ninguna corrección cosmológica.
///
/// ## Unidades
///
/// `h0` es H₀ en **unidades internas de tiempo** (1/t_sim). Para simulaciones
/// cosmológicas en unidades naturales (L=Mpc/h, M=10¹⁰ M☉/h, V=km/s) el valor
/// habitual es `h0 ≈ 0.1` (≈ H₀ en unidades de km/s/kpc).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CosmologySection {
    /// `false` (default) → integración Newtoniana plana (sin factor de escala).
    /// `true` → integrar Friedmann y usar factores drift/kick cosmológicos.
    #[serde(default)]
    pub enabled: bool,
    /// Condiciones de contorno periódicas (Fase 18).
    ///
    /// `false` (default) → caja no periódica. Las fuerzas usan árbol Barnes-Hut o
    ///   distancias euclídeas sin imagen mínima; las posiciones no se envuelven.
    ///
    /// `true` → caja periódica. Requiere `gravity.solver = "pm"` o `"tree_pm"`:
    ///   el solver PM usa CIC+FFT periódica para calcular las fuerzas correctamente.
    ///   Las posiciones se envuelven a `[0, box_size)` tras cada paso de drift.
    ///   Las fuerzas de árbol (BarnesHut) NO son periódicas; usar PM o TreePM.
    #[serde(default)]
    pub periodic: bool,
    /// Fracción de densidad de materia (sin dimensiones). Default: 0.3.
    #[serde(default = "default_omega_m")]
    pub omega_m: f64,
    /// Fracción de energía oscura (sin dimensiones). Default: 0.7.
    #[serde(default = "default_omega_lambda")]
    pub omega_lambda: f64,
    /// H₀ en unidades internas (1/t_sim). Default: 0.1.
    #[serde(default = "default_h0")]
    pub h0: f64,
    /// Factor de escala inicial. Default: 1.0 (z=0).
    /// Para simulaciones de alta redshift, p. ej. z=49 → `a_init = 0.02`.
    #[serde(default = "default_a_init")]
    pub a_init: f64,
    /// Si `true`, el motor calcula automáticamente G a partir de `omega_m` y `h0`
    /// usando la condición de Friedmann `G = 3·Ω_m·H₀²/(8π)` (ρ̄_m = 1).
    ///
    /// Requiere `enabled = true`. Cuando está activo, el campo
    /// `simulation.gravitational_constant` se ignora para el modo cosmológico
    /// y se emite un `info!` con el valor calculado. Si el campo
    /// `simulation.gravitational_constant` difiere del G auto en más de 1 %,
    /// también se emite un `warn!` de inconsistencia.
    ///
    /// Default: `false` (retrocompatible — se usa `simulation.gravitational_constant`).
    #[serde(default)]
    pub auto_g: bool,
    /// Parámetro CPL w₀ para energía oscura dinámica (Phase 155).
    /// `w(a) = w0 + wa*(1-a)`. Default: -1.0 (ΛCDM).
    #[serde(default = "default_w0")]
    pub w0: f64,
    /// Parámetro CPL wₐ para energía oscura dinámica (Phase 155).
    /// Default: 0.0 (ΛCDM).
    #[serde(default)]
    pub wa: f64,
    /// Suma de masas de neutrinos en eV (Phase 156).
    /// `Ω_ν = m_ν / (93.14 eV × h²)`. Default: 0.0 (sin neutrinos).
    #[serde(default)]
    pub m_nu_ev: f64,
    /// Reparto fenomenológico de Σm_ν (referencia `split_m_nu_ev` en cosmología).
    #[serde(default)]
    pub neutrino_hierarchy: NeutrinoHierarchyKind,
}

fn default_omega_m() -> f64 {
    0.3
}
fn default_omega_lambda() -> f64 {
    0.7
}
fn default_h0() -> f64 {
    0.1
}
fn default_a_init() -> f64 {
    1.0
}
fn default_w0() -> f64 {
    -1.0
}

impl Default for CosmologySection {
    fn default() -> Self {
        Self {
            enabled: false,
            periodic: false,
            omega_m: default_omega_m(),
            omega_lambda: default_omega_lambda(),
            h0: default_h0(),
            a_init: default_a_init(),
            auto_g: false,
            w0: default_w0(),
            wa: 0.0,
            m_nu_ev: 0.0,
            neutrino_hierarchy: NeutrinoHierarchyKind::default(),
        }
    }
}

// ── Sistema de unidades físicas ───────────────────────────────────────────────

/// G en kpc Msun⁻¹ (km/s)² (NIST 2018 redondeado a 5 cifras).
pub const G_KPC_MSUN_KMPS: f64 = 4.3009e-6;

/// Sistema de unidades físicas (opcional; retrocompatible: `enabled = false`).
///
/// Cuando `enabled = true`, la constante gravitacional interna se calcula
/// automáticamente como
///
/// ```text
/// G_int = G_kpc × mass_in_msun / length_in_kpc / velocity_in_km_s²
/// ```
///
/// donde `G_kpc = 4.3009 × 10⁻⁶ kpc Msun⁻¹ (km/s)²`.
///
/// # Ejemplo TOML
/// ```toml
/// [units]
/// enabled        = true
/// length_in_kpc  = 1.0       # 1 u.l. = 1 kpc
/// mass_in_msun   = 1.0e10    # 1 u.m. = 10¹⁰ M☉  (unidades GADGET clásicas)
/// velocity_in_km_s = 1.0     # 1 u.v. = 1 km/s
/// # G_int calculado = 4.3009e-6 × 1e10 / 1 / 1 = 4.3009e4
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnitsSection {
    /// `true` → calcular G internamente a partir de las escalas de unidad.
    /// `false` (default) → usar `simulation.gravitational_constant` sin cambios.
    #[serde(default)]
    pub enabled: bool,
    /// 1 unidad interna de longitud equivale a este número de kpc.
    #[serde(default = "default_unit_one")]
    pub length_in_kpc: f64,
    /// 1 unidad interna de masa equivale a este número de masas solares.
    #[serde(default = "default_unit_one")]
    pub mass_in_msun: f64,
    /// 1 unidad interna de velocidad equivale a este número de km/s.
    #[serde(default = "default_unit_one")]
    pub velocity_in_km_s: f64,
}

fn default_unit_one() -> f64 {
    1.0
}

impl Default for UnitsSection {
    fn default() -> Self {
        Self {
            enabled: false,
            length_in_kpc: 1.0,
            mass_in_msun: 1.0,
            velocity_in_km_s: 1.0,
        }
    }
}

impl UnitsSection {
    /// G en unidades internas calculado a partir de las escalas.
    pub fn compute_g(&self) -> f64 {
        G_KPC_MSUN_KMPS * self.mass_in_msun
            / self.length_in_kpc
            / (self.velocity_in_km_s * self.velocity_in_km_s)
    }

    /// Unidad de tiempo interna expresada en Gyr
    /// (1 kpc / (km/s) = 0.97779 Gyr).
    pub fn time_unit_in_gyr(&self) -> f64 {
        0.97779 * self.length_in_kpc / self.velocity_in_km_s
    }

    /// Hubble time en unidades internas dado H₀ en km/s/Mpc.
    pub fn hubble_time(&self, h0_km_s_mpc: f64) -> f64 {
        // 1 Mpc = 1000 kpc; t_H = 1/H₀
        // H₀ en unidades internas = h0_km_s_mpc × (velocity_in_km_s / (1000 × length_in_kpc))
        let h0_int = h0_km_s_mpc * self.velocity_in_km_s / (1000.0 * self.length_in_kpc);
        1.0 / h0_int
    }
}
