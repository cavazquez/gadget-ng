use serde::{Deserialize, Serialize};

// ── SphSection ───────────────────────────────────────────────────────────────

/// Tipo de enfriamiento radiativo para partículas de gas.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum CoolingKind {
    /// Sin enfriamiento (default).
    #[default]
    None,
    /// Enfriamiento atómico H+He: `Λ(T) = Λ₀ · T^β` con floor en `t_floor_k`.
    AtomicHHe,
    /// Enfriamiento con contribución metálica (Phase 111).
    ///
    /// `Λ(T, Z) = Λ_HHe(T) + (Z/Z_sun) × Λ_metal(T)` — Sutherland & Dopita (1993).
    MetalCooling,
    /// Enfriamiento tabulado con interpolación bilineal (Phase 119).
    ///
    /// Tabla interna 7×20 derivada de Sutherland & Dopita (1993): 7 bins en Z/Z_sun
    /// y 20 bins en log10(T) de 4.0 a 8.5. Interpolación bilineal en (Z, log T).
    MetalTabular,
}

/// Configuración del módulo SPH cosmológico (Phase 66).
///
/// Añadir `[sph]` al TOML:
///
/// ```toml
/// [sph]
/// enabled       = true
/// gamma         = 1.6667
/// alpha_visc    = 1.0
/// n_neigh       = 32
/// cooling       = "atomic_h_he"
/// t_floor_k     = 1e4
/// gas_fraction  = 0.1
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SphSection {
    /// Activa el módulo SPH (default: `false`).
    #[serde(default)]
    pub enabled: bool,
    /// Índice adiabático γ (default: 5/3).
    #[serde(default = "default_gamma")]
    pub gamma: f64,
    /// Parámetro de viscosidad artificial α (default: 1.0).
    #[serde(default = "default_alpha_visc")]
    pub alpha_visc: f64,
    /// Número objetivo de vecinos SPH (default: 32).
    #[serde(default = "default_n_neigh")]
    pub n_neigh: usize,
    /// Tipo de enfriamiento radiativo.
    #[serde(default)]
    pub cooling: CoolingKind,
    /// Temperatura de floor en Kelvin (default: 10⁴ K).
    #[serde(default = "default_t_floor_k")]
    pub t_floor_k: f64,
    /// Fracción de partículas inicializadas como gas (default: 0.0).
    #[serde(default)]
    pub gas_fraction: f64,
    /// Configuración del feedback estelar (Phase 78).
    #[serde(default)]
    pub feedback: FeedbackSection,
    /// Configuración del feedback AGN (Phase 96).
    #[serde(default)]
    pub agn: AgnSection,
    /// Configuración del enriquecimiento químico metálico (Phase 109).
    #[serde(default)]
    pub enrichment: EnrichmentSection,
    /// Configuración del modelo ISM multifase (Phase 114).
    #[serde(default)]
    pub ism: IsmSection,
    /// Configuración del módulo de rayos cósmicos (Phase 117).
    #[serde(default)]
    pub cr: CrSection,
    /// Configuración de la conducción térmica del ICM (Phase 121).
    #[serde(default)]
    pub conduction: ConductionSection,
    /// Configuración del gas molecular H₂ (Phase 122).
    #[serde(default)]
    pub molecular: MolecularSection,
    /// Configuración del polvo intersticial (Phase 130).
    #[serde(default)]
    pub dust: DustSection,
    /// Factor de supresión del cooling por campo magnético (Phase 134).
    ///
    /// `Λ_eff = Λ(T) / (1 + mag_suppress_cooling / β)`.
    /// `0.0` = sin supresión magnética (default).
    #[serde(default)]
    pub mag_suppress_cooling: f64,
}

fn default_gamma() -> f64 {
    5.0 / 3.0
}
fn default_alpha_visc() -> f64 {
    1.0
}
fn default_n_neigh() -> usize {
    32
}
fn default_t_floor_k() -> f64 {
    1e4
}

impl Default for SphSection {
    fn default() -> Self {
        Self {
            enabled: false,
            gamma: default_gamma(),
            alpha_visc: default_alpha_visc(),
            n_neigh: default_n_neigh(),
            cooling: CoolingKind::None,
            t_floor_k: default_t_floor_k(),
            gas_fraction: 0.0,
            feedback: FeedbackSection::default(),
            agn: AgnSection::default(),
            enrichment: EnrichmentSection::default(),
            ism: IsmSection::default(),
            cr: CrSection::default(),
            conduction: ConductionSection::default(),
            molecular: MolecularSection::default(),
            dust: DustSection::default(),
            mag_suppress_cooling: 0.0,
        }
    }
}

/// Configuración del feedback estelar por supernovas (Phase 78).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackSection {
    /// Activa el feedback estelar (default: `false`).
    #[serde(default)]
    pub enabled: bool,
    /// Velocidad del kick de SN en km/s (default: 350 km/s).
    #[serde(default = "default_v_kick")]
    pub v_kick_km_s: f64,
    /// Eficiencia de energía SN ε_SN: fracción de E_SN=10⁵¹ erg transferida (default: 0.1).
    #[serde(default = "default_eps_sn")]
    pub eps_sn: f64,
    /// Densidad umbral para SFR en unidades internas (default: 0.1).
    #[serde(default = "default_rho_sf")]
    pub rho_sf: f64,
    /// SFR mínima para activar el feedback (default: 1e-4 en unidades internas).
    #[serde(default = "default_sfr_min")]
    pub sfr_min: f64,
    /// Vientos galácticos (Phase 108). Default: desactivado.
    #[serde(default)]
    pub wind: WindParams,
    /// Fracción de masa que se convierte en estrella al hacer spawning (Phase 112).
    /// Default: 0.5 (el gas padre retiene el 50% de su masa).
    #[serde(default = "default_m_star_fraction")]
    pub m_star_fraction: f64,
    /// Masa mínima de gas para que la partícula no se elimine tras el spawning (Phase 112).
    /// Partículas con masa < m_gas_min se marcan para eliminación.
    #[serde(default = "default_m_gas_min")]
    pub m_gas_min: f64,
    /// Normalización de la DTD de SN Ia: A_Ia [SN / Gyr / M_sun] (Phase 113).
    /// Basado en Maoz & Mannucci (2012): ~2 × 10⁻³ SN/Gyr/M_sun.
    #[serde(default = "default_a_ia")]
    pub a_ia: f64,
    /// Tiempo mínimo para SN Ia tras la formación estelar [Gyr] (Phase 113).
    /// Default: 0.1 Gyr.
    #[serde(default = "default_t_ia_min_gyr")]
    pub t_ia_min_gyr: f64,
    /// Energía inyectada por SN Ia en unidades internas (Phase 113).
    /// E_Ia ≈ 1e51 erg × 1.3 en unidades gadget-ng.
    #[serde(default = "default_e_ia_code")]
    pub e_ia_code: f64,
    /// Activa el feedback mecánico de vientos estelares pre-SN (Phase 115).
    /// Modela vientos de estrellas OB y Wolf-Rayet (~10-30 Myr antes de SN II).
    #[serde(default)]
    pub stellar_wind_enabled: bool,
    /// Velocidad terminal del viento estelar [km/s] (Phase 115). Default: 2000 km/s.
    #[serde(default = "default_v_stellar_wind")]
    pub v_stellar_wind_km_s: f64,
    /// Factor de carga másica del viento estelar η_w = Ṁ_wind / SFR (Phase 115). Default: 0.1.
    #[serde(default = "default_eta_stellar_wind")]
    pub eta_stellar_wind: f64,
}

/// Parámetros del modelo de vientos galácticos (Phase 108).
///
/// Basado en la prescripción de Springel & Hernquist (2003) MNRAS 339, 289.
///
/// La velocidad del viento es `v_wind_km_s` y el factor de carga de masa `η`
/// determina cuánta masa se eyecta por unidad de masa estelar formada.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindParams {
    /// Activa los vientos galácticos (default: `false`).
    #[serde(default)]
    pub enabled: bool,
    /// Velocidad terminal del viento en km/s (default: 480 km/s ≈ 2× v_SN).
    #[serde(default = "default_v_wind")]
    pub v_wind_km_s: f64,
    /// Factor de carga de masa η = Ṁ_wind / SFR (default: 2.0).
    #[serde(default = "default_mass_loading")]
    pub mass_loading: f64,
    /// Tiempo de desacoplamiento hidrológico en Myr (default: 0.0 = sin desacoplamiento).
    #[serde(default)]
    pub t_decoupling_myr: f64,
}

fn default_v_wind() -> f64 {
    480.0
}
fn default_mass_loading() -> f64 {
    2.0
}

impl Default for WindParams {
    fn default() -> Self {
        Self {
            enabled: false,
            v_wind_km_s: default_v_wind(),
            mass_loading: default_mass_loading(),
            t_decoupling_myr: 0.0,
        }
    }
}

fn default_v_kick() -> f64 {
    350.0
}
fn default_eps_sn() -> f64 {
    0.1
}
fn default_rho_sf() -> f64 {
    0.1
}
fn default_sfr_min() -> f64 {
    1e-4
}
fn default_m_star_fraction() -> f64 {
    0.5
}
fn default_m_gas_min() -> f64 {
    0.01
}
fn default_a_ia() -> f64 {
    2e-3
}
fn default_t_ia_min_gyr() -> f64 {
    0.1
}
fn default_e_ia_code() -> f64 {
    1.54e-3 * 1.3
} // E_Ia ≈ 1.3 × E_SN
fn default_v_stellar_wind() -> f64 {
    2000.0
}
fn default_eta_stellar_wind() -> f64 {
    0.1
}

impl Default for FeedbackSection {
    fn default() -> Self {
        Self {
            enabled: false,
            v_kick_km_s: default_v_kick(),
            eps_sn: default_eps_sn(),
            rho_sf: default_rho_sf(),
            sfr_min: default_sfr_min(),
            wind: WindParams::default(),
            m_star_fraction: default_m_star_fraction(),
            m_gas_min: default_m_gas_min(),
            a_ia: default_a_ia(),
            t_ia_min_gyr: default_t_ia_min_gyr(),
            e_ia_code: default_e_ia_code(),
            stellar_wind_enabled: false,
            v_stellar_wind_km_s: default_v_stellar_wind(),
            eta_stellar_wind: default_eta_stellar_wind(),
        }
    }
}

/// Configuración del enriquecimiento químico metálico (Phase 109).
///
/// Controla los yields de SN II y AGB usados en `apply_enrichment`.
///
/// ```toml
/// [sph.enrichment]
/// enabled    = true
/// yield_snii = 0.02   # fracción de masa en metales eyectada por SN II
/// yield_agb  = 0.04   # fracción de masa en metales eyectada por AGB
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnrichmentSection {
    /// Activa el enriquecimiento químico (default: `false`).
    #[serde(default)]
    pub enabled: bool,
    /// Yield metálico de SN II: fracción de masa en Z eyectada (default: 0.02).
    #[serde(default = "default_yield_snii")]
    pub yield_snii: f64,
    /// Yield metálico de AGB: fracción de masa en Z eyectada (default: 0.04).
    #[serde(default = "default_yield_agb")]
    pub yield_agb: f64,
}

fn default_yield_snii() -> f64 {
    0.02
}
fn default_yield_agb() -> f64 {
    0.04
}

impl Default for EnrichmentSection {
    fn default() -> Self {
        Self {
            enabled: false,
            yield_snii: default_yield_snii(),
            yield_agb: default_yield_agb(),
        }
    }
}

/// Configuración del modelo ISM multifase fría-caliente (Phase 114).
///
/// Basado en el modelo de presión efectiva de Springel & Hernquist (2003).
///
/// ```toml
/// [sph.ism]
/// enabled = true
/// q_star  = 2.5
/// f_cold  = 0.5
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IsmSection {
    /// Activa el modelo ISM multifase (default: `false`).
    #[serde(default)]
    pub enabled: bool,
    /// Parámetro de escala de presión efectiva q* (default: 2.5).
    /// Controla la rigidez del ISM frío: P_eff = (γ-1) ρ (u + q_star × u_cold).
    #[serde(default = "default_q_star")]
    pub q_star: f64,
    /// Fracción fría inicial del ISM (default: 0.5).
    #[serde(default = "default_f_cold")]
    pub f_cold: f64,
}

fn default_q_star() -> f64 {
    2.5
}
fn default_f_cold() -> f64 {
    0.5
}

impl Default for IsmSection {
    fn default() -> Self {
        Self {
            enabled: false,
            q_star: default_q_star(),
            f_cold: default_f_cold(),
        }
    }
}

/// Configuración del gas molecular simple HI → H₂ (Phase 122).
///
/// ```toml
/// [sph.molecular]
/// enabled          = true
/// rho_h2_threshold = 100.0
/// sfr_h2_boost     = 2.0
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MolecularSection {
    /// Activa el módulo de gas molecular (default: `false`).
    #[serde(default)]
    pub enabled: bool,
    /// Densidad umbral de formación de H₂ en unidades internas (default: `100.0`).
    #[serde(default = "default_rho_h2_threshold")]
    pub rho_h2_threshold: f64,
    /// Factor multiplicativo de SFR en gas molecular (default: `2.0`).
    #[serde(default = "default_sfr_h2_boost")]
    pub sfr_h2_boost: f64,
}

fn default_rho_h2_threshold() -> f64 {
    100.0
}
fn default_sfr_h2_boost() -> f64 {
    2.0
}

impl Default for MolecularSection {
    fn default() -> Self {
        Self {
            enabled: false,
            rho_h2_threshold: default_rho_h2_threshold(),
            sfr_h2_boost: default_sfr_h2_boost(),
        }
    }
}

/// Configuración de la conducción térmica del ICM (Phase 121).
///
/// Modelo de Spitzer (1962) con factor de supresión ψ por campo B / turbulencia.
///
/// ```toml
/// [sph.conduction]
/// enabled        = true
/// kappa_spitzer  = 1e-4
/// psi_suppression = 0.1
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConductionSection {
    /// Activa la conducción térmica (default: `false`).
    #[serde(default)]
    pub enabled: bool,
    /// Conductividad Spitzer en unidades internas (default: `1e-4`).
    #[serde(default = "default_kappa_spitzer")]
    pub kappa_spitzer: f64,
    /// Factor de supresión ψ ∈ [0,1] por campo magnético o turbulencia (default: `0.1`).
    #[serde(default = "default_psi_suppression")]
    pub psi_suppression: f64,
    /// Activa conducción anisótropa ∥B (Phase 133). Si `false` usa Spitzer isótropo.
    #[serde(default)]
    pub anisotropic: bool,
    /// Conductividad paralela a B (Phase 133). Default: igual a `kappa_spitzer`.
    #[serde(default = "default_kappa_par")]
    pub kappa_par: f64,
    /// Conductividad perpendicular a B (Phase 133). Default: `1e-6` (muy suprimida).
    #[serde(default = "default_kappa_perp")]
    pub kappa_perp: f64,
}

fn default_kappa_spitzer() -> f64 {
    1e-4
}
fn default_psi_suppression() -> f64 {
    0.1
}
fn default_kappa_par() -> f64 {
    1e-4
}
fn default_kappa_perp() -> f64 {
    1e-6
}

impl Default for ConductionSection {
    fn default() -> Self {
        Self {
            enabled: false,
            kappa_spitzer: default_kappa_spitzer(),
            psi_suppression: default_psi_suppression(),
            anisotropic: false,
            kappa_par: default_kappa_par(),
            kappa_perp: default_kappa_perp(),
        }
    }
}

/// Configuración del módulo de rayos cósmicos (Phase 117).
///
/// ```toml
/// [sph.cr]
/// enabled     = true
/// cr_fraction = 0.1
/// kappa_cr    = 3e-3
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrSection {
    /// Activa el módulo de rayos cósmicos (default: `false`).
    #[serde(default)]
    pub enabled: bool,
    /// Fracción de energía de SN II que va a CRs (default: 0.1).
    #[serde(default = "default_cr_fraction")]
    pub cr_fraction: f64,
    /// Coeficiente de difusión isótropa κ_CR [unidades internas] (default: 3e-3).
    #[serde(default = "default_kappa_cr")]
    pub kappa_cr: f64,
    /// Factor de supresión de difusión CR por campo magnético (Phase 129).
    ///
    /// `f_suppress = 1 / (1 + b_cr_suppress × |B|²)`.
    /// `1.0` → supresión moderada; `0.0` → sin supresión (recupera comportamiento antiguo).
    #[serde(default = "default_b_cr_suppress")]
    pub b_cr_suppress: f64,
    /// Usa `gadget_ng_mhd::diffuse_cr_anisotropic` en lugar de difusión isótropa (Phase 133).
    /// Requiere `[mhd] enabled = true` y campo B en las partículas.
    #[serde(default)]
    pub anisotropic_diffusion: bool,
    /// Pérdidas hadrónicas ~ `e_cr × exp(−coeff × ρ × dt)`; `0.0` desactiva (roadmap Fase A).
    #[serde(default)]
    pub hadronic_loss_coeff: f64,
    /// Coeficiente de streaming lungo-B para rayos cósmicos (Phase 170). Default: `0.0` (desactivado).
    ///
    /// `streaming_coefficient > 0` activa `streaming_crk` + `cr_pressure_backreaction`.
    #[serde(default)]
    pub streaming_coefficient: f64,
}

fn default_cr_fraction() -> f64 {
    0.1
}
fn default_kappa_cr() -> f64 {
    3e-3
}
fn default_b_cr_suppress() -> f64 {
    1.0
}

impl Default for CrSection {
    fn default() -> Self {
        Self {
            enabled: false,
            cr_fraction: default_cr_fraction(),
            kappa_cr: default_kappa_cr(),
            b_cr_suppress: default_b_cr_suppress(),
            anisotropic_diffusion: false,
            hadronic_loss_coeff: 0.0,
            streaming_coefficient: 0.0,
        }
    }
}

/// Configuración del feedback de Agujeros Negros Supermasivos (AGN) (Phase 96).
///
/// Configura el modelo de acreción Bondi-Hoyle y feedback térmico/cinético.
///
/// ```toml
/// [sph.agn]
/// enabled      = true
/// eps_feedback = 0.05
/// m_seed       = 1e5
/// v_kick_agn   = 500.0
/// r_influence  = 1.0
/// n_agn_bh     = 1
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgnSection {
    /// Activa el feedback AGN (default: `false`).
    #[serde(default)]
    pub enabled: bool,
    /// Eficiencia radiativa del feedback ε_feedback (default: 0.05).
    #[serde(default = "default_eps_feedback")]
    pub eps_feedback: f64,
    /// Masa semilla del agujero negro [M_sol/h] (default: 1e5).
    #[serde(default = "default_m_seed")]
    pub m_seed: f64,
    /// Velocidad de kick cinético AGN [km/s] (default: 500.0; 0 = solo térmico).
    #[serde(default = "default_v_kick_agn")]
    pub v_kick_agn: f64,
    /// Radio de influencia máximo para depositar energía (default: 1.0).
    #[serde(default = "default_r_influence")]
    pub r_influence: f64,
    /// Número de BH semilla a colocar en los N halos FoF más masivos (default: 1).
    /// Si no hay halos identificados, se coloca uno en el centro de la caja.
    #[serde(default = "default_n_agn_bh")]
    pub n_agn_bh: usize,
    /// Umbral de Eddington para bifurcar modo quasar ↔ radio (Phase 116).
    /// `Ṁ / Ṁ_Edd < f_edd_threshold` → modo radio (jets mecánicos).
    /// Default: 0.01 (1% de la tasa de Eddington).
    #[serde(default = "default_f_edd_threshold")]
    pub f_edd_threshold: f64,
    /// Radio de la burbuja de jets mecánicos AGN [unidades internas] (Phase 116).
    /// Default: 2.0.
    #[serde(default = "default_r_bubble")]
    pub r_bubble: f64,
    /// Eficiencia del modo radio `ε_radio` (Phase 116).
    /// Fracción de la energía de acreción que va a kicks mecánicos. Default: 0.2.
    #[serde(default = "default_eps_radio")]
    pub eps_radio: f64,
}

fn default_eps_feedback() -> f64 {
    0.05
}
fn default_m_seed() -> f64 {
    1e5
}
fn default_v_kick_agn() -> f64 {
    500.0
}
fn default_r_influence() -> f64 {
    1.0
}
fn default_n_agn_bh() -> usize {
    1
}
fn default_f_edd_threshold() -> f64 {
    0.01
}
fn default_r_bubble() -> f64 {
    2.0
}
fn default_eps_radio() -> f64 {
    0.2
}

impl Default for AgnSection {
    fn default() -> Self {
        Self {
            enabled: false,
            eps_feedback: default_eps_feedback(),
            m_seed: default_m_seed(),
            v_kick_agn: default_v_kick_agn(),
            r_influence: default_r_influence(),
            n_agn_bh: default_n_agn_bh(),
            f_edd_threshold: default_f_edd_threshold(),
            r_bubble: default_r_bubble(),
            eps_radio: default_eps_radio(),
        }
    }
}
/// Configuración del módulo de polvo intersticial básico (Phase 130).
///
/// Modelo simplificado de acreción D/G por metalicidad y destrucción por sputtering.
///
/// ```toml
/// [sph.dust]
/// enabled     = true
/// d_to_g_max  = 0.01
/// t_destroy_k = 1e6
/// tau_grow    = 1.0
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DustSection {
    /// Activa el módulo de polvo (default: `false`).
    #[serde(default)]
    pub enabled: bool,
    /// D/G máximo solar (default: `0.01`).
    #[serde(default = "default_d_to_g_max")]
    pub d_to_g_max: f64,
    /// Temperatura de destrucción por sputtering en K (default: `1e6`).
    #[serde(default = "default_t_destroy_k")]
    pub t_destroy_k: f64,
    /// Tiempo de crecimiento por accreción en unidades internas (default: `1.0`).
    #[serde(default = "default_tau_grow")]
    pub tau_grow: f64,
    /// Opacidad del polvo en UV: `κ_dust [cm²/g]` (Phase 137). Default: `1000.0`.
    #[serde(default = "default_kappa_dust_uv")]
    pub kappa_dust_uv: f64,
    /// Impulso cinético fenomenológico por presión de radiación sobre el gas con polvo (Fase A roadmap).
    #[serde(default)]
    pub radiation_pressure_enabled: bool,
    /// Acoplamiento κ_rp × (D/G) × J_UV / ρ en dirección ±z (default 0).
    #[serde(default)]
    pub radiation_pressure_kappa: f64,
    /// Intensidad UV media de referencia [unidades internas] (default 0).
    #[serde(default)]
    pub radiation_pressure_j_uv: f64,
}

fn default_d_to_g_max() -> f64 {
    0.01
}
fn default_t_destroy_k() -> f64 {
    1e6
}
fn default_tau_grow() -> f64 {
    1.0
}
fn default_kappa_dust_uv() -> f64 {
    1000.0
}

impl Default for DustSection {
    fn default() -> Self {
        Self {
            enabled: false,
            d_to_g_max: default_d_to_g_max(),
            t_destroy_k: default_t_destroy_k(),
            tau_grow: default_tau_grow(),
            kappa_dust_uv: default_kappa_dust_uv(),
            radiation_pressure_enabled: false,
            radiation_pressure_kappa: 0.0,
            radiation_pressure_j_uv: 0.0,
        }
    }
}
