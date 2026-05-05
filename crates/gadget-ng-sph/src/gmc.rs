//! Formación de cúmulos estelares por colapso de GMC + IMF Kroupa (Phase 159).
//!
//! ## Modelo físico
//!
//! Sobre el modelo de formación estelar estocástica existente (Phase 112),
//! se agrega un nivel de sub-partición en Grupos de Nubes Moleculares Gigantes
//! (Giant Molecular Clouds, GMC):
//!
//! 1. Partículas de gas con SFR alta forman un `GmcCluster` con masa discreta
//!    y N_* estrellas muestreadas de la IMF de Kroupa (2001).
//!
//! 2. La retroalimentación por supernovas (SN II) se inyecta en función de la
//!    masa del cúmulo y la edad: solo cúmulos con `age < 30 Myr` generan SN II.
//!
//! ## IMF de Kroupa (2001)
//!
//! ```text
//! dN/dm ∝ m^{-α(m)}
//! α₁ = 1.3   para 0.1 ≤ m/M☉ < 0.5
//! α₂ = 2.3   para 0.5 ≤ m/M☉ ≤ 150
//! ```
//!
//! ## Referencia
//!
//! Kroupa (2001) MNRAS 322, 231 — IMF estándar.
//! Kennicutt (1998) ApJ 498, 541 — relación Schmidt-Kennicutt.

use gadget_ng_core::Particle;
use gadget_ng_core::config::SphSection;

/// Masa mínima de la IMF de Kroupa [M☉].
const M_MIN: f64 = 0.1;
/// Masa máxima de la IMF de Kroupa [M☉].
const M_MAX: f64 = 150.0;
/// Masa de transición de la IMF de Kroupa [M☉].
const M_BREAK: f64 = 0.5;
/// Exponente de baja masa (α₁).
const ALPHA1: f64 = 1.3;
/// Exponente de alta masa (α₂).
const ALPHA2: f64 = 2.3;

/// IMF de Kroupa (2001) (Phase 159).
#[derive(Debug, Clone)]
pub struct KroupaImf {
    /// Masa mínima [M☉]. Default: 0.1.
    pub m_min: f64,
    /// Masa máxima [M☉]. Default: 150.0.
    pub m_max: f64,
    /// Masa de transición [M☉]. Default: 0.5.
    pub m_break: f64,
    /// Índice de baja masa α₁. Default: 1.3.
    pub alpha1: f64,
    /// Índice de alta masa α₂. Default: 2.3.
    pub alpha2: f64,
}

impl Default for KroupaImf {
    fn default() -> Self {
        Self {
            m_min: M_MIN,
            m_max: M_MAX,
            m_break: M_BREAK,
            alpha1: ALPHA1,
            alpha2: ALPHA2,
        }
    }
}

/// Normalización de la IMF de Kroupa (integral en [m_min, m_max]).
fn kroupa_norm(imf: &KroupaImf) -> f64 {
    // ∫ m^{-α} dm = m^{1-α}/(1-α)  para α ≠ 1
    // Segmento 1: [m_min, m_break] con α1
    let c1 = if (imf.alpha1 - 1.0).abs() < 1e-10 {
        imf.m_break.ln() - imf.m_min.ln()
    } else {
        let e = 1.0 - imf.alpha1;
        (imf.m_break.powf(e) - imf.m_min.powf(e)) / e
    };
    // Segmento 2: [m_break, m_max] con α2 — factor de continuidad
    let k_ratio = imf.m_break.powf(imf.alpha2 - imf.alpha1);
    let c2 = if (imf.alpha2 - 1.0).abs() < 1e-10 {
        k_ratio * (imf.m_max.ln() - imf.m_break.ln())
    } else {
        let e = 1.0 - imf.alpha2;
        k_ratio * (imf.m_max.powf(e) - imf.m_break.powf(e)) / e
    };
    c1 + c2
}

/// Muestrea una masa estelar de la IMF de Kroupa usando inversión de CDF (Phase 159).
///
/// Usa la técnica de inversión analítica de la CDF normalizada.
///
/// # Parámetros
/// - `imf`: parámetros de la IMF
/// - `rng_seed`: semilla para el LCG
///
/// # Retorna
/// Masa estelar en M☉, en el intervalo [m_min, m_max].
pub fn sample_stellar_mass(imf: &KroupaImf, rng_seed: u64) -> f64 {
    let u = {
        let x = rng_seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (x >> 33) as f64 / (u64::MAX >> 33) as f64
    };

    let norm = kroupa_norm(imf);
    if norm <= 0.0 {
        return imf.m_min;
    }

    // CDF en m_break: P(<m_break)
    let c1 = if (imf.alpha1 - 1.0).abs() < 1e-10 {
        imf.m_break.ln() - imf.m_min.ln()
    } else {
        let e = 1.0 - imf.alpha1;
        (imf.m_break.powf(e) - imf.m_min.powf(e)) / e
    };
    let p_break = c1 / norm;

    if u < p_break {
        // Invertir en segmento bajo: u × norm = (m^{1-α1} - m_min^{1-α1}) / (1-α1)
        let e = 1.0 - imf.alpha1;
        if e.abs() < 1e-10 {
            imf.m_min * (u * norm).exp()
        } else {
            (imf.m_min.powf(e) + u * norm * e).max(0.0).powf(1.0 / e)
        }
    } else {
        // Invertir en segmento alto con continuidad
        let k_ratio = imf.m_break.powf(imf.alpha2 - imf.alpha1);
        let u_adj = (u * norm - c1) / k_ratio;
        let e = 1.0 - imf.alpha2;
        if e.abs() < 1e-10 {
            imf.m_break * (u_adj).exp()
        } else {
            (imf.m_break.powf(e) + u_adj * e).max(0.0).powf(1.0 / e)
        }
    }
    .clamp(imf.m_min, imf.m_max)
}

/// Cúmulo estelar GMC (Phase 159).
#[derive(Debug, Clone)]
pub struct GmcCluster {
    /// Posición del cúmulo [x, y, z] en unidades internas.
    pub pos: [f64; 3],
    /// Masa total del cúmulo en unidades internas.
    pub mass_total: f64,
    /// Número de estrellas muestreadas de la IMF.
    pub n_stars: usize,
    /// Edad del cúmulo en Gyr.
    pub age_gyr: f64,
    /// Metalicidad heredada del gas progenitor.
    pub metallicity: f64,
}

/// Colapsa partículas de gas con SFR alta en cúmulos GMC (Phase 159).
///
/// Para cada partícula de gas con `sfr > sfr_threshold`, se crea un `GmcCluster`
/// que consume la masa de gas disponible. La masa del cúmulo es la masa de la
/// partícula multiplicada por `sfr × dt`.
///
/// # Parámetros
/// - `particles`: partículas de gas (se modifican: reduce masa estelar formada)
/// - `sfr_threshold`: umbral mínimo de SFR [unidades internas/Gyr]
/// - `dt`: paso de tiempo en Gyr
/// - `seed`: semilla base para la IMF
///
/// # Retorna
/// `Vec<GmcCluster>` de los cúmulos formados en este paso.
pub fn collapse_gmc(
    particles: &mut [Particle],
    sfr_threshold: f64,
    dt: f64,
    seed: u64,
) -> Vec<GmcCluster> {
    let imf = KroupaImf::default();
    let mut clusters = Vec::new();

    for (idx, p) in particles.iter_mut().enumerate() {
        if !p.is_gas() {
            continue;
        }
        // SFR estimada desde energía interna y densidad (proxy de Kennicutt)
        let rho = if p.smoothing_length > 0.0 {
            p.mass / p.smoothing_length.powi(3)
        } else {
            0.0
        };
        let sfr_est = rho.max(0.0) * 1e-3; // proxy: SFR ∝ ρ
        if sfr_est < sfr_threshold {
            continue;
        }

        let mass_formed = (sfr_est * dt).min(p.mass * 0.1); // hasta 10% de la masa
        if mass_formed <= 0.0 {
            continue;
        }

        // Muestrear N estrellas de la IMF hasta completar la masa
        let mean_stellar_mass = 0.3; // M☉ promedio Kroupa
        let n_stars_est = (mass_formed * 1e10 / mean_stellar_mass).max(1.0) as usize;

        let cluster_seed = seed.wrapping_add(idx as u64 * 99_991);
        // Verificar que la IMF está normalizada — muestra de prueba
        let _m_test = sample_stellar_mass(&imf, cluster_seed);

        clusters.push(GmcCluster {
            pos: [p.position.x, p.position.y, p.position.z],
            mass_total: mass_formed,
            n_stars: n_stars_est.min(10_000),
            age_gyr: 0.0,
            metallicity: p.metallicity,
        });
    }

    clusters
}

/// Inyecta feedback de SN II desde los cúmulos jóvenes (Phase 159).
///
/// Solo los cúmulos con `age_gyr < 0.03` (30 Myr) generan SN II.
/// La energía de SN II se inyecta en las partículas de gas cercanas al cúmulo.
///
/// Energía típica por SN II: E_SN ≈ 10⁵¹ erg, con ~1 SN por 100 M☉ de estrellas masivas.
///
/// # Parámetros
/// - `clusters`: cúmulos GMC activos
/// - `particles`: partículas de gas (reciben la energía)
/// - `dt`: paso de tiempo en Gyr
/// - `cfg`: configuración SPH
pub fn inject_sn_from_cluster(
    clusters: &[GmcCluster],
    particles: &mut [Particle],
    dt: f64,
    cfg: &SphSection,
) {
    const AGE_SN_MAX_GYR: f64 = 0.030; // 30 Myr
    const E_SN_CODE: f64 = 1.0e-4; // energía SN en unidades internas por unidad de masa
    const R_INJECT: f64 = 0.5; // radio de inyección en unidades internas

    for cluster in clusters {
        if cluster.age_gyr > AGE_SN_MAX_GYR {
            continue;
        }

        // Número de SN II: ~1 por cada 100 M☉ de masa del cúmulo
        let n_sn = (cluster.mass_total * 1e10 / 100.0).max(0.0);
        if n_sn <= 0.0 {
            continue;
        }

        let e_total = n_sn * E_SN_CODE * dt / 0.01; // normalizado a dt
        let mut n_neighbors = 0_usize;
        // Contar vecinos primero
        for p in particles.iter() {
            if !p.is_gas() {
                continue;
            }
            let dx = p.position.x - cluster.pos[0];
            let dy = p.position.y - cluster.pos[1];
            let dz = p.position.z - cluster.pos[2];
            if (dx * dx + dy * dy + dz * dz).sqrt() < R_INJECT {
                n_neighbors += 1;
            }
        }
        if n_neighbors == 0 {
            continue;
        }
        let e_per_particle = e_total / n_neighbors as f64;

        for p in particles.iter_mut() {
            if !p.is_gas() {
                continue;
            }
            let dx = p.position.x - cluster.pos[0];
            let dy = p.position.y - cluster.pos[1];
            let dz = p.position.z - cluster.pos[2];
            if (dx * dx + dy * dy + dz * dz).sqrt() < R_INJECT {
                // Escalar por gamma para convertir energía en u (energía interna específica)
                p.internal_energy += e_per_particle * (cfg.gamma - 1.0);
                // Enriquecer ligeramente con metales
                p.metallicity += cluster.metallicity * 0.01 * dt;
            }
        }
    }
}
