//! Formación estelar Pop III y feedback PISN (Phase 180).
//!
//! Este módulo usa criterios locales de gas primordial frío: baja metalicidad,
//! densidad alta y enfriamiento molecular por H2/HD. No depende de `gadget-ng-rt`
//! para evitar ciclos entre crates; el llamador pasa las fracciones H2/HD.

use crate::cooling::u_to_temperature;
use crate::periodic_delta;
use gadget_ng_core::{Particle, PopIIISection, Vec3};

/// Cúmulo Pop III formado desde una partícula de gas primordial.
#[derive(Debug, Clone, PartialEq)]
pub struct PopIIICluster {
    /// Posición del cúmulo en unidades internas.
    pub pos: Vec3,
    /// Masa estelar formada en unidades internas.
    pub mass_total: f64,
    /// Número efectivo de estrellas muestreadas.
    pub n_stars: usize,
    /// Masa media de la IMF muestreada \[M_sun\].
    pub mean_stellar_mass_msun: f64,
    /// Metalicidad heredada del gas.
    pub metallicity: f64,
    /// Fracción H2/H usada por el criterio.
    pub h2_fraction: f64,
    /// Fracción HD/H usada por el criterio.
    pub hd_fraction: f64,
}

#[inline]
fn gas_density(p: &Particle) -> f64 {
    let h = p.smoothing_length.max(1e-10);
    p.mass / ((4.0 / 3.0) * std::f64::consts::PI * h * h * h)
}

/// Devuelve `true` si el gas cumple el criterio Pop III local.
pub fn is_pop_iii_candidate(
    p: &Particle,
    h2_fraction: f64,
    hd_fraction: f64,
    cfg: &PopIIISection,
    gamma: f64,
) -> bool {
    if !cfg.enabled || !p.is_gas() {
        return false;
    }
    if p.metallicity > cfg.critical_metallicity {
        return false;
    }
    if gas_density(p) < cfg.density_threshold {
        return false;
    }
    let t = u_to_temperature(p.internal_energy, gamma);
    if t > cfg.max_temperature_k {
        return false;
    }
    h2_fraction >= cfg.min_h2_fraction || hd_fraction >= cfg.min_hd_fraction
}

#[inline]
fn lcg_rand01(seed: &mut u64) -> f64 {
    *seed = seed
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*seed >> 33) as f64 / (u64::MAX >> 33) as f64
}

/// Muestrea una masa de la IMF top-heavy Pop III `dN/dm ∝ m^-alpha`.
pub fn sample_pop_iii_mass(cfg: &PopIIISection, seed: &mut u64) -> f64 {
    let m_min = cfg.imf_m_min_msun.max(1e-6);
    let m_max = cfg.imf_m_max_msun.max(m_min);
    let alpha = cfg.imf_alpha;
    let u = lcg_rand01(seed).clamp(0.0, 1.0);
    if (alpha - 1.0).abs() < 1e-12 {
        m_min * (m_max / m_min).powf(u)
    } else {
        let e = 1.0 - alpha;
        (m_min.powf(e) + u * (m_max.powf(e) - m_min.powf(e))).powf(1.0 / e)
    }
    .clamp(m_min, m_max)
}

/// Forma cúmulos Pop III desde gas candidato.
///
/// `molecular_fractions[i] = (x_h2, x_hd)` debe estar alineado con `particles`.
pub fn form_pop_iii_clusters(
    particles: &mut [Particle],
    molecular_fractions: &[(f64, f64)],
    cfg: &PopIIISection,
    gamma: f64,
    seed: u64,
) -> Vec<PopIIICluster> {
    if !cfg.enabled {
        return Vec::new();
    }
    let mut rng = seed;
    let mut clusters = Vec::new();
    for (i, p) in particles.iter_mut().enumerate() {
        let (x_h2, x_hd) = molecular_fractions
            .get(i)
            .copied()
            .unwrap_or((p.h2_fraction, 0.0));
        if !is_pop_iii_candidate(p, x_h2, x_hd, cfg, gamma) {
            continue;
        }
        let mass_total = (cfg.cluster_efficiency.clamp(0.0, 1.0) * p.mass).min(0.5 * p.mass);
        if mass_total <= 0.0 {
            continue;
        }
        let n_sample = 16usize;
        let mut sum_mass = 0.0;
        for _ in 0..n_sample {
            sum_mass += sample_pop_iii_mass(cfg, &mut rng);
        }
        let mean_stellar_mass_msun = sum_mass / n_sample as f64;
        let n_stars =
            ((mass_total * 1.0e10) / mean_stellar_mass_msun).clamp(1.0, 100_000.0) as usize;

        clusters.push(PopIIICluster {
            pos: p.position,
            mass_total,
            n_stars,
            mean_stellar_mass_msun,
            metallicity: p.metallicity,
            h2_fraction: x_h2,
            hd_fraction: x_hd,
        });
        p.mass -= mass_total;
    }
    clusters
}

/// Aplica feedback de pair-instability SN desde cúmulos Pop III.
///
/// Deposita energía térmica y metales en gas dentro de `feedback_radius`.
/// Retorna la energía total inyectada.
pub fn apply_pop_iii_pisn_feedback(
    clusters: &[PopIIICluster],
    particles: &mut [Particle],
    cfg: &PopIIISection,
    periodic_box: Option<f64>,
) -> f64 {
    if clusters.is_empty() || cfg.pisn_energy_code <= 0.0 {
        return 0.0;
    }
    let r_fb = cfg.feedback_radius.max(1e-10);
    let mut total_energy = 0.0;
    for cluster in clusters {
        let neighbors: Vec<usize> = particles
            .iter()
            .enumerate()
            .filter(|(_, p)| p.is_gas())
            .filter(|(_, p)| periodic_delta(cluster.pos, p.position, periodic_box).norm() <= r_fb)
            .map(|(i, _)| i)
            .collect();
        if neighbors.is_empty() {
            continue;
        }
        let e_each = cfg.pisn_energy_code * cluster.mass_total / neighbors.len() as f64;
        let z_add = cfg.metal_yield.max(0.0) * cluster.mass_total / neighbors.len() as f64;
        for i in neighbors {
            let m = particles[i].mass.max(1e-30);
            particles[i].internal_energy += e_each / m;
            particles[i].metallicity = (particles[i].metallicity + z_add / m).clamp(0.0, 1.0);
            total_energy += e_each;
        }
    }
    total_energy
}

#[cfg(test)]
mod tests {
    use super::*;
    use gadget_ng_core::{Particle, PopIIISection};

    fn cfg() -> PopIIISection {
        PopIIISection {
            enabled: true,
            density_threshold: 1.0,
            max_temperature_k: 5.0e3,
            ..Default::default()
        }
    }

    fn gas(z: f64, h: f64, u: f64) -> Particle {
        let mut p = Particle::new_gas(0, 1.0, Vec3::zero(), Vec3::zero(), u, h);
        p.metallicity = z;
        p
    }

    #[test]
    fn candidate_requires_low_metallicity_and_molecules() {
        let cfg = cfg();
        let p = gas(0.0, 0.1, 1.0);
        assert!(is_pop_iii_candidate(&p, 1e-3, 0.0, &cfg, 5.0 / 3.0));
        let enriched = gas(1e-2, 0.1, 1.0);
        assert!(!is_pop_iii_candidate(&enriched, 1e-3, 0.0, &cfg, 5.0 / 3.0));
        assert!(!is_pop_iii_candidate(&p, 0.0, 0.0, &cfg, 5.0 / 3.0));
    }

    #[test]
    fn top_heavy_imf_samples_in_range() {
        let cfg = cfg();
        let mut seed = 42;
        for _ in 0..64 {
            let m = sample_pop_iii_mass(&cfg, &mut seed);
            assert!(m >= cfg.imf_m_min_msun && m <= cfg.imf_m_max_msun);
        }
    }

    #[test]
    fn pisn_feedback_heats_and_enriches_gas() {
        let cfg = cfg();
        let cluster = PopIIICluster {
            pos: Vec3::zero(),
            mass_total: 0.1,
            n_stars: 10,
            mean_stellar_mass_msun: 180.0,
            metallicity: 0.0,
            h2_fraction: 1e-3,
            hd_fraction: 1e-8,
        };
        let mut p = vec![gas(0.0, 0.1, 1.0)];
        let e = apply_pop_iii_pisn_feedback(&[cluster], &mut p, &cfg, None);
        assert!(e > 0.0);
        assert!(p[0].internal_energy > 1.0);
        assert!(p[0].metallicity > 0.0);
    }
}
