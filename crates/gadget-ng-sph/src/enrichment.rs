//! Enriquecimiento químico metálico por SPH (Phase 110).
//!
//! Distribuye metales eyectados por supernovas (SN II) y estrellas AGB
//! a las partículas de gas vecinas usando el kernel SPH.
//!
//! ## Algoritmo
//!
//! Para cada partícula de gas con `sfr[i] > sfr_min` que dispara un evento SN
//! en el paso `dt`:
//!
//! 1. Se calcula la masa metálica eyectada: `ΔZ = cfg.yield_snii × sfr[i] × dt`.
//! 2. Se distribuye a todos los vecinos de gas dentro de `2 × h_sml` ponderados
//!    por el kernel W(r, h).
//! 3. El incremento de metalicidad del vecino es:
//!    `ΔZ_j = w_ij / Σ_k w_ik × ΔZ / m_j`
//!
//! ## Referencias
//!
//! - Woosley & Weaver (1995) ApJS 101, 181 — yields de SN II
//! - Portinari, Chiosi & Bressan (1998) A&A 334, 505 — yields AGB

use gadget_ng_core::{EnrichmentSection, Particle, ParticleType};

/// Factor de normalización del kernel SPH (usado solo para peso relativo).
#[inline]
fn kernel_w(r: f64, h: f64) -> f64 {
    if h <= 0.0 {
        return 0.0;
    }
    let q = r / h;
    if q > 2.0 {
        return 0.0;
    }
    // Wendland C2 3D (mismo kernel que el módulo sph principal)
    let t = 1.0 - 0.5 * q;
    let t4 = t * t * t * t;
    (21.0 / (2.0 * std::f64::consts::PI)) / (h * h * h) * t4 * (1.0 + 2.0 * q)
}

/// Distribuye metales de SN al gas vecino (Phase 110).
///
/// Para cada partícula gas con `sfr[i] > sfr_min`, calcula la masa metálica
/// eyectada y la distribuye a vecinos ponderados por kernel SPH.
///
/// El enriquecimiento AGB se aplica a las **partículas estelares** (Star) de forma
/// gradual: cada estrella "gotea" `cfg.yield_agb × m_star × dt / t_agb` a vecinos
/// de gas. Para simplificar, usamos `t_agb = 1.0` en unidades internas.
///
/// # Parámetros
///
/// - `particles` — slice mutable con todas las partículas del rank local.
/// - `sfr` — tasa de formación estelar por partícula en unidades internas [1/tiempo].
/// - `dt` — paso de tiempo en unidades internas.
/// - `cfg` — configuración de enriquecimiento.
pub fn apply_enrichment(particles: &mut [Particle], sfr: &[f64], dt: f64, cfg: &EnrichmentSection) {
    if !cfg.enabled || particles.is_empty() {
        return;
    }

    let n = particles.len();
    assert_eq!(sfr.len(), n, "sfr.len() debe coincidir con particles.len()");

    // Acumular incrementos de metalicidad para evitar aliasing
    let mut delta_z = vec![0.0_f64; n];

    // ── SN II desde partículas de gas ─────────────────────────────────────
    for i in 0..n {
        if particles[i].ptype != ParticleType::Gas {
            continue;
        }
        if sfr[i] <= 0.0 {
            continue;
        }

        let h_i = particles[i].smoothing_length.max(1e-10);
        let delta_metal = cfg.yield_snii * sfr[i] * dt;
        if delta_metal <= 0.0 {
            continue;
        }

        // Encontrar vecinos y acumular pesos del kernel
        let pos_i = particles[i].position;
        let mut weights = vec![0.0_f64; n];
        let mut weight_sum = 0.0_f64;

        for j in 0..n {
            if i == j {
                continue;
            }
            if particles[j].ptype != ParticleType::Gas {
                continue;
            }

            let dx = particles[j].position.x - pos_i.x;
            let dy = particles[j].position.y - pos_i.y;
            let dz = particles[j].position.z - pos_i.z;
            let r = (dx * dx + dy * dy + dz * dz).sqrt();

            let w = kernel_w(r, 2.0 * h_i);
            if w > 0.0 {
                weights[j] = w;
                weight_sum += w;
            }
        }

        if weight_sum <= 0.0 {
            continue;
        }

        // Distribuir metales a vecinos
        for j in 0..n {
            if weights[j] <= 0.0 {
                continue;
            }
            let m_j = particles[j].mass.max(1e-30);
            delta_z[j] += (weights[j] / weight_sum) * delta_metal / m_j;
        }
    }

    // ── AGB desde partículas estelares ────────────────────────────────────
    // Rata gradual: ΔZ_agb = yield_agb × m_star × dt (normalizada por tiempo AGB = 1)
    for i in 0..n {
        if particles[i].ptype != ParticleType::Star {
            continue;
        }

        let h_i = particles[i].smoothing_length.max(1e-10).max(
            // Usar media de smoothing lengths de vecinos si h_star = 0
            0.1,
        );
        let delta_metal = cfg.yield_agb * particles[i].mass * dt;
        if delta_metal <= 0.0 {
            continue;
        }

        let pos_i = particles[i].position;
        let mut weights = vec![0.0_f64; n];
        let mut weight_sum = 0.0_f64;

        for j in 0..n {
            if i == j {
                continue;
            }
            if particles[j].ptype != ParticleType::Gas {
                continue;
            }

            let dx = particles[j].position.x - pos_i.x;
            let dy = particles[j].position.y - pos_i.y;
            let dz = particles[j].position.z - pos_i.z;
            let r = (dx * dx + dy * dy + dz * dz).sqrt();

            let w = kernel_w(r, 2.0 * h_i);
            if w > 0.0 {
                weights[j] = w;
                weight_sum += w;
            }
        }

        if weight_sum <= 0.0 {
            continue;
        }

        for j in 0..n {
            if weights[j] <= 0.0 {
                continue;
            }
            let m_j = particles[j].mass.max(1e-30);
            delta_z[j] += (weights[j] / weight_sum) * delta_metal / m_j;
        }
    }

    // ── Aplicar incrementos ───────────────────────────────────────────────
    for i in 0..n {
        if delta_z[i] > 0.0 {
            particles[i].metallicity = (particles[i].metallicity + delta_z[i]).min(1.0);
        }
    }
}
