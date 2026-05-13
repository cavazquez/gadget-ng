//! Materia oscura auto-interactuante (SIDM) — Phase 157.
//!
//! ## Modelo físico
//!
//! Las partículas de materia oscura con sección eficaz σ/m pueden sufrir
//! scattering elástico isótropo. La probabilidad de scattering entre dos
//! partículas vecinas es:
//!
//! ```text
//! P = (σ/m) × ρ_local × v_rel × dt
//! ```
//!
//! donde `ρ_local` se estima como la densidad media en un radio de 2h.
//! El scattering es **elástico**: conserva masa, momento total y energía
//! cinética. La dirección de salida es isótropa en el sistema CM.
//!
//! ## Referencia
//!
//! Spergel & Steinhardt (2000) PRL 84, 3760 — propuesta SIDM.
//! Rocha et al. (2013) MNRAS 430, 81 — implementación N-body.

use gadget_ng_core::Particle;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

/// Parámetros del modelo SIDM (Phase 157).
#[derive(Debug, Clone)]
pub struct SidmParams {
    /// Sección eficaz por masa σ/m en cm²/g (unidades de referencia).
    /// En unidades internas: se escala como `sigma_m × conv_factor`.
    /// Para σ/m = 1 cm²/g con unidades kpc/10¹⁰M☉: conv ≈ 1e-5.
    pub sigma_m: f64,
    /// Corte de velocidad [unidades internas]. Scattering inhibido para v_rel > v_max.
    pub v_max: f64,
}

impl Default for SidmParams {
    fn default() -> Self {
        Self {
            sigma_m: 1.0e-5,
            v_max: 1.0e6,
        }
    }
}

/// LCG simple para reproducibilidad sin dependencias externas (rng_seed por par).
#[inline]
fn lcg_rand(seed: u64) -> f64 {
    let x = seed
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (x >> 33) as f64 / (u64::MAX >> 33) as f64
}

/// Probabilidad de scattering para un par de partículas (Phase 157).
///
/// `P = (σ/m) × ρ_local × v_rel × dt`
///
/// # Parámetros
/// - `v_rel`: velocidad relativa en unidades internas
/// - `rho_local`: densidad local estimada en unidades internas
/// - `sigma_m`: sección eficaz por masa en unidades internas
/// - `dt`: paso de tiempo
pub fn scatter_probability(v_rel: f64, rho_local: f64, sigma_m: f64, dt: f64) -> f64 {
    (sigma_m * rho_local * v_rel * dt).min(1.0)
}

/// Aplica el scattering SIDM a todas las partículas (Phase 157).
///
/// Para cada partícula busca vecinos en radio `2h` y sortea scattering
/// elástico isótropo si se da la probabilidad. El scattering se aplica
/// en el sistema de referencia del CM del par.
///
/// # Parámetros
/// - `particles`: partículas de materia oscura (se ignoran las de gas/estrella)
/// - `params`: parámetros SIDM
/// - `dt`: paso de tiempo
/// - `rng_seed`: semilla para el generador de números aleatorios
pub fn apply_sidm_scattering(
    particles: &mut [Particle],
    params: &SidmParams,
    dt: f64,
    rng_seed: u64,
) {
    if params.sigma_m <= 0.0 {
        return;
    }
    let n = particles.len();
    if n < 2 {
        return;
    }

    // Precalcular densidades locales (estimación simple: suma de masas en 2h)
    let positions: Vec<[f64; 3]> = particles
        .iter()
        .map(|p| [p.position.x, p.position.y, p.position.z])
        .collect();
    let masses: Vec<f64> = particles.iter().map(|p| p.mass).collect();
    let velocities: Vec<[f64; 3]> = particles
        .iter()
        .map(|p| [p.velocity.x, p.velocity.y, p.velocity.z])
        .collect();
    let is_dm: Vec<bool> = particles
        .iter()
        .map(|p| !p.is_gas() && !p.is_star())
        .collect();
    let h_vals: Vec<f64> = particles
        .iter()
        .map(|p| p.smoothing_length.max(0.01))
        .collect();

    #[cfg(feature = "rayon")]
    let rho_local: Vec<f64> = (0..n)
        .into_par_iter()
        .map(|i| sidm_local_density(i, &positions, &masses, &h_vals))
        .collect();

    #[cfg(not(feature = "rayon"))]
    let rho_local: Vec<f64> = (0..n)
        .map(|i| sidm_local_density(i, &positions, &masses, &h_vals))
        .collect();

    // Aplicar scattering por pares (i < j)
    let mut delta_v: Vec<[f64; 3]> = vec![[0.0; 3]; n];

    #[cfg(feature = "rayon")]
    {
        let mut pair_deltas: Vec<SidmPairDelta> = (0..n)
            .into_par_iter()
            .flat_map_iter(|i| {
                let positions = &positions;
                let velocities = &velocities;
                let masses = &masses;
                let h_vals = &h_vals;
                let rho_local = &rho_local;
                let is_dm = &is_dm;
                ((i + 1)..n).filter_map(move |j| {
                    sidm_pair_delta(
                        i, j, positions, velocities, masses, h_vals, rho_local, is_dm, params, dt,
                        rng_seed,
                    )
                })
            })
            .collect();
        pair_deltas.sort_by_key(|delta| (delta.i, delta.j));
        for pair in pair_deltas {
            delta_v[pair.i][0] += pair.dvi[0];
            delta_v[pair.i][1] += pair.dvi[1];
            delta_v[pair.i][2] += pair.dvi[2];
            delta_v[pair.j][0] += pair.dvj[0];
            delta_v[pair.j][1] += pair.dvj[1];
            delta_v[pair.j][2] += pair.dvj[2];
        }
    }

    #[cfg(not(feature = "rayon"))]
    for i in 0..n {
        if !is_dm[i] {
            continue;
        }

        for j in (i + 1)..n {
            if let Some(pair) = sidm_pair_delta(
                i,
                j,
                &positions,
                &velocities,
                &masses,
                &h_vals,
                &rho_local,
                &is_dm,
                params,
                dt,
                rng_seed,
            ) {
                delta_v[pair.i][0] += pair.dvi[0];
                delta_v[pair.i][1] += pair.dvi[1];
                delta_v[pair.i][2] += pair.dvi[2];
                delta_v[pair.j][0] += pair.dvj[0];
                delta_v[pair.j][1] += pair.dvj[1];
                delta_v[pair.j][2] += pair.dvj[2];
            }
        }
    }

    // Aplicar cambios de velocidad
    for (i, p) in particles.iter_mut().enumerate() {
        p.velocity.x += delta_v[i][0];
        p.velocity.y += delta_v[i][1];
        p.velocity.z += delta_v[i][2];
    }
}

fn sidm_local_density(i: usize, positions: &[[f64; 3]], masses: &[f64], h_vals: &[f64]) -> f64 {
    let h2 = 2.0 * h_vals[i];
    let mut rho = 0.0_f64;
    for j in 0..positions.len() {
        let dx = positions[j][0] - positions[i][0];
        let dy = positions[j][1] - positions[i][1];
        let dz = positions[j][2] - positions[i][2];
        let r2 = dx * dx + dy * dy + dz * dz;
        if r2 < h2 * h2 {
            rho += masses[j];
        }
    }
    let vol = std::f64::consts::FRAC_PI_6 * h2.powi(3);
    if vol > 0.0 { rho / vol } else { 0.0 }
}

struct SidmPairDelta {
    i: usize,
    j: usize,
    dvi: [f64; 3],
    dvj: [f64; 3],
}

#[expect(
    clippy::too_many_arguments,
    reason = "SIDM pair kernel keeps SoA slices explicit for Rayon"
)]
fn sidm_pair_delta(
    i: usize,
    j: usize,
    positions: &[[f64; 3]],
    velocities: &[[f64; 3]],
    masses: &[f64],
    h_vals: &[f64],
    rho_local: &[f64],
    is_dm: &[bool],
    params: &SidmParams,
    dt: f64,
    rng_seed: u64,
) -> Option<SidmPairDelta> {
    if !is_dm[i] || !is_dm[j] {
        return None;
    }

    let h2_i = 2.0 * h_vals[i];
    let dx = positions[j][0] - positions[i][0];
    let dy = positions[j][1] - positions[i][1];
    let dz = positions[j][2] - positions[i][2];
    let r2 = dx * dx + dy * dy + dz * dz;
    if r2 >= h2_i * h2_i {
        return None;
    }

    let dvx = velocities[j][0] - velocities[i][0];
    let dvy = velocities[j][1] - velocities[i][1];
    let dvz = velocities[j][2] - velocities[i][2];
    let v_rel = (dvx * dvx + dvy * dvy + dvz * dvz).sqrt();
    if v_rel > params.v_max || v_rel <= 0.0 {
        return None;
    }

    let rho_ij = 0.5 * (rho_local[i] + rho_local[j]);
    let prob = scatter_probability(v_rel, rho_ij, params.sigma_m, dt);
    let pair_seed = rng_seed
        .wrapping_add(i as u64 * 1_000_003)
        .wrapping_add(j as u64 * 7_919);
    if lcg_rand(pair_seed) > prob {
        return None;
    }

    let m_i = masses[i];
    let m_j = masses[j];
    let m_tot = m_i + m_j;
    let vcm_x = (m_i * velocities[i][0] + m_j * velocities[j][0]) / m_tot;
    let vcm_y = (m_i * velocities[i][1] + m_j * velocities[j][1]) / m_tot;
    let vcm_z = (m_i * velocities[i][2] + m_j * velocities[j][2]) / m_tot;

    let seed2 = pair_seed.wrapping_add(0xDEAD_BEEF);
    let seed3 = pair_seed.wrapping_add(0xCAFE_BABE);
    let cos_theta = 2.0 * lcg_rand(seed2) - 1.0;
    let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
    let phi = 2.0 * std::f64::consts::PI * lcg_rand(seed3);
    let nx = sin_theta * phi.cos();
    let ny = sin_theta * phi.sin();
    let nz = cos_theta;

    let vrel_out_x = v_rel * nx;
    let vrel_out_y = v_rel * ny;
    let vrel_out_z = v_rel * nz;

    let new_vi_x = vcm_x - (m_j / m_tot) * vrel_out_x;
    let new_vi_y = vcm_y - (m_j / m_tot) * vrel_out_y;
    let new_vi_z = vcm_z - (m_j / m_tot) * vrel_out_z;
    let new_vj_x = vcm_x + (m_i / m_tot) * vrel_out_x;
    let new_vj_y = vcm_y + (m_i / m_tot) * vrel_out_y;
    let new_vj_z = vcm_z + (m_i / m_tot) * vrel_out_z;

    Some(SidmPairDelta {
        i,
        j,
        dvi: [
            new_vi_x - velocities[i][0],
            new_vi_y - velocities[i][1],
            new_vi_z - velocities[i][2],
        ],
        dvj: [
            new_vj_x - velocities[j][0],
            new_vj_y - velocities[j][1],
            new_vj_z - velocities[j][2],
        ],
    })
}
