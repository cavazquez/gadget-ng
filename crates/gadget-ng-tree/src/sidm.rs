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
    let h_vals: Vec<f64> = particles
        .iter()
        .map(|p| p.smoothing_length.max(0.01))
        .collect();

    let rho_local: Vec<f64> = (0..n)
        .map(|i| {
            let h2 = 2.0 * h_vals[i];
            let mut rho = 0.0_f64;
            for j in 0..n {
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
        })
        .collect();

    // Aplicar scattering por pares (i < j)
    let mut delta_v: Vec<[f64; 3]> = vec![[0.0; 3]; n];

    for i in 0..n {
        if particles[i].is_gas() || particles[i].is_star() {
            continue;
        }
        let h2_i = 2.0 * h_vals[i];

        for j in (i + 1)..n {
            if particles[j].is_gas() || particles[j].is_star() {
                continue;
            }

            let dx = positions[j][0] - positions[i][0];
            let dy = positions[j][1] - positions[i][1];
            let dz = positions[j][2] - positions[i][2];
            let r2 = dx * dx + dy * dy + dz * dz;
            if r2 >= h2_i * h2_i {
                continue;
            }

            let dvx = particles[j].velocity.x - particles[i].velocity.x;
            let dvy = particles[j].velocity.y - particles[i].velocity.y;
            let dvz = particles[j].velocity.z - particles[i].velocity.z;
            let v_rel = (dvx * dvx + dvy * dvy + dvz * dvz).sqrt();

            if v_rel > params.v_max || v_rel <= 0.0 {
                continue;
            }

            let rho_ij = 0.5 * (rho_local[i] + rho_local[j]);
            let prob = scatter_probability(v_rel, rho_ij, params.sigma_m, dt);

            // Sortear con LCG (semilla determinista por par)
            let pair_seed = rng_seed
                .wrapping_add(i as u64 * 1_000_003)
                .wrapping_add(j as u64 * 7_919);
            if lcg_rand(pair_seed) > prob {
                continue;
            }

            // Scattering elástico isótropo en sistema CM:
            // velocidades CM del par
            let m_i = masses[i];
            let m_j = masses[j];
            let m_tot = m_i + m_j;
            let vcm_x = (m_i * particles[i].velocity.x + m_j * particles[j].velocity.x) / m_tot;
            let vcm_y = (m_i * particles[i].velocity.y + m_j * particles[j].velocity.y) / m_tot;
            let vcm_z = (m_i * particles[i].velocity.z + m_j * particles[j].velocity.z) / m_tot;

            // Dirección aleatoria isótropa para la velocidad relativa saliente
            let seed2 = pair_seed.wrapping_add(0xDEAD_BEEF);
            let seed3 = pair_seed.wrapping_add(0xCAFE_BABE);
            let cos_theta = 2.0 * lcg_rand(seed2) - 1.0;
            let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
            let phi = 2.0 * std::f64::consts::PI * lcg_rand(seed3);
            let nx = sin_theta * phi.cos();
            let ny = sin_theta * phi.sin();
            let nz = cos_theta;

            // Velocidad relativa entrante en CM
            let vrel_in_x = particles[j].velocity.x - particles[i].velocity.x;
            let vrel_in_y = particles[j].velocity.y - particles[i].velocity.y;
            let vrel_in_z = particles[j].velocity.z - particles[i].velocity.z;
            let v_rel_mag =
                (vrel_in_x * vrel_in_x + vrel_in_y * vrel_in_y + vrel_in_z * vrel_in_z).sqrt();

            // Velocidad relativa saliente (misma magnitud, nueva dirección)
            let vrel_out_x = v_rel_mag * nx;
            let vrel_out_y = v_rel_mag * ny;
            let vrel_out_z = v_rel_mag * nz;

            // Nuevas velocidades en lab: v_i = v_CM - (m_j/m_tot)*v_rel_out
            let new_vi_x = vcm_x - (m_j / m_tot) * vrel_out_x;
            let new_vi_y = vcm_y - (m_j / m_tot) * vrel_out_y;
            let new_vi_z = vcm_z - (m_j / m_tot) * vrel_out_z;
            let new_vj_x = vcm_x + (m_i / m_tot) * vrel_out_x;
            let new_vj_y = vcm_y + (m_i / m_tot) * vrel_out_y;
            let new_vj_z = vcm_z + (m_i / m_tot) * vrel_out_z;

            // Acumular delta_v (se aplica después del loop para evitar conflictos)
            delta_v[i][0] += new_vi_x - particles[i].velocity.x;
            delta_v[i][1] += new_vi_y - particles[i].velocity.y;
            delta_v[i][2] += new_vi_z - particles[i].velocity.z;
            delta_v[j][0] += new_vj_x - particles[j].velocity.x;
            delta_v[j][1] += new_vj_y - particles[j].velocity.y;
            delta_v[j][2] += new_vj_z - particles[j].velocity.z;
        }
    }

    // Aplicar cambios de velocidad
    for (i, p) in particles.iter_mut().enumerate() {
        p.velocity.x += delta_v[i][0];
        p.velocity.y += delta_v[i][1];
        p.velocity.z += delta_v[i][2];
    }
}
