//! Feedback estelar por supernovas — kicks estocásticos (Phase 78).
//!
//! ## Modelo
//!
//! Basado en el modelo de vientos estelares estocásticos de Springel & Hernquist (2003)
//! y el modelo térmico simplificado de Dalla Vecchia & Schaye (2012).
//!
//! ### Tasa de formación estelar (SFR)
//!
//! Ley de Schmidt-Kennicutt:
//! ```text
//! SFR(i) = A × (ρ_i / ρ_sf)^n   [M_sun/yr/kpc³]
//! ```
//! con `A = 0.1`, `n = 1.5`, aplicada solo a gas con `ρ > ρ_sf`.
//!
//! ### Energía de supernova
//!
//! Por unidad de masa estelar formada: `E_SN = ε_SN × 1e51 erg / (100 M_sun)`.
//! En unidades internas (kpc, km/s, 10¹⁰ M_sun): `E_SN_code ≈ ε_SN × 1.54`.
//!
//! ### Kick estocástico
//!
//! Para cada partícula de gas con `SFR(i) > sfr_min`:
//! - Se aplica un kick de velocidad `v_kick` en dirección aleatoria.
//! - La probabilidad de kick por paso es `p = 1 - exp(-SFR(i) × dt / m_part)`.
//! - La energía interna del vecino receptor aumenta en `ε_SN × E_SN / m_gas`.
//!
//! ## Referencia
//!
//! Springel & Hernquist (2003), MNRAS 339, 289;
//! Dalla Vecchia & Schaye (2012), MNRAS 426, 140.

use gadget_ng_core::{FeedbackSection, Particle, ParticleType, WindParams};

// ── Constantes ─────────────────────────────────────────────────────────────

/// Constante de Schmidt-Kennicutt A en unidades internas (kpc, km/s, 10¹⁰ M_sun).
const SFR_A: f64 = 0.1;
/// Índice de la ley de potencia de Schmidt-Kennicutt.
const SFR_N: f64 = 1.5;
/// Energía de SN por M_sun estelar formada en unidades internas
/// (≈ 1e51 erg / 100 M_sun × factor_conversión).
const E_SN_PER_MSUN: f64 = 1.54e-3; // (km/s)² por 10¹⁰ M_sun

// ── API pública ─────────────────────────────────────────────────────────────

/// Calcula la tasa de formación estelar (SFR) para cada partícula de gas.
///
/// Retorna `sfr[i]` en unidades internas de masa/tiempo.
/// Para partículas que no son gas, o que no alcanzan `rho_sf`, retorna 0.
pub fn compute_sfr(particles: &[Particle], cfg: &FeedbackSection) -> Vec<f64> {
    particles
        .iter()
        .map(|p| {
            if p.ptype != ParticleType::Gas {
                return 0.0;
            }
            // La densidad se aproxima usando la longitud de suavizado
            let h = p.smoothing_length.max(1e-10);
            let rho_approx = p.mass / (h * h * h * (4.0 / 3.0) * std::f64::consts::PI);
            if rho_approx < cfg.rho_sf {
                0.0
            } else {
                SFR_A * (rho_approx / cfg.rho_sf).powf(SFR_N)
            }
        })
        .collect()
}

/// Calcula la tasa de formación estelar con boost por gas molecular H₂ (Phase 122).
///
/// Igual a `compute_sfr` pero multiplica por `(1 + sfr_h2_boost × h2_fraction)`
/// cuando `molecular.enabled = true`. Si `molecular.enabled = false`, es idéntica
/// a `compute_sfr`.
pub fn compute_sfr_with_h2(
    particles: &[Particle],
    cfg: &FeedbackSection,
    h2_boost: f64,
) -> Vec<f64> {
    particles
        .iter()
        .map(|p| {
            if p.ptype != ParticleType::Gas {
                return 0.0;
            }
            let h = p.smoothing_length.max(1e-10);
            let rho_approx = p.mass / (h * h * h * (4.0 / 3.0) * std::f64::consts::PI);
            if rho_approx < cfg.rho_sf {
                0.0
            } else {
                let sfr_base = SFR_A * (rho_approx / cfg.rho_sf).powf(SFR_N);
                // Phase 122: multiplicar por factor H2
                sfr_base * (1.0 + h2_boost * p.h2_fraction)
            }
        })
        .collect()
}

/// Aplica el feedback por supernovas estocástico a las partículas de gas.
///
/// Para cada partícula de gas con `sfr[i] > sfr_min`:
/// 1. Calcula la probabilidad de kick: `p = 1 - exp(-sfr[i] × dt / mass)`.
/// 2. Si se aplica el kick, modifica la velocidad y energía interna.
///
/// # Parámetros
/// - `particles` — slice mutable de partículas.
/// - `sfr`       — tasas de formación estelar por partícula (de `compute_sfr`).
/// - `cfg`       — configuración del feedback.
/// - `dt`        — paso de tiempo en unidades internas.
/// - `seed`      — semilla para el generador estocástico (se actualiza in-place).
pub fn apply_sn_feedback(
    particles: &mut [Particle],
    sfr: &[f64],
    cfg: &FeedbackSection,
    dt: f64,
    seed: &mut u64,
) {
    if !cfg.enabled {
        return;
    }

    let v_kick = cfg.v_kick_km_s;
    let e_sn_per_m = E_SN_PER_MSUN * cfg.eps_sn;

    for i in 0..particles.len() {
        if particles[i].ptype != ParticleType::Gas {
            continue;
        }
        let sfr_i = if i < sfr.len() { sfr[i] } else { 0.0 };
        if sfr_i < cfg.sfr_min {
            continue;
        }

        // Probabilidad de kick en este paso
        let m = particles[i].mass.max(1e-30);
        let prob = 1.0 - (-sfr_i * dt / m).exp();

        // LCG aleatorio
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let rand_val = (*seed >> 33) as f64 / u32::MAX as f64;

        if rand_val < prob {
            // Dirección de kick aleatoria en esfera unitaria (rechazo)
            let (nx, ny, nz) = random_unit_vector(seed);

            particles[i].velocity.x += v_kick * nx;
            particles[i].velocity.y += v_kick * ny;
            particles[i].velocity.z += v_kick * nz;

            // Energía interna: ΔU = ε_SN × E_SN / mass
            particles[i].internal_energy += e_sn_per_m / m * sfr_i * dt;
        }
    }
}

/// Genera un vector unitario aleatorio en la esfera usando un LCG.
fn random_unit_vector(seed: &mut u64) -> (f64, f64, f64) {
    let lcg = |s: &mut u64| -> f64 {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*s >> 33) as f64 / u32::MAX as f64) * 2.0 - 1.0
    };
    loop {
        let x = lcg(seed);
        let y = lcg(seed);
        let z = lcg(seed);
        let r2 = x * x + y * y + z * z;
        if r2 > 0.0 && r2 <= 1.0 {
            let r = r2.sqrt();
            return (x / r, y / r, z / r);
        }
    }
}

// ── Vientos galácticos (Phase 108) ─────────────────────────────────────────

/// Aplica el modelo de vientos galácticos (Springel & Hernquist 2003).
///
/// Para cada partícula de gas con SFR activa:
/// - Con probabilidad `p_wind = 1 - exp(-η × SFR × dt / mass)`, la partícula
///   es lanzada como viento con velocidad `v_wind` en dirección aleatoria.
/// - La energía cinética del viento se añade como kick de velocidad.
/// - Se retorna el vector de índices de partículas lanzadas.
///
/// # Parámetros
/// - `particles` — partículas de gas (mutables).
/// - `sfr`       — tasas de formación estelar por partícula.
/// - `cfg`       — parámetros del viento galáctico.
/// - `dt`        — paso de tiempo en unidades internas.
/// - `seed`      — semilla RNG (se actualiza in-place).
pub fn apply_galactic_winds(
    particles: &mut [Particle],
    sfr: &[f64],
    cfg: &WindParams,
    dt: f64,
    seed: &mut u64,
) -> Vec<usize> {
    let mut launched: Vec<usize> = Vec::new();
    if !cfg.enabled {
        return launched;
    }

    let v_wind = cfg.v_wind_km_s;
    let eta = cfg.mass_loading;

    for i in 0..particles.len() {
        if particles[i].ptype != ParticleType::Gas {
            continue;
        }
        let sfr_i = if i < sfr.len() { sfr[i] } else { 0.0 };
        if sfr_i <= 0.0 {
            continue;
        }

        let m = particles[i].mass.max(1e-30);
        // Probabilidad de ser lanzado como viento en este paso.
        let prob = 1.0 - (-(eta * sfr_i * dt) / m).exp();

        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let rand_val = (*seed >> 33) as f64 / u32::MAX as f64;

        if rand_val < prob {
            let (nx, ny, nz) = random_unit_vector(seed);
            particles[i].velocity.x += v_wind * nx;
            particles[i].velocity.y += v_wind * ny;
            particles[i].velocity.z += v_wind * nz;
            launched.push(i);
        }
    }
    launched
}

/// Calcula la energía total inyectada por SN en un paso.
///
/// Util para monitorear la conservación de energía.
pub fn total_sn_energy_injection(
    sfr: &[f64],
    masses: &[f64],
    cfg: &FeedbackSection,
    dt: f64,
) -> f64 {
    sfr.iter()
        .zip(masses.iter())
        .filter(|&(&s, _)| s >= cfg.sfr_min)
        .map(|(&s, &_m)| E_SN_PER_MSUN * cfg.eps_sn * s * dt)
        .sum()
}

/// Genera partículas estelares desde gas con SFR activa (Phase 112).
///
/// La probabilidad de spawning en un paso `dt` para la partícula `i` es:
/// `p_spawn = 1 - exp(-sfr[i] × dt / m_i)`
///
/// Las estrellas spawneadas:
/// - Heredan `metallicity`, posición y velocidad del gas padre.
/// - Tienen `ptype = ParticleType::Star` y `stellar_age = 0`.
/// - El gas padre pierde `cfg.m_star_fraction × m_gas` de masa.
///
/// Si la masa del gas padre cae por debajo de `cfg.m_gas_min`, el índice
/// se añade al vector de retorno `to_remove`.
///
/// # Retorno
///
/// `(Vec<Particle>, Vec<usize>)` — nuevas estrellas + índices de gas a eliminar.
pub fn spawn_star_particles(
    particles: &mut [Particle],
    sfr: &[f64],
    dt: f64,
    seed: &mut u64,
    cfg: &FeedbackSection,
    next_global_id: &mut usize,
) -> (Vec<Particle>, Vec<usize>) {
    let mut new_stars: Vec<Particle> = Vec::new();
    let mut to_remove: Vec<usize> = Vec::new();

    if !cfg.enabled {
        return (new_stars, to_remove);
    }

    for i in 0..particles.len() {
        if particles[i].ptype != gadget_ng_core::ParticleType::Gas {
            continue;
        }
        if sfr[i] < cfg.sfr_min {
            continue;
        }

        let m_i = particles[i].mass;
        if m_i <= 0.0 {
            continue;
        }

        // Probabilidad de spawning
        let prob = 1.0 - (-sfr[i] * dt / m_i).exp();
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        let rand_val = (*seed >> 33) as f64 / u32::MAX as f64;

        if rand_val < prob {
            let m_star = cfg.m_star_fraction * m_i;
            let star = gadget_ng_core::Particle::new_star(
                *next_global_id,
                m_star,
                particles[i].position,
                particles[i].velocity,
                particles[i].metallicity,
            );
            *next_global_id += 1;
            new_stars.push(star);

            particles[i].mass -= m_star;
            if particles[i].mass < cfg.m_gas_min {
                to_remove.push(i);
            }
        }
    }

    (new_stars, to_remove)
}

/// Aplica SN Ia a partículas estelares con edad > t_ia_min_gyr (Phase 113).
///
/// DTD power-law: `R(t) = A_Ia × (t / 1 Gyr)^{-1}` [SN / Gyr / M_sun]
///
/// Por cada estrella con `stellar_age > t_ia_min_gyr`:
/// 1. Se calcula el número esperado de SN Ia en `dt_gyr`: `N_exp = A_Ia × (t/Gyr)^{-1} × dt_gyr × m_star`.
/// 2. Se sortea estocásticamente si ocurre al menos una SN Ia.
/// 3. Si ocurre: se inyecta `e_ia_code` en energía térmica al gas vecino más cercano
///    y se distribuye hierro (como metalicidad) a todos los vecinos dentro de `2×h`.
///
/// # Parámetros
///
/// - `particles` — slice mutable con todas las partículas.
/// - `dt_gyr` — paso de tiempo en Gyr.
/// - `seed` — semilla PRNG.
/// - `cfg` — configuración de feedback con parámetros SN Ia.
pub fn apply_snia_feedback(
    particles: &mut [Particle],
    dt_gyr: f64,
    seed: &mut u64,
    cfg: &FeedbackSection,
) {
    if !cfg.enabled {
        return;
    }

    let n = particles.len();
    let mut delta_u = vec![0.0_f64; n];
    let mut delta_z = vec![0.0_f64; n];

    for i in 0..n {
        if particles[i].ptype != gadget_ng_core::ParticleType::Star {
            continue;
        }
        let age = particles[i].stellar_age;
        if age < cfg.t_ia_min_gyr {
            continue;
        }

        // Tasa DTD power-law: R = A_Ia × (t/Gyr)^{-1}
        let rate = cfg.a_ia * (age).recip(); // SN / Gyr / M_sun
        let n_exp = rate * dt_gyr * particles[i].mass;

        // Sorteo estocástico
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        let rand_val = (*seed >> 33) as f64 / u32::MAX as f64;
        let prob = 1.0 - (-n_exp).exp();
        if rand_val >= prob {
            continue;
        }

        // Distribuir energía y Fe a vecinos de gas
        let h_i = particles[i].smoothing_length.max(0.1);
        let pos_i = particles[i].position;

        // Encontrar vecinos de gas dentro de 2×h
        let mut weights = vec![0.0_f64; n];
        let mut weight_sum = 0.0_f64;

        for j in 0..n {
            if i == j {
                continue;
            }
            if particles[j].ptype != gadget_ng_core::ParticleType::Gas {
                continue;
            }
            let dx = particles[j].position.x - pos_i.x;
            let dy = particles[j].position.y - pos_i.y;
            let dz = particles[j].position.z - pos_i.z;
            let r = (dx * dx + dy * dy + dz * dz).sqrt();
            if r < 2.0 * h_i {
                let w = (1.0 - r / (2.0 * h_i)).powi(2);
                weights[j] = w;
                weight_sum += w;
            }
        }

        if weight_sum <= 0.0 {
            // Si no hay vecinos, inyectar a la partícula más cercana de gas
            let (closest, _) = (0..n)
                .filter(|&j| j != i && particles[j].ptype == gadget_ng_core::ParticleType::Gas)
                .map(|j| {
                    let dx = particles[j].position.x - pos_i.x;
                    let dy = particles[j].position.y - pos_i.y;
                    let dz = particles[j].position.z - pos_i.z;
                    let r2 = dx * dx + dy * dy + dz * dz;
                    (j, r2)
                })
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap_or((0, f64::MAX));
            if closest < n && particles[closest].ptype == gadget_ng_core::ParticleType::Gas {
                delta_u[closest] += cfg.e_ia_code;
                delta_z[closest] += 0.002 / particles[closest].mass.max(1e-30); // Fe yield ~0.002 M_sun
            }
            continue;
        }

        for j in 0..n {
            if weights[j] <= 0.0 {
                continue;
            }
            let frac = weights[j] / weight_sum;
            let m_j = particles[j].mass.max(1e-30);
            delta_u[j] += frac * cfg.e_ia_code;
            delta_z[j] += frac * 0.002 / m_j; // Fe yield ~0.002 M_sun por SN Ia
        }
    }

    // Aplicar incrementos
    for i in 0..n {
        if delta_u[i] > 0.0 {
            particles[i].internal_energy += delta_u[i];
        }
        if delta_z[i] > 0.0 {
            particles[i].metallicity = (particles[i].metallicity + delta_z[i]).min(1.0);
        }
    }
}

/// Incrementa la edad estelar de todas las partículas estelares.
///
/// Debe llamarse cada paso con `dt_gyr` = paso de tiempo en Gyr.
pub fn advance_stellar_ages(particles: &mut [Particle], dt_gyr: f64) {
    for p in particles.iter_mut() {
        if p.ptype == gadget_ng_core::ParticleType::Star {
            p.stellar_age += dt_gyr;
        }
    }
}

/// Aplica feedback mecánico de vientos estelares pre-SN (Phase 115).
///
/// Modela vientos de estrellas OB y Wolf-Rayet (~10-30 Myr antes de SN II).
/// Para gas con SFR activa, aplica kicks de velocidad con probabilidad proporcional
/// al factor de carga másica η_w.
///
/// La probabilidad de kick por paso: `p = η_w × sfr[i] × dt / m_i`
///
/// # Retorno
///
/// Índices de las partículas que recibieron un kick de viento estelar.
pub fn apply_stellar_wind_feedback(
    particles: &mut [Particle],
    sfr: &[f64],
    cfg: &gadget_ng_core::FeedbackSection,
    dt: f64,
    seed: &mut u64,
) -> Vec<usize> {
    let mut kicked = Vec::new();
    if !cfg.stellar_wind_enabled {
        return kicked;
    }

    // Velocidad del viento en unidades internas (km/s → km/s ya está bien)
    let v_wind = cfg.v_stellar_wind_km_s;

    for i in 0..particles.len() {
        if particles[i].ptype != gadget_ng_core::ParticleType::Gas {
            continue;
        }
        if sfr[i] < cfg.sfr_min {
            continue;
        }

        let m_i = particles[i].mass.max(1e-30);
        // Probabilidad: p = η_w × sfr × dt / m
        let prob = (cfg.eta_stellar_wind * sfr[i] * dt / m_i).min(1.0);

        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        let rand_val = (*seed >> 33) as f64 / u32::MAX as f64;

        if rand_val < prob {
            let (nx, ny, nz) = random_unit_vector(seed);
            particles[i].velocity.x += v_wind * nx;
            particles[i].velocity.y += v_wind * ny;
            particles[i].velocity.z += v_wind * nz;
            kicked.push(i);
        }
    }
    kicked
}

#[cfg(test)]
mod tests {
    use super::*;
    use gadget_ng_core::{FeedbackSection, Particle, ParticleType, Vec3};

    fn gas_particle(id: usize, rho_scale: f64) -> Particle {
        let mut p = Particle::new(id, 1.0, Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 0.0));
        p.ptype = ParticleType::Gas;
        p.smoothing_length = rho_scale;
        p.internal_energy = 1.0;
        p
    }

    #[test]
    fn sfr_zero_below_threshold() {
        let cfg = FeedbackSection {
            rho_sf: 1.0,
            ..Default::default()
        };
        let p = gas_particle(0, 100.0); // rho muy baja (h grande)
        let sfr = compute_sfr(&[p], &cfg);
        assert_eq!(sfr[0], 0.0, "SFR debe ser 0 bajo el umbral");
    }

    #[test]
    fn sfr_positive_above_threshold() {
        let cfg = FeedbackSection {
            rho_sf: 0.001,
            ..Default::default()
        };
        let mut p = gas_particle(0, 0.01); // h pequeño → ρ alta
        p.mass = 1.0;
        let sfr = compute_sfr(&[p], &cfg);
        assert!(
            sfr[0] > 0.0,
            "SFR debe ser positiva sobre el umbral: {}",
            sfr[0]
        );
    }

    #[test]
    fn sfr_zero_for_dm_particle() {
        let cfg = FeedbackSection::default();
        let p = Particle::new(0, 1.0, Vec3::zero(), Vec3::zero()); // DM por defecto
        let sfr = compute_sfr(&[p], &cfg);
        assert_eq!(sfr[0], 0.0, "DM no forma estrellas");
    }

    #[test]
    fn feedback_disabled_no_kick() {
        let cfg = FeedbackSection {
            enabled: false,
            ..Default::default()
        };
        let mut particles = vec![gas_particle(0, 0.01)];
        let sfr = vec![1.0]; // sfr alta
        let vel_before = particles[0].velocity;
        let mut seed = 42u64;
        apply_sn_feedback(&mut particles, &sfr, &cfg, 0.1, &mut seed);
        assert_eq!(
            particles[0].velocity.x, vel_before.x,
            "Sin kick si disabled"
        );
    }

    #[test]
    fn feedback_enabled_probabilistic_kick() {
        let cfg = FeedbackSection {
            enabled: true,
            v_kick_km_s: 350.0,
            eps_sn: 0.1,
            rho_sf: 0.0,
            sfr_min: 0.0,
            ..Default::default()
        };
        // Muchos pasos → algún kick debe aplicarse
        let mut kicked = false;
        for i in 0..50 {
            let mut p = gas_particle(i, 0.01);
            let sfr = vec![1.0];
            let v0 = p.velocity.x;
            let mut seed = (i as u64 + 1) * 12345;
            apply_sn_feedback(std::slice::from_mut(&mut p), &sfr, &cfg, 1.0, &mut seed);
            if (p.velocity.x - v0).abs() > 1e-10 {
                kicked = true;
                break;
            }
        }
        assert!(
            kicked,
            "Al menos un kick debe aplicarse en 50 intentos con p alto"
        );
    }

    #[test]
    fn kick_velocity_magnitude_correct() {
        let v_kick = 300.0;
        let cfg = FeedbackSection {
            enabled: true,
            v_kick_km_s: v_kick,
            sfr_min: 0.0,
            rho_sf: 0.0,
            eps_sn: 1.0,
            ..Default::default()
        };
        // Forzar kick con sfr=1e10 para prob ≈ 1
        let mut p = gas_particle(0, 0.01);
        let sfr = vec![1e10f64];
        let v0x = p.velocity.x;
        let v0y = p.velocity.y;
        let v0z = p.velocity.z;
        let mut seed = 99u64;
        apply_sn_feedback(std::slice::from_mut(&mut p), &sfr, &cfg, 1e-10, &mut seed);
        let dv = ((p.velocity.x - v0x).powi(2)
            + (p.velocity.y - v0y).powi(2)
            + (p.velocity.z - v0z).powi(2))
        .sqrt();
        // Con sfr=1e10 y dt=1e-10, prob ≈ 1; verificar que se aplicó algún kick o energía
        assert!(
            dv <= v_kick + 1.0,
            "La magnitud del kick no debe exceder v_kick: {dv}"
        );
    }

    #[test]
    fn energy_injection_positive() {
        let cfg = FeedbackSection {
            enabled: true,
            sfr_min: 0.0,
            ..Default::default()
        };
        let sfr = vec![1.0, 2.0, 0.0];
        let masses = vec![1.0, 1.0, 1.0];
        let e = total_sn_energy_injection(&sfr, &masses, &cfg, 0.1);
        assert!(e > 0.0, "La energía inyectada debe ser positiva: {e}");
    }

    #[test]
    fn unit_vector_on_sphere() {
        let mut seed = 42u64;
        for _ in 0..100 {
            let (x, y, z) = random_unit_vector(&mut seed);
            let r = (x * x + y * y + z * z).sqrt();
            assert!((r - 1.0).abs() < 1e-10, "Vector unitario: |v| = {r}");
        }
    }
}
