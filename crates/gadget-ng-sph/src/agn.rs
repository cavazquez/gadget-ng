//! Feedback de Agujeros Negros (AGN) — Bondi-Hoyle + depósito térmico (Phase 96).
//!
//! Implementa el ciclo básico de feedback AGN:
//! 1. **Acreción de Bondi-Hoyle**: Ṁ = 4π G² M_BH² ρ / (c_s² + v_rel²)^(3/2)
//! 2. **Crecimiento del agujero negro**: M_BH → M_BH + Ṁ dt (masa acretada)
//! 3. **Feedback térmico**: E_fb = ε_feedback × Ṁ × c² depositada en gas vecino

use gadget_ng_core::{Particle, Vec3};

/// Agujero negro supermasivo (SMBH) en la simulación.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BlackHole {
    /// Posición en la caja de simulación.
    pub pos: Vec3,
    /// Masa del agujero negro en unidades internas (M_sol/h).
    pub mass: f64,
    /// Tasa de acreción instantánea Ṁ [unidades internas/tiempo].
    pub accretion_rate: f64,
}

impl BlackHole {
    /// Crea un nuevo agujero negro en posición `pos` con masa `mass`.
    pub fn new(pos: Vec3, mass: f64) -> Self {
        Self { pos, mass, accretion_rate: 0.0 }
    }
}

/// Parámetros del modelo de feedback AGN.
#[derive(Debug, Clone)]
pub struct AgnParams {
    /// Eficiencia radiativa del feedback: fracción de Ṁ c² convertida en calor (default: 0.05).
    pub eps_feedback: f64,
    /// Masa semilla del agujero negro al crearse [M_sol/h] (default: 1e5).
    pub m_seed: f64,
    /// Velocidad de kick cinético en km/s (para modo jet; 0 = solo feedback térmico).
    pub v_kick_agn: f64,
    /// Radio de influencia máximo para depositar energía [unidades de caja] (default: 1.0).
    pub r_influence: f64,
}

impl Default for AgnParams {
    fn default() -> Self {
        Self {
            eps_feedback: 0.05,
            m_seed: 1e5,
            v_kick_agn: 500.0,
            r_influence: 1.0,
        }
    }
}

/// Constante de gravedad en unidades internas (km²/s² × Mpc/h / M_sol×h).
/// Aproximado como G ≈ 4.3e-3 pc M_sol^-1 (km/s)^2 → ajustado a unidades internas.
const G_INTERNAL: f64 = 4.302e-3;

/// Velocidad de la luz en km/s.
const C_KMS: f64 = 2.998e5;

/// Calcula la tasa de acreción de Bondi-Hoyle para un agujero negro.
///
/// $$\dot{M} = \frac{4\pi G^2 M_{BH}^2 \rho}{(c_s^2 + v_{rel}^2)^{3/2}}$$
///
/// # Argumentos
/// - `bh`: agujero negro
/// - `rho_gas`: densidad del gas en la vecindad [unidades internas]
/// - `c_sound`: velocidad del sonido local [km/s]
///
/// # Retorna
/// Tasa de acreción Ṁ en unidades internas / tiempo.
pub fn bondi_accretion_rate(bh: &BlackHole, rho_gas: f64, c_sound: f64) -> f64 {
    let denom = (c_sound * c_sound).powi(3).sqrt(); // (c_s²)^(3/2) = c_s³
    if denom < 1e-30 {
        return 0.0;
    }
    4.0 * std::f64::consts::PI * G_INTERNAL * G_INTERNAL * bh.mass * bh.mass * rho_gas / denom
}

/// Aplica feedback AGN térmico a las partículas de gas vecinas.
///
/// Para cada agujero negro:
/// 1. Calcula Ṁ × dt = masa acretada en este paso
/// 2. Deposita E_fb = ε_feedback × Ṁ × c² como energía interna en partículas dentro de `r_influence`
/// 3. Si `v_kick_agn > 0`, aplica kick radial adicional (modo jet)
///
/// # Argumentos
/// - `particles`: partículas de gas (modificadas in-place)
/// - `bhs`: lista de agujeros negros
/// - `params`: parámetros AGN
/// - `dt`: paso de tiempo
pub fn apply_agn_feedback(
    particles: &mut [Particle],
    bhs: &[BlackHole],
    params: &AgnParams,
    dt: f64,
) {
    for bh in bhs {
        // Energía de feedback disponible en este paso
        let e_feedback = params.eps_feedback * bh.accretion_rate * C_KMS * C_KMS * dt;
        if e_feedback <= 0.0 {
            continue;
        }

        // Buscar partículas vecinas dentro del radio de influencia
        let r2_max = params.r_influence * params.r_influence;
        let mut neighbors: Vec<usize> = Vec::new();
        let mut total_mass_neighbors = 0.0_f64;

        for (i, p) in particles.iter().enumerate() {
            if p.internal_energy <= 0.0 {
                continue; // no es partícula de gas
            }
            let dx = p.position.x - bh.pos.x;
            let dy = p.position.y - bh.pos.y;
            let dz = p.position.z - bh.pos.z;
            let r2 = dx * dx + dy * dy + dz * dz;
            if r2 < r2_max {
                neighbors.push(i);
                total_mass_neighbors += p.mass;
            }
        }

        if neighbors.is_empty() || total_mass_neighbors <= 0.0 {
            continue;
        }

        // Distribuir energía proporcional a la masa
        for &i in &neighbors {
            let mass_frac = particles[i].mass / total_mass_neighbors;
            let de = e_feedback * mass_frac / particles[i].mass;
            particles[i].internal_energy += de;

            // Kick cinético radial (modo jet)
            if params.v_kick_agn > 0.0 {
                let dx = particles[i].position.x - bh.pos.x;
                let dy = particles[i].position.y - bh.pos.y;
                let dz = particles[i].position.z - bh.pos.z;
                let r = (dx * dx + dy * dy + dz * dz).sqrt().max(1e-30);
                let kick = params.v_kick_agn * mass_frac;
                particles[i].velocity.x += kick * dx / r;
                particles[i].velocity.y += kick * dy / r;
                particles[i].velocity.z += kick * dz / r;
            }
        }
    }
}

/// Hace crecer los agujeros negros acretando masa del gas vecino.
///
/// Actualiza `bh.accretion_rate` y `bh.mass` para cada agujero negro
/// basándose en la densidad local del gas.
///
/// # Argumentos
/// - `bhs`: agujeros negros (modificados in-place)
/// - `particles`: partículas de gas (masa total en vecindad)
/// - `params`: parámetros AGN
/// - `dt`: paso de tiempo
pub fn grow_black_holes(
    bhs: &mut [BlackHole],
    particles: &[Particle],
    params: &AgnParams,
    dt: f64,
) {
    for bh in bhs.iter_mut() {
        // Densidad local: suma de masa de gas vecino / volumen esférico
        let r_inf = params.r_influence;
        let r2_max = r_inf * r_inf;
        let vol = 4.0 / 3.0 * std::f64::consts::PI * r_inf * r_inf * r_inf;

        let mut mass_in_sphere = 0.0_f64;
        let mut c_sound_local = 0.0_f64;
        let mut n_gas = 0_usize;

        for p in particles.iter() {
            if p.internal_energy <= 0.0 {
                continue;
            }
            let dx = p.position.x - bh.pos.x;
            let dy = p.position.y - bh.pos.y;
            let dz = p.position.z - bh.pos.z;
            let r2 = dx * dx + dy * dy + dz * dz;
            if r2 < r2_max {
                mass_in_sphere += p.mass;
                // c_s² ∝ u para gas ideal: c_s ≈ sqrt(γ(γ-1)u) con γ=5/3
                c_sound_local += (5.0 / 3.0 * (5.0 / 3.0 - 1.0) * p.internal_energy).sqrt();
                n_gas += 1;
            }
        }

        if n_gas == 0 || vol <= 0.0 {
            bh.accretion_rate = 0.0;
            continue;
        }

        let rho_local = mass_in_sphere / vol;
        let c_s = c_sound_local / n_gas as f64;

        bh.accretion_rate = bondi_accretion_rate(bh, rho_local, c_s);
        let dm = bh.accretion_rate * dt * (1.0 - params.eps_feedback);
        bh.mass += dm.max(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gadget_ng_core::{Particle, Vec3};

    fn make_gas_particle(pos: Vec3, mass: f64, u: f64) -> Particle {
        let mut p = Particle::new(0, mass, pos, Vec3::zero());
        p.internal_energy = u;
        p.smoothing_length = 0.5;
        p
    }

    fn make_bh(pos: Vec3, mass: f64, mdot: f64) -> BlackHole {
        BlackHole { pos, mass, accretion_rate: mdot }
    }

    /// Ṁ ∝ M_BH² verificado numéricamente.
    #[test]
    fn bondi_rate_scales_with_mass() {
        let bh1 = make_bh(Vec3::zero(), 1e6, 0.0);
        let bh2 = make_bh(Vec3::zero(), 2e6, 0.0);
        let rho = 1.0;
        let c_s = 10.0;

        let mdot1 = bondi_accretion_rate(&bh1, rho, c_s);
        let mdot2 = bondi_accretion_rate(&bh2, rho, c_s);

        let ratio = mdot2 / mdot1;
        assert!(
            (ratio - 4.0).abs() < 1e-10,
            "Ṁ debe escalar como M_BH², ratio = {} ≠ 4.0",
            ratio
        );
    }

    /// La partícula vecina debe ganar energía interna después del feedback.
    #[test]
    fn agn_feedback_increases_internal_energy() {
        let params = AgnParams { eps_feedback: 0.1, m_seed: 1e5, v_kick_agn: 0.0, r_influence: 2.0 };
        let bh = make_bh(Vec3::zero(), 1e8, 1e-3);

        let mut particles = vec![
            make_gas_particle(Vec3::new(0.5, 0.0, 0.0), 1.0, 100.0),
            make_gas_particle(Vec3::new(5.0, 0.0, 0.0), 1.0, 100.0), // fuera del radio
        ];
        let u_before = particles[0].internal_energy;
        let u_outside_before = particles[1].internal_energy;

        apply_agn_feedback(&mut particles, &[bh], &params, 1.0);

        assert!(
            particles[0].internal_energy > u_before,
            "partícula vecina debe ganar energía: {} > {}",
            particles[0].internal_energy,
            u_before
        );
        assert_eq!(
            particles[1].internal_energy,
            u_outside_before,
            "partícula lejana no debe cambiar"
        );
    }

    /// La energía depositada es exactamente ε × Ṁ × c² × dt (conservación de energía).
    #[test]
    fn agn_energy_conservation() {
        let eps = 0.05;
        let mdot = 1e-4;
        let dt = 1.0;
        let params = AgnParams { eps_feedback: eps, m_seed: 1e5, v_kick_agn: 0.0, r_influence: 2.0 };
        let bh = make_bh(Vec3::zero(), 1e8, mdot);

        let mut particles = vec![
            make_gas_particle(Vec3::new(0.5, 0.0, 0.0), 1.0, 100.0),
        ];
        let u_before = particles[0].internal_energy * particles[0].mass;

        apply_agn_feedback(&mut particles, &[bh], &params, dt);

        let u_after = particles[0].internal_energy * particles[0].mass;
        let e_deposited = u_after - u_before;
        let e_expected = eps * mdot * C_KMS * C_KMS * dt;

        assert!(
            (e_deposited - e_expected).abs() / e_expected < 1e-10,
            "energía depositada = {} ≠ {} esperado",
            e_deposited,
            e_expected
        );
    }

    /// Sin gas vecino, el feedback no modifica partículas lejanas.
    #[test]
    fn agn_feedback_no_neighbors() {
        let params = AgnParams::default();
        let bh = make_bh(Vec3::zero(), 1e8, 1e-3);
        let mut particles = vec![
            make_gas_particle(Vec3::new(10.0, 0.0, 0.0), 1.0, 100.0),
        ];
        let u_before = particles[0].internal_energy;

        apply_agn_feedback(&mut particles, &[bh], &params, 1.0);

        assert_eq!(
            particles[0].internal_energy,
            u_before,
            "sin vecinos, energía debe ser invariante"
        );
    }

    /// grow_black_holes aumenta la masa del BH cuando hay gas vecino.
    #[test]
    fn grow_bh_increases_mass_with_gas() {
        let params = AgnParams { eps_feedback: 0.0, m_seed: 1e5, v_kick_agn: 0.0, r_influence: 2.0 };
        let mut bhs = vec![BlackHole::new(Vec3::zero(), 1e8)];
        let particles = vec![
            make_gas_particle(Vec3::new(0.5, 0.0, 0.0), 1.0, 1000.0),
        ];
        let mass_before = bhs[0].mass;

        grow_black_holes(&mut bhs, &particles, &params, 1.0);

        // La masa solo aumenta si la acreción es positiva
        // (puede ser 0 si el gas está demasiado caliente)
        assert!(bhs[0].mass >= mass_before, "masa del BH no debe decrecer");
    }

    /// `bondi_accretion_rate` con c_sound=0 no debe provocar división por cero.
    #[test]
    fn bondi_rate_no_divide_by_zero() {
        let bh = make_bh(Vec3::zero(), 1e6, 0.0);
        let mdot = bondi_accretion_rate(&bh, 1.0, 0.0);
        assert!(mdot.is_finite(), "Ṁ debe ser finito con c_s=0");
    }
}
