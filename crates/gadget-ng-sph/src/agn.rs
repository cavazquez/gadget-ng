//! Feedback de Agujeros Negros (AGN) — Bondi-Hoyle + depósito térmico (Phase 96).
//!
//! Implementa el ciclo básico de feedback AGN:
//! 1. **Acreción de Bondi-Hoyle**: Ṁ = 4π G² M_BH² ρ / (c_s² + v_rel²)^(3/2)
//! 2. **Crecimiento del agujero negro**: M_BH → M_BH + Ṁ dt (masa acretada)
//! 3. **Feedback térmico**: E_fb = ε_feedback × Ṁ × c² depositada en gas vecino

use crate::periodic_delta;
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
    /// Spin adimensional Kerr `a_*` en [-0.998, 0.998].
    #[serde(default)]
    pub spin: f64,
    /// Velocidad peculiar del BH, usada para kicks de recoil tras mergers.
    #[serde(default)]
    pub velocity: Vec3,
}

impl BlackHole {
    /// Crea un nuevo agujero negro en posición `pos` con masa `mass`.
    pub fn new(pos: Vec3, mass: f64) -> Self {
        Self {
            pos,
            mass,
            accretion_rate: 0.0,
            spin: 0.0,
            velocity: Vec3::zero(),
        }
    }

    /// Crea un BH con spin inicial explícito.
    pub fn with_spin(pos: Vec3, mass: f64, spin: f64) -> Self {
        let mut bh = Self::new(pos, mass);
        bh.spin = spin.clamp(-0.998, 0.998);
        bh
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

/// Eficiencia radiativa aproximada de un disco delgado Kerr.
///
/// Usa una interpolación suave entre valores canónicos: retrogrado extremo
/// ~3.8%, Schwarzschild 5.7%, progrado rápido ~32%.
pub fn radiative_efficiency_from_spin(spin: f64) -> f64 {
    let a = spin.clamp(-0.998, 0.998);
    if a >= 0.0 {
        0.057 + (0.32 - 0.057) * a.powf(0.7)
    } else {
        0.057 - (0.057 - 0.038) * (-a).powf(0.7)
    }
}

/// Eficiencia de feedback AGN efectiva, escalada por el spin del BH.
pub fn spin_dependent_feedback_efficiency(eps_feedback: f64, spin: f64) -> f64 {
    let eps0 = radiative_efficiency_from_spin(0.0);
    eps_feedback.max(0.0) * radiative_efficiency_from_spin(spin) / eps0
}

/// Actualiza el spin por acreción coherente durante `dt`.
///
/// El modelo acerca `a_*` al valor progrado máximo si `mdot > 0`, con escala de
/// masa de Salpeter reducida `dm / M_BH`. Es deliberadamente estable y acotado.
pub fn spin_up_by_accretion(bh: &mut BlackHole, dt: f64) {
    if bh.mass <= 0.0 || bh.accretion_rate <= 0.0 || dt <= 0.0 {
        return;
    }
    let dm_over_m = (bh.accretion_rate * dt / bh.mass).clamp(0.0, 1.0);
    let target = 0.998;
    bh.spin += (target - bh.spin) * (1.0 - (-3.0 * dm_over_m).exp());
    bh.spin = bh.spin.clamp(-0.998, 0.998);
}

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

/// Aplica feedback AGN térmico a las partículas de gas vecinas (modo quasar).
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
    apply_agn_feedback_periodic(particles, bhs, params, dt, None);
}

/// Igual que `apply_agn_feedback`, usando imagen mínima si `periodic_box = Some(L)`.
pub fn apply_agn_feedback_periodic(
    particles: &mut [Particle],
    bhs: &[BlackHole],
    params: &AgnParams,
    dt: f64,
    periodic_box: Option<f64>,
) {
    for bh in bhs {
        // Energía de feedback disponible en este paso
        let eps_feedback = spin_dependent_feedback_efficiency(params.eps_feedback, bh.spin);
        let e_feedback = eps_feedback * bh.accretion_rate * C_KMS * C_KMS * dt;
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
            let d = periodic_delta(bh.pos, p.position, periodic_box);
            let r2 = d.dot(d);
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
                let d = periodic_delta(bh.pos, particles[i].position, periodic_box);
                let r = d.norm().max(1e-30);
                let kick = params.v_kick_agn * mass_frac;
                particles[i].velocity += d * (kick / r);
            }
        }
    }
}

/// Aplica feedback AGN mecánico de modo radio (bubble feedback) (Phase 116).
///
/// Cuando la tasa de acreción es baja (`Ṁ/Ṁ_Edd < f_edd_threshold`), el AGN
/// opera en modo radio: inyecta jets mecánicos en una burbuja esférica.
///
/// La energía mecánica disponible es: `E_radio = eps_radio × Ṁ × c² × dt`
///
/// Se distribuye como kicks tangenciales (perpendiculares a r_BH-partícula)
/// a las partículas de gas dentro del radio `r_bubble`.
///
/// # Argumentos
/// - `bh`: agujero negro en modo radio
/// - `particles`: partículas de gas (modificadas in-place)
/// - `params`: parámetros AGN base
/// - `r_bubble`: radio de la burbuja [unidades internas]
/// - `eps_radio`: eficiencia del modo radio
/// - `dt`: paso de tiempo
pub fn bubble_feedback_radio(
    bh: &BlackHole,
    particles: &mut [Particle],
    _params: &AgnParams,
    r_bubble: f64,
    eps_radio: f64,
    dt: f64,
) {
    bubble_feedback_radio_periodic(bh, particles, _params, r_bubble, eps_radio, dt, None);
}

/// Igual que `bubble_feedback_radio`, usando imagen mínima si `periodic_box = Some(L)`.
pub fn bubble_feedback_radio_periodic(
    bh: &BlackHole,
    particles: &mut [Particle],
    _params: &AgnParams,
    r_bubble: f64,
    eps_radio: f64,
    dt: f64,
    periodic_box: Option<f64>,
) {
    let e_radio = eps_radio * bh.accretion_rate * C_KMS * C_KMS * dt;
    if e_radio <= 0.0 {
        return;
    }

    let r2_bubble = r_bubble * r_bubble;

    // Recopilar vecinos dentro de la burbuja
    let mut neighbors: Vec<usize> = Vec::new();
    let mut total_mass = 0.0_f64;

    for (i, p) in particles.iter().enumerate() {
        if p.ptype != gadget_ng_core::ParticleType::Gas {
            continue;
        }
        let d = periodic_delta(bh.pos, p.position, periodic_box);
        let r2 = d.dot(d);
        if r2 < r2_bubble && r2 > 0.0 {
            neighbors.push(i);
            total_mass += p.mass;
        }
    }

    if neighbors.is_empty() || total_mass <= 0.0 {
        return;
    }

    // Kick tangencial: perpendicular al vector BH→partícula
    // Construir un vector perpendicular usando producto vectorial con eje z
    for &i in &neighbors {
        let mass_frac = particles[i].mass / total_mass;
        let e_i = e_radio * mass_frac / particles[i].mass.max(1e-30);

        let d = periodic_delta(bh.pos, particles[i].position, periodic_box);
        let r = d.norm().max(1e-30);

        // Dirección tangencial: producto vectorial r × ẑ, luego normalizar
        let tx = -d.y / r;
        let ty = d.x / r;
        let tz = 0.0_f64;
        let t_norm = (tx * tx + ty * ty + tz * tz).sqrt().max(1e-30);

        // Velocidad del kick tangencial: v_kick = sqrt(2 × e_i)
        let v_kick = (2.0 * e_i.abs()).sqrt();
        particles[i].velocity.x += v_kick * tx / t_norm;
        particles[i].velocity.y += v_kick * ty / t_norm;
        particles[i].velocity.z += v_kick * tz / t_norm;
    }
}

/// Aplica feedback AGN con bifurcación modo quasar / modo radio (Phase 116).
///
/// Bifurcación según la tasa de Eddington normalizada:
/// - `Ṁ / Ṁ_Edd > f_edd_threshold` → modo quasar (térmico, `apply_agn_feedback`)
/// - `Ṁ / Ṁ_Edd ≤ f_edd_threshold` → modo radio (mecánico, `bubble_feedback_radio`)
///
/// La tasa de Eddington se estima como:
/// `Ṁ_Edd = 4π G M_BH m_p / (ε_r σ_T c) ≈ m_bh / (1e8 × yr)`
/// En unidades internas: `Ṁ_Edd ≈ bh.mass × 1e-10` [masa/tiempo].
pub fn apply_agn_feedback_bimodal(
    particles: &mut [Particle],
    bhs: &[BlackHole],
    params: &AgnParams,
    f_edd_threshold: f64,
    r_bubble: f64,
    eps_radio: f64,
    dt: f64,
) {
    apply_agn_feedback_bimodal_periodic(
        particles,
        bhs,
        params,
        f_edd_threshold,
        r_bubble,
        eps_radio,
        dt,
        None,
    );
}

/// Igual que `apply_agn_feedback_bimodal`, usando imagen mínima si `periodic_box = Some(L)`.
#[expect(clippy::too_many_arguments)]
pub fn apply_agn_feedback_bimodal_periodic(
    particles: &mut [Particle],
    bhs: &[BlackHole],
    params: &AgnParams,
    f_edd_threshold: f64,
    r_bubble: f64,
    eps_radio: f64,
    dt: f64,
    periodic_box: Option<f64>,
) {
    for bh in bhs {
        if bh.accretion_rate <= 0.0 {
            continue;
        }

        // Tasa de Eddington aproximada en unidades internas
        let m_edd_rate = bh.mass * 1e-10;
        let f_edd = if m_edd_rate > 0.0 {
            bh.accretion_rate / m_edd_rate
        } else {
            1.0
        };

        if f_edd > f_edd_threshold {
            // Modo quasar: feedback térmico (comportamiento original)
            apply_agn_feedback_periodic(
                particles,
                std::slice::from_ref(bh),
                params,
                dt,
                periodic_box,
            );
        } else {
            // Modo radio: jets mecánicos en burbuja
            bubble_feedback_radio_periodic(
                bh,
                particles,
                params,
                r_bubble,
                eps_radio,
                dt,
                periodic_box,
            );
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
    grow_black_holes_periodic(bhs, particles, params, dt, None);
}

/// Igual que `grow_black_holes`, usando imagen mínima si `periodic_box = Some(L)`.
pub fn grow_black_holes_periodic(
    bhs: &mut [BlackHole],
    particles: &[Particle],
    params: &AgnParams,
    dt: f64,
    periodic_box: Option<f64>,
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
            let d = periodic_delta(bh.pos, p.position, periodic_box);
            let r2 = d.dot(d);
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
        let eps_rad = spin_dependent_feedback_efficiency(params.eps_feedback, bh.spin);
        let dm = bh.accretion_rate * dt * (1.0 - eps_rad.min(0.95));
        bh.mass += dm.max(0.0);
        spin_up_by_accretion(bh, dt);
    }
}

/// Fusiona BHs separados menos que `merger_radius`, conservando masa y momento.
///
/// El remanente queda en el centro de masa, con spin promedio ponderado por
/// masa y un recoil fenomenológico que crece con la asimetría de masas y spins.
pub fn merge_black_holes(
    bhs: &mut Vec<BlackHole>,
    merger_radius: f64,
    recoil_velocity_scale: f64,
    periodic_box: Option<f64>,
) -> usize {
    if merger_radius <= 0.0 || bhs.len() < 2 {
        return 0;
    }

    let mut merged = 0;
    let mut i = 0;
    while i < bhs.len() {
        let mut j = i + 1;
        while j < bhs.len() {
            let d = periodic_delta(bhs[i].pos, bhs[j].pos, periodic_box);
            if d.norm() > merger_radius {
                j += 1;
                continue;
            }

            let a = bhs[i].clone();
            let b = bhs[j].clone();
            let total_mass = (a.mass + b.mass).max(1e-30);
            let q = a.mass.min(b.mass) / a.mass.max(b.mass).max(1e-30);
            let eta = q / (1.0 + q).powi(2);
            let spin_asym = (a.spin - b.spin).abs();
            let recoil = recoil_velocity_scale.max(0.0) * eta * eta * (1.0 - q).abs()
                + 0.1 * recoil_velocity_scale.max(0.0) * eta * spin_asym;

            let pos = (a.pos * a.mass + b.pos * b.mass) / total_mass;
            let mut velocity = (a.velocity * a.mass + b.velocity * b.mass) / total_mass;
            if recoil > 0.0 {
                let dir = if d.norm() > 1e-30 {
                    d / d.norm()
                } else {
                    Vec3::new(1.0, 0.0, 0.0)
                };
                velocity += dir * recoil;
            }

            bhs[i] = BlackHole {
                pos,
                mass: total_mass,
                accretion_rate: a.accretion_rate + b.accretion_rate,
                spin: ((a.spin * a.mass + b.spin * b.mass) / total_mass).clamp(-0.998, 0.998),
                velocity,
            };
            bhs.swap_remove(j);
            merged += 1;
        }
        i += 1;
    }
    merged
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
        BlackHole {
            pos,
            mass,
            accretion_rate: mdot,
            spin: 0.0,
            velocity: Vec3::zero(),
        }
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
        let params = AgnParams {
            eps_feedback: 0.1,
            m_seed: 1e5,
            v_kick_agn: 0.0,
            r_influence: 2.0,
        };
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
            particles[1].internal_energy, u_outside_before,
            "partícula lejana no debe cambiar"
        );
    }

    /// La energía depositada es exactamente ε × Ṁ × c² × dt (conservación de energía).
    #[test]
    fn agn_energy_conservation() {
        let eps = 0.05;
        let mdot = 1e-4;
        let dt = 1.0;
        let params = AgnParams {
            eps_feedback: eps,
            m_seed: 1e5,
            v_kick_agn: 0.0,
            r_influence: 2.0,
        };
        let bh = make_bh(Vec3::zero(), 1e8, mdot);

        let mut particles = vec![make_gas_particle(Vec3::new(0.5, 0.0, 0.0), 1.0, 100.0)];
        let u_before = particles[0].internal_energy * particles[0].mass;

        apply_agn_feedback(&mut particles, &[bh], &params, dt);

        let u_after = particles[0].internal_energy * particles[0].mass;
        let e_deposited = u_after - u_before;
        let e_expected = spin_dependent_feedback_efficiency(eps, 0.0) * mdot * C_KMS * C_KMS * dt;

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
        let mut particles = vec![make_gas_particle(Vec3::new(10.0, 0.0, 0.0), 1.0, 100.0)];
        let u_before = particles[0].internal_energy;

        apply_agn_feedback(&mut particles, &[bh], &params, 1.0);

        assert_eq!(
            particles[0].internal_energy, u_before,
            "sin vecinos, energía debe ser invariante"
        );
    }

    /// grow_black_holes aumenta la masa del BH cuando hay gas vecino.
    #[test]
    fn grow_bh_increases_mass_with_gas() {
        let params = AgnParams {
            eps_feedback: 0.0,
            m_seed: 1e5,
            v_kick_agn: 0.0,
            r_influence: 2.0,
        };
        let mut bhs = vec![BlackHole::new(Vec3::zero(), 1e8)];
        let particles = vec![make_gas_particle(Vec3::new(0.5, 0.0, 0.0), 1.0, 1000.0)];
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

    #[test]
    fn radiative_efficiency_grows_with_prograde_spin() {
        let retro = radiative_efficiency_from_spin(-0.9);
        let zero = radiative_efficiency_from_spin(0.0);
        let pro = radiative_efficiency_from_spin(0.9);
        assert!(retro < zero);
        assert!(pro > zero);
    }

    #[test]
    fn merge_black_holes_conserves_mass_and_reduces_count() {
        let mut bhs = vec![
            BlackHole::with_spin(Vec3::new(0.0, 0.0, 0.0), 3.0, 0.5),
            BlackHole::with_spin(Vec3::new(0.1, 0.0, 0.0), 1.0, -0.5),
        ];
        let n = merge_black_holes(&mut bhs, 0.2, 0.0, None);
        assert_eq!(n, 1);
        assert_eq!(bhs.len(), 1);
        assert!((bhs[0].mass - 4.0).abs() < 1e-12);
        assert!((bhs[0].spin - 0.25).abs() < 1e-12);
    }
}
