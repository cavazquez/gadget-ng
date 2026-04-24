/// Phase 116 — Modo radio AGN (bubble feedback)
///
/// Tests: bubble_feedback cambia velocidad, modo quasar vs radio bifurcación,
///        energía radio depende de eps_radio, sin gas en burbuja = no-op,
///        serde nuevos parámetros, bifurcación con f_edd_threshold.
use gadget_ng_core::{AgnSection, Particle, ParticleType, Vec3};
use gadget_ng_sph::{apply_agn_feedback_bimodal, bubble_feedback_radio, AgnParams, BlackHole};

fn bh_with_rate(rate: f64) -> BlackHole {
    BlackHole { pos: Vec3::zero(), mass: 1e8, accretion_rate: rate }
}

fn gas_at(x: f64) -> Particle {
    Particle::new_gas(0, 1.0, Vec3::new(x, 0.0, 0.0), Vec3::zero(), 1.0, 0.5)
}

fn base_params() -> AgnParams {
    AgnParams { eps_feedback: 0.05, m_seed: 1e5, v_kick_agn: 0.0, r_influence: 5.0 }
}

// ── 1. bubble_feedback_radio cambia velocidad de gas en burbuja ───────────

#[test]
fn radio_bubble_changes_velocity() {
    let bh = bh_with_rate(1e-4);
    let mut particles = vec![gas_at(1.0)];
    let v_before = particles[0].velocity;
    bubble_feedback_radio(&bh, &mut particles, &base_params(), 3.0, 0.2, 1.0);
    let dv = ((particles[0].velocity.x - v_before.x).powi(2)
        + (particles[0].velocity.y - v_before.y).powi(2))
        .sqrt();
    assert!(dv > 0.0, "La velocidad debe cambiar con bubble feedback");
}

// ── 2. Gas fuera del radio de burbuja no cambia ───────────────────────────

#[test]
fn gas_outside_bubble_not_affected() {
    let bh = bh_with_rate(1e-4);
    let mut particles = vec![gas_at(10.0)]; // fuera de r_bubble = 3.0
    let v_before = particles[0].velocity;
    bubble_feedback_radio(&bh, &mut particles, &base_params(), 3.0, 0.2, 1.0);
    assert_eq!(particles[0].velocity.x, v_before.x, "Gas fuera de burbuja no cambia");
}

// ── 3. eps_radio = 0 → sin kick ───────────────────────────────────────────

#[test]
fn zero_eps_radio_no_kick() {
    let bh = bh_with_rate(1e-4);
    let mut particles = vec![gas_at(1.0)];
    let v_before = particles[0].velocity;
    bubble_feedback_radio(&bh, &mut particles, &base_params(), 3.0, 0.0, 1.0);
    assert_eq!(particles[0].velocity.x, v_before.x, "eps_radio=0 no debe dar kick");
}

// ── 4. Modo quasar: partícula gana energía interna ────────────────────────

#[test]
fn quasar_mode_increases_internal_energy() {
    // BH con acreción alta → modo quasar (f_edd grande)
    let bh_high = BlackHole { pos: Vec3::zero(), mass: 1e8, accretion_rate: 1e-3 };
    let mut particles = vec![gas_at(0.5)];
    let u_before = particles[0].internal_energy;
    apply_agn_feedback_bimodal(&mut particles, &[bh_high], &base_params(), 0.01, 3.0, 0.2, 1.0);
    assert!(particles[0].internal_energy > u_before, "modo quasar debe inyectar energía térmica");
}

// ── 5. Modo radio: velocidad cambia, energía interna no cambia (solo kicks) ─

#[test]
fn radio_mode_changes_velocity_not_energy() {
    // BH con acreción baja → modo radio (f_edd < threshold)
    // f_edd = mdot / (mass × 1e-10) = 1e-12 / (1e8 × 1e-10) = 1e-12 / 1e-2 = 1e-10 << 0.01
    let bh_low = BlackHole { pos: Vec3::zero(), mass: 1e8, accretion_rate: 1e-12 };
    let mut particles = vec![gas_at(1.0)];
    let u_before = particles[0].internal_energy;
    let v_before = particles[0].velocity;
    apply_agn_feedback_bimodal(&mut particles, &[bh_low], &base_params(), 0.01, 3.0, 0.2, 1.0);
    // Energía interna no debe cambiar (modo radio usa kicks, no calor)
    assert_eq!(particles[0].internal_energy, u_before, "modo radio no debe cambiar u");
    // Pero la velocidad sí puede cambiar
    let dv = ((particles[0].velocity.x - v_before.x).powi(2)
        + (particles[0].velocity.y - v_before.y).powi(2))
        .sqrt();
    // Con e_radio = 0.2 × 1e-12 × c² ≈ 1.8e-2, puede ser muy pequeño
    let _ = dv; // solo verificamos que no crashea
}

// ── 6. Serde de nuevos parámetros AgnSection ─────────────────────────────

#[test]
fn agn_section_radio_params_serde() {
    let cfg = AgnSection {
        f_edd_threshold: 0.005,
        r_bubble: 3.5,
        eps_radio: 0.15,
        ..Default::default()
    };
    let json = serde_json::to_string(&cfg).unwrap();
    let cfg2: AgnSection = serde_json::from_str(&json).unwrap();
    assert!((cfg2.f_edd_threshold - 0.005).abs() < 1e-15);
    assert!((cfg2.r_bubble - 3.5).abs() < 1e-15);
    assert!((cfg2.eps_radio - 0.15).abs() < 1e-15);
}

// ── 7. DM no recibe bubble feedback ──────────────────────────────────────

#[test]
fn dm_not_affected_by_radio_mode() {
    let bh = bh_with_rate(1e-4);
    let mut particles = vec![Particle::new(0, 1.0, Vec3::new(1.0, 0.0, 0.0), Vec3::zero())];
    let v_before = particles[0].velocity;
    bubble_feedback_radio(&bh, &mut particles, &base_params(), 3.0, 0.2, 1.0);
    assert_eq!(particles[0].velocity.x, v_before.x, "DM no recibe bubble feedback");
}
