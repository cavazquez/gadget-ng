/// Phase 115 — Vientos estelares pre-SN (feedback mecánico OB/Wolf-Rayet)
///
/// Tests: kick ocurre con prob alta, DM no recibe kick, desactivado = no-op,
///        velocidad del kick correcta, sfr=0 sin kick, serde de nuevos params.
use gadget_ng_core::{FeedbackSection, Particle, Vec3};
use gadget_ng_sph::apply_stellar_wind_feedback;

fn cfg_wind(enabled: bool) -> FeedbackSection {
    FeedbackSection {
        enabled: true,
        sfr_min: 0.0,
        rho_sf: 0.0,
        stellar_wind_enabled: enabled,
        v_stellar_wind_km_s: 500.0,
        eta_stellar_wind: 10.0, // factor alto para forzar prob ≈ 1
        ..Default::default()
    }
}

fn gas(id: usize) -> Particle {
    Particle::new_gas(id, 1.0, Vec3::zero(), Vec3::zero(), 1.0, 0.5)
}

// ── 1. Kick ocurre con prob alta (eta grande) ─────────────────────────────

#[test]
fn stellar_wind_kick_occurs_eventually() {
    let cfg = cfg_wind(true);
    let mut kicked_any = false;
    for attempt in 0..200u64 {
        let mut particles = vec![gas(0)];
        let sfr = vec![1.0];
        let mut seed = attempt.wrapping_mul(6271) + 1;
        let k = apply_stellar_wind_feedback(&mut particles, &sfr, &cfg, 1.0, &mut seed);
        if !k.is_empty() {
            kicked_any = true;
            break;
        }
    }
    assert!(
        kicked_any,
        "Debe haber al menos un kick en 200 intentos con eta=10"
    );
}

// ── 2. DM no recibe kick ───────────────────────────────────────────────────

#[test]
fn dm_not_kicked_by_stellar_wind() {
    let cfg = cfg_wind(true);
    let mut particles = vec![Particle::new(0, 1.0, Vec3::zero(), Vec3::zero())];
    let sfr = vec![1e10_f64];
    let mut seed = 42u64;
    let k = apply_stellar_wind_feedback(&mut particles, &sfr, &cfg, 1.0, &mut seed);
    assert!(k.is_empty(), "DM no debe recibir kick de viento estelar");
}

// ── 3. Desactivado = no-op ─────────────────────────────────────────────────

#[test]
fn disabled_stellar_wind_no_kick() {
    let cfg = cfg_wind(false);
    let mut particles = vec![gas(0)];
    let sfr = vec![1e10_f64];
    let vel_before = particles[0].velocity;
    let mut seed = 99u64;
    apply_stellar_wind_feedback(&mut particles, &sfr, &cfg, 1.0, &mut seed);
    assert_eq!(particles[0].velocity.x, vel_before.x);
    assert_eq!(particles[0].velocity.y, vel_before.y);
    assert_eq!(particles[0].velocity.z, vel_before.z);
}

// ── 4. SFR = 0 → sin kick ─────────────────────────────────────────────────

#[test]
fn zero_sfr_no_stellar_wind() {
    let cfg = cfg_wind(true);
    let mut particles = vec![gas(0)];
    let sfr = vec![0.0];
    let mut seed = 7u64;
    let k = apply_stellar_wind_feedback(&mut particles, &sfr, &cfg, 1.0, &mut seed);
    assert!(k.is_empty(), "SFR=0 no debe producir kick de viento");
}

// ── 5. El kick cambia la velocidad ────────────────────────────────────────

#[test]
fn stellar_wind_changes_velocity() {
    let cfg = cfg_wind(true);
    let mut changed = false;
    for attempt in 0..500u64 {
        let mut particles = vec![gas(0)];
        let sfr = vec![1e6_f64]; // sfr altísimo
        let v0 = particles[0].velocity;
        let mut seed = attempt.wrapping_add(1234);
        apply_stellar_wind_feedback(&mut particles, &sfr, &cfg, 1.0, &mut seed);
        let dv = ((particles[0].velocity.x - v0.x).powi(2)
            + (particles[0].velocity.y - v0.y).powi(2)
            + (particles[0].velocity.z - v0.z).powi(2))
        .sqrt();
        if dv > 0.0 {
            changed = true;
            break;
        }
    }
    assert!(
        changed,
        "La velocidad debe cambiar tras un kick de viento estelar"
    );
}

// ── 6. Serde de nuevos parámetros ─────────────────────────────────────────

#[test]
fn stellar_wind_params_serde() {
    let cfg = FeedbackSection {
        stellar_wind_enabled: true,
        v_stellar_wind_km_s: 3000.0,
        eta_stellar_wind: 0.2,
        ..Default::default()
    };
    let json = serde_json::to_string(&cfg).unwrap();
    let cfg2: FeedbackSection = serde_json::from_str(&json).unwrap();
    assert!(cfg2.stellar_wind_enabled);
    assert!((cfg2.v_stellar_wind_km_s - 3000.0).abs() < 1e-10);
    assert!((cfg2.eta_stellar_wind - 0.2).abs() < 1e-15);
}
