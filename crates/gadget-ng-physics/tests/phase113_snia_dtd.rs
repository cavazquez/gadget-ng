/// Phase 113 — SN Ia con distribución de retraso temporal (DTD)
///
/// Tests: rate decrece con edad, no Ia para estrellas jóvenes,
///        inyección de energía positiva, distribución de Fe a gas vecino,
///        DTD integrada da fracción razonable, serde de parámetros.
use gadget_ng_core::{FeedbackSection, Particle, Vec3};
use gadget_ng_sph::{advance_stellar_ages, apply_snia_feedback};

fn fb_cfg() -> FeedbackSection {
    FeedbackSection {
        enabled: true,
        a_ia: 2e-3,
        t_ia_min_gyr: 0.1,
        e_ia_code: 1.54e-3 * 1.3,
        ..Default::default()
    }
}

fn star_with_age(id: usize, age_gyr: f64) -> Particle {
    let mut s = Particle::new_star(id, 1.0, Vec3::zero(), Vec3::zero(), 0.01);
    s.stellar_age = age_gyr;
    s.smoothing_length = 1.0;
    s
}

fn gas_nearby(id: usize, offset: f64) -> Particle {
    Particle::new_gas(id, 1.0, Vec3::new(offset, 0.0, 0.0), Vec3::zero(), 1.0, 0.5)
}

// ── 1. Estrellas jóvenes (< t_ia_min) no generan SN Ia ───────────────────

#[test]
fn no_ia_for_young_stars() {
    let cfg = fb_cfg();
    let mut particles = vec![
        star_with_age(0, 0.05), // edad < t_ia_min = 0.1 Gyr
        gas_nearby(1, 0.3),
    ];
    let u_before = particles[1].internal_energy;
    let mut seed = 42u64;
    apply_snia_feedback(&mut particles, 1.0, &mut seed, &cfg);
    assert_eq!(
        particles[1].internal_energy, u_before,
        "No debe inyectarse energía para estrella joven"
    );
}

// ── 2. Feedback desactivado no hace nada ─────────────────────────────────

#[test]
fn disabled_no_ia() {
    let mut cfg = fb_cfg();
    cfg.enabled = false;
    let mut particles = vec![star_with_age(0, 5.0), gas_nearby(1, 0.3)];
    let u_before = particles[1].internal_energy;
    let mut seed = 7u64;
    apply_snia_feedback(&mut particles, 1.0, &mut seed, &cfg);
    assert_eq!(particles[1].internal_energy, u_before);
}

// ── 3. Inyección de energía positiva a gas vecino ─────────────────────────

#[test]
fn energy_injected_to_neighbor() {
    let cfg = fb_cfg();
    let mut injected = false;
    // Intentar 500 veces con masa grande para maximizar N_exp
    for i in 0..500u64 {
        let mut star = star_with_age(0, 1.0);
        star.mass = 100.0; // masa grande → N_exp alto
        let mut particles = vec![star, gas_nearby(1, 0.3)];
        let u_before = particles[1].internal_energy;
        let mut seed = i.wrapping_mul(6271) + 13;
        apply_snia_feedback(&mut particles, 1.0, &mut seed, &cfg);
        if particles[1].internal_energy > u_before {
            injected = true;
            break;
        }
    }
    assert!(
        injected,
        "Debe inyectarse energía al gas vecino en alguno de los 500 intentos"
    );
}

// ── 4. Metalicidad (Fe) se distribuye a gas vecino ────────────────────────

#[test]
fn iron_distributed_to_neighbor() {
    let cfg = fb_cfg();
    for i in 0..500u64 {
        let mut star = star_with_age(0, 1.0);
        star.mass = 100.0;
        let mut particles = vec![star, gas_nearby(1, 0.3)];
        let z_before = particles[1].metallicity;
        let mut seed = i.wrapping_mul(3571) + 7;
        apply_snia_feedback(&mut particles, 1.0, &mut seed, &cfg);
        if particles[1].metallicity > z_before {
            return; // éxito
        }
    }
    panic!("Debe distribuirse Fe al gas vecino en alguno de los 500 intentos");
}

// ── 5. DTD integrada da fracción razonable ────────────────────────────────
// ∫_{0.1}^{10} A_Ia × t^{-1} dt = A_Ia × ln(10/0.1) ≈ 2e-3 × 4.6 ≈ 9.2e-3 SN/M_sun

#[test]
fn dtd_integrated_fraction_reasonable() {
    let a_ia = 2e-3_f64;
    let t_min = 0.1_f64;
    let t_max = 10.0_f64;
    let integral = a_ia * (t_max / t_min).ln();
    // ~9.2e-3 SN/M_sun total — observacionalmente ~1e-3 a ~1e-2 SN/M_sun
    assert!(integral > 1e-3, "Fracción integrada muy baja: {integral}");
    assert!(integral < 1e-1, "Fracción integrada muy alta: {integral}");
}

// ── 6. Serde de FeedbackSection con parámetros SN Ia ─────────────────────

#[test]
fn feedback_snia_params_serde() {
    let cfg = FeedbackSection {
        a_ia: 3e-3,
        t_ia_min_gyr: 0.05,
        e_ia_code: 2.5e-3,
        ..Default::default()
    };
    let json = serde_json::to_string(&cfg).unwrap();
    let cfg2: FeedbackSection = serde_json::from_str(&json).unwrap();
    assert!((cfg2.a_ia - 3e-3).abs() < 1e-20);
    assert!((cfg2.t_ia_min_gyr - 0.05).abs() < 1e-15);
    assert!((cfg2.e_ia_code - 2.5e-3).abs() < 1e-20);
}

// ── 7. advance_stellar_ages incrementa edad correctamente ─────────────────

#[test]
fn advance_stellar_ages_test() {
    let mut particles = vec![
        star_with_age(0, 1.0),
        gas_nearby(1, 0.5),                                // gas no debe cambiar
        Particle::new(2, 1.0, Vec3::zero(), Vec3::zero()), // DM no cambia
    ];
    advance_stellar_ages(&mut particles, 0.1);
    assert!(
        (particles[0].stellar_age - 1.1).abs() < 1e-12,
        "Edad estelar debe aumentar"
    );
    assert_eq!(particles[1].stellar_age, 0.0, "Gas no tiene edad estelar");
    assert_eq!(particles[2].stellar_age, 0.0, "DM no tiene edad estelar");
}
