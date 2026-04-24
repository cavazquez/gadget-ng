/// Phase 120 — Integración engine: ISM, CRs, vientos estelares, AGN bimodal
///
/// Tests: ISM + CR activos simultáneamente, stellar winds + CR conjuntos,
///        AGN bimodal con f_edd bajo usa modo radio, módulos desactivados no-op,
///        advance_stellar_ages + snia integración, stack bariónico completo no crashea.
use gadget_ng_core::{AgnSection, CrSection, FeedbackSection, IsmSection, Particle, Vec3};
use gadget_ng_sph::{
    advance_stellar_ages, apply_agn_feedback_bimodal, apply_stellar_wind_feedback,
    diffuse_cr, inject_cr_from_sn, update_ism_phases, AgnParams, BlackHole,
};

fn gas(id: usize, h: f64) -> Particle {
    Particle::new_gas(id, 1.0, Vec3::zero(), Vec3::zero(), 2.0, h)
}

fn star_particle(id: usize, age: f64) -> Particle {
    let mut p = Particle::new_star(id, 1.0, Vec3::zero(), Vec3::zero(), 0.02);
    p.stellar_age = age;
    p
}

// ── 1. ISM y CRs activos simultáneamente no se interfieren ────────────────

#[test]
fn ism_and_cr_independent() {
    let ism_cfg = IsmSection { enabled: true, q_star: 2.5, f_cold: 0.5 };
    let mut particles = vec![gas(0, 0.01)];
    let sfr = vec![1.0];
    let u_before = particles[0].internal_energy;

    update_ism_phases(&mut particles, &sfr, 0.001, &ism_cfg, 0.1);
    let e_total_after_ism = particles[0].internal_energy + particles[0].u_cold;
    assert!((e_total_after_ism - u_before).abs() < 1e-9, "ISM debe conservar energía total");

    // CR injection no afecta energía térmica
    let u_thermal = particles[0].internal_energy;
    inject_cr_from_sn(&mut particles, &sfr, 0.1, 0.1);
    assert_eq!(particles[0].internal_energy, u_thermal, "CR injection no cambia u térmica");
    assert!(particles[0].cr_energy > 0.0, "CR energy debe aumentar");
}

// ── 2. Vientos estelares + CRs conjuntos ─────────────────────────────────

#[test]
fn stellar_winds_and_cr_together() {
    let fb_cfg = FeedbackSection {
        enabled: true,
        sfr_min: 0.0,
        rho_sf: 0.0,
        stellar_wind_enabled: true,
        v_stellar_wind_km_s: 500.0,
        eta_stellar_wind: 10.0,
        ..Default::default()
    };
    let mut particles = vec![gas(0, 0.5)];
    let sfr = vec![1e6_f64];

    // Aplicar CR + vientos en secuencia (como en engine.rs)
    inject_cr_from_sn(&mut particles, &sfr, 0.1, 0.01);
    assert!(particles[0].cr_energy > 0.0);

    // Vientos son estocásticos — solo verificar que no crashea
    let mut seed = 42u64;
    apply_stellar_wind_feedback(&mut particles, &sfr, &fb_cfg, 0.01, &mut seed);
    // No assertion on velocity — just ensure no panic
}

// ── 3. AGN bimodal con f_edd bajo → modo radio (sin cambio en u) ──────────

#[test]
fn agn_bimodal_low_edd_radio_mode() {
    // f_edd muy bajo → modo radio: v cambia, u no cambia
    let bh = BlackHole { pos: Vec3::new(1.0, 0.0, 0.0), mass: 1e8, accretion_rate: 1e-12 };
    let mut particles = vec![gas(0, 0.5)];
    particles[0].position = Vec3::new(1.5, 0.0, 0.0);
    let params = AgnParams { eps_feedback: 0.05, m_seed: 1e5, v_kick_agn: 0.0, r_influence: 5.0 };
    let u_before = particles[0].internal_energy;

    apply_agn_feedback_bimodal(&mut particles, &[bh], &params, 0.01, 3.0, 0.2, 1.0);

    assert_eq!(particles[0].internal_energy, u_before, "Modo radio no modifica u");
}

// ── 4. Módulos desactivados = no-op ───────────────────────────────────────

#[test]
fn disabled_modules_no_effect() {
    let ism_off = IsmSection { enabled: false, ..Default::default() };
    let mut particles = vec![gas(0, 0.01)];
    let sfr = vec![1.0];
    let u_before = particles[0].internal_energy;

    update_ism_phases(&mut particles, &sfr, 0.001, &ism_off, 0.1);
    assert_eq!(particles[0].internal_energy, u_before, "ISM desactivado = no-op");
    assert_eq!(particles[0].u_cold, 0.0, "ISM desactivado = no-op en u_cold");

    // CRs con cr_fraction = 0 → sin inyección
    inject_cr_from_sn(&mut particles, &sfr, 0.0, 0.1);
    assert_eq!(particles[0].cr_energy, 0.0, "cr_fraction=0 → sin CRs");
}

// ── 5. advance_stellar_ages incrementa edad correctamente ─────────────────

#[test]
fn advance_ages_increments_stellar_age() {
    let mut particles = vec![
        star_particle(0, 0.5),
        gas(1, 0.5), // gas no debe cambiar
    ];
    let age_before = particles[0].stellar_age;
    let gas_age_before = particles[1].stellar_age;

    advance_stellar_ages(&mut particles, 0.01);

    assert!((particles[0].stellar_age - (age_before + 0.01)).abs() < 1e-15);
    assert_eq!(particles[1].stellar_age, gas_age_before, "Gas no cambia edad estelar");
}

// ── 6. Stack bariónico completo no crashea con 50 partículas ─────────────

#[test]
fn full_baryonic_stack_no_panic() {
    let ism_cfg = IsmSection { enabled: true, q_star: 2.5, f_cold: 0.5 };
    let fb_cfg = FeedbackSection {
        enabled: true,
        sfr_min: 0.0,
        rho_sf: 0.01,
        stellar_wind_enabled: true,
        v_stellar_wind_km_s: 500.0,
        eta_stellar_wind: 0.1,
        ..Default::default()
    };

    let n = 50;
    let mut particles: Vec<Particle> = (0..n)
        .map(|i| gas(i, 0.1 + (i as f64) * 0.01))
        .collect();
    let sfr = vec![0.1_f64; n];

    // Secuencia igual a engine.rs
    update_ism_phases(&mut particles, &sfr, 0.01, &ism_cfg, 0.01);
    inject_cr_from_sn(&mut particles, &sfr, 0.1, 0.01);
    diffuse_cr(&mut particles, 3e-3, 0.0, 0.01);
    let mut seed = 77u64;
    apply_stellar_wind_feedback(&mut particles, &sfr, &fb_cfg, 0.01, &mut seed);
    advance_stellar_ages(&mut particles, 0.001);

    // Solo verifica que no haya NaN
    for p in &particles {
        assert!(p.internal_energy.is_finite(), "u no debe ser NaN");
        assert!(p.cr_energy.is_finite(), "cr_energy no debe ser NaN");
        assert!(p.u_cold.is_finite(), "u_cold no debe ser NaN");
    }
}
