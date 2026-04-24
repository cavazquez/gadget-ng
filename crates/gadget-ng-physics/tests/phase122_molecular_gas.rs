/// Phase 122 — Gas molecular HI → H₂
///
/// Tests: desactivado no-op, gas denso forma H2, gas diluido fotodisociación,
///        h2_fraction bounded [0,1], DM no afectado, SFR boost con H2.
use gadget_ng_core::{FeedbackSection, MolecularSection, Particle, ParticleType, Vec3};
use gadget_ng_sph::{compute_sfr_with_h2, update_h2_fraction};

fn gas_dense(id: usize) -> Particle {
    // h=0.01 → rho ≈ mass/h³ = 1.0/1e-6 = 1e6 >> rho_h2_threshold=100
    Particle::new_gas(id, 1.0, Vec3::zero(), Vec3::zero(), 1.0, 0.01)
}

fn gas_diffuse(id: usize) -> Particle {
    // h=1.0 → rho ≈ 1.0 << rho_h2_threshold=100
    Particle::new_gas(id, 1.0, Vec3::zero(), Vec3::zero(), 1.0, 1.0)
}

const CFG: MolecularSection = MolecularSection {
    enabled: true,
    rho_h2_threshold: 100.0,
    sfr_h2_boost: 2.0,
};

// ── 1. Desactivado = no-op ────────────────────────────────────────────────

#[test]
fn disabled_no_op() {
    let cfg_off = MolecularSection { enabled: false, ..Default::default() };
    let mut p = gas_dense(0);
    update_h2_fraction(&mut [p.clone()], &cfg_off, 0.1);
    assert_eq!(p.h2_fraction, 0.0);
}

// ── 2. Gas muy denso aumenta h2_fraction hacia el equilibrio ─────────────

#[test]
fn dense_gas_gains_h2() {
    let mut particles = vec![gas_dense(0)];
    particles[0].h2_fraction = 0.0;
    update_h2_fraction(&mut particles, &CFG, 0.1);
    assert!(particles[0].h2_fraction > 0.0, "Gas denso debe ganar H2");
}

// ── 3. Gas diluido pierde h2_fraction por fotodisociación ─────────────────

#[test]
fn diffuse_gas_loses_h2() {
    let mut particles = vec![gas_diffuse(0)];
    particles[0].h2_fraction = 0.8;
    update_h2_fraction(&mut particles, &CFG, 0.1);
    assert!(particles[0].h2_fraction < 0.8, "Gas diluido debe perder H2 por fotodisociación");
}

// ── 4. h2_fraction siempre en [0, 1] ─────────────────────────────────────

#[test]
fn h2_fraction_bounded() {
    let mut particles = vec![gas_dense(0), gas_diffuse(1)];
    particles[0].h2_fraction = 0.99;
    particles[1].h2_fraction = 0.01;
    update_h2_fraction(&mut particles, &CFG, 1.0);
    for p in &particles {
        assert!(p.h2_fraction >= 0.0 && p.h2_fraction <= 1.0,
            "h2_fraction fuera de [0,1]: {}", p.h2_fraction);
    }
}

// ── 5. DM no afectado ─────────────────────────────────────────────────────

#[test]
fn dm_not_affected() {
    let mut dm = Particle::new(0, 1.0, Vec3::zero(), Vec3::zero());
    dm.h2_fraction = 0.5;
    let mut particles = vec![dm];
    update_h2_fraction(&mut particles, &CFG, 0.1);
    assert_eq!(particles[0].h2_fraction, 0.5, "DM no debe cambiar h2_fraction");
}

// ── 6. SFR boost mayor con más H2 ────────────────────────────────────────

#[test]
fn sfr_boosted_by_h2() {
    let fb_cfg = FeedbackSection {
        enabled: true,
        sfr_min: 0.0,
        rho_sf: 0.001, // baja umbral para que trigger con gas_dense
        ..Default::default()
    };
    let p_no_h2 = gas_dense(0); // h2_fraction = 0.0
    let mut p_with_h2 = gas_dense(1);
    p_with_h2.h2_fraction = 1.0; // máximo H2

    let sfr_no_h2 = compute_sfr_with_h2(&[p_no_h2], &fb_cfg, 2.0);
    let sfr_with_h2 = compute_sfr_with_h2(&[p_with_h2], &fb_cfg, 2.0);

    assert!(sfr_with_h2[0] > sfr_no_h2[0],
        "SFR con H2 debe ser mayor: {} vs {}", sfr_with_h2[0], sfr_no_h2[0]);
    // Con h2_fraction=1 y boost=2: sfr_h2 = sfr_base * (1 + 2*1) = 3 * sfr_base
    let ratio = sfr_with_h2[0] / sfr_no_h2[0];
    assert!((ratio - 3.0).abs() < 1e-10, "Ratio esperado 3.0, obtenido {}", ratio);
}
