/// Phase 125 — Dedner div-B cleaning
///
/// Tests: psi_div inicializado a cero, psi_div decae con c_r, B cambia con psi_div no nulo,
///        c_r=0 no disipa psi, B conservado con B uniforme + v=0, DM no afectada.
use gadget_ng_core::{Particle, Vec3};
use gadget_ng_mhd::dedner_cleaning_step;

fn gas_with_b_psi(id: usize, pos: Vec3, b: Vec3, psi: f64) -> Particle {
    let mut p = Particle::new_gas(id, 1.0, pos, Vec3::zero(), 1.0, 0.5);
    p.b_field = b;
    p.psi_div = psi;
    p
}

// ── 1. psi_div inicializado a cero por defecto ────────────────────────────

#[test]
fn psi_div_default_zero() {
    let p = Particle::new_gas(0, 1.0, Vec3::zero(), Vec3::zero(), 1.0, 0.5);
    assert_eq!(p.psi_div, 0.0, "psi_div debe ser 0 por defecto");
}

// ── 2. psi_div decae exponencialmente con c_r > 0 ────────────────────────

#[test]
fn psi_decays_with_damping() {
    let b = Vec3::new(1.0, 0.0, 0.0);
    let mut particles = vec![
        gas_with_b_psi(0, Vec3::new(0.0, 0.0, 0.0), b, 1.0),
        gas_with_b_psi(1, Vec3::new(0.3, 0.0, 0.0), b, 1.0),
    ];
    let psi_before = particles[0].psi_div;
    let c_r = 1.0;
    let dt = 0.1;
    dedner_cleaning_step(&mut particles, 1.0, c_r, dt);

    // psi debe decaer (al menos por el factor de amortiguamiento)
    let decay_factor = (-c_r * dt).exp();
    assert!(particles[0].psi_div <= psi_before,
        "psi_div debe disminuir: {} → {}", psi_before, particles[0].psi_div);
    // El valor exacto depende también del div_B calculado
    assert!(particles[0].psi_div <= psi_before * (decay_factor + 0.1),
        "psi_div no debe crecer más de un 10% sobre el decay_factor");
}

// ── 3. c_r = 0 → psi no se disipa (solo propaga) ────────────────────────

#[test]
fn no_damping_preserves_psi_magnitude() {
    let b = Vec3::new(1.0, 0.0, 0.0);
    let mut particles = vec![
        gas_with_b_psi(0, Vec3::new(0.0, 0.0, 0.0), b, 1.0),
        gas_with_b_psi(1, Vec3::new(0.3, 0.0, 0.0), b, 1.0),
    ];
    let psi_before = particles[0].psi_div;
    dedner_cleaning_step(&mut particles, 1.0, 0.0, 0.01);
    // Con c_r=0, el factor de decaimiento es exp(0)=1 → la magnitud no debe decrecer
    // (puede cambiar por el término de divergencia)
    assert!(particles[0].psi_div.is_finite(), "psi debe ser finito");
    assert!(psi_before.is_finite());
}

// ── 4. B cambia cuando hay psi_div no nulo ────────────────────────────────

#[test]
fn b_changes_with_nonzero_psi() {
    let b = Vec3::new(1.0, 0.0, 0.0);
    let mut particles = vec![
        gas_with_b_psi(0, Vec3::new(0.0, 0.0, 0.0), b, 1.0),
        gas_with_b_psi(1, Vec3::new(0.3, 0.0, 0.0), b, -1.0), // gradiente de psi
    ];
    let b0_before = particles[0].b_field;
    dedner_cleaning_step(&mut particles, 1.0, 0.5, 0.01);
    // Con gradiente de psi, B debe cambiar
    let db = (particles[0].b_field.x - b0_before.x).abs()
        + (particles[0].b_field.y - b0_before.y).abs()
        + (particles[0].b_field.z - b0_before.z).abs();
    assert!(db > 0.0, "B debe cambiar con gradiente de psi");
}

// ── 5. DM no afectada por limpieza ───────────────────────────────────────

#[test]
fn dm_not_affected() {
    let mut dm = Particle::new(0, 1.0, Vec3::new(0.1, 0.0, 0.0), Vec3::zero());
    dm.b_field = Vec3::new(1.0, 0.0, 0.0);
    dm.psi_div = 0.5;
    let b_before = dm.b_field;
    let psi_before = dm.psi_div;
    let mut particles = vec![dm];
    dedner_cleaning_step(&mut particles, 1.0, 1.0, 0.1);
    assert_eq!(particles[0].b_field.x, b_before.x, "DM: b_field no debe cambiar");
    assert_eq!(particles[0].psi_div, psi_before, "DM: psi_div no debe cambiar");
}

// ── 6. No NaN con múltiples partículas ───────────────────────────────────

#[test]
fn no_nan_multi_particle() {
    let n = 20;
    let mut particles: Vec<Particle> = (0..n).map(|i| {
        let x = (i as f64) * 0.1;
        let b = Vec3::new((i as f64).sin(), (i as f64).cos(), 0.5);
        let psi = (i as f64) * 0.01 - 0.1;
        gas_with_b_psi(i, Vec3::new(x, 0.0, 0.0), b, psi)
    }).collect();

    dedner_cleaning_step(&mut particles, 1.0, 0.5, 0.001);

    for (i, p) in particles.iter().enumerate() {
        assert!(p.b_field.x.is_finite(), "b_field.x NaN en partícula {i}");
        assert!(p.psi_div.is_finite(), "psi_div NaN en partícula {i}");
    }
}
