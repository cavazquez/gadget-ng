//! Tests de integración — Phase 156: Neutrinos masivos.

use gadget_ng_core::{CosmologyParams, neutrino_suppression, omega_nu_from_mass};

// T1: m_nu=0 → omega_nu = 0
#[test]
fn m_nu_zero_gives_omega_nu_zero() {
    let omega_nu = omega_nu_from_mass(0.0, 0.674);
    assert_eq!(omega_nu, 0.0, "omega_nu debe ser 0 para m_nu=0");
}

// T2: m_nu=0.06 eV suprime P(k) — neutrino_suppression < 1
#[test]
fn m_nu_006_suppresses_pk() {
    let h = 0.674;
    let omega_m = 0.315;
    let m_nu = 0.06;
    let omega_nu = omega_nu_from_mass(m_nu, h);
    let f_nu = omega_nu / omega_m;
    let sup = neutrino_suppression(f_nu);
    assert!(
        sup < 1.0,
        "La supresión debe ser < 1 para m_nu=0.06 eV, got {}",
        sup
    );
    assert!(sup > 0.0, "La supresión debe ser > 0");
}

// T3: formula Omega_nu correcta
#[test]
fn omega_nu_formula_correct() {
    // Omega_nu = m_nu / (93.14 eV * h^2)
    let h = 0.674;
    let m_nu = 0.3;
    let expected = m_nu / (93.14 * h * h);
    let omega_nu = omega_nu_from_mass(m_nu, h);
    assert!(
        (omega_nu - expected).abs() < 1e-10,
        "Formula Omega_nu incorrecta"
    );
}

// T4: advance_a estable con Omega_nu
#[test]
fn advance_a_stable_with_omega_nu() {
    let h = 0.674;
    let m_nu = 0.06;
    let omega_nu = omega_nu_from_mass(m_nu, h);
    let cosmo = CosmologyParams::new_with_nu(0.315, 0.685, 0.1, omega_nu);
    let mut a = 0.02_f64;
    let dt = 1e-4;
    for _ in 0..100 {
        a = cosmo.advance_a(a, dt);
        assert!(a.is_finite() && a > 0.0, "a debe permanecer positivo");
    }
}

// T5: supresion lineal — 8*f_nu
#[test]
fn neutrino_suppression_linear() {
    let f_nu = 0.05;
    let expected = 1.0 - 8.0 * f_nu;
    let sup = neutrino_suppression(f_nu);
    assert!(
        (sup - expected).abs() < 1e-12,
        "Supresión debe ser 1-8*f_nu, expected={}, got={}",
        expected,
        sup
    );
}

// T6: f_nu grande → supresión clampada a 0
#[test]
fn large_f_nu_suppression_clamped() {
    let sup = neutrino_suppression(0.5);
    assert_eq!(sup, 0.0, "Para f_nu=0.5, supresión debe ser 0 (clampada)");
}
