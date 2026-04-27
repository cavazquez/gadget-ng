//! Tests de integración — Phase 151: Emisión de rayos X.

use gadget_ng_analysis::xray::{
    compute_xray_profile, mass_weighted_temperature, spectroscopic_temperature,
    total_xray_luminosity,
};
use gadget_ng_core::{Particle, Vec3};

const GAMMA: f64 = 5.0 / 3.0;

fn gas_particle(u: f64, h: f64, mass: f64, x: f64) -> Particle {
    Particle::new_gas(0, mass, Vec3 { x, y: 0.0, z: 0.0 }, Vec3::zero(), u, h)
}

// T1: L_X positiva para gas caliente
#[test]
fn lx_positive_hot_gas() {
    let u_hot = 1e4;
    let particles = vec![gas_particle(u_hot, 0.5, 1.0, 0.0)];
    let lx = total_xray_luminosity(&particles, GAMMA);
    assert!(
        lx > 0.0,
        "L_X debe ser positiva para gas caliente, got {}",
        lx
    );
}

// T2: T_X > 0 para gas caliente
#[test]
fn tx_positive_hot_gas() {
    let u_hot = 1e4;
    let particles = vec![gas_particle(u_hot, 0.5, 1.0, 0.0)];
    let tx = spectroscopic_temperature(&particles, GAMMA);
    assert!(
        tx > 0.0,
        "T_X debe ser positiva para gas caliente, got {}",
        tx
    );
}

// T3: L_X = 0 sin gas caliente
#[test]
fn lx_zero_no_hot_gas() {
    let u_cold = 0.01;
    let particles = vec![gas_particle(u_cold, 0.5, 1.0, 0.0)];
    let lx = total_xray_luminosity(&particles, GAMMA);
    assert_eq!(lx, 0.0, "L_X debe ser 0 para gas frío, got {}", lx);
}

// T4: perfil radial tiene luminosidad total positiva
#[test]
fn xray_profile_has_luminosity() {
    let u_hot = 1e4;
    let mut particles = Vec::new();
    for i in 0..10 {
        let r = i as f64 * 0.5;
        let mass = 1.0 / (1.0 + r * r);
        particles.push(gas_particle(u_hot, 0.3, mass, r));
    }
    let r_edges: Vec<f64> = (0..6).map(|i| i as f64).collect();
    let profile = compute_xray_profile(&particles, [0.0, 0.0, 0.0], &r_edges, GAMMA);
    assert!(!profile.is_empty(), "El perfil no debe estar vacío");
    let lx_total: f64 = profile.iter().map(|b| b.luminosity_x).sum();
    assert!(
        lx_total > 0.0,
        "Luminosidad total del perfil debe ser positiva"
    );
}

// T5: T_X espectroscópica difiere de T_media
#[test]
fn tx_spectroscopic_differs_from_mass_weighted() {
    let mut particles = Vec::new();
    for _ in 0..5 {
        particles.push(gas_particle(1e3, 0.3, 0.1, 0.0));
    }
    for _ in 0..5 {
        particles.push(gas_particle(1e5, 0.3, 0.1, 1.0));
    }
    let t_sl = spectroscopic_temperature(&particles, GAMMA);
    let t_mw = mass_weighted_temperature(&particles, GAMMA);
    assert!(t_sl > 0.0 && t_mw > 0.0, "T_sl={}, T_mw={}", t_sl, t_mw);
    assert!((t_sl - t_mw).abs() > 0.0, "T_sl y T_mw deben diferir");
}

// T6: integración con N=50 partículas sin panics
#[test]
fn xray_integration_n50() {
    let mut particles = Vec::new();
    for i in 0..50 {
        let u = if i % 3 == 0 { 1e4 } else { 0.01 };
        let r = i as f64 * 0.2;
        particles.push(gas_particle(u, 0.2, 0.1, r));
    }
    let lx = total_xray_luminosity(&particles, GAMMA);
    let tx = spectroscopic_temperature(&particles, GAMMA);
    let r_edges: Vec<f64> = (0..11).map(|i| i as f64).collect();
    let profile = compute_xray_profile(&particles, [0.0, 0.0, 0.0], &r_edges, GAMMA);
    assert!(lx >= 0.0 && tx >= 0.0 && profile.len() == 10);
}
