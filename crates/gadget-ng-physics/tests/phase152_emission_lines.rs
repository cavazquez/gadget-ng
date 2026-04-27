//! Tests de integración — Phase 152: Líneas de emisión nebular.

use gadget_ng_analysis::emission_lines::{
    bpt_diagram, compute_emission_lines, emissivity_halpha, emissivity_nii, emissivity_oiii,
};
use gadget_ng_core::{Particle, Vec3};

const GAMMA: f64 = 5.0 / 3.0;

fn gas_particle(u: f64, h: f64, metallicity: f64) -> Particle {
    let mut p = Particle::new_gas(0, 0.1, Vec3::zero(), Vec3::zero(), u, h);
    p.metallicity = metallicity;
    p
}

// T1: Hα finita para gas caliente ionizado
#[test]
fn halpha_positive_hot_gas() {
    let rho = 0.01;
    let t_ion = 1.0e4;
    let ha = emissivity_halpha(rho, t_ion);
    assert!(
        ha > 0.0,
        "Hα debe ser positiva para gas ionizado, got {}",
        ha
    );
}

// T2: Líneas = 0 sin gas ionizado (T < T_min)
#[test]
fn lines_zero_cold_gas() {
    let rho = 0.01;
    let t_cold = 1000.0;
    let ha = emissivity_halpha(rho, t_cold);
    let oiii = emissivity_oiii(rho, t_cold, 0.02);
    let nii = emissivity_nii(rho, t_cold, 0.02);
    assert_eq!(ha, 0.0, "Hα debe ser 0 para gas frío");
    assert_eq!(oiii, 0.0, "[OIII] debe ser 0 para gas frío");
    assert_eq!(nii, 0.0, "[NII] debe ser 0 para gas frío");
}

// T3: Cociente NII/Hα crece con metalicidad
#[test]
fn nii_halpha_increases_with_metallicity() {
    let rho = 0.01;
    let t = 1.0e4;
    let ha = emissivity_halpha(rho, t);
    let nii_low = emissivity_nii(rho, t, 0.001);
    let nii_high = emissivity_nii(rho, t, 0.05);
    assert!(ha > 0.0, "Hα debe ser positiva");
    assert!(
        nii_high > nii_low,
        "[NII] debe crecer con Z: low={}, high={}",
        nii_low,
        nii_high
    );
}

// T4: [OIII] crece con metalicidad (proporcional a Z/Z_sun)
#[test]
fn oiii_grows_with_metallicity() {
    let rho = 0.01;
    let t = 2.0e4;
    let oiii_low = emissivity_oiii(rho, t, 0.001);
    let oiii_high = emissivity_oiii(rho, t, 0.05);
    assert!(
        oiii_high > oiii_low,
        "[OIII] debe crecer con Z: low={}, high={}",
        oiii_low,
        oiii_high
    );
}

// T5: diagrama BPT tiene puntos para gas ionizado
#[test]
fn bpt_diagram_has_points() {
    let mut particles = Vec::new();
    for _ in 0..5 {
        particles.push(gas_particle(1e4, 0.3, 0.02));
    }
    let lines = compute_emission_lines(&particles, GAMMA);
    let bpt = bpt_diagram(&lines);
    assert!(
        !bpt.is_empty(),
        "El diagrama BPT debe tener puntos para gas ionizado caliente"
    );
}

// T6: N=100 partículas sin panics
#[test]
fn emission_lines_n100_no_panic() {
    let mut particles = Vec::new();
    for i in 0..100 {
        let u = if i % 4 == 0 {
            0.001
        } else {
            5e3 + i as f64 * 100.0
        };
        let z = 0.001 + i as f64 * 0.0004;
        particles.push(gas_particle(u, 0.2, z));
    }
    let lines = compute_emission_lines(&particles, GAMMA);
    assert_eq!(lines.len(), 100);
    let _bpt = bpt_diagram(&lines);
}
