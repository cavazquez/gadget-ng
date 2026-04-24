//! Tests de integración — Phase 157: SIDM scattering elástico.

use gadget_ng_tree::{apply_sidm_scattering, scatter_probability, SidmParams};
use gadget_ng_core::{Particle, Vec3};

fn dm_particle(x: f64, vx: f64, h: f64, m: f64) -> Particle {
    let mut p = Particle::new(0, m, Vec3 { x, y: 0.0, z: 0.0 }, Vec3 { x: vx, y: 0.0, z: 0.0 });
    p.smoothing_length = h;
    p
}

// T1: momento total conservado tras scattering
#[test]
fn momentum_conserved_after_scatter() {
    let mut particles = vec![
        dm_particle(0.0, 100.0, 1.0, 1.0),
        dm_particle(0.5, -100.0, 1.0, 1.0),
    ];
    let px_before: f64 = particles.iter().map(|p| p.mass * p.velocity.x).sum();
    let py_before: f64 = particles.iter().map(|p| p.mass * p.velocity.y).sum();
    let pz_before: f64 = particles.iter().map(|p| p.mass * p.velocity.z).sum();

    let params = SidmParams { sigma_m: 1e-3, v_max: 1e6 };
    apply_sidm_scattering(&mut particles, &params, 0.1, 42);

    let px_after: f64 = particles.iter().map(|p| p.mass * p.velocity.x).sum();
    let py_after: f64 = particles.iter().map(|p| p.mass * p.velocity.y).sum();
    let pz_after: f64 = particles.iter().map(|p| p.mass * p.velocity.z).sum();

    assert!((px_after - px_before).abs() < 1e-6, "p_x no conservado: Δ={}", px_after - px_before);
    assert!((py_after - py_before).abs() < 1e-6, "p_y no conservado");
    assert!((pz_after - pz_before).abs() < 1e-6, "p_z no conservado");
}

// T2: energía cinética conservada (scattering elástico)
#[test]
fn kinetic_energy_conserved() {
    let mut particles = vec![
        dm_particle(0.0, 100.0, 1.0, 1.0),
        dm_particle(0.5, -50.0, 1.0, 1.0),
    ];
    let ek_before: f64 = particles.iter().map(|p| {
        let v2 = p.velocity.x*p.velocity.x + p.velocity.y*p.velocity.y + p.velocity.z*p.velocity.z;
        0.5 * p.mass * v2
    }).sum();

    let params = SidmParams { sigma_m: 1e-3, v_max: 1e6 };
    apply_sidm_scattering(&mut particles, &params, 0.1, 7);

    let ek_after: f64 = particles.iter().map(|p| {
        let v2 = p.velocity.x*p.velocity.x + p.velocity.y*p.velocity.y + p.velocity.z*p.velocity.z;
        0.5 * p.mass * v2
    }).sum();

    assert!((ek_after - ek_before).abs() / ek_before.max(1e-30) < 1e-6,
        "Energía cinética no conservada: E_antes={:.6e}, E_después={:.6e}", ek_before, ek_after);
}

// T3: P_scatter crece con densidad
#[test]
fn scatter_prob_grows_with_density() {
    let p_low = scatter_probability(100.0, 0.01, 1e-5, 0.1);
    let p_high = scatter_probability(100.0, 1.0, 1e-5, 0.1);
    assert!(p_high > p_low, "P_scatter debe crecer con densidad: {:.6e} vs {:.6e}", p_low, p_high);
}

// T4: sigma_m=0 es noop
#[test]
fn sigma_m_zero_is_noop() {
    let mut particles = vec![
        dm_particle(0.0, 100.0, 1.0, 1.0),
        dm_particle(0.3, -100.0, 1.0, 1.0),
    ];
    let v0_before = particles[0].velocity.x;
    let v1_before = particles[1].velocity.x;

    let params = SidmParams { sigma_m: 0.0, v_max: 1e6 };
    apply_sidm_scattering(&mut particles, &params, 0.1, 42);

    assert_eq!(particles[0].velocity.x, v0_before, "sigma_m=0 no debe cambiar velocidades");
    assert_eq!(particles[1].velocity.x, v1_before);
}

// T5: N=0 no crash
#[test]
fn sidm_n0_no_crash() {
    let mut particles: Vec<Particle> = Vec::new();
    let params = SidmParams::default();
    apply_sidm_scattering(&mut particles, &params, 0.1, 42);
}

// T6: N=20 partículas — momento conservado
#[test]
fn sidm_n20_moment_conserved() {
    let mut particles: Vec<Particle> = (0..20).map(|i| {
        dm_particle(i as f64 * 0.1, (i as f64 - 10.0) * 10.0, 0.3, 0.1)
    }).collect();

    let px_before: f64 = particles.iter().map(|p| p.mass * p.velocity.x).sum();
    let params = SidmParams { sigma_m: 1e-3, v_max: 1e6 };
    apply_sidm_scattering(&mut particles, &params, 0.05, 123);
    let px_after: f64 = particles.iter().map(|p| p.mass * p.velocity.x).sum();
    assert!((px_after - px_before).abs() < 1e-4, "Momento total debe conservarse para N=20");
}
