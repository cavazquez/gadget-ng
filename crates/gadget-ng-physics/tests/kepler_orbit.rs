//! Validación de la órbita de Kepler (problema de 2 cuerpos).
//!
//! ## Configuración
//!
//! - Sol: masa M = 1 en el origen, fijo (masa ≫ planeta).
//! - Planeta: masa m = 1e-6, órbita circular de radio r = 1.
//! - G = 1.
//!
//! ## Condiciones iniciales (órbita circular)
//!
//! ```text
//! r_0 = (1, 0, 0)
//! v_0 = (0, v_circ, 0)   con v_circ = √(GM/r) = 1
//! ```
//!
//! ## Cantidades conservadas
//!
//! - Energía específica: E = ½v² − GM/r = −½ (para órbita circular con M=1, r=1, G=1).
//! - Momento angular específico: L = |r × v| = r·v_circ = 1.
//! - Período orbital: T = 2πr/v_circ = 2π.
//!
//! ## Criterios de validación
//!
//! Después de exactamente un período, el planeta debe volver a su posición inicial.
//! Se verifica también la conservación de E y L a lo largo de la integración.

use gadget_ng_core::Vec3;
use gadget_ng_integrators::leapfrog_kdk_step;

const G: f64 = 1.0;
const M_SUN: f64 = 1.0;
const R_ORBIT: f64 = 1.0;

/// Calcula la aceleración gravitatoria del Sol sobre el planeta.
/// El Sol se asume fijo en el origen con masa M_SUN.
fn solar_accel(r_planet: Vec3) -> Vec3 {
    let r = r_planet.norm();
    let eps2 = 1e-14; // suavizado mínimo para estabilidad
    let r2_soft = r * r + eps2;
    let r3_soft = r2_soft * r2_soft.sqrt();
    r_planet * (-G * M_SUN / r3_soft)
}

/// Energía específica del planeta (referencia: potencial → 0 en r → ∞).
fn specific_energy(pos: Vec3, vel: Vec3) -> f64 {
    let r = pos.norm();
    0.5 * vel.dot(vel) - G * M_SUN / r
}

/// Momento angular específico (escalar = componente z en órbita XY).
fn specific_angular_momentum(pos: Vec3, vel: Vec3) -> f64 {
    pos.x * vel.y - pos.y * vel.x
}

/// Tipo auxiliar: partícula representada como (pos, vel) para el integrador.
struct TwoBodyState {
    pos: Vec3,
    vel: Vec3,
}

#[test]
fn kepler_circular_orbit_one_period() {
    // Condiciones iniciales: órbita circular perfecta.
    let v_circ = (G * M_SUN / R_ORBIT).sqrt(); // = 1.0
    let pos0 = Vec3::new(R_ORBIT, 0.0, 0.0);
    let vel0 = Vec3::new(0.0, v_circ, 0.0);

    let period = 2.0 * std::f64::consts::PI * R_ORBIT / v_circ;

    // Paso de tiempo: 1000 pasos por período → dt = T/1000.
    let n_steps = 1000usize;
    let dt = period / n_steps as f64;

    let state = TwoBodyState {
        pos: pos0,
        vel: vel0,
    };
    let e0 = specific_energy(state.pos, state.vel);
    let l0 = specific_angular_momentum(state.pos, state.vel);

    // Simular usando leapfrog_kdk_step con una partícula "gadget_ng_core::Particle".
    // Como el integrador trabaja con &mut [Particle], envolvemos el estado manualmente.
    use gadget_ng_core::Particle;

    let mut particles = vec![Particle::new(0, 1e-6, pos0, vel0)];
    let mut scratch = vec![Vec3::zero(); 1];

    for _ in 0..n_steps {
        leapfrog_kdk_step(&mut particles, dt, &mut scratch, |ps, acc| {
            acc[0] = solar_accel(ps[0].position);
        });
    }

    let pos_f = particles[0].position;
    let vel_f = particles[0].velocity;
    let e_f = specific_energy(pos_f, vel_f);
    let l_f = specific_angular_momentum(pos_f, vel_f);

    // El planeta debe volver a su posición inicial con <0.5% de error.
    let pos_err = (pos_f - pos0).norm() / R_ORBIT;
    assert!(
        pos_err < 0.005,
        "Posición final: error = {pos_err:.4e} (esperado < 0.005)"
    );

    // Energía conservada con <0.1% de error relativo.
    let e_err = (e_f - e0).abs() / e0.abs();
    assert!(
        e_err < 0.001,
        "Energía: e0={e0:.6}, e_f={e_f:.6}, error relativo={e_err:.4e}"
    );

    // Momento angular conservado con <0.1% de error.
    let l_err = (l_f - l0).abs() / l0.abs();
    assert!(
        l_err < 0.001,
        "Momento angular: L0={l0:.6}, L_f={l_f:.6}, error relativo={l_err:.4e}"
    );
}

#[test]
fn kepler_elliptic_orbit_energy_conserved() {
    // Órbita elíptica: v_0 = 0.8 * v_circ (sublanzada → excentricidad e ≈ 0.22).
    let v_init = 0.8 * (G * M_SUN / R_ORBIT).sqrt();
    let pos0 = Vec3::new(R_ORBIT, 0.0, 0.0);
    let vel0 = Vec3::new(0.0, v_init, 0.0);

    use gadget_ng_core::Particle;
    let mut particles = vec![Particle::new(0, 1e-6, pos0, vel0)];
    let mut scratch = vec![Vec3::zero(); 1];

    let e0 = specific_energy(pos0, vel0);
    let l0 = specific_angular_momentum(pos0, vel0);

    let period_approx = 2.0 * std::f64::consts::PI; // sobreestimación del período
    let n_steps = 2000usize;
    let dt = period_approx / n_steps as f64;

    for _ in 0..n_steps {
        leapfrog_kdk_step(&mut particles, dt, &mut scratch, |ps, acc| {
            acc[0] = solar_accel(ps[0].position);
        });
    }

    let e_f = specific_energy(particles[0].position, particles[0].velocity);
    let l_f = specific_angular_momentum(particles[0].position, particles[0].velocity);

    // Energía y momento angular conservados a <0.5 %.
    let e_err = (e_f - e0).abs() / e0.abs();
    let l_err = (l_f - l0).abs() / l0.abs();
    assert!(e_err < 0.005, "Energía elíptica: error = {e_err:.4e}");
    assert!(l_err < 0.005, "L angular elíptico: error = {l_err:.4e}");
}

#[test]
fn kepler_orbit_radius_nearly_constant_circular() {
    // En órbita circular perfecta el radio debe mantenerse constante.
    let v_circ = (G * M_SUN / R_ORBIT).sqrt();
    let pos0 = Vec3::new(R_ORBIT, 0.0, 0.0);
    let vel0 = Vec3::new(0.0, v_circ, 0.0);

    use gadget_ng_core::Particle;
    let mut particles = vec![Particle::new(0, 1e-6, pos0, vel0)];
    let mut scratch = vec![Vec3::zero(); 1];

    let period = 2.0 * std::f64::consts::PI;
    let n_steps = 1000usize;
    let dt = period / n_steps as f64;

    let mut r_min = f64::INFINITY;
    let mut r_max = 0.0_f64;

    for _ in 0..n_steps {
        leapfrog_kdk_step(&mut particles, dt, &mut scratch, |ps, acc| {
            acc[0] = solar_accel(ps[0].position);
        });
        let r = particles[0].position.norm();
        r_min = r_min.min(r);
        r_max = r_max.max(r);
    }

    // Variación del radio < 0.5 % (órbita es casi circular con leapfrog).
    let delta_r = (r_max - r_min) / R_ORBIT;
    assert!(
        delta_r < 0.005,
        "Variación del radio: Δr/r0 = {delta_r:.4e}"
    );
}
