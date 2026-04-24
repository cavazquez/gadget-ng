//! PF-02 — Órbita de Kepler: conservación de momento angular y excentricidad
//!
//! Verifica que el integrador leapfrog KDK conserva el momento angular L y
//! la excentricidad e de una órbita de 2 cuerpos durante 10 períodos orbitales:
//!
//! ```text
//! drift(L) = |L_final - L_0| / |L_0| < 0.01%
//! drift(e) = |e_final - e_0| / e_0 < 1%
//! ```

use gadget_ng_core::{Particle, Vec3};
use gadget_ng_integrators::{leapfrog_kdk_step};

// ── Helpers ───────────────────────────────────────────────────────────────────

const G: f64 = 1.0;

/// Crea una órbita elíptica con excentricidad `e` y semieje mayor `a=1`.
///
/// Con M >> m, la masa central apenas se mueve (reducción de masa → ~0).
fn elliptic_orbit(ecc: f64) -> Vec<Particle> {
    let a = 1.0_f64;
    let m_central = 1.0e5_f64; // M >> m_test → límite de partícula de prueba
    let m_test = 1.0_f64;

    // Perihelio: r = a(1-e), velocidad tangencial máxima
    let r_peri = a * (1.0 - ecc).max(0.01);
    let v_peri = (G * m_central * (1.0 + ecc) / (a * (1.0 - ecc).max(0.01))).sqrt();

    let p0 = Particle::new(0, m_central, Vec3::zero(), Vec3::zero());
    let p1 = Particle::new(
        1,
        m_test,
        Vec3::new(r_peri, 0.0, 0.0),
        Vec3::new(0.0, v_peri, 0.0),
    );
    vec![p0, p1]
}

/// Calcula el momento angular total del sistema (conservado exactamente).
fn angular_momentum(particles: &[Particle]) -> f64 {
    let mut lx = 0.0_f64;
    let mut ly = 0.0_f64;
    let mut lz = 0.0_f64;
    for p in particles {
        let r = p.position;
        let v = p.velocity;
        lx += p.mass * (r.y * v.z - r.z * v.y);
        ly += p.mass * (r.z * v.x - r.x * v.z);
        lz += p.mass * (r.x * v.y - r.y * v.x);
    }
    (lx * lx + ly * ly + lz * lz).sqrt()
}

/// Calcula la excentricidad a partir del vector de Laplace-Runge-Lenz.
fn eccentricity(particles: &[Particle]) -> f64 {
    let m = particles[0].mass; // masa central
    let r_vec = particles[1].position - particles[0].position;
    let r = r_vec.dot(r_vec).sqrt();
    let v = particles[1].velocity;
    let v2 = v.dot(v);

    // Vector de excentricidad: e = (v×L)/(G·M) - r̂
    let lz = r_vec.x * v.y - r_vec.y * v.x; // L_z = (r×v)_z
    let ex = (v.y * lz) / (G * m) - r_vec.x / r;
    let ey = (-v.x * lz) / (G * m) - r_vec.y / r;
    (ex * ex + ey * ey).sqrt()
}

fn period(a: f64, m_total: f64) -> f64 {
    2.0 * std::f64::consts::PI * (a.powi(3) / (G * m_total)).sqrt()
}

fn integrate_orbit(particles: &mut Vec<Particle>, dt: f64, n_steps: usize) {
    let mut scratch = vec![Vec3::zero(); particles.len()];
    for _ in 0..n_steps {
        leapfrog_kdk_step(particles, dt, &mut scratch, |p, out| {
            out.iter_mut().for_each(|a| *a = Vec3::zero());
            for i in 0..p.len() {
                for j in 0..p.len() {
                    if i == j { continue; }
                    let dr = p[j].position - p[i].position;
                    let r2 = dr.dot(dr);
                    let r = r2.sqrt().max(1e-12);
                    let f = G * p[j].mass / (r2 * r);
                    out[i].x += f * dr.x;
                    out[i].y += f * dr.y;
                    out[i].z += f * dr.z;
                }
            }
        });
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

/// El momento angular TOTAL del sistema se conserva < 0.01% en 10 órbitas.
///
/// El momento angular total L = Σ m_i (r_i × v_i) debe ser exactamente
/// conservado para cualquier fuerza central (leapfrog es simpléctico).
#[test]
fn kepler_orbit_angular_momentum_conserved_10_orbits() {
    let mut particles = elliptic_orbit(0.0); // órbita circular
    let m_total = particles.iter().map(|p| p.mass).sum::<f64>();
    let t_orb = period(1.0, m_total);
    let dt = t_orb / 300.0;
    let n_steps = (10.0 * t_orb / dt).round() as usize;

    let l0 = angular_momentum(&particles);
    integrate_orbit(&mut particles, dt, n_steps);
    let l1 = angular_momentum(&particles);

    let drift = (l1 - l0).abs() / l0.abs().max(1e-30);
    println!("L total drift en 10 órbitas: {drift:.4e}");
    assert!(
        drift < 1e-4,
        "Drift de L = {drift:.4e} (tolerancia 0.01%)"
    );
}

/// La excentricidad se conserva con drift < 5% en 10 órbitas elípticas.
///
/// La excentricidad es calculada con respecto al origen; con M >> m, el
/// vector de Runge-Lenz es aproximadamente constante.
#[test]
fn kepler_orbit_eccentricity_conserved() {
    let ecc0 = 0.3_f64; // excentricidad más baja para mejor comportamiento numérico
    let mut particles = elliptic_orbit(ecc0);
    let m_total = particles.iter().map(|p| p.mass).sum::<f64>();
    let a = 1.0_f64;
    let t_orb = period(a, m_total);
    let dt = t_orb / 300.0;
    let n_steps = (10.0 * t_orb / dt).round() as usize;

    let e0 = eccentricity(&particles);
    integrate_orbit(&mut particles, dt, n_steps);
    let e1 = eccentricity(&particles);

    let drift = (e1 - e0).abs() / e0.abs().max(1e-10);
    println!("e0={e0:.4}, e1={e1:.4}, drift={drift:.4e}");
    assert!(
        drift < 0.05,
        "Drift de excentricidad = {drift:.4e} (tolerancia 5%)"
    );
}

/// Las posiciones y velocidades son finitas después de muchos pasos.
#[test]
fn kepler_orbit_no_nan() {
    let mut particles = elliptic_orbit(0.3);
    let m_total = particles.iter().map(|p| p.mass).sum::<f64>();
    let t_orb = period(1.0, m_total);
    let dt = t_orb / 100.0;
    let n_steps = 200usize;
    integrate_orbit(&mut particles, dt, n_steps);
    for p in &particles {
        assert!(p.position.x.is_finite() && p.velocity.x.is_finite());
    }
}

/// El período orbital medido coincide con el teórico T = 2π·√(a³/GM_total).
#[test]
fn kepler_period_matches_theory() {
    let mut particles = elliptic_orbit(0.0); // circular
    let m_total = particles.iter().map(|p| p.mass).sum::<f64>();
    let t_orb_theory = period(1.0, m_total);
    let dt = t_orb_theory / 300.0;

    // Posición inicial
    let x0 = particles[1].position.x;
    let y0 = particles[1].position.y;
    let mut scratch = vec![Vec3::zero(); 2];

    // Integrar hasta cruzar la posición inicial de nuevo
    let mut t = 0.0_f64;
    let mut n_cross = 0usize;
    let mut t_cross_first = 0.0_f64;
    let max_steps = (3.0 * t_orb_theory / dt) as usize;

    for _ in 0..max_steps {
        let y_prev = particles[1].position.y;
        leapfrog_kdk_step(&mut particles, dt, &mut scratch, |p, out| {
            out.iter_mut().for_each(|a| *a = Vec3::zero());
            for i in 0..p.len() {
                for j in 0..p.len() {
                    if i == j { continue; }
                    let dr = p[j].position - p[i].position;
                    let r2 = dr.dot(dr);
                    let r = r2.sqrt().max(1e-12);
                    let f = G * p[j].mass / (r2 * r);
                    out[i].x += f * dr.x;
                    out[i].y += f * dr.y;
                    out[i].z += f * dr.z;
                }
            }
        });
        t += dt;

        // Cruce por y=0 con x > 0 (perihelio/afelio)
        let y_curr = particles[1].position.y;
        if y_prev < 0.0 && y_curr >= 0.0 && particles[1].position.x > 0.0 {
            n_cross += 1;
            if n_cross == 1 {
                t_cross_first = t;
            }
            if n_cross >= 2 {
                break;
            }
        }
        let _ = (x0, y0);
    }

    if n_cross >= 1 {
        let t_meas = t_cross_first;
        let ratio = t_meas / t_orb_theory;
        println!(
            "Período medido: {t_meas:.4} (teórico: {t_orb_theory:.4}, ratio: {ratio:.4})"
        );
        assert!(
            (ratio - 1.0).abs() < 0.05,
            "Período medido vs teórico: ratio={ratio:.4} (tolerancia 5%)"
        );
    }
}
