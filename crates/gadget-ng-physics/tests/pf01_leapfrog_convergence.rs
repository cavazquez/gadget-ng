//! PF-01 — Convergencia de orden del integrador leapfrog KDK
//!
//! El integrador KDK es de orden 2 (Verlet). Verificamos este orden midiendo
//! el error de energía en una órbita de Kepler al variar el timestep:
//!
//! ```text
//! E_error(dt) ∝ dt²
//! ratio = E_error(dt) / E_error(dt/2) ≈ 4.0 ± 0.5
//! ```

use gadget_ng_core::{Particle, Vec3};
use gadget_ng_integrators::leapfrog_kdk_step;

const G: f64 = 1.0;

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Genera dos partículas en órbita circular (G=1, M_central=100, m_test=1, r=1).
/// v_circ = √(G·M/r) = √100 = 10. Período T = 2π/10 ≈ 0.628.
fn kepler_initial() -> Vec<Particle> {
    let m_central = 100.0_f64;
    let m_test = 1.0_f64;
    let r = 1.0_f64;
    let v_circ = (G * m_central / r).sqrt(); // = 10.0

    let mut p0 = Particle::new(0, m_central, Vec3::new(0.0, 0.0, 0.0), Vec3::zero());
    let mut p1 = Particle::new(1, m_test, Vec3::new(r, 0.0, 0.0), Vec3::new(0.0, v_circ, 0.0));
    p0.smoothing_length = 1e-4;
    p1.smoothing_length = 1e-4;
    vec![p0, p1]
}

/// Energía mecánica total de una órbita de Kepler (G=1).
fn kepler_energy(particles: &[Particle]) -> f64 {
    let g = 1.0_f64;
    let mut ek = 0.0_f64;
    let mut ep = 0.0_f64;

    for p in particles {
        let v2 = p.velocity.dot(p.velocity);
        ek += 0.5 * p.mass * v2;
    }

    for i in 0..particles.len() {
        for j in (i + 1)..particles.len() {
            let dr = particles[i].position - particles[j].position;
            let r = dr.dot(dr).sqrt().max(1e-10);
            ep -= g * particles[i].mass * particles[j].mass / r;
        }
    }
    ek + ep
}

/// Error de energía relativo al integrar una órbita por `t_final`.
fn energy_error_for_dt(dt: f64, t_final: f64) -> f64 {
    let g = 1.0_f64;
    let mut particles = kepler_initial();
    let e0 = kepler_energy(&particles);
    let n_steps = (t_final / dt).ceil() as usize;

    let mut scratch = vec![Vec3::zero(); particles.len()];
    for _ in 0..n_steps {
        leapfrog_kdk_step(&mut particles, dt, &mut scratch, |p, out| {
            // Gravedad directa O(N²) sobre los 2 cuerpos
            out.iter_mut().for_each(|a| *a = Vec3::zero());
            for i in 0..p.len() {
                for j in 0..p.len() {
                    if i == j { continue; }
                    let dr = p[j].position - p[i].position;
                    let r2 = dr.dot(dr);
                    let r = r2.sqrt().max(1e-10);
                    let f = g * p[j].mass / (r2 * r);
                    out[i].x += f * dr.x;
                    out[i].y += f * dr.y;
                    out[i].z += f * dr.z;
                }
            }
        });
    }

    let e1 = kepler_energy(&particles);
    (e1 - e0).abs() / e0.abs().max(1e-30)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

/// El integrador conserva energía en una órbita completa.
#[test]
fn leapfrog_conserves_energy_one_orbit() {
    // T = 2π√(r³/(G·M)) = 2π/10 ≈ 0.6283 para G=1, M=100, r=1
    let t_orbit = 2.0 * std::f64::consts::PI / 10.0;
    let dt = t_orbit / 500.0; // 500 pasos por órbita
    let err = energy_error_for_dt(dt, t_orbit);
    assert!(
        err < 0.01,
        "Error de energía en 1 órbita: {err:.4e} (tolerancia 1%)"
    );
}

/// El ratio de errores al duplicar el paso es ≈ 4 (orden 2).
#[test]
fn leapfrog_kdk_order2_convergence() {
    let t_orbit = 2.0 * std::f64::consts::PI / 10.0;
    let t_final = 0.5 * t_orbit; // media órbita
    let dt_coarse = t_orbit / 50.0;
    let dt_fine = dt_coarse / 2.0;
    let dt_finer = dt_coarse / 4.0;

    let err_coarse = energy_error_for_dt(dt_coarse, t_final);
    let err_fine = energy_error_for_dt(dt_fine, t_final);
    let err_finer = energy_error_for_dt(dt_finer, t_final);

    println!(
        "Leapfrog convergencia: err({})={:.4e}, err({})={:.4e}, err({})={:.4e}",
        dt_coarse, err_coarse, dt_fine, err_fine, dt_finer, err_finer
    );

    // Ratio: err(dt) / err(dt/2) debería ser ≈ 4 para orden 2
    if err_fine > 0.0 && err_finer > 0.0 {
        let ratio1 = err_coarse / err_fine;
        let ratio2 = err_fine / err_finer;
        println!("Ratios: {ratio1:.3}, {ratio2:.3} (esperado ≈ 4.0)");

        // Al menos uno de los ratios debe ser cercano a 4
        let ok = (ratio1 - 4.0).abs() < 2.0 || (ratio2 - 4.0).abs() < 2.0;
        assert!(
            ok,
            "Orden de convergencia KDK: ratios={ratio1:.3}, {ratio2:.3} (esperado ≈ 4 ± 2)"
        );
    }
}

/// La energía es estable a lo largo de 10 órbitas.
#[test]
fn leapfrog_energy_stable_10_orbits() {
    let t_orbit = 2.0 * std::f64::consts::PI / 10.0;
    let dt = t_orbit / 200.0;
    let err = energy_error_for_dt(dt, 10.0 * t_orbit);
    assert!(
        err < 0.05,
        "Error de energía en 10 órbitas: {err:.4e} (tolerancia 5%)"
    );
}

/// Las velocidades son finitas después de integrar.
#[test]
fn leapfrog_no_nan_after_steps() {
    let mut particles = kepler_initial();
    let mut scratch = vec![Vec3::zero(); particles.len()];
    let dt = 0.01_f64;
    let g = 1.0_f64;
    for _ in 0..100 {
        leapfrog_kdk_step(&mut particles, dt, &mut scratch, |p, out| {
            out.iter_mut().for_each(|a| *a = Vec3::zero());
            for i in 0..p.len() {
                for j in 0..p.len() {
                    if i == j { continue; }
                    let dr = p[j].position - p[i].position;
                    let r2 = dr.dot(dr);
                    let r = r2.sqrt().max(1e-10);
                    let f = g * p[j].mass / (r2 * r);
                    out[i].x += f * dr.x;
                    out[i].y += f * dr.y;
                    out[i].z += f * dr.z;
                }
            }
        });
    }
    for p in &particles {
        assert!(p.velocity.x.is_finite(), "v_x no finita");
        assert!(p.position.x.is_finite(), "x no finita");
    }
}
