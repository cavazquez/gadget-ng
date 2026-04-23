//! Problema de tres cuerpos — Órbita en figura-8 (Chenciner & Montgomery 2000).
//!
//! ## Descripción física
//!
//! La órbita figura-8 es una solución periódica exacta del problema de 3 cuerpos
//! con masas iguales m=1 y G=1. Fue descubierta numéricamente por Moore (1993) y
//! demostrada existir analíticamente por Chenciner & Montgomery (2000).
//!
//! ## Condiciones iniciales
//!
//! ```text
//! r₁ = (-0.97000436,  0.24308753, 0)
//! r₂ = ( 0.00000000,  0.00000000, 0)
//! r₃ = ( 0.97000436, -0.24308753, 0)
//!
//! v₁ = ( 0.93240737/2,  0.86473146/2, 0)  =  (0.46620368, 0.43236573, 0)
//! v₂ = (-0.93240737,   -0.86473146,   0)
//! v₃ = ( 0.93240737/2,  0.86473146/2, 0)  =  (0.46620368, 0.43236573, 0)
//! ```
//!
//! ## Propiedades conservadas
//!
//! - Período orbital: T ≈ 6.3259
//! - Energía total: E = -1.287... (negativa; sistema ligado)
//! - Momento angular total: L = 0 (por simetría de la solución)
//! - Momento lineal total: p = 0 (sistema en el COM)
//!
//! ## Criterios de validación
//!
//! - `|ΔE/E| < 0.1%` tras 1 período completo
//! - `|p_total| < 1e-12` (momento lineal conservado)
//! - Retorno a posición inicial: `|r_i(T) - r_i(0)| < 0.01`
//!
//! ## Nota sobre precisión
//!
//! La órbita figura-8 es CAÓTICA — pequeños errores en las ICs llevan a divergencia
//! exponencial. El leapfrog KDK de segundo orden con dt=0.001 proporciona buena
//! conservación durante ~1 período. Para múltiples períodos se necesita dt < 1e-4.
//!
//! ## Referencias
//!
//! - Chenciner, A. & Montgomery, R. (2000). Ann. of Math. 152, 881–901.
//! - Moore, C. (1993). Phys. Rev. Lett. 70, 3675.

use gadget_ng_core::{Particle, Vec3};
use gadget_ng_integrators::leapfrog_kdk_step;

// ── Constantes físicas ────────────────────────────────────────────────────────
const G: f64 = 1.0;
const M: f64 = 1.0;
const EPS2: f64 = 0.0; // sin suavizado: la figura-8 evita encuentros cercanos

// Período orbital de la figura-8 (Chenciner & Montgomery 2000).
const T_PERIOD: f64 = 6.3259;

// ── Condiciones iniciales ────────────────────────────────────────────────────
fn figure8_ics() -> [Particle; 3] {
    let r1 = Vec3::new(-0.97000436, 0.24308753, 0.0);
    let r2 = Vec3::new(0.0, 0.0, 0.0);
    let r3 = Vec3::new(0.97000436, -0.24308753, 0.0);

    // v₂ = -(v₁ + v₃); v₁ = v₃ (simetría)
    let v1 = Vec3::new(0.93240737 / 2.0, 0.86473146 / 2.0, 0.0);
    let v2 = Vec3::new(-0.93240737, -0.86473146, 0.0);
    let v3 = Vec3::new(0.93240737 / 2.0, 0.86473146 / 2.0, 0.0);

    [
        Particle::new(0, M, r1, v1),
        Particle::new(1, M, r2, v2),
        Particle::new(2, M, r3, v3),
    ]
}

// ── Aceleración gravitatoria ──────────────────────────────────────────────────
/// Calcula la aceleración gravitatoria sobre todas las partículas (interacción directa).
fn compute_accels(particles: &[Particle]) -> [Vec3; 3] {
    let mut acc = [Vec3::zero(); 3];
    for i in 0..3 {
        for j in 0..3 {
            if i == j {
                continue;
            }
            let dr = particles[j].position - particles[i].position;
            let r2 = dr.dot(dr) + EPS2;
            let r3 = r2 * r2.sqrt();
            acc[i] += dr * (G * M / r3);
        }
    }
    acc
}

// ── Energías y momentos ───────────────────────────────────────────────────────
fn total_energy(particles: &[Particle]) -> f64 {
    // Energía cinética
    let ke: f64 = particles
        .iter()
        .map(|p| 0.5 * p.mass * p.velocity.dot(p.velocity))
        .sum();
    // Energía potencial
    let mut pe = 0.0_f64;
    for i in 0..3 {
        for j in (i + 1)..3 {
            let r = (particles[i].position - particles[j].position).norm();
            pe -= G * particles[i].mass * particles[j].mass / r;
        }
    }
    ke + pe
}

fn total_momentum(particles: &[Particle]) -> Vec3 {
    particles
        .iter()
        .map(|p| p.velocity * p.mass)
        .fold(Vec3::zero(), |a, b| a + b)
}

fn total_angular_momentum(particles: &[Particle]) -> Vec3 {
    particles
        .iter()
        .map(|p| {
            let r = p.position;
            let v = p.velocity;
            // L = r × (m·v)
            Vec3::new(
                r.y * v.z - r.z * v.y,
                r.z * v.x - r.x * v.z,
                r.x * v.y - r.y * v.x,
            ) * p.mass
        })
        .fold(Vec3::zero(), |a, b| a + b)
}

// ── Integración ───────────────────────────────────────────────────────────────
fn integrate_figure8(particles: &mut Vec<Particle>, n_steps: usize, dt: f64) {
    let mut scratch = vec![Vec3::zero(); 3];
    for _ in 0..n_steps {
        leapfrog_kdk_step(particles, dt, &mut scratch, |ps, acc| {
            let accs = compute_accels(ps);
            for (a, &fa) in acc.iter_mut().zip(accs.iter()) {
                *a = fa;
            }
        });
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[test]
fn figure8_initial_conditions_symmetric() {
    let ps = figure8_ics();

    // Momento lineal total = 0 (v₁ + v₂ + v₃ = 0 por construcción).
    let p_tot: Vec3 = ps
        .iter()
        .map(|p| p.velocity * p.mass)
        .fold(Vec3::zero(), |a, b| a + b);
    assert!(
        p_tot.norm() < 1e-12,
        "|p_total| = {:.4e} (debe ser 0)",
        p_tot.norm()
    );

    // Simetría de posiciones: r₁ = -r₃.
    let diff = ps[0].position + ps[2].position;
    assert!(
        diff.norm() < 1e-10,
        "r₁ + r₃ = {:.4e} (debe ser 0; simetría figura-8)",
        diff.norm()
    );

    // Energía total negativa (sistema ligado).
    let e = total_energy(&ps.to_vec());
    assert!(
        e < 0.0,
        "Energía total E = {e:.6} (debe ser negativa para sistema ligado)"
    );

    // Momento angular total ≈ 0 en z (simetría de la solución).
    let l = total_angular_momentum(&ps.to_vec());
    assert!(
        l.z.abs() < 1e-10,
        "|L_z| = {:.4e} (debe ser ~0 para figura-8)",
        l.z.abs()
    );
}

#[test]
fn figure8_energy_conserved_half_period() {
    // Verifica conservación de energía durante medio período con dt=0.001.
    let ics = figure8_ics();
    let mut particles: Vec<Particle> = ics.to_vec();

    let dt = 0.001;
    let n_steps = (0.5 * T_PERIOD / dt).ceil() as usize;

    let e0 = total_energy(&particles);
    integrate_figure8(&mut particles, n_steps, dt);
    let e_f = total_energy(&particles);

    let de_rel = (e_f - e0).abs() / e0.abs();
    assert!(
        de_rel < 0.001,
        "|ΔE/E| = {de_rel:.4e} tras T/2 (esperado < 0.1% con dt=0.001)"
    );
}

#[test]
fn figure8_momentum_conserved() {
    // El momento lineal debe ser 0 a máquina para toda la integración.
    let ics = figure8_ics();
    let mut particles: Vec<Particle> = ics.to_vec();

    let dt = 0.001;
    let n_steps = (T_PERIOD / dt).ceil() as usize;

    integrate_figure8(&mut particles, n_steps, dt);

    let p = total_momentum(&particles);
    assert!(
        p.norm() < 1e-10,
        "|p_total| = {:.4e} (debe ser ~0; leapfrog conserva momento linealmente)",
        p.norm()
    );
}

#[test]
fn figure8_angular_momentum_nearly_zero() {
    // El momento angular total es 0 por simetría; el leapfrog lo preserva.
    let ics = figure8_ics();
    let mut particles: Vec<Particle> = ics.to_vec();

    let dt = 0.001;
    let n_steps = (T_PERIOD / dt).ceil() as usize;

    let l0 = total_angular_momentum(&particles);
    integrate_figure8(&mut particles, n_steps, dt);
    let l_f = total_angular_momentum(&particles);

    assert!(
        (l_f.z - l0.z).abs() < 1e-9,
        "ΔL_z = {:.4e} (esperado ~0; L₀={:.4e})",
        (l_f.z - l0.z).abs(),
        l0.z
    );
}

/// Test de retorno posicional tras un período completo. Más costoso; requiere
/// que la integración sea suficientemente precisa (dt=0.0005).
#[test]
#[ignore = "test fino: cargo test -p gadget-ng-physics --release -- --include-ignored"]
fn figure8_returns_to_start_one_period() {
    let ics = figure8_ics();
    let mut particles: Vec<Particle> = ics.to_vec();

    let dt = 0.0005; // dt más fino para retorno preciso
    let n_steps = (T_PERIOD / dt).round() as usize;

    integrate_figure8(&mut particles, n_steps, dt);

    // Cada partícula debe volver a su posición inicial con error < 1% de la separación.
    let separation = 2.0 * 0.97000436; // separación típica ≈ 1.94 UA
    for (i, (p_f, p_0)) in particles.iter().zip(ics.iter()).enumerate() {
        let pos_err = (p_f.position - p_0.position).norm();
        assert!(
            pos_err < 0.01 * separation,
            "Partícula {i}: |r_f - r₀| = {pos_err:.4e} (esperado < {:.4e})",
            0.01 * separation
        );
    }

    // Energía conservada con dt=0.0005: < 0.01%.
    let e0 = total_energy(&ics.to_vec());
    let e_f = total_energy(&particles);
    let de_rel = (e_f - e0).abs() / e0.abs();
    assert!(
        de_rel < 0.0001,
        "|ΔE/E| = {de_rel:.4e} tras T completo con dt=0.0005 (esperado < 0.01%)"
    );
}
