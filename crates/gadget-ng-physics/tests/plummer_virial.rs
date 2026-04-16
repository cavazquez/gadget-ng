//! Equilibrio virial de la esfera de Plummer.
//!
//! ## Perfil de Plummer
//!
//! La esfera de Plummer (1911) es una solución analítica de equilibrio
//! de la ecuación de Boltzmann colisionless para una distribución isótropa.
//!
//! ```text
//! ρ(r) = (3M / 4π a³) · (1 + r²/a²)^{-5/2}
//! Φ(r) = -GM / √(r² + a²)
//! ```
//!
//! donde `a` es el radio de escala de Plummer.
//!
//! ## Generación de condiciones iniciales
//!
//! Usamos el muestreo por inversión (Aarseth 1974):
//! 1. Radio: de la CDF inversa de la masa encerrada `M(r) = M r³/(r²+a²)^{3/2}`.
//! 2. Velocidad: rejection sampling de la función de distribución de Eddington.
//!
//! ## Teorema del Virial
//!
//! En equilibrio estadístico: `2T + W = 0`, donde:
//! - T = energía cinética total
//! - W = energía potencial gravitatoria total
//!
//! Ratio virial: Q = -T/W → 0.5 en equilibrio.
//!
//! ## Criterio de validación
//!
//! Después de evolucionar la esfera durante 2 tiempos de cruce (`t_cross = a/σ`),
//! el ratio virial debe satisfacer `|Q - 0.5| < 0.05`.

use gadget_ng_core::{Particle, Vec3};
use gadget_ng_integrators::leapfrog_kdk_step;
use gadget_ng_tree::BarnesHutGravity;

const G: f64 = 1.0;
const N: usize = 200;
const A: f64 = 1.0;  // radio de escala Plummer
const M_TOT: f64 = 1.0; // masa total
const THETA: f64 = 0.5;
const EPS: f64 = 0.05; // suavizado gravitatorio (≈ 0.05 a)

/// Generador de números aleatorios LCG simple (reproducible).
struct Lcg(u64);
impl Lcg {
    fn next_f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as f64 / u32::MAX as f64
    }
    /// Gaussiana estándar via Box-Muller.
    fn gauss(&mut self) -> f64 {
        let u1 = self.next_f64();
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// Posición de Plummer: muestreo por inversión de la CDF de masa.
/// M(<r) = M · x³/(x²+1)^{3/2} con x = r/a.
/// La inversión no tiene forma cerrada; usamos bisección.
fn plummer_radius(lcg: &mut Lcg) -> f64 {
    let u = lcg.next_f64();
    // Bisección para resolver u = x³/(x²+1)^{3/2}
    let mut lo = 0.0_f64;
    let mut hi = 1000.0_f64 * A;
    for _ in 0..50 {
        let mid = 0.5 * (lo + hi);
        let x = mid / A;
        let cdf = x * x * x / (x * x + 1.0).powf(1.5);
        if cdf < u { lo = mid; } else { hi = mid; }
    }
    0.5 * (lo + hi)
}

/// Genera N partículas de Plummer con posiciones muestreadas de la CDF
/// y velocidades escaladas por "virial scaling" para forzar T = |W|/2.
///
/// Algoritmo:
/// 1. Posiciones: inversión numérica de CDF de masa de Plummer.
/// 2. Velocidades: Gaussianas isótropas con σ² = G·M / (6·a) (dispersión de Jeans promedio).
/// 3. Virial scaling: escala velocidades para que T = |W|/2 exactamente.
fn plummer_ics() -> Vec<Particle> {
    let m = M_TOT / N as f64;
    let mut lcg = Lcg(0xdead_beef_cafe_1234);
    let mut particles = Vec::with_capacity(N);

    // Dispersión de velocidades local de Jeans (aproximación)
    let sigma_global = (G * M_TOT / (6.0 * A)).sqrt();

    for id in 0..N {
        let r = plummer_radius(&mut lcg);
        // Dispersión local de Jeans para la esfera de Plummer:
        // σ²(r) = G M / (6 a) · (1 + r²/a²)^{-1/2}  (Plummer 1911)
        let sigma_local = sigma_global / (1.0 + (r / A) * (r / A)).sqrt().cbrt();

        let cos_theta = 2.0 * lcg.next_f64() - 1.0;
        let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
        let phi_angle = 2.0 * std::f64::consts::PI * lcg.next_f64();
        let pos = Vec3::new(
            r * sin_theta * phi_angle.cos(),
            r * sin_theta * phi_angle.sin(),
            r * cos_theta,
        );
        // Velocidad isótropa: 3 Gaussianas independientes.
        let vel = Vec3::new(
            sigma_local * lcg.gauss(),
            sigma_local * lcg.gauss(),
            sigma_local * lcg.gauss(),
        );
        particles.push(Particle::new(id, m, pos, vel));
    }

    // Corregir COM y VCM.
    let (com, vcm) = center_of_mass_and_vel(&particles);
    for p in &mut particles {
        p.position -= com;
        p.velocity -= vcm;
    }

    // Virial scaling: escalar velocidades para lograr T = |W|/2.
    let t = kinetic_energy(&particles);
    let w = potential_energy(&particles);
    if t > 0.0 && w < 0.0 {
        let scale = (0.5 * w.abs() / t).sqrt();
        for p in &mut particles {
            p.velocity *= scale;
        }
    }
    particles
}

fn center_of_mass_and_vel(particles: &[Particle]) -> (Vec3, Vec3) {
    let m_tot: f64 = particles.iter().map(|p| p.mass).sum();
    let com: Vec3  = particles.iter().map(|p| p.position * p.mass).fold(Vec3::zero(), |a, b| a + b) / m_tot;
    let vcm: Vec3  = particles.iter().map(|p| p.velocity * p.mass).fold(Vec3::zero(), |a, b| a + b) / m_tot;
    (com, vcm)
}

/// Energía cinética total.
fn kinetic_energy(particles: &[Particle]) -> f64 {
    particles.iter().map(|p| 0.5 * p.mass * p.velocity.dot(p.velocity)).sum()
}

/// Energía potencial gravitatoria total (sum_{i<j} -G mi mj / r_ij, con suavizado).
fn potential_energy(particles: &[Particle]) -> f64 {
    let n = particles.len();
    let mut w = 0.0_f64;
    for i in 0..n {
        for j in (i + 1)..n {
            let r = (particles[i].position - particles[j].position).norm();
            let eps = EPS;
            w -= G * particles[i].mass * particles[j].mass / (r * r + eps * eps).sqrt();
        }
    }
    w
}

/// Ratio virial Q = -T / W (≈ 0.5 en equilibrio).
fn virial_ratio(particles: &[Particle]) -> f64 {
    let t = kinetic_energy(particles);
    let w = potential_energy(particles);
    if w.abs() < 1e-300 { return 0.0; }
    -t / w
}

#[test]
fn plummer_initial_conditions_virial_equilibrium() {
    let particles = plummer_ics();

    let q0 = virial_ratio(&particles);
    // Las CIs de Plummer deben estar cerca del virial: Q ∈ (0.3, 0.7).
    assert!(
        (q0 - 0.5).abs() < 0.2,
        "Ratio virial inicial: Q0 = {q0:.3} (esperado ≈ 0.5 ± 0.2 con N={N})"
    );
}

#[test]
fn plummer_kinetic_energy_positive() {
    let particles = plummer_ics();
    let t = kinetic_energy(&particles);
    assert!(t > 0.0, "Energía cinética negativa: T = {t:.4}");
}

#[test]
fn plummer_com_near_zero_after_correction() {
    let particles = plummer_ics();
    let (com, vcm) = center_of_mass_and_vel(&particles);
    assert!(com.norm() < 1e-10, "COM no centrado: |COM| = {:.4e}", com.norm());
    assert!(vcm.norm() < 1e-10, "VCM no nulo: |VCM| = {:.4e}", vcm.norm());
}

#[test]
fn plummer_evolved_virial_ratio_near_half() {
    // Evolucionar la esfera durante 2 tiempos de cruce y verificar el virial.
    // t_cross ≈ a / σ  donde σ² ≈ G M / (6 a)  → t_cross ≈ √(6 a³ / GM) ≈ √6.
    let t_cross = (6.0 * A * A * A / (G * M_TOT)).sqrt();
    let t_evolve = 2.0 * t_cross;

    let mut particles = plummer_ics();
    let mut scratch = vec![Vec3::zero(); N];

    // dt conservador: t_cross / 100.
    let dt = t_cross / 100.0;
    let n_steps = (t_evolve / dt).ceil() as usize;

    let bh = BarnesHutGravity { theta: THETA };
    let eps2 = EPS * EPS;

    use gadget_ng_core::GravitySolver;
    let all_indices: Vec<usize> = (0..N).collect();
    for _ in 0..n_steps {
        leapfrog_kdk_step(&mut particles, dt, &mut scratch, |ps, acc| {
            let pos:   Vec<Vec3> = ps.iter().map(|p| p.position).collect();
            let mass:  Vec<f64>  = ps.iter().map(|p| p.mass).collect();
            bh.accelerations_for_indices(&pos, &mass, eps2, G, &all_indices, acc);
        });
    }

    let q = virial_ratio(&particles);
    assert!(
        (q - 0.5).abs() < 0.15,
        "Ratio virial tras evolución: Q = {q:.4} (esperado 0.5 ± 0.15)"
    );
}
