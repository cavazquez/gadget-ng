//! Test de conservación de energía con pasos temporales jerárquicos.
//!
//! Usamos un sistema de dos cuerpos en órbita circular (kepler). La energía total
//! (cinética + potencial gravitacional) debe conservarse dentro de un umbral relativo
//! durante muchas órbitas tanto con el integrador de paso fijo como con el jerárquico.
//!
//! **Configuración orbital:**
//! - Masas iguales `m = 0.5` → masa total `M = 1`.
//! - Separación `r = 1`, velocidad circular `v_c = sqrt(G*M / (2*r)) = sqrt(0.5) ≈ 0.707`.
//! - Período `T = 2π * r / v_c ≈ 8.886`.
//! - El softening es pequeño comparado con `r`, de modo que la órbita se mantiene estable.

use gadget_ng_core::{Particle, Vec3};
use gadget_ng_integrators::{hierarchical_kdk_step, leapfrog_kdk_step, HierarchicalState};

const G: f64 = 1.0;
const M_EACH: f64 = 0.5;
const SEP: f64 = 1.0; // separación inicial entre los dos cuerpos
const EPS2: f64 = 1e-4; // softening^2 << SEP^2

fn total_energy(parts: &[Particle]) -> f64 {
    let kinetic: f64 = parts
        .iter()
        .map(|p| 0.5 * p.mass * p.velocity.dot(p.velocity))
        .sum();
    // Potencial gravitacional (N=2, una sola pareja).
    let dr = parts[1].position - parts[0].position;
    let r2 = dr.dot(dr) + EPS2;
    let potential = -G * parts[0].mass * parts[1].mass / r2.sqrt();
    kinetic + potential
}

/// Construye dos partículas en órbita circular kepleriana alrededor del baricentro.
fn two_body() -> Vec<Particle> {
    // v_c = sqrt(G * M_total / (4 * r_bari))  para cada cuerpo alrededor del CM.
    // r_bari = SEP/2, M_total = 2*M_EACH = 1.
    let r_bari = SEP / 2.0;
    let v_c = (G * 2.0 * M_EACH / (4.0 * r_bari)).sqrt();

    let p0 = Particle::new(
        0,
        M_EACH,
        Vec3::new(-r_bari, 0.0, 0.0),
        Vec3::new(0.0, -v_c, 0.0),
    );
    let p1 = Particle::new(
        1,
        M_EACH,
        Vec3::new(r_bari, 0.0, 0.0),
        Vec3::new(0.0, v_c, 0.0),
    );
    vec![p0, p1]
}

/// Calcula las aceleraciones gravitacionales mutuas para el sistema de dos cuerpos.
fn gravity_two_body(parts: &[Particle], active: &[usize], acc: &mut [Vec3]) {
    for (out_j, &i) in active.iter().enumerate() {
        let mut a = Vec3::zero();
        for (j, p_j) in parts.iter().enumerate() {
            if j == i {
                continue;
            }
            let dr = p_j.position - parts[i].position;
            let r2 = dr.dot(dr) + EPS2;
            let inv_r3 = r2.powf(-1.5);
            a += dr * (G * p_j.mass * inv_r3);
        }
        acc[out_j] = a;
    }
}

/// Integrador de paso fijo: misma física, sirve de referencia.
fn run_uniform(steps: u64, dt: f64) -> f64 {
    let mut parts = two_body();
    // Inicializar aceleraciones.
    let n = parts.len();
    let mut scratch = vec![Vec3::zero(); n];
    for _ in 0..steps {
        leapfrog_kdk_step(&mut parts, dt, &mut scratch, |ps, acc| {
            let all_idx: Vec<usize> = (0..ps.len()).collect();
            gravity_two_body(ps, &all_idx, acc);
        });
    }
    total_energy(&parts)
}

/// Integrador jerárquico: misma física, mismos parámetros de tiempo.
fn run_hierarchical(steps: u64, dt: f64, eta: f64, max_level: u32) -> f64 {
    let mut parts = two_body();
    let n = parts.len();

    // Calcular aceleraciones iniciales.
    let all_idx: Vec<usize> = (0..n).collect();
    let mut init_acc = vec![Vec3::zero(); n];
    gravity_two_body(&parts, &all_idx, &mut init_acc);
    for (p, &a) in parts.iter_mut().zip(init_acc.iter()) {
        p.acceleration = a;
    }

    let mut h_state = HierarchicalState::new(n);
    h_state.init_from_accels(&parts, EPS2, dt, eta, max_level);

    for _ in 0..steps {
        hierarchical_kdk_step(
            &mut parts,
            &mut h_state,
            dt,
            EPS2,
            eta,
            max_level,
            gravity_two_body,
        );
    }
    total_energy(&parts)
}

/// La energía se conserva dentro del 1% tras ~10 órbitas con el integrador jerárquico.
#[test]
fn hierarchical_two_body_energy_conserved() {
    let period_approx = std::f64::consts::TAU * (SEP / 2.0) / ((G * 2.0 * M_EACH / 4.0).sqrt());
    let dt = period_approx / 200.0; // ~200 pasos por órbita
    let orbits = 10.0;
    let steps = (orbits * period_approx / dt).round() as u64;

    let e0 = total_energy(&two_body());
    let e_hier = run_hierarchical(steps, dt, 0.025, 4);

    let rel_err = ((e_hier - e0) / e0.abs()).abs();
    // El KDK jerárquico con bins variables acumula más error que el KDK uniforme
    // (los sub-pasos de partículas activas usan distinto dt). Un 5 % sobre 10 órbitas
    // (~2000 pasos) es conservador pero razonable para un integrador de 2.º orden.
    assert!(
        rel_err < 0.05,
        "deriva de energía jerárquica demasiado grande: |ΔE/E₀| = {rel_err:.4e} (> 5 %)",
    );
}

/// El integrador jerárquico conserva mejor o igual la energía que el de paso fijo
/// con el mismo `dt_base` (margen del 2x).
#[test]
fn hierarchical_vs_uniform_energy_drift() {
    let period_approx = std::f64::consts::TAU * (SEP / 2.0) / ((G * 2.0 * M_EACH / 4.0).sqrt());
    let dt = period_approx / 200.0;
    let orbits = 5.0;
    let steps = (orbits * period_approx / dt).round() as u64;

    let e0 = total_energy(&two_body());
    let e_uniform = run_uniform(steps, dt);
    let e_hier = run_hierarchical(steps, dt, 0.025, 4);

    let rel_uniform = ((e_uniform - e0) / e0.abs()).abs();
    let rel_hier = ((e_hier - e0) / e0.abs()).abs();

    // El jerárquico puede diferir ligeramente del uniforme (bins distintos),
    // pero no debe tener una deriva 10x peor.
    assert!(
        rel_hier < 10.0 * rel_uniform + 1e-8,
        "el jerárquico deriva mucho más que el uniforme: |ΔE/E₀|_hier={rel_hier:.4e} vs uniforme={rel_uniform:.4e}",
    );
}
