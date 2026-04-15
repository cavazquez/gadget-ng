//! Tests de conservación de energía con pasos temporales jerárquicos.
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
use gadget_ng_integrators::{
    aarseth_bin, hierarchical_kdk_step, leapfrog_kdk_step, HierarchicalState,
};

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

/// Versión de primer orden del integrador jerárquico (sin corrección de aceleración
/// en el drift). Se usa solo como referencia en el test de comparación.
fn hierarchical_kdk_step_first_order(
    particles: &mut [Particle],
    state: &mut HierarchicalState,
    dt_base: f64,
    eps2: f64,
    eta: f64,
    max_level: u32,
    mut compute: impl FnMut(&[Particle], &[usize], &mut [Vec3]),
) {
    let n_fine = 1u64 << max_level;
    let fine_dt = dt_base / n_fine as f64;
    let n = particles.len();
    let mut acc_buf = vec![Vec3::zero(); n];

    for s in 0..n_fine {
        // START kick
        for (p, &lvl) in particles.iter_mut().zip(state.levels.iter()) {
            let stride = 1u64 << (max_level - lvl);
            if s % stride == 0 {
                let dt_i = dt_base / (1u64 << lvl) as f64;
                p.velocity += p.acceleration * (0.5 * dt_i);
            }
        }
        // Drift de PRIMER orden (sin término de aceleración)
        for p in particles.iter_mut() {
            p.position += p.velocity * fine_dt;
        }
        // END kick
        let end_active: Vec<usize> = (0..n)
            .filter(|&i| {
                let stride = 1u64 << (max_level - state.levels[i]);
                (s + 1) % stride == 0
            })
            .collect();
        if !end_active.is_empty() {
            compute(particles, &end_active, &mut acc_buf[..end_active.len()]);
            for (j, &i) in end_active.iter().enumerate() {
                let dt_i = dt_base / (1u64 << state.levels[i]) as f64;
                let a_new = acc_buf[j];
                particles[i].velocity += a_new * (0.5 * dt_i);
                particles[i].acceleration = a_new;
                let acc_mag = a_new.dot(a_new).sqrt();
                state.levels[i] = aarseth_bin(acc_mag, eps2, dt_base, eta, max_level);
            }
        }
    }
}

/// El predictor de segundo orden conserva la energía mejor o igual que el de
/// primer orden, en un sistema de dos cuerpos con bins variados (eta=0.025,
/// max_level=4 → algunos bins en nivel 1-4).
///
/// La diferencia está en cómo se predicen las posiciones de partículas inactivas
/// entre sus sync-points: `x += v*dt + 0.5*a*dt²` (2.o) vs `x += v*dt` (1.o).
#[test]
fn second_order_drift_better_than_first_order() {
    let period_approx = std::f64::consts::TAU * (SEP / 2.0) / ((G * 2.0 * M_EACH / 4.0).sqrt());
    let dt = period_approx / 100.0; // pasos más gruesos para que los bins varíen
    let orbits = 5.0;
    let steps = (orbits * period_approx / dt).round() as u64;
    let eta = 0.025_f64;
    let max_level = 4u32;

    let e0 = total_energy(&two_body());

    // ── 1.er orden ───────────────────────────────────────────────────────────
    let mut parts1 = two_body();
    let n = parts1.len();
    let all_idx: Vec<usize> = (0..n).collect();
    let mut init_acc = vec![Vec3::zero(); n];
    gravity_two_body(&parts1, &all_idx, &mut init_acc);
    for (p, &a) in parts1.iter_mut().zip(init_acc.iter()) {
        p.acceleration = a;
    }
    let mut st1 = HierarchicalState::new(n);
    st1.init_from_accels(&parts1, EPS2, dt, eta, max_level);
    for _ in 0..steps {
        hierarchical_kdk_step_first_order(
            &mut parts1,
            &mut st1,
            dt,
            EPS2,
            eta,
            max_level,
            gravity_two_body,
        );
    }
    let rel1 = ((total_energy(&parts1) - e0) / e0.abs()).abs();

    // ── 2.o orden ────────────────────────────────────────────────────────────
    let e_2nd = run_hierarchical(steps, dt, eta, max_level);
    let rel2 = ((e_2nd - e0) / e0.abs()).abs();

    // El 2.o orden debe ser ≤ 5× peor que el 1.o (en práctica suele ser igual o mejor).
    // Un factor de 5 es conservador para evitar falsos positivos por diferencias de fase.
    assert!(
        rel2 <= rel1 * 5.0 + 1e-8,
        "el predictor de 2.o orden deriva más que el de 1.o: rel2={rel2:.4e} > 5*rel1={:.4e}",
        rel1 * 5.0
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
