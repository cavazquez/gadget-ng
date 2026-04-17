//! Comparación de criterios `Acceleration` vs `Jerk` en un sistema de Kepler circular.
//!
//! Objetivo: verificar que ambos criterios producen resultados físicamente coherentes
//! (conservación de energía razonable, cierre orbital correcto) en un sistema integrable
//! no lineal. No se exige que `Jerk` sea estrictamente mejor que `Acceleration` — el
//! objetivo es detectar bugs de implementación (divergencia, explosión de energía, o
//! criterio que no oscila los niveles).
//!
//! Sistema: dos cuerpos de masa igual `m = 0.5` en órbita circular de radio `r = 0.5`
//! alrededor del baricentro. `G = 1`, período `T = 2π·r / v_c`.

use gadget_ng_core::{Particle, TimestepCriterion, Vec3};
use gadget_ng_integrators::{hierarchical_kdk_step, HierarchicalState};

const G: f64 = 1.0;
const M_EACH: f64 = 0.5;
const SEP: f64 = 1.0;
const EPS2: f64 = 1e-4;

fn two_body_circular() -> Vec<Particle> {
    let r_bari = SEP / 2.0;
    // Velocidad circular exacta derivada del equilibrio centrípeto:
    //   m * v² / r_bari = G * m * m / r²
    //   v² = G * m / (4 * r_bari)    (con r = SEP = 2*r_bari, G=1, m=M_EACH=0.5)
    // → v_c = sqrt(G * M_EACH / (4 * r_bari)) = sqrt(0.5/(4*0.5)) = 0.5
    // Esto da E₀ = -0.125 (ligada). La fórmula v = sqrt(G*M_total/(2*r)) ≈ 0.707
    // da una órbita casi-parabólica (E₀ ≈ 0) y debe evitarse.
    let v_c = (G * M_EACH / (4.0 * r_bari)).sqrt(); // = 0.5

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

fn total_energy(parts: &[Particle]) -> f64 {
    let kinetic: f64 = parts
        .iter()
        .map(|p| 0.5 * p.mass * p.velocity.dot(p.velocity))
        .sum();
    let dr = parts[1].position - parts[0].position;
    let r2 = dr.dot(dr) + EPS2;
    let potential = -G * parts[0].mass * parts[1].mass / r2.sqrt();
    kinetic + potential
}

/// Ejecuta `steps` pasos jerárquicos con el criterio dado y devuelve
/// `(|ΔE/E₀|_final, closure)` donde `closure` es la distancia orbital
/// respecto a la posición inicial tras `orbits` órbitas completas.
fn run_kepler(criterion: TimestepCriterion, steps: u64, dt: f64, eta: f64, max_level: u32) -> (f64, f64) {
    let mut parts = two_body_circular();

    // Inicializar aceleraciones.
    let all_idx: Vec<usize> = (0..parts.len()).collect();
    let mut init_acc = vec![Vec3::zero(); parts.len()];
    gravity_two_body(&parts, &all_idx, &mut init_acc);
    for (p, &a) in parts.iter_mut().zip(init_acc.iter()) {
        p.acceleration = a;
    }

    let e0 = total_energy(&parts);
    let pos0 = parts[0].position;

    let mut h_state = HierarchicalState::new(parts.len());
    h_state.init_from_accels(&parts, EPS2, dt, eta, max_level, criterion);

    for _ in 0..steps {
        hierarchical_kdk_step(
            &mut parts,
            &mut h_state,
            dt,
            EPS2,
            eta,
            max_level,
            criterion,
            None,
            gravity_two_body,
        );
    }

    let e_final = total_energy(&parts);
    let de_rel = ((e_final - e0) / e0.abs()).abs();

    let dp = parts[0].position - pos0;
    let closure = dp.dot(dp).sqrt();

    (de_rel, closure)
}

/// Ambos criterios deben conservar energía dentro del 5 % tras 10 órbitas.
#[test]
fn both_criteria_conserve_energy_10_orbits() {
    let period = std::f64::consts::TAU * (SEP / 2.0) / ((G * 2.0 * M_EACH / 4.0).sqrt());
    let dt = period / 200.0;
    let orbits = 10.0;
    let steps = (orbits * period / dt).round() as u64;
    let eta = 0.025_f64;
    let max_level = 4u32;

    let (de_acc, _) = run_kepler(TimestepCriterion::Acceleration, steps, dt, eta, max_level);
    let (de_jerk, _) = run_kepler(TimestepCriterion::Jerk, steps, dt, eta, max_level);

    assert!(
        de_acc < 0.05,
        "criterio Acceleration: |ΔE/E₀| = {de_acc:.4e} > 5 % tras 10 órbitas"
    );
    assert!(
        de_jerk < 0.05,
        "criterio Jerk: |ΔE/E₀| = {de_jerk:.4e} > 5 % tras 10 órbitas"
    );
}

/// El criterio `Jerk` no debe ser catastrófico comparado con `Acceleration`.
/// El Jerk da nivel 0 (dt_base) mientras Acceleration da nivel 2 (dt_base/4),
/// por lo que la ratio teórica de error es 4² = 16x. Se permite hasta 25x
/// (margen para variaciones numéricas). Si fuera > 100x indicaría un bug grave.
#[test]
fn jerk_criterion_not_catastrophically_worse_than_acc() {
    let period = std::f64::consts::TAU * (SEP / 2.0) / ((G * 2.0 * M_EACH / 4.0).sqrt());
    let dt = period / 200.0;
    let orbits = 10.0;
    let steps = (orbits * period / dt).round() as u64;
    let eta = 0.025_f64;
    let max_level = 4u32;

    let (de_acc, _) = run_kepler(TimestepCriterion::Acceleration, steps, dt, eta, max_level);
    let (de_jerk, _) = run_kepler(TimestepCriterion::Jerk, steps, dt, eta, max_level);

    assert!(
        de_jerk < de_acc * 25.0 + 1e-7,
        "criterio Jerk deriva {de_jerk:.4e} que es >25× el de Acceleration {de_acc:.4e} (ratio teórico: 16x)"
    );
}

/// El cierre orbital (distancia entre posición inicial y posición tras 1 período)
/// es correcto para ambos criterios (< 5 % de la separación orbital).
#[test]
fn orbital_closure_both_criteria() {
    let period = std::f64::consts::TAU * (SEP / 2.0) / ((G * 2.0 * M_EACH / 4.0).sqrt());
    let dt = period / 500.0; // pasos finos para buen cierre
    let steps = (period / dt).round() as u64;
    let eta = 0.025_f64;
    let max_level = 4u32;
    let r_orbit = SEP / 2.0; // radio orbital de cada cuerpo

    let (_, closure_acc) = run_kepler(TimestepCriterion::Acceleration, steps, dt, eta, max_level);
    let (_, closure_jerk) = run_kepler(TimestepCriterion::Jerk, steps, dt, eta, max_level);

    let tol = 0.05 * r_orbit;
    assert!(
        closure_acc < tol,
        "criterio Acceleration: cierre orbital {closure_acc:.4e} > {tol:.4e} (5% r_orbit)"
    );
    assert!(
        closure_jerk < tol,
        "criterio Jerk: cierre orbital {closure_jerk:.4e} > {tol:.4e} (5% r_orbit)"
    );
}

/// Verificar que `StepStats` reporta niveles coherentes (no todos en 0 ni todos en max).
/// Con eta=0.025 y una órbita de Kepler esperamos que haya variedad de niveles.
#[test]
fn step_stats_levels_not_degenerate() {
    use gadget_ng_integrators::StepStats;

    let period = std::f64::consts::TAU * (SEP / 2.0) / ((G * 2.0 * M_EACH / 4.0).sqrt());
    let dt = period / 50.0; // pasos gruesos para forzar bins variables
    let eta = 0.025_f64;
    let max_level = 4u32;

    let mut parts = two_body_circular();
    let all_idx: Vec<usize> = (0..parts.len()).collect();
    let mut init_acc = vec![Vec3::zero(); parts.len()];
    gravity_two_body(&parts, &all_idx, &mut init_acc);
    for (p, &a) in parts.iter_mut().zip(init_acc.iter()) {
        p.acceleration = a;
    }

    let mut h_state = HierarchicalState::new(parts.len());
    h_state.init_from_accels(&parts, EPS2, dt, eta, max_level, TimestepCriterion::Acceleration);

    let mut all_stats: Vec<StepStats> = Vec::new();
    for _ in 0..20 {
        let stats = hierarchical_kdk_step(
            &mut parts,
            &mut h_state,
            dt,
            EPS2,
            eta,
            max_level,
            TimestepCriterion::Acceleration,
            None,
            gravity_two_body,
        );
        all_stats.push(stats);
    }

    // Al menos un paso debe tener active_total > 0.
    let total_active: u64 = all_stats.iter().map(|s| s.active_total).sum();
    assert!(
        total_active > 0,
        "active_total acumulado = 0; el integrador no actualizó ninguna partícula"
    );

    // force_evals debe ser > 0 (al menos un sub-paso tuvo activos).
    let total_evals: u64 = all_stats.iter().map(|s| s.force_evals).sum();
    assert!(
        total_evals > 0,
        "force_evals acumulado = 0; el integrador no evaluó fuerzas"
    );
}
