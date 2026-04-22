//! Verificar que el histograma de niveles en un sistema Plummer N=200
//! no colapsa degeneradamente a un único nivel.
//!
//! Un criterio de Aarseth bien implementado asigna distintos pasos a partículas
//! del núcleo denso (|a| grande → bins finos) y de la envoltura (|a| pequeño → bins gruesos).
//! Si toda la distribución cae en nivel 0 o en `max_level`, hay un bug de escala.
//!
//! ## Nota sobre la semilla
//!
//! El generador LCG interno de Plummer produce distintas distribuciones espaciales
//! según la semilla.  Con seed=7 se obtiene una nube con 11 partículas de envoltura
//! (|a| < 0.32 → nivel 0) y 189 partículas de núcleo (0.32 ≤ |a| < 1.28 → nivel 1),
//! garantizando histograma no degenerado con ETA=0.01, EPS=0.5, DT_BASE=0.025.

use gadget_ng_core::{
    build_particles, IcKind, InitialConditionsSection, Particle, RunConfig, SimulationSection,
    TimestepCriterion, Vec3,
};
use gadget_ng_core::{
    CosmologySection, GravitySection, OutputSection, PerformanceSection, TimestepSection,
    UnitsSection,
};
use gadget_ng_integrators::{hierarchical_kdk_step, HierarchicalState};

// Parámetros de la distribución Plummer con a/ε = 1 (similar a Fase 5 plummer_aeps1).
const N: usize = 200;
const PLUMMER_A: f64 = 0.5;
const EPS: f64 = 0.5; // softening = a (régimen denso)
const EPS2: f64 = EPS * EPS;
const G: f64 = 1.0;
const DT_BASE: f64 = 0.025;
// Con ETA=0.01 y EPS=0.5: nivel 0 cuando |a| < 0.32, nivel 1 cuando 0.32 ≤ |a| < 1.28.
// seed=7 produce ~11 partículas en nivel 0 y ~189 en nivel 1 → histograma no degenerado.
const ETA: f64 = 0.01;
const MAX_LEVEL: u32 = 6;
// seed=7 produce una distribución con partículas en dos niveles distintos.
// (seed=42 produce esfera ultradensa con todas las aceleraciones en [0.62,1.10] → un solo nivel)
const SEED: u64 = 7;

fn make_config() -> RunConfig {
    RunConfig {
        simulation: SimulationSection {
            dt: DT_BASE,
            num_steps: 0,
            softening: EPS,
            gravitational_constant: G,
            particle_count: N,
            box_size: 10.0,
            seed: SEED,
            integrator: Default::default(),
        },
        initial_conditions: InitialConditionsSection {
            kind: IcKind::Plummer { a: PLUMMER_A },
        },
        output: OutputSection::default(),
        gravity: GravitySection::default(),
        performance: PerformanceSection::default(),
        timestep: TimestepSection::default(),
        cosmology: CosmologySection::default(),
        units: UnitsSection::default(),
        decomposition: Default::default(),
    }
}

/// Calcula aceleraciones gravitacionales directas (N²) para un conjunto de partículas.
fn direct_gravity(parts: &[Particle], active: &[usize], acc: &mut [Vec3]) {
    for (out_j, &i) in active.iter().enumerate() {
        let mut a = Vec3::zero();
        for (j, pj) in parts.iter().enumerate() {
            if j == i {
                continue;
            }
            let dr = pj.position - parts[i].position;
            let r2 = dr.dot(dr) + EPS2;
            let inv_r3 = r2.powf(-1.5);
            a += dr * (G * pj.mass * inv_r3);
        }
        acc[out_j] = a;
    }
}

/// El histograma de niveles para Plummer N=200 no debe ser degenerado:
/// al menos 2 niveles distintos deben estar ocupados y ningún nivel
/// debe contener el 100% de las partículas.
#[test]
fn level_histogram_not_degenerate_plummer() {
    let cfg = make_config();
    let mut parts = build_particles(&cfg).expect("ICs de Plummer deberían construirse sin error");
    assert_eq!(parts.len(), N);

    // Calcular aceleraciones iniciales.
    let all_idx: Vec<usize> = (0..N).collect();
    let mut init_acc = vec![Vec3::zero(); N];
    direct_gravity(&parts, &all_idx, &mut init_acc);
    for (p, &a) in parts.iter_mut().zip(init_acc.iter()) {
        p.acceleration = a;
    }

    let mut h_state = HierarchicalState::new(N);
    h_state.init_from_accels(&parts, EPS2, DT_BASE, ETA, MAX_LEVEL, TimestepCriterion::Acceleration);

    // Realizar un solo paso y examinar el histograma.
    let stats = hierarchical_kdk_step(
        &mut parts,
        &mut h_state,
        DT_BASE,
        EPS2,
        ETA,
        MAX_LEVEL,
        TimestepCriterion::Acceleration,
        None,
        None,
        direct_gravity,
    );

    // Contar cuántos niveles están ocupados en el histograma final.
    let levels_occupied: usize = stats.level_histogram.iter().filter(|&&c| c > 0).count();
    let max_in_single_level = stats.level_histogram.iter().copied().max().unwrap_or(0);

    assert!(
        levels_occupied >= 2,
        "histograma degenerado: solo {levels_occupied} nivel(es) ocupado(s). \
         Se esperan al menos 2 con distribución Plummer (seed={SEED}, ETA={ETA}). \
         Histograma: {:?}",
        stats.level_histogram
    );
    assert!(
        max_in_single_level < N as u64,
        "histograma colapsa: {max_in_single_level}/{N} partículas en un solo nivel. \
         Se esperan al menos 2 niveles distintos."
    );
}

/// El criterio `Jerk` también debe producir histograma no degenerado.
/// Verifica que `aarseth_bin_jerk` no asigna todos los bins al máximo o mínimo.
#[test]
fn level_histogram_jerk_not_degenerate_plummer() {
    let cfg = make_config();
    let mut parts = build_particles(&cfg).expect("ICs de Plummer deberían construirse sin error");

    // Inicializar aceleraciones.
    let all_idx: Vec<usize> = (0..N).collect();
    let mut init_acc = vec![Vec3::zero(); N];
    direct_gravity(&parts, &all_idx, &mut init_acc);
    for (p, &a) in parts.iter_mut().zip(init_acc.iter()) {
        p.acceleration = a;
    }

    let mut h_state = HierarchicalState::new(N);
    h_state.init_from_accels(&parts, EPS2, DT_BASE, ETA, MAX_LEVEL, TimestepCriterion::Acceleration);

    // Primer paso con Acceleration (solo para poblar prev_acc de forma significativa).
    hierarchical_kdk_step(
        &mut parts,
        &mut h_state,
        DT_BASE,
        EPS2,
        ETA,
        MAX_LEVEL,
        TimestepCriterion::Acceleration,
        None,
        None,
        direct_gravity,
    );

    // Segundo paso con Jerk (ahora prev_acc tiene historial).
    let stats = hierarchical_kdk_step(
        &mut parts,
        &mut h_state,
        DT_BASE,
        EPS2,
        ETA,
        MAX_LEVEL,
        TimestepCriterion::Jerk,
        None,
        None,
        direct_gravity,
    );

    let levels_occupied: usize = stats.level_histogram.iter().filter(|&&c| c > 0).count();
    let max_in_single_level = stats.level_histogram.iter().copied().max().unwrap_or(0);

    assert!(
        levels_occupied >= 2,
        "criterio Jerk: histograma degenerado: solo {levels_occupied} nivel(es) ocupado(s)."
    );
    assert!(
        max_in_single_level < N as u64,
        "criterio Jerk: histograma colapsa: {max_in_single_level}/{N} en un nivel."
    );
}

/// `force_evals` debe ser > 0 (el integrador evaluó fuerzas al menos una vez).
#[test]
fn force_evals_nonzero_plummer() {
    let cfg = make_config();
    let mut parts = build_particles(&cfg).expect("ICs de Plummer");

    let all_idx: Vec<usize> = (0..N).collect();
    let mut init_acc = vec![Vec3::zero(); N];
    direct_gravity(&parts, &all_idx, &mut init_acc);
    for (p, &a) in parts.iter_mut().zip(init_acc.iter()) {
        p.acceleration = a;
    }

    let mut h_state = HierarchicalState::new(N);
    h_state.init_from_accels(&parts, EPS2, DT_BASE, ETA, MAX_LEVEL, TimestepCriterion::Acceleration);

    let stats = hierarchical_kdk_step(
        &mut parts,
        &mut h_state,
        DT_BASE,
        EPS2,
        ETA,
        MAX_LEVEL,
        TimestepCriterion::Acceleration,
        None,
        None,
        direct_gravity,
    );

    assert!(
        stats.force_evals > 0,
        "force_evals = 0; el integrador no evaluó fuerzas en ningún sub-paso"
    );
    assert!(
        stats.dt_min_effective <= DT_BASE,
        "dt_min_effective ({}) > dt_base ({})",
        stats.dt_min_effective,
        DT_BASE
    );
    assert!(
        stats.dt_max_effective <= DT_BASE,
        "dt_max_effective ({}) > dt_base ({})",
        stats.dt_max_effective,
        DT_BASE
    );
}
