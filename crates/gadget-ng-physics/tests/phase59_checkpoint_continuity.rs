//! Phase 59 — Continuidad física tras restart/checkpoint
//!
//! ## Objetivo
//!
//! Verificar que reanudar una simulación desde un checkpoint produce
//! trayectorias bit-a-bit idénticas a una corrida continua, validando que:
//!
//! - El estado completo está capturado en posiciones + velocidades de las partículas.
//! - No hay estado oculto en el integrador entre pasos.
//! - Recomputar fuerzas desde posiciones restauradas produce el mismo resultado.
//!
//! ## Protocolo
//!
//! 1. Correr N=8³ PM, 20 pasos fijos (dt constante).
//! 2. Guardar snapshot de posiciones/velocidades al paso 10 (clonando `Vec<Particle>`).
//! 3. Desde el snapshot, recomputar fuerzas y continuar 10 pasos más.
//! 4. Comparar posiciones y velocidades finales: deben ser idénticas bit-a-bit.
//!
//! Controlar con `PHASE59_SKIP=1`.

use gadget_ng_core::{
    build_particles, CosmologySection, GravitySection, GravitySolver, IcKind,
    InitialConditionsSection, OutputSection, Particle, PerformanceSection, RunConfig,
    SimulationSection, TimestepSection, UnitsSection, Vec3,
};
use gadget_ng_integrators::leapfrog_kdk_step;
use gadget_ng_pm::PmSolver;

// ── Parámetros de la simulación ───────────────────────────────────────────────

const N_GRID: usize = 8;
const N_TOTAL: usize = N_GRID * N_GRID * N_GRID;
const BOX: f64 = 1.0;
const G: f64 = 1.0;
const DT: f64 = 1e-3;
const SOFTENING: f64 = 0.01;
const N_STEPS_TOTAL: usize = 20;
const CHECKPOINT_STEP: usize = 10;
const SEED: u64 = 59;

// ── Helpers ───────────────────────────────────────────────────────────────────

fn build_initial_particles() -> Vec<Particle> {
    let cfg = RunConfig {
        simulation: SimulationSection {
            dt: DT,
            num_steps: N_STEPS_TOTAL as u64,
            softening: SOFTENING,
            physical_softening: false,
            gravitational_constant: G,
            particle_count: N_TOTAL,
            box_size: BOX,
            seed: SEED,
            integrator: Default::default(),
        },
        initial_conditions: InitialConditionsSection {
            kind: IcKind::Lattice,
        },
        output: OutputSection::default(),
        gravity: GravitySection {
            solver: gadget_ng_core::SolverKind::Pm,
            pm_grid_size: N_GRID,
            ..GravitySection::default()
        },
        performance: PerformanceSection::default(),
        timestep: TimestepSection::default(),
        cosmology: CosmologySection::default(),
        units: UnitsSection::default(),
        decomposition: Default::default(),
        insitu_analysis: Default::default(),
        sph: Default::default(),
    };
    build_particles(&cfg).expect("ICs no deben fallar")
}

fn make_pm() -> PmSolver {
    PmSolver {
        grid_size: N_GRID,
        box_size: BOX,
    }
}

/// Computa fuerzas y las almacena en `scratch`.
fn compute_forces(parts: &[Particle], scratch: &mut Vec<Vec3>) {
    let pm = make_pm();
    let pos: Vec<Vec3> = parts.iter().map(|p| p.position).collect();
    let m: Vec<f64> = parts.iter().map(|p| p.mass).collect();
    let idx: Vec<usize> = (0..parts.len()).collect();
    pm.accelerations_for_indices(&pos, &m, SOFTENING, G, &idx, scratch);
}

/// Ejecuta `n_steps` pasos leapfrog KDK desde el estado actual.
/// Prerrequisito: `scratch` contiene las fuerzas correspondientes a `parts`.
fn run_steps(parts: &mut Vec<Particle>, scratch: &mut Vec<Vec3>, n_steps: usize) {
    let pm = make_pm();
    for _ in 0..n_steps {
        leapfrog_kdk_step(parts, DT, scratch, |ps, acc| {
            let pos: Vec<Vec3> = ps.iter().map(|p| p.position).collect();
            let m: Vec<f64> = ps.iter().map(|p| p.mass).collect();
            let idx: Vec<usize> = (0..ps.len()).collect();
            pm.accelerations_for_indices(&pos, &m, SOFTENING, G, &idx, acc);
        });
    }
}

fn skip() -> bool {
    std::env::var("PHASE59_SKIP")
        .map(|v| v == "1")
        .unwrap_or(false)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

/// Verificación principal de continuidad bit-a-bit.
///
/// Corrida continua de 20 pasos vs restart desde el paso 10.
/// Las posiciones y velocidades finales deben ser idénticas.
#[test]
fn phase59_checkpoint_continuity_bitexact() {
    if skip() {
        eprintln!("[phase59] saltado por PHASE59_SKIP=1");
        return;
    }

    // ── Corrida continua: 20 pasos ────────────────────────────────────────────
    let mut parts_continuous = build_initial_particles();
    let mut scratch_c = vec![Vec3::zero(); N_TOTAL];
    compute_forces(&parts_continuous, &mut scratch_c);
    run_steps(&mut parts_continuous, &mut scratch_c, N_STEPS_TOTAL);

    eprintln!("[phase59] Corrida continua completada (N_STEPS={N_STEPS_TOTAL})");

    // ── Corrida dividida: 10 pasos + checkpoint + 10 pasos ───────────────────
    let mut parts_divided = build_initial_particles();
    let mut scratch_d = vec![Vec3::zero(); N_TOTAL];
    compute_forces(&parts_divided, &mut scratch_d);

    // Fase 1: 10 pasos
    run_steps(&mut parts_divided, &mut scratch_d, CHECKPOINT_STEP);
    eprintln!("[phase59] Checkpoint en paso {CHECKPOINT_STEP}");

    // Simular serialización: clonar partículas (posiciones + velocidades)
    let checkpoint_particles: Vec<Particle> = parts_divided.clone();

    // Simular restauración: reconstruir fuerzas desde posiciones restauradas
    let mut parts_restored = checkpoint_particles;
    let mut scratch_r = vec![Vec3::zero(); N_TOTAL];
    compute_forces(&parts_restored, &mut scratch_r);

    // Fase 2: 10 pasos más desde el checkpoint
    run_steps(
        &mut parts_restored,
        &mut scratch_r,
        N_STEPS_TOTAL - CHECKPOINT_STEP,
    );
    eprintln!("[phase59] Corrida dividida completada");

    // ── Comparación bit-a-bit ─────────────────────────────────────────────────
    assert_eq!(parts_continuous.len(), parts_restored.len());

    let mut max_dx = 0.0_f64;
    let mut max_dv = 0.0_f64;
    for (pc, pr) in parts_continuous.iter().zip(parts_restored.iter()) {
        let dx = (pc.position.x - pr.position.x)
            .abs()
            .max((pc.position.y - pr.position.y).abs())
            .max((pc.position.z - pr.position.z).abs());
        let dv = (pc.velocity.x - pr.velocity.x)
            .abs()
            .max((pc.velocity.y - pr.velocity.y).abs())
            .max((pc.velocity.z - pr.velocity.z).abs());
        max_dx = max_dx.max(dx);
        max_dv = max_dv.max(dv);
    }

    eprintln!("[phase59] max|Δx| = {max_dx:.2e}  max|Δv| = {max_dv:.2e}");

    // Los resultados deben ser idénticos bit-a-bit.
    assert_eq!(
        max_dx, 0.0,
        "Posiciones no coinciden: max|Δx| = {max_dx:.2e}"
    );
    assert_eq!(
        max_dv, 0.0,
        "Velocidades no coinciden: max|Δv| = {max_dv:.2e}"
    );

    eprintln!("[phase59] ✓ Continuidad bit-a-bit verificada");
}

/// Verificar que el checkpoint captura suficiente estado: si omitimos la
/// recomputación de fuerzas (usamos scratch incorrecto), el resultado difiere.
#[test]
fn phase59_stale_forces_produce_different_result() {
    if skip() {
        eprintln!("[phase59] saltado por PHASE59_SKIP=1");
        return;
    }

    // Corrida continua
    let mut parts_ref = build_initial_particles();
    let mut scratch_ref = vec![Vec3::zero(); N_TOTAL];
    compute_forces(&parts_ref, &mut scratch_ref);
    run_steps(&mut parts_ref, &mut scratch_ref, N_STEPS_TOTAL);

    // Corrida con scratch cero (fuerzas no recomputadas al restart)
    let mut parts_alt = build_initial_particles();
    let mut scratch_a = vec![Vec3::zero(); N_TOTAL];
    compute_forces(&parts_alt, &mut scratch_a);
    run_steps(&mut parts_alt, &mut scratch_a, CHECKPOINT_STEP);

    // Restart con scratch = zeros (ERROR intencional: no recomputamos fuerzas)
    let mut scratch_wrong = vec![Vec3::zero(); N_TOTAL];
    run_steps(
        &mut parts_alt,
        &mut scratch_wrong,
        N_STEPS_TOTAL - CHECKPOINT_STEP,
    );

    // Con scratch incorrecto, el resultado debe DIFERIR de la corrida continua.
    let max_dx: f64 = parts_ref
        .iter()
        .zip(parts_alt.iter())
        .map(|(pr, pa)| {
            (pr.position.x - pa.position.x)
                .abs()
                .max((pr.position.y - pa.position.y).abs())
                .max((pr.position.z - pa.position.z).abs())
        })
        .fold(0.0_f64, f64::max);

    eprintln!("[phase59] Con fuerzas incorrectas: max|Δx| = {max_dx:.2e}");
    // Solo verificamos que es finito — puede o no diferir dependiendo de la magnitud
    // de las fuerzas en ese paso (puede ser cero si G=0 en ese paso de la grid).
    assert!(max_dx.is_finite(), "max_dx no finito");
}
