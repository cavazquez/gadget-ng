//! Tests de validación para condiciones iniciales 2LPT — Fase 28.
//!
//! ## Cobertura
//!
//! 1. `lpt2_reproducible`:
//!    Misma seed → partículas bit-a-bit iguales (reproducibilidad determinista).
//!
//! 2. `lpt2_psi2_mean_near_zero`:
//!    `⟨Ψ²⟩ ≈ 0` — el modo DC del campo de segundo orden es nulo.
//!
//! 3. `lpt2_psi2_smaller_than_psi1`:
//!    `|Ψ²|_rms < |Ψ¹|_rms` — la corrección de segundo orden es subleading.
//!
//! 4. `lpt2_positions_in_box`:
//!    Todas las posiciones 2LPT están en `[0, box_size)` (wrap periódico correcto).
//!
//! 5. `lpt2_no_nan_inf`:
//!    Sin NaN ni Inf en posición ni velocidad de las partículas 2LPT.
//!
//! 6. `lpt2_pm_run_stable`:
//!    10 pasos de PM con ICs 2LPT no produce NaN/Inf (estabilidad numérica).
//!
//! 7. `lpt2_treepm_run_stable`:
//!    10 pasos de TreePM con ICs 2LPT no produce NaN/Inf.
//!
//! 8. `lpt2_vs_1lpt_differ`:
//!    El desplazamiento total con 2LPT difiere del 1LPT (Ψ² ≠ 0).

use gadget_ng_core::{
    build_particles,
    cosmology::CosmologyParams,
    wrap_position, CosmologySection, GravitySection, GravitySolver, IcKind,
    InitialConditionsSection, OutputSection, PerformanceSection, RunConfig, SimulationSection,
    TimestepSection, TransferKind, UnitsSection, Vec3,
};
use gadget_ng_integrators::{leapfrog_cosmo_kdk_step, CosmoFactors};
use gadget_ng_pm::PmSolver;
use gadget_ng_treepm::TreePmSolver;

// ── Constantes ────────────────────────────────────────────────────────────────

const G: f64 = 1.0;
const BOX: f64 = 1.0;
const GRID: usize = 8; // 8³ = 512 partículas (rápido)
const N_PART: usize = 512;
const NM: usize = 8;

const OMEGA_M: f64 = 0.315;
const OMEGA_L: f64 = 0.685;
const H0: f64 = 0.1;
const A_INIT: f64 = 0.02;

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Configuración ΛCDM con ICs 2LPT (Eisenstein–Hu + σ₈).
fn lpt2_config(seed: u64) -> RunConfig {
    RunConfig {
        simulation: SimulationSection {
            dt: 0.002,
            num_steps: 10,
            softening: 0.02,
            gravitational_constant: G,
            particle_count: N_PART,
            box_size: BOX,
            seed,
            integrator: Default::default(),
        },
        initial_conditions: InitialConditionsSection {
            kind: IcKind::Zeldovich {
                seed,
                grid_size: GRID,
                spectral_index: 0.965,
                amplitude: 1.0e-4,
                transfer: TransferKind::EisensteinHu,
                sigma8: Some(0.8),
                omega_b: 0.049,
                h: 0.674,
                t_cmb: 2.7255,
                box_size_mpc_h: Some(100.0),
                use_2lpt: true,
                normalization_mode: gadget_ng_core::NormalizationMode::Legacy,
            },
        },
        output: OutputSection::default(),
        gravity: GravitySection {
            solver: gadget_ng_core::SolverKind::Pm,
            pm_grid_size: NM,
            ..GravitySection::default()
        },
        performance: PerformanceSection::default(),
        timestep: TimestepSection::default(),
        cosmology: CosmologySection {
            enabled: true,
            periodic: true,
            omega_m: OMEGA_M,
            omega_lambda: OMEGA_L,
            h0: H0,
            a_init: A_INIT,
                auto_g: false,
        },
        units: UnitsSection::default(),
        decomposition: Default::default(),
    }
}

/// Configuración 1LPT equivalente (mismos parámetros, `use_2lpt = false`).
fn lpt1_config(seed: u64) -> RunConfig {
    let mut cfg = lpt2_config(seed);
    if let IcKind::Zeldovich { ref mut use_2lpt, .. } = cfg.initial_conditions.kind {
        *use_2lpt = false;
    }
    cfg
}

// ── Test 1: reproducibilidad ──────────────────────────────────────────────────

/// Misma seed → partículas idénticas bit-a-bit.
///
/// Garantía de reproducibilidad determinista independiente del número de rangos MPI.
#[test]
fn lpt2_reproducible() {
    let cfg = lpt2_config(42);
    let parts_a = build_particles(&cfg).expect("2LPT build A");
    let parts_b = build_particles(&cfg).expect("2LPT build B");

    assert_eq!(parts_a.len(), parts_b.len());
    for (a, b) in parts_a.iter().zip(parts_b.iter()) {
        assert_eq!(
            a.position.x.to_bits(),
            b.position.x.to_bits(),
            "x difiere en gid={}", a.global_id
        );
        assert_eq!(
            a.position.y.to_bits(),
            b.position.y.to_bits(),
            "y difiere en gid={}", a.global_id
        );
        assert_eq!(
            a.position.z.to_bits(),
            b.position.z.to_bits(),
            "z difiere en gid={}", a.global_id
        );
    }
}

// ── Test 2: ⟨Ψ²⟩ ≈ 0 ────────────────────────────────────────────────────────

/// El desplazamiento medio de la corrección de segundo orden debe ser ≈ 0.
///
/// Esto es consecuencia de que el modo DC de φ²(k) es forzado a cero.
/// El test compara las posiciones medias de 1LPT vs 2LPT: ambas deben ser
/// indistinguibles (la corrección de segundo orden tiene media nula).
#[test]
fn lpt2_psi2_mean_near_zero() {
    let parts_1lpt = build_particles(&lpt1_config(7)).expect("1LPT build");
    let parts_2lpt = build_particles(&lpt2_config(7)).expect("2LPT build");

    let mean1 = |ps: &[gadget_ng_core::Particle], f: fn(&gadget_ng_core::Particle) -> f64| -> f64 {
        ps.iter().map(f).sum::<f64>() / ps.len() as f64
    };

    // Ψ² = (pos_2lpt − pos_1lpt) en promedio debe ser ≈ 0 (módulo box wrapping)
    // Usamos la diferencia de medias (aproximación válida para desplazamientos pequeños)
    let dx = mean1(&parts_2lpt, |p| p.position.x) - mean1(&parts_1lpt, |p| p.position.x);
    let dy = mean1(&parts_2lpt, |p| p.position.y) - mean1(&parts_1lpt, |p| p.position.y);
    let dz = mean1(&parts_2lpt, |p| p.position.z) - mean1(&parts_1lpt, |p| p.position.z);

    // La tolerancia es 1% del espaciado de la retícula (d = BOX/GRID = 0.125)
    let tol = 0.01 * BOX / GRID as f64;
    assert!(
        dx.abs() < tol,
        "⟨Ψ²_x⟩ = {:.2e} no es ≈ 0 (tol = {:.2e})", dx, tol
    );
    assert!(
        dy.abs() < tol,
        "⟨Ψ²_y⟩ = {:.2e} no es ≈ 0 (tol = {:.2e})", dy, tol
    );
    assert!(
        dz.abs() < tol,
        "⟨Ψ²_z⟩ = {:.2e} no es ≈ 0 (tol = {:.2e})", dz, tol
    );
}

// ── Test 3: |Ψ²|_rms < |Ψ¹|_rms ─────────────────────────────────────────────

/// La corrección de segundo orden es subleading: su RMS es menor que el de 1LPT.
///
/// Esto verifica que la aproximación 2LPT mejora (no domina) sobre Zel'dovich.
#[test]
fn lpt2_psi2_smaller_than_psi1() {
    let parts_1lpt = build_particles(&lpt1_config(13)).expect("1LPT build");
    let parts_2lpt = build_particles(&lpt2_config(13)).expect("2LPT build");

    // Desplazamiento 1LPT desde la retícula perfecta
    let d_grid = BOX / GRID as f64;
    let psi1_rms_sq: f64 = parts_1lpt.iter().enumerate().map(|(gid, p)| {
        let ix = gid / (GRID * GRID);
        let iy = (gid % (GRID * GRID)) / GRID;
        let iz = gid % GRID;
        let qx = (ix as f64 + 0.5) * d_grid;
        let qy = (iy as f64 + 0.5) * d_grid;
        let qz = (iz as f64 + 0.5) * d_grid;
        let dpx = (p.position.x - qx + BOX / 2.0).rem_euclid(BOX) - BOX / 2.0;
        let dpy = (p.position.y - qy + BOX / 2.0).rem_euclid(BOX) - BOX / 2.0;
        let dpz = (p.position.z - qz + BOX / 2.0).rem_euclid(BOX) - BOX / 2.0;
        dpx * dpx + dpy * dpy + dpz * dpz
    }).sum::<f64>() / N_PART as f64;

    // Diferencia entre 2LPT y 1LPT ≈ (D₂/D₁²) Ψ²
    let psi2_rms_sq: f64 = parts_1lpt.iter().zip(parts_2lpt.iter()).map(|(p1, p2)| {
        let dpx = (p2.position.x - p1.position.x + BOX / 2.0).rem_euclid(BOX) - BOX / 2.0;
        let dpy = (p2.position.y - p1.position.y + BOX / 2.0).rem_euclid(BOX) - BOX / 2.0;
        let dpz = (p2.position.z - p1.position.z + BOX / 2.0).rem_euclid(BOX) - BOX / 2.0;
        dpx * dpx + dpy * dpy + dpz * dpz
    }).sum::<f64>() / N_PART as f64;

    let psi1_rms = psi1_rms_sq.sqrt();
    let psi2_rms = psi2_rms_sq.sqrt();

    assert!(
        psi2_rms < psi1_rms,
        "|Ψ²|_rms = {:.4e} ≥ |Ψ¹|_rms = {:.4e} — corrección 2LPT no es subleading",
        psi2_rms, psi1_rms
    );
}

// ── Test 4: posiciones en caja ────────────────────────────────────────────────

/// Todas las posiciones 2LPT están en `[0, box_size)`.
#[test]
fn lpt2_positions_in_box() {
    let parts = build_particles(&lpt2_config(55)).expect("2LPT build");
    assert_eq!(parts.len(), N_PART);

    for p in &parts {
        assert!(
            p.position.x >= 0.0 && p.position.x < BOX,
            "x = {} fuera de [0, {})", p.position.x, BOX
        );
        assert!(
            p.position.y >= 0.0 && p.position.y < BOX,
            "y = {} fuera de [0, {})", p.position.y, BOX
        );
        assert!(
            p.position.z >= 0.0 && p.position.z < BOX,
            "z = {} fuera de [0, {})", p.position.z, BOX
        );
    }
}

// ── Test 5: sin NaN/Inf ───────────────────────────────────────────────────────

/// Sin NaN ni Inf en posición ni velocidad de las partículas 2LPT.
#[test]
fn lpt2_no_nan_inf() {
    let parts = build_particles(&lpt2_config(99)).expect("2LPT build");

    for p in &parts {
        assert!(
            p.position.x.is_finite() && p.position.y.is_finite() && p.position.z.is_finite(),
            "Posición NaN/Inf en gid={}: {:?}", p.global_id, p.position
        );
        assert!(
            p.velocity.x.is_finite() && p.velocity.y.is_finite() && p.velocity.z.is_finite(),
            "Velocidad NaN/Inf en gid={}: {:?}", p.global_id, p.velocity
        );
    }
}

// ── Test 6: PM estable con ICs 2LPT ──────────────────────────────────────────

/// 10 pasos de leapfrog cosmológico con PM y ICs 2LPT no produce NaN/Inf.
#[test]
fn lpt2_pm_run_stable() {
    let cfg = lpt2_config(888);
    let mut parts = build_particles(&cfg).expect("2LPT build");
    let cosmo = CosmologyParams::new(OMEGA_M, OMEGA_L, H0);
    let mut a = cfg.cosmology.a_init;
    let dt = cfg.simulation.dt;

    let pm = PmSolver { grid_size: NM, box_size: BOX };
    let mut scratch = vec![Vec3::zero(); N_PART];

    for _ in 0..10 {
        let g_cosmo = G / a;
        let (drift, kick_half, kick_half2) = cosmo.drift_kick_factors(a, dt);
        let cf = CosmoFactors { drift, kick_half, kick_half2 };
        a = cosmo.advance_a(a, dt);

        leapfrog_cosmo_kdk_step(&mut parts, cf, &mut scratch, |ps, acc| {
            let pos: Vec<Vec3> = ps.iter().map(|p| p.position).collect();
            let m: Vec<f64> = ps.iter().map(|p| p.mass).collect();
            let idx: Vec<usize> = (0..ps.len()).collect();
            pm.accelerations_for_indices(&pos, &m, 0.0, g_cosmo, &idx, acc);
        });

        for p in parts.iter_mut() {
            p.position = wrap_position(p.position, BOX);
        }
    }

    for p in &parts {
        assert!(
            p.position.x.is_finite() && p.position.y.is_finite() && p.position.z.is_finite(),
            "Posición NaN/Inf PM+2LPT gid={}: {:?}", p.global_id, p.position
        );
        assert!(
            p.velocity.x.is_finite() && p.velocity.y.is_finite() && p.velocity.z.is_finite(),
            "Velocidad NaN/Inf PM+2LPT gid={}: {:?}", p.global_id, p.velocity
        );
    }
}

// ── Test 7: TreePM estable con ICs 2LPT ──────────────────────────────────────

/// 10 pasos de leapfrog cosmológico con TreePM y ICs 2LPT no produce NaN/Inf.
#[test]
fn lpt2_treepm_run_stable() {
    let cfg = {
        let mut c = lpt2_config(999);
        c.gravity.solver = gadget_ng_core::SolverKind::TreePm;
        c.gravity.theta = 0.5;
        c
    };

    let mut parts = build_particles(&cfg).expect("2LPT build");
    let cosmo = CosmologyParams::new(OMEGA_M, OMEGA_L, H0);
    let mut a = cfg.cosmology.a_init;
    let dt = cfg.simulation.dt;

    let treepm = TreePmSolver {
        grid_size: NM,
        box_size: BOX,
        r_split: 0.0,
    };
    let mut scratch = vec![Vec3::zero(); N_PART];

    for _ in 0..10 {
        let g_cosmo = G / a;
        let (drift, kick_half, kick_half2) = cosmo.drift_kick_factors(a, dt);
        let cf = CosmoFactors { drift, kick_half, kick_half2 };
        a = cosmo.advance_a(a, dt);

        leapfrog_cosmo_kdk_step(&mut parts, cf, &mut scratch, |ps, acc| {
            let pos: Vec<Vec3> = ps.iter().map(|p| p.position).collect();
            let m: Vec<f64> = ps.iter().map(|p| p.mass).collect();
            let idx: Vec<usize> = (0..ps.len()).collect();
            treepm.accelerations_for_indices(&pos, &m, 0.0, g_cosmo, &idx, acc);
        });

        for p in parts.iter_mut() {
            p.position = wrap_position(p.position, BOX);
        }
    }

    for p in &parts {
        assert!(
            p.position.x.is_finite() && p.position.y.is_finite() && p.position.z.is_finite(),
            "Posición NaN/Inf TreePM+2LPT gid={}: {:?}", p.global_id, p.position
        );
        assert!(
            p.velocity.x.is_finite() && p.velocity.y.is_finite() && p.velocity.z.is_finite(),
            "Velocidad NaN/Inf TreePM+2LPT gid={}: {:?}", p.global_id, p.velocity
        );
    }
}

// ── Test 8: 2LPT difiere de 1LPT ─────────────────────────────────────────────

/// El desplazamiento total 2LPT difiere del 1LPT — confirma que Ψ² ≠ 0.
///
/// Con la corrección de segundo orden activa (D₂/D₁² ≈ −0.43), al menos
/// la mayoría de las partículas debe tener posición o velocidad diferente a
/// la 1LPT a nivel de bits floating-point.
///
/// Para una rejilla 8³ la corrección 2LPT (cuadrática en el desplazamiento)
/// es pequeña (~3×10⁻⁵ en unidades de caja) pero claramente no nula. Por
/// eso se usa comparación bit-a-bit: si Ψ² = 0, todas las posiciones serían
/// idénticas. Con Ψ² ≠ 0, virtualmente todos los bits deben diferir dado que
/// la corrección (~3×10⁻⁵) es mucho mayor que la precisión de máquina (2×10⁻¹⁶).
#[test]
fn lpt2_vs_1lpt_differ() {
    let parts_1lpt = build_particles(&lpt1_config(77)).expect("1LPT build");
    let parts_2lpt = build_particles(&lpt2_config(77)).expect("2LPT build");

    assert_eq!(parts_1lpt.len(), parts_2lpt.len());

    // Comparación bit-a-bit: detecta cualquier diferencia floating-point.
    // La corrección 2LPT (~3×10⁻⁵) >> epsilon de máquina (2×10⁻¹⁶).
    let differ_count = parts_1lpt.iter().zip(parts_2lpt.iter()).filter(|(p1, p2)| {
        p1.position.x.to_bits() != p2.position.x.to_bits()
            || p1.position.y.to_bits() != p2.position.y.to_bits()
            || p1.position.z.to_bits() != p2.position.z.to_bits()
    }).count();

    assert!(
        differ_count > N_PART / 2,
        "{}/{} partículas difieren entre 1LPT y 2LPT (esperado > {})\n\
        Si differ_count = 0: Ψ² es cero (bug en 2LPT).\n\
        Si differ_count pequeño: la corrección tiene magnitud esperada pero la rejilla es insuficiente.",
        differ_count, N_PART, N_PART / 2
    );
}
