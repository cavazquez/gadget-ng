//! Phase 51 — G auto-consistente integrado en el motor cosmológico.
//!
//! ## Motivación
//!
//! Phase 50 introdujo `g_code_consistent()` como función standalone.
//! Phase 51 integra esa lógica directamente en `RunConfig`:
//!
//! - `CosmologySection::auto_g = true` → `effective_g()` calcula G desde Friedmann.
//! - `RunConfig::cosmo_g_diagnostic()` → error relativo para diagnóstico.
//! - El motor CLI emite `warn!` si G manual difiere > 1 % del valor consistente.
//!
//! ## Tests
//!
//! 1. **`phase51_auto_g_effective_g`** — Verifica que con `auto_g=true`,
//!    `effective_g()` devuelve `g_code_consistent(omega_m, h0)` exactamente.
//!
//! 2. **`phase51_legacy_g_diagnostic`** — Con `auto_g=false` y G=1 (legacy),
//!    `cosmo_g_diagnostic()` devuelve error > 2000 %.
//!
//! 3. **`phase51_consistent_g_no_warning`** — Con G ya consistente manualmente
//!    (`G = g_code_consistent(om, h0)`), `cosmo_g_diagnostic()` da error < 1e-10.
//!
//! 4. **`phase51_units_priority`** — Cuando `units.enabled=true` y `auto_g=true`,
//!    `effective_g()` prioriza `UnitsSection::compute_g()`.
//!
//! 5. **`phase51_auto_g_simulation_stable`** — Simulación corta (N=8, a=0.02→0.04)
//!    con `auto_g=true`. Verifica que la simulación completa sin explosión.

use gadget_ng_core::{
    build_particles, cosmo_consistency_error,
    cosmology::{gravity_coupling_qksl, CosmologyParams},
    g_code_consistent, wrap_position, CosmologySection, GravitySection, GravitySolver, IcKind,
    InitialConditionsSection, NormalizationMode, OutputSection, PerformanceSection, RunConfig,
    SimulationSection, TimestepSection, TransferKind, UnitsSection, Vec3,
};
use gadget_ng_integrators::{leapfrog_cosmo_kdk_step, CosmoFactors};
use gadget_ng_pm::PmSolver;

// ── Constantes ────────────────────────────────────────────────────────────────

const OMEGA_M: f64 = 0.315;
const OMEGA_L: f64 = 0.685;
const H0: f64 = 0.1;
const A_INIT: f64 = 0.02;
const BOX: f64 = 1.0;

fn base_cfg(g_override: f64, auto_g: bool) -> RunConfig {
    RunConfig {
        simulation: SimulationSection {
            dt: 1e-3,
            num_steps: 1,
            softening: 1.0 / (8.0 * 20.0),
            gravitational_constant: g_override,
            particle_count: 8 * 8 * 8,
            box_size: BOX,
            seed: 42,
            integrator: Default::default(),
        },
        initial_conditions: InitialConditionsSection {
            kind: IcKind::Zeldovich {
                seed: 42,
                grid_size: 8,
                spectral_index: 0.965,
                amplitude: 1e-4,
                transfer: TransferKind::EisensteinHu,
                sigma8: Some(0.8),
                omega_b: 0.049,
                h: 0.674,
                t_cmb: 2.7255,
                box_size_mpc_h: Some(100.0),
                use_2lpt: false,
                normalization_mode: NormalizationMode::Z0Sigma8,
            },
        },
        output: OutputSection::default(),
        gravity: GravitySection {
            solver: gadget_ng_core::SolverKind::Pm,
            pm_grid_size: 8,
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
            auto_g,
        },
        units: UnitsSection::default(),
        decomposition: Default::default(),
    }
}

// ── TEST 1 — effective_g con auto_g=true ──────────────────────────────────────

/// Con `auto_g=true`, `effective_g()` debe devolver `g_code_consistent(omega_m, h0)`.
#[test]
fn phase51_auto_g_effective_g() {
    let cfg = base_cfg(1.0, true); // G=1 manual, pero auto_g lo ignora
    let g_effective = cfg.effective_g();
    let g_expected = g_code_consistent(OMEGA_M, H0);

    let err = (g_effective - g_expected).abs() / g_expected;
    println!(
        "[phase51_auto_g] effective_g={g_effective:.6e}  g_code_consistent={g_expected:.6e}  err={err:.2e}"
    );

    assert!(
        err < 1e-12,
        "auto_g=true debe producir G=g_code_consistent exacto: err={err:.3e}"
    );

    // Verificar que satisface Friedmann.
    let friedmann_err = cosmo_consistency_error(g_effective, OMEGA_M, H0, 1.0);
    assert!(
        friedmann_err < 1e-12,
        "G de auto_g no satisface Friedmann: err={friedmann_err:.3e}"
    );
}

// ── TEST 2 — Diagnóstico con G legacy ────────────────────────────────────────

/// Con `auto_g=false` y G=1.0, `cosmo_g_diagnostic()` debe reportar error > 2000%.
#[test]
fn phase51_legacy_g_diagnostic() {
    let cfg = base_cfg(1.0, false); // G=1 legacy, auto_g desactivado
    let (g_consistent, rel_err) = cfg
        .cosmo_g_diagnostic()
        .expect("cosmo_g_diagnostic debe devolver Some cuando cosmology.enabled=true");

    println!(
        "[phase51_legacy] G_used=1.0  G_consistent={g_consistent:.4e}  err={:.1}%",
        rel_err * 100.0
    );

    assert!(
        rel_err > 20.0, // >2000%
        "G=1 legacy debería dar error >2000% pero dio {:.1}%",
        rel_err * 100.0
    );

    // La advertencia se emite en el motor (no testeable aquí), pero podemos
    // verificar que el umbral del 1% se supera claramente.
    assert!(
        rel_err > 0.01,
        "Error debería superar umbral del 1%: rel_err={rel_err:.3e}"
    );
}

// ── TEST 3 — G manual ya consistente → sin warning ───────────────────────────

/// Con G manualmente puesto al valor consistente, el error debe ser < 1e-10.
#[test]
fn phase51_consistent_g_no_warning() {
    let g_correct = g_code_consistent(OMEGA_M, H0);
    let cfg = base_cfg(g_correct, false); // G correcto, auto_g desactivado

    let (g_consistent, rel_err) = cfg
        .cosmo_g_diagnostic()
        .expect("cosmo_g_diagnostic debe devolver Some");

    println!(
        "[phase51_consistent] G_used={g_correct:.6e}  G_consistent={g_consistent:.6e}  err={rel_err:.2e}"
    );

    // Error < 1e-10: no se emitiría warning.
    assert!(
        rel_err < 1e-10,
        "G manual consistente debe dar error < 1e-10: {rel_err:.3e}"
    );
    assert!(
        rel_err < 0.01,
        "Error debe estar bajo el umbral del 1%: {:.2}%",
        rel_err * 100.0
    );
}

// ── TEST 4 — Prioridad UnitsSection sobre auto_g ──────────────────────────────

/// Cuando `units.enabled=true` y `auto_g=true`, `effective_g()` debe usar
/// `UnitsSection::compute_g()` (prioridad mayor).
#[test]
fn phase51_units_priority() {
    let mut cfg = base_cfg(1.0, true);
    // Configurar UnitsSection con valores GADGET clásicos.
    cfg.units = gadget_ng_core::UnitsSection {
        enabled: true,
        length_in_kpc: 1.0,
        mass_in_msun: 1.0e10,
        velocity_in_km_s: 1.0,
    };
    // G_int = G_KPC_MSUN_KMPS × 1e10 / 1 / 1 = 4.3009e-6 × 1e10 = 4.3009e4
    let g_units = cfg.units.compute_g();
    let g_effective = cfg.effective_g();

    println!(
        "[phase51_units_priority] G_units={g_units:.4e}  G_effective={g_effective:.4e}  \
         G_auto_g={:.4e}",
        g_code_consistent(OMEGA_M, H0)
    );

    assert!(
        (g_effective - g_units).abs() / g_units < 1e-12,
        "units.enabled=true debe tomar prioridad sobre auto_g: \
         g_effective={g_effective:.4e} != g_units={g_units:.4e}"
    );
}

// ── TEST 5 — Simulación estable con auto_g=true ───────────────────────────────

/// N=8, a=0.02→0.04, con `auto_g=true`.
///
/// Verifica que la simulación con G auto-consistente arranca, avanza el factor
/// de escala correctamente y no produce NaN/Inf.
#[test]
fn phase51_auto_g_simulation_stable() {
    let cfg = base_cfg(1.0, true); // G=1 manual ignorado; auto_g calcula G_cons
    let g = cfg.effective_g(); // debe ser g_code_consistent(OMEGA_M, H0)

    let mut parts = build_particles(&cfg).expect("ICs");
    let c = CosmologyParams::new(OMEGA_M, OMEGA_L, H0);
    let pm = PmSolver {
        grid_size: 8,
        box_size: BOX,
    };
    let softening = cfg.simulation.softening;
    let mut scratch = vec![Vec3::zero(); parts.len()];

    let mut a = A_INIT;
    let a_target = 0.04_f64;
    let dt = 1.0e-3;

    for _ in 0..200 {
        if a >= a_target {
            break;
        }
        let g_cosmo = gravity_coupling_qksl(g, a);
        let (drift, kh, kh2) = c.drift_kick_factors(a, dt);
        let cf = CosmoFactors {
            drift,
            kick_half: kh,
            kick_half2: kh2,
        };
        a = c.advance_a(a, dt);
        leapfrog_cosmo_kdk_step(&mut parts, cf, &mut scratch, |ps, out| {
            let pos: Vec<Vec3> = ps.iter().map(|p| p.position).collect();
            let m: Vec<f64> = ps.iter().map(|p| p.mass).collect();
            let idx: Vec<usize> = (0..ps.len()).collect();
            pm.accelerations_for_indices(&pos, &m, softening, g_cosmo, &idx, out);
        });
        for p in parts.iter_mut() {
            p.position = wrap_position(p.position, BOX);
        }
    }

    let v_rms: f64 = {
        let s: f64 = parts
            .iter()
            .map(|p| p.velocity.norm() * p.velocity.norm())
            .sum();
        (s / parts.len() as f64).sqrt()
    };

    println!("[phase51_sim] G_auto={g:.4e}  a_final={a:.4}  v_rms={v_rms:.4e}");

    assert!(a >= a_target * 0.99, "No alcanzó a_target");
    assert!(
        v_rms.is_finite() && !v_rms.is_nan(),
        "v_rms no finito → explosión"
    );
    assert!(v_rms < 1e3, "v_rms excesivo → inestabilidad: {v_rms:.3e}");

    // Confirmar que el G usado es el consistente, no 1.0.
    let g_expected = g_code_consistent(OMEGA_M, H0);
    assert!(
        (g - g_expected).abs() / g_expected < 1e-12,
        "G de la simulación debería ser g_code_consistent: {g:.4e} != {g_expected:.4e}"
    );

    println!("  ✓ auto_g=true: G={g:.4e} (Friedmann), simulación estable.");
}
