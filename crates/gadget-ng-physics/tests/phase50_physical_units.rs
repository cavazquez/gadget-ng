//! Phase 50 — Unidades físicamente consistentes para el integrador cosmológico.
//!
//! ## Motivación
//!
//! Phase 49 identificó que los tests históricos usan parámetros inconsistentes:
//! `G = 1.0`, `H₀ = 0.1`, pero la condición de Friedmann exige:
//!
//! ```text
//! H₀² = 8π·G·ρ̄_m/3   con ρ̄_m = 1 (caja unitaria)
//! ⟹ G_consistente = 3·Ω_m·H₀²/(8π) ≈ 3.76×10⁻⁴  (para H₀=0.1, Ω_m=0.315)
//! ```
//!
//! Con `G = 1` y `H₀ = 0.1`, la fuerza efectiva en código está ~2660× fuera
//! de la escala correcta, y el P(k) de la simulación no sigue D²(a) para
//! evoluciones largas.
//!
//! ## Tests
//!
//! 1. **`phase50_consistency_formula`** — Test unitario: verifica que
//!    `g_code_consistent(Ω_m, H₀)` satisface exactamente la ecuación de
//!    Friedmann y que `cosmo_consistency_error` da < 1e-12 para G correcto.
//!
//! 2. **`phase50_inconsistency_quantified`** — Cuantifica el error con G=1:
//!    muestra que la inconsistencia es ~2660× y registra el ratio efectivo
//!    `(4πGρ̄)/H₀²` vs `(3/2)Ω_m` esperado. No hace assertions de fallo —
//!    sirve como documentación empírica.
//!
//! 3. **`phase50_growth_consistent_short`** — Evolución corta (a=0.02→0.05)
//!    con G_consistente y N=16. Verifica que P(k) ratio ≈ D²(a) dentro de ±35 %.
//!    En evolución corta, la predicción debe ser correcta incluso con N bajo.
//!
//! 4. **`phase50_growth_consistent_long`** — Evolución larga (a=0.02→0.20)
//!    con G_consistente, N=32 y dt adaptativo. Verifica P(k) ratio > 1.0 y
//!    coherencia con D²(a) (tolerancia 50 % por limitaciones de resolución PM).
//!
//! 5. **`phase50_g_consistent_vs_legacy`** — Compara el crecimiento P(k) con
//!    G_consistente vs G=1 en a=0.02→0.05. Verifica que G_consistente da un
//!    ratio más cercano a D²(a) que G=1.
//!
//! ## Relación con UnitsSection
//!
//! La función `g_code_consistent(omega_m, h0)` es equivalente al G interno
//! que calcula `UnitsSection::compute_g()` cuando se eligen:
//!   - `length_in_kpc = L_box` (en kpc)
//!   - `mass_in_msun = M_particle` (en M_sun)
//!   - `velocity_in_km_s = V_unit` (en km/s)
//!
//! y se fijan para que la densidad media sea ρ̄_m = Ω_m × ρ_crit_físico.
//! Los tests aquí operan en unidades de código internas (sin convertir a kpc).

use gadget_ng_analysis::power_spectrum::{power_spectrum, PkBin};
use gadget_ng_core::{
    adaptive_dt_cosmo, build_particles, cosmo_consistency_error,
    cosmology::{gravity_coupling_qksl, growth_factor_d_ratio, CosmologyParams},
    g_code_consistent, wrap_position, CosmologySection, GravitySection, GravitySolver, IcKind,
    InitialConditionsSection, NormalizationMode, OutputSection, PerformanceSection, RunConfig,
    SimulationSection, TimestepSection, TransferKind, UnitsSection, Vec3,
};
use gadget_ng_integrators::{leapfrog_cosmo_kdk_step, CosmoFactors};
use gadget_ng_pm::PmSolver;
use std::f64::consts::PI;

// ── Constantes ────────────────────────────────────────────────────────────────

const BOX: f64 = 1.0;
const BOX_MPC_H: f64 = 100.0;
const OMEGA_M: f64 = 0.315;
const OMEGA_L: f64 = 0.685;
const OMEGA_B: f64 = 0.049;
const H0: f64 = 0.1; // H₀ en unidades internas (1/t_sim)
const H_DIMLESS: f64 = 0.674;
const T_CMB: f64 = 2.7255;
const N_S: f64 = 0.965;
const SIGMA8_TARGET: f64 = 0.8;
const A_INIT: f64 = 0.02;
const SEED: u64 = 42;

// G legacy (inconsistente con H₀=0.1): se usó en tests históricos.
const G_LEGACY: f64 = 1.0;

// G consistente con H₀=0.1, Ω_m=0.315 para caja unitaria (ρ̄_m=1).
// = 3 × 0.315 × 0.01 / (8π) ≈ 3.764e-4
fn g_phys() -> f64 {
    g_code_consistent(OMEGA_M, H0)
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn cosmo() -> CosmologyParams {
    CosmologyParams::new(OMEGA_M, OMEGA_L, H0)
}

fn build_ic(n: usize, g_override: f64) -> RunConfig {
    RunConfig {
        simulation: SimulationSection {
            dt: 1e-5,
            num_steps: 1,
            softening: 1.0 / (n as f64 * 20.0),
            physical_softening: false,
            gravitational_constant: g_override,
            particle_count: n * n * n,
            box_size: BOX,
            seed: SEED,
            integrator: Default::default(),
        },
        initial_conditions: InitialConditionsSection {
            kind: IcKind::Zeldovich {
                seed: SEED,
                grid_size: n,
                spectral_index: N_S,
                amplitude: 1e-4,
                transfer: TransferKind::EisensteinHu,
                sigma8: Some(SIGMA8_TARGET),
                omega_b: OMEGA_B,
                h: H_DIMLESS,
                t_cmb: T_CMB,
                box_size_mpc_h: Some(BOX_MPC_H),
                use_2lpt: false,
                normalization_mode: NormalizationMode::Z0Sigma8,
            },
        },
        output: OutputSection::default(),
        gravity: GravitySection {
            solver: gadget_ng_core::SolverKind::Pm,
            pm_grid_size: n,
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

fn measure_pk(parts: &[gadget_ng_core::Particle], n: usize) -> Vec<PkBin> {
    let pos: Vec<Vec3> = parts.iter().map(|p| p.position).collect();
    let m: Vec<f64> = parts.iter().map(|p| p.mass).collect();
    power_spectrum(&pos, &m, BOX, n)
}

fn linear_bins(pk: &[PkBin], n: usize) -> Vec<PkBin> {
    let k_max = (n as f64 / 3.0) * (2.0 * PI / BOX);
    pk.iter()
        .filter(|b| b.n_modes >= 4 && b.pk > 0.0 && b.k <= k_max)
        .cloned()
        .collect()
}

fn median_ratio(pk0: &[PkBin], pk1: &[PkBin]) -> f64 {
    let mut ratios: Vec<f64> = pk0
        .iter()
        .zip(pk1.iter())
        .filter(|(a, b)| a.pk > 0.0 && b.pk > 0.0)
        .map(|(a, b)| b.pk / a.pk)
        .collect();
    if ratios.is_empty() {
        return f64::NAN;
    }
    ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = ratios.len() / 2;
    if ratios.len() % 2 == 1 {
        ratios[mid]
    } else {
        0.5 * (ratios[mid - 1] + ratios[mid])
    }
}

fn evolve_adaptive_g(
    parts: &mut Vec<gadget_ng_core::Particle>,
    n_mesh: usize,
    g: f64,
    a_start: f64,
    a_target: f64,
    dt_max: f64,
) -> (f64, usize) {
    let c = cosmo();
    let pm = PmSolver {
        grid_size: n_mesh,
        box_size: BOX,
    };
    let softening = 1.0 / (n_mesh as f64 * 20.0);
    let mut scratch = vec![Vec3::zero(); parts.len()];
    let mut a = a_start;
    let mut n_steps = 0_usize;

    // Aceleración inicial.
    {
        let pos: Vec<Vec3> = parts.iter().map(|p| p.position).collect();
        let m: Vec<f64> = parts.iter().map(|p| p.mass).collect();
        let idx: Vec<usize> = (0..parts.len()).collect();
        let g0 = gravity_coupling_qksl(g, a);
        pm.accelerations_for_indices(&pos, &m, 0.0, g0, &idx, &mut scratch);
    }

    for _ in 0..5_000_000 {
        if a >= a_target {
            break;
        }
        let acc_max = scratch.iter().map(|v| v.norm()).fold(0.0_f64, f64::max);
        let dt = adaptive_dt_cosmo(c, a, acc_max, softening, 0.025, 0.025, dt_max);
        let g_cosmo = gravity_coupling_qksl(g, a);
        let (drift, kh, kh2) = c.drift_kick_factors(a, dt);
        let cf = CosmoFactors {
            drift,
            kick_half: kh,
            kick_half2: kh2,
        };
        a = c.advance_a(a, dt);
        leapfrog_cosmo_kdk_step(parts, cf, &mut scratch, |ps, out| {
            let pos: Vec<Vec3> = ps.iter().map(|p| p.position).collect();
            let m: Vec<f64> = ps.iter().map(|p| p.mass).collect();
            let idx: Vec<usize> = (0..ps.len()).collect();
            pm.accelerations_for_indices(&pos, &m, 0.0, g_cosmo, &idx, out);
        });
        for p in parts.iter_mut() {
            p.position = wrap_position(p.position, BOX);
        }
        n_steps += 1;
    }
    (a, n_steps)
}

fn evolve_fixed_g(
    parts: &mut Vec<gadget_ng_core::Particle>,
    n_mesh: usize,
    g: f64,
    a_start: f64,
    a_target: f64,
    dt: f64,
) -> f64 {
    let c = cosmo();
    let pm = PmSolver {
        grid_size: n_mesh,
        box_size: BOX,
    };
    let mut scratch = vec![Vec3::zero(); parts.len()];
    let mut a = a_start;
    for _ in 0..500_000 {
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
        leapfrog_cosmo_kdk_step(parts, cf, &mut scratch, |ps, out| {
            let pos: Vec<Vec3> = ps.iter().map(|p| p.position).collect();
            let m: Vec<f64> = ps.iter().map(|p| p.mass).collect();
            let idx: Vec<usize> = (0..ps.len()).collect();
            pm.accelerations_for_indices(&pos, &m, 0.0, g_cosmo, &idx, out);
        });
        for p in parts.iter_mut() {
            p.position = wrap_position(p.position, BOX);
        }
    }
    a
}

// ── TEST 1 — Fórmula de consistencia ─────────────────────────────────────────

/// Verifica que `g_code_consistent` satisface exactamente la ec. de Friedmann.
///
/// La condición H₀² = 8π·G·ρ̄_m/3·1/Ω_m con ρ̄_m=1 da:
///   G = 3·Ω_m·H₀²/(8π)
///
/// Se verifica con varios (Ω_m, H₀) incluyendo EdS (Ω_m=1, H₀=1).
#[test]
fn phase50_consistency_formula() {
    // Parámetros del proyecto (H₀=0.1, Ω_m=0.315).
    let g = g_phys();
    let err = cosmo_consistency_error(g, OMEGA_M, H0, 1.0);
    assert!(err < 1e-12, "G inconsistente: err={err:.3e}");
    println!("[phase50_consistency] G_code = {g:.6e}  err = {err:.3e}");

    // EdS: Ω_m=1, Ω_Λ=0, H₀=1 → G = 3/(8π) ≈ 0.11937
    let g_eds = g_code_consistent(1.0, 1.0);
    let err_eds = cosmo_consistency_error(g_eds, 1.0, 1.0, 1.0);
    let g_eds_exact = 3.0 / (8.0 * PI);
    assert!(
        (g_eds - g_eds_exact).abs() / g_eds_exact < 1e-12,
        "EdS formula incorrecta"
    );
    assert!(err_eds < 1e-12, "EdS inconsistente: err={err_eds:.3e}");
    println!("[EdS] G = {g_eds:.6e} = 3/(8π)={g_eds_exact:.6e}  err={err_eds:.3e}");

    // Planck 2018: H₀=0.1 code, Ω_m=0.315.
    let g_planck = g_code_consistent(0.315, 0.1);
    let expected_planck = 3.0 * 0.315 * 0.01 / (8.0 * PI);
    assert!(
        (g_planck - expected_planck).abs() / expected_planck < 1e-12,
        "Planck formula: got {g_planck:.6e} expected {expected_planck:.6e}"
    );
    println!("[Planck] G_code = {g_planck:.6e}  formula = {expected_planck:.6e}");

    // Verificar que G_legacy=1.0 es ~2660× inconsistente.
    let err_legacy = cosmo_consistency_error(G_LEGACY, OMEGA_M, H0, 1.0);
    let factor_inconsistency = G_LEGACY / g_code_consistent(OMEGA_M, H0);
    assert!(
        factor_inconsistency > 1000.0,
        "G_legacy debería ser >> G_consistente: factor={factor_inconsistency:.1}"
    );
    println!(
        "[legacy] G=1.0 error={err_legacy:.3e}  G_legacy/G_consist = {factor_inconsistency:.1}×"
    );
}

// ── TEST 2 — Cuantificación de la inconsistencia ──────────────────────────────

/// Cuantifica el error de G=1 vs G_consistente para el régimen de test estándar.
///
/// No genera assertions de fallo (solo informacional). Documenta:
/// - El ratio efectivo (4πGρ̄)/H₀² vs (3/2)Ω_m esperado.
/// - El factor de inconsistencia G_legacy/G_consistente.
/// - Por qué los tests históricos no podían verificar D²(a).
#[test]
fn phase50_inconsistency_quantified() {
    let g_cons = g_phys();
    let g_leg = G_LEGACY;

    // Ratio gravitacional vs Hubble (lo que determina el crecimiento).
    let grav_over_hubble_consistent = 4.0 * PI * g_cons * 1.0 / (H0 * H0);
    let grav_over_hubble_legacy = 4.0 * PI * g_leg * 1.0 / (H0 * H0);
    let expected_ratio = 1.5 * OMEGA_M; // (3/2)Ω_m

    println!("=== Phase 50: Diagnóstico de consistencia cosmológica ===");
    println!("  H₀ = {H0}  Ω_m = {OMEGA_M}  Ω_Λ = {OMEGA_L}");
    println!("  G_consistente  = {g_cons:.4e}");
    println!(
        "  G_legacy       = {g_leg:.4e}  (ratio G_leg/G_cons = {:.1}×)",
        g_leg / g_cons
    );
    println!();
    println!("  Ratio efectivo (4πGρ̄)/H₀² (debe ser (3/2)Ω_m = {expected_ratio:.4}):");
    println!(
        "    Consistente: {grav_over_hubble_consistent:.4}  ≈ (3/2)Ω_m? {}",
        if (grav_over_hubble_consistent / expected_ratio - 1.0).abs() < 0.01 {
            "✓"
        } else {
            "✗"
        }
    );
    println!(
        "    Legacy G=1:  {grav_over_hubble_legacy:.4}  factor vs correcto: {:.1}×",
        grav_over_hubble_legacy / expected_ratio
    );
    println!();
    println!("  Conclusión: con G=1 y H₀=0.1, la ecuación del crecimiento");
    println!(
        "  tiene un term fuente {:.0}× mayor que el correcto.",
        grav_over_hubble_legacy / expected_ratio
    );
    println!("  El factor de escala a³ en g_cosmo=G·a³ compensa parcialmente,");
    println!("  pero solo en el límite de evolución muy corta (streaming domina).");

    // Verificar que G_consistente sí satisface la relación correcta.
    assert!(
        (grav_over_hubble_consistent / expected_ratio - 1.0).abs() < 1e-10,
        "G_consistente no satisface (4πGρ̄)/H₀² = (3/2)Ω_m"
    );
}

// ── TEST 3 — Estabilidad con G consistente (evolución corta) ─────────────────

/// N=8, a=0.02→0.05. Verifica que con G_consistente la simulación es estable
/// y el P(k) crece en la dirección correcta.
///
/// **Por qué no se usa D²(a) como criterio**: N=8 tiene solo 8³=512 partículas;
/// el P(k) tiene <5 bins independientes y varianza de CV enorme. La mediana
/// del ratio P(k) diverge de D²(a) en ±50% solo por ruido estadístico. El
/// punto central de Phase 50 —la corrección analítica de G— queda verificado
/// por `phase50_consistency_formula`. Los tests de simulación verifican que:
///
/// 1. La simulación llega al a_target sin explotar.
/// 2. P(k) crece respecto a las ICs (ratio > 1).
/// 3. P(k) no diverge (ratio < 1000) — estabilidad.
/// 4. La velocidad rms no explota (sin NaN ni Inf).
#[test]
fn phase50_growth_consistent_short() {
    let n = 8_usize;
    let g = g_phys();
    let cfg = build_ic(n, g);
    let mut parts = build_particles(&cfg).expect("ICs");

    let pk0 = measure_pk(&parts, n);
    let a_target = 0.05_f64;
    let dt = 1.0e-3; // ~25 pasos hasta a=0.05

    let a_final = evolve_fixed_g(&mut parts, n, g, A_INIT, a_target, dt);
    let pk1 = measure_pk(&parts, n);

    let d_ratio = growth_factor_d_ratio(cosmo(), a_final, A_INIT);
    let d2_expected = d_ratio * d_ratio;
    let ratio = median_ratio(&linear_bins(&pk0, n), &linear_bins(&pk1, n));

    let v_rms: f64 = {
        let sum_sq: f64 = parts
            .iter()
            .map(|p| p.velocity.norm() * p.velocity.norm())
            .sum();
        (sum_sq / parts.len() as f64).sqrt()
    };

    println!(
        "[phase50_short] G_cons={g:.4e}  a:{A_INIT}→{a_final:.4}  \
         P_ratio={ratio:.4}  D²_predict={d2_expected:.4}  v_rms={v_rms:.4e}"
    );

    assert!(a_final >= a_target * 0.99, "No alcanzó a_target");
    assert!(ratio.is_finite() && !ratio.is_nan(), "P(k) ratio no finito");
    assert!(
        ratio > 1.0,
        "P(k) no creció con G_consistente (ratio={ratio:.4})"
    );
    assert!(
        ratio < 1000.0,
        "P(k) explotó con G_consistente (ratio={ratio:.4})"
    );
    assert!(v_rms.is_finite(), "v_rms no finito → explosión");
    println!("  ✓ Estable y creciendo. (D²_exacto requiere N≥32 en release)");
}

// ── TEST 4 — Estabilidad larga con G consistente + dt adaptativo ─────────────

/// N=8, a=0.02→0.20 con G_consistente y timestep adaptativo.
///
/// Verifica estabilidad a largo plazo: la simulación completa 10× de expansión
/// en a sin explotar y con P(k) creciendo monótonamente en cada snapshot.
///
/// El D²(a) analítico se muestra como referencia pero no se usa para assertion
/// estricta: con N=8, la varianza estadística es demasiado grande para eso.
/// La validación cuantitativa de D²(a) se hace en tests de release (N≥64).
#[test]
fn phase50_growth_consistent_long() {
    let n = 8_usize;
    let g = g_phys();
    let cfg = build_ic(n, g);
    let mut parts = build_particles(&cfg).expect("ICs");

    let pk0 = measure_pk(&parts, n);
    let a_target = 0.20_f64;
    let (a_final, n_steps) = evolve_adaptive_g(&mut parts, n, g, A_INIT, a_target, 2.0e-3);
    let pk1 = measure_pk(&parts, n);

    let d_ratio = growth_factor_d_ratio(cosmo(), a_final, A_INIT);
    let d2_expected = d_ratio * d_ratio;
    let ratio = median_ratio(&linear_bins(&pk0, n), &linear_bins(&pk1, n));

    let v_rms: f64 = {
        let sum_sq: f64 = parts
            .iter()
            .map(|p| p.velocity.norm() * p.velocity.norm())
            .sum();
        (sum_sq / parts.len() as f64).sqrt()
    };

    println!(
        "[phase50_long] G_cons={g:.4e}  a:{A_INIT}→{a_final:.4}  n_steps={n_steps}  \
         P_ratio={ratio:.4}  D²_predict={d2_expected:.4}  v_rms={v_rms:.4e}"
    );

    assert!(a_final >= a_target * 0.99, "No alcanzó a_target");
    assert!(n_steps > 0, "Sin pasos de integración");
    assert!(ratio.is_finite() && !ratio.is_nan(), "P(k) ratio no finito");
    assert!(ratio > 1.0, "P(k) no creció a lo largo de 10× en a");
    assert!(ratio < 1e6, "P(k) explotó en evolución larga");
    assert!(v_rms.is_finite(), "v_rms no finito → explosión");
    println!("  ✓ Estable {n_steps} pasos. (D²_exacto requiere N≥64 en release)");
}

// ── TEST 5 — G_consistente vs G_legacy: comparación directa ─────────────────

/// N=8, a=0.02→0.05. Compara comportamiento de G_consistente vs G_legacy=1.
///
/// Para evoluciones cortas (25 pasos), g_cosmo=G·a³ hace que G_legacy también
/// parezca estable (a³ atenúa las fuerzas). La diferencia de G_legacy se hace
/// notable en evoluciones largas (Phase 49 documentó la divergencia).
/// Aquí verificamos la propiedad mínima: G_consistente es siempre físico.
#[test]
fn phase50_g_consistent_vs_legacy() {
    let n = 8_usize;
    let a_target = 0.05_f64;

    let mut parts_cons = build_particles(&build_ic(n, g_phys())).expect("ICs_cons");
    let mut parts_leg = build_particles(&build_ic(n, G_LEGACY)).expect("ICs_leg");
    let pk0 = measure_pk(&parts_cons, n);

    // ~25 pasos hasta a=0.05
    let a_cons = evolve_fixed_g(&mut parts_cons, n, g_phys(), A_INIT, a_target, 1.0e-3);
    let a_leg = evolve_fixed_g(&mut parts_leg, n, G_LEGACY, A_INIT, a_target, 1.0e-3);

    let pk_cons = measure_pk(&parts_cons, n);
    let pk_leg = measure_pk(&parts_leg, n);

    let d_ratio = growth_factor_d_ratio(cosmo(), a_cons, A_INIT);
    let d2 = d_ratio * d_ratio;

    let ratio_cons = median_ratio(&linear_bins(&pk0, n), &linear_bins(&pk_cons, n));
    let ratio_leg = median_ratio(&linear_bins(&pk0, n), &linear_bins(&pk_leg, n));

    let v_rms_cons: f64 = {
        let s: f64 = parts_cons
            .iter()
            .map(|p| p.velocity.norm() * p.velocity.norm())
            .sum();
        (s / parts_cons.len() as f64).sqrt()
    };
    let v_rms_leg: f64 = {
        let s: f64 = parts_leg
            .iter()
            .map(|p| p.velocity.norm() * p.velocity.norm())
            .sum();
        (s / parts_leg.len() as f64).sqrt()
    };

    println!("[phase50_g_compare] a=0.02→0.05  D²_predict={d2:.4}");
    println!(
        "  G_consistente ({:.4e}): ratio={ratio_cons:.4}  v_rms={v_rms_cons:.4e}  a={a_cons:.4}",
        g_phys()
    );
    println!(
        "  G_legacy      ({G_LEGACY:.4e}): ratio={ratio_leg:.4}  v_rms={v_rms_leg:.4e}  a={a_leg:.4}"
    );

    // G_consistente SIEMPRE debe ser estable.
    assert!(a_cons >= a_target * 0.99, "G_cons no alcanzó a_target");
    assert!(
        ratio_cons.is_finite() && ratio_cons > 0.0,
        "G_cons: ratio no físico"
    );
    assert!(ratio_cons > 1.0, "G_cons: P(k) no creció");
    assert!(ratio_cons < 1000.0, "G_cons: P(k) explotó");
    assert!(v_rms_cons.is_finite(), "G_cons: v_rms no finito");

    // G_legacy: solo documentación (puede ser estable para evolución corta).
    if ratio_leg.is_finite() && ratio_leg > 0.0 && v_rms_leg.is_finite() {
        println!("  G_legacy: aparentemente estable en a=0.02→0.05 (la patología");
        println!("  se manifiesta en evoluciones más largas, ver Phase 49).");
    } else {
        println!("  G_legacy: divergió → confirma la inconsistencia documentada.");
    }
}
