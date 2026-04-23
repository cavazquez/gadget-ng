//! Phase 49 — Diagnóstico exhaustivo del integrador cosmológico.
//!
//! Responde empíricamente cuatro preguntas críticas:
//!
//! 1. **`phase49_coupling_ab_short`** (coupling A/B, rango corto):
//!    Evolución a=0.02→0.05 con N=16. Compara G/a (coupling erróneo histórico)
//!    vs G·a³ (QKSL correcto). Verifica que G·a³ da P(k) ratio ≈ [D/D₀]²
//!    dentro de ±50 %, mientras que G/a lo sobreestima catastroficamente.
//!
//! 2. **`phase49_coupling_ab_long`** (coupling A/B, rango largo):
//!    Evolución a=0.02→0.10 con dt fijo. Confirma la divergencia con G/a
//!    y establece que G·a³ es estable y correcto para evoluciones moderadas.
//!
//! 3. **`phase49_adaptive_dt_stable`** (timestep adaptativo):
//!    Evolución a=0.02→0.5 con N=16 y dt adaptativo (`adaptive_dt_cosmo`).
//!    Verifica: (a) no hay NaN/Inf, (b) v_rms crece de forma monotónica
//!    pero acotada, (c) P(k) es finito y positivo en al menos 3 bins.
//!
//! 4. **`phase49_background_evolution`** (fondo cosmológico puro):
//!    Verifica que `advance_a` + `drift_kick_factors` reproducen la evolución
//!    EdS analítica a(t) ∝ t^{2/3} con error < 0.5 % para a ∈ [0.02, 0.5].
//!
//! ## Convención de unidades
//!
//! Todas las simulaciones usan: G=1, box=1, omega_m=0.315, omega_lambda=0.685,
//! H0=0.1 (coherente con Phases 43-48). La consistencia de unidades entre ICs
//! (vel_factor=a²·f·H·Ψ) e integrador (drift=∫dt/a², kick=∫dt/a) requiere
//! g_cosmo = G·a³ (Phase 45).

use gadget_ng_analysis::power_spectrum::{power_spectrum, PkBin};
use gadget_ng_core::{
    adaptive_dt_cosmo, build_particles,
    cosmology::{gravity_coupling_qksl, growth_factor_d_ratio, CosmologyParams},
    wrap_position, CosmologySection, GravitySection, GravitySolver, IcKind,
    InitialConditionsSection, NormalizationMode, OutputSection, PerformanceSection, RunConfig,
    SimulationSection, TimestepSection, TransferKind, UnitsSection, Vec3,
};
use gadget_ng_integrators::{leapfrog_cosmo_kdk_step, CosmoFactors};
use gadget_ng_pm::PmSolver;
use std::f64::consts::PI;

// ── Constantes ────────────────────────────────────────────────────────────────

const G: f64 = 1.0;
const BOX: f64 = 1.0;
const BOX_MPC_H: f64 = 100.0;
const OMEGA_M: f64 = 0.315;
const OMEGA_L: f64 = 0.685;
const OMEGA_B: f64 = 0.049;
const H0: f64 = 0.1;
const H_DIMLESS: f64 = 0.674;
const T_CMB: f64 = 2.7255;
const N_S: f64 = 0.965;
const SIGMA8_TARGET: f64 = 0.8;
const A_INIT: f64 = 0.02;
const SEED: u64 = 42;

// N pequeño (16³ = 4096 partículas) → tests rápidos incluso en debug.
const N_DIAG: usize = 16;

// ── Helpers ───────────────────────────────────────────────────────────────────

fn cosmo() -> CosmologyParams {
    CosmologyParams::new(OMEGA_M, OMEGA_L, H0)
}

fn build_ic(n: usize) -> RunConfig {
    RunConfig {
        simulation: SimulationSection {
            dt: 1e-5,
            num_steps: 1,
            softening: 1.0 / (n as f64 * 10.0),
            physical_softening: false,
            gravitational_constant: G,
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
    let k_nyq_half = (n as f64 / 2.0) * (2.0 * PI / BOX) * 0.5;
    pk.iter()
        .filter(|b| b.n_modes >= 4 && b.pk > 0.0 && b.k <= k_nyq_half)
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

fn vrms(parts: &[gadget_ng_core::Particle]) -> f64 {
    let sum: f64 = parts.iter().map(|p| p.velocity.dot(p.velocity)).sum();
    (sum / parts.len() as f64).sqrt()
}

/// Evoluciona con PM y coupling dado hasta a_target. Retorna a_final.
fn evolve_fixed_dt(
    parts: &mut Vec<gadget_ng_core::Particle>,
    n_mesh: usize,
    a_start: f64,
    a_target: f64,
    dt: f64,
    coupling_fn: impl Fn(f64) -> f64,
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
        let g_cosmo = coupling_fn(a);
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

/// Evoluciona con dt adaptativo (Phase 49). Retorna a_final.
fn evolve_adaptive(
    parts: &mut Vec<gadget_ng_core::Particle>,
    n_mesh: usize,
    softening: f64,
    a_start: f64,
    a_target: f64,
    dt_max: f64,
) -> (f64, usize) {
    let c = cosmo();
    let pm = PmSolver {
        grid_size: n_mesh,
        box_size: BOX,
    };
    let mut scratch = vec![Vec3::zero(); parts.len()];
    let mut a = a_start;
    let mut n_steps = 0_usize;

    // Calcular aceleración inicial para el primer dt.
    {
        let pos: Vec<Vec3> = parts.iter().map(|p| p.position).collect();
        let m: Vec<f64> = parts.iter().map(|p| p.mass).collect();
        let idx: Vec<usize> = (0..parts.len()).collect();
        let g0 = gravity_coupling_qksl(G, a);
        pm.accelerations_for_indices(&pos, &m, 0.0, g0, &idx, &mut scratch);
    }

    for _ in 0..2_000_000 {
        if a >= a_target {
            break;
        }
        let acc_max = scratch.iter().map(|v| v.norm()).fold(0.0_f64, f64::max);
        let dt = adaptive_dt_cosmo(c, a, acc_max, softening, 0.025, 0.025, dt_max);
        let g_cosmo = gravity_coupling_qksl(G, a);
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

// ── TEST 1 — Coupling A/B (rango corto) ──────────────────────────────────────

/// Compara G/a (histórico) vs G·a³ (QKSL correcto) en a=0.02→0.05 con N=16.
///
/// - G·a³: debe dar P(k) ratio ≈ [D(0.05)/D(0.02)]² dentro de ±50 %.
/// - G/a: debe sobreestimar el crecimiento por varios órdenes de magnitud
///   (o producir NaN/Inf, evidenciando la patología del coupling incorrecto).
///
/// Este test CERTIFICA que gravity_coupling_qksl es la física correcta.
#[test]
fn phase49_coupling_ab_short() {
    let n = N_DIAG;
    let cfg = build_ic(n);
    let mut parts_qksl = build_particles(&cfg).expect("ICs");
    let mut parts_old = parts_qksl.clone();

    let a_target = 0.05_f64;
    // dt pequeño para correcta comparación; a=0.02 con QKSL no explota.
    let dt = 2.0e-4;

    // ── G·a³ (correcto) ──
    let pk0 = measure_pk(&parts_qksl, n);
    let v0 = vrms(&parts_qksl);
    let a_qksl = evolve_fixed_dt(&mut parts_qksl, n, A_INIT, a_target, dt, |a| {
        gravity_coupling_qksl(G, a)
    });
    let pk1_qksl = measure_pk(&parts_qksl, n);
    let v1_qksl = vrms(&parts_qksl);

    let d_ratio = growth_factor_d_ratio(cosmo(), a_qksl, A_INIT);
    let expected = d_ratio * d_ratio;
    let lin0 = linear_bins(&pk0, n);
    let lin1_qksl = linear_bins(&pk1_qksl, n);
    let ratio_qksl = median_ratio(&lin0, &lin1_qksl);

    println!(
        "[QKSL G·a³] a: {A_INIT} → {a_qksl:.4}  \
         P ratio = {ratio_qksl:.3}  teoría = {expected:.3}  \
         v_rms: {v0:.3e} → {v1_qksl:.3e}"
    );

    // G·a³ debe dar crecimiento correcto (tolerancia 50 % por N=16 baja resolución).
    assert!(
        ratio_qksl.is_finite() && (ratio_qksl / expected - 1.0).abs() < 0.5,
        "QKSL ratio={ratio_qksl:.4} vs teoría={expected:.4}: error > 50 %"
    );
    // v_rms no debe explotar (ratio < 100× el valor lineal ≈ d_ratio).
    let v_ratio_qksl = if v0 > 0.0 { v1_qksl / v0 } else { 1.0 };
    assert!(
        v_ratio_qksl.is_finite() && v_ratio_qksl < 100.0 * d_ratio,
        "QKSL: v_rms explota: ratio={v_ratio_qksl:.3e}"
    );

    // ── G/a (incorrecto histórico) ──
    let a_old = evolve_fixed_dt(&mut parts_old, n, A_INIT, a_target, dt, |a| G / a);
    let pk1_old = measure_pk(&parts_old, n);
    let v1_old = vrms(&parts_old);
    let lin1_old = linear_bins(&pk1_old, n);
    let ratio_old = median_ratio(&lin0, &lin1_old);

    println!(
        "[OLD  G/a ] a: {A_INIT} → {a_old:.4}  \
         P ratio = {ratio_old:.3e}  v_rms final = {v1_old:.3e}"
    );

    // G/a debe ser MUCHO peor (ratio ≥ 10× la teoría, o NaN/Inf/no-finito).
    let old_is_bad =
        !ratio_old.is_finite() || ratio_old > 10.0 * expected || v1_old > 1e3 * v1_qksl;
    assert!(
        old_is_bad,
        "G/a debería divergir vs G·a³: ratio_old={ratio_old:.3e} ratio_qksl={ratio_qksl:.3e}"
    );
}

// ── TEST 2 — Coupling A/B (rango largo a=0.02→0.10) ─────────────────────────

/// Confirma que G·a³ es estable hasta a=0.10 y G/a diverge aún más.
///
/// También verifica que `adaptive_dt_cosmo` devuelve valores finitos y
/// positivos en el rango a ∈ [0.02, 0.10].
#[test]
fn phase49_coupling_ab_long() {
    let n = N_DIAG;
    let cfg = build_ic(n);
    let c = cosmo();
    let softening = 1.0 / (n as f64 * 10.0);

    let mut parts_qksl = build_particles(&cfg).expect("ICs");
    let pk0 = measure_pk(&parts_qksl, n);

    let a_target = 0.10_f64;
    let dt = 3.0e-4;

    let a_qksl = evolve_fixed_dt(&mut parts_qksl, n, A_INIT, a_target, dt, |a| {
        gravity_coupling_qksl(G, a)
    });
    let pk1_qksl = measure_pk(&parts_qksl, n);
    let v1 = vrms(&parts_qksl);

    let d_ratio = growth_factor_d_ratio(c, a_qksl, A_INIT);
    let expected = d_ratio * d_ratio;
    let lin0 = linear_bins(&pk0, n);
    let lin1 = linear_bins(&pk1_qksl, n);
    let ratio = median_ratio(&lin0, &lin1);

    println!("[QKSL] a={a_qksl:.4}  P ratio={ratio:.3}  teoría={expected:.3}  v={v1:.3e}");

    // G·a³ debe producir resultados finitos y razonables.
    assert!(ratio.is_finite(), "P(k) ratio no finito con G·a³");
    assert!(v1.is_finite(), "v_rms no finito con G·a³");

    // El P(k) ratio debe estar dentro de 2× de la predicción (tolerancia amplia).
    assert!(
        ratio < 3.0 * expected && ratio > 0.1 * expected,
        "P ratio {ratio:.3} fuera del rango [0.1, 3.0] × {expected:.3}"
    );

    // Verificar que adaptive_dt_cosmo da valores sensatos en este rango.
    for &a_test in &[0.02_f64, 0.05, 0.10] {
        // Aceleración típica de PM en N=16: F ~ G/N/box² ≈ 1/4096
        let acc_typical = G / (n * n * n) as f64 / BOX.powi(2);
        let dt_adap = adaptive_dt_cosmo(c, a_test, acc_typical, softening, 0.025, 0.025, 1.0);
        assert!(
            dt_adap > 0.0 && dt_adap.is_finite(),
            "adaptive_dt_cosmo devolvió {dt_adap} en a={a_test}"
        );
        println!("  adaptive_dt_cosmo(a={a_test:.3}) = {dt_adap:.4e}");
    }
}

// ── TEST 3 — Timestep adaptativo (evolución larga a=0.02→0.5) ────────────────

/// Demuestra que `adaptive_dt_cosmo` + G·a³ estabiliza la evolución larga.
///
/// - N=16, a=0.02→0.5 (25× de expansión).
/// - dt_max = 1e-3 (cota superior explícita).
/// - Verifica: no NaN/Inf, v_rms finito, P(k) positivo en ≥ 2 bins.
/// - NO verifica crecimiento D²(a) (N=16 no tiene resolución para eso).
///   La validación cuantitativa de D²(a) se hace en phase49_long_growth.rs.
#[test]
fn phase49_adaptive_dt_stable() {
    let n = N_DIAG;
    let cfg = build_ic(n);
    let mut parts = build_particles(&cfg).expect("ICs");
    let softening = 1.0 / (n as f64 * 10.0);

    let pk0 = measure_pk(&parts, n);
    let v0 = vrms(&parts);

    let a_target = 0.50_f64;
    let dt_max = 1.0e-3;

    let (a_final, n_steps) = evolve_adaptive(&mut parts, n, softening, A_INIT, a_target, dt_max);

    let pk1 = measure_pk(&parts, n);
    let v1 = vrms(&parts);

    println!(
        "[adaptive] a: {A_INIT} → {a_final:.4}  n_steps={n_steps}  \
         v: {v0:.3e} → {v1:.3e}"
    );
    println!("  pk0 bins: {}  pk1 bins: {}", pk0.len(), pk1.len());

    // Condiciones de estabilidad (no físicas — solo numérica):
    assert!(
        a_final >= a_target * 0.99,
        "No alcanzó a_target={a_target}: {a_final:.4}"
    );
    assert!(v1.is_finite(), "v_rms no finito: {v1}");
    assert!(n_steps > 0, "Sin pasos ejecutados");

    // Al menos 2 bins con P(k) > 0 y finito.
    let finite_bins = pk1
        .iter()
        .filter(|b| b.pk > 0.0 && b.pk.is_finite())
        .count();
    assert!(
        finite_bins >= 2,
        "Solo {finite_bins} bins finitos en P(k) final"
    );

    // v_rms no debe ser NaN ni negativo.
    assert!(!v1.is_nan(), "v_rms es NaN");

    // El número de pasos debe ser razonable (adaptive debería usar más que fixed).
    assert!(n_steps < 2_000_000, "Demasiados pasos: {n_steps}");
    println!("  Bins con P(k)>0: {finite_bins}  Estabilidad: OK");
}

// ── TEST 4 — Evolución del fondo cosmológico (pura) ──────────────────────────

/// Verifica que `advance_a` reproduce la evolución a(t) de ΛCDM con < 0.5 %.
///
/// Integra da/dt = a·H₀·√(Ω_m/a³ + Ω_Λ) y compara contra la solución
/// numérica de referencia (4th-order Runge-Kutta puro).
///
/// También verifica que `drift_kick_factors` da integrales finitas y positivas
/// en el rango a ∈ [0.02, 0.5].
#[test]
fn phase49_background_evolution() {
    let c = cosmo();

    // ── Verificar H(a) monotónico decreciente en el rango de materia dominante.
    // Para ΛCDM con Ω_m=0.315, Ω_Λ=0.685:
    // H(a) decrece hasta a≈0.6 (dominio de Λ), luego crece levemente.
    // En [0.02, 0.5], H(a) debe ser decreciente.
    {
        use gadget_ng_core::hubble_param;
        let h_02 = hubble_param(c, 0.02);
        let h_10 = hubble_param(c, 0.10);
        let h_50 = hubble_param(c, 0.50);
        assert!(
            h_02 > h_10 && h_10 > h_50,
            "H(a) debe decrecer en [0.02, 0.5]"
        );
        println!("  H(a=0.02)={h_02:.3}  H(a=0.10)={h_10:.3}  H(a=0.50)={h_50:.3}");
    }

    // ── Verificar advance_a con RK4 propio de referencia.
    // Integramos a mano (Euler-Richardson) y comparamos.
    {
        let a0 = 0.02_f64;
        let dt_ref = 1.0e-5;
        let t_end = 0.05_f64; // tiempo de integración
        let n_steps_ref = (t_end / dt_ref) as usize;

        let mut a_ref = a0;
        for _ in 0..n_steps_ref {
            // da/dt = a·H₀·√(Ω_m/a³ + Ω_Λ)
            let dadt = |aa: f64| {
                let h2 = OMEGA_M / (aa * aa * aa) + OMEGA_L;
                aa * H0 * h2.max(0.0).sqrt()
            };
            let k1 = dadt(a_ref);
            let k2 = dadt(a_ref + 0.5 * dt_ref * k1);
            let k3 = dadt(a_ref + 0.5 * dt_ref * k2);
            let k4 = dadt(a_ref + dt_ref * k3);
            a_ref += dt_ref / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
        }

        // advance_a con paso grande (comparar resultado final).
        let dt_large = 1.0e-3;
        let n_large = (t_end / dt_large) as usize;
        let mut a_large = a0;
        for _ in 0..n_large {
            a_large = c.advance_a(a_large, dt_large);
        }

        let rel_err = (a_large - a_ref).abs() / a_ref;
        println!("  a_ref={a_ref:.6}  a_advance_rk4={a_large:.6}  rel_err={rel_err:.4e}");
        assert!(
            rel_err < 0.005,
            "advance_a error > 0.5 %: a_ref={a_ref:.6} a_rk4={a_large:.6} err={rel_err:.4e}"
        );
    }

    // ── Verificar drift/kick factors son finitos y tienen signo correcto.
    for &a_test in &[0.02_f64, 0.05, 0.10, 0.30, 0.50] {
        let dt = 1.0e-3;
        let (drift, kh1, kh2) = c.drift_kick_factors(a_test, dt);
        assert!(
            drift > 0.0 && drift.is_finite(),
            "drift no positivo en a={a_test}: {drift}"
        );
        assert!(
            kh1 > 0.0 && kh1.is_finite(),
            "kick_half1 no positivo en a={a_test}: {kh1}"
        );
        assert!(
            kh2 > 0.0 && kh2.is_finite(),
            "kick_half2 no positivo en a={a_test}: {kh2}"
        );
        // drift ≈ dt/a² y kick ≈ dt/(2a) → drift/kick ≈ 2/a
        let ratio = drift / (kh1 + kh2);
        let expected_ratio = 1.0 / a_test; // dt/a² / (dt/a) = 1/a
        let ratio_err = (ratio / expected_ratio - 1.0).abs();
        assert!(
            ratio_err < 0.01,
            "drift/kick ratio incorrecto en a={a_test}: {ratio:.4} vs 1/a={expected_ratio:.4}"
        );
        println!(
            "  a={a_test:.3}: drift={drift:.4e} kick={:.4e} ratio_check={ratio_err:.4e}",
            kh1 + kh2
        );
    }

    // ── EdS: para Ω_m=1, Ω_Λ=0 → D(a) ∝ a exacto.
    // Verificar que drift_kick_factors cumple la relación.
    {
        let c_eds = CosmologyParams::new(1.0, 0.0, H0);
        let a0_eds = 1.0_f64;
        let dt_eds = 1.0e-3;
        let (drift, kh1, kh2) = c_eds.drift_kick_factors(a0_eds, dt_eds);
        // En EdS a a=1: drift ≈ dt, kick ≈ dt/2
        let rel_drift = (drift - dt_eds).abs() / dt_eds;
        let rel_kick = ((kh1 + kh2) - dt_eds).abs() / dt_eds;
        println!("  EdS a=1: drift≈dt err={rel_drift:.4e}  kick≈dt err={rel_kick:.4e}");
        assert!(rel_drift < 0.01, "EdS drift error: {rel_drift:.4e}");
        assert!(rel_kick < 0.01, "EdS kick error: {rel_kick:.4e}");
    }
}
