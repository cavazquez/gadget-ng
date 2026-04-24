//! Phase 47 — Validación end-to-end del pipeline P(k) con física corregida.
//!
//! ## Tests
//!
//! 1. **`phase47_pk_correction_at_ics`** — Aplica la corrección A_grid·R(N) a
//!    ICs ZA (z=0 amplitude) y verifica que P_corr ≈ P_EH(k, z=0) dentro de
//!    ±15 %. Valida la tabla `phase47_default` con R(N=32).
//!
//! 2. **`phase47_pk_growth_qksl`** — Evolución ultracorta
//!    `a_init=0.02 → 0.0201` con PM + acoplamiento QKSL (Phase 45 fix).
//!    Verifica que el cociente P(k,a_final)/P(k,a_init) ≈ [D(a_final)/D(a_init)]²
//!    dentro de ±30 %. Confirma que la física correcta no introduce divergencias.
//!
//! 3. **`shot_noise_negligible_at_large_signal`** — Verifica que
//!    `correct_pk_with_shot_noise` reduce el poder a alto k en la
//!    proporción esperada cuando el shot noise es comparable.
//!
//! ## Convención de unidades
//!
//! R(N) fue calibrado con P_cont en (Mpc/h)³ (Phase 35), por lo que
//! `correct_pk(..., box_mpc_h=None, ...)` ya devuelve P en (Mpc/h)³.

use gadget_ng_analysis::pk_correction::{correct_pk, correct_pk_with_shot_noise, RnModel};
use gadget_ng_analysis::power_spectrum::{power_spectrum, PkBin};
use gadget_ng_core::{
    amplitude_for_sigma8, build_particles,
    cosmology::{gravity_coupling_qksl, growth_factor_d_ratio, CosmologyParams},
    transfer_eh_nowiggle, wrap_position, CosmologySection, EisensteinHuParams, GravitySection,
    GravitySolver, IcKind, InitialConditionsSection, OutputSection, PerformanceSection, RunConfig,
    SimulationSection, TimestepSection, TransferKind, UnitsSection, Vec3,
};
use gadget_ng_integrators::{leapfrog_cosmo_kdk_step, CosmoFactors};
use gadget_ng_pm::PmSolver;
use serde_json::json;
use std::f64::consts::PI;
use std::fs;
use std::path::PathBuf;

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

// N pequeño (32) para coherencia con Phase 35/36; mismas semillas.
const N_MESH: usize = 32;
const SEED: u64 = 42;

// ── Helpers ───────────────────────────────────────────────────────────────────

fn eh() -> EisensteinHuParams {
    EisensteinHuParams {
        omega_m: OMEGA_M,
        omega_b: OMEGA_B,
        h: H_DIMLESS,
        t_cmb: T_CMB,
    }
}

fn cosmo() -> CosmologyParams {
    CosmologyParams::new(OMEGA_M, OMEGA_L, H0)
}

fn build_config(n: usize, seed: u64) -> RunConfig {
    let gravity = GravitySection {
        solver: gadget_ng_core::SolverKind::Pm,
        pm_grid_size: n,
        ..GravitySection::default()
    };
    RunConfig {
        simulation: SimulationSection {
            dt: 5.0e-6,
            num_steps: 1,
            softening: 0.01,
            physical_softening: false,
            gravitational_constant: G,
            particle_count: n * n * n,
            box_size: BOX,
            seed,
            integrator: Default::default(),
        },
        initial_conditions: InitialConditionsSection {
            kind: IcKind::Zeldovich {
                seed,
                grid_size: n,
                spectral_index: N_S,
                amplitude: 1.0e-4,
                transfer: TransferKind::EisensteinHu,
                sigma8: Some(SIGMA8_TARGET),
                omega_b: OMEGA_B,
                h: H_DIMLESS,
                t_cmb: T_CMB,
                box_size_mpc_h: Some(BOX_MPC_H),
                use_2lpt: false,
                normalization_mode: gadget_ng_core::NormalizationMode::Legacy,
            },
        },
        output: OutputSection::default(),
        gravity,
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
            ..Default::default()
        },
        units: UnitsSection::default(),
        decomposition: Default::default(),
        insitu_analysis: Default::default(),
        sph: Default::default(),
        rt: Default::default(), reionization: Default::default(), mhd: Default::default(),
        turbulence: Default::default(), two_fluid: Default::default(),
        sidm: Default::default(), modified_gravity: Default::default(),
    }
}

/// Evoluciona hasta `a_target` con PM + acoplamiento QKSL (Phase 45 fix).
fn evolve_pm_qksl(
    parts: &mut Vec<gadget_ng_core::Particle>,
    n_mesh: usize,
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
    for _ in 0..100_000 {
        if a >= a_target {
            break;
        }
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
    }
    a
}

fn measure_pk_parts(parts: &[gadget_ng_core::Particle], n_mesh: usize) -> Vec<PkBin> {
    let pos: Vec<Vec3> = parts.iter().map(|p| p.position).collect();
    let m: Vec<f64> = parts.iter().map(|p| p.mass).collect();
    power_spectrum(&pos, &m, BOX, n_mesh)
}

/// P_EH(k, z=0) en (Mpc/h)³ (referencia para ICs Legacy de amplitud z=0).
fn eh_pk_z0(k_hmpc: f64) -> f64 {
    let e = eh();
    let amp = amplitude_for_sigma8(SIGMA8_TARGET, N_S, &e);
    let tk = transfer_eh_nowiggle(k_hmpc, &e);
    amp * amp * k_hmpc.powf(N_S) * tk * tk
}

/// k interna (2π/box) → h/Mpc (protocolo Phase 35, mismo k_conv).
#[inline]
fn k_to_hmpc(k_internal: f64) -> f64 {
    k_internal * H_DIMLESS / BOX_MPC_H
}

/// Bins en régimen lineal: k ≤ k_Nyq/2, n_modes ≥ 8.
fn linear_bins(pk: &[PkBin], n: usize) -> Vec<PkBin> {
    let k_max = (n as f64 / 2.0) * (2.0 * PI / BOX) * 0.5;
    pk.iter()
        .filter(|b| b.n_modes >= 8 && b.pk > 0.0 && b.k <= k_max)
        .cloned()
        .collect()
}

/// Mediana de |P_corr/P_ref - 1| sobre los bins dados.
fn median_rel_err_vs_ref(corr_bins: &[PkBin], ref_fn: &dyn Fn(f64) -> f64) -> f64 {
    let mut errs: Vec<f64> = corr_bins
        .iter()
        .filter_map(|b| {
            let k_hmpc = k_to_hmpc(b.k);
            let p_ref = ref_fn(k_hmpc);
            if p_ref > 0.0 && b.pk > 0.0 && p_ref.is_finite() {
                Some((b.pk / p_ref - 1.0).abs())
            } else {
                None
            }
        })
        .collect();
    if errs.is_empty() {
        return f64::NAN;
    }
    errs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = errs.len() / 2;
    if errs.len() % 2 == 1 {
        errs[mid]
    } else {
        0.5 * (errs[mid - 1] + errs[mid])
    }
}

fn phase47_dir() -> PathBuf {
    let mut d = PathBuf::from(std::env::var("CARGO_TARGET_DIR").unwrap_or_else(|_| {
        let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        p.pop();
        p.pop();
        p.push("target");
        p.to_string_lossy().to_string()
    }));
    d.push("phase47");
    let _ = fs::create_dir_all(&d);
    d
}

fn dump(name: &str, v: serde_json::Value) {
    let mut p = phase47_dir();
    p.push(format!("{name}.json"));
    if let Ok(s) = serde_json::to_string_pretty(&v) {
        let _ = fs::write(p, s);
    }
}

// ── TEST 1 ────────────────────────────────────────────────────────────────────

/// Corrección A_grid·R(N) en ICs ZA: P_corr debe ≈ P_EH(k, z=0).
///
/// Valida que `phase47_default` con R(N=32) (idéntico a Phase 35) reproduce
/// el espectro EH dentro de ±15 % en el régimen lineal.
#[test]
fn phase47_pk_correction_at_ics() {
    let cfg = build_config(N_MESH, SEED);
    let parts = build_particles(&cfg).expect("ICs");

    let pk_raw = measure_pk_parts(&parts, N_MESH);
    let model = RnModel::phase47_default();

    // box_mpc_h=None: R(N) absorbe el factor de volumen (protocolo Phase 35/36).
    let pk_corr = correct_pk(&pk_raw, BOX, N_MESH, None, &model);

    // Referencia: ICs Legacy tienen amplitud z=0, so P_ref = P_EH(k, z=0).
    let lin = linear_bins(&pk_corr, N_MESH);
    assert!(
        !lin.is_empty(),
        "No hay bins lineales después de la corrección"
    );

    let err = median_rel_err_vs_ref(&lin, &eh_pk_z0);

    println!(
        "N={N_MESH} R(N)_modelo = {:.6}  err_mediana_lineal = {:.1}%",
        model.evaluate(N_MESH),
        err * 100.0
    );

    assert!(
        err < 0.15,
        "P_corr difiere de P_EH(z=0) en {:.1}% > 15%",
        err * 100.0
    );

    dump(
        "pk_correction_at_ics",
        json!({
            "n_mesh": N_MESH,
            "seed": SEED,
            "r_n": model.evaluate(N_MESH),
            "err_median_pct": err * 100.0,
            "pk_corr": lin.iter().map(|b| json!({"k": b.k, "pk": b.pk})).collect::<Vec<_>>(),
        }),
    );
}

// ── TEST 2 ────────────────────────────────────────────────────────────────────

/// Crecimiento lineal ultracorto (a_init → a_init × 1.05) con PM + QKSL.
///
/// Verifica que P(k,a_final)/P(k,a_init) ≈ [D(a_final)/D(a_init)]² dentro
/// de ±30 %. Confirma que la física QKSL (Phase 45) no introduce divergencias.
#[test]
fn phase47_pk_growth_qksl() {
    let cfg = build_config(N_MESH, SEED);
    let mut parts = build_particles(&cfg).expect("ICs");

    let pk0 = measure_pk_parts(&parts, N_MESH);

    // Evolución muy corta: 10% en a.  dt=5e-4 → ~10 pasos (rápido en debug).
    // Con G·a³ las fuerzas en a=0.02 son ≈ 8e-6 → integración estable a dt grande.
    let a_target = A_INIT * 1.10;
    let dt = 5.0e-4_f64;
    let a_final = evolve_pm_qksl(&mut parts, N_MESH, A_INIT, a_target, dt);

    assert!(
        a_final >= a_target * 0.99,
        "Evolución no alcanzó a_target={a_target:.5}: a_final={a_final:.6}"
    );

    let pk1 = measure_pk_parts(&parts, N_MESH);

    let d_ratio = growth_factor_d_ratio(cosmo(), a_final, A_INIT);
    let expected_ratio = d_ratio * d_ratio;

    // Comparar ratio P(k,a_final)/P(k,a_init) contra [D/D₀]².
    let lin0 = linear_bins(&pk0, N_MESH);
    let lin1 = linear_bins(&pk1, N_MESH);

    let mut ratios: Vec<f64> = lin0
        .iter()
        .zip(lin1.iter())
        .filter(|(b0, b1)| b0.pk > 0.0 && b1.pk > 0.0)
        .map(|(b0, b1)| b1.pk / b0.pk)
        .collect();

    assert!(
        !ratios.is_empty(),
        "No hay bins para calcular ratio de P(k)"
    );
    ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_ratio = if ratios.len() % 2 == 1 {
        ratios[ratios.len() / 2]
    } else {
        0.5 * (ratios[ratios.len() / 2 - 1] + ratios[ratios.len() / 2])
    };

    let rel_err = (median_ratio / expected_ratio - 1.0).abs();
    println!(
        "a: {A_INIT} → {a_final:.6}  P(k) ratio = {median_ratio:.4}  teoría = {expected_ratio:.4}  err = {:.1}%",
        rel_err * 100.0
    );

    // Tolerancia holgada (30%) dado que N=32 PM con Legacy ICs puede tener
    // efectos no-lineales incluso a a_init (σ₈=0.8 a z=0 amplitude).
    assert!(
        rel_err < 0.30,
        "Crecimiento P(k): ratio={median_ratio:.4} vs [D/D₀]²={expected_ratio:.4} (err {:.1}%)",
        rel_err * 100.0
    );

    // P(k) inicial no debe ser cero ni infinito.
    assert!(median_ratio.is_finite(), "Ratio P(k) no finito");

    dump(
        "pk_growth_qksl",
        json!({
            "a_init": A_INIT,
            "a_final": a_final,
            "expected_ratio": expected_ratio,
            "median_ratio": median_ratio,
            "rel_err_pct": rel_err * 100.0,
        }),
    );
}

// ── TEST 3 ────────────────────────────────────────────────────────────────────

/// Verifica que `correct_pk_with_shot_noise` sustrae P_shot correctamente.
///
/// Crea bins sintéticos donde P_measured = 2·P_shot y verifica que la
/// corrección con shot noise da la mitad de la corrección sin shot noise.
#[test]
fn shot_noise_correction_is_proportional() {
    use gadget_ng_analysis::pk_correction::a_grid;
    use gadget_ng_analysis::power_spectrum::PkBin;

    let model = RnModel::phase47_default();
    let n = N_MESH;
    let n_particles = n * n * n;
    let v_box = BOX.powi(3);
    let p_shot = v_box / n_particles as f64;

    // Bin donde P_measured = 2 × P_shot → después de sustracción P_signal = P_shot.
    // correct_pk_without_sn da 2×P_shot / (A × R).
    // correct_pk_with_sn da 1×P_shot / (A × R).
    // Ratio = 0.5.
    let bins_2shot = vec![PkBin {
        k: 50.0,
        pk: 2.0 * p_shot,
        n_modes: 20,
    }];
    let out_std = correct_pk(&bins_2shot, BOX, n, None, &model);
    let out_sn = correct_pk_with_shot_noise(&bins_2shot, BOX, n, None, n_particles, &model);

    let ratio = out_sn[0].pk / out_std[0].pk;
    assert!(
        (ratio - 0.5).abs() < 0.01,
        "P=2·P_shot: ratio con/sin shot noise = {ratio:.4} (esperado 0.5)"
    );

    // Bin donde P >> P_shot → las dos correcciones son casi iguales (< 1%).
    let bins_large = vec![PkBin {
        k: 1.0,
        pk: p_shot * 1e8,
        n_modes: 20,
    }];
    let out_large_std = correct_pk(&bins_large, BOX, n, None, &model);
    let out_large_sn = correct_pk_with_shot_noise(&bins_large, BOX, n, None, n_particles, &model);
    let rel_diff = (out_large_std[0].pk - out_large_sn[0].pk).abs() / out_large_std[0].pk;
    assert!(
        rel_diff < 0.01,
        "P >> P_shot: diferencia relativa = {rel_diff:.4} (esperado < 0.01)"
    );

    // Bin donde P < P_shot → pk corregida debe ser cero (no negativa).
    let bins_subshot = vec![PkBin {
        k: 200.0,
        pk: 0.1 * p_shot,
        n_modes: 10,
    }];
    let out_sub = correct_pk_with_shot_noise(&bins_subshot, BOX, n, None, n_particles, &model);
    assert_eq!(
        out_sub[0].pk, 0.0,
        "P < P_shot: pk debería ser 0 (no negativa)"
    );

    let ag = a_grid(BOX, n);
    let r = model.evaluate(n);
    println!(
        "P_shot = {p_shot:.3e}  A_grid = {ag:.3e}  R(N={n}) = {r:.6}  ratio(2·P_shot) = {ratio:.4}"
    );
}
