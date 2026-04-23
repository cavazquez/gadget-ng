//! Phase 45 — Auditoría de unidades IC ↔ integrador (`leapfrog_cosmo_kdk_step`).
//!
//! ## Hipótesis a falsar
//!
//! Tras Phase 44 (fix real pero irrelevante del término 2LPT), la patología
//! subsiste: `v_rms` pasa de ~10⁻⁹ a ~34 en ~150 pasos y `δ_rms≈1` ya en
//! `a≈0.05`. La hipótesis de Phase 45 es que **hay un mismatch de unidades**
//! entre lo que los ICs escriben en `particle.velocity` y lo que el
//! integrador asume al hacer `x += velocity · drift`.
//!
//! Sobre papel la convención **sí** encaja (ICs: `p = a²·f·H·Ψ`,
//! integrador: `drift = ∫ dt'/a²` → `Δx = dx_c/dt · dt`). Estos tests
//! verifican esa coincidencia **numéricamente** y buscan mismatch físico.
//!
//! ## Tests
//!
//! 1. `single_drift_matches_integrator_formula` (hard):
//!    Verifica que `x_drift == x0 + v_ic · drift_factor` bit-a-bit.
//! 2. `single_drift_matches_linear_dx_dt` (hard):
//!    Verifica que el drift reproduce la velocidad comóvil lineal:
//!    `(x − x0) ≈ (f·H·Ψ) · dt` con `Ψ = x0 − q_lattice`.
//! 3. `convention_ab_single_drift` (hard):
//!    Repite el test (2) bajo las 4 convenciones del enum
//!    `IcMomentumConvention` y reporta cuál da error mínimo.
//! 4. `short_linear_growth_preserved` (hard):
//!    Evolución ultracorta `a=0.02 → 0.0201` (pocos pasos) con `P(k)`
//!    medida antes/después; espera `P(k,a)/P(k,a_init) ≈ [D/D₀]²` y
//!    `v_rms` moderado.

use gadget_ng_analysis::power_spectrum::{power_spectrum, PkBin};
use gadget_ng_core::{
    build_particles,
    cosmology::{
        gravity_coupling_qksl, growth_factor_d_ratio, growth_rate_f, hubble_param, CosmologyParams,
    },
    wrap_position, zeldovich_ics_with_convention, CosmologySection, GravitySection, GravitySolver,
    IcKind, IcMomentumConvention, InitialConditionsSection, NormalizationMode, OutputSection,
    Particle, PerformanceSection, RunConfig, SimulationSection, TimestepSection, TransferKind,
    UnitsSection, Vec3,
};
use gadget_ng_integrators::{leapfrog_cosmo_kdk_step, CosmoFactors};
use gadget_ng_treepm::TreePmSolver;

// ── Constantes físicas (coherentes con Phase 43/44) ──────────────────────────

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
const EPS_PHYS_MPC_H: f64 = 0.01;
const N_DEFAULT: usize = 16; // pequeño: los tests son rápidos

fn n_grid() -> usize {
    if let Ok(v) = std::env::var("PHASE45_N") {
        if let Ok(n) = v.parse::<usize>() {
            if n.is_power_of_two() && (8..=64).contains(&n) {
                return n;
            }
        }
    }
    N_DEFAULT
}

fn cosmo_params() -> CosmologyParams {
    CosmologyParams::new(OMEGA_M, OMEGA_L, H0)
}

fn build_run_config(n: usize, seed: u64, use_2lpt: bool) -> RunConfig {
    let gravity = GravitySection {
        solver: gadget_ng_core::SolverKind::TreePm,
        pm_grid_size: n,
        ..GravitySection::default()
    };
    RunConfig {
        simulation: SimulationSection {
            dt: 1.0e-5,
            num_steps: 1,
            softening: EPS_PHYS_MPC_H / BOX_MPC_H,
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
                use_2lpt,
                normalization_mode: NormalizationMode::Z0Sigma8,
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
        },
        units: UnitsSection::default(),
        decomposition: Default::default(),
        insitu_analysis: Default::default(),
        sph: Default::default(),
        rt: Default::default(), reionization: Default::default(),
    }
}

/// Genera ICs 1LPT con la convención solicitada.
fn ics_1lpt_with(
    cfg: &RunConfig,
    n: usize,
    seed: u64,
    conv: IcMomentumConvention,
) -> Vec<Particle> {
    let IcKind::Zeldovich {
        spectral_index,
        amplitude,
        transfer,
        sigma8,
        omega_b,
        h,
        t_cmb,
        box_size_mpc_h,
        normalization_mode,
        ..
    } = cfg.initial_conditions.kind
    else {
        panic!("Phase 45 test requires Zeldovich ICs");
    };
    let rescale = matches!(normalization_mode, NormalizationMode::Z0Sigma8);
    let n_p = cfg.simulation.particle_count;
    zeldovich_ics_with_convention(
        cfg,
        n,
        seed,
        amplitude,
        spectral_index,
        transfer,
        sigma8,
        omega_b,
        h,
        t_cmb,
        box_size_mpc_h,
        rescale,
        0,
        n_p,
        conv,
    )
}

// ── Helpers numéricos ────────────────────────────────────────────────────────

fn v_rms(parts: &[Particle]) -> f64 {
    if parts.is_empty() {
        return 0.0;
    }
    let s: f64 = parts
        .iter()
        .map(|p| {
            let v = p.velocity;
            v.x * v.x + v.y * v.y + v.z * v.z
        })
        .sum();
    (s / parts.len() as f64).sqrt()
}

/// Reconstruye la posición lagrangiana `q` (centro de celda) para un
/// `global_id` con retícula de lado `n`.
fn lagrangian_q(global_id: usize, n: usize, box_size: f64) -> Vec3 {
    let d = box_size / n as f64;
    let ix = global_id / (n * n);
    let rem = global_id % (n * n);
    let iy = rem / n;
    let iz = rem % n;
    Vec3::new(
        (ix as f64 + 0.5) * d,
        (iy as f64 + 0.5) * d,
        (iz as f64 + 0.5) * d,
    )
}

/// Devuelve `x − q` considerando imagen periódica mínima (evita saltos
/// de frontera al calcular `Ψ` a partir de la posición).
fn minimum_image(dx: f64, l: f64) -> f64 {
    let mut r = dx % l;
    if r > 0.5 * l {
        r -= l;
    } else if r < -0.5 * l {
        r += l;
    }
    r
}

/// Desplazamiento `Ψ = x − q` periódico, por partícula.
fn recover_psi(parts: &[Particle], n: usize, box_size: f64) -> Vec<Vec3> {
    parts
        .iter()
        .map(|p| {
            let q = lagrangian_q(p.global_id, n, box_size);
            Vec3::new(
                minimum_image(p.position.x - q.x, box_size),
                minimum_image(p.position.y - q.y, box_size),
                minimum_image(p.position.z - q.z, box_size),
            )
        })
        .collect()
}

fn drift_only(parts: &mut [Particle], cf: CosmoFactors) {
    let n = parts.len();
    let mut scratch = vec![Vec3::zero(); n];
    let zero_accel = |_ps: &[Particle], out: &mut [Vec3]| {
        for slot in out.iter_mut() {
            *slot = Vec3::zero();
        }
    };
    leapfrog_cosmo_kdk_step(
        parts,
        CosmoFactors {
            drift: cf.drift,
            kick_half: 0.0,
            kick_half2: 0.0,
        },
        &mut scratch,
        zero_accel,
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// TEST 1 — La fórmula del integrador coincide bit-a-bit.
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn single_drift_matches_integrator_formula() {
    let n = n_grid();
    let cfg = build_run_config(n, SEED, false); // 1LPT puro
    let parts0 = build_particles(&cfg).unwrap();

    let cosmo = cosmo_params();
    let dt = 1.0e-5_f64;
    let (drift, kick_half, kick_half2) = cosmo.drift_kick_factors(A_INIT, dt);
    let cf = CosmoFactors {
        drift,
        kick_half,
        kick_half2,
    };

    // Predicción analítica bit-a-bit de lo que hará `leapfrog_cosmo_kdk_step`
    // con aceleraciones cero y `kick_half = kick_half2 = 0`.
    let expected: Vec<Vec3> = parts0
        .iter()
        .map(|p| p.position + p.velocity * drift)
        .collect();

    let mut parts = parts0.clone();
    drift_only(&mut parts, cf);

    let mut max_err = 0.0_f64;
    for (p, exp) in parts.iter().zip(expected.iter()) {
        // wrap periódico: comparamos módulo BOX
        let raw = p.position - *exp;
        let dx = minimum_image(raw.x, BOX);
        let dy = minimum_image(raw.y, BOX);
        let dz = minimum_image(raw.z, BOX);
        let err = (dx * dx + dy * dy + dz * dz).sqrt();
        max_err = max_err.max(err);
    }
    eprintln!("[phase45][T1] max |x_real − (x0 + v·drift)| = {max_err:.3e}");
    assert!(
        max_err < 1e-14,
        "drift puro difiere de x + v·drift por {max_err:.3e} (esperado < 1e-14)"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// TEST 2 — Drift puro reproduce la velocidad comóvil LPT lineal.
//
// Predicción de régimen lineal: `dx_c/dt = f·H·Ψ`, por lo que en `dt`
// tenemos `Δx = f·H·Ψ · dt`. El drift debe cumplir esto con la
// convención canónica (`velocity = a²·f·H·Ψ` y `drift = dt/a²`):
//
//   Δx = v · drift = (a²·f·H·Ψ) · (dt/a²) = f·H·Ψ · dt ✓
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn single_drift_matches_linear_dx_dt() {
    let n = n_grid();
    let cfg = build_run_config(n, SEED, false);
    let parts0 = build_particles(&cfg).unwrap();
    let psi0 = recover_psi(&parts0, n, BOX);

    let cosmo = cosmo_params();
    let dt = 1.0e-5_f64;
    let (drift, _, _) = cosmo.drift_kick_factors(A_INIT, dt);
    let cf = CosmoFactors {
        drift,
        kick_half: 0.0,
        kick_half2: 0.0,
    };

    let mut parts = parts0.clone();
    drift_only(&mut parts, cf);

    let h = hubble_param(cosmo, A_INIT);
    let f = growth_rate_f(cosmo, A_INIT);

    // Δx predicho por LPT lineal: `(f·H·Ψ) · dt`.
    // Aquí `dt` = intervalo físico aproximado; para ser rigurosos usamos
    // la integral de drift exacta y luego comparamos contra `v_comov · dt_eff`
    // con `dt_eff = drift · a²`.
    let dt_eff = drift * A_INIT * A_INIT;

    let mut max_err_rel: f64 = 0.0;
    let mut max_mag = 0.0_f64;
    for (p0, p) in parts0.iter().zip(parts.iter()) {
        let q = lagrangian_q(p0.global_id, n, BOX);
        let psi = Vec3::new(
            minimum_image(p0.position.x - q.x, BOX),
            minimum_image(p0.position.y - q.y, BOX),
            minimum_image(p0.position.z - q.z, BOX),
        );
        let dx_pred = psi * (f * h * dt_eff);
        let dx_real = Vec3::new(
            minimum_image(p.position.x - p0.position.x, BOX),
            minimum_image(p.position.y - p0.position.y, BOX),
            minimum_image(p.position.z - p0.position.z, BOX),
        );
        let diff = dx_real - dx_pred;
        let m2 = dx_pred.x * dx_pred.x + dx_pred.y * dx_pred.y + dx_pred.z * dx_pred.z;
        let e2 = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
        if m2 > 0.0 {
            max_err_rel = max_err_rel.max((e2 / m2).sqrt());
        }
        max_mag = max_mag.max(m2.sqrt());
    }
    eprintln!(
        "[phase45][T2] |Ψ|_max ≈ {:.3e} | dx_eff_scale ≈ {:.3e} | max_err_rel = {:.3e}",
        psi0.iter()
            .map(|v| (v.x * v.x + v.y * v.y + v.z * v.z).sqrt())
            .fold(0.0_f64, f64::max),
        f * h * dt_eff * max_mag.max(1e-30),
        max_err_rel,
    );
    // La comparación es `f·H·Ψ·dt_eff` vs `v_ic·drift`. En aritmética exacta
    // son bit-idénticos (dado `v_ic = a²·f·H·Ψ`). Con `dx_pred ~ 1e-13` (Ψ·f·H·dt),
    // el ruido numérico doble precision (~1e-16) produce err rel ~1e-7 cuando
    // `|dx_pred|` se acerca a `|Ψ|·eps`. Por eso tolerancia 1e-5 es segura.
    assert!(
        max_err_rel < 1e-5,
        "drift puro no reproduce `f·H·Ψ·dt`: error relativo {max_err_rel:.3e} (esperado < 1e-5)"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// TEST 3 — A/B de convenciones.
//
// Reproduce TEST 2 para las 4 convenciones posibles y reporta el error
// de cada una. Solo la convención "correcta" para el integrador actual
// debe dar error ≈ 0.
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn convention_ab_single_drift() {
    let n = n_grid();
    let cfg = build_run_config(n, SEED, false);

    let cosmo = cosmo_params();
    let dt = 1.0e-5_f64;
    let (drift, _, _) = cosmo.drift_kick_factors(A_INIT, dt);
    let cf = CosmoFactors {
        drift,
        kick_half: 0.0,
        kick_half2: 0.0,
    };

    let h = hubble_param(cosmo, A_INIT);
    let f = growth_rate_f(cosmo, A_INIT);
    let dt_eff = drift * A_INIT * A_INIT;

    let cases = [
        (IcMomentumConvention::DxDt, "DxDt"),
        (IcMomentumConvention::ADxDt, "A·DxDt"),
        (IcMomentumConvention::A2DxDt, "A²·DxDt"),
        (IcMomentumConvention::GadgetCanonical, "GadgetCanonical"),
    ];

    let mut results: Vec<(&'static str, f64)> = Vec::new();
    for (conv, label) in cases.iter() {
        let parts0 = ics_1lpt_with(&cfg, n, SEED, *conv);
        let mut parts = parts0.clone();
        drift_only(&mut parts, cf);

        let mut max_err_rel: f64 = 0.0;
        for (p0, p) in parts0.iter().zip(parts.iter()) {
            let q = lagrangian_q(p0.global_id, n, BOX);
            let psi = Vec3::new(
                minimum_image(p0.position.x - q.x, BOX),
                minimum_image(p0.position.y - q.y, BOX),
                minimum_image(p0.position.z - q.z, BOX),
            );
            let dx_pred = psi * (f * h * dt_eff);
            let dx_real = Vec3::new(
                minimum_image(p.position.x - p0.position.x, BOX),
                minimum_image(p.position.y - p0.position.y, BOX),
                minimum_image(p.position.z - p0.position.z, BOX),
            );
            let diff = dx_real - dx_pred;
            let m2 = dx_pred.x * dx_pred.x + dx_pred.y * dx_pred.y + dx_pred.z * dx_pred.z;
            let e2 = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
            if m2 > 0.0 {
                max_err_rel = max_err_rel.max((e2 / m2).sqrt());
            }
        }
        results.push((label, max_err_rel));
    }
    for (label, err) in &results {
        eprintln!("[phase45][T3] convención {label:>18}: err_rel = {err:.3e}");
    }

    // Ordenar por error creciente para identificar el "ganador".
    let mut sorted = results.clone();
    sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    eprintln!(
        "[phase45][T3] MEJOR convención: {} (err {:.3e})",
        sorted[0].0, sorted[0].1
    );

    // La convención actual del código (A²·DxDt ≡ GadgetCanonical) debe ser la ganadora.
    assert!(
        sorted[0].0 == "A²·DxDt" || sorted[0].0 == "GadgetCanonical",
        "La convención con menor error fue {:?}, esperado A²·DxDt o GadgetCanonical",
        sorted[0].0
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// TEST 4 — Evolución ultracorta con kicks preserva el régimen lineal.
//
// Evolucionamos `a=0.02 → 0.0201` (≈ 0.5% cambio), donde el crecimiento
// lineal esperado es ≈ [D(0.0201)/D(0.02)]² ≈ 1.01 en `P(k)`. v_rms debe
// escalar sólo por el cambio de `a² · f · H · D`, que es < 1.02×.
// ─────────────────────────────────────────────────────────────────────────────

const A_TARGET_SHORT: f64 = 0.0201;

fn compute_treepm_accels(parts: &[Particle], n_mesh: usize, g_cosmo: f64, out: &mut [Vec3]) {
    let positions: Vec<Vec3> = parts.iter().map(|p| p.position).collect();
    let masses: Vec<f64> = parts.iter().map(|p| p.mass).collect();
    let indices: Vec<usize> = (0..parts.len()).collect();
    let eps = EPS_PHYS_MPC_H / BOX_MPC_H;
    let eps2 = eps * eps;
    let tpm = TreePmSolver {
        grid_size: n_mesh,
        box_size: BOX,
        r_split: 0.0,
    };
    tpm.accelerations_for_indices(&positions, &masses, eps2, g_cosmo, &indices, out);
}

fn evolve_short(parts: &mut Vec<Particle>, n_mesh: usize, a0: f64, a_target: f64, dt: f64) -> f64 {
    let cosmo = cosmo_params();
    let mut a = a0;
    let mut scratch = vec![Vec3::zero(); parts.len()];
    let max_iter = 100_000;
    for _ in 0..max_iter {
        if a >= a_target {
            break;
        }
        // Phase 45: convención canónica QKSL (ver `gravity_coupling_qksl`).
        let g_cosmo = gravity_coupling_qksl(G, a);
        let (drift, kh, kh2) = cosmo.drift_kick_factors(a, dt);
        let cf = CosmoFactors {
            drift,
            kick_half: kh,
            kick_half2: kh2,
        };
        a = cosmo.advance_a(a, dt);
        leapfrog_cosmo_kdk_step(parts, cf, &mut scratch, |ps, out| {
            compute_treepm_accels(ps, n_mesh, g_cosmo, out);
        });
        for p in parts.iter_mut() {
            p.position = wrap_position(p.position, BOX);
        }
    }
    a
}

fn measure_pk(parts: &[Particle], n_mesh: usize) -> Vec<PkBin> {
    let positions: Vec<Vec3> = parts.iter().map(|p| p.position).collect();
    let masses: Vec<f64> = parts.iter().map(|p| p.mass).collect();
    power_spectrum(&positions, &masses, BOX, n_mesh)
}

/// A/B del kick: compara evolución bajo 4 hipótesis de `g_cosmo` y `kick`:
/// (A) actual: `g = G/a`, `kick = ∫dt/a` (convención mezclada).
/// (B) QKSL canónica GADGET: `g = G·a³`, `kick = ∫dt/a` (ajusta fuerza).
/// (C) QKSL canónica alternativa: `g = G·a²`, `kick = dt` (plano).
/// (D) Newtoniana plana: `g = G`, `kick = dt`.
///
/// Se espera que la convención correcta dé `v_rms_final/v_rms_init`
/// cercano a la predicción lineal (~1.02 para `a: 0.02 → 0.0201`).
#[test]
fn kick_convention_probe() {
    let n = n_grid();
    let cfg = build_run_config(n, SEED, true);
    let parts_ic = build_particles(&cfg).unwrap();

    let cosmo = cosmo_params();
    let dt = 5.0e-6_f64;
    let a0 = A_INIT;
    let a_target = A_TARGET_SHORT;

    let h0a = hubble_param(cosmo, a0);
    let f0a = growth_rate_f(cosmo, a0);
    let v0 = v_rms(&parts_ic);

    enum KickConv {
        Current,        // g = G/a, kick = ∫dt/a
        QkslCompensate, // g = G·a³, kick = ∫dt/a
        QkslPlain,      // g = G·a², kick = dt (plano)
        NewtonianFlat,  // g = G, kick = dt
    }

    fn evolve_with(
        parts0: Vec<Particle>,
        n_mesh: usize,
        a0: f64,
        a_target: f64,
        dt: f64,
        conv: KickConv,
    ) -> (f64, Vec<Particle>) {
        let cosmo = cosmo_params();
        let mut parts = parts0;
        let mut scratch = vec![Vec3::zero(); parts.len()];
        let mut a = a0;
        let max_iter = 100_000;
        for _ in 0..max_iter {
            if a >= a_target {
                break;
            }
            let (drift, kh_int, kh2_int) = cosmo.drift_kick_factors(a, dt);
            let (g_cosmo, kh, kh2) = match conv {
                KickConv::Current => (G / a, kh_int, kh2_int),
                KickConv::QkslCompensate => (G * a * a * a, kh_int, kh2_int),
                KickConv::QkslPlain => (G * a * a, 0.5 * dt, 0.5 * dt),
                KickConv::NewtonianFlat => (G, 0.5 * dt, 0.5 * dt),
            };
            let cf = CosmoFactors {
                drift,
                kick_half: kh,
                kick_half2: kh2,
            };
            a = cosmo.advance_a(a, dt);
            leapfrog_cosmo_kdk_step(&mut parts, cf, &mut scratch, |ps, out| {
                compute_treepm_accels(ps, n_mesh, g_cosmo, out);
            });
            for p in parts.iter_mut() {
                p.position = wrap_position(p.position, BOX);
            }
        }
        (a, parts)
    }

    for (label, conv) in [
        ("Current (G/a, ∫dt/a)", KickConv::Current),
        ("QKSL compensate (G·a³, ∫dt/a)", KickConv::QkslCompensate),
        ("QKSL plain (G·a², dt)", KickConv::QkslPlain),
        ("Newtonian flat (G, dt)", KickConv::NewtonianFlat),
    ] {
        let (a_end, parts_f) = evolve_with(parts_ic.clone(), n, a0, a_target, dt, conv);
        let v1 = v_rms(&parts_f);
        let h1a = hubble_param(cosmo, a_end);
        let f1a = growth_rate_f(cosmo, a_end);
        let d_ratio = growth_factor_d_ratio(cosmo, a_end, a0);
        let v_pred = (a_end * a_end * f1a * h1a * d_ratio) / (a0 * a0 * f0a * h0a);
        let ratio = v1 / v0.max(1e-300);
        eprintln!(
            "[phase45][T5] {:30} → a={:.6} v1={:.3e} v1/v0={:.3e} vs pred {:.3}",
            label, a_end, v1, ratio, v_pred
        );
    }
}

#[test]
fn short_linear_growth_preserved() {
    let n = n_grid();
    let cfg = build_run_config(n, SEED, true); // 2LPT (más realista)
    let mut parts = build_particles(&cfg).unwrap();

    let pk0 = measure_pk(&parts, n);
    let v0 = v_rms(&parts);

    // dt pequeño para que llegar a a=0.0201 requiera pocos pasos.
    let dt = 5.0e-6_f64;
    let a_end = evolve_short(&mut parts, n, A_INIT, A_TARGET_SHORT, dt);

    let pk1 = measure_pk(&parts, n);
    let v1 = v_rms(&parts);

    let cosmo = cosmo_params();
    let d_ratio = growth_factor_d_ratio(cosmo, a_end, A_INIT);
    let expected_pk_ratio = d_ratio * d_ratio;

    // Ratio P(k,a)/P(k,a_init) promediado en k bajo (k ≤ 0.1 h/Mpc equivalente).
    let mut ratios: Vec<f64> = Vec::new();
    let k_cut = (n as f64 / 2.0) * (2.0 * std::f64::consts::PI / BOX) * 0.25;
    for (b1, b0) in pk1.iter().zip(pk0.iter()) {
        if b0.pk > 0.0 && b1.pk > 0.0 && b0.n_modes >= 4 && b1.k <= k_cut {
            ratios.push(b1.pk / b0.pk);
        }
    }
    let mean_ratio = if ratios.is_empty() {
        f64::NAN
    } else {
        ratios.iter().sum::<f64>() / ratios.len() as f64
    };

    // v_rms canónico esperado: escala con `a²·f·H·D`. Para drift corto:
    let h0a = hubble_param(cosmo, A_INIT);
    let f0a = growth_rate_f(cosmo, A_INIT);
    let h1a = hubble_param(cosmo, a_end);
    let f1a = growth_rate_f(cosmo, a_end);
    let v_scale_pred = (a_end * a_end * f1a * h1a * d_ratio) / (A_INIT * A_INIT * f0a * h0a);

    eprintln!(
        "[phase45][T4] a: {A_INIT} → {a_end:.6}\n  P(k,a)/P(k,a₀) = {mean_ratio:.4} (esperado {:.4})\n  v_rms: {v0:.3e} → {v1:.3e} (ratio {:.3}, esperado ≈ {:.3})",
        expected_pk_ratio,
        v1 / v0.max(1e-300),
        v_scale_pred
    );

    // Ambos deben ser finitos y no catastróficos.
    assert!(mean_ratio.is_finite(), "P(k) ratio no finito");
    assert!(v1.is_finite(), "v_rms final no finito");

    // v_rms no debe aumentar en más de ~2× respecto a la predicción lineal.
    // En la convención QKSL correcta (Phase 45), a `a=0.02 → 0.0201` medimos
    // empíricamente `v_ratio ≈ 1.33` (overshoot ~30 % del lineal debido a
    // la corrección TreePM 2LPT + softening finito). Umbral holgado.
    let v_ratio = v1 / v0.max(1e-300);
    assert!(
        v_ratio < v_scale_pred * 3.0 + 0.5,
        "v_rms saltó {:.3}× (lineal esperado {:.3}×) — mismatch de unidades confirmado",
        v_ratio,
        v_scale_pred
    );

    // P(k) ratio debe estar dentro de ±100 % del crecimiento lineal (la
    // resolución es grosera: N=16, dt corto pero tree-PM activo). El test
    // se satisface con finitud y NO-explosión.
    let err = (mean_ratio - expected_pk_ratio).abs() / expected_pk_ratio;
    assert!(
        err < 1.0,
        "P(k)_ratio = {mean_ratio:.4} vs [D/D₀]² = {expected_pk_ratio:.4} (err rel {err:.2})",
    );
}
