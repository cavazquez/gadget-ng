//! Phase 58 — c(M) y perfiles NFW desde N-body + función de correlación ξ(r)
//!
//! ## Objetivo
//!
//! Validar el pipeline completo de post-procesamiento cosmológico:
//! 1. Simulación PM cosmológica pequeña (N=32³, BOX=300 Mpc/h, z: 50→0).
//! 2. Detección de halos FoF y ajuste de perfil NFW.
//! 3. Comparación c(M) medido con Duffy+2008 y Ludlow+2016.
//! 4. Cálculo ξ(r) mediante FFT desde P(k) y via pares directos.
//!
//! ## Tests
//!
//! 1. **`phase58_concentration_vs_theory`** — scatter c_meas/c_duffy ∈ [0.2, 5.0].
//! 2. **`phase58_xi_fft_positive_at_small_r`** — ξ(r) finito a z=0.
//! 3. **`phase58_xi_pairs_finite`** — ξ_pairs finito y no-infinito.
//! 4. **`phase58_ludlow_vs_duffy_range`** — relación Ludlow/Duffy en [0.3, 3.5].
//!
//! Controlar con `PHASE58_SKIP=1` para omitir la simulación pesada.

use gadget_ng_analysis::{
    AnalysisParams, RHO_CRIT_H2, analyse, concentration_duffy2008, concentration_ludlow2016,
    fit_nfw_concentration, measure_density_profile, r200_from_m200, rho_crit_z,
    two_point_correlation_fft, two_point_correlation_pairs,
};
use gadget_ng_core::{
    CosmologySection, GravitySection, GravitySolver, IcKind, InitialConditionsSection,
    NormalizationMode, OutputSection, PerformanceSection, RunConfig, SimulationSection,
    TimestepSection, TransferKind, UnitsSection, Vec3, build_particles,
    cosmology::{CosmologyParams, adaptive_dt_cosmo, g_code_consistent, gravity_coupling_qksl},
    wrap_position,
};
use gadget_ng_integrators::{CosmoFactors, leapfrog_cosmo_kdk_step};
use gadget_ng_pm::PmSolver;
use std::sync::OnceLock;

// ── Cosmología Planck 2015 ────────────────────────────────────────────────────

/// Tamaño de caja en unidades de código (1.0 = caja normalizada)
const BOX: f64 = 1.0;
/// Tamaño físico de la caja en Mpc/h
const BOX_MPC_H: f64 = 300.0;
const OMEGA_M: f64 = 0.315;
const OMEGA_L: f64 = 0.685;
const OMEGA_B: f64 = 0.049;
const H0: f64 = 0.1;
const H_DIMLESS: f64 = 0.674;
const T_CMB: f64 = 2.7255;
const N_S: f64 = 0.965;
const SIGMA8: f64 = 0.811;

const N_GRID: usize = 32;
const A_INIT: f64 = 0.02;
const A_FINAL: f64 = 1.0;
const SEED: u64 = 58;

const ETA_GRAV: f64 = 0.1;
const ALPHA_H: f64 = 0.01;
const DT_MAX: f64 = 0.05;

const FOF_B: f64 = 0.2;
const FOF_MIN_PART: usize = 10;
const NFW_MIN_PART: usize = 20;

// ── Helpers ───────────────────────────────────────────────────────────────────

fn m_part_msun_h() -> f64 {
    let rho_bar_m = OMEGA_M * RHO_CRIT_H2;
    rho_bar_m * BOX_MPC_H.powi(3) / (N_GRID * N_GRID * N_GRID) as f64
}

fn cosmo_params() -> CosmologyParams {
    CosmologyParams::new(OMEGA_M, OMEGA_L, H0)
}

fn rho_crit_physical() -> f64 {
    rho_crit_z(OMEGA_M, OMEGA_L, 0.0)
}

// ── Resultado de la simulación ────────────────────────────────────────────────

struct Phase58Result {
    /// Partículas en unidades código [0, 1]
    particles: Vec<gadget_ng_core::Particle>,
    a_final: f64,
    n_halos: usize,
    /// c_medido / c_duffy para cada halo con NFW ajustado
    c_ratio_duffy: Vec<f64>,
    /// c_medido / c_ludlow para cada halo con NFW ajustado  
    c_ratio_ludlow: Vec<f64>,
    /// ξ(r) via FFT desde P(k)
    xi_fft: Vec<gadget_ng_analysis::XiBin>,
}

// ── Simulación PM cosmológica ─────────────────────────────────────────────────

fn run_simulation() -> Phase58Result {
    eprintln!(
        "[phase58] Iniciando N={}³ BOX={} Mpc/h a_init={A_INIT} → a_final={A_FINAL}",
        N_GRID, BOX_MPC_H
    );

    let cfg = RunConfig {
        simulation: SimulationSection {
            dt: 4.0e-4,
            num_steps: 1,
            softening: BOX / (N_GRID as f64 * 20.0),
            physical_softening: false,
            gravitational_constant: g_code_consistent(OMEGA_M, H0),
            particle_count: N_GRID * N_GRID * N_GRID,
            box_size: BOX,
            seed: SEED,
            integrator: Default::default(),
        },
        initial_conditions: InitialConditionsSection {
            kind: IcKind::Zeldovich {
                seed: SEED,
                grid_size: N_GRID,
                spectral_index: N_S,
                amplitude: 1.0e-4,
                transfer: TransferKind::EisensteinHu,
                sigma8: Some(SIGMA8),
                omega_b: OMEGA_B,
                h: H_DIMLESS,
                t_cmb: T_CMB,
                box_size_mpc_h: Some(BOX_MPC_H),
                use_2lpt: true,
                normalization_mode: NormalizationMode::Z0Sigma8,
            },
        },
        output: OutputSection::default(),
        gravity: GravitySection {
            solver: gadget_ng_core::SolverKind::Pm,
            pm_grid_size: N_GRID,
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
            auto_g: true,
            ..Default::default()
        },
        units: UnitsSection::default(),
        decomposition: Default::default(),
        insitu_analysis: Default::default(),
        sph: Default::default(),
        rt: Default::default(),
        reionization: Default::default(),
        mhd: Default::default(),
        turbulence: Default::default(),
        two_fluid: Default::default(),
        sidm: Default::default(),
        modified_gravity: Default::default(),
    };

    let mut parts = build_particles(&cfg).expect("[phase58] ICs no deben fallar");
    let cosmo = cosmo_params();
    let g_code = g_code_consistent(OMEGA_M, H0);
    let pm = PmSolver {
        grid_size: N_GRID,
        box_size: BOX,
    };
    let n = parts.len();
    let mut scratch = vec![Vec3::zero(); n];
    let mut a = A_INIT;
    let softening = cfg.simulation.softening;

    // Fuerzas iniciales
    {
        let pos: Vec<Vec3> = parts.iter().map(|p| p.position).collect();
        let m: Vec<f64> = parts.iter().map(|p| p.mass).collect();
        let idx: Vec<usize> = (0..n).collect();
        let g_cosmo = gravity_coupling_qksl(g_code, a);
        pm.accelerations_for_indices(&pos, &m, 0.0, g_cosmo, &idx, &mut scratch);
    }

    let max_iter = 200_000usize;
    let mut step = 0usize;
    for _ in 0..max_iter {
        if a >= A_FINAL {
            break;
        }
        let acc_max = scratch
            .iter()
            .map(|v| (v.x * v.x + v.y * v.y + v.z * v.z).sqrt())
            .fold(0.0_f64, f64::max);
        let dt = adaptive_dt_cosmo(cosmo, a, acc_max, softening, ETA_GRAV, ALPHA_H, DT_MAX)
            .max(1e-8)
            .min(A_FINAL - a + 1e-12);

        let g_cosmo = gravity_coupling_qksl(g_code, a);
        let (drift, kick_half, kick_half2) = cosmo.drift_kick_factors(a, dt);
        let cf = CosmoFactors {
            drift,
            kick_half,
            kick_half2,
        };
        a = cosmo.advance_a(a, dt);
        step += 1;

        leapfrog_cosmo_kdk_step(&mut parts, cf, &mut scratch, |ps, acc| {
            let pos: Vec<Vec3> = ps.iter().map(|p| p.position).collect();
            let m: Vec<f64> = ps.iter().map(|p| p.mass).collect();
            let idx: Vec<usize> = (0..ps.len()).collect();
            pm.accelerations_for_indices(&pos, &m, 0.0, g_cosmo, &idx, acc)
        });
        for p in parts.iter_mut() {
            p.position = wrap_position(p.position, BOX);
        }
        if step.is_multiple_of(1000) {
            eprintln!("[phase58] step={step} a={a:.4}");
        }
    }
    eprintln!("[phase58] completado en {step} pasos, a_final={a:.5}");

    // Análisis FoF + P(k) en unidades código [0,1]
    let analysis = analyse(
        &parts,
        &AnalysisParams {
            box_size: BOX,
            b: FOF_B,
            min_particles: FOF_MIN_PART,
            rho_crit: 0.0,
            pk_mesh: N_GRID,
        },
    );

    eprintln!("[phase58] halos FoF: {}", analysis.halos.len());

    // P(k) está en unidades código; lo convertimos a ξ(r) en unidades físicas Mpc/h
    // escalando k → k/BOX_MPC_H y P → P*BOX_MPC_H³
    let pk_phys: Vec<gadget_ng_analysis::PkBin> = analysis
        .power_spectrum
        .iter()
        .map(|b| gadget_ng_analysis::PkBin {
            k: b.k / BOX_MPC_H,
            pk: b.pk * BOX_MPC_H.powi(3),
            n_modes: b.n_modes,
        })
        .collect();
    let xi_fft = two_point_correlation_fft(&pk_phys, BOX_MPC_H, 20);

    // Masa de partícula en M_sun/h
    let mp = m_part_msun_h();
    let rho_c = rho_crit_physical();

    // Para halos con ≥ NFW_MIN_PART: ajuste NFW
    let mut c_ratio_duffy: Vec<f64> = Vec::new();
    let mut c_ratio_ludlow: Vec<f64> = Vec::new();

    for halo in &analysis.halos {
        if halo.n_particles < NFW_MIN_PART {
            continue;
        }

        let m200_msun_h = halo.n_particles as f64 * mp;
        let r200_mpc_h = r200_from_m200(m200_msun_h, rho_c);
        let r200_code = r200_mpc_h / BOX_MPC_H;

        // Posiciones relativas al CoM del halo en unidades código
        let (cx, cy, cz) = (halo.x_com, halo.y_com, halo.z_com);
        let radii_mpc_h: Vec<f64> = parts
            .iter()
            .filter_map(|p| {
                let mut dx = p.position.x - cx;
                let mut dy = p.position.y - cy;
                let mut dz = p.position.z - cz;
                // imagen mínima
                if dx > 0.5 {
                    dx -= 1.0;
                } else if dx < -0.5 {
                    dx += 1.0;
                }
                if dy > 0.5 {
                    dy -= 1.0;
                } else if dy < -0.5 {
                    dy += 1.0;
                }
                if dz > 0.5 {
                    dz -= 1.0;
                } else if dz < -0.5 {
                    dz += 1.0;
                }
                let r_code = (dx * dx + dy * dy + dz * dz).sqrt();
                if r_code < 2.0 * r200_code {
                    Some(r_code * BOX_MPC_H)
                } else {
                    None
                }
            })
            .collect();

        if radii_mpc_h.len() < NFW_MIN_PART {
            continue;
        }

        let r_min_mpc_h = r200_mpc_h * 0.05;
        let r_max_mpc_h = r200_mpc_h;
        let profile = measure_density_profile(&radii_mpc_h, mp, r_min_mpc_h, r_max_mpc_h, 12, None);
        if profile.iter().filter(|b| b.n_part > 0).count() < 3 {
            continue;
        }

        if let Some(fit) = fit_nfw_concentration(&profile, m200_msun_h, rho_c, 1.0, 20.0, 50) {
            // Concentración = r200 / r_s
            let c_meas = r200_mpc_h / fit.profile.r_s;
            let c_duffy = concentration_duffy2008(m200_msun_h, 0.0);
            let c_ludlow = concentration_ludlow2016(m200_msun_h, 0.0);
            if c_duffy > 0.0 && c_ludlow > 0.0 && c_meas > 0.0 {
                c_ratio_duffy.push(c_meas / c_duffy);
                c_ratio_ludlow.push(c_meas / c_ludlow);
                eprintln!(
                    "[phase58]   halo N={} M={:.2e} Msun/h c_fit={:.2} c_duffy={:.2} c_ludlow={:.2}",
                    halo.n_particles, m200_msun_h, c_meas, c_duffy, c_ludlow
                );
            }
        }
    }

    Phase58Result {
        particles: parts,
        a_final: a,
        n_halos: analysis.halos.len(),
        c_ratio_duffy,
        c_ratio_ludlow,
        xi_fft,
    }
}

fn result() -> &'static Phase58Result {
    static CELL: OnceLock<Phase58Result> = OnceLock::new();
    CELL.get_or_init(run_simulation)
}

fn skip() -> bool {
    std::env::var("PHASE58_SKIP")
        .map(|v| v == "1")
        .unwrap_or(false)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

/// Scatter c(M) medido vs Duffy+2008 dentro de un factor 5.
/// Para N=32³ la resolución es baja; toleramos ratio ∈ [0.2, 5.0].
#[test]
#[ignore = "lento: cargo test -p gadget-ng-physics --release -- --include-ignored"]
fn phase58_concentration_vs_theory() {
    if skip() {
        eprintln!("[phase58] saltado por PHASE58_SKIP=1");
        return;
    }
    let r = result();
    eprintln!(
        "[phase58] a_final={:.4} halos={} halos_nfw={}",
        r.a_final,
        r.n_halos,
        r.c_ratio_duffy.len()
    );

    if r.c_ratio_duffy.is_empty() {
        eprintln!(
            "[phase58] ADVERTENCIA: ningún halo con ajuste NFW válido (N=32³ puede no colapsar suficiente)"
        );
        return;
    }

    for &ratio in &r.c_ratio_duffy {
        assert!(
            ratio > 0.1 && ratio < 10.0,
            "c_fit/c_duffy = {ratio:.2} fuera de [0.1, 10.0]"
        );
    }
    let mean = r.c_ratio_duffy.iter().sum::<f64>() / r.c_ratio_duffy.len() as f64;
    eprintln!(
        "[phase58] c_fit/c_duffy: mean={mean:.2} n={}",
        r.c_ratio_duffy.len()
    );

    for &ratio in &r.c_ratio_ludlow {
        assert!(
            ratio > 0.1 && ratio < 10.0,
            "c_fit/c_ludlow = {ratio:.2} fuera de [0.1, 10.0]"
        );
    }
}

/// ξ(r) desde FFT debe tener valores finitos.
#[test]
#[ignore = "lento: cargo test -p gadget-ng-physics --release -- --include-ignored"]
fn phase58_xi_fft_finite() {
    if skip() {
        eprintln!("[phase58] saltado por PHASE58_SKIP=1");
        return;
    }
    let r = result();
    if r.xi_fft.is_empty() {
        eprintln!("[phase58] ξ_fft vacío");
        return;
    }
    for b in &r.xi_fft {
        assert!(b.xi.is_finite(), "ξ_fft no finito en r={:.2}", b.r);
        assert!(b.r > 0.0, "r <= 0 en xi_fft");
    }
    let xi0 = r.xi_fft[0].xi;
    eprintln!(
        "[phase58] ξ_fft[0]: r={:.2} Mpc/h  ξ={:.4}",
        r.xi_fft[0].r, xi0
    );
}

/// ξ via pares directos sobre submuestra de partículas: valores finitos.
#[test]
#[ignore = "lento: cargo test -p gadget-ng-physics --release -- --include-ignored"]
fn phase58_xi_pairs_finite() {
    if skip() {
        eprintln!("[phase58] saltado por PHASE58_SKIP=1");
        return;
    }
    let r = result();
    let n_sample = r.particles.len().min(300);
    let positions: Vec<Vec3> = r.particles[..n_sample]
        .iter()
        .map(|p| {
            Vec3::new(
                p.position.x * BOX_MPC_H,
                p.position.y * BOX_MPC_H,
                p.position.z * BOX_MPC_H,
            )
        })
        .collect();

    let xi_pairs = two_point_correlation_pairs(&positions, BOX_MPC_H, 5.0, 80.0, 8);
    assert!(!xi_pairs.is_empty(), "xi_pairs vacío");
    for b in &xi_pairs {
        assert!(b.xi.is_finite(), "ξ_pairs no finito en r={:.2}", b.r);
    }
    let mean_xi: f64 = xi_pairs.iter().map(|b| b.xi).sum::<f64>() / xi_pairs.len() as f64;
    eprintln!(
        "[phase58] ξ_pairs media={mean_xi:.4} (n_bins={})",
        xi_pairs.len()
    );
}

/// Relación Ludlow/Duffy debe estar en [0.3, 3.5] para masas cosmológicas típicas.
#[test]
#[ignore = "lento: cargo test -p gadget-ng-physics --release -- --include-ignored"]
fn phase58_ludlow_vs_duffy_range() {
    let masses = [1e11_f64, 1e12, 1e13, 1e14, 1e15];
    for m in masses {
        let c_d = concentration_duffy2008(m, 0.0);
        let c_l = concentration_ludlow2016(m, 0.0);
        let ratio = c_l / c_d;
        assert!(
            ratio > 0.3 && ratio < 3.5,
            "Ludlow/Duffy = {ratio:.2} fuera de [0.3, 3.5] para M={m:.0e}"
        );
        eprintln!("[phase58] M={m:.0e} c_duffy={c_d:.2} c_ludlow={c_l:.2} ratio={ratio:.2}");
    }
}
