//! Phase 55 — Comparación espectro de masas FoF vs HMF analítica (PS/ST)
//!
//! Evoluciona a z=0 con G consistente y compara la distribución de masas de
//! halos FoF con la función de masa de halos de Sheth-Tormen y Press-Schechter.
//!
//! ## Matriz
//!
//! `N ∈ {64, 128, 256}`, BOX = 300 Mpc/h, ZA + Z0Sigma8, `seed = 42`,
//! timestep adaptativo, `auto_g = true`, evolución desde `a=0.02` hasta `a=1.0`.
//!
//! Correr con: `cargo test -p gadget-ng-physics --release --test phase55_fof_vs_hmf -- --test-threads=1 --nocapture`

use gadget_ng_analysis::{
    analyse, mass_function_table, AnalysisParams, HmfBin, HmfParams, RHO_CRIT_H2,
};
use gadget_ng_core::{
    build_particles,
    cosmology::{adaptive_dt_cosmo, g_code_consistent, gravity_coupling_qksl, CosmologyParams},
    wrap_position, CosmologySection, GravitySection, GravitySolver, IcKind,
    InitialConditionsSection, NormalizationMode, OutputSection, PerformanceSection, RunConfig,
    SimulationSection, TimestepSection, TransferKind, UnitsSection, Vec3,
};
use gadget_ng_integrators::{leapfrog_cosmo_kdk_step, CosmoFactors};
use gadget_ng_pm::PmSolver;
use serde_json::json;
use std::fs;
use std::path::PathBuf;
use std::sync::OnceLock;

// ── Constantes ────────────────────────────────────────────────────────────────

const BOX: f64 = 1.0;
const BOX_MPC_H: f64 = 300.0;
const OMEGA_M: f64 = 0.315;
const OMEGA_L: f64 = 0.685;
const OMEGA_B: f64 = 0.049;
const H0: f64 = 0.1;
const H_DIMLESS: f64 = 0.674;
const T_CMB: f64 = 2.7255;
const N_S: f64 = 0.965;
const SIGMA8: f64 = 0.811;

const A_INIT: f64 = 0.02;
const A_FINAL: f64 = 1.0;
const N_VALUES: [usize; 3] = [64, 128, 256];
const SEED: u64 = 42;

const ETA_GRAV: f64 = 0.1;
const ALPHA_H: f64 = 0.01;
const DT_MAX: f64 = 0.05;

const FOF_B: f64 = 0.2;
const FOF_MIN_PART: usize = 20;

// ── Helpers ───────────────────────────────────────────────────────────────────

fn cosmo_params() -> CosmologyParams {
    CosmologyParams::new(OMEGA_M, OMEGA_L, H0)
}

fn g_consistent() -> f64 {
    g_code_consistent(OMEGA_M, H0)
}

/// Masa de partícula en M_sun/h (unidades físicas)
fn m_part_msun_h(n: usize) -> f64 {
    let rho_bar_m = OMEGA_M * RHO_CRIT_H2;
    rho_bar_m * BOX_MPC_H.powi(3) / (n * n * n) as f64
}

fn build_run_config(n: usize) -> RunConfig {
    RunConfig {
        simulation: SimulationSection {
            dt: 4.0e-4,
            num_steps: 1,
            softening: 0.01,
            physical_softening: false,
            gravitational_constant: g_consistent(),
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
            auto_g: true,
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

// ── Evolución PM adaptativa ───────────────────────────────────────────────────

fn evolve_pm_to_a_adaptive(
    parts: &mut Vec<gadget_ng_core::Particle>,
    n_mesh: usize,
    a_start: f64,
    a_target: f64,
) -> f64 {
    if a_start >= a_target {
        return a_start;
    }
    let cosmo = cosmo_params();
    let g_code = g_consistent();
    let softening = BOX / (n_mesh as f64 * 20.0);
    let pm = PmSolver {
        grid_size: n_mesh,
        box_size: BOX,
    };
    let n = parts.len();
    let mut scratch = vec![Vec3::zero(); n];
    let mut a = a_start;

    // Pre-compute initial forces
    {
        let pos: Vec<Vec3> = parts.iter().map(|p| p.position).collect();
        let m: Vec<f64> = parts.iter().map(|p| p.mass).collect();
        let idx: Vec<usize> = (0..n).collect();
        let g_cosmo = gravity_coupling_qksl(g_code, a);
        pm.accelerations_for_indices(&pos, &m, 0.0, g_cosmo, &idx, &mut scratch);
    }

    let max_iter = 300_000usize;
    let mut step = 0usize;
    for _ in 0..max_iter {
        if a >= a_target {
            break;
        }
        let acc_max = scratch
            .iter()
            .map(|v| (v.x * v.x + v.y * v.y + v.z * v.z).sqrt())
            .fold(0.0_f64, f64::max);
        let dt =
            adaptive_dt_cosmo(cosmo, a, acc_max, softening, ETA_GRAV, ALPHA_H, DT_MAX).max(1e-8);

        let g_cosmo = gravity_coupling_qksl(g_code, a);
        let (drift, kick_half, kick_half2) = cosmo.drift_kick_factors(a, dt);
        let cf = CosmoFactors {
            drift,
            kick_half,
            kick_half2,
        };
        a = cosmo.advance_a(a, dt);
        step += 1;

        leapfrog_cosmo_kdk_step(parts, cf, &mut scratch, |ps, acc| {
            let pos: Vec<Vec3> = ps.iter().map(|p| p.position).collect();
            let m: Vec<f64> = ps.iter().map(|p| p.mass).collect();
            let idx: Vec<usize> = (0..ps.len()).collect();
            pm.accelerations_for_indices(&pos, &m, 0.0, g_cosmo, &idx, acc)
        });
        for p in parts.iter_mut() {
            p.position = wrap_position(p.position, BOX);
        }
        if step % 500 == 0 {
            eprintln!("[phase55] N={n_mesh} step={step} a={a:.4}");
        }
    }
    eprintln!("[phase55] N={n_mesh} completado en {step} pasos, a_final={a:.5}");
    a
}

// ── Resultado por N ───────────────────────────────────────────────────────────

struct SimResult55 {
    n: usize,
    a_final: f64,
    /// Masas de halos FoF en M_sun/h
    halo_masses_msun_h: Vec<f64>,
    /// v_rms de todas las partículas en unidades internas
    v_rms: f64,
    /// Masa mínima resoluble (20 partículas × m_part)
    m_min_resoluble: f64,
}

impl SimResult55 {
    fn to_json(&self) -> serde_json::Value {
        json!({
            "n": self.n,
            "a_final": self.a_final,
            "n_halos": self.halo_masses_msun_h.len(),
            "halo_masses_msun_h": self.halo_masses_msun_h,
            "v_rms": self.v_rms,
            "m_min_resoluble_msun_h": self.m_min_resoluble,
        })
    }
}

// ── Simulación ────────────────────────────────────────────────────────────────

fn run_simulation_n(n: usize) -> SimResult55 {
    let t0 = std::time::Instant::now();
    eprintln!(
        "[phase55] Iniciando N={n}³  BOX={BOX_MPC_H} Mpc/h  G_consistent={:.4e}",
        g_consistent()
    );
    let cfg = build_run_config(n);
    let mut parts = build_particles(&cfg).expect("build_particles falló");
    let a_final = evolve_pm_to_a_adaptive(&mut parts, n, A_INIT, A_FINAL);

    // v_rms en unidades internas
    let n_total = parts.len();
    let v_rms = {
        let sq_sum: f64 = parts
            .iter()
            .map(|p| {
                p.velocity.x * p.velocity.x
                    + p.velocity.y * p.velocity.y
                    + p.velocity.z * p.velocity.z
            })
            .sum();
        (sq_sum / n_total as f64).sqrt()
    };

    // FoF en unidades internas — rho_crit=0 para usar r_max como r_vir
    let analysis = analyse(
        &parts,
        &AnalysisParams {
            box_size: BOX,
            b: FOF_B,
            min_particles: FOF_MIN_PART,
            rho_crit: 0.0,
            pk_mesh: 32,
        },
    );

    let mp = m_part_msun_h(n);
    // FofHalo.mass es la suma de p.mass; cada p.mass = 1/N_total en ICs
    // → masa en M_sun/h = h.n_particles * mp
    let halo_masses_physical: Vec<f64> = analysis
        .halos
        .iter()
        .map(|h| h.n_particles as f64 * mp)
        .collect();
    eprintln!(
        "[phase55] N={n} halos={} v_rms={:.4e} tiempo={:.1}s",
        halo_masses_physical.len(),
        v_rms,
        t0.elapsed().as_secs_f64()
    );

    SimResult55 {
        n,
        a_final,
        halo_masses_msun_h: halo_masses_physical,
        v_rms,
        m_min_resoluble: FOF_MIN_PART as f64 * mp,
    }
}

// ── Matriz global con OnceLock ────────────────────────────────────────────────

fn run_full_matrix() -> Vec<SimResult55> {
    let skip_n256 = std::env::var("PHASE55_SKIP_N256")
        .map(|v| v == "1")
        .unwrap_or(false);
    let skip_n128 = std::env::var("PHASE55_SKIP_N128")
        .map(|v| v == "1")
        .unwrap_or(false);
    let mut all = Vec::new();
    for &n in N_VALUES.iter() {
        if n == 256 && skip_n256 {
            eprintln!("[phase55] N=256 saltado por PHASE55_SKIP_N256=1");
            continue;
        }
        if n == 128 && skip_n128 {
            eprintln!("[phase55] N=128 saltado por PHASE55_SKIP_N128=1");
            continue;
        }
        all.push(run_simulation_n(n));
    }
    all
}

fn matrix() -> &'static [SimResult55] {
    static CELL: OnceLock<Vec<SimResult55>> = OnceLock::new();
    CELL.get_or_init(|| {
        let m = run_full_matrix();
        dump_results(&m);
        m
    })
}

fn dump_results(sims: &[SimResult55]) {
    use std::sync::atomic::{AtomicBool, Ordering};
    static DUMPED: AtomicBool = AtomicBool::new(false);
    if DUMPED.swap(true, Ordering::SeqCst) {
        return;
    }
    let dir = phase55_dir();
    let all: Vec<_> = sims.iter().map(|s| s.to_json()).collect();
    let txt = serde_json::to_string_pretty(&json!({
        "box_mpc_h": BOX_MPC_H,
        "sigma8": SIGMA8,
        "omega_m": OMEGA_M,
        "fof_b": FOF_B,
        "simulations": all,
    }))
    .unwrap_or_default();
    let _ = fs::write(dir.join("fof_results.json"), txt);
}

fn phase55_dir() -> PathBuf {
    let mut d = PathBuf::from(std::env::var("CARGO_TARGET_DIR").unwrap_or_else(|_| {
        let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        p.pop();
        p.pop();
        p.push("target");
        p.to_string_lossy().to_string()
    }));
    d.push("phase55");
    let _ = fs::create_dir_all(&d);
    d
}

fn find_sim(n: usize) -> Option<&'static SimResult55> {
    matrix().iter().find(|r| r.n == n)
}

fn hmf_params() -> HmfParams {
    HmfParams {
        omega_m: OMEGA_M,
        omega_lambda: OMEGA_L,
        omega_b: OMEGA_B,
        h: H_DIMLESS,
        sigma8: SIGMA8,
        n_s: N_S,
        t_cmb: T_CMB,
    }
}

/// Densidad analítica ST para un rango de masas → dn/dlnM en h³/Mpc³
fn hmf_st_density(m_min: f64, m_max: f64, n_bins: usize) -> Vec<HmfBin> {
    if m_min >= m_max || n_bins == 0 {
        return Vec::new();
    }
    mass_function_table(&hmf_params(), m_min, m_max, n_bins, 0.0)
}

/// Histograma FoF dn/dlnM en h³/Mpc³ para las masas dadas
fn fof_dn_dlnm(masses: &[f64], m_min: f64, m_max: f64, n_bins: usize) -> Vec<(f64, f64)> {
    if masses.is_empty() || m_min >= m_max || n_bins == 0 {
        return Vec::new();
    }
    let ln_m_min = m_min.ln();
    let ln_m_max = m_max.ln();
    let d_ln_m = (ln_m_max - ln_m_min) / n_bins as f64;
    let vol = BOX_MPC_H.powi(3);
    let mut counts = vec![0usize; n_bins];
    for &m in masses {
        if m < m_min || m >= m_max {
            continue;
        }
        let bin = ((m.ln() - ln_m_min) / d_ln_m) as usize;
        if bin < n_bins {
            counts[bin] += 1;
        }
    }
    (0..n_bins)
        .map(|i| {
            let ln_m_center = ln_m_min + (i as f64 + 0.5) * d_ln_m;
            let m_center = ln_m_center.exp();
            let dn_dlnm = counts[i] as f64 / (vol * d_ln_m);
            (m_center, dn_dlnm)
        })
        .collect()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

/// La simulación N=64 llega a a=1.0 sin explosión (v_rms < 50 en unidades internas)
#[test]
#[ignore = "lento: cargo test -p gadget-ng-physics --release -- --include-ignored"]
fn phase55_evolution_stable_n64() {
    let n = 64;
    let sim = find_sim(n).expect("N=64 no encontrado en la matriz");
    eprintln!(
        "[phase55] N={n} a_final={:.5} v_rms={:.4e}",
        sim.a_final, sim.v_rms
    );
    assert!(
        (sim.a_final - A_FINAL).abs() < 0.01,
        "N={n}: a_final={:.5} demasiado lejos de A_FINAL={A_FINAL}",
        sim.a_final
    );
    assert!(
        sim.v_rms < 50.0,
        "N={n}: v_rms={:.4e} — posible explosión numérica",
        sim.v_rms
    );
}

/// FoF encuentra ≥ 1 halo para N=64 a z=0 en BOX=300 Mpc/h
#[test]
#[ignore = "lento: cargo test -p gadget-ng-physics --release -- --include-ignored"]
fn phase55_halos_found_n64() {
    let n = 64;
    let sim = find_sim(n).expect("N=64 no encontrado");
    let n_halos = sim.halo_masses_msun_h.len();
    eprintln!(
        "[phase55] N={n} halos={n_halos}  m_min={:.2e}  m_max={:.2e}  [M_sun/h]",
        sim.halo_masses_msun_h
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min),
        sim.halo_masses_msun_h
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max),
    );
    assert!(
        n_halos >= 1,
        "N={n}: FoF no encontró ningún halo — simulación puede no estar evolucionando"
    );
}

/// FoF encuentra ≥ 20 halos para N=128
#[test]
#[ignore = "lento: cargo test -p gadget-ng-physics --release -- --include-ignored"]
fn phase55_halos_found_n128() {
    let n = 128;
    if find_sim(n).is_none() {
        eprintln!("[phase55] N=128 no disponible, saltando");
        return;
    }
    let sim = find_sim(n).unwrap();
    let n_halos = sim.halo_masses_msun_h.len();
    eprintln!("[phase55] N={n} halos={n_halos}");
    assert!(
        n_halos >= 20,
        "N={n}: sólo {n_halos} halos — se esperan ≥20 en BOX=300 Mpc/h"
    );
}

/// FoF encuentra ≥ 100 halos para N=256
#[test]
#[ignore = "lento: cargo test -p gadget-ng-physics --release -- --include-ignored"]
fn phase55_halos_found_n256() {
    let n = 256;
    if find_sim(n).is_none() {
        eprintln!("[phase55] N=256 no disponible (PHASE55_SKIP_N256=1), saltando");
        return;
    }
    let sim = find_sim(n).unwrap();
    let n_halos = sim.halo_masses_msun_h.len();
    eprintln!("[phase55] N={n} halos={n_halos}");
    assert!(
        n_halos >= 100,
        "N={n}: sólo {n_halos} halos — se esperan ≥100 en BOX=300 Mpc/h"
    );
}

/// Ratio dn/dlnM(FoF) / dn/dlnM(ST) ∈ [0.05, 20] para N=128
#[test]
#[ignore = "lento: cargo test -p gadget-ng-physics --release -- --include-ignored"]
fn phase55_fof_vs_hmf_ratio_n128() {
    let n = 128;
    if find_sim(n).is_none() {
        eprintln!("[phase55] N=128 no disponible, saltando");
        return;
    }
    let sim = find_sim(n).unwrap();
    let masses = &sim.halo_masses_msun_h;
    if masses.is_empty() {
        eprintln!("[phase55] N=128 sin halos — saltando comparación HMF");
        return;
    }

    let m_min = sim.m_min_resoluble;
    let m_max = masses.iter().cloned().fold(f64::NEG_INFINITY, f64::max) * 2.0;
    let n_bins = 8usize;
    let fof_hist = fof_dn_dlnm(masses, m_min, m_max, n_bins);
    let hmf_table = hmf_st_density(m_min, m_max, n_bins);

    eprintln!("[phase55] N={n} comparación FoF vs ST:");
    let mut ratios_ok = 0;
    let mut ratios_total = 0;
    for (i, (fof_m, fof_dn)) in fof_hist.iter().enumerate() {
        if i >= hmf_table.len() {
            break;
        }
        let st_dn = hmf_table[i].n_st;
        if st_dn <= 0.0 || *fof_dn <= 0.0 {
            continue;
        }
        let ratio = fof_dn / st_dn;
        eprintln!(
            "  M={:.2e} M_sun/h: FoF={:.3e}  ST={:.3e}  ratio={:.3}",
            fof_m, fof_dn, st_dn, ratio
        );
        ratios_total += 1;
        if ratio >= 0.05 && ratio <= 20.0 {
            ratios_ok += 1;
        }
    }
    eprintln!("[phase55] Bins con ratio dentro de [0.05, 20]: {ratios_ok}/{ratios_total}");
    if ratios_total > 0 {
        assert!(
            ratios_ok * 2 >= ratios_total,
            "N={n}: sólo {ratios_ok}/{ratios_total} bins tienen ratio FoF/ST ∈ [0.05, 20]"
        );
    }
}

/// Halos de N=256 tienen masas mínimas más bajas que N=64 (convergencia de masa mínima)
#[test]
#[ignore = "lento: cargo test -p gadget-ng-physics --release -- --include-ignored"]
fn phase55_mass_function_convergence() {
    let n_low = 64;
    let n_high = 256;
    let sim_low = match find_sim(n_low) {
        Some(s) => s,
        None => {
            eprintln!("N=64 no disponible");
            return;
        }
    };
    let sim_high = match find_sim(n_high) {
        Some(s) => s,
        None => {
            eprintln!("[phase55] N=256 no disponible, saltando convergencia");
            return;
        }
    };

    if sim_low.halo_masses_msun_h.is_empty() || sim_high.halo_masses_msun_h.is_empty() {
        eprintln!("[phase55] Sin halos en N=64 o N=256, saltando");
        return;
    }

    let m_min_low: f64 = sim_low
        .halo_masses_msun_h
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);
    let m_min_high: f64 = sim_high
        .halo_masses_msun_h
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);
    eprintln!(
        "[phase55] Masa mínima: N={n_low}={:.3e}  N={n_high}={:.3e} [M_sun/h]",
        m_min_low, m_min_high
    );
    assert!(
        m_min_high < m_min_low,
        "N={n_high} debería tener masa mínima más baja ({:.3e}) que N={n_low} ({:.3e})",
        m_min_high,
        m_min_low
    );
}
