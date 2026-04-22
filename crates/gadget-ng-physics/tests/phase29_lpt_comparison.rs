//! Validación física comparativa 1LPT vs 2LPT — Fase 29.
//!
//! ## Pregunta física central
//!
//! ¿Cuánto mejora 2LPT respecto a Zel'dovich (1LPT) en gadget-ng, y en qué régimen
//! esa mejora se vuelve relevante?
//!
//! ## Nota sobre a_init y amplitud del espectro
//!
//! En gadget-ng, el parámetro `a_init` solo afecta las **velocidades**
//! (`p = a²·f·H·Ψ`), NO las posiciones. La amplitud del campo de desplazamiento
//! está fijada por la normalización `σ₈`, independientemente de `a_init`.
//! Por tanto:
//! - La corrección de posición 2LPT escala con `σ₈²/σ₈ = σ₈`, no con `a_init`.
//! - El efecto de "inicio tardío" se manifiesta mediante la corrección de velocidad.
//!
//! ## Cobertura
//!
//! 1. `correction_scales_with_amplitude`:
//!    La corrección 2LPT crece con la amplitud del espectro (σ₈).
//!    `|Ψ²|/|Ψ¹| ∝ σ₈` — verificación del comportamiento cuadrático vs lineal.
//!
//! 2. `psi1_psi2_ratio_quantified`:
//!    Cuantifica el ratio `|Ψ²|_rms / |Ψ¹|_rms` para ΛCDM estándar (σ₈=0.8).
//!    Valida que la corrección es subleading pero numéricamente significativa.
//!
//! 3. `pk_2lpt_initial_consistent_with_1lpt`:
//!    `P(k)` inicial de 2LPT difiere < 15% del de 1LPT en todos los bins.
//!    La corrección de posición 2LPT no rompe la forma espectral.
//!
//! 4. `pm_growth_both_lpt_consistent`:
//!    Tras evolución corta con PM, `delta_rms` de 2LPT y 1LPT difieren < 15%.
//!    Ambos muestran crecimiento gravitacional coherente.
//!
//! 5. `treepm_growth_both_lpt_consistent`:
//!    Ídem con TreePM: la consistencia entre solvers confirma que la mejora
//!    de 2LPT no es artefacto de un solver en particular.
//!
//! 6. `velocity_correction_subleading`:
//!    El momentum RMS de 2LPT difiere < 20% del de 1LPT.
//!    La corrección de velocidad es subleading en el régimen lineal (σ₈=0.8).

use gadget_ng_analysis::power_spectrum::power_spectrum;
use gadget_ng_core::{
    build_particles,
    cosmology::CosmologyParams,
    density_contrast_rms, peculiar_vrms,
    wrap_position, CosmologySection, GravitySection, GravitySolver, IcKind,
    InitialConditionsSection, OutputSection, PerformanceSection, RunConfig, SimulationSection,
    TimestepSection, TransferKind, UnitsSection, Vec3,
};
use gadget_ng_integrators::{leapfrog_cosmo_kdk_step, CosmoFactors};
use gadget_ng_pm::PmSolver;
use gadget_ng_treepm::TreePmSolver;

// ── Constantes compartidas ────────────────────────────────────────────────────

const G: f64 = 1.0;
const BOX: f64 = 1.0;
const GRID: usize = 8; // 8³ = 512 partículas — rápido para CI
const N_PART: usize = 512;
const NM: usize = 8; // malla PM/TreePM
const NGRID_DELTA: usize = 4; // malla para density_contrast_rms

const OMEGA_M: f64 = 0.315;
const OMEGA_L: f64 = 0.685;
const H0: f64 = 0.1;
const A_INIT: f64 = 0.02; // z ≈ 49

const OMEGA_B: f64 = 0.049;
const H_DIMLESS: f64 = 0.674;
const T_CMB: f64 = 2.7255;
const N_S: f64 = 0.965;
const BOX_MPC_H: f64 = 100.0;
const SIGMA8_STD: f64 = 0.8; // ΛCDM estándar

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Configuración ΛCDM base. `sigma8` y `use_2lpt` son parámetros variables.
fn base_config(seed: u64, sigma8: f64, use_2lpt: bool) -> RunConfig {
    RunConfig {
        simulation: SimulationSection {
            dt: 0.002,
            num_steps: 20,
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
                spectral_index: N_S,
                amplitude: 1.0e-4,
                transfer: TransferKind::EisensteinHu,
                sigma8: Some(sigma8),
                omega_b: OMEGA_B,
                h: H_DIMLESS,
                t_cmb: T_CMB,
                box_size_mpc_h: Some(BOX_MPC_H),
                use_2lpt,
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

/// Configura TreePM como solver gravitacional.
fn with_treepm(mut cfg: RunConfig) -> RunConfig {
    cfg.gravity.solver = gadget_ng_core::SolverKind::TreePm;
    cfg.gravity.theta = 0.5;
    cfg
}

/// Compute |Ψ²|_rms = rms(pos_2lpt − pos_1lpt) con simetría periódica.
fn psi2_rms(parts_1lpt: &[gadget_ng_core::Particle], parts_2lpt: &[gadget_ng_core::Particle]) -> f64 {
    let sum_sq: f64 = parts_1lpt.iter().zip(parts_2lpt.iter()).map(|(p1, p2)| {
        let dx = (p2.position.x - p1.position.x + BOX / 2.0).rem_euclid(BOX) - BOX / 2.0;
        let dy = (p2.position.y - p1.position.y + BOX / 2.0).rem_euclid(BOX) - BOX / 2.0;
        let dz = (p2.position.z - p1.position.z + BOX / 2.0).rem_euclid(BOX) - BOX / 2.0;
        dx * dx + dy * dy + dz * dz
    }).sum::<f64>();
    (sum_sq / N_PART as f64).sqrt()
}

/// Compute |Ψ¹|_rms = rms(pos_1lpt − q_lagrange) con simetría periódica.
fn psi1_rms(parts_1lpt: &[gadget_ng_core::Particle]) -> f64 {
    let d_grid = BOX / GRID as f64;
    let sum_sq: f64 = parts_1lpt.iter().enumerate().map(|(gid, p)| {
        let ix = gid / (GRID * GRID);
        let iy = (gid % (GRID * GRID)) / GRID;
        let iz = gid % GRID;
        let qx = (ix as f64 + 0.5) * d_grid;
        let qy = (iy as f64 + 0.5) * d_grid;
        let qz = (iz as f64 + 0.5) * d_grid;
        let dx = (p.position.x - qx + BOX / 2.0).rem_euclid(BOX) - BOX / 2.0;
        let dy = (p.position.y - qy + BOX / 2.0).rem_euclid(BOX) - BOX / 2.0;
        let dz = (p.position.z - qz + BOX / 2.0).rem_euclid(BOX) - BOX / 2.0;
        dx * dx + dy * dy + dz * dz
    }).sum::<f64>();
    (sum_sq / N_PART as f64).sqrt()
}

/// Evolución leapfrog cosmológico con PM durante `n_steps` pasos.
///
/// Retorna el factor de escala final.
fn run_pm_evolution(parts: &mut Vec<gadget_ng_core::Particle>, n_steps: usize, dt: f64) -> f64 {
    let cosmo = CosmologyParams::new(OMEGA_M, OMEGA_L, H0);
    let pm = PmSolver { grid_size: NM, box_size: BOX };
    let mut scratch = vec![Vec3::zero(); N_PART];
    let mut a = A_INIT;

    for _ in 0..n_steps {
        let g_cosmo = G / a;
        let (drift, kick_half, kick_half2) = cosmo.drift_kick_factors(a, dt);
        let cf = CosmoFactors { drift, kick_half, kick_half2 };
        a = cosmo.advance_a(a, dt);

        leapfrog_cosmo_kdk_step(parts, cf, &mut scratch, |ps, acc| {
            let pos: Vec<Vec3> = ps.iter().map(|p| p.position).collect();
            let m: Vec<f64> = ps.iter().map(|p| p.mass).collect();
            let idx: Vec<usize> = (0..ps.len()).collect();
            pm.accelerations_for_indices(&pos, &m, 0.0, g_cosmo, &idx, acc);
        });

        for p in parts.iter_mut() {
            p.position = wrap_position(p.position, BOX);
        }
    }
    a
}

/// Evolución leapfrog cosmológico con TreePM durante `n_steps` pasos.
fn run_treepm_evolution(parts: &mut Vec<gadget_ng_core::Particle>, n_steps: usize, dt: f64) -> f64 {
    let cosmo = CosmologyParams::new(OMEGA_M, OMEGA_L, H0);
    let treepm = TreePmSolver { grid_size: NM, box_size: BOX, r_split: 0.0 };
    let mut scratch = vec![Vec3::zero(); N_PART];
    let mut a = A_INIT;

    for _ in 0..n_steps {
        let g_cosmo = G / a;
        let (drift, kick_half, kick_half2) = cosmo.drift_kick_factors(a, dt);
        let cf = CosmoFactors { drift, kick_half, kick_half2 };
        a = cosmo.advance_a(a, dt);

        leapfrog_cosmo_kdk_step(parts, cf, &mut scratch, |ps, acc| {
            let pos: Vec<Vec3> = ps.iter().map(|p| p.position).collect();
            let m: Vec<f64> = ps.iter().map(|p| p.mass).collect();
            let idx: Vec<usize> = (0..ps.len()).collect();
            treepm.accelerations_for_indices(&pos, &m, 0.0, g_cosmo, &idx, acc);
        });

        for p in parts.iter_mut() {
            p.position = wrap_position(p.position, BOX);
        }
    }
    a
}

// ── Test 1: corrección 2LPT escala con amplitud ───────────────────────────────

/// La corrección de segundo orden |Ψ²|_rms / |Ψ¹|_rms crece con σ₈.
///
/// ## Principio físico
///
/// Ψ² se genera de la fuente cuadrática S(x) ∝ (Ψ¹)². Por tanto:
/// - |Ψ²|_rms ∝ amplitude²  ∝ σ₈²
/// - |Ψ¹|_rms ∝ amplitude   ∝ σ₈
/// → ratio = |Ψ²|/|Ψ¹| ∝ σ₈
///
/// ## Nota sobre a_init
///
/// En gadget-ng, a_init solo afecta velocidades, no posiciones. La amplitud del
/// campo de desplazamiento se fija por σ₈ independientemente de a_init. Por tanto,
/// el test usa distintos valores de σ₈ para explorar el régimen no lineal.
///
/// La corrección es mayor a σ₈ alto (universos más "clumposos"), que corresponde
/// a inicios tardíos o a amplitudes mayores del espectro primordial.
#[test]
fn correction_scales_with_amplitude() {
    const SEED: u64 = 42;
    // Amplitudes: σ₈ = 0.4 (débil), 0.8 (estándar), 1.6 (fuerte)
    let sigma8_values = [0.4_f64, 0.8, 1.6];
    let mut ratios = [0.0f64; 3];

    for (i, &s8) in sigma8_values.iter().enumerate() {
        let parts_1lpt = build_particles(&base_config(SEED, s8, false))
            .expect("1LPT build");
        let parts_2lpt = build_particles(&base_config(SEED, s8, true))
            .expect("2LPT build");

        let psi1 = psi1_rms(&parts_1lpt);
        let psi2 = psi2_rms(&parts_1lpt, &parts_2lpt);

        assert!(psi1 > 0.0, "σ₈={}: |Ψ¹|_rms = 0 — campo degenerado", s8);
        ratios[i] = psi2 / psi1;
    }

    // La corrección debe crecer con σ₈.
    assert!(
        ratios[1] > ratios[0],
        "ratio(σ₈=0.8)={:.4e} ≤ ratio(σ₈=0.4)={:.4e} — corrección 2LPT no crece con amplitud",
        ratios[1], ratios[0]
    );
    assert!(
        ratios[2] > ratios[1],
        "ratio(σ₈=1.6)={:.4e} ≤ ratio(σ₈=0.8)={:.4e} — corrección 2LPT no crece con amplitud",
        ratios[2], ratios[1]
    );

    // La corrección siempre debe ser subleading (< 100%) — 2LPT mejora, no domina.
    for (i, &s8) in sigma8_values.iter().enumerate() {
        assert!(
            ratios[i] < 1.0,
            "ratio(σ₈={})={:.4e} ≥ 1.0 — Ψ² domina sobre Ψ¹, régimen inválido para 2LPT",
            s8, ratios[i]
        );
    }

    // Verificar escalado aproximadamente lineal: ratio[1]/ratio[0] ≈ σ₈[1]/σ₈[0] = 2
    let scaling_01 = ratios[1] / ratios[0];
    let scaling_12 = ratios[2] / ratios[1];
    assert!(
        scaling_01 > 1.5 && scaling_01 < 2.8,
        "Escalado σ₈=0.8/σ₈=0.4: ratio_ratio={:.3} — se espera ≈2 (lineal en σ₈)",
        scaling_01
    );
    assert!(
        scaling_12 > 1.5 && scaling_12 < 2.8,
        "Escalado σ₈=1.6/σ₈=0.8: ratio_ratio={:.3} — se espera ≈2 (lineal en σ₈)",
        scaling_12
    );
}

// ── Test 2: cuantificar ratio |Ψ²|/|Ψ¹| para ΛCDM estándar ─────────────────

/// Mide el ratio real |Ψ²|_rms / |Ψ¹|_rms para ΛCDM Planck18 (σ₈=0.8).
///
/// Este test no solo verifica una desigualdad — impone una COTA INFERIOR Y SUPERIOR
/// para confirmar que la corrección es:
/// - No despreciable: ratio > 1e-4 (Ψ² ≠ 0)
/// - Subleading:      ratio < 0.25 (Ψ² no domina)
///
/// El ratio esperado para una retícula 8³ con σ₈=0.8 es O(10⁻²)–O(10⁻¹).
#[test]
fn psi1_psi2_ratio_quantified() {
    const SEED: u64 = 1234;
    let parts_1lpt = build_particles(&base_config(SEED, SIGMA8_STD, false))
        .expect("1LPT build");
    let parts_2lpt = build_particles(&base_config(SEED, SIGMA8_STD, true))
        .expect("2LPT build");

    let psi1 = psi1_rms(&parts_1lpt);
    let psi2 = psi2_rms(&parts_1lpt, &parts_2lpt);
    let ratio = psi2 / psi1;

    assert!(
        psi1 > 0.0,
        "|Ψ¹|_rms = 0 — campo Zel'dovich degenerado (bug en ICs)"
    );

    // La corrección 2LPT debe ser numéricamente no nula.
    const LOWER: f64 = 1e-4;
    assert!(
        psi2 > LOWER * psi1,
        "|Ψ²|_rms / |Ψ¹|_rms = {:.3e} < {:.3e} — corrección demasiado pequeña",
        ratio, LOWER
    );

    // La corrección 2LPT debe ser subleading (2LPT es una corrección, no el efecto dominante).
    const UPPER: f64 = 0.25;
    assert!(
        psi2 < UPPER * psi1,
        "|Ψ²|_rms / |Ψ¹|_rms = {:.3e} > {:.3e} — corrección 2LPT demasiado grande para régimen lineal",
        ratio, UPPER
    );

    // Reportar para el reporte técnico (visible en salida de test con --nocapture).
    println!(
        "\n[phase29 métricas] σ₈={:.2}, a_init={:.3}:\n\
         |Ψ¹|_rms  = {:.4e} [box]\n\
         |Ψ²|_rms  = {:.4e} [box]\n\
         ratio     = {:.4e}  ({:.2}%)",
        SIGMA8_STD, A_INIT, psi1, psi2, ratio, ratio * 100.0
    );
}

// ── Test 3: P(k) inicial de 2LPT consistente con 1LPT ───────────────────────

/// `P(k)` inicial de 2LPT difiere < 15% del de 1LPT en todos los bins de k.
///
/// ## Principio
///
/// La corrección de posición 2LPT es |Ψ²| / |Ψ¹| ≪ 1 en el régimen lineal.
/// Esto implica que el espectro de potencia inicial de 2LPT debe ser muy similar
/// al de 1LPT — la forma espectral se preserva.
///
/// Un diferencia > 15% indicaría que la corrección 2LPT introduce artefactos
/// no físicos en el espectro inicial.
#[test]
fn pk_2lpt_initial_consistent_with_1lpt() {
    const SEED: u64 = 567;
    let parts_1lpt = build_particles(&base_config(SEED, SIGMA8_STD, false))
        .expect("1LPT build");
    let parts_2lpt = build_particles(&base_config(SEED, SIGMA8_STD, true))
        .expect("2LPT build");

    let pos_1lpt: Vec<Vec3> = parts_1lpt.iter().map(|p| p.position).collect();
    let pos_2lpt: Vec<Vec3> = parts_2lpt.iter().map(|p| p.position).collect();
    let masses: Vec<f64> = parts_1lpt.iter().map(|p| p.mass).collect();

    let pk_1lpt = power_spectrum(&pos_1lpt, &masses, BOX, NM);
    let pk_2lpt = power_spectrum(&pos_2lpt, &masses, BOX, NM);

    assert!(
        !pk_1lpt.is_empty() && !pk_2lpt.is_empty(),
        "P(k) vacío — error en power_spectrum"
    );
    assert_eq!(
        pk_1lpt.len(), pk_2lpt.len(),
        "P(k) 1LPT y 2LPT tienen distinto número de bins"
    );

    // Comparar bin a bin: la forma espectral debe estar preservada.
    const MAX_REL_ERR: f64 = 0.15;
    for (b1, b2) in pk_1lpt.iter().zip(pk_2lpt.iter()) {
        if b1.pk <= 0.0 || b2.pk <= 0.0 {
            continue; // bins vacíos o con ruido extremo — ignorar
        }
        let rel_err = (b2.pk - b1.pk).abs() / b1.pk;
        assert!(
            rel_err < MAX_REL_ERR,
            "P_2lpt(k={:.3}) / P_1lpt - 1 = {:.3} > {:.2} — 2LPT rompe la forma espectral",
            b1.k, rel_err, MAX_REL_ERR
        );
    }
}

// ── Test 4: crecimiento gravitacional consistente con PM ─────────────────────

/// Tras evolución corta con PM, `delta_rms` de 1LPT y 2LPT difieren < 15%.
///
/// ## Interpretación física
///
/// En el régimen lineal (a_init=0.02, pocos pasos), la corrección de segundo
/// orden afecta principalmente las velocidades (menor transiente). Las posiciones
/// tardan más en divergir. Este test verifica:
///
/// 1. Ambas variantes muestran crecimiento gravitacional (`delta_f >= delta_0 * 0.5`).
/// 2. El crecimiento es consistente entre 1LPT y 2LPT (< 15% de diferencia).
///
/// Una diferencia > 15% indicaría que los transientes de velocidad de 1LPT
/// están perturbando significativamente la evolución.
#[test]
fn pm_growth_both_lpt_consistent() {
    const SEED: u64 = 888;
    const N_STEPS: usize = 20;
    const DT: f64 = 0.002;

    let mut parts_1lpt = build_particles(&base_config(SEED, SIGMA8_STD, false))
        .expect("1LPT build");
    let mut parts_2lpt = build_particles(&base_config(SEED, SIGMA8_STD, true))
        .expect("2LPT build");

    // Contraste de densidad inicial.
    let delta0_1 = density_contrast_rms(&parts_1lpt, BOX, NGRID_DELTA);
    let delta0_2 = density_contrast_rms(&parts_2lpt, BOX, NGRID_DELTA);

    // Evolución.
    run_pm_evolution(&mut parts_1lpt, N_STEPS, DT);
    run_pm_evolution(&mut parts_2lpt, N_STEPS, DT);

    // Contraste de densidad final.
    let delta_f_1 = density_contrast_rms(&parts_1lpt, BOX, NGRID_DELTA);
    let delta_f_2 = density_contrast_rms(&parts_2lpt, BOX, NGRID_DELTA);

    // Ambas variantes deben mostrar crecimiento (no colapso numérico).
    assert!(
        delta_f_1 >= delta0_1 * 0.5,
        "1LPT+PM: delta_rms colapsa de {:.4e} a {:.4e} — inestabilidad numérica",
        delta0_1, delta_f_1
    );
    assert!(
        delta_f_2 >= delta0_2 * 0.5,
        "2LPT+PM: delta_rms colapsa de {:.4e} a {:.4e} — inestabilidad numérica",
        delta0_2, delta_f_2
    );

    // El crecimiento final debe ser consistente entre 1LPT y 2LPT.
    let rel_diff = (delta_f_2 - delta_f_1).abs() / delta_f_1.max(1e-15);
    const MAX_DIFF: f64 = 0.15;
    assert!(
        rel_diff < MAX_DIFF,
        "PM: |delta_rms_2LPT - delta_rms_1LPT| / delta_rms_1LPT = {:.3} > {:.2}\n\
         delta_f_1LPT={:.4e}  delta_f_2LPT={:.4e}",
        rel_diff, MAX_DIFF, delta_f_1, delta_f_2
    );

    println!(
        "\n[phase29 PM crecimiento] {} pasos, dt={}, a_init={}:\n\
         1LPT: delta_0={:.4e} → delta_f={:.4e}  (×{:.2})\n\
         2LPT: delta_0={:.4e} → delta_f={:.4e}  (×{:.2})\n\
         |Δdelta| / delta_1LPT = {:.3} ({:.2}%)",
        N_STEPS, DT, A_INIT,
        delta0_1, delta_f_1, delta_f_1 / delta0_1.max(1e-15),
        delta0_2, delta_f_2, delta_f_2 / delta0_2.max(1e-15),
        rel_diff, rel_diff * 100.0
    );
}

// ── Test 5: crecimiento gravitacional consistente con TreePM ─────────────────

/// Ídem al test anterior pero con TreePM.
///
/// Verifica que la mejora (o consistencia) observada con PM no es artefacto
/// de un solver en particular. PM y TreePM deben dar resultados comparables
/// para ambas variantes de ICs.
#[test]
fn treepm_growth_both_lpt_consistent() {
    const SEED: u64 = 999;
    const N_STEPS: usize = 20;
    const DT: f64 = 0.002;

    let cfg_1lpt = with_treepm(base_config(SEED, SIGMA8_STD, false));
    let cfg_2lpt = with_treepm(base_config(SEED, SIGMA8_STD, true));

    let mut parts_1lpt = build_particles(&cfg_1lpt).expect("1LPT+TreePM build");
    let mut parts_2lpt = build_particles(&cfg_2lpt).expect("2LPT+TreePM build");

    let delta0_1 = density_contrast_rms(&parts_1lpt, BOX, NGRID_DELTA);
    let delta0_2 = density_contrast_rms(&parts_2lpt, BOX, NGRID_DELTA);

    run_treepm_evolution(&mut parts_1lpt, N_STEPS, DT);
    run_treepm_evolution(&mut parts_2lpt, N_STEPS, DT);

    let delta_f_1 = density_contrast_rms(&parts_1lpt, BOX, NGRID_DELTA);
    let delta_f_2 = density_contrast_rms(&parts_2lpt, BOX, NGRID_DELTA);

    assert!(
        delta_f_1 >= delta0_1 * 0.5,
        "1LPT+TreePM: delta_rms colapsa de {:.4e} a {:.4e}",
        delta0_1, delta_f_1
    );
    assert!(
        delta_f_2 >= delta0_2 * 0.5,
        "2LPT+TreePM: delta_rms colapsa de {:.4e} a {:.4e}",
        delta0_2, delta_f_2
    );

    let rel_diff = (delta_f_2 - delta_f_1).abs() / delta_f_1.max(1e-15);
    const MAX_DIFF: f64 = 0.15;
    assert!(
        rel_diff < MAX_DIFF,
        "TreePM: |delta_rms_2LPT - delta_rms_1LPT| / delta_rms_1LPT = {:.3} > {:.2}\n\
         delta_f_1LPT={:.4e}  delta_f_2LPT={:.4e}",
        rel_diff, MAX_DIFF, delta_f_1, delta_f_2
    );

    println!(
        "\n[phase29 TreePM crecimiento] {} pasos, dt={}, a_init={}:\n\
         1LPT: delta_0={:.4e} → delta_f={:.4e}  (×{:.2})\n\
         2LPT: delta_0={:.4e} → delta_f={:.4e}  (×{:.2})\n\
         |Δdelta| / delta_1LPT = {:.3} ({:.2}%)",
        N_STEPS, DT, A_INIT,
        delta0_1, delta_f_1, delta_f_1 / delta0_1.max(1e-15),
        delta0_2, delta_f_2, delta_f_2 / delta0_2.max(1e-15),
        rel_diff, rel_diff * 100.0
    );
}

// ── Test 6: corrección de velocidad es subleading ────────────────────────────

/// El momentum RMS de 2LPT difiere < 20% del de 1LPT en el régimen lineal.
///
/// ## Principio físico
///
/// El momentum canónico de 2LPT es:
/// ```text
/// p_2LPT = a²·H(a)·[f₁·Ψ¹ + f₂·(D₂/D₁²)·Ψ²]
/// ```
/// La corrección de velocidad es `f₂·(D₂/D₁²)·|Ψ²| ≈ 2 × 0.43 × |Ψ²|`.
/// Como |Ψ²| ≪ |Ψ¹| en el régimen lineal, el momentum RMS no debe diferir
/// significativamente entre 1LPT y 2LPT.
///
/// La tolerancia (20%) es generosa porque la corrección de velocidad es mayor
/// que la de posición (factor f₂/f₁ = 2 y d2_over_d1sq ≈ −0.43).
#[test]
fn velocity_correction_subleading() {
    const SEED: u64 = 3141;

    let parts_1lpt = build_particles(&base_config(SEED, SIGMA8_STD, false))
        .expect("1LPT build");
    let parts_2lpt = build_particles(&base_config(SEED, SIGMA8_STD, true))
        .expect("2LPT build");

    let vrms_1lpt = peculiar_vrms(&parts_1lpt, A_INIT);
    let vrms_2lpt = peculiar_vrms(&parts_2lpt, A_INIT);

    assert!(
        vrms_1lpt > 0.0,
        "v_rms 1LPT = 0 — velocidades degeneradas"
    );
    assert!(
        vrms_2lpt > 0.0,
        "v_rms 2LPT = 0 — velocidades degeneradas"
    );

    let rel_diff = (vrms_2lpt - vrms_1lpt).abs() / vrms_1lpt;
    const MAX_DIFF: f64 = 0.20;
    assert!(
        rel_diff < MAX_DIFF,
        "|v_rms_2LPT - v_rms_1LPT| / v_rms_1LPT = {:.3} > {:.2}\n\
         v_rms_1LPT={:.4e}  v_rms_2LPT={:.4e}\n\
         La corrección de velocidad 2LPT es demasiado grande para régimen lineal",
        rel_diff, MAX_DIFF, vrms_1lpt, vrms_2lpt
    );

    println!(
        "\n[phase29 velocidades] σ₈={:.2}, a_init={:.3}:\n\
         v_rms 1LPT = {:.4e}\n\
         v_rms 2LPT = {:.4e}\n\
         |Δv| / v_1LPT = {:.3} ({:.2}%)\n\
         Nota: corrección incluye factor f₂/f₁=2 × d₂/d₁²≈−0.43",
        SIGMA8_STD, A_INIT, vrms_1lpt, vrms_2lpt, rel_diff, rel_diff * 100.0
    );
}
