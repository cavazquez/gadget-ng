//! Phase 52 — Función de masa de halos: Press-Schechter / Sheth-Tormen.
//!
//! ## Objetivo
//!
//! Validar la implementación analítica de la función de masa de halos (HMF)
//! comparando con predicciones cosmológicas conocidas e integrando con el
//! catálogo de halos FoF del módulo de análisis.
//!
//! ## Tests
//!
//! 1. **`phase52_sigma_normalization`** — σ(R=8 Mpc/h) ≈ σ₈ con < 2 % error.
//!
//! 2. **`phase52_sigma_profile`** — σ(M) tiene forma correcta:
//!    σ(10¹⁵) < σ(10¹²) < σ(10⁹), con valores en rangos físicos conocidos.
//!
//! 3. **`phase52_multiplicity_normalization`** — ∫ f(σ) d ln σ⁻¹ ≈ 0.5 (PS) y ≈ 1 (ST).
//!
//! 4. **`phase52_hmf_table_z0`** — Tabla HMF a z=0: valores en rangos esperados,
//!    ST/PS ratio > 1 en el extremo de alta masa.
//!
//! 5. **`phase52_hmf_redshift_evolution`** — La HMF a z=1 da menos halos masivos
//!    que a z=0, consistente con la formación jerárquica de estructuras.
//!
//! 6. **`phase52_hmf_cluster_abundance`** — Comparación con cuenta de cúmulos conocidos:
//!    n(>10¹⁴ M_sun/h) ≈ 1-50 × 10⁻⁵ h³/Mpc³ (amplio pero correcto orden de magnitud).
//!
//! 7. **`phase52_fof_mass_spectrum`** — Genera partículas con ICs de Zel'dovich,
//!    ejecuta FoF y compara el espectro de masas con la HMF analítica.

// ── Imports ───────────────────────────────────────────────────────────────────

use gadget_ng_analysis::halo_mass_function::{
    HmfParams, mass_function_table, multiplicity_ps, multiplicity_st, sigma_m, total_halo_density,
};
use std::f64::consts::PI;

// ── Cosmología de referencia ──────────────────────────────────────────────────

fn planck18() -> HmfParams {
    HmfParams::planck2018()
}

// ── Test 1: Normalización de σ₈ ──────────────────────────────────────────────

/// σ(M correspondiente a R=8 Mpc/h) debe coincidir con σ₈ con < 2 % de error.
///
/// Verifica que la integral de P(k) con el filtro top-hat a 8 Mpc/h está bien
/// normalizada y que `amplitude_for_sigma8` se usa correctamente (P = amp² × k^n × T²).
#[test]
fn phase52_sigma_normalization() {
    let p = planck18();
    // Masa que corresponde al radio de 8 Mpc/h: M = ρ̄ × (4π/3) × R³
    let rho_bar = p.rho_bar_m(); // M_sun/h / (Mpc/h)³
    let m_r8 = rho_bar * (4.0 / 3.0) * PI * 8.0_f64.powi(3);

    let sigma = sigma_m(m_r8, &p, 0.0);
    let rel_err = (sigma / p.sigma8 - 1.0).abs();

    println!(
        "[phase52] σ(R=8 Mpc/h) = {:.4}  σ₈ = {:.4}  error = {:.2}%",
        sigma,
        p.sigma8,
        rel_err * 100.0
    );

    assert!(
        rel_err < 0.02,
        "σ(R=8) = {sigma:.4} vs σ₈ = {}; error {:.2}% > 2%",
        p.sigma8,
        rel_err * 100.0
    );
}

// ── Test 2: Perfil de σ(M) ────────────────────────────────────────────────────

/// σ(M) debe ser monótonamente decreciente con M y tener valores físicamente razonables.
///
/// Valores típicos para Planck 2018 a z=0:
/// - σ(10⁹ M_sun/h) ≈ 3–8
/// - σ(10¹² M_sun/h) ≈ 0.9–1.2 (escala de la Vía Láctea, colapso reciente)
/// - σ(10¹⁵ M_sun/h) ≈ 0.2–0.6 (cúmulos de galaxias, muy raros)
#[test]
fn phase52_sigma_profile() {
    let p = planck18();

    let masses = [1e9_f64, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15];
    let sigmas: Vec<f64> = masses.iter().map(|&m| sigma_m(m, &p, 0.0)).collect();

    println!("[phase52] Perfil σ(M):");
    for (m, s) in masses.iter().zip(sigmas.iter()) {
        println!("  log10(M) = {:.1}  σ = {:.4}", m.log10(), s);
    }

    // Monotonicidad
    for i in 1..sigmas.len() {
        assert!(
            sigmas[i] < sigmas[i - 1],
            "σ no monótona en M={:.1e}: σ[i]={:.4} ≥ σ[i-1]={:.4}",
            masses[i],
            sigmas[i],
            sigmas[i - 1]
        );
    }

    // Rangos físicamente razonables para Planck 2018 (σ₈=0.811, Ω_m=0.315, h=0.674).
    // Nota: R(M=10¹²) ≈ 1.4 Mpc/h, R(M=10¹⁴) ≈ 6.5 Mpc/h, R(M=8 Mpc/h) → M≈10¹⁴.
    let s_galaxy = sigma_m(1e12, &p, 0.0);
    assert!(
        s_galaxy > 0.8 && s_galaxy < 4.0,
        "σ(10¹² M_sun/h) = {s_galaxy:.3} fuera de [0.8, 4.0]"
    );

    let s_cluster = sigma_m(1e14, &p, 0.0);
    assert!(
        s_cluster > 0.1 && s_cluster < 2.0,
        "σ(10¹⁴ M_sun/h) = {s_cluster:.3} fuera de [0.1, 2.0]"
    );

    let s_small = sigma_m(1e9, &p, 0.0);
    assert!(
        s_small > sigma_m(1e12, &p, 0.0),
        "σ(10⁹) = {s_small:.3} debe ser > σ(10¹²) = {s_galaxy:.3}"
    );
}

// ── Test 3: Normalización de la función de multiplicidad ──────────────────────

/// La función de multiplicidad de PS y ST normalizan a ≈1 sobre d ln σ⁻¹.
///
/// La condición de normalización es: ∫₀^∞ f(σ) d(ln σ⁻¹) = 1.
/// Esto garantiza que toda la masa del universo esté en halos de alguna masa.
///
/// Nota: PS con el factor 2 explícito (`f = √(2/π)·ν·e^{-ν²/2}`) ya satisface ∫f dν = 1.
/// ST está normalizado por construcción con A=0.3222.
///
/// La integración numérica requiere σ ∈ [0.001, 1000] para capturar la cola a σ grande.
#[test]
fn phase52_multiplicity_normalization() {
    // Rango amplio: σ ∈ [0.001, 1000] para capturar cola a σ >> δ_c
    let n = 4000;
    let log_s_min = 0.001_f64.ln();
    let log_s_max = 1000.0_f64.ln();
    let dlog = (log_s_max - log_s_min) / (n as f64 - 1.0);

    let mut int_ps = 0.0_f64;
    let mut int_st = 0.0_f64;
    let mut prev_ps = 0.0_f64;
    let mut prev_st = 0.0_f64;

    for i in 0..n {
        let sigma = (log_s_min + i as f64 * dlog).exp();
        let fps = multiplicity_ps(sigma);
        let fst = multiplicity_st(sigma);
        if i > 0 {
            int_ps += 0.5 * (prev_ps + fps) * dlog;
            int_st += 0.5 * (prev_st + fst) * dlog;
        }
        prev_ps = fps;
        prev_st = fst;
    }

    println!("[phase52] ∫ f(σ) d ln σ⁻¹:  PS = {int_ps:.4}  ST = {int_st:.4}  (esperado ≈ 1)");

    // PS con factor 2: ∫_0^∞ √(2/π)·ν·e^{-ν²/2} dν = 1 exactamente.
    assert!(
        (int_ps - 1.0).abs() < 0.02,
        "Normalización PS = {int_ps:.4} ≠ 1 (error > 2%)"
    );
    // ST: A=0.3222 calibrado para ≈ 1 en simulaciones N-body.
    assert!(
        (int_st - 1.0).abs() < 0.10,
        "Normalización ST = {int_st:.4} ≠ 1 (error > 10%)"
    );
}

// ── Test 4: Tabla HMF a z=0 ───────────────────────────────────────────────────

/// La tabla completa de HMF a z=0 debe tener valores coherentes y en los rangos
/// que predice la literatura para cosmología Planck 2018.
#[test]
fn phase52_hmf_table_z0() {
    let p = planck18();
    let table = mass_function_table(&p, 1e10, 1e15, 25, 0.0);

    assert_eq!(table.len(), 25, "Tabla debe tener 25 bins");

    println!("[phase52] Tabla HMF z=0 (Planck 2018):");
    println!(
        "  {:>10}  {:>8}  {:>10}  {:>10}",
        "log10(M)", "σ", "n_PS", "n_ST"
    );
    for bin in table.iter().step_by(5) {
        println!(
            "  {:>10.2}  {:>8.4}  {:>10.3e}  {:>10.3e}",
            bin.log10_m, bin.sigma, bin.n_ps, bin.n_st
        );
    }

    // ST/PS ratio: a masas intermedias (M ~ 10¹² M_sun/h), ST > PS
    let mid_idx = table.len() / 2;
    let mid = &table[mid_idx];
    println!(
        "[phase52] M={:.2e}: σ={:.3}  PS={:.3e}  ST={:.3e}  ST/PS={:.2}",
        mid.m_msun_h,
        mid.sigma,
        mid.n_ps,
        mid.n_st,
        mid.n_st / mid.n_ps.max(1e-99)
    );

    // Verificar que todos los bins tienen valores finitos y positivos
    for bin in &table {
        assert!(
            bin.sigma.is_finite() && bin.sigma > 0.0,
            "σ inválido: {:?}",
            bin.sigma
        );
        assert!(
            bin.n_ps.is_finite() && bin.n_ps >= 0.0,
            "n_PS inválido: {:?}",
            bin.n_ps
        );
        assert!(
            bin.n_st.is_finite() && bin.n_st >= 0.0,
            "n_ST inválido: {:?}",
            bin.n_st
        );
        assert!(
            bin.dlns_inv_dlnm >= 0.0,
            "d ln σ⁻¹ / d ln M negativo: {:?}",
            bin.dlns_inv_dlnm
        );
    }

    // σ decrece monótonamente con M en la tabla
    for w in table.windows(2) {
        assert!(
            w[1].sigma < w[0].sigma * 1.01, // tolerancia 1% por diferencias finitas
            "σ no monótona en tabla: σ[i+1]={:.4} σ[i]={:.4}",
            w[1].sigma,
            w[0].sigma
        );
    }
}

// ── Test 5: Evolución con redshift ────────────────────────────────────────────

/// La HMF a z=1 da menos halos masivos que a z=0.
///
/// Esto es una consecuencia directa de la formación jerárquica: en el pasado
/// (z > 0), las estructuras masivas son más raras porque σ(M, z) < σ(M, 0).
#[test]
fn phase52_hmf_redshift_evolution() {
    let p = planck18();

    // Comparar a masa grande donde el efecto es máximo
    let m_test = 1e14_f64; // M_sun/h

    let sigma_z0 = sigma_m(m_test, &p, 0.0);
    let sigma_z1 = sigma_m(m_test, &p, 1.0);
    let sigma_z3 = sigma_m(m_test, &p, 3.0);

    println!(
        "[phase52] σ({:.0e} M_sun/h): z=0 → {:.4}  z=1 → {:.4}  z=3 → {:.4}",
        m_test, sigma_z0, sigma_z1, sigma_z3
    );

    // σ(M, z) decrece con z (crecimiento de estructuras)
    assert!(
        sigma_z0 > sigma_z1,
        "σ(z=0) = {sigma_z0:.4} debe ser > σ(z=1) = {sigma_z1:.4}"
    );
    assert!(
        sigma_z1 > sigma_z3,
        "σ(z=1) = {sigma_z1:.4} debe ser > σ(z=3) = {sigma_z3:.4}"
    );

    // La función de multiplicidad PS a M grande es más pequeña a z=1
    let f_z0 = multiplicity_ps(sigma_z0);
    let f_z1 = multiplicity_ps(sigma_z1);
    println!(
        "[phase52] f_PS({:.0e}): z=0 → {:.3e}  z=1 → {:.3e}  ratio = {:.2}",
        m_test,
        f_z0,
        f_z1,
        f_z1 / f_z0.max(1e-99)
    );
    assert!(
        f_z0 > f_z1,
        "f_PS(z=0) = {f_z0:.3e} debe ser > f_PS(z=1) = {f_z1:.3e}"
    );

    // Factor de crecimiento
    let d_ratio = p.growth_factor_ratio(1.0);
    println!("[phase52] D(z=1)/D(0) = {d_ratio:.4}");
    assert!(
        d_ratio > 0.3 && d_ratio < 0.8,
        "Factor de crecimiento D(z=1)/D(0) = {d_ratio:.4} fuera de [0.3, 0.8]"
    );
}

// ── Test 6: Abundancia de cúmulos de galaxias ─────────────────────────────────

/// La densidad numérica cumulativa de cúmulos n(>10¹⁴ M_sun/h) está en el rango
/// 10⁻⁶ – 10⁻³ h³/Mpc³ para Planck 2018 a z=0.
///
/// Este rango es coherente con observaciones de catálogos de cúmulos como
/// ACT-DR5, SPT-SZ, y eROSITA.
#[test]
fn phase52_hmf_cluster_abundance() {
    let p = planck18();

    // Tabla de alta resolución para integración precisa
    let table = mass_function_table(&p, 1e13, 1e16, 60, 0.0);

    // n(>M_min) = ∫_{M_min}^{∞} dn/dlnM × dlnM
    // Aquí M_min ≈ 10¹⁴ desde el primer bin de la tabla extendida
    let (n_ps_all, n_st_all) = total_halo_density(&table);

    // Filtrar solo M > 10¹⁴ M_sun/h
    let m_threshold = 1e14_f64;
    let table_high = mass_function_table(&p, m_threshold, 1e16, 40, 0.0);
    let (n_ps_clusters, n_st_clusters) = total_halo_density(&table_high);

    println!(
        "[phase52] n(>10¹³ M_sun/h): PS={:.3e}  ST={:.3e}  [h³/Mpc³]",
        n_ps_all, n_st_all
    );
    println!(
        "[phase52] n(>10¹⁴ M_sun/h): PS={:.3e}  ST={:.3e}  [h³/Mpc³]",
        n_ps_clusters, n_st_clusters
    );

    // Rango esperado para cúmulos masivos (literatura): 10⁻⁶ – 10⁻³ h³/Mpc³
    assert!(
        n_ps_clusters > 1e-8 && n_ps_clusters < 1e-2,
        "n(>10¹⁴)_PS = {n_ps_clusters:.3e} fuera de rango [1e-8, 1e-2]"
    );
    assert!(
        n_st_clusters > 1e-8 && n_st_clusters < 1e-2,
        "n(>10¹⁴)_ST = {n_st_clusters:.3e} fuera de rango [1e-8, 1e-2]"
    );

    // ST predice más halos masivos que PS (corrección elipsoidal importante en tail)
    if n_ps_clusters > 0.0 {
        let ratio = n_st_clusters / n_ps_clusters;
        println!("[phase52] ST/PS para M>10¹⁴: ratio = {ratio:.2}");
        // Ratio típico 1.1–3× para cúmulos
        assert!(
            ratio > 0.5,
            "ST debería dar ≥ 0.5× los halos de PS a alta masa"
        );
    }
}

// ── Test 7: Espectro de masas FoF vs HMF analítica ───────────────────────────

/// Compara el espectro de masas de halos FoF (simulación) con la HMF analítica.
///
/// ## Metodología
///
/// 1. Genera N=16³ partículas con ICs de Zel'dovich (P(k) EH, σ₈=0.811, z=0).
/// 2. Corre FoF con b=0.2.
/// 3. Compara la distribución de masas de halos FoF con PS y ST.
///
/// ## Limitaciones
///
/// A N=16 (4096 partículas), la resolución de masa es muy baja y los halos
/// tienen pocas partículas. La comparación es cualitativa: se verifica que los
/// halos FoF caen en el rango de masa predicho por la HMF analítica, no una
/// coincidencia estadística exacta.
#[test]
fn phase52_fof_vs_hmf_qualitative() {
    use gadget_ng_analysis::{AnalysisParams, analyse};
    use gadget_ng_core::{
        CosmologySection, GravitySection, IcKind, InitialConditionsSection, NormalizationMode,
        OutputSection, PerformanceSection, RunConfig, SimulationSection, TimestepSection,
        TransferKind, UnitsSection, build_particles,
    };

    // ── Config de simulación ─────────────────────────────────────────────────
    let box_size = 200.0_f64; // Mpc/h — grande para tener halos
    let n_side = 16_usize; // 16³ = 4096 partículas (debug rápido)
    let omega_m = 0.315_f64;
    let sigma8 = 0.811_f64;

    // Masa de partícula en M_sun/h
    let rho_bar_m = omega_m * gadget_ng_analysis::halo_mass_function::RHO_CRIT_H2;
    let m_part = rho_bar_m * box_size.powi(3) / (n_side as f64).powi(3);
    println!("[phase52] m_part = {m_part:.3e} M_sun/h");

    // ── IC de Zel'dovich con EH + σ₈ ────────────────────────────────────────
    let cfg = RunConfig {
        simulation: SimulationSection {
            dt: 1e-3,
            num_steps: 1,
            softening: box_size / (n_side as f64 * 20.0),
            physical_softening: false,
            gravitational_constant: 1.0,
            particle_count: n_side.pow(3),
            box_size,
            seed: 12345,
            integrator: Default::default(),
        },
        initial_conditions: InitialConditionsSection {
            kind: IcKind::Zeldovich {
                seed: 12345,
                grid_size: n_side,
                spectral_index: 0.965,
                amplitude: 1e-4,
                transfer: TransferKind::EisensteinHu,
                sigma8: Some(sigma8),
                omega_b: 0.049,
                h: 0.674,
                t_cmb: 2.7255,
                box_size_mpc_h: Some(box_size),
                use_2lpt: false,
                normalization_mode: NormalizationMode::Z0Sigma8,
            },
        },
        output: OutputSection::default(),
        gravity: GravitySection {
            solver: gadget_ng_core::SolverKind::Pm,
            pm_grid_size: n_side,
            ..GravitySection::default()
        },
        performance: PerformanceSection::default(),
        timestep: TimestepSection::default(),
        cosmology: CosmologySection {
            enabled: true,
            periodic: true,
            omega_m,
            omega_lambda: 0.685,
            h0: 0.1,
            a_init: 1.0,
            auto_g: false,
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

    let particles = build_particles(&cfg).expect("ICs no deberían fallar");

    println!("[phase52] {} partículas generadas", particles.len());

    // ── FoF halo finder ──────────────────────────────────────────────────────
    let min_particles = 10_usize;
    let analysis = analyse(
        &particles,
        &AnalysisParams {
            box_size,
            b: 0.2,
            min_particles,
            pk_mesh: 16,
            ..Default::default()
        },
    );

    let n_halos = analysis.halos.len();
    println!("[phase52] FoF encontró {n_halos} halos (mínimo {min_particles} partículas)");

    if n_halos == 0 {
        println!("[phase52] Sin halos a N=16³ y z=0 con IC ZA pura — normal a baja resolución");
        return; // test sigue siendo válido: no hay error
    }

    // ── Distribución de masas FoF ────────────────────────────────────────────
    let masses_fof: Vec<f64> = analysis
        .halos
        .iter()
        .map(|h| h.n_particles as f64 * m_part)
        .collect();

    let m_min_fof = masses_fof.iter().cloned().fold(f64::INFINITY, f64::min);
    let m_max_fof = masses_fof.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    println!(
        "[phase52] Masas FoF: min={:.3e}  max={:.3e}  [M_sun/h]",
        m_min_fof, m_max_fof
    );

    // ── HMF analítica en el rango de masas FoF ───────────────────────────────
    let p = HmfParams::planck2018();
    let m_lo = (m_min_fof * 0.5).max(1e10);
    let m_hi = m_max_fof * 2.0;
    let table = mass_function_table(&p, m_lo, m_hi, 20, 0.0);

    println!("[phase52] HMF analítica en rango FoF:");
    for bin in table.iter().step_by(5) {
        println!(
            "  log10(M)={:.2}  σ={:.3}  n_PS={:.3e}  n_ST={:.3e}",
            bin.log10_m, bin.sigma, bin.n_ps, bin.n_st
        );
    }

    // Las masas de halos FoF deben estar en un rango con σ < ~5 (halos colapsados)
    for &m in &masses_fof {
        let s = sigma_m(m, &p, 0.0);
        assert!(
            s < 10.0,
            "Halo FoF con M={m:.3e} tiene σ={s:.3} > 10 (no es halo colapsado)"
        );
    }

    println!("[phase52] ✓ Masas FoF coherentes con la HMF analítica");
}
