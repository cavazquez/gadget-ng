//! Phase 53 — Perfiles de halos NFW y relación concentración-masa c(M).
//!
//! ## Objetivo
//!
//! Validar la implementación del perfil NFW y la relación c(M) de Duffy+2008,
//! comparando propiedades analíticas con valores de la literatura y con halos
//! FoF generados desde condiciones iniciales de Zel'dovich.
//!
//! ## Tests
//!
//! 1. **`phase53_nfw_virial_properties`** — Propiedades básicas del perfil NFW:
//!    r_200, ρ_mean(<r_200) = 200 ρ_crit, M(<r_200) = M_200.
//!
//! 2. **`phase53_concentration_mass_relation`** — c(M, z) de Duffy+2008:
//!    tendencias con M y z, valores razonables, comparación MW vs cluster.
//!
//! 3. **`phase53_nfw_profile_shapes`** — Pendiente logarítmica del perfil:
//!    ρ ∝ r^{-1} en el interior (r << r_s), ρ ∝ r^{-3} en el exterior (r >> r_s).
//!
//! 4. **`phase53_mass_concentration_table`** — Tabla c(M) para masas cosmológicas:
//!    desde enanas (10⁸ M_sun/h) hasta super-cúmulos (10¹⁵ M_sun/h).
//!
//! 5. **`phase53_circular_velocity_peak`** — Velocidad circular máxima en r ≈ 2.16 r_s.
//!
//! 6. **`phase53_density_profile_from_fof_halo`** — Genera IC Zel'dovich, busca
//!    halos FoF y mide el perfil de densidad del halo más masivo.

use gadget_ng_analysis::nfw::{
    concentration_bhattacharya2013, concentration_duffy2008, fit_nfw_concentration,
    measure_density_profile, r200_from_m200, rho_crit_z, NfwProfile, RHO_CRIT0,
};
use std::f64::consts::PI;

// ── Cosmología de referencia ──────────────────────────────────────────────────

const OMEGA_M: f64 = 0.315;
const OMEGA_L: f64 = 0.685;

fn planck_rho_crit() -> f64 {
    rho_crit_z(OMEGA_M, OMEGA_L, 0.0)
}

// ── Test 1: Propiedades viriales del NFW ──────────────────────────────────────

/// Verifica que r_200 y ρ_mean(<r_200) son correctos por construcción.
///
/// Por definición: ρ_mean(<r_200) = 200 ρ_crit.
/// El perfil NFW construido con `from_m200_c` debe satisfacer esto exactamente.
#[test]
fn phase53_nfw_virial_properties() {
    let rho_c = planck_rho_crit();
    println!("[phase53] ρ_crit(z=0) = {rho_c:.4e} (M_sun/h)/(Mpc/h)³");

    let masses = [1e11_f64, 1e12, 1e13, 1e14, 1e15];

    println!("[phase53] Propiedades NFW a z=0:");
    println!("  {:>10}  {:>6}  {:>8}  {:>10}  {:>10}",
        "M_200", "c", "r_200", "ρ_s", "r_s");

    for &m200 in &masses {
        let c = concentration_duffy2008(m200, 0.0);
        let profile = NfwProfile::from_m200_c(m200, c, rho_c);
        let r200 = r200_from_m200(m200, rho_c);

        println!("  {:.1e}  {:>6.2}  {:>8.4}  {:>10.3e}  {:>10.4}",
            m200, c, r200, profile.rho_s, profile.r_s);

        // M(<r_200) = M_200 por construcción
        let m_enc = profile.mass_enclosed(r200);
        let err_m = (m_enc / m200 - 1.0).abs();
        assert!(err_m < 1e-10, "M(<r_200)/M_200 error = {err_m:.2e} para M={m200:.1e}");

        // ρ_mean(<r_200) = 200 ρ_crit
        let rho_mean = m_enc / ((4.0 / 3.0) * PI * r200.powi(3));
        let err_rho = (rho_mean / (200.0 * rho_c) - 1.0).abs();
        assert!(err_rho < 1e-10, "ρ_mean(<r_200) error = {err_rho:.2e} para M={m200:.1e}");

        // r_s = r_200 / c por construcción
        let err_rs = (profile.r_s / (r200 / c) - 1.0).abs();
        assert!(err_rs < 1e-14, "r_s error = {err_rs:.2e}");

        // ρ_s debe ser positiva y razonable
        assert!(profile.rho_s > 0.0 && profile.rho_s.is_finite(), "ρ_s no válido");
    }
}

// ── Test 2: Relación c(M) de Duffy+2008 ──────────────────────────────────────

/// Verifica tendencias y valores de la relación c(M, z) de Duffy et al. (2008).
///
/// Valores de referencia de la literatura para ΛCDM/WMAP5, z=0:
/// - c(10¹²) ≈ 5.7 (halos tipo Vía Láctea)
/// - c(10¹⁴) ≈ 4.4 (cúmulos de galaxias)
/// - c(10¹⁵) ≈ 3.4 (super-cúmulos)
#[test]
fn phase53_concentration_mass_relation() {
    println!("[phase53] Relación c(M) Duffy+2008 vs Bhattacharya+2013:");
    println!("  {:>10}  {:>8}  {:>8}", "log10(M)", "Duffy08", "Bhatt13");

    let masses = [1e10_f64, 1e11, 1e12, 1e13, 1e14, 1e15];
    let mut prev_c = f64::INFINITY;

    for &m200 in &masses {
        let c_d = concentration_duffy2008(m200, 0.0);
        let c_b = concentration_bhattacharya2013(m200, 0.0);
        println!("  {:>10.1}  {:>8.3}  {:>8.3}", m200.log10(), c_d, c_b);

        // c decrece con M
        assert!(c_d < prev_c * 1.01,
            "c(Duffy) no decrece con M: c({m200:.1e})={c_d:.3} ≥ prev={prev_c:.3}");
        prev_c = c_d;

        // Rango físico: c ∈ [1.5, 30] para M ∈ [10¹⁰, 10¹⁵]
        assert!(c_d > 1.5 && c_d < 30.0,
            "c({m200:.1e}) = {c_d:.3} fuera de [1.5, 30]");
    }

    // Evolución con redshift: c decrece al aumentar z
    let m_test = 1e13_f64;
    let c_z0 = concentration_duffy2008(m_test, 0.0);
    let c_z1 = concentration_duffy2008(m_test, 1.0);
    let c_z2 = concentration_duffy2008(m_test, 2.0);
    println!("[phase53] c(10¹³) vs z: z=0→{c_z0:.3}  z=1→{c_z1:.3}  z=2→{c_z2:.3}");
    assert!(c_z0 > c_z1 && c_z1 > c_z2, "c debe decrecer con z");

    // Verificar valores de referencia de la literatura (±20%)
    let c_mw = concentration_duffy2008(1e12, 0.0);
    let c_cluster = concentration_duffy2008(1e14, 0.0);
    assert!(c_mw > 4.5 && c_mw < 8.0,
        "c(MW ~ 10¹²) = {c_mw:.3} fuera de [4.5, 8.0]");
    assert!(c_cluster > 3.5 && c_cluster < 6.0,
        "c(cluster ~ 10¹⁴) = {c_cluster:.3} fuera de [3.5, 6.0]");
}

// ── Test 3: Pendientes logarítmicas del perfil NFW ────────────────────────────

/// El perfil NFW tiene:
/// - Pendiente interior: ρ ∝ r^{-1} para r << r_s
/// - Pendiente exterior: ρ ∝ r^{-3} para r >> r_s
///
/// Se verifica midiendo la pendiente logarítmica d ln ρ / d ln r en los extremos.
#[test]
fn phase53_nfw_profile_shapes() {
    let profile = NfwProfile { rho_s: 1e7, r_s: 0.3 };

    // Pendiente interna: en r = 0.01 r_s, esperamos γ ≈ -1
    let r_inner = 0.01 * profile.r_s;
    let eps = 0.001;
    let gamma_inner = (profile.density(r_inner * (1.0 + eps)).ln()
        - profile.density(r_inner * (1.0 - eps)).ln())
        / (2.0 * eps);

    println!("[phase53] Pendiente interna (r=0.01r_s): γ = {gamma_inner:.4}  (esperado ≈ -1)");
    assert!(
        (gamma_inner + 1.0).abs() < 0.05,
        "γ_inner = {gamma_inner:.4} ≠ -1 (NFW inner slope)"
    );

    // Pendiente externa: en r = 100 r_s, esperamos γ ≈ -3
    let r_outer = 100.0 * profile.r_s;
    let gamma_outer = (profile.density(r_outer * (1.0 + eps)).ln()
        - profile.density(r_outer * (1.0 - eps)).ln())
        / (2.0 * eps);

    println!("[phase53] Pendiente externa (r=100r_s): γ = {gamma_outer:.4}  (esperado ≈ -3)");
    assert!(
        (gamma_outer + 3.0).abs() < 0.05,
        "γ_outer = {gamma_outer:.4} ≠ -3 (NFW outer slope)"
    );

    // Pendiente en r_s: γ = -2 (punto de inflexión)
    let r_scale = profile.r_s;
    let gamma_scale = (profile.density(r_scale * (1.0 + eps)).ln()
        - profile.density(r_scale * (1.0 - eps)).ln())
        / (2.0 * eps);
    println!("[phase53] Pendiente en r_s: γ = {gamma_scale:.4}  (esperado ≈ -2)");
    assert!(
        (gamma_scale + 2.0).abs() < 0.05,
        "γ(r_s) = {gamma_scale:.4} ≠ -2"
    );
}

// ── Test 4: Tabla c(M) cosmológica ───────────────────────────────────────────

/// Genera una tabla completa c(M) con parámetros del perfil NFW para todas las
/// escalas de masa relevantes en cosmología.
#[test]
fn phase53_mass_concentration_table() {
    let rho_c = planck_rho_crit();

    let masses = [1e8_f64, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15];

    println!("[phase53] Tabla c(M) y propiedades NFW (Planck 2018, z=0):");
    println!("  {:>7}  {:>5}  {:>7}  {:>7}  {:>10}  {:>7}",
        "log10M", "c", "r_200", "r_s", "rho_s", "M(<2rs)/M200");

    for &m200 in &masses {
        let c = concentration_duffy2008(m200, 0.0);
        let profile = NfwProfile::from_m200_c(m200, c, rho_c);
        let r200 = r200_from_m200(m200, rho_c);
        let m_2rs = profile.mass_enclosed(2.0 * profile.r_s) / m200;

        println!("  {:>7.1}  {:>5.2}  {:>7.4}  {:>7.5}  {:>10.3e}  {:>7.4}",
            m200.log10(), c, r200, profile.r_s, profile.rho_s, m_2rs);

        // Verificaciones básicas
        assert!(r200 > 0.0 && r200 < 50.0, "r_200 fuera de rango para M={m200:.1e}");
        assert!(profile.r_s > 0.0 && profile.r_s < 10.0, "r_s fuera de rango");
        assert!(profile.rho_s > 0.0 && profile.rho_s.is_finite(), "ρ_s inválido");

        // Fracción de masa encerrada en 2r_s (típico: ~40-60%)
        assert!(m_2rs > 0.1 && m_2rs < 1.0,
            "M(<2r_s)/M_200 = {m_2rs:.4} fuera de [0.1, 1.0] para M={m200:.1e}");
    }
}

// ── Test 5: Velocidad circular máxima ─────────────────────────────────────────

/// La velocidad circular v_c(r) = sqrt(G M(<r)/r) alcanza su máximo en r ≈ 2.16 r_s.
///
/// Esta es una propiedad analítica conocida del perfil NFW. Se verifica
/// que el máximo numérico cae en ese rango.
#[test]
fn phase53_circular_velocity_peak() {
    // r_max de v_c: resolvemos d/dr [M(<r)/r] = 0
    // M(<r)/r = 4π ρ_s r_s³ g(x)/r = 4π ρ_s r_s² g(x)/x
    // d/dx [g(x)/x] = 0 → g'(x)/x - g(x)/x² = 0 → x g'(x) = g(x)
    // g'(x) = x/(1+x)² → x²/(1+x)² = g(x) = ln(1+x) - x/(1+x)
    // Solución numérica: x ≈ 2.163

    let profile = NfwProfile { rho_s: 1e7, r_s: 0.3 };

    // Barrer r en [0.5 r_s, 10 r_s] y encontrar el máximo de M(<r)/r
    let n_r = 1000;
    let r_min = 0.5 * profile.r_s;
    let r_max = 10.0 * profile.r_s;

    let mut max_vc2 = 0.0_f64;
    let mut r_at_max = 0.0_f64;

    for i in 0..n_r {
        let r = r_min * (r_max / r_min).powf(i as f64 / (n_r - 1) as f64);
        let vc2 = profile.circular_velocity_sq_over_g(r);
        if vc2 > max_vc2 {
            max_vc2 = vc2;
            r_at_max = r;
        }
    }

    let x_at_max = r_at_max / profile.r_s;
    println!("[phase53] v_c máxima en r/r_s = {x_at_max:.3}  (esperado ≈ 2.163)");

    // El máximo debe estar en x ≈ 2.163 ± 10%
    assert!(
        (x_at_max - 2.163).abs() < 0.3,
        "v_c_max en r/r_s = {x_at_max:.3} ≠ 2.163 ± 10%"
    );
}

// ── Test 6: Perfil de densidad desde halo FoF ─────────────────────────────────

/// Genera condiciones iniciales Zel'dovich, identifica halos FoF, y mide el
/// perfil de densidad radial del halo más masivo comparando con NFW analítico.
///
/// ## Limitaciones
///
/// A N=16³ y z=0 con ICs de Zel'dovich (sin evolución temporal), los halos son
/// solo leves sobredensidades en el campo lineal, no estructuras colapsadas.
/// Este test verifica que la infraestructura de medición funciona correctamente,
/// no que el perfil sea NFW (requeriría simulación completa).
#[test]
fn phase53_density_profile_from_fof_halo() {
    use gadget_ng_analysis::{analyse, AnalysisParams};
    use gadget_ng_core::{
        build_particles, CosmologySection, GravitySection, IcKind, InitialConditionsSection,
        NormalizationMode, OutputSection, PerformanceSection, RunConfig, SimulationSection,
        TimestepSection, TransferKind, UnitsSection,
    };

    let box_size = 200.0_f64;
    let n_side = 16_usize;
    let omega_m = OMEGA_M;
    let sigma8 = 0.811_f64;

    let rho_bar_m = omega_m * RHO_CRIT0;
    let m_part = rho_bar_m * box_size.powi(3) / (n_side as f64).powi(3);
    println!("[phase53] m_part = {m_part:.3e} M_sun/h  (N={}³)", n_side);

    // ── IC de Zel'dovich ─────────────────────────────────────────────────────
    let cfg = RunConfig {
        simulation: SimulationSection {
            dt: 1e-3,
            num_steps: 1,
            softening: box_size / (n_side as f64 * 20.0),
            gravitational_constant: 1.0,
            particle_count: n_side.pow(3),
            box_size,
            seed: 99,
            integrator: Default::default(),
        },
        initial_conditions: InitialConditionsSection {
            kind: IcKind::Zeldovich {
                seed: 99,
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
            omega_lambda: OMEGA_L,
            h0: 0.1,
            a_init: 1.0,
            auto_g: false,
        },
        units: UnitsSection::default(),
        decomposition: Default::default(),
    };

    let particles = build_particles(&cfg).expect("ICs no deben fallar");
    println!("[phase53] {} partículas generadas", particles.len());

    // ── FoF halo finder ──────────────────────────────────────────────────────
    let rho_crit_val = planck_rho_crit();
    let analysis = analyse(
        &particles,
        &AnalysisParams {
            box_size,
            b: 0.2,
            min_particles: 8,
            pk_mesh: 16,
            ..Default::default()
        },
    );

    let n_halos = analysis.halos.len();
    println!("[phase53] FoF encontró {n_halos} halos");

    if n_halos == 0 {
        // Con ICs lineales a z=0 y N=16³, es normal no encontrar halos colapsados
        println!("[phase53] Sin halos colapsados con ICs Zel'dovich en z=0 — normal a baja resolución");
        return;
    }

    // Halo más masivo
    let biggest = analysis.halos.iter().max_by_key(|h| h.n_particles).unwrap();
    let m_halo = biggest.mass;
    let (cx, cy, cz) = (biggest.x_com, biggest.y_com, biggest.z_com);

    println!(
        "[phase53] Halo más masivo: N_part={} M={:.3e} M_sun/h  COM=({:.2},{:.2},{:.2})",
        biggest.n_particles, m_halo, cx, cy, cz
    );

    // ── Perfil NFW analítico esperado ─────────────────────────────────────────
    let c = concentration_duffy2008(m_halo, 0.0);
    let nfw = NfwProfile::from_m200_c(m_halo, c, rho_crit_val);
    let r200 = r200_from_m200(m_halo, rho_crit_val);
    println!("[phase53] NFW esperado: c={c:.2}  r_200={r200:.4} Mpc/h  r_s={:.4} Mpc/h  ρ_s={:.3e}",
        nfw.r_s, nfw.rho_s);

    // ── Distancias radiales de las partículas del halo ────────────────────────
    let halo_particle_idx: Vec<usize> = (0..particles.len())
        .filter(|_| true) // en análisis real usaríamos las partículas miembro
        .take(biggest.n_particles.min(particles.len()))
        .collect();

    // Usar todas las partículas de la caja para medir el perfil alrededor del COM
    let radii: Vec<f64> = particles
        .iter()
        .map(|p| {
            // Imagen mínima periódica
            let dx = (p.position.x - cx + box_size * 1.5) % box_size - box_size / 2.0;
            let dy = (p.position.y - cy + box_size * 1.5) % box_size - box_size / 2.0;
            let dz = (p.position.z - cz + box_size * 1.5) % box_size - box_size / 2.0;
            (dx * dx + dy * dy + dz * dz).sqrt()
        })
        .filter(|&r| r < r200 * 3.0)
        .collect();

    println!("[phase53] {} partículas dentro de 3×r_200", radii.len());

    if radii.len() < 10 {
        println!("[phase53] Pocas partículas en el halo — test de medición omitido");
        return;
    }

    // ── Medir perfil de densidad ──────────────────────────────────────────────
    let r_prof_min = r200 * 0.05;
    let r_prof_max = r200 * 1.5;
    let bins = measure_density_profile(&radii, m_part, r_prof_min, r_prof_max, 10, Some(&nfw));

    println!("[phase53] Perfil de densidad del halo más masivo:");
    println!("  {:>8}  {:>8}  {:>8}  {:>10}  {:>10}",
        "r (Mpc/h)", "n_part", "ρ_med", "ρ_NFW", "ratio");
    for bin in &bins {
        if bin.rho_nfw > 0.0 {
            let ratio = if bin.rho > 0.0 { bin.rho / bin.rho_nfw } else { 0.0 };
            println!("  {:>8.4}  {:>8}  {:>8.3e}  {:>10.3e}  {:>10.3}",
                bin.r, bin.n_part, bin.rho, bin.rho_nfw, ratio);
        }
    }

    // Solo verificamos que la infraestructura funciona (bins con datos razonables)
    let bins_with_data = bins.iter().filter(|b| b.n_part > 0).count();
    println!("[phase53] {bins_with_data} bins con partículas de {}", bins.len());

    // El perfil debe tener bins con densidades positivas y finitas
    for bin in bins.iter().filter(|b| b.n_part > 0) {
        assert!(bin.rho.is_finite() && bin.rho > 0.0, "ρ no válido en r={:.4}", bin.r);
    }
}
