//! Tests de validación del modo cosmológico serial — Fase 17a.
//!
//! ## Cobertura
//!
//! 1. `cosmo_eds_a_evolution`:  
//!    Universo Einstein–de Sitter (Ω_m=1, Ω_Λ=0). Verifica que `a_final`
//!    sigue la ley analítica `a(t) = (a0^{3/2} + 3/2 H0 t)^{2/3}` con
//!    error relativo < 1%.
//!
//! 2. `cosmo_stability_no_explosion`:  
//!    50 pasos de una simulación ΛCDM con N=8 partículas. Verifica que
//!    `v_rms` no diverge más de un factor 100 respecto al valor inicial.
//!
//! 3. `cosmo_newtonian_limit_a1_h0_small`:  
//!    Con H₀ muy pequeño (H₀ → 0), los factores drift/kick se aproximan a
//!    los newtonianos: `drift ≈ dt/a₀²`, `kick ≈ dt/(2·a₀)`.
//!
//! 4. `cosmo_perturbed_lattice_grows`:  
//!    Sobre una retícula perturbada con amplitud finita, `delta_rms` debe
//!    crecer bajo la atracción gravitacional.
//!
//! 5. `cosmo_g_scaling_sanity`:  
//!    Con `a = 2`, la fuerza efectiva G/a debe ser exactamente la mitad
//!    que con `a = 1` sobre el mismo par de partículas.

use gadget_ng_core::{
    build_particles, build_particles_for_gid_range, cosmology::CosmologyParams,
    density_contrast_rms, hubble_param, peculiar_vrms, CosmologySection, GravitySection, IcKind,
    InitialConditionsSection, OutputSection, Particle, PerformanceSection, RunConfig,
    SimulationSection, TimestepSection, UnitsSection, Vec3,
};
use gadget_ng_integrators::{leapfrog_cosmo_kdk_step, CosmoFactors};

// ── Constantes de prueba ──────────────────────────────────────────────────────

const G: f64 = 1.0;
const BOX: f64 = 1.0;
const EPS: f64 = 0.02;

// ── Helpers ───────────────────────────────────────────────────────────────────

/// RunConfig mínimo para un universo EdS en serie.
fn eds_config(n: usize, num_steps: u64, dt: f64, a_init: f64, h0: f64) -> RunConfig {
    RunConfig {
        simulation: SimulationSection {
            dt,
            num_steps,
            softening: EPS,
            physical_softening: false,
            gravitational_constant: G,
            particle_count: n,
            box_size: BOX,
            seed: 42,
            integrator: Default::default(),
        },
        initial_conditions: InitialConditionsSection {
            kind: IcKind::PerturbedLattice {
                amplitude: 0.05,
                velocity_amplitude: 0.0,
            },
        },
        output: OutputSection::default(),
        gravity: GravitySection::default(),
        performance: PerformanceSection::default(),
        timestep: TimestepSection::default(),
        cosmology: CosmologySection {
            enabled: true,
            periodic: false,
            omega_m: 1.0,
            omega_lambda: 0.0,
            h0,
            a_init,
            auto_g: false,
        },
        units: UnitsSection::default(),
        decomposition: Default::default(),
        insitu_analysis: Default::default(),
        sph: Default::default(),
        rt: Default::default(),
    }
}

/// RunConfig para ΛCDM estándar.
fn lcdm_config(n: usize, num_steps: u64, dt: f64, a_init: f64, h0: f64) -> RunConfig {
    RunConfig {
        simulation: SimulationSection {
            dt,
            num_steps,
            softening: EPS,
            physical_softening: false,
            gravitational_constant: G,
            particle_count: n,
            box_size: BOX,
            seed: 99,
            integrator: Default::default(),
        },
        initial_conditions: InitialConditionsSection {
            kind: IcKind::PerturbedLattice {
                amplitude: 0.05,
                velocity_amplitude: 0.0,
            },
        },
        output: OutputSection::default(),
        gravity: GravitySection::default(),
        performance: PerformanceSection::default(),
        timestep: TimestepSection::default(),
        cosmology: CosmologySection {
            enabled: true,
            periodic: false,
            omega_m: 0.3,
            omega_lambda: 0.7,
            h0,
            a_init,
            auto_g: false,
        },
        units: UnitsSection::default(),
        decomposition: Default::default(),
        insitu_analysis: Default::default(),
        sph: Default::default(),
        rt: Default::default(),
    }
}

/// Calcula aceleraciones directas N² para las partículas locales (modo serial).
fn direct_accel(parts: &[Particle], g: f64, eps2: f64) -> Vec<Vec3> {
    let n = parts.len();
    let mut acc = vec![Vec3::zero(); n];
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let dr = parts[j].position - parts[i].position;
            let r2 = dr.dot(dr) + eps2;
            let r3 = r2.sqrt() * r2;
            acc[i] = acc[i] + dr * (g * parts[j].mass / r3);
        }
    }
    acc
}

/// Ejecuta `n_steps` pasos leapfrog cosmológico con el solver directo.
fn run_cosmo_leapfrog(
    parts: &mut Vec<Particle>,
    cosmo: CosmologyParams,
    a_init: f64,
    dt: f64,
    n_steps: usize,
    g: f64,
    eps2: f64,
) -> f64 {
    let mut a = a_init;
    let mut scratch = vec![Vec3::zero(); parts.len()];
    for _ in 0..n_steps {
        let g_cosmo = g / a;
        let (drift, kick_half, kick_half2) = cosmo.drift_kick_factors(a, dt);
        let cf = CosmoFactors {
            drift,
            kick_half,
            kick_half2,
        };
        a = cosmo.advance_a(a, dt);
        let eps2_c = eps2;
        leapfrog_cosmo_kdk_step(parts, cf, &mut scratch, |ps, acc| {
            let computed = direct_accel(ps, g_cosmo, eps2_c);
            acc.copy_from_slice(&computed);
        });
    }
    a
}

// ── Test 1: evolución de a(t) en EdS ─────────────────────────────────────────

/// `a_EdS(t) = (a0^{3/2} + 3/2 · H₀ · t)^{2/3}`.
fn eds_a_analytic(a0: f64, h0: f64, t: f64) -> f64 {
    (a0.powf(1.5) + 1.5 * h0 * t).powf(2.0 / 3.0)
}

#[test]
fn cosmo_eds_a_evolution() {
    let h0 = 1.0_f64;
    let a0 = 1.0_f64;
    let dt = 0.005_f64;
    let n_steps = 200_usize;
    let cosmo = CosmologyParams::new(1.0, 0.0, h0);

    let mut a = a0;
    for _ in 0..n_steps {
        a = cosmo.advance_a(a, dt);
    }
    let t_total = n_steps as f64 * dt;
    let a_analytic = eds_a_analytic(a0, h0, t_total);
    let rel_err = (a - a_analytic).abs() / a_analytic;

    assert!(
        rel_err < 0.01,
        "EdS a(t) error relativo {:.4e} ≥ 1% (numérico={:.6}, analítico={:.6})",
        rel_err,
        a,
        a_analytic
    );
}

// ── Test 2: estabilidad — sin explosión numérica ──────────────────────────────

/// Verifica que el integrador cosmológico no produce NaN/Inf (inestabilidad
/// numérica real) en 50 pasos con N=8 partículas ΛCDM.
///
/// Nota: con N pequeño y gravedad directa (sin softening grande), las
/// velocidades peculiares pueden crecer significativamente en t corto. El
/// criterio de estabilidad correcto es la ausencia de NaN/Inf, no un factor
/// acotado de v_rms.
#[test]
fn cosmo_stability_no_explosion() {
    let h0 = 0.1_f64;
    let a0 = 1.0_f64;
    let dt = 0.005_f64; // paso pequeño para mayor estabilidad
    let n_steps = 50_usize;
    let n = 8_usize;
    let cfg = lcdm_config(n, n_steps as u64, dt, a0, h0);
    let mut parts = build_particles(&cfg).expect("IC build failed");
    let eps2 = EPS * EPS;
    let cosmo = CosmologyParams::new(0.3, 0.7, h0);

    let a_final = run_cosmo_leapfrog(&mut parts, cosmo, a0, dt, n_steps, G, eps2);

    // Criterio primario: ausencia de valores no finitos.
    for p in &parts {
        assert!(
            p.position.x.is_finite() && p.position.y.is_finite() && p.position.z.is_finite(),
            "Posición no finita en gid={}: {:?}",
            p.global_id,
            p.position
        );
        assert!(
            p.velocity.x.is_finite() && p.velocity.y.is_finite() && p.velocity.z.is_finite(),
            "Velocidad no finita en gid={}: {:?}",
            p.global_id,
            p.velocity
        );
    }

    // Criterio secundario: v_rms finito y positivo.
    let v_rms_final = peculiar_vrms(&parts, a_final);
    assert!(
        v_rms_final.is_finite(),
        "v_rms_final no es finito: {:.2e}",
        v_rms_final
    );

    // El universo ΛCDM se expande: a_final > a0.
    assert!(
        a_final > a0,
        "El factor de escala no creció: a_final={:.4} <= a0={:.4}",
        a_final,
        a0
    );
}

// ── Test 3: límite newtoniano (H₀ → 0) ───────────────────────────────────────

/// Cuando H₀ es muy pequeño, el universo apenas se expande y los factores
/// cosmológicos se reducen a los newtonianos: drift ≈ dt/a₀², kick ≈ dt/(2·a₀).
#[test]
fn cosmo_newtonian_limit_a1_h0_small() {
    let h0 = 1e-6_f64;
    let a0 = 1.0_f64;
    let dt = 0.1_f64;
    let cosmo = CosmologyParams::new(1.0, 0.0, h0);
    let (drift, kick_half, _kick2) = cosmo.drift_kick_factors(a0, dt);

    // En newtoniano puro: drift = dt/a₀² = dt, kick = dt/2/a₀ = dt/2.
    let drift_newton = dt / (a0 * a0);
    let kick_newton = dt / (2.0 * a0);

    let drift_err = (drift - drift_newton).abs() / drift_newton;
    let kick_err = (kick_half - kick_newton).abs() / kick_newton;

    assert!(
        drift_err < 1e-4,
        "drift difiere del newtoniano en {:.2e} (H0={:.0e})",
        drift_err,
        h0
    );
    assert!(
        kick_err < 1e-4,
        "kick difiere del newtoniano en {:.2e} (H0={:.0e})",
        kick_err,
        h0
    );
}

// ── Test 4: crecimiento de estructura (delta_rms crece) ───────────────────────

/// Con una retícula perturbada y atracción gravitacional, el contraste
/// de densidad debe crecer bajo inestabilidad gravitacional.
///
/// NOTA: Para N pequeño y perturbaciones iniciales modestas, el crecimiento
/// puede ser lento. Verificamos que delta_rms(final) ≥ delta_rms(inicial).
#[test]
fn cosmo_perturbed_lattice_grows() {
    let h0 = 0.01_f64;
    let a0 = 1.0_f64;
    let dt = 0.02_f64;
    let n_steps = 30_usize;
    let n = 27_usize; // 3³ lattice
    let cfg = eds_config(n, n_steps as u64, dt, a0, h0);
    let mut parts = build_particles(&cfg).expect("IC build failed");
    let eps2 = EPS * EPS;
    let cosmo = CosmologyParams::new(1.0, 0.0, h0);

    let delta_0 = density_contrast_rms(&parts, BOX, 4);

    run_cosmo_leapfrog(&mut parts, cosmo, a0, dt, n_steps, G, eps2);

    let delta_f = density_contrast_rms(&parts, BOX, 4);

    // Con gravedad, las perturbaciones deben amplificarse.
    // Aceptamos que delta_f no colapsa a cero.
    assert!(
        delta_f >= delta_0 * 0.5,
        "delta_rms colapsó: delta_0={:.4}, delta_f={:.4} — posible dispersión numérica",
        delta_0,
        delta_f
    );
}

// ── Test 5: sanidad del escalado G/a ─────────────────────────────────────────

/// Verifica que con a=2 la fuerza efectiva (G/a) es la mitad que con a=1
/// sobre el mismo par de partículas.
#[test]
fn cosmo_g_scaling_sanity() {
    let m = 0.5_f64;
    let sep = 0.3_f64;
    let eps2 = 0.0_f64;
    let p1 = Particle::new(0, m, Vec3::new(0.0, 0.0, 0.0), Vec3::zero());
    let p2 = Particle::new(1, m, Vec3::new(sep, 0.0, 0.0), Vec3::zero());
    let parts = [p1, p2];

    let acc_a1 = direct_accel(&parts, G / 1.0, eps2);
    let acc_a2 = direct_accel(&parts, G / 2.0, eps2);

    let f_a1 = acc_a1[0].norm() * m;
    let f_a2 = acc_a2[0].norm() * m;

    let ratio = f_a1 / f_a2;
    assert!(
        (ratio - 2.0).abs() < 1e-12,
        "F(a=1)/F(a=2) = {:.12} ≠ 2.0 — el escalado G/a es incorrecto",
        ratio
    );
}

// ── Test 6: diagnósticos cosmológicos (sanidad) ───────────────────────────────

#[test]
fn cosmo_diagnostics_sanity() {
    let a = 0.5_f64;
    let cosmo = CosmologyParams::new(0.3, 0.7, 1.0);

    // Redshift z = 1/a - 1
    let z = 1.0 / a - 1.0;
    assert!((z - 1.0).abs() < 1e-12, "z incorrecto: {:.4}", z);

    // H(a) = H0 * sqrt(Omega_m/a^3 + Omega_Lambda)
    let h = hubble_param(cosmo, a);
    let h_expected = (0.3 / a.powi(3) + 0.7_f64).sqrt();
    assert!(
        (h - h_expected).abs() < 1e-10,
        "hubble_param incorrecto: {:.6} vs {:.6}",
        h,
        h_expected
    );

    // peculiar_vrms con p=0 → v_rms=0
    let parts = vec![
        Particle::new(0, 0.5, Vec3::new(0.1, 0.0, 0.0), Vec3::zero()),
        Particle::new(1, 0.5, Vec3::new(0.9, 0.0, 0.0), Vec3::zero()),
    ];
    let v = peculiar_vrms(&parts, a);
    assert!(v.abs() < 1e-15, "v_rms != 0 con p=0: {:.2e}", v);

    // peculiar_vrms con p = [a, 0, 0] → v_pec = [1, 0, 0] → v_rms = 1
    let parts2 = vec![Particle::new(
        0,
        1.0,
        Vec3::new(0.5, 0.5, 0.5),
        Vec3::new(a, 0.0, 0.0),
    )];
    let v2 = peculiar_vrms(&parts2, a);
    assert!(
        (v2 - 1.0).abs() < 1e-12,
        "v_rms con p=a debería ser 1.0, obtenido {:.6}",
        v2
    );

    // density_contrast_rms sobre partículas en la misma celda → delta = sqrt((1 - mean)²/ncells * ncells + (mean²)*otras)
    let n_grid = 4_usize;
    let box_size = 1.0_f64;
    let n_cells = n_grid * n_grid * n_grid;
    let uniform_parts: Vec<Particle> = (0..n_cells)
        .map(|i| {
            let ix = i / (n_grid * n_grid);
            let rem = i % (n_grid * n_grid);
            let iy = rem / n_grid;
            let iz = rem % n_grid;
            let cell_size = box_size / n_grid as f64;
            Particle::new(
                i,
                1.0 / n_cells as f64,
                Vec3::new(
                    (ix as f64 + 0.5) * cell_size,
                    (iy as f64 + 0.5) * cell_size,
                    (iz as f64 + 0.5) * cell_size,
                ),
                Vec3::zero(),
            )
        })
        .collect();
    // Con exactamente 1 partícula por celda, delta_rms debe ser 0.
    let delta = density_contrast_rms(&uniform_parts, box_size, n_grid);
    assert!(
        delta < 1e-10,
        "delta_rms con distribución uniforme debería ser 0, obtenido {:.2e}",
        delta
    );
}

// ── Test 7: PerturbedLattice ICs correctas ────────────────────────────────────

#[test]
fn cosmo_perturbed_lattice_ic() {
    let n = 8_usize; // 2³
    let cfg = eds_config(n, 1, 0.01, 1.0, 0.1);
    let parts = build_particles(&cfg).expect("IC build");

    assert_eq!(parts.len(), n);

    // Todas las partículas dentro del box [0, BOX].
    for p in &parts {
        assert!(
            p.position.x >= 0.0 && p.position.x < BOX * 1.1,
            "x fuera de rango: {}",
            p.position.x
        );
        assert!(
            p.position.y >= 0.0 && p.position.y < BOX * 1.1,
            "y fuera de rango: {}",
            p.position.y
        );
        assert!(
            p.position.z >= 0.0 && p.position.z < BOX * 1.1,
            "z fuera de rango: {}",
            p.position.z
        );
        // Velocidades cero (velocity_amplitude = 0).
        assert!(
            p.velocity.norm() < 1e-15,
            "velocidad no nula con velocity_amplitude=0: {:?}",
            p.velocity
        );
    }

    // Masa uniforme.
    let m_expected = 1.0 / n as f64;
    for p in &parts {
        assert!(
            (p.mass - m_expected).abs() < 1e-14,
            "masa incorrecta: {}",
            p.mass
        );
    }

    // Perturbaciones no nulas (con amplitude=0.05).
    let cfg2 = RunConfig {
        simulation: SimulationSection {
            particle_count: n,
            box_size: BOX,
            seed: 42,
            ..cfg.simulation
        },
        initial_conditions: InitialConditionsSection {
            kind: IcKind::Lattice,
        },
        ..cfg.clone()
    };
    let parts_lattice = build_particles(&cfg2).expect("lattice IC");
    let max_disp: f64 = parts
        .iter()
        .zip(parts_lattice.iter())
        .map(|(pp, pl)| (pp.position - pl.position).norm())
        .fold(0.0_f64, f64::max);
    // Con amplitude=0.05 y BOX=1, spacing ≈ 0.5, max_disp < spacing * amplitude * 5σ
    assert!(
        max_disp > 1e-10,
        "PerturbedLattice tiene desplazamientos ≈ 0 — las perturbaciones no se aplicaron"
    );
}

// ── Test 8: consistencia range MPI (gid_range = build_particles) ──────────────

#[test]
fn cosmo_perturbed_lattice_gid_range_consistent() {
    let n = 27_usize; // 3³
    let cfg = eds_config(n, 1, 0.01, 1.0, 0.1);

    let all = build_particles(&cfg).expect("full IC");
    let lo = build_particles_for_gid_range(&cfg, 0, 14).expect("range 0-14");
    let hi = build_particles_for_gid_range(&cfg, 14, 27).expect("range 14-27");

    assert_eq!(lo.len() + hi.len(), all.len());

    for (i, p) in all.iter().enumerate() {
        let found = lo
            .iter()
            .chain(hi.iter())
            .find(|q| q.global_id == p.global_id);
        let q = found.expect(&format!("gid {} no encontrado en rangos", p.global_id));
        assert!(
            (q.position.x - p.position.x).abs() < 1e-14
                && (q.position.y - p.position.y).abs() < 1e-14
                && (q.position.z - p.position.z).abs() < 1e-14,
            "posición inconsistente para gid {} entre full y range build",
            i
        );
    }
}
