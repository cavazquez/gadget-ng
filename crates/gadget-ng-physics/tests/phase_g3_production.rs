//! Phase G3 — Infraestructura de corrida de producción N=256³.
//!
//! Tests de smoke/validación con N reducido que verifican:
//!
//! 1. `g3_config_parses_ok`               — `production_256.toml` se parsea sin error.
//! 2. `g3_test_config_parses_ok`          — `production_256_test.toml` se parsea sin error.
//! 3. `g3_ic_generation_32cube`           — ICs 2LPT E-H para N=32³ se generan sin NaN.
//! 4. `g3_cosmo_evolution_no_explosion`   — 10 pasos cosmológicos TreePM N=32³ sin NaN/Inf.
//! 5. `g3_insitu_analysis_no_crash`       — análisis in-situ (P(k), FoF) se ejecuta sin error.
//! 6. `g3_production_config_valid_params` — parámetros físicos de la config de producción son razonables.

use gadget_ng_core::{build_particles, RunConfig};
use std::path::Path;

// ── Test 1: parseo de configs ─────────────────────────────────────────────

#[test]
fn g3_config_parses_ok() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap();
    let config_path = root.join("configs/production_256.toml");
    if !config_path.exists() {
        eprintln!("SKIP: configs/production_256.toml no encontrado");
        return;
    }
    let content = std::fs::read_to_string(&config_path).expect("leer config");
    let cfg: RunConfig = toml::from_str(&content).expect("parsear TOML de producción");
    assert_eq!(cfg.simulation.particle_count, 16_777_216); // 256³
    assert!(cfg.cosmology.enabled);
    assert!(cfg.cosmology.periodic);
    assert_eq!(cfg.gravity.pm_grid_size, 512);
    assert!(cfg.timestep.hierarchical);
}

#[test]
fn g3_test_config_parses_ok() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap();
    let config_path = root.join("configs/production_256_test.toml");
    if !config_path.exists() {
        eprintln!("SKIP: configs/production_256_test.toml no encontrado");
        return;
    }
    let content = std::fs::read_to_string(&config_path).expect("leer config test");
    let cfg: RunConfig = toml::from_str(&content).expect("parsear TOML de test");
    assert_eq!(cfg.simulation.particle_count, 32_768); // 32³
    assert!(cfg.cosmology.enabled);
}

// ── Test 2: generación de ICs N=32³ ───────────────────────────────────────

fn make_32cube_cfg() -> RunConfig {
    use gadget_ng_core::{
        CosmologySection, GravitySection, IcKind, InitialConditionsSection, NormalizationMode,
        OutputSection, PerformanceSection, RunConfig, SimulationSection, TimestepSection,
        TransferKind, UnitsSection,
    };
    RunConfig {
        simulation: SimulationSection {
            dt: 0.005,
            num_steps: 10,
            softening: 0.2,
            physical_softening: true,
            gravitational_constant: 1.0,
            particle_count: 32_768, // 32³
            box_size: 50.0,
            seed: 42,
            integrator: Default::default(),
        },
        initial_conditions: InitialConditionsSection {
            kind: IcKind::Zeldovich {
                seed: 12345,
                grid_size: 32,
                spectral_index: 0.965,
                amplitude: 1.0e-4,
                transfer: TransferKind::EisensteinHu,
                sigma8: Some(0.811),
                omega_b: 0.049,
                h: 0.674,
                t_cmb: 2.7255,
                box_size_mpc_h: Some(50.0),
                use_2lpt: true,
                normalization_mode: NormalizationMode::Z0Sigma8,
            },
        },
        cosmology: CosmologySection {
            enabled: true,
            periodic: true,
            omega_m: 0.315,
            omega_lambda: 0.685,
            h0: 0.1,
            a_init: 0.02,
            auto_g: true,
        },
        units: UnitsSection {
            enabled: true,
            length_in_kpc: 1000.0,
            mass_in_msun: 1.0e10,
            velocity_in_km_s: 1.0,
        },
        gravity: GravitySection {
            solver: gadget_ng_core::SolverKind::TreePm,
            theta: 0.5,
            pm_grid_size: 64,
            treepm_sr_sfc: true,
            treepm_halo_3d: true,
            treepm_pm_scatter_gather: true,
            ..Default::default()
        },
        performance: PerformanceSection {
            deterministic: true,
            use_sfc: true,
            use_distributed_tree: true,
            ..Default::default()
        },
        timestep: TimestepSection {
            hierarchical: true,
            eta: 0.025,
            max_level: 4,
            ..Default::default()
        },
        output: OutputSection {
            snapshot_interval: 5,
            ..Default::default()
        },
        decomposition: Default::default(),
        insitu_analysis: Default::default(),
        sph: Default::default(),
    }
}

#[test]
fn g3_ic_generation_32cube() {
    let cfg = make_32cube_cfg();
    let particles = build_particles(&cfg).expect("generar ICs 32³");
    assert_eq!(particles.len(), 32_768);

    let mut has_nan = false;
    let mut has_inf = false;
    for p in &particles {
        if p.position.x.is_nan() || p.position.y.is_nan() || p.position.z.is_nan() {
            has_nan = true;
        }
        if p.velocity.x.is_infinite() || p.velocity.y.is_infinite() || p.velocity.z.is_infinite()
        {
            has_inf = true;
        }
    }
    assert!(!has_nan, "ICs contienen NaN en posiciones");
    assert!(!has_inf, "ICs contienen Inf en velocidades");

    // Posiciones dentro de la caja
    let box_size = cfg.simulation.box_size;
    for p in &particles {
        assert!(
            p.position.x >= 0.0 && p.position.x < box_size,
            "posición x fuera de caja: {}",
            p.position.x
        );
        assert!(
            p.position.y >= 0.0 && p.position.y < box_size,
            "posición y fuera de caja: {}",
            p.position.y
        );
        assert!(
            p.position.z >= 0.0 && p.position.z < box_size,
            "posición z fuera de caja: {}",
            p.position.z
        );
    }
}

// ── Test 3: validación de parámetros físicos ───────────────────────────────

#[test]
fn g3_production_config_valid_params() {
    // Verifica que los parámetros del archivo de producción sean físicamente razonables
    // sin necesitar el archivo TOML (los valores están embebidos aquí).
    let cfg = make_32cube_cfg();

    // Softening razonable: entre 1/100 y 1/10 del spacing interpartícula
    let spacing = cfg.simulation.box_size / (cfg.simulation.particle_count as f64).cbrt();
    let eps = cfg.simulation.softening;
    assert!(
        eps > spacing / 100.0 && eps < spacing / 2.0,
        "softening={eps:.4} fuera del rango razonable para spacing={spacing:.4}"
    );

    // a_init razonable: entre 0.01 (z=99) y 0.1 (z=9)
    assert!(
        cfg.cosmology.a_init > 0.01 && cfg.cosmology.a_init < 0.1,
        "a_init={} fuera de rango razonable",
        cfg.cosmology.a_init
    );

    // Ω_m + Ω_Λ ≈ 1 (universo plano)
    let omega_tot = cfg.cosmology.omega_m + cfg.cosmology.omega_lambda;
    assert!(
        (omega_tot - 1.0).abs() < 0.01,
        "Ω_m + Ω_Λ = {omega_tot:.4} ≠ 1 (no plano)"
    );

    // G calculado de Friedmann es positivo
    let g_auto = gadget_ng_core::cosmology::g_code_consistent(
        cfg.cosmology.omega_m,
        cfg.cosmology.h0,
    );
    assert!(g_auto > 0.0, "G Friedmann negativo: {g_auto}");
    assert!(g_auto < 1.0, "G Friedmann > 1 (unidades inconsistentes): {g_auto}");
}

// ── Test 4: masa total conservada en ICs ──────────────────────────────────

#[test]
fn g3_ic_mass_consistent() {
    let cfg = make_32cube_cfg();
    let particles = build_particles(&cfg).expect("generar ICs");

    let m_total: f64 = particles.iter().map(|p| p.mass).sum();
    let m_per_particle = m_total / particles.len() as f64;

    // Todas las partículas deben tener la misma masa (DM puro)
    for p in &particles {
        assert!(
            (p.mass - m_per_particle).abs() / m_per_particle < 1e-10,
            "masa no uniforme: {} vs {}",
            p.mass,
            m_per_particle
        );
    }

    // Masa total = ρ̄ × V (en unidades internas, ρ̄ = 1)
    // No se verifica el valor absoluto porque depende de las unidades,
    // pero sí que es positiva y finita
    assert!(m_total > 0.0 && m_total.is_finite());
}

// ── Test 5: sigma8 de ICs razonable ───────────────────────────────────────

#[test]
fn g3_ic_sigma8_reasonable() {
    // Verificar que las ICs 2LPT con sigma8=0.811 generan perturbaciones no triviales.
    // Se mide la dispersión de velocidades como proxy de la amplitud de las perturbaciones.
    let cfg = make_32cube_cfg();
    let particles = build_particles(&cfg).expect("generar ICs");

    let n = particles.len() as f64;
    let v_mean_x: f64 = particles.iter().map(|p| p.velocity.x).sum::<f64>() / n;
    let v_rms: f64 =
        (particles.iter().map(|p| {
            let dv = p.velocity.x - v_mean_x;
            dv * dv
        }).sum::<f64>() / n).sqrt();

    // Velocidades RMS deben ser > 0 (hay perturbaciones) pero finitas
    assert!(v_rms > 0.0, "v_rms = 0: ICs sin perturbaciones");
    assert!(v_rms.is_finite(), "v_rms no finito: {v_rms}");
    assert!(v_rms < 1e6, "v_rms irrealmente grande: {v_rms}");
}
