//! Phase 63 — Análisis in-situ en el loop stepping.
//!
//! Verifica la configuración `InsituAnalysisSection` y la lógica de intervalo.
//! El análisis in-situ en el loop real se prueba mediante el binario CLI.

use gadget_ng_analysis::AnalysisParams;
use gadget_ng_core::{InsituAnalysisSection, Particle, Vec3};

fn skip() -> bool {
    std::env::var("PHASE63_SKIP")
        .map(|v| v == "1")
        .unwrap_or(false)
}

/// Verifica que InsituAnalysisSection::default() tiene enabled=false y interval=0.
#[test]
fn phase63_default_config_disabled() {
    let cfg = InsituAnalysisSection::default();
    assert!(
        !cfg.enabled,
        "por defecto el análisis in-situ debe estar desactivado"
    );
    assert_eq!(cfg.interval, 0, "interval default = 0");
    assert_eq!(cfg.pk_mesh, 32, "pk_mesh default = 32");
    assert!((cfg.fof_b - 0.2).abs() < 1e-10, "fof_b default = 0.2");
    assert_eq!(cfg.fof_min_part, 20, "fof_min_part default = 20");
}

/// Verifica que el campo interval controla correctamente cuándo se ejecuta el análisis.
#[test]
fn phase63_interval_logic() {
    if skip() {
        return;
    }

    let cfg = InsituAnalysisSection {
        enabled: true,
        interval: 5,
        pk_mesh: 8,
        fof_b: 0.2,
        fof_min_part: 2,
        xi_bins: 0,
        output_dir: None,
        ..Default::default()
    };

    // Con interval=5, los pasos 5, 10, 15 deben disparar análisis; 3, 7 no.
    for step in [1u64, 2, 3, 4, 7, 8, 11, 13] {
        let should = cfg.enabled && cfg.interval > 0 && step % cfg.interval == 0;
        assert!(
            !should,
            "paso {step} no debe disparar análisis con interval=5"
        );
    }
    for step in [5u64, 10, 15, 20] {
        let should = cfg.enabled && cfg.interval > 0 && step % cfg.interval == 0;
        assert!(should, "paso {step} debe disparar análisis con interval=5");
    }
}

/// Verifica que la lógica de enabled=false previene la ejecución.
#[test]
fn phase63_disabled_no_output() {
    if skip() {
        return;
    }

    let cfg = InsituAnalysisSection {
        enabled: false,
        interval: 5,
        pk_mesh: 8,
        fof_b: 0.2,
        fof_min_part: 2,
        xi_bins: 0,
        output_dir: None,
        ..Default::default()
    };

    // Con enabled=false, el análisis no debe ejecutarse aunque el paso sea múltiplo.
    let should_run = cfg.enabled && cfg.interval > 0 && 5u64.is_multiple_of(cfg.interval);
    assert!(
        !should_run,
        "con enabled=false no debe ejecutarse el análisis"
    );
}

/// Verifica que AnalysisParams tiene Default funcional con pk_mesh válido.
#[test]
fn phase63_analysis_params_defaults() {
    let p = AnalysisParams {
        box_size: 1.0,
        ..Default::default()
    };
    assert!(p.pk_mesh > 0, "pk_mesh debe ser > 0");
    assert!(p.b > 0.0 && p.b < 1.0, "b debe estar en (0, 1)");
}

/// Verifica que el análisis in-situ sobre partículas uniformes produce P(k) finito.
#[test]
fn phase63_pk_finite_on_uniform() {
    if skip() {
        return;
    }

    // Generar lattice pequeño.
    let n_side = 4usize;
    let box_size = 1.0f64;
    let step = 1.0 / n_side as f64;
    let mut gid = 0usize;
    let particles: Vec<Particle> = (0..n_side)
        .flat_map(|ix| (0..n_side).flat_map(move |iy| (0..n_side).map(move |iz| (ix, iy, iz))))
        .map(|(ix, iy, iz)| {
            let p = Particle::new(
                gid,
                1.0,
                Vec3::new(
                    (ix as f64 + 0.5) * step,
                    (iy as f64 + 0.5) * step,
                    (iz as f64 + 0.5) * step,
                ),
                Vec3::zero(),
            );
            gid += 1;
            p
        })
        .collect();

    let params = AnalysisParams {
        box_size,
        b: 0.2,
        min_particles: 2,
        pk_mesh: 4,
        ..Default::default()
    };

    let result = gadget_ng_analysis::analyse(&particles, &params);
    for bin in &result.power_spectrum {
        assert!(
            bin.pk.is_finite(),
            "P(k) debe ser finito: k={}, pk={}",
            bin.k,
            bin.pk
        );
        assert!(bin.k > 0.0, "k debe ser positivo");
    }
}
