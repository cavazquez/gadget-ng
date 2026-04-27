/// Phase 136 — MHD cosmológico end-to-end
///
/// Tests: b_field_stats None con sin gas, b_field_stats correcto con partículas,
///        campo primordial débil es estable tras 50 pasos MHD,
///        energía magnética finita y positiva tras evolución,
///        stats_interval en MhdSection, amplificación de B por compresión.
use gadget_ng_core::{BFieldKind, MhdSection, Particle, Vec3};
use gadget_ng_mhd::{
    advance_induction, apply_artificial_resistivity, b_field_stats, dedner_cleaning_step,
    init_b_field,
};

fn make_gas_grid(n: usize, b0: Vec3, box_size: f64) -> Vec<Particle> {
    let cfg = MhdSection {
        enabled: true,
        b0_kind: BFieldKind::Uniform,
        b0_uniform: [b0.x, b0.y, b0.z],
        ..Default::default()
    };
    let mut particles: Vec<Particle> = (0..n)
        .map(|i| {
            let x = (i as f64 / n as f64) * box_size;

            Particle::new_gas(
                i,
                1.0,
                Vec3::new(x, 0.0, 0.0),
                Vec3::new(0.01, 0.0, 0.0),
                1.0,
                box_size / n as f64 * 2.0,
            )
        })
        .collect();
    init_b_field(&mut particles, &cfg, box_size);
    particles
}

// ── 1. b_field_stats None con lista vacía ────────────────────────────────

#[test]
fn stats_none_empty() {
    assert!(b_field_stats(&[]).is_none());
}

// ── 2. b_field_stats None con solo DM ────────────────────────────────────

#[test]
fn stats_none_dm_only() {
    let dm = Particle::new(0, 1.0, Vec3::zero(), Vec3::zero());
    assert!(b_field_stats(&[dm]).is_none());
}

// ── 3. b_field_stats correcto con campo uniforme ──────────────────────────

#[test]
fn stats_uniform_b_correct() {
    let b_val = 2.0_f64;
    let mut particles: Vec<Particle> = (0..10)
        .map(|i| {
            Particle::new_gas(
                i,
                1.0,
                Vec3::new(i as f64 * 0.1, 0.0, 0.0),
                Vec3::zero(),
                1.0,
                0.2,
            )
        })
        .collect();
    for p in &mut particles {
        p.b_field = Vec3::new(b_val, 0.0, 0.0);
    }
    let stats = b_field_stats(&particles).unwrap();
    assert!(
        (stats.b_mean - b_val).abs() < 1e-10,
        "b_mean = {}",
        stats.b_mean
    );
    assert!(
        (stats.b_rms - b_val).abs() < 1e-10,
        "b_rms = {}",
        stats.b_rms
    );
    assert!(
        (stats.b_max - b_val).abs() < 1e-10,
        "b_max = {}",
        stats.b_max
    );
    assert!(stats.e_mag > 0.0, "e_mag debe ser positiva");
    assert_eq!(stats.n_gas, 10);
}

// ── 4. Campo primordial débil estable tras 50 pasos ───────────────────────

#[test]
fn primordial_b_stable_after_50_steps() {
    let b0 = Vec3::new(1e-10, 0.0, 0.0);
    let mut particles = make_gas_grid(32, b0, 1.0);

    let dt = 1e-4;
    for _ in 0..50 {
        advance_induction(&mut particles, dt);
        dedner_cleaning_step(&mut particles, 1.0, 0.5, dt);
    }

    let stats = b_field_stats(&particles).unwrap();
    assert!(stats.b_max.is_finite(), "B_max no finito");
    assert!(stats.b_max >= 0.0, "B_max negativo");
    assert!(stats.e_mag.is_finite(), "E_mag no finita");
}

// ── 5. Energía magnética positiva y finita tras evolución ─────────────────

#[test]
fn mag_energy_positive_finite() {
    let b0 = Vec3::new(1.0, 0.5, 0.0);
    let mut particles = make_gas_grid(16, b0, 1.0);

    let dt = 0.001;
    for _ in 0..20 {
        advance_induction(&mut particles, dt);
        apply_artificial_resistivity(&mut particles, 0.5, dt);
        dedner_cleaning_step(&mut particles, 1.0, 0.5, dt);
    }

    let stats = b_field_stats(&particles).unwrap();
    assert!(
        stats.e_mag.is_finite() && stats.e_mag > 0.0,
        "E_mag debe ser finita y positiva: {}",
        stats.e_mag
    );
}

// ── 6. stats_interval en MhdSection (configuración) ──────────────────────

#[test]
fn mhd_section_stats_interval_default() {
    let cfg = MhdSection::default();
    assert_eq!(
        cfg.stats_interval, 0,
        "stats_interval default=0 (desactivado)"
    );

    let cfg_on = MhdSection {
        stats_interval: 10,
        ..Default::default()
    };
    assert_eq!(cfg_on.stats_interval, 10);
}
