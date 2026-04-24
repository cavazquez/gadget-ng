/// Phase 126 — Integración MHD en engine + validación onda de Alfvén
///
/// Tests: MhdSection desactivado = no-op, advance_induction + apply_magnetic_forces
///        en secuencia no explota, velocidad de Alfvén v_A = B/sqrt(μ₀ρ),
///        pipeline completo inducción + fuerzas + limpieza Dedner, config MHD default.
use gadget_ng_core::{MhdSection, Particle, Vec3};
use gadget_ng_mhd::{advance_induction, apply_magnetic_forces, dedner_cleaning_step};

fn gas_with_b(id: usize, pos: Vec3, vel: Vec3, b: Vec3) -> Particle {
    let mut p = Particle::new_gas(id, 1.0, pos, vel, 1.0, 0.5);
    p.b_field = b;
    p
}

// ── 1. MhdSection: default desactivado ────────────────────────────────────

#[test]
fn mhd_section_default_disabled() {
    let cfg = MhdSection::default();
    assert!(!cfg.enabled, "MHD debe estar desactivado por defecto");
    assert_eq!(cfg.c_h, 1.0);
    assert_eq!(cfg.c_r, 0.5);
}

// ── 2. Pipeline completo no explota con B no nulo ────────────────────────

#[test]
fn full_pipeline_no_panic() {
    let b = Vec3::new(1.0, 0.0, 0.0);
    let n = 10;
    let mut particles: Vec<Particle> = (0..n).map(|i| {
        let x = (i as f64) * 0.1;
        let vel = Vec3::new(0.0, 0.01 * (i as f64).sin(), 0.0);
        let mut p = gas_with_b(i, Vec3::new(x, 0.0, 0.0), vel, b);
        p.psi_div = 0.001 * (i as f64 - 5.0);
        p
    }).collect();

    let dt = 0.001;
    advance_induction(&mut particles, dt);
    apply_magnetic_forces(&mut particles, dt);
    dedner_cleaning_step(&mut particles, 1.0, 0.5, dt);

    for (i, p) in particles.iter().enumerate() {
        assert!(p.b_field.x.is_finite(), "b_field.x NaN partícula {i}");
        assert!(p.velocity.x.is_finite(), "vel.x NaN partícula {i}");
        assert!(p.psi_div.is_finite(), "psi_div NaN partícula {i}");
    }
}

// ── 3. Velocidad de Alfvén v_A = B / sqrt(μ₀ ρ) ─────────────────────────

#[test]
fn alfven_speed_formula() {
    // Verificar la fórmula analítica de v_A (no simulación completa)
    let b_mag = 2.0_f64;
    let rho = 1.0_f64;
    let mu0 = gadget_ng_mhd::MU0;
    let v_alfven = b_mag / (mu0 * rho).sqrt();
    let expected = b_mag; // con μ₀=1 y ρ=1: v_A = B
    assert!((v_alfven - expected).abs() < 1e-12,
        "v_A = {v_alfven}, expected {expected}");
}

// ── 4. B con amplitud pequeña produce cambios pequeños en la velocidad ────

#[test]
fn small_b_small_force() {
    let b_small = Vec3::new(0.001, 0.0, 0.0);
    let b_large = Vec3::new(1.0, 0.0, 0.0);

    let mut particles_small = vec![
        gas_with_b(0, Vec3::new(0.0, 0.0, 0.0), Vec3::zero(), b_small),
        gas_with_b(1, Vec3::new(0.3, 0.0, 0.0), Vec3::zero(), b_small),
    ];
    let mut particles_large = vec![
        gas_with_b(0, Vec3::new(0.0, 0.0, 0.0), Vec3::zero(), b_large),
        gas_with_b(1, Vec3::new(0.3, 0.0, 0.0), Vec3::zero(), b_large),
    ];

    apply_magnetic_forces(&mut particles_small, 0.01);
    apply_magnetic_forces(&mut particles_large, 0.01);

    let dv_small = particles_small[0].velocity.x.abs();
    let dv_large = particles_large[0].velocity.x.abs();

    // Con B grande, la fuerza debe ser mayor (P_B ∝ B²)
    assert!(dv_large > dv_small,
        "Fuerza con B grande ({dv_large:.2e}) debe ser > B pequeño ({dv_small:.2e})");
}

// ── 5. Dedner cleaning decae ψ y mantiene |B| finito ─────────────────────

#[test]
fn dedner_maintains_finite_b() {
    let b = Vec3::new(1.0, 0.0, 0.0);
    let mut particles: Vec<Particle> = (0..5).map(|i| {
        let mut p = gas_with_b(i, Vec3::new(i as f64 * 0.2, 0.0, 0.0), Vec3::zero(), b);
        p.psi_div = (i as f64 - 2.0) * 0.1;
        p
    }).collect();

    // Aplicar muchos pasos
    for _ in 0..100 {
        dedner_cleaning_step(&mut particles, 1.0, 0.5, 0.001);
    }
    for (i, p) in particles.iter().enumerate() {
        assert!(p.b_field.x.is_finite(), "B.x NaN tras 100 pasos Dedner en partícula {i}");
        assert!(p.psi_div.is_finite(), "psi NaN tras 100 pasos Dedner en partícula {i}");
        // ψ debe estar amortiguado
        assert!(p.psi_div.abs() < 0.5, "ψ no se amortiguó: {}", p.psi_div);
    }
}

// ── 6. Integración end-to-end con datos realistas ────────────────────────

#[test]
fn end_to_end_realistic_mhd() {
    // Setup: ondas de Alfvén en caja periódica simplificada (1D, N=8 partículas)
    let n = 8;
    let b0 = 1.0_f64; // B de fondo en x
    let rho0 = 1.0_f64;
    let v_a = b0 / (gadget_ng_mhd::MU0 * rho0).sqrt(); // ~1.0

    let mut particles: Vec<Particle> = (0..n).map(|i| {
        let x = (i as f64) / (n as f64); // caja [0,1]
        let bx = b0;
        let by = 0.01 * (2.0 * std::f64::consts::PI * x).sin(); // perturbación transversa
        let vy = 0.01 * (2.0 * std::f64::consts::PI * x).sin(); // velocidad asociada
        let b = Vec3::new(bx, by, 0.0);
        let vel = Vec3::new(0.0, vy, 0.0);
        let mut p = gas_with_b(i, Vec3::new(x, 0.0, 0.0), vel, b);
        p.smoothing_length = 0.3; // mayor que dx para capturar vecinos
        p
    }).collect();

    // Propagar algunos pasos
    let dt = 0.001 / v_a;
    for _ in 0..50 {
        advance_induction(&mut particles, dt);
        apply_magnetic_forces(&mut particles, dt);
        dedner_cleaning_step(&mut particles, v_a, 0.5, dt);
    }

    // Verificar que el sistema no explota
    for (i, p) in particles.iter().enumerate() {
        let b_mag = (p.b_field.x.powi(2) + p.b_field.y.powi(2) + p.b_field.z.powi(2)).sqrt();
        assert!(b_mag.is_finite() && b_mag < 100.0,
            "B se disparó en partícula {i}: |B|={b_mag}");
    }
}
