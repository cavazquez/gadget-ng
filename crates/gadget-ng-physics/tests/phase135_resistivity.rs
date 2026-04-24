/// Phase 135 — Resistividad numérica artificial (Price 2008)
///
/// Tests: alpha_b=0 → no-op, resistividad suaviza discontinuidad B,
///        campo uniforme no cambia con resistividad, B finito tras muchos pasos,
///        resistividad reduce gradiente |ΔB|/h, alpha_b escala linealmente.
use gadget_ng_core::{Particle, Vec3};
use gadget_ng_mhd::apply_artificial_resistivity;

fn gas_with_b_vel(id: usize, pos: Vec3, b: Vec3, vel: Vec3) -> Particle {
    let mut p = Particle::new_gas(id, 1.0, pos, vel, 1.0, 0.2);
    p.b_field = b;
    p
}

// ── 1. alpha_b=0 → no-op ─────────────────────────────────────────────────

#[test]
fn zero_alpha_no_op() {
    let vel = Vec3::new(1.0, 0.0, 0.0);
    let mut particles = vec![
        gas_with_b_vel(0, Vec3::new(0.0, 0.0, 0.0), Vec3::new(2.0, 0.0, 0.0), vel),
        gas_with_b_vel(1, Vec3::new(0.1, 0.0, 0.0), Vec3::new(0.0, 0.0, 0.0), vel),
    ];
    let b0_before = particles[0].b_field.x;
    let b1_before = particles[1].b_field.x;
    apply_artificial_resistivity(&mut particles, 0.0, 0.01);
    assert_eq!(particles[0].b_field.x, b0_before, "alpha_b=0: B no debe cambiar");
    assert_eq!(particles[1].b_field.x, b1_before, "alpha_b=0: B no debe cambiar");
}

// ── 2. Resistividad suaviza discontinuidad de B ───────────────────────────

#[test]
fn resistivity_smooths_b_discontinuity() {
    // velocidades distintas para que v_sig > 0
    let mut particles = vec![
        gas_with_b_vel(0, Vec3::new(0.0, 0.0, 0.0), Vec3::new(2.0, 0.0, 0.0), Vec3::new(2.0, 0.0, 0.0)),
        gas_with_b_vel(1, Vec3::new(0.1, 0.0, 0.0), Vec3::new(0.0, 0.0, 0.0), Vec3::new(-1.0, 0.0, 0.0)),
    ];
    let b0_before = particles[0].b_field.x;
    let b1_before = particles[1].b_field.x;
    apply_artificial_resistivity(&mut particles, 0.5, 0.01);
    let delta_b0 = particles[0].b_field.x - b0_before;
    let delta_b1 = particles[1].b_field.x - b1_before;
    assert!(delta_b0 <= 0.0, "B alta debe decrecer: Δ={delta_b0:.4e}");
    assert!(delta_b1 >= 0.0, "B baja debe crecer: Δ={delta_b1:.4e}");
}

// ── 3. Campo uniforme → sin cambio ────────────────────────────────────────

#[test]
fn uniform_b_no_change() {
    let vel = Vec3::new(1.0, 0.0, 0.0);
    let b = Vec3::new(1.0, 0.0, 0.0);
    let mut particles = vec![
        gas_with_b_vel(0, Vec3::new(0.0, 0.0, 0.0), b, vel),
        gas_with_b_vel(1, Vec3::new(0.1, 0.0, 0.0), b, vel),
        gas_with_b_vel(2, Vec3::new(0.2, 0.0, 0.0), b, vel),
    ];
    let b_before: Vec<f64> = particles.iter().map(|p| p.b_field.x).collect();
    apply_artificial_resistivity(&mut particles, 0.5, 0.01);
    for (i, p) in particles.iter().enumerate() {
        assert!((p.b_field.x - b_before[i]).abs() < 1e-10,
            "Campo uniforme no debe cambiar: p{i}, ΔB={:.2e}", (p.b_field.x - b_before[i]).abs());
    }
}

// ── 4. B finito tras muchos pasos de resistividad ─────────────────────────

#[test]
fn b_remains_finite_after_many_steps() {
    let vel = Vec3::new(1.0, 0.0, 0.0);
    let mut particles = vec![
        gas_with_b_vel(0, Vec3::new(0.0, 0.0, 0.0), Vec3::new(5.0, 0.0, 0.0), vel),
        gas_with_b_vel(1, Vec3::new(0.1, 0.0, 0.0), Vec3::new(0.0, 0.0, 0.0), vel),
        gas_with_b_vel(2, Vec3::new(0.2, 0.0, 0.0), Vec3::new(3.0, 0.0, 0.0), vel),
    ];
    for _ in 0..100 {
        apply_artificial_resistivity(&mut particles, 0.5, 0.001);
    }
    for (i, p) in particles.iter().enumerate() {
        assert!(p.b_field.x.is_finite(), "p{i}: B.x no finito");
        assert!(p.b_field.y.is_finite(), "p{i}: B.y no finito");
    }
}

// ── 5. Resistividad reduce el gradiente de B ─────────────────────────────

#[test]
fn resistivity_reduces_b_gradient() {
    let make = || vec![
        gas_with_b_vel(0, Vec3::new(0.0, 0.0, 0.0), Vec3::new(10.0, 0.0, 0.0), Vec3::new(3.0, 0.0, 0.0)),
        gas_with_b_vel(1, Vec3::new(0.1, 0.0, 0.0), Vec3::new(0.0, 0.0, 0.0), Vec3::new(-2.0, 0.0, 0.0)),
    ];
    let mut particles = make();
    let db_before = (particles[0].b_field.x - particles[1].b_field.x).abs();

    for _ in 0..20 {
        apply_artificial_resistivity(&mut particles, 0.5, 0.001);
    }

    let db_after = (particles[0].b_field.x - particles[1].b_field.x).abs();
    assert!(db_after < db_before,
        "Gradiente ΔB debe reducirse: {db_before:.4} → {db_after:.4}");
}

// ── 6. Alpha_b mayor → mayor difusión en un paso ─────────────────────────

#[test]
fn larger_alpha_b_more_diffusion() {
    let make = || vec![
        gas_with_b_vel(0, Vec3::new(0.0, 0.0, 0.0), Vec3::new(4.0, 0.0, 0.0), Vec3::new(3.0, 0.0, 0.0)),
        gas_with_b_vel(1, Vec3::new(0.1, 0.0, 0.0), Vec3::new(0.0, 0.0, 0.0), Vec3::new(-2.0, 0.0, 0.0)),
    ];
    let mut p_low = make();
    let mut p_high = make();
    let b0 = p_low[0].b_field.x;
    apply_artificial_resistivity(&mut p_low, 0.1, 0.01);
    apply_artificial_resistivity(&mut p_high, 0.9, 0.01);
    let d_low = (p_low[0].b_field.x - b0).abs();
    let d_high = (p_high[0].b_field.x - b0).abs();
    assert!(d_high > d_low, "alpha_b mayor → más difusión: {d_high:.4e} vs {d_low:.4e}");
}
