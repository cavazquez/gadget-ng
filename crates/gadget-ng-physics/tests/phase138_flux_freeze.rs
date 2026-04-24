/// Phase 138 — Freeze-out de B en ICM: criterio β-plasma + B ∝ ρ^{2/3}
///
/// Tests: B=0 no cambia, beta>>beta_freeze aplica escala, beta<<beta_freeze no cambia,
///        B ∝ ρ^{2/3} correcto, mean_gas_density, flux_freeze_error.
use gadget_ng_core::{Particle, Vec3};
use gadget_ng_mhd::{apply_flux_freeze, flux_freeze_error, mean_gas_density};

const GAMMA: f64 = 5.0 / 3.0;

fn gas_with_b(id: usize, mass: f64, h: f64, u: f64, b: Vec3) -> Particle {
    let mut p = Particle::new_gas(id, mass, Vec3::new(id as f64 * 0.1, 0.0, 0.0),
                                  Vec3::zero(), u, h);
    p.b_field = b;
    p
}

// ── 1. B=0 → no-op ──────────────────────────────────────────────────────

#[test]
fn zero_b_no_change() {
    let mut particles = vec![
        gas_with_b(0, 1.0, 0.3, 1.0, Vec3::zero()),
    ];
    apply_flux_freeze(&mut particles, GAMMA, 100.0, 1.0);
    assert_eq!(particles[0].b_field.x, 0.0);
}

// ── 2. β >> β_freeze → aplica escala B ∝ ρ^{2/3} ────────────────────────

#[test]
fn high_beta_applies_scale() {
    // β elevado: campo muy débil (B=1e-9) y gas caliente (u grande)
    // rho_ref << rho_actual → escala > 1 → B aumenta
    let b0 = 1e-9;
    let h = 0.3;
    let u = 1e12; // alta P_th → β >> 1
    let mut particles = vec![
        gas_with_b(0, 1.0, h, u, Vec3::new(b0, 0.0, 0.0)),
    ];
    let rho = 1.0 / (h * h * h);
    let rho_ref = rho * 0.1; // partícula 10x comprimida respecto a ref
    apply_flux_freeze(&mut particles, GAMMA, 100.0, rho_ref);
    // B debe aumentar con la compresión
    assert!(particles[0].b_field.x > b0, "B debe aumentar por compresión: {:.4e} > {:.4e}",
            particles[0].b_field.x, b0);
}

// ── 3. β << β_freeze → no cambia ─────────────────────────────────────────

#[test]
fn low_beta_no_change() {
    // β << 100: campo fuerte (B=1e6) → campo magnético domina
    let b_val = 1e6_f64;
    let mut particles = vec![
        gas_with_b(0, 1.0, 0.3, 1.0, Vec3::new(b_val, 0.0, 0.0)),
    ];
    let b_before = particles[0].b_field.x;
    let rho_ref = 1.0 / (0.3_f64 * 0.3 * 0.3);
    apply_flux_freeze(&mut particles, GAMMA, 100.0, rho_ref);
    assert_eq!(particles[0].b_field.x, b_before, "β bajo: B no debe cambiar");
}

// ── 4. Escala correcta B ∝ ρ^{2/3} ──────────────────────────────────────

#[test]
fn scale_follows_rho_power_2_3() {
    let b0 = 1e-9;
    let h = 0.3;
    let rho = 1.0 / (h * h * h);
    let rho_ref = rho / 8.0; // 8× compresión → B debe multiplicarse por 8^(2/3) = 4
    let mut particles = vec![
        gas_with_b(0, 1.0, h, 1e12, Vec3::new(b0, 0.0, 0.0)),
    ];
    apply_flux_freeze(&mut particles, GAMMA, 100.0, rho_ref);
    let expected = b0 * 8.0_f64.powf(2.0 / 3.0);
    let actual = particles[0].b_field.x;
    assert!((actual / expected - 1.0).abs() < 1e-10,
        "B_actual = {actual:.4e}, B_expected = {expected:.4e}");
}

// ── 5. mean_gas_density con partículas uniformes ──────────────────────────

#[test]
fn mean_density_uniform() {
    let particles: Vec<Particle> = (0..5).map(|i| {
        gas_with_b(i, 1.0, 0.3, 1.0, Vec3::zero())
    }).collect();
    let rho_expected = 1.0 / (0.3_f64 * 0.3 * 0.3);
    let rho_actual = mean_gas_density(&particles);
    assert!((rho_actual / rho_expected - 1.0).abs() < 1e-10,
        "rho_mean = {rho_actual:.4e}, esperado {rho_expected:.4e}");
}

// ── 6. flux_freeze_error = 0 cuando B = B0 * (ρ/ρ0)^{2/3} ───────────────

#[test]
fn flux_freeze_error_exact_zero() {
    let b0 = 1.0_f64;
    let rho0 = 1.0_f64;
    let rho = 8.0_f64;
    let b_exact = b0 * (rho / rho0).powf(2.0 / 3.0);
    let err = flux_freeze_error(b_exact, b0, rho, rho0);
    assert!(err < 1e-14, "Error debe ser 0: {err:.2e}");
}
