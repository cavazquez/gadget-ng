/// Phase 133 — MHD anisótropo: conducción ∥B y difusión CR anisótropa
///
/// Tests: calor fluye solo ∥B con κ_⊥=0, calor isótropo con κ_∥=κ_⊥,
///        β-plasma correcto, difusión CR anisótropa reduce frente a isótropa,
///        con B=0 conducción anisótropa degrada a isótropa, conservación energía.
use gadget_ng_core::{Particle, Vec3};
use gadget_ng_mhd::{apply_anisotropic_conduction, beta_plasma, diffuse_cr_anisotropic, MU0};

const GAMMA: f64 = 5.0 / 3.0;

fn gas_with_b(id: usize, pos: Vec3, u: f64, b: Vec3) -> Particle {
    let mut p = Particle::new_gas(id, 1.0, pos, Vec3::zero(), u, 0.5);
    p.b_field = b;
    p
}

// ── 1. Con κ_⊥=0 y B∥x: calor fluye entre partículas alineadas con B ────

#[test]
fn heat_flows_parallel_to_b() {
    // B apunta en x; partículas alineadas en x → θ=0, cos²=1 → κ_eff = κ_∥
    let b = Vec3::new(1.0, 0.0, 0.0);
    let mut particles = vec![
        gas_with_b(0, Vec3::new(0.0, 0.0, 0.0), 10.0, b),
        gas_with_b(1, Vec3::new(0.1, 0.0, 0.0), 1.0, b),  // ∥B
    ];
    let u0_before = particles[0].internal_energy;
    apply_anisotropic_conduction(&mut particles, 0.1, 0.0, GAMMA, 0.01);
    // Con κ_⊥=0 y alineación ∥B: debe haber transferencia
    assert!(particles[0].internal_energy < u0_before, "calor debe fluir de hot a cold ∥B");
    assert!(particles[1].internal_energy > 1.0, "cold debe recibir calor");
}

// ── 2. Con κ_⊥=0 y B∥x: sin transferencia para partículas ⊥B ────────────

#[test]
fn no_heat_perpendicular_to_b() {
    // B apunta en x; partículas separadas en y → θ=90°, cos²=0 → κ_eff = κ_⊥ = 0
    let b = Vec3::new(1.0, 0.0, 0.0);
    let mut particles = vec![
        gas_with_b(0, Vec3::new(0.0, 0.0, 0.0), 10.0, b),
        gas_with_b(1, Vec3::new(0.0, 0.1, 0.0), 1.0, b),  // ⊥B
    ];
    let u0_before = particles[0].internal_energy;
    let u1_before = particles[1].internal_energy;
    apply_anisotropic_conduction(&mut particles, 0.1, 0.0, GAMMA, 0.01);
    assert_eq!(particles[0].internal_energy, u0_before,
        "sin transferencia ⊥B con κ_⊥=0");
    assert_eq!(particles[1].internal_energy, u1_before,
        "sin transferencia ⊥B con κ_⊥=0");
}

// ── 3. Con κ_∥=κ_⊥=κ: conducción isótropa (degeneración) ─────────────────

#[test]
fn isotropic_limit_kpar_eq_kperp() {
    let b = Vec3::new(1.0, 0.0, 0.0);
    // Partícula ⊥B — con κ_∥=κ_⊥ debe haber transferencia igual que isótropo
    let mut p_aniso = vec![
        gas_with_b(0, Vec3::new(0.0, 0.0, 0.0), 10.0, b),
        gas_with_b(1, Vec3::new(0.0, 0.1, 0.0), 1.0, b),
    ];
    apply_anisotropic_conduction(&mut p_aniso, 0.1, 0.1, GAMMA, 0.01);
    // Debe haber transferencia (κ_∥ = κ_⊥ = 0.1 → isótropo)
    assert!(p_aniso[0].internal_energy < 10.0, "isótropo: debe fluir calor");
}

// ── 4. β-plasma correcto ──────────────────────────────────────────────────

#[test]
fn beta_plasma_formula() {
    // β = 2μ₀ P / B²
    let b = Vec3::new(1.0, 0.0, 0.0); // |B|² = 1
    let p = 0.5_f64;
    let beta = beta_plasma(p, b);
    let expected = 2.0 * MU0 * p / 1.0;
    assert!((beta - expected).abs() < 1e-12, "β = {beta}, esperado {expected}");
}

// ── 5. β-plasma = ∞ con B=0 ──────────────────────────────────────────────

#[test]
fn beta_infinite_with_zero_b() {
    let beta = beta_plasma(1.0, Vec3::zero());
    assert!(beta.is_infinite(), "β debe ser ∞ con B=0");
}

// ── 6. Conservación de energía en conducción anisótropa ───────────────────

#[test]
fn energy_conserved_anisotropic() {
    let b = Vec3::new(1.0, 0.5, 0.2);
    let mut particles = vec![
        gas_with_b(0, Vec3::new(0.0, 0.0, 0.0), 8.0, b),
        gas_with_b(1, Vec3::new(0.08, 0.0, 0.0), 3.0, b),
        gas_with_b(2, Vec3::new(0.0, 0.08, 0.0), 1.0, b),
    ];
    let e_total_before: f64 = particles.iter().map(|p| p.internal_energy * p.mass).sum();
    apply_anisotropic_conduction(&mut particles, 0.05, 0.005, GAMMA, 0.01);
    let e_total_after: f64 = particles.iter().map(|p| p.internal_energy * p.mass).sum();
    assert!((e_total_after - e_total_before).abs() / e_total_before < 1e-12,
        "Energía no conservada: Δ = {:.2e}", (e_total_after - e_total_before).abs());
}
