/// Phase 139 — RMHD: MHD especial-relativista
///
/// Tests: lorentz_factor=1 para v=0, γ→∞ para v→c, primitivización Newton-Raphson,
///        v supralumínica retorna None, em_energy_density correcto, advance_srmhd.
use gadget_ng_core::{MhdSection, Particle, Vec3};
use gadget_ng_mhd::{advance_srmhd, em_energy_density, lorentz_factor, srmhd_conserved_to_primitive, C_LIGHT};

const GAMMA_AD: f64 = 4.0 / 3.0; // gas relativista

// ── 1. Lorentz γ=1 para v=0 ──────────────────────────────────────────────

#[test]
fn lorentz_factor_zero_vel() {
    let gamma = lorentz_factor(Vec3::zero(), C_LIGHT);
    assert!((gamma - 1.0).abs() < 1e-12, "γ(v=0) debe ser 1: {gamma}");
}

// ── 2. γ crece con velocidad ──────────────────────────────────────────────

#[test]
fn lorentz_factor_increases_with_v() {
    let g1 = lorentz_factor(Vec3::new(0.1, 0.0, 0.0), C_LIGHT);
    let g5 = lorentz_factor(Vec3::new(0.5, 0.0, 0.0), C_LIGHT);
    let g9 = lorentz_factor(Vec3::new(0.9, 0.0, 0.0), C_LIGHT);
    assert!(g1 < g5 && g5 < g9, "γ debe crecer: {g1:.4} < {g5:.4} < {g9:.4}");
}

// ── 3. v ≥ c → ∞ ──────────────────────────────────────────────────────────

#[test]
fn lorentz_factor_superluminal() {
    let gamma = lorentz_factor(Vec3::new(C_LIGHT, 0.0, 0.0), C_LIGHT);
    assert!(gamma.is_infinite(), "v=c debe dar γ=∞");
}

// ── 4. Primitivización Newton-Raphson: gas en reposo sin B ─────────────────

#[test]
fn primitive_rest_no_b() {
    // Gas en reposo: D=ρ, S=0, τ≈(γ_ad-1)ρε=P, B=0
    let rho0 = 1.0;
    let p0 = 0.5;
    let eps0 = p0 / ((GAMMA_AD - 1.0) * rho0);
    let d = rho0;          // D = γρ = ρ (reposo)
    let tau = p0 + rho0 * eps0 - d; // τ = E − D ≈ P/(γ_ad-1)
    let result = srmhd_conserved_to_primitive(
        d, [0.0, 0.0, 0.0], tau.max(0.0), [0.0, 0.0, 0.0], GAMMA_AD, C_LIGHT,
    );
    assert!(result.is_some(), "Primitivización debe converger");
    let (rho, vel, p) = result.unwrap();
    assert!((rho - rho0).abs() / rho0 < 0.1, "ρ = {rho:.4}, esperado {rho0}");
    assert!(vel.iter().all(|v| v.abs() < 1e-3), "v debe ser ~0");
    assert!(p >= 0.0, "P debe ser positiva");
}

// ── 5. em_energy_density correcto ─────────────────────────────────────────

#[test]
fn em_energy_density_correct() {
    let b = Vec3::new(3.0, 4.0, 0.0);
    let u_em = em_energy_density(b);
    // u_EM = B²/2 = (9+16)/2 = 12.5
    assert!((u_em - 12.5).abs() < 1e-12, "u_EM = {u_em:.4}, esperado 12.5");
}

// ── 6. advance_srmhd no modifica partículas sub-relativistas ──────────────

#[test]
fn srmhd_no_effect_below_threshold() {
    let mut particles = vec![
        Particle::new_gas(0, 1.0, Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.05, 0.0, 0.0), 1.0, 0.3),
    ];
    let pos_before = particles[0].position;
    advance_srmhd(&mut particles, 0.01, C_LIGHT, 0.1);
    // |v|/c = 0.05 < threshold=0.1 → no aplica corrección relativista → posición no cambia por SRMHD
    assert_eq!(particles[0].position.x, pos_before.x,
        "Partícula sub-relativista no debe cambiar posición");
}
