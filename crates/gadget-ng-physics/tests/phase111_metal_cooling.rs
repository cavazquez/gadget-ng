/// Phase 111 — Enfriamiento por metales (MetalCooling)
///
/// Tests: Λ_metal > Λ_HHe a T=10⁶K con Z=Z_sun, suelo de temperatura,
///        Z=0 produce mismo resultado que AtomicHHe, monotonía en Z,
///        serde de CoolingKind, apply_cooling MetalCooling.
use gadget_ng_core::{CoolingKind, Particle, SphSection, Vec3};
use gadget_ng_sph::{apply_cooling, cooling_rate_atomic, cooling_rate_metal, temperature_to_u};

const GAMMA: f64 = 5.0 / 3.0;
const T_FLOOR: f64 = 1e4;
const Z_SUN: f64 = 0.0127;

fn u_at(t_k: f64) -> f64 { temperature_to_u(t_k, GAMMA) }

fn base_sph(cooling: CoolingKind) -> SphSection {
    SphSection { cooling, t_floor_k: T_FLOOR, ..SphSection::default() }
}

// ── 1. MetalCooling > AtomicHHe a T=10⁶K con Z=Z_sun ─────────────────────

#[test]
fn metal_cooling_exceeds_atomic_at_high_z() {
    let u = u_at(1e6);
    let lambda_hhe = cooling_rate_atomic(u, 1.0, GAMMA, T_FLOOR);
    let lambda_metal = cooling_rate_metal(u, 1.0, Z_SUN, GAMMA, T_FLOOR);
    assert!(lambda_metal > lambda_hhe, "Λ_metal debe superar Λ_HHe con Z=Z_sun a T=10⁶K");
}

// ── 2. Z=0 produce mismo resultado que AtomicHHe ─────────────────────────

#[test]
fn metal_cooling_z_zero_equals_atomic() {
    let u = u_at(1e6);
    let lambda_hhe = cooling_rate_atomic(u, 1.0, GAMMA, T_FLOOR);
    let lambda_zero_z = cooling_rate_metal(u, 1.0, 0.0, GAMMA, T_FLOOR);
    assert!((lambda_zero_z - lambda_hhe).abs() < 1e-20, "Z=0 debe dar igual que AtomicHHe");
}

// ── 3. Suelo de temperatura: Λ=0 por debajo del floor ─────────────────────

#[test]
fn metal_cooling_zero_below_floor() {
    let u_cold = u_at(T_FLOOR * 0.5);
    let lambda = cooling_rate_metal(u_cold, 1.0, Z_SUN, GAMMA, T_FLOOR);
    assert_eq!(lambda, 0.0, "No debe haber enfriamiento por debajo del floor");
}

// ── 4. Monotonía: mayor Z → mayor tasa de enfriamiento ───────────────────

#[test]
fn cooling_rate_monotone_in_z() {
    let u = u_at(5e5);
    let lambda_low = cooling_rate_metal(u, 1.0, 0.001, GAMMA, T_FLOOR);
    let lambda_mid = cooling_rate_metal(u, 1.0, Z_SUN, GAMMA, T_FLOOR);
    let lambda_high = cooling_rate_metal(u, 1.0, 0.1, GAMMA, T_FLOOR);
    assert!(lambda_low <= lambda_mid, "Λ debe crecer con Z");
    assert!(lambda_mid <= lambda_high, "Λ debe crecer con Z");
}

// ── 5. Serde de CoolingKind::MetalCooling ────────────────────────────────

#[test]
fn cooling_kind_metal_serde_roundtrip() {
    let kind = CoolingKind::MetalCooling;
    let json = serde_json::to_string(&kind).unwrap();
    let kind2: CoolingKind = serde_json::from_str(&json).unwrap();
    assert_eq!(kind2, CoolingKind::MetalCooling);
}

// ── 6. apply_cooling despacha a MetalCooling correctamente ────────────────

#[test]
fn apply_cooling_metal_reduces_energy() {
    let cfg = base_sph(CoolingKind::MetalCooling);
    let u_hot = u_at(1e6);
    let mut p = Particle::new_gas(0, 1.0, Vec3::zero(), Vec3::zero(), u_hot, 0.5);
    p.metallicity = Z_SUN;

    let u_before = p.internal_energy;
    apply_cooling(&mut [p.clone()], &cfg, 0.01);

    // Para verificar que la energía baja, corremos sobre un slice mutable
    let mut particles = vec![p];
    apply_cooling(&mut particles, &cfg, 0.1);
    assert!(particles[0].internal_energy < u_before, "la energía debe bajar con MetalCooling");
}
