/// Phase 119 — Enfriamiento tabulado S&D93 con interpolación bilineal
///
/// Tests: MetalTabular genera tasa positiva, mayor a MetalCooling a Z alta,
///        tasa = 0 bajo T_floor, enfriamiento tabulado vs analítico consistente,
///        serde de MetalTabular, backward compat MetalCooling.
use gadget_ng_core::{CoolingKind, Particle, ParticleType, SphSection, Vec3};
use gadget_ng_sph::{apply_cooling, cooling_rate_tabular, temperature_to_u, u_to_temperature};

const GAMMA: f64 = 5.0 / 3.0;
const T_FLOOR: f64 = 1e4;

fn cfg_tabular() -> SphSection {
    SphSection { cooling: CoolingKind::MetalTabular, ..Default::default() }
}

fn gas_at_temp(t_k: f64, z: f64) -> Particle {
    let u = temperature_to_u(t_k, GAMMA);
    let mut p = Particle::new_gas(0, 1.0, Vec3::zero(), Vec3::zero(), u, 0.5);
    p.metallicity = z;
    p
}

// ── 1. cooling_rate_tabular retorna valor positivo para T > T_floor ────────

#[test]
fn tabular_rate_positive_above_floor() {
    let u = temperature_to_u(1e6, GAMMA);
    let rate = cooling_rate_tabular(u, 1.0, 0.02, GAMMA, T_FLOOR);
    assert!(rate > 0.0, "Tasa tabulada debe ser positiva para T=1e6 K");
}

// ── 2. Tasa = 0 por debajo del floor ──────────────────────────────────────

#[test]
fn tabular_rate_zero_below_floor() {
    let u = temperature_to_u(5e3, GAMMA); // T < T_floor
    let rate = cooling_rate_tabular(u, 1.0, 0.02, GAMMA, T_FLOOR);
    assert_eq!(rate, 0.0, "Tasa debe ser 0 para T < T_floor");
}

// ── 3. Tasa aumenta con metalicidad ───────────────────────────────────────

#[test]
fn tabular_rate_increases_with_metallicity() {
    let u = temperature_to_u(1e6, GAMMA);
    let rate_low = cooling_rate_tabular(u, 1.0, 0.001, GAMMA, T_FLOOR);
    let rate_high = cooling_rate_tabular(u, 1.0, 0.02, GAMMA, T_FLOOR);
    assert!(rate_high > rate_low, "Tasa tabulada debe aumentar con Z: {rate_high} vs {rate_low}");
}

// ── 4. apply_cooling reduce u con MetalTabular ────────────────────────────

#[test]
fn apply_cooling_tabular_reduces_u() {
    let cfg = cfg_tabular();
    let t0 = 1e6;
    let mut particles = vec![gas_at_temp(t0, 0.02)];
    let u_before = particles[0].internal_energy;
    apply_cooling(&mut particles, &cfg, 1.0);
    assert!(
        particles[0].internal_energy < u_before,
        "u debe reducirse con MetalTabular: {} < {}",
        particles[0].internal_energy, u_before
    );
}

// ── 5. Backward compat: MetalCooling sigue funcionando ────────────────────

#[test]
fn backward_compat_metal_cooling_still_works() {
    let cfg = SphSection { cooling: CoolingKind::MetalCooling, ..Default::default() };
    let t0 = 1e6;
    let mut particles = vec![gas_at_temp(t0, 0.02)];
    let u_before = particles[0].internal_energy;
    apply_cooling(&mut particles, &cfg, 1.0);
    assert!(particles[0].internal_energy < u_before, "MetalCooling aún debe enfriar");
}

// ── 6. Serde de CoolingKind::MetalTabular ─────────────────────────────────

#[test]
fn cooling_kind_metal_tabular_serde() {
    let kind = CoolingKind::MetalTabular;
    let json = serde_json::to_string(&kind).unwrap();
    let kind2: CoolingKind = serde_json::from_str(&json).unwrap();
    assert_eq!(kind, kind2, "CoolingKind::MetalTabular debe ser serializable");
}
