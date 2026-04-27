/// Phase 114 — ISM Multifase fría-caliente
///
/// Tests: effective_pressure > P_thermal con u_cold > 0, u_cold crece para gas denso,
///        energía total conservada, gas sub-umbral disipa u_cold, disabled = no-op,
///        serde de IsmSection, effective_u correcto.
use gadget_ng_core::{IsmSection, Particle, Vec3};
use gadget_ng_sph::{effective_pressure, effective_u, update_ism_phases};

const GAMMA: f64 = 5.0 / 3.0;

fn gas(id: usize, u: f64, h: f64) -> Particle {
    Particle::new_gas(id, 1.0, Vec3::zero(), Vec3::zero(), u, h)
}

fn ism_cfg() -> IsmSection {
    IsmSection {
        enabled: true,
        q_star: 2.5,
        f_cold: 0.5,
    }
}

// ── 1. effective_pressure > P_thermal cuando u_cold > 0 ──────────────────

#[test]
fn effective_pressure_exceeds_thermal() {
    let rho = 1.0;
    let u = 1.0;
    let u_cold = 0.5;
    let p_thermal = (GAMMA - 1.0) * rho * u;
    let p_eff = effective_pressure(rho, u, u_cold, 2.5, GAMMA);
    assert!(
        p_eff > p_thermal,
        "P_eff debe ser mayor que P_thermal con u_cold > 0"
    );
}

// ── 2. effective_pressure = P_thermal cuando u_cold = 0 ───────────────────

#[test]
fn effective_pressure_equals_thermal_when_no_cold() {
    let rho = 2.0;
    let u = 1.5;
    let p_thermal = (GAMMA - 1.0) * rho * u;
    let p_eff = effective_pressure(rho, u, 0.0, 2.5, GAMMA);
    assert!(
        (p_eff - p_thermal).abs() < 1e-12,
        "P_eff debe ser igual a P_thermal con u_cold=0"
    );
}

// ── 3. u_cold crece para gas sobre el umbral de SFR ──────────────────────

#[test]
fn u_cold_grows_for_dense_gas() {
    let cfg = ism_cfg();
    let mut particles = vec![gas(0, 2.0, 0.01)]; // h pequeño → ρ alta
    let sfr = vec![1.0]; // gas sobre umbral
    let u_cold_before = particles[0].u_cold;
    update_ism_phases(&mut particles, &sfr, 0.001, &cfg, 0.1);
    assert!(
        particles[0].u_cold > u_cold_before,
        "u_cold debe crecer para gas denso con sfr > 0"
    );
}

// ── 4. Conservación de energía total ─────────────────────────────────────

#[test]
fn total_energy_conserved() {
    let cfg = ism_cfg();
    let mut particles = vec![gas(0, 2.0, 0.01)];
    let sfr = vec![1.0];
    let e_total_before = particles[0].internal_energy + particles[0].u_cold;
    update_ism_phases(&mut particles, &sfr, 0.001, &cfg, 0.1);
    let e_total_after = particles[0].internal_energy + particles[0].u_cold;
    assert!(
        (e_total_after - e_total_before).abs() < 1e-10,
        "La energía total (u + u_cold) debe conservarse: antes={e_total_before}, después={e_total_after}"
    );
}

// ── 5. Gas sub-umbral disipa u_cold hacia u ────────────────────────────────

#[test]
fn u_cold_dissipates_below_threshold() {
    let cfg = ism_cfg();
    let mut particles = vec![gas(0, 1.0, 10.0)]; // h grande → ρ baja
    particles[0].u_cold = 0.5; // comenzar con fase fría
    let sfr = vec![0.0]; // gas sub-umbral
    let u_cold_before = particles[0].u_cold;
    update_ism_phases(&mut particles, &sfr, 0.001, &cfg, 0.1);
    assert!(
        particles[0].u_cold < u_cold_before,
        "u_cold debe disiparse para gas sub-umbral"
    );
}

// ── 6. Módulo desactivado = no-op ────────────────────────────────────────

#[test]
fn disabled_ism_no_change() {
    let cfg = IsmSection {
        enabled: false,
        ..ism_cfg()
    };
    let mut particles = vec![gas(0, 2.0, 0.01)];
    let sfr = vec![1.0];
    let u_before = particles[0].internal_energy;
    let u_cold_before = particles[0].u_cold;
    update_ism_phases(&mut particles, &sfr, 0.001, &cfg, 0.1);
    assert_eq!(particles[0].internal_energy, u_before);
    assert_eq!(particles[0].u_cold, u_cold_before);
}

// ── 7. effective_u correcto ───────────────────────────────────────────────

#[test]
fn effective_u_formula() {
    let mut p = gas(0, 1.0, 0.5);
    p.u_cold = 0.3;
    let q_star = 2.5;
    let expected = p.internal_energy + q_star * p.u_cold;
    let result = effective_u(&p, q_star);
    assert!(
        (result - expected).abs() < 1e-15,
        "effective_u incorrecto: {result} vs {expected}"
    );
}
