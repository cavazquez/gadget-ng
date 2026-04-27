/// Phase 137 — Polvo + RT: absorción UV por polvo
///
/// Tests: tau_dust=0 con dust_to_gas=0, tau_dust crece con D/G,
///        tau_dust crece con kappa_dust_uv, atenuación exp(-tau) es [0,1],
///        kappa_dust_uv default en DustSection, radiation_gas_coupling con polvo.
use gadget_ng_core::{DustSection, Particle, Vec3};
use gadget_ng_rt::{M1Params, RadiationField, radiation_gas_coupling_step_with_dust};
use gadget_ng_sph::dust_uv_opacity;

// ── 1. tau_dust = 0 con D/G = 0 ──────────────────────────────────────────

#[test]
fn tau_zero_with_no_dust() {
    let tau = dust_uv_opacity(1000.0, 0.0, 1.0, 0.1);
    assert_eq!(tau, 0.0);
}

// ── 2. tau_dust crece con D/G ─────────────────────────────────────────────

#[test]
fn tau_grows_with_dust_to_gas() {
    let tau1 = dust_uv_opacity(1000.0, 0.001, 1.0, 0.1);
    let tau2 = dust_uv_opacity(1000.0, 0.01, 1.0, 0.1);
    assert!(
        tau2 > tau1,
        "tau debe crecer con D/G: {tau1:.4e} vs {tau2:.4e}"
    );
}

// ── 3. tau_dust crece con kappa_dust_uv ──────────────────────────────────

#[test]
fn tau_grows_with_kappa() {
    let tau1 = dust_uv_opacity(100.0, 0.01, 1.0, 0.1);
    let tau2 = dust_uv_opacity(1000.0, 0.01, 1.0, 0.1);
    assert!(
        (tau2 / tau1 - 10.0).abs() < 1e-10,
        "tau ∝ kappa: ratio = {}",
        tau2 / tau1
    );
}

// ── 4. Atenuación exp(-tau) ∈ [0, 1] ─────────────────────────────────────

#[test]
fn attenuation_in_range() {
    for &dtg in &[0.0, 0.001, 0.005, 0.01] {
        let tau = dust_uv_opacity(1000.0, dtg, 1.0, 0.1);
        let att = (-tau).exp();
        assert!(
            (0.0..=1.0).contains(&att),
            "att={att:.4} fuera de [0,1] con D/G={dtg}"
        );
    }
}

// ── 5. kappa_dust_uv default en DustSection ──────────────────────────────

#[test]
fn dust_section_kappa_default() {
    let cfg = DustSection::default();
    assert_eq!(
        cfg.kappa_dust_uv, 1000.0,
        "kappa_dust_uv default debe ser 1000"
    );
}

// ── 6. radiation_gas_coupling_step_with_dust funciona ────────────────────

#[test]
fn coupling_with_dust_runs() {
    let mut particles: Vec<Particle> = (0..4)
        .map(|i| {
            let mut p = Particle::new_gas(
                i,
                1.0,
                Vec3::new(0.25 * i as f64, 0.5, 0.5),
                Vec3::zero(),
                1e10,
                0.2,
            );
            p.dust_to_gas = 0.005;
            p
        })
        .collect();

    let mut rad = RadiationField::uniform(4, 4, 4, 0.25, 1.0);
    let params = M1Params::default();

    let u_before: f64 = particles.iter().map(|p| p.internal_energy).sum();
    radiation_gas_coupling_step_with_dust(&mut particles, &mut rad, &params, 1000.0, 0.01, 1.0);
    let u_after: f64 = particles.iter().map(|p| p.internal_energy).sum();

    // Con campo radiativo → fotocalentamiento puede ocurrir
    assert!(
        u_after >= u_before || u_after.is_finite(),
        "u debe ser finita"
    );
}
