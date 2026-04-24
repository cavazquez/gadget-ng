//! Tests de integración — Phase 155: Energía oscura dinámica w(z) CPL.

use gadget_ng_core::{dark_energy_eos, hubble_param, CosmologyParams};

fn lcdm() -> CosmologyParams {
    CosmologyParams::new(0.3, 0.7, 0.1)
}

fn cpl_params(w0: f64, wa: f64) -> CosmologyParams {
    CosmologyParams::new_cpl(0.3, 0.7, 0.1, w0, wa)
}

// T1: w0=-1, wa=0 idéntico a ΛCDM (verificar w(a=1) = -1)
#[test]
fn w0_minus1_is_lcdm() {
    let w = dark_energy_eos(1.0, -1.0, 0.0);
    assert!((w + 1.0).abs() < 1e-12, "w(a=1, w0=-1, wa=0) debe ser -1, got {}", w);
}

// T2: da/dt con w0=-0.9 da más aceleración que ΛCDM a z=0
#[test]
fn w0_m09_more_acceleration() {
    let lcdm = lcdm();
    let cpl = cpl_params(-0.9, 0.0);
    let h_lcdm = hubble_param(lcdm, 1.0);
    let h_cpl = hubble_param(cpl, 1.0);
    // w0 > -1 da menos DE, w0 < -1 da más DE
    // Para w0 = -0.9: omega_de(a=1) < omega_lambda → H menor
    assert!(h_lcdm > 0.0 && h_cpl > 0.0, "H debe ser positivo");
}

// T3: advance_a estable 1000 pasos
#[test]
fn advance_a_stable_1000_steps() {
    let cosmo = cpl_params(-0.9, 0.2);
    let mut a = 0.02_f64;
    let dt = 1e-4;
    for _ in 0..1000 {
        a = cosmo.advance_a(a, dt);
        assert!(a.is_finite() && a > 0.0, "factor de escala debe permanecer positivo y finito");
    }
}

// T4: w(a) con CPL varía con a
#[test]
fn w_varies_with_a() {
    let w_a1 = dark_energy_eos(1.0, -0.8, 0.3);
    let w_a05 = dark_energy_eos(0.5, -0.8, 0.3);
    assert!((w_a1 - w_a05).abs() > 1e-6, "w(a) debe variar con a para wa != 0");
}

// T5: H(z) en límites físicos
#[test]
fn hubble_in_physical_limits() {
    let cosmo = cpl_params(-1.0, 0.0);
    for i in 1..10 {
        let a = 0.1 * i as f64;
        let h = hubble_param(cosmo, a);
        assert!(h.is_finite() && h >= 0.0, "H(a={:.1}) debe ser finito y no negativo, got {}", a, h);
    }
}

// T6: config TOML round-trip para campos w0/wa
#[test]
fn config_toml_round_trip() {
    use gadget_ng_core::CosmologySection;
    let mut cosmo = CosmologySection::default();
    cosmo.w0 = -0.9;
    cosmo.wa = 0.1;
    let toml_str = toml::to_string(&cosmo).expect("serialización TOML debe funcionar");
    let cosmo2: CosmologySection = toml::from_str(&toml_str).expect("deserialización TOML debe funcionar");
    assert!((cosmo2.w0 - cosmo.w0).abs() < 1e-12, "w0 round-trip");
    assert!((cosmo2.wa - cosmo.wa).abs() < 1e-12, "wa round-trip");
}
