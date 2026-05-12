//! Phase 184 — Warm / fuzzy dark matter.

use gadget_ng_core::{
    DarkMatterModel, DarkMatterSection, fdm_half_mode_k, fdm_quantum_pressure_cs2,
    fdm_transfer_suppression, wdm_half_mode_k, wdm_transfer_suppression,
};

#[test]
fn wdm_transfer_is_unity_on_large_scales_and_suppressed_on_small_scales() {
    let low_k = wdm_transfer_suppression(1.0e-4, 3.0, 0.315, 0.674);
    let high_k = wdm_transfer_suppression(100.0, 3.0, 0.315, 0.674);

    assert!((low_k - 1.0).abs() < 1.0e-5);
    assert!(high_k < 0.5);
}

#[test]
fn heavier_wdm_has_larger_half_mode_scale() {
    let light = wdm_half_mode_k(1.0, 0.315, 0.674);
    let heavy = wdm_half_mode_k(5.0, 0.315, 0.674);

    assert!(heavy > light);
}

#[test]
fn fuzzy_dm_transfer_cuts_high_k_and_mass_shifts_cutoff() {
    let k_half_light = fdm_half_mode_k(0.5, 0.315, 0.674);
    let k_half_heavy = fdm_half_mode_k(5.0, 0.315, 0.674);
    let t_low_k = fdm_transfer_suppression(0.01, 1.0, 0.315, 0.674);
    let t_high_k = fdm_transfer_suppression(100.0, 1.0, 0.315, 0.674);

    assert!(k_half_heavy > k_half_light);
    assert!(t_low_k > 0.999);
    assert!(t_high_k < 0.2);
}

#[test]
fn fuzzy_quantum_pressure_proxy_scales_with_k_and_a() {
    let small_k = fdm_quantum_pressure_cs2(1.0, 1.0, 1.0);
    let large_k = fdm_quantum_pressure_cs2(2.0, 1.0, 1.0);
    let early = fdm_quantum_pressure_cs2(1.0, 1.0, 0.5);

    assert!(large_k > small_k * 15.9);
    assert!(early > small_k);
}

#[test]
fn dark_matter_section_serde_accepts_warm_and_fuzzy_models() {
    let warm: DarkMatterSection = toml::from_str(
        r#"
enabled = true
model = "warm"
m_wdm_kev = 2.5
"#,
    )
    .expect("warm dark matter section should deserialize");
    assert_eq!(warm.model, DarkMatterModel::Warm);
    assert_eq!(warm.m_wdm_kev, 2.5);

    let fuzzy: DarkMatterSection = toml::from_str(
        r#"
enabled = true
model = "fuzzy"
m_fdm_22 = 3.0
"#,
    )
    .expect("fuzzy dark matter section should deserialize");
    assert_eq!(fuzzy.model, DarkMatterModel::Fuzzy);
    assert_eq!(fuzzy.m_fdm_22, 3.0);
}
