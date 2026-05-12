//! Phase 179 — química deuterio/HD + cooling primordial.

use gadget_ng_rt::{
    ChemState, F_D, cooling_rate_hd, k_d_ionization_exchange, k_d_recombination_exchange,
    k_hd_destruction, k_hd_formation, solve_chemistry_implicit,
};

#[test]
fn chem_state_tracks_deuterium_species() {
    let st = ChemState::neutral();
    assert_eq!(ChemState::species_count(), 12);
    assert!((st.deuterium_nuclei_fraction() - F_D).abs() < 1e-15);
    assert_eq!(st.x_d, F_D);
    assert_eq!(st.x_dp, 0.0);
    assert_eq!(st.x_hd, 0.0);
}

#[test]
fn deuterium_rates_are_positive() {
    for t in [30.0_f64, 100.0, 1e3, 1e4] {
        assert!(k_d_ionization_exchange(t) > 0.0);
        assert!(k_d_recombination_exchange(t) > 0.0);
        assert!(k_hd_formation(t) > 0.0);
        assert!(k_hd_destruction(t) >= 0.0);
    }
}

#[test]
fn hd_forms_from_h2_and_deuterium_ion() {
    let mut st = ChemState::neutral();
    st.x_h2 = 1e-3;
    st.x_hi = 1.0 - 2.0 * st.x_h2;
    st.x_hii = 1e-4;
    st.x_dp = 0.5 * F_D;
    st.x_d = F_D - st.x_dp;
    st.x_e = st.x_hii + st.x_dp;

    let out = solve_chemistry_implicit(&st, 0.0, 0.0, 200.0, 1e10);
    assert!(out.x_hd > st.x_hd, "HD debe crecer cuando hay H2 y D+");
    assert!((out.hydrogen_nuclei_fraction() - 1.0).abs() < 1e-8);
    assert!((out.deuterium_nuclei_fraction() - F_D).abs() < 1e-10);
}

#[test]
fn hd_cooling_is_positive_and_scales_with_abundance() {
    let base = cooling_rate_hd(200.0, 1e-6, 1e-3);
    let twice = cooling_rate_hd(200.0, 2e-6, 1e-3);
    assert!(base > 0.0);
    assert!((twice / base - 2.0).abs() < 1e-12);

    let sph_base = gadget_ng_sph::cooling_rate_hd(200.0, 1e-3, 1e-6);
    let sph_twice = gadget_ng_sph::cooling_rate_hd(200.0, 1e-3, 2e-6);
    assert!(sph_base > 0.0);
    assert!((sph_twice / sph_base - 2.0).abs() < 1e-12);
}

#[test]
fn legacy_six_species_json_defaults_deuterium() {
    let json = r#"{
        "x_hi": 1.0,
        "x_hii": 0.0,
        "x_hei": 0.0789,
        "x_heii": 0.0,
        "x_heiii": 0.0,
        "x_e": 0.0
    }"#;
    let st: ChemState = serde_json::from_str(json).expect("legacy ChemState");
    assert!((st.deuterium_nuclei_fraction() - F_D).abs() < 1e-15);
    assert_eq!(st.x_hd, 0.0);
}
