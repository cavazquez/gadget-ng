//! Tests helpers cosmología roadmap (neutrinos, from_cosmology_toml).

use gadget_ng_core::{
    hubble_param, omega_nu_from_mass, split_m_nu_ev, CosmologyParams, NeutrinoHierarchyKind,
};

#[test]
fn split_m_nu_preserves_sum() {
    let s = 0.06_f64;
    for h in [
        NeutrinoHierarchyKind::Degenerate,
        NeutrinoHierarchyKind::Normal,
        NeutrinoHierarchyKind::Inverted,
    ] {
        let m = split_m_nu_ev(s, h);
        let sum: f64 = m.iter().sum();
        assert!(
            (sum - s).abs() < 1e-14,
            "hierarchy {:?} sum {} vs {}",
            h,
            sum,
            s
        );
    }
}

#[test]
fn from_cosmology_toml_matches_omega_nu() {
    let h100 = 0.674_f64;
    let m_nu = 0.12_f64;
    let om = omega_nu_from_mass(m_nu, h100);
    let p = CosmologyParams::from_cosmology_toml(0.3, 0.7, 0.1, -1.0, 0.0, m_nu, h100);
    assert!((p.omega_nu - om).abs() < 1e-12);
    let h0 = hubble_param(p, 1.0);
    assert!(h0.is_finite() && h0 > 0.0);
}
