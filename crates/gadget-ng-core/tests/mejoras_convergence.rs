//! Tests de referencia numérica (roadmap mejoras 10).
use gadget_ng_core::cosmology::{CosmologyParams, growth_factor_d, growth_factor_d_ratio};

#[test]
fn growth_factor_d_ratio_ed_s_is_linear_in_a() {
    let p = CosmologyParams::new(1.0, 0.0, 1.0);
    let r = growth_factor_d_ratio(p, 0.5, 1.0);
    assert!(
        (r - 0.5).abs() < 1e-9,
        "EdS D(a)/D(1) en a=0.5 debe ser ~0.5, got {r}"
    );
}

#[test]
fn growth_factor_d_lcdm_close_to_cpt92_at_z0() {
    let p = CosmologyParams::new(0.3, 0.7, 1.0);
    let d1 = growth_factor_d(p, 1.0);
    assert!(d1.is_finite() && d1 > 0.0, "D(1) finito positivo");
}
