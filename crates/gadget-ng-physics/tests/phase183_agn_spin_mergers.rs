//! Phase 183 — AGN spin + mergers.

use gadget_ng_core::{AgnSection, Vec3};
use gadget_ng_sph::{
    BlackHole, merge_black_holes, radiative_efficiency_from_spin,
    spin_dependent_feedback_efficiency, spin_up_by_accretion,
};

#[test]
fn kerr_efficiency_increases_for_prograde_spin() {
    let retro = radiative_efficiency_from_spin(-0.9);
    let zero = radiative_efficiency_from_spin(0.0);
    let prograde = radiative_efficiency_from_spin(0.9);

    assert!(retro < zero);
    assert!(prograde > zero);
}

#[test]
fn spin_feedback_efficiency_scales_base_feedback() {
    let eps0 = 0.05;
    let low = spin_dependent_feedback_efficiency(eps0, -0.9);
    let high = spin_dependent_feedback_efficiency(eps0, 0.9);

    assert!(high > eps0);
    assert!(low < eps0);
}

#[test]
fn accretion_spins_bh_toward_prograde_limit() {
    let mut bh = BlackHole::with_spin(Vec3::zero(), 1.0e8, 0.0);
    bh.accretion_rate = 1.0e7;

    spin_up_by_accretion(&mut bh, 1.0);

    assert!(bh.spin > 0.0);
    assert!(bh.spin <= 0.998);
}

#[test]
fn close_black_holes_merge_with_mass_weighted_spin() {
    let mut bhs = vec![
        BlackHole::with_spin(Vec3::new(0.0, 0.0, 0.0), 3.0, 0.6),
        BlackHole::with_spin(Vec3::new(0.05, 0.0, 0.0), 1.0, -0.2),
        BlackHole::with_spin(Vec3::new(10.0, 0.0, 0.0), 5.0, 0.0),
    ];

    let merged = merge_black_holes(&mut bhs, 0.1, 0.0, None);

    assert_eq!(merged, 1);
    assert_eq!(bhs.len(), 2);
    assert!(bhs.iter().any(|bh| (bh.mass - 4.0).abs() < 1e-12));
}

#[test]
fn agn_section_serde_accepts_spin_and_merger_knobs() {
    let cfg: AgnSection = toml::from_str(
        r#"
enabled = true
spin_enabled = true
initial_spin = 0.7
mergers_enabled = true
merger_radius = 0.25
recoil_velocity_scale = 800.0
"#,
    )
    .expect("agn section should deserialize");

    assert!(cfg.spin_enabled);
    assert_eq!(cfg.initial_spin, 0.7);
    assert!(cfg.mergers_enabled);
    assert_eq!(cfg.merger_radius, 0.25);
    assert_eq!(cfg.recoil_velocity_scale, 800.0);
}
