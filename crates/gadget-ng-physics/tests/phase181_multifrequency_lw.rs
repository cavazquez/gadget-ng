//! Phase 181 — RT multifrecuencia + Lyman-Werner.
//!
//! Verifica grupos fotónicos HI/HeI/HeII/LW/IR y el acoplamiento reducido de
//! fotodisociación LW con la red primordial H2/HD.

use gadget_ng_core::config::RtSection;
use gadget_ng_rt::{
    ChemState, M1Params, MultiFrequencyField, N_PHOTON_GROUPS, PhotonGroup,
    apply_lw_photodissociation, single_group_rates,
};

fn m1_params() -> M1Params {
    M1Params {
        c_red_factor: 100.0,
        kappa_abs: 0.0,
        kappa_scat: 0.0,
        substeps: 1,
        sigma_dust: 0.1,
    }
}

#[test]
fn photon_groups_have_stable_indices_and_energies() {
    assert_eq!(N_PHOTON_GROUPS, 5);
    assert!(PhotonGroup::LymanWerner.energy_ev() < PhotonGroup::HiIonizing.energy_ev());
    assert_eq!(PhotonGroup::Infrared.index(), 4);
}

#[test]
fn hi_group_produces_hi_photoionization_only() {
    let rates = single_group_rates(PhotonGroup::HiIonizing, 1.0e-14, &m1_params());
    assert!(rates.gamma_hi > 0.0);
    assert_eq!(rates.gamma_hei, 0.0);
    assert_eq!(rates.k_lw_h2, 0.0);
}

#[test]
fn lyman_werner_group_dissociates_molecules_without_ionizing_hi() {
    let rates = single_group_rates(PhotonGroup::LymanWerner, 1.0e-13, &m1_params());
    assert_eq!(rates.gamma_hi, 0.0);
    assert!(rates.k_lw_h2 > rates.k_lw_hd);

    let mut state = ChemState {
        x_h2: 1.0e-3,
        x_hd: 1.0e-6,
        ..ChemState::neutral()
    };
    let h2_before = state.x_h2;
    let hd_before = state.x_hd;

    apply_lw_photodissociation(&mut state, &rates, 1.0e12);

    assert!(state.x_h2 < h2_before);
    assert!(state.x_hd < hd_before);
    assert!((state.deuterium_nuclei_fraction() - gadget_ng_rt::F_D).abs() < 1e-12);
}

#[test]
fn multifrequency_field_keeps_groups_separate() {
    let mut energies = [0.0; N_PHOTON_GROUPS];
    energies[PhotonGroup::HeiIonizing.index()] = 2.0e-14;
    energies[PhotonGroup::LymanWerner.index()] = 5.0e-14;

    let field = MultiFrequencyField::uniform(2, 2, 2, 0.5, energies);
    let rates = field.rates_at_cell(0, &m1_params());

    assert_eq!(rates.gamma_hi, 0.0);
    assert!(rates.gamma_hei > 0.0);
    assert!(rates.k_lw_h2 > 0.0);
}

#[test]
fn rt_section_serde_accepts_multifrequency_knobs() {
    let cfg: RtSection = toml::from_str(
        r#"
enabled = true
rt_mesh = 16
multifrequency_enabled = true
lw_h2_factor = 0.5
lw_hd_factor = 0.25
"#,
    )
    .expect("rt section should deserialize");

    assert!(cfg.enabled);
    assert!(cfg.multifrequency_enabled);
    assert_eq!(cfg.rt_mesh, 16);
    assert_eq!(cfg.lw_h2_factor, 0.5);
    assert_eq!(cfg.lw_hd_factor, 0.25);
}
