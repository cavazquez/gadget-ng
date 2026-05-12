//! Phase 192 — Polvo activo: especies silicato/grafito + shielding H2.

use gadget_ng_core::{DustSection, DustSpeciesModel, MolecularSection, Particle, Vec3};
use gadget_ng_sph::{
    dust_h2_shielding_factor, dust_species_fractions, dust_uv_opacity_active,
    effective_dust_uv_opacity, update_h2_fraction_with_dust,
};

fn active_dust_cfg() -> DustSection {
    DustSection {
        enabled: true,
        species_model: DustSpeciesModel::SilicateGraphite,
        silicate_fraction: 0.25,
        graphite_fraction: 0.75,
        kappa_silicate_uv: 800.0,
        kappa_graphite_uv: 1600.0,
        h2_shielding_boost: 3.0,
        ..Default::default()
    }
}

fn dusty_gas(h2: f64) -> Particle {
    let mut p = Particle::new_gas(0, 1.0, Vec3::zero(), Vec3::zero(), 1.0, 0.2);
    p.dust_to_gas = 0.01;
    p.h2_fraction = h2;
    p
}

#[test]
fn silicate_graphite_fractions_are_normalized() {
    let cfg = active_dust_cfg();
    let (sil, gra) = dust_species_fractions(&cfg);
    assert!((sil - 0.25).abs() < 1e-12);
    assert!((gra - 0.75).abs() < 1e-12);
    assert!((sil + gra - 1.0).abs() < 1e-12);
}

#[test]
fn active_dust_uses_species_weighted_uv_opacity() {
    let cfg = active_dust_cfg();
    let kappa = effective_dust_uv_opacity(&cfg);
    assert!((kappa - 1400.0).abs() < 1e-12);

    let tau = dust_uv_opacity_active(&cfg, 0.01, 2.0, 0.5);
    assert!((tau - 14.0).abs() < 1e-12);
}

#[test]
fn dust_shielding_increases_h2_survival() {
    let cfg = MolecularSection {
        enabled: true,
        rho_h2_threshold: 1000.0,
        sfr_h2_boost: 1.0,
    };
    let dust = active_dust_cfg();

    let mut unshielded = vec![dusty_gas(0.8)];
    let mut shielded = vec![dusty_gas(0.8)];
    shielded[0].dust_to_gas = 0.02;

    update_h2_fraction_with_dust(&mut unshielded, &cfg, None, 5.0);
    update_h2_fraction_with_dust(&mut shielded, &cfg, Some(&dust), 5.0);

    assert!(dust_h2_shielding_factor(&shielded[0], &dust) > 1.0);
    assert!(
        shielded[0].h2_fraction > unshielded[0].h2_fraction,
        "dust shielding should slow H2 photodissociation"
    );
}
