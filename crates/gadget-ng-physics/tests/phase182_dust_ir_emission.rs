//! Phase 182 — polvo IR / emisión térmica.

use gadget_ng_core::{DustSection, Particle, ParticleType, Vec3};
use gadget_ng_rt::{MultiFrequencyField, N_PHOTON_GROUPS, PhotonGroup, deposit_dust_ir_emission};
use gadget_ng_sph::{dust_equilibrium_temperature, dust_ir_luminosity};

fn dust_cfg() -> DustSection {
    DustSection {
        enabled: true,
        ir_emission_enabled: true,
        kappa_dust_uv: 1000.0,
        kappa_dust_ir: 10.0,
        ir_emissivity: 1.0,
        dust_temperature_floor_k: 2.725,
        dust_temperature_cap_k: 2000.0,
        ..Default::default()
    }
}

fn dusty_gas(id: usize, x: f64) -> Particle {
    let mut p = Particle::new_gas(id, 2.0, Vec3::new(x, 0.5, 0.5), Vec3::zero(), 1.0, 0.1);
    p.dust_to_gas = 0.01;
    p
}

#[test]
fn dust_temperature_grows_with_radiation_field() {
    let cfg = dust_cfg();
    let cold = dust_equilibrium_temperature(0.0, &cfg);
    let warm = dust_equilibrium_temperature(1.0e-12, &cfg);
    let hot = dust_equilibrium_temperature(1.0e-8, &cfg);

    assert!(warm > cold);
    assert!(hot > warm);
    assert!(hot <= cfg.dust_temperature_cap_k);
}

#[test]
fn dust_ir_luminosity_requires_gas_and_dust() {
    let cfg = dust_cfg();
    let gas = dusty_gas(0, 0.25);
    let mut no_dust = gas.clone();
    no_dust.dust_to_gas = 0.0;
    let mut dm = gas.clone();
    dm.ptype = ParticleType::DarkMatter;

    assert!(dust_ir_luminosity(&gas, 30.0, &cfg) > 0.0);
    assert_eq!(dust_ir_luminosity(&no_dust, 30.0, &cfg), 0.0);
    assert_eq!(dust_ir_luminosity(&dm, 30.0, &cfg), 0.0);
}

#[test]
fn dust_ir_deposits_only_into_infrared_group() {
    let cfg = dust_cfg();
    let particles = vec![dusty_gas(0, 0.25), dusty_gas(1, 0.75)];
    let mut field = MultiFrequencyField::uniform(4, 4, 4, 0.25, [0.0; N_PHOTON_GROUPS]);

    deposit_dust_ir_emission(&particles, &mut field, &cfg, 1.0e-10, 0.5, 1.0);

    let ir_energy: f64 = field
        .group(PhotonGroup::Infrared)
        .energy_density
        .iter()
        .sum();
    let hi_energy: f64 = field
        .group(PhotonGroup::HiIonizing)
        .energy_density
        .iter()
        .sum();

    assert!(ir_energy > 0.0);
    assert_eq!(hi_energy, 0.0);
}

#[test]
fn dust_section_serde_accepts_ir_knobs() {
    let cfg: DustSection = toml::from_str(
        r#"
enabled = true
ir_emission_enabled = true
kappa_dust_ir = 20.0
ir_emissivity = 0.7
dust_temperature_floor_k = 10.0
dust_temperature_cap_k = 1500.0
"#,
    )
    .expect("dust section should deserialize");

    assert!(cfg.ir_emission_enabled);
    assert_eq!(cfg.kappa_dust_ir, 20.0);
    assert_eq!(cfg.ir_emissivity, 0.7);
    assert_eq!(cfg.dust_temperature_floor_k, 10.0);
    assert_eq!(cfg.dust_temperature_cap_k, 1500.0);
}
