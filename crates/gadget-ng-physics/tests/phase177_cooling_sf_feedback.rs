//! Phase 177 — Cooling+SF+feedback de producción (UVB + pressure-law + thermal stochastic).

use gadget_ng_core::{
    CoolingKind, FeedbackSection, Particle, SphSection, StarFormationModel, StellarFeedbackMode,
    UvBackgroundModel, Vec3,
};
use gadget_ng_sph::{
    apply_thermal_feedback_stochastic, compute_sfr_model, cooling_rate_tabular, cooling_rate_uvb,
    temperature_to_u,
};

const GAMMA: f64 = 5.0 / 3.0;

fn gas_particle(id: usize, mass: f64, u: f64, h: f64) -> Particle {
    let mut p = Particle::new_gas(id, mass, Vec3::zero(), Vec3::zero(), u, h);
    p.metallicity = 0.01;
    p
}

#[test]
fn uvb_reduces_net_cooling_after_reionization() {
    let u = temperature_to_u(2.0e5, GAMMA);
    let rho = 0.05;
    let mut cfg = SphSection {
        cooling: CoolingKind::UvBackground,
        t_floor_k: 1.0e4,
        uv_background_model: UvBackgroundModel::Hm2012,
        reionization_redshift: 8.0,
        self_shielding_nh_cm3: 1.0e-2,
        ..SphSection::default()
    };
    let lambda_tab = cooling_rate_tabular(u, rho, 0.01, GAMMA, cfg.t_floor_k);
    let lambda_uvb = cooling_rate_uvb(u, rho, 0.01, GAMMA, cfg.t_floor_k, &cfg, 1.0);
    assert!(
        lambda_uvb < lambda_tab,
        "Con UVB activo a z<z_reion, el cooling neto debe reducirse"
    );

    cfg.self_shielding_nh_cm3 = 1.0e-6;
    let lambda_uvb_shielded = cooling_rate_uvb(u, rho, 0.01, GAMMA, cfg.t_floor_k, &cfg, 1.0);
    assert!(
        lambda_uvb_shielded > lambda_uvb,
        "Mayor auto-apantallamiento debe reducir el heating y subir Λ_net"
    );
}

#[test]
fn pressure_law_sfr_increases_with_pressure() {
    let fb = FeedbackSection {
        sf_model: StarFormationModel::PressureLaw,
        rho_sf: 0.0,
        sf_pressure_norm: 1.0e-2,
        sf_pressure_index: 1.0,
        ..FeedbackSection::default()
    };
    let cold = gas_particle(0, 1.0, 0.1, 0.05);
    let hot = gas_particle(1, 1.0, 1.0, 0.05);
    let sfr = compute_sfr_model(&[cold, hot], &fb, GAMMA);
    assert!(sfr[1] > sfr[0], "Mayor presión térmica debe aumentar SFR");
}

#[test]
fn thermal_stochastic_feedback_injects_energy() {
    let fb = FeedbackSection {
        enabled: true,
        sf_model: StarFormationModel::DensityLaw,
        feedback_mode: StellarFeedbackMode::ThermalStochastic,
        delta_t_heat_k: 1.0e7,
        n_heat_neighbors: 1,
        sfr_min: 0.0,
        ..FeedbackSection::default()
    };
    let mut particles = vec![gas_particle(0, 1.0, 0.2, 0.05)];
    let sfr = vec![1.0e8];
    let u0 = particles[0].internal_energy;
    let mut seed = 17_u64;
    let injected = apply_thermal_feedback_stochastic(
        &mut particles,
        &sfr,
        &fb,
        GAMMA,
        1.0e-6,
        &mut seed,
        None,
    );
    assert!(injected > 0.0, "Debe inyectar energía térmica positiva");
    assert!(
        particles[0].internal_energy > u0,
        "La energía interna debe aumentar tras feedback térmico"
    );
}
