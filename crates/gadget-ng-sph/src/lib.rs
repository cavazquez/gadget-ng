//! `gadget-ng-sph` — Hidrodinámica de partículas suavizadas (SPH) básica.
//!
//! ## Características
//!
//! - `ParticleType` enum `{DarkMatter, Gas}` con `GasData`
//!   (u, A, ρ, P, h_sml, acc_sph, du/dt, da/dt, Balsara, max_vsig).
//! - Kernel **Wendland C2** 3D con normalización correcta.
//! - Estimación de densidad SPH con **suavizado adaptativo** h_sml (Newton-Raphson,
//!   objetivo N_neigh = 32 vecinos).
//! - **Ecuaciones de movimiento SPH** simétricas (Springel & Hernquist 2002) con
//!   **viscosidad artificial** de Monaghan (α = 1).
//! - **EOS adiabática**: P = (γ−1) ρ u, γ = 5/3.
//! - Integrador **leapfrog KDK** combinado (gravedad + SPH + energía interna).
//! - **Formulación de entropía Gadget-2** (Springel & Hernquist 2002):
//!   A = P/ρ^γ, evolución exacta con viscosidad de señal y **limitador de Balsara**.
//! - Integrador `sph_kdk_step_gadget2` + función `courant_dt` para timestep adaptativo.
use gadget_ng_core::Vec3;

/// Vector mínima imagen en caja cúbica de lado `L` (`p_j - p_i` por eje en `[-L/2, L/2]`).
#[inline]
pub fn periodic_delta(pi: Vec3, pj: Vec3, periodic_box: Option<f64>) -> Vec3 {
    let mut d = pj - pi;
    if let Some(l) = periodic_box
        && l > 0.0
    {
        d.x -= l * (d.x / l).round();
        d.y -= l * (d.y / l).round();
        d.z -= l * (d.z / l).round();
    }
    d
}

pub mod agn;
pub mod cooling;
pub mod cosmic_rays;
pub mod density;
pub mod dust;
pub mod enrichment;
pub mod feedback;
pub mod forces;
pub mod gmc;
pub mod integrator;
pub mod ism;
pub mod kernel;
pub mod molecular_gas;
pub mod particle;
pub mod phase_transitions;
pub mod pop_iii;
pub mod thermal_conduction;
pub mod viscosity;

pub use agn::{
    AgnParams, BlackHole, PbhSeedingParams, apply_agn_feedback, apply_agn_feedback_bimodal,
    apply_agn_feedback_bimodal_periodic, apply_agn_feedback_periodic, bondi_accretion_rate,
    bubble_feedback_radio, bubble_feedback_radio_periodic, grow_black_holes,
    grow_black_holes_periodic, merge_black_holes, radiative_efficiency_from_spin,
    seed_primordial_black_holes, spin_dependent_feedback_efficiency, spin_up_by_accretion,
};
pub use cooling::{
    apply_cooling, apply_cooling_mhd, apply_cooling_mhd_with_redshift, apply_cooling_with_redshift,
    cooling_rate_atomic, cooling_rate_hd, cooling_rate_metal, cooling_rate_tabular,
    cooling_rate_uvb, temperature_to_u, u_to_temperature,
};
pub use cosmic_rays::diffuse_cr_periodic;
pub use cosmic_rays::{apply_cr_hadronic_losses, cr_pressure, diffuse_cr, inject_cr_from_sn};
pub use density::{GAMMA, compute_density, compute_density_with_periodic};
pub use dust::{
    apply_dust_radiation_pressure_kick, dust_equilibrium_temperature, dust_h2_shielding_factor,
    dust_ir_luminosity, dust_species_fractions, dust_uv_opacity, dust_uv_opacity_active,
    effective_dust_uv_opacity, update_dust,
};
pub use enrichment::apply_enrichment;
pub use enrichment::apply_enrichment_periodic;
pub use feedback::compute_sfr_with_h2;
pub use feedback::{
    advance_stellar_ages, apply_galactic_winds, apply_sn_feedback, apply_snia_feedback,
    apply_snia_feedback_periodic, apply_stellar_wind_feedback, apply_thermal_feedback_stochastic,
    compute_sfr, compute_sfr_model, compute_sfr_pressure, spawn_star_particles,
    total_sn_energy_injection,
};
pub use forces::{
    compute_sph_forces, compute_sph_forces_gadget2, compute_sph_forces_gadget2_with_periodic,
    compute_sph_forces_with_periodic,
};
#[cfg(feature = "bench-sph-forces-ref")]
pub use forces::{
    compute_sph_forces_gadget2_scalar_ref, sph_gadget2_update_for_particle_scalar_ref,
};
pub use gmc::inject_sn_from_cluster_periodic;
pub use gmc::{GmcCluster, KroupaImf, collapse_gmc, inject_sn_from_cluster, sample_stellar_mass};
pub use integrator::{courant_dt, sph_cosmo_kdk_step, sph_kdk_step, sph_kdk_step_gadget2};
pub use ism::{effective_pressure, effective_u, update_ism_phases};
pub use kernel::{grad_w, grad_w_batch, w, w_and_grad_w_batch, w_batch};
pub use molecular_gas::{update_h2_fraction, update_h2_fraction_with_dust};
pub use particle::{GasData, ParticleType, SphParticle};
pub use phase_transitions::{
    apply_phase_transitions, classify_phase, cooling_time, field_length, free_fall_time,
    phase_fractions, thermal_instability_criterion,
};
pub use pop_iii::{
    PopIIICluster, apply_pop_iii_pisn_feedback, form_pop_iii_clusters, is_pop_iii_candidate,
    sample_pop_iii_mass,
};
pub use thermal_conduction::{apply_thermal_conduction, apply_thermal_conduction_periodic};
pub use viscosity::{compute_balsara_factors, compute_balsara_factors_with_periodic};
