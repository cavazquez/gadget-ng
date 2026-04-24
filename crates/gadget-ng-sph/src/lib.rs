//! `gadget-ng-sph` — Hidrodinámica de partículas suavizadas (SPH) básica.
//!
//! ## Características
//!
//! - `ParticleType` enum `{DarkMatter, Gas}` con `GasData` (u, ρ, P, h_sml, acc_sph, du/dt).
//! - Kernel **Wendland C2** 3D con normalización correcta.
//! - Estimación de densidad SPH con **suavizado adaptativo** h_sml (Newton-Raphson,
//!   objetivo N_neigh = 32 vecinos).
//! - **Ecuaciones de movimiento SPH** simétricas (Springel & Hernquist 2002) con
//!   **viscosidad artificial** de Monaghan (α = 1).
//! - **EOS adiabática**: P = (γ−1) ρ u, γ = 5/3.
//! - Integrador **leapfrog KDK** combinado (gravedad + SPH + energía interna).
pub mod agn;
pub mod cooling;
pub mod cosmic_rays;
pub mod density;
pub mod dust;
pub mod enrichment;
pub mod feedback;
pub mod forces;
pub mod integrator;
pub mod ism;
pub mod kernel;
pub mod molecular_gas;
pub mod particle;
pub mod thermal_conduction;

pub use agn::{apply_agn_feedback, apply_agn_feedback_bimodal, bondi_accretion_rate, bubble_feedback_radio, grow_black_holes, AgnParams, BlackHole};
pub use cooling::{apply_cooling, cooling_rate_atomic, cooling_rate_metal, cooling_rate_tabular, temperature_to_u, u_to_temperature};
pub use density::{compute_density, GAMMA};
pub use enrichment::apply_enrichment;
pub use cosmic_rays::{cr_pressure, diffuse_cr, inject_cr_from_sn};
pub use ism::{effective_pressure, effective_u, update_ism_phases};
pub use cooling::apply_cooling_mhd;
pub use dust::{dust_uv_opacity, update_dust};
pub use feedback::compute_sfr_with_h2;
pub use molecular_gas::update_h2_fraction;
pub use thermal_conduction::apply_thermal_conduction;
pub use feedback::{advance_stellar_ages, apply_galactic_winds, apply_sn_feedback, apply_snia_feedback, apply_stellar_wind_feedback, compute_sfr, spawn_star_particles, total_sn_energy_injection};
pub use forces::compute_sph_forces;
pub use integrator::{sph_cosmo_kdk_step, sph_kdk_step};
pub use kernel::{grad_w, w};
pub use particle::{GasData, ParticleType, SphParticle};
