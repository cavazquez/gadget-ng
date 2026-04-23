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
pub mod density;
pub mod feedback;
pub mod forces;
pub mod integrator;
pub mod kernel;
pub mod particle;

pub use agn::{apply_agn_feedback, bondi_accretion_rate, grow_black_holes, AgnParams, BlackHole};
pub use cooling::{apply_cooling, cooling_rate_atomic, temperature_to_u, u_to_temperature};
pub use density::{compute_density, GAMMA};
pub use feedback::{apply_sn_feedback, compute_sfr, total_sn_energy_injection};
pub use forces::compute_sph_forces;
pub use integrator::{sph_cosmo_kdk_step, sph_kdk_step};
pub use kernel::{grad_w, w};
pub use particle::{GasData, ParticleType, SphParticle};
