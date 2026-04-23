//! `gadget-ng-rt` — Transferencia radiativa M1 (Phase 81).
//!
//! ## Descripción
//!
//! Implementa el solver de transferencia radiativa con cierre M1 (Levermore 1984;
//! Morel 2000) acoplado al gas SPH de `gadget-ng-sph`. El solver modela el
//! transporte de fotones UV (fotoionización de HI) e IR (calentamiento del polvo).
//!
//! ## Características
//!
//! - **Solver M1**: solver hiperbólico HLL (Harten-Lax-van Leer) en malla cartesiana.
//! - **Factor de Eddington**: cierre M1 de Levermore, interpolando entre el límite
//!   isótropo (f=1/3) y el límite de streaming libre (f=1).
//! - **Velocidad de luz reducida**: `c_red = c / factor` para pasos de tiempo manejables.
//! - **Fotoionización**: tasa Γ_HI ∝ σ_HI × c_red × E_UV.
//! - **Fotocalentamiento**: ΔU ∝ Γ_HI × dt depositado en partículas de gas.
//! - **Emisión del gas**: depósito de bremsstrahlung/recombinación al campo.
//! - **Acoplamiento splitting**: gas→rad (emisión) + rad→gas (calentamiento) por paso.
//!
//! ## Uso
//!
//! ```rust,ignore
//! use gadget_ng_rt::{RadiationField, M1Params, m1_update, radiation_gas_coupling_step};
//!
//! let mut rad = RadiationField::uniform(32, 32, 32, dx, e0_uv);
//! let params = M1Params { c_red_factor: 100.0, kappa_abs: 0.1, ..Default::default() };
//!
//! // En el loop de simulación:
//! m1_update(&mut rad, dt, &params);
//! radiation_gas_coupling_step(&mut particles, &mut rad, &params, dt, box_size);
//! ```
//!
//! ## Módulos
//!
//! - [`m1`] — Solver M1 hiperbólico: factor de Eddington, RadiationField, m1_update.
//! - [`coupling`] — Acoplamiento rad-gas: fotoionización, fotocalentamiento, emisión.

pub mod coupling;
pub mod m1;

pub use coupling::{
    apply_photoheating, deposit_gas_emission, photoionization_rate,
    radiation_gas_coupling_step,
};
pub use m1::{
    eddington_factor, m1_update, M1Params, RadiationField, C_KMS,
};
