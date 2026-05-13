//! Radiative-transfer photon groups for spectral feedback (Phase 181).
//!
//! This module keeps the existing single-field M1 solver intact and adds a
//! compact spectral layer for chemistry and feedback. The first user is the
//! Lyman-Werner band, which photodissociates `H2` and `HD` around Pop III
//! sources.

use crate::chemistry::ChemState;
use crate::m1::{C_KMS, M1Params, RadiationField};
use gadget_ng_core::{DustSection, Particle, ParticleType};
#[cfg(feature = "rayon")]
use rayon::prelude::*;

/// Photon groups transported or sampled by the reduced RT model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PhotonGroup {
    /// Hydrogen ionizing photons, h nu >= 13.6 eV.
    HiIonizing,
    /// Neutral-helium ionizing photons, h nu >= 24.6 eV.
    HeiIonizing,
    /// Singly-ionized helium ionizing photons, h nu >= 54.4 eV.
    HeiiIonizing,
    /// Lyman-Werner photons, 11.2 eV <= h nu < 13.6 eV.
    LymanWerner,
    /// Infrared photons for dust heating and re-emission.
    Infrared,
}

impl PhotonGroup {
    /// Stable index used by [`MultiFrequencyField`].
    #[inline]
    pub const fn index(self) -> usize {
        match self {
            Self::HiIonizing => 0,
            Self::HeiIonizing => 1,
            Self::HeiiIonizing => 2,
            Self::LymanWerner => 3,
            Self::Infrared => 4,
        }
    }

    /// Representative photon energy in eV.
    #[inline]
    pub const fn energy_ev(self) -> f64 {
        match self {
            Self::HiIonizing => 18.0,
            Self::HeiIonizing => 30.0,
            Self::HeiiIonizing => 60.0,
            Self::LymanWerner => 12.4,
            Self::Infrared => 0.1,
        }
    }

    /// Approximate absorption cross section in cm^2.
    #[inline]
    pub const fn cross_section_cm2(self) -> f64 {
        match self {
            Self::HiIonizing => 6.3e-18,
            Self::HeiIonizing => 7.8e-18,
            Self::HeiiIonizing => 1.6e-18,
            Self::LymanWerner => 2.0e-18,
            Self::Infrared => 1.0e-21,
        }
    }
}

/// Number of photon groups in the reduced spectral basis.
pub const N_PHOTON_GROUPS: usize = 5;

/// Group order used for compact arrays.
pub const PHOTON_GROUPS: [PhotonGroup; N_PHOTON_GROUPS] = [
    PhotonGroup::HiIonizing,
    PhotonGroup::HeiIonizing,
    PhotonGroup::HeiiIonizing,
    PhotonGroup::LymanWerner,
    PhotonGroup::Infrared,
];

const EV_TO_ERG: f64 = 1.602_176_634e-12;

/// Cell-local rates induced by a multi-frequency radiation field.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MultiFrequencyRates {
    /// HI photoionization rate [s^-1].
    pub gamma_hi: f64,
    /// HeI photoionization rate [s^-1].
    pub gamma_hei: f64,
    /// HeII photoionization rate [s^-1].
    pub gamma_heii: f64,
    /// H2 Lyman-Werner photodissociation rate [s^-1].
    pub k_lw_h2: f64,
    /// HD Lyman-Werner photodissociation rate [s^-1].
    pub k_lw_hd: f64,
    /// IR dust-heating proxy rate [s^-1].
    pub k_ir_dust: f64,
}

impl MultiFrequencyRates {
    /// Returns a zero-rate packet.
    #[inline]
    pub const fn zero() -> Self {
        Self {
            gamma_hi: 0.0,
            gamma_hei: 0.0,
            gamma_heii: 0.0,
            k_lw_h2: 0.0,
            k_lw_hd: 0.0,
            k_ir_dust: 0.0,
        }
    }
}

impl Default for MultiFrequencyRates {
    fn default() -> Self {
        Self::zero()
    }
}

/// A compact collection of one M1 field per photon group.
#[derive(Debug, Clone)]
pub struct MultiFrequencyField {
    /// M1 fields ordered by [`PHOTON_GROUPS`].
    pub groups: [RadiationField; N_PHOTON_GROUPS],
}

impl MultiFrequencyField {
    /// Creates a uniform multi-frequency field.
    pub fn uniform(
        nx: usize,
        ny: usize,
        nz: usize,
        dx: f64,
        energy_density_by_group: [f64; N_PHOTON_GROUPS],
    ) -> Self {
        Self {
            groups: energy_density_by_group.map(|e0| RadiationField::uniform(nx, ny, nz, dx, e0)),
        }
    }

    /// Borrows the field for one photon group.
    #[inline]
    pub fn group(&self, group: PhotonGroup) -> &RadiationField {
        &self.groups[group.index()]
    }

    /// Mutably borrows the field for one photon group.
    #[inline]
    pub fn group_mut(&mut self, group: PhotonGroup) -> &mut RadiationField {
        &mut self.groups[group.index()]
    }

    /// Applies the standard M1 update to every photon group.
    pub fn m1_update_all(&mut self, dt: f64, params: &M1Params) {
        for field in &mut self.groups {
            crate::m1::m1_update(field, dt, params);
        }
    }

    /// Computes local chemistry rates at a cell index.
    pub fn rates_at_cell(&self, cell: usize, params: &M1Params) -> MultiFrequencyRates {
        let c_red = C_KMS * 1e5 / params.c_red_factor;
        let rate = |group: PhotonGroup| -> f64 {
            let field = self.group(group);
            if cell >= field.energy_density.len() {
                return 0.0;
            }
            let e = field.energy_density[cell].max(0.0);
            let photon_energy = group.energy_ev() * EV_TO_ERG;
            group.cross_section_cm2() * c_red * e / photon_energy
        };

        let gamma_hi = rate(PhotonGroup::HiIonizing);
        let gamma_hei = rate(PhotonGroup::HeiIonizing);
        let gamma_heii = rate(PhotonGroup::HeiiIonizing);
        let k_lw_h2 = rate(PhotonGroup::LymanWerner);
        MultiFrequencyRates {
            gamma_hi,
            gamma_hei,
            gamma_heii,
            k_lw_h2,
            k_lw_hd: 0.35 * k_lw_h2,
            k_ir_dust: rate(PhotonGroup::Infrared),
        }
    }
}

pub fn apply_lw_photodissociation(state: &mut ChemState, rates: &MultiFrequencyRates, dt: f64) {
    if dt <= 0.0 {
        return;
    }

    let h2_survival = (-rates.k_lw_h2.max(0.0) * dt).exp();
    let hd_survival = (-rates.k_lw_hd.max(0.0) * dt).exp();
    state.x_h2 *= h2_survival;
    state.x_hd *= hd_survival;
    state.clamp_and_normalize();
}

/// Convenience helper for uniform backgrounds in one photon group.
pub fn single_group_rates(
    group: PhotonGroup,
    energy_density: f64,
    params: &M1Params,
) -> MultiFrequencyRates {
    let mut energies = [0.0; N_PHOTON_GROUPS];
    energies[group.index()] = energy_density;
    let field = MultiFrequencyField::uniform(1, 1, 1, 1.0, energies);
    field.rates_at_cell(0, params)
}

/// Deposits thermal dust emission into the IR photon group.
///
/// `radiation_energy_density` is the local heating proxy used to estimate the
/// equilibrium dust temperature. The deposited quantity is energy density:
/// luminosity times `dt` divided by the cell volume.
#[cfg(not(feature = "rayon"))]
fn deposit_dust_ir_emission_impl(
    particles: &[Particle],
    field: &mut MultiFrequencyField,
    cfg: &DustSection,
    radiation_energy_density: f64,
    dt: f64,
    box_size: f64,
) {
    if !cfg.enabled || !cfg.ir_emission_enabled || dt <= 0.0 || box_size <= 0.0 {
        return;
    }

    let ir = field.group_mut(PhotonGroup::Infrared);
    let dv = ir.dx.powi(3).max(1e-30);
    let dust_temperature =
        gadget_ng_sph::dust_equilibrium_temperature(radiation_energy_density, cfg);

    for p in particles {
        if p.ptype != ParticleType::Gas || p.dust_to_gas <= 0.0 {
            continue;
        }

        let ix = ((p.position.x / box_size * ir.nx as f64).floor() as usize).min(ir.nx - 1);
        let iy = ((p.position.y / box_size * ir.ny as f64).floor() as usize).min(ir.ny - 1);
        let iz = ((p.position.z / box_size * ir.nz as f64).floor() as usize).min(ir.nz - 1);
        let cell = ir.idx(ix, iy, iz);
        let luminosity = gadget_ng_sph::dust_ir_luminosity(p, dust_temperature, cfg);
        ir.energy_density[cell] += luminosity * dt / dv;
    }
}

#[cfg(feature = "rayon")]
fn deposit_dust_ir_emission_par(
    particles: &[Particle],
    field: &mut MultiFrequencyField,
    cfg: &DustSection,
    radiation_energy_density: f64,
    dt: f64,
    box_size: f64,
) {
    if !cfg.enabled || !cfg.ir_emission_enabled || dt <= 0.0 || box_size <= 0.0 {
        return;
    }

    let ir = field.group_mut(PhotonGroup::Infrared);
    let dv = ir.dx.powi(3).max(1e-30);
    let dust_temperature =
        gadget_ng_sph::dust_equilibrium_temperature(radiation_energy_density, cfg);
    let nx = ir.nx;
    let ny = ir.ny;
    let nz = ir.nz;

    let contributions: Vec<(usize, f64)> = particles
        .par_iter()
        .filter_map(|p| {
            if p.ptype != ParticleType::Gas || p.dust_to_gas <= 0.0 {
                return None;
            }

            let ix = ((p.position.x / box_size * nx as f64).floor() as usize).min(nx - 1);
            let iy = ((p.position.y / box_size * ny as f64).floor() as usize).min(ny - 1);
            let iz = ((p.position.z / box_size * nz as f64).floor() as usize).min(nz - 1);
            let cell = ix * ny * nz + iy * nz + iz;
            let luminosity = gadget_ng_sph::dust_ir_luminosity(p, dust_temperature, cfg);
            Some((cell, luminosity * dt / dv))
        })
        .collect();

    let ir_field = field.group_mut(PhotonGroup::Infrared);
    for (cell, delta) in contributions {
        if cell < ir_field.energy_density.len() {
            ir_field.energy_density[cell] += delta;
        }
    }
}

pub fn deposit_dust_ir_emission(
    particles: &[Particle],
    field: &mut MultiFrequencyField,
    cfg: &DustSection,
    radiation_energy_density: f64,
    dt: f64,
    box_size: f64,
) {
    #[cfg(feature = "rayon")]
    {
        deposit_dust_ir_emission_par(
            particles,
            field,
            cfg,
            radiation_energy_density,
            dt,
            box_size,
        );
    }

    #[cfg(not(feature = "rayon"))]
    {
        deposit_dust_ir_emission_impl(
            particles,
            field,
            cfg,
            radiation_energy_density,
            dt,
            box_size,
        );
    }
}
