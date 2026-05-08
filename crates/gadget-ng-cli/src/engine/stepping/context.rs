//! Shared execution context for all solver branches in `run_stepping`.
//!
//! Holds all mutable state that the `maybe_*!` macros and per-branch code
//! need, so each solver-path function receives a single `&mut SteppingCtx`
//! instead of ~20 separate borrows.

use gadget_ng_core::{
    Particle, RunConfig, SfcKind, Vec3, build_particles_for_gid_range,
    cosmology::CosmologyParams,
};
use gadget_ng_integrators::HierarchicalState;
use gadget_ng_parallel::{ParallelRuntime, gid_block_range};
use std::path::Path;
use std::time::Instant;

use crate::config_load;
use crate::error::CliError;
use gadget_ng_io::Provenance;

use crate::engine::checkpoint::{load_checkpoint, save_checkpoint};
use crate::engine::diagnostics::write_diagnostic_line;
use crate::engine::gravity::{local_bh_walk_params, local_bh_use_rayon, LocalBhWalkParams};
use crate::engine::provenance::{provenance_for_run, snapshot_env_for};
use crate::engine::timings::{
    HpcTimingsAggregate, TreePmAggregate, TreePmStepDiag,
};

/// Accumulated timings written to `timings.json` at the end of a run.
pub(crate) struct SteppingTimings {
    pub acc_comm_ns: u64,
    pub acc_gravity_ns: u64,
    pub acc_step_ns: u64,
    pub steps_run: u64,
    pub wall_loop_start: Instant,
    pub hpc_aggregate_opt: Option<HpcTimingsAggregate>,
    pub treepm_hpc_opt: Option<TreePmAggregate>,
    pub acc_tpm: TreePmStepDiag,
    pub tpm_step_count: u64,
}

impl Default for SteppingTimings {
    fn default() -> Self {
        Self {
            acc_comm_ns: 0,
            acc_gravity_ns: 0,
            acc_step_ns: 0,
            steps_run: 0,
            wall_loop_start: Instant::now(),
            hpc_aggregate_opt: None,
            treepm_hpc_opt: None,
            acc_tpm: TreePmStepDiag::default(),
            tpm_step_count: 0,
        }
    }
}

/// Context shared by all solver branches.
///
/// Borrows are held for the duration of the main integration loop.
pub(crate) struct SteppingCtx<'a, R: ParallelRuntime + ?Sized> {
    pub rt: &'a R,
    pub cfg: &'a RunConfig,
    pub out_dir: &'a Path,
    pub total: usize,

    pub local: Vec<Particle>,
    pub scratch: Vec<Vec3>,

    pub start_step: u64,
    pub a_current: f64,

    pub eps2_base: f64,
    pub g: f64,

    pub checkpoint_interval: u64,
    pub snapshot_interval: u64,
    pub sfc_rebalance: u64,

    pub sfc_kind: SfcKind,
    pub bh_walk: LocalBhWalkParams,
    pub bh_parallel: bool,

    pub cosmo_state: Option<(CosmologyParams, f64)>,

    pub h_state_opt: Option<HierarchicalState>,
    pub h_state_resume: Option<HierarchicalState>,

    pub rt_field_opt: Option<gadget_ng_rt::RadiationField>,
    pub sph_chem_states: Vec<gadget_ng_rt::ChemState>,
    pub agn_bhs: Vec<gadget_ng_sph::BlackHole>,
    pub halo_centers: Vec<Vec3>,

    pub global_pos: Vec<Vec3>,
    pub global_mass: Vec<f64>,

    pub cfg_hash: String,
    pub prov: Provenance,

    pub timings: SteppingTimings,
}

impl<'a, R: ParallelRuntime + ?Sized> SteppingCtx<'a, R> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        rt: &'a R,
        cfg: &'a RunConfig,
        out_dir: &'a Path,
        _write_final_snapshot: bool,
        resume_from: Option<&Path>,
    ) -> Result<Self, CliError> {
        let total = cfg.simulation.particle_count;
        let (lo, hi) = gid_block_range(total, rt.rank(), rt.size());

        let g = cfg.effective_g();

        if let Some((g_consistent, rel_err)) = cfg.cosmo_g_diagnostic() {
            if cfg.cosmology.auto_g {
                rt.root_eprintln(&format!(
                    "[gadget-ng] cosmology.auto_g=true → G auto-consistente: {g:.4e} \
                     (3·Ω_m·H₀²/8π, condición de Friedmann satisfecha)"
                ));
            } else if rel_err > 0.01 {
                rt.root_eprintln(&format!(
                    "[gadget-ng] ADVERTENCIA: G ({g:.4e}) inconsistente con cosmología \
                     ({:.1}% fuera de G_consistente={g_consistent:.4e}). \
                     Usa [cosmology] auto_g = true para corregir automáticamente.",
                    rel_err * 100.0
                ));
            }
        }

        let eps2_base = cfg.softening_squared();

        let bh_walk = local_bh_walk_params(cfg);
        let bh_parallel = local_bh_use_rayon(cfg);
        let checkpoint_interval = cfg.output.checkpoint_interval;
        let snapshot_interval = cfg.output.snapshot_interval;

        let cfg_hash =
            config_load::config_canonical_hash(cfg).unwrap_or_else(|_| "unknown".to_owned());

        let mut resume_agn_bhs: Option<Vec<gadget_ng_sph::BlackHole>> = None;
        let mut resume_chem_states: Option<Vec<gadget_ng_rt::ChemState>> = None;

        let (local, start_step, a_current, h_state_resume) =
            if let Some(resume_dir) = resume_from {
                rt.root_eprintln(&format!(
                    "[gadget-ng] Reanudando desde checkpoint en {:?}",
                    resume_dir.join("checkpoint")
                ));
                let (p, completed, a, hs, agn_opt, chem_opt) =
                    load_checkpoint(rt, resume_dir, lo, hi, &cfg_hash)?;
                resume_agn_bhs = agn_opt;
                resume_chem_states = chem_opt;
                (p, completed + 1, a, hs)
            } else if let gadget_ng_core::IcKind::External { path, format } =
                &cfg.initial_conditions.kind
            {
                let ic_path = Path::new(path);
                rt.root_eprintln(&format!(
                    "[gadget-ng] Cargando ICs externas desde {:?} (formato: {:?})",
                    ic_path, format
                ));
                let data =
                    gadget_ng_io::read_snapshot_formatted(*format, ic_path)
                        .map_err(CliError::Snapshot)?;
                let mut p = data.particles;
                p.retain(|part| part.global_id >= lo && part.global_id < hi);
                let a0 = data.time;
                (p, 1u64, a0, None)
            } else {
                let p = build_particles_for_gid_range(cfg, lo, hi)?;
                let a0 = if cfg.cosmology.enabled {
                    cfg.cosmology.a_init
                } else {
                    1.0
                };
                (p, 1u64, a0, None)
            };

        let scratch = vec![Vec3::zero(); local.len()];
        let global_pos: Vec<Vec3> = Vec::new();
        let global_mass: Vec<f64> = Vec::new();

        std::fs::create_dir_all(out_dir).map_err(|e| CliError::io(out_dir, e))?;
        let prov = provenance_for_run(cfg)?;

        let diag_path = out_dir.join("diagnostics.jsonl");
        let mut diag_file = if rt.rank() == 0 {
            Some(
                std::fs::File::create(&diag_path)
                    .map_err(|e| CliError::io(&diag_path, e))?,
            )
        } else {
            None
        };

        write_diagnostic_line(rt, 0, &local, &diag_path, &mut diag_file, None, None, None)?;

        let rt_field_opt: Option<gadget_ng_rt::RadiationField> = if cfg.rt.enabled {
            let n = cfg.rt.rt_mesh;
            let dx = if n > 0 {
                cfg.simulation.box_size / n as f64
            } else {
                1.0
            };
            Some(gadget_ng_rt::RadiationField::uniform(n, n, n, dx, 0.0))
        } else {
            None
        };

        let sph_chem_states: Vec<gadget_ng_rt::ChemState> = if cfg.reionization.enabled {
            if let Some(restored) = resume_chem_states.take() {
                restored
            } else {
                local
                    .iter()
                    .map(|_| gadget_ng_rt::ChemState::neutral())
                    .collect()
            }
        } else {
            Vec::new()
        };

        let sfc_kind = cfg.performance.sfc_kind;
        let sfc_rebalance = cfg.performance.sfc_rebalance_interval;

        let cosmo_state: Option<(CosmologyParams, f64)> = if cfg.cosmology.enabled {
            let params = CosmologyParams::from_cosmology_toml(
                cfg.cosmology.omega_m,
                cfg.cosmology.omega_lambda,
                cfg.cosmology.h0,
                cfg.cosmology.w0,
                cfg.cosmology.wa,
                cfg.cosmology.m_nu_ev,
                cfg.cosmology.h0 * 10.0,
            );
            Some((params, cfg.cosmology.a_init))
        } else {
            None
        };

        Ok(Self {
            rt,
            cfg,
            out_dir,
            total,
            local,
            scratch,
            start_step,
            a_current,
            eps2_base,
            g,
            checkpoint_interval,
            snapshot_interval,
            sfc_rebalance,
            sfc_kind,
            bh_walk,
            bh_parallel,
            cosmo_state,
            h_state_opt: None,
            h_state_resume,
            rt_field_opt,
            sph_chem_states,
            agn_bhs: resume_agn_bhs.unwrap_or_default(),
            halo_centers: Vec::new(),
            global_pos,
            global_mass,
            cfg_hash,
            prov,
            timings: SteppingTimings::default(),
        })
    }

    pub(crate) fn checkpoint(&mut self, step: u64) {
        if self.checkpoint_interval > 0 && step % self.checkpoint_interval == 0 {
            let _ = save_checkpoint(
                self.rt,
                step,
                self.a_current,
                &self.local,
                self.total,
                self.h_state_opt.as_ref(),
                self.out_dir,
                &self.cfg_hash,
                &self.agn_bhs,
                &self.sph_chem_states,
            );
        }
    }

    pub(crate) fn snap_frame(&mut self, step: u64) {
        if self.snapshot_interval > 0 && step % self.snapshot_interval == 0 {
            if let Some(all_parts) = self.rt.root_gather_particles(&self.local, self.total) {
                let frame_dir =
                    self.out_dir.join("frames").join(format!("snap_{:06}", step));
                let _ = std::fs::create_dir_all(&frame_dir);
                let t = step as f64 * self.cfg.simulation.dt;
                let z = if self.cfg.cosmology.enabled {
                    1.0 / self.a_current - 1.0
                } else {
                    0.0
                };
                let env = snapshot_env_for(self.cfg, t, z);
                let _ = gadget_ng_io::write_snapshot_formatted(
                    self.cfg.output.snapshot_format,
                    &frame_dir,
                    &all_parts,
                    &self.prov,
                    &env,
                );
            }
        }
    }

    pub(crate) fn insitu(&mut self, step: u64) {
        let (_insitu_ran, _insitu_fx) = crate::insitu::maybe_run_insitu(
            &self.local,
            &self.cfg.insitu_analysis,
            self.cfg.simulation.box_size,
            self.a_current,
            step,
            self.out_dir,
            if self.sph_chem_states.is_empty() {
                None
            } else {
                Some(&self.sph_chem_states)
            },
        );
        if _insitu_ran && !_insitu_fx.halo_centers.is_empty() {
            self.halo_centers = _insitu_fx.halo_centers;
        }
    }

    pub(crate) fn sph(&mut self, sph_step: u64) {
        if !self.cfg.sph.enabled {
            return;
        }
        let cf_sph = match &self.cosmo_state {
            Some((cp, _)) => {
                let (d, k, k2) = cp.drift_kick_factors(self.a_current, self.cfg.simulation.dt);
                gadget_ng_integrators::CosmoFactors { drift: d, kick_half: k, kick_half2: k2 }
            }
            None => gadget_ng_integrators::CosmoFactors::flat(self.cfg.simulation.dt),
        };
        let gamma = self.cfg.sph.gamma;
        let alpha = self.cfg.sph.alpha_visc;
        let n_neigh = self.cfg.sph.n_neigh as f64;
        let periodic_box = self.cfg.cosmology.periodic.then_some(self.cfg.simulation.box_size);

        gadget_ng_sph::sph_cosmo_kdk_step(
            &mut self.local,
            cf_sph,
            gamma,
            alpha,
            n_neigh,
            periodic_box,
            |_parts| {},
        );

        if self.cfg.sph.cooling != gadget_ng_core::CoolingKind::None {
            if self.cfg.sph.mag_suppress_cooling > 0.0 && self.cfg.mhd.enabled {
                gadget_ng_sph::apply_cooling_mhd(&mut self.local, &self.cfg.sph, self.cfg.simulation.dt);
            } else {
                gadget_ng_sph::apply_cooling(&mut self.local, &self.cfg.sph, self.cfg.simulation.dt);
            }
        }

        if self.cfg.sph.dust.enabled {
            gadget_ng_sph::update_dust(&mut self.local, &self.cfg.sph.dust, self.cfg.sph.gamma, self.cfg.simulation.dt);
        }
        if self.cfg.sph.dust.enabled && self.cfg.sph.dust.radiation_pressure_enabled {
            let z_ref = 0.5 * self.cfg.simulation.box_size;
            gadget_ng_sph::apply_dust_radiation_pressure_kick(&mut self.local, &self.cfg.sph.dust, z_ref, self.cfg.simulation.dt);
        }

        if self.cfg.sph.conduction.enabled {
            if self.cfg.sph.conduction.anisotropic {
                gadget_ng_mhd::apply_anisotropic_conduction(
                    &mut self.local,
                    self.cfg.sph.conduction.kappa_par,
                    self.cfg.sph.conduction.kappa_perp,
                    self.cfg.sph.gamma,
                    self.cfg.simulation.dt,
                );
            } else {
                gadget_ng_sph::apply_thermal_conduction_periodic(
                    &mut self.local,
                    &self.cfg.sph.conduction,
                    self.cfg.sph.gamma,
                    self.cfg.sph.t_floor_k,
                    self.cfg.simulation.dt,
                    periodic_box,
                );
            }
        }

        if self.cfg.sph.molecular.enabled {
            gadget_ng_sph::update_h2_fraction(&mut self.local, &self.cfg.sph.molecular, self.cfg.simulation.dt);
        }

        if self.cfg.turbulence.enabled {
            gadget_ng_mhd::apply_turbulent_forcing(&mut self.local, &self.cfg.turbulence, self.cfg.simulation.dt, sph_step as u64);
        }

        if self.cfg.two_fluid.enabled {
            gadget_ng_mhd::apply_electron_ion_coupling(&mut self.local, &self.cfg.two_fluid, self.cfg.simulation.dt);
        }

        if self.cfg.sph.ism.enabled {
            let sfr_ism = gadget_ng_sph::compute_sfr(&self.local, &self.cfg.sph.feedback);
            let rho_sf = self.cfg.sph.feedback.rho_sf;
            gadget_ng_sph::update_ism_phases(&mut self.local, &sfr_ism, rho_sf, &self.cfg.sph.ism, self.cfg.simulation.dt);
        }

        if self.cfg.sph.feedback.enabled {
            let sfr = gadget_ng_sph::compute_sfr(&self.local, &self.cfg.sph.feedback);
            let mut fb_seed = (sph_step as u64).wrapping_mul(2654435761).wrapping_add(self.rt.rank() as u64);
            gadget_ng_sph::apply_sn_feedback(&mut self.local, &sfr, &self.cfg.sph.feedback, self.cfg.simulation.dt, &mut fb_seed);

            if self.cfg.sph.feedback.stellar_wind_enabled {
                let mut wind_seed = fb_seed.wrapping_add(0xC0FFEE);
                gadget_ng_sph::apply_stellar_wind_feedback(&mut self.local, &sfr, &self.cfg.sph.feedback, self.cfg.simulation.dt, &mut wind_seed);
            }

            if self.cfg.sph.cr.enabled {
                gadget_ng_sph::inject_cr_from_sn(&mut self.local, &sfr, self.cfg.sph.cr.cr_fraction, self.cfg.simulation.dt);
                if self.cfg.sph.cr.anisotropic_diffusion && self.cfg.mhd.enabled {
                    gadget_ng_mhd::diffuse_cr_anisotropic(&mut self.local, self.cfg.sph.cr.kappa_cr, self.cfg.sph.cr.b_cr_suppress, self.cfg.simulation.dt);
                } else {
                    gadget_ng_sph::diffuse_cr_periodic(&mut self.local, self.cfg.sph.cr.kappa_cr, self.cfg.sph.cr.b_cr_suppress, self.cfg.simulation.dt, periodic_box);
                }
                if self.cfg.sph.cr.hadronic_loss_coeff > 0.0 {
                    gadget_ng_sph::apply_cr_hadronic_losses(&mut self.local, self.cfg.sph.cr.hadronic_loss_coeff, self.cfg.simulation.dt);
                }
            }

            let mut spawn_seed = fb_seed.wrapping_add(0xDEAD_BEEF);
            let mut next_gid = self.local.iter().map(|p| p.global_id).max().unwrap_or(0) + 1;
            let (new_stars, to_remove) = gadget_ng_sph::spawn_star_particles(
                &mut self.local, &sfr, self.cfg.simulation.dt, &mut spawn_seed,
                &self.cfg.sph.feedback, &mut next_gid,
            );
            let mut remove_sorted = to_remove;
            remove_sorted.sort_unstable_by(|a, b| b.cmp(a));
            remove_sorted.dedup();
            for idx in remove_sorted {
                self.local.swap_remove(idx);
            }
            self.local.extend(new_stars);

            let dt_gyr = self.cfg.simulation.dt * 1e-3;
            gadget_ng_sph::advance_stellar_ages(&mut self.local, dt_gyr);
            let mut ia_seed = fb_seed.wrapping_add(0xF00D);
            gadget_ng_sph::apply_snia_feedback_periodic(&mut self.local, dt_gyr, &mut ia_seed, &self.cfg.sph.feedback, periodic_box);
        }
    }

    pub(crate) fn agn(&mut self) {
        if !self.cfg.sph.agn.enabled {
            return;
        }
        let periodic_box_agn = self.cfg.cosmology.periodic.then_some(self.cfg.simulation.box_size);
        let agn_params = gadget_ng_sph::AgnParams {
            eps_feedback: self.cfg.sph.agn.eps_feedback,
            m_seed: self.cfg.sph.agn.m_seed,
            v_kick_agn: self.cfg.sph.agn.v_kick_agn,
            r_influence: self.cfg.sph.agn.r_influence,
        };
        let n_bh = self.cfg.sph.agn.n_agn_bh.max(1);

        if !self.halo_centers.is_empty() {
            let n_new = self.halo_centers.len().min(n_bh);
            if self.agn_bhs.len() != n_new {
                self.agn_bhs.resize_with(n_new, || {
                    gadget_ng_sph::BlackHole::new(gadget_ng_core::Vec3::zero(), agn_params.m_seed)
                });
            }
            for (bh, &pos) in self.agn_bhs.iter_mut().zip(self.halo_centers.iter()) {
                bh.pos = pos;
            }
        } else if self.agn_bhs.is_empty() {
            let center = self.cfg.simulation.box_size * 0.5;
            self.agn_bhs.push(gadget_ng_sph::BlackHole::new(
                gadget_ng_core::Vec3::new(center, center, center),
                agn_params.m_seed,
            ));
        }

        gadget_ng_sph::grow_black_holes_periodic(
            &mut self.agn_bhs,
            &self.local,
            &agn_params,
            self.cfg.simulation.dt,
            periodic_box_agn,
        );
        gadget_ng_sph::apply_agn_feedback_bimodal_periodic(
            &mut self.local,
            &self.agn_bhs,
            &agn_params,
            self.cfg.sph.agn.f_edd_threshold,
            self.cfg.sph.agn.r_bubble,
            self.cfg.sph.agn.eps_radio,
            self.cfg.simulation.dt,
            periodic_box_agn,
        );
    }

    pub(crate) fn mhd(&mut self) {
        if !self.cfg.mhd.enabled {
            return;
        }
        let dt_alfven = gadget_ng_mhd::alfven_dt(&self.local, self.cfg.mhd.cfl_mhd);
        let dt_mhd = self.cfg.simulation.dt.min(dt_alfven);
        gadget_ng_mhd::advance_induction(&mut self.local, dt_mhd);

        if self.cfg.mhd.alpha_b > 0.0 {
            gadget_ng_mhd::apply_artificial_resistivity(&mut self.local, self.cfg.mhd.alpha_b, dt_mhd);
        }
        gadget_ng_mhd::apply_magnetic_forces(&mut self.local, dt_mhd);
        gadget_ng_mhd::dedner_cleaning_step(&mut self.local, self.cfg.mhd.c_h, self.cfg.mhd.c_r, dt_mhd);

        if self.cfg.mhd.relativistic_mhd {
            gadget_ng_mhd::advance_srmhd(&mut self.local, dt_mhd, gadget_ng_mhd::C_LIGHT, self.cfg.mhd.v_rel_threshold);
        }

        {
            let rho_ref_mhd = gadget_ng_mhd::mean_gas_density(&self.local);
            gadget_ng_mhd::apply_flux_freeze(&mut self.local, self.cfg.sph.gamma, self.cfg.mhd.beta_freeze, rho_ref_mhd);
        }

        if self.cfg.mhd.eta_braginskii > 0.0 {
            gadget_ng_mhd::apply_braginskii_viscosity(&mut self.local, self.cfg.mhd.eta_braginskii, dt_mhd);
        }

        if self.cfg.mhd.reconnection_enabled {
            gadget_ng_mhd::apply_magnetic_reconnection(&mut self.local, self.cfg.mhd.f_reconnection, self.cfg.sph.gamma, dt_mhd);
        }
    }

    pub(crate) fn rt(&mut self) {
        if !self.cfg.rt.enabled {
            return;
        }
        if let Some(ref mut rf) = self.rt_field_opt {
            let m1p = gadget_ng_rt::M1Params {
                c_red_factor: self.cfg.rt.c_red_factor,
                kappa_abs: self.cfg.rt.kappa_abs,
                kappa_scat: 0.0,
                substeps: self.cfg.rt.substeps,
                sigma_dust: 0.1,
            };
            gadget_ng_rt::m1_update(rf, self.cfg.simulation.dt, &m1p);
            gadget_ng_rt::radiation_gas_coupling_step(&mut self.local, rf, &m1p, self.cfg.simulation.dt, self.cfg.simulation.box_size);
        }
    }

    pub(crate) fn sidm(&mut self, step: u64) {
        if !self.cfg.sidm.enabled {
            return;
        }
        let sidm_params = gadget_ng_tree::SidmParams {
            sigma_m: self.cfg.sidm.sigma_m,
            v_max: self.cfg.sidm.v_max,
        };
        gadget_ng_tree::apply_sidm_scattering(
            &mut self.local,
            &sidm_params,
            self.cfg.simulation.dt,
            self.cfg.simulation.seed + step,
        );
    }

    pub(crate) fn fr(&mut self) {
        if !self.cfg.modified_gravity.enabled {
            return;
        }
        let fr_params = gadget_ng_core::FRParams {
            f_r0: self.cfg.modified_gravity.f_r0,
            n: self.cfg.modified_gravity.n,
        };
        let cosmo_params = gadget_ng_core::CosmologyParams::from_cosmology_toml(
            self.cfg.cosmology.omega_m,
            self.cfg.cosmology.omega_lambda,
            self.cfg.cosmology.h0,
            self.cfg.cosmology.w0,
            self.cfg.cosmology.wa,
            self.cfg.cosmology.m_nu_ev,
            self.cfg.cosmology.h0 * 10.0,
        );
        gadget_ng_core::apply_modified_gravity(&mut self.local, &fr_params, &cosmo_params, self.a_current);
    }

    pub(crate) fn reionization(&mut self) {
        if !self.cfg.reionization.enabled {
            return;
        }
        let _a_cur = self.a_current;
        let _z_eor = if _a_cur > 0.0 {
            1.0 / _a_cur - 1.0
        } else {
            f64::INFINITY
        };
        if !(_z_eor >= self.cfg.reionization.z_end && _z_eor <= self.cfg.reionization.z_start) {
            return;
        }
        if let Some(ref mut rf) = self.rt_field_opt {
            let m1p = gadget_ng_rt::M1Params {
                c_red_factor: self.cfg.rt.c_red_factor,
                kappa_abs: self.cfg.rt.kappa_abs,
                kappa_scat: 0.0,
                substeps: self.cfg.rt.substeps,
                sigma_dust: 0.1,
            };

            if self.sph_chem_states.len() < self.local.len() {
                let extra = self.local.len() - self.sph_chem_states.len();
                self.sph_chem_states.extend(
                    std::iter::repeat(gadget_ng_rt::ChemState::neutral()).take(extra),
                );
            } else if self.sph_chem_states.len() > self.local.len() {
                self.sph_chem_states.truncate(self.local.len());
            }

            let n_src = self.cfg.reionization.n_sources.max(1);
            let lum = self.cfg.reionization.uv_luminosity;
            let bsz = self.cfg.simulation.box_size;
            let sources: Vec<gadget_ng_rt::UvSource> = (0..n_src)
                .map(|i| {
                    let frac = (i as f64 + 0.5) / n_src as f64;
                    gadget_ng_rt::UvSource {
                        pos: gadget_ng_core::Vec3::new(frac * bsz, frac * bsz, frac * bsz),
                        luminosity: lum,
                    }
                })
                .collect();

            let _reion_state = gadget_ng_rt::reionization_step(
                rf,
                &mut self.sph_chem_states,
                &sources,
                &m1p,
                self.cfg.simulation.dt,
                bsz,
                _z_eor,
            );
        }
    }
}
