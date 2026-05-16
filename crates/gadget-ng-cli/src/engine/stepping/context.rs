//! Standalone physics step functions for `run_stepping`.
//!
//! Each `step_*` function corresponds to a physics module (SPH, MHD, RT, etc.)
//! and is called directly from the integration loops in `mod.rs`.

use gadget_ng_parallel::ParallelRuntime;

// ── Standalone physics step functions ──────────────────────────────────
//
// Called directly from the integration loops in mod.rs.

pub(crate) fn step_mhd(local: &mut [gadget_ng_core::Particle], cfg: &gadget_ng_core::RunConfig) {
    if !cfg.mhd.enabled {
        return;
    }
    let dt_alfven = gadget_ng_mhd::alfven_dt(local, cfg.mhd.cfl_mhd);
    let dt_mhd = cfg.simulation.dt.min(dt_alfven);

    // Induction + resistivity: CUDA si `cuda_mhd` opt-in, else CPU.
    #[cfg(feature = "cuda")]
    let cuda_induction_ok = if cfg.performance.use_gpu_cuda && cfg.accelerators.cuda_mhd {
        if let Ok(solver) = gadget_ng_cuda::CudaMhdSolver::try_new_checked() {
            solver
                .try_induction_resistivity(local, dt_mhd, cfg.mhd.alpha_b, cfg.simulation.box_size)
                .is_ok()
        } else {
            false
        }
    } else {
        false
    };
    #[cfg(not(feature = "cuda"))]
    let cuda_induction_ok = false;

    if !cuda_induction_ok {
        gadget_ng_mhd::advance_induction(local, dt_mhd);
        if cfg.mhd.alpha_b > 0.0 {
            gadget_ng_mhd::apply_artificial_resistivity(local, cfg.mhd.alpha_b, dt_mhd);
        }
    }

    if cfg.mhd.ambipolar_diffusion_enabled {
        #[cfg(feature = "cuda")]
        let ambipolar_cuda_ok = if cfg.performance.use_gpu_cuda && cfg.accelerators.cuda_mhd {
            if let Ok(solver) = gadget_ng_cuda::CudaMhdSolver::try_new_checked() {
                solver
                    .try_ambipolar_diffusion(
                        local,
                        cfg.mhd.ambipolar_eta,
                        cfg.mhd.ambipolar_ion_floor,
                        cfg.mhd.ambipolar_dust_coupling,
                        cfg.sph.gamma,
                        dt_mhd,
                    )
                    .is_ok()
            } else {
                false
            }
        } else {
            false
        };
        #[cfg(not(feature = "cuda"))]
        let ambipolar_cuda_ok = false;

        if !ambipolar_cuda_ok {
            gadget_ng_mhd::apply_ambipolar_diffusion(
                local,
                cfg.mhd.ambipolar_eta,
                cfg.mhd.ambipolar_ion_floor,
                cfg.mhd.ambipolar_dust_coupling,
                cfg.sph.gamma,
                dt_mhd,
            );
        }
    }

    // Magnetic forces: CUDA si `cuda_mhd` opt-in, else CPU.
    // El kernel CUDA retorna aceleraciones puras (sin dt), mismas unidades que CPU.
    #[cfg(feature = "cuda")]
    let cuda_mag_ok = if cfg.performance.use_gpu_cuda && cfg.accelerators.cuda_mhd {
        if let Ok(solver) = gadget_ng_cuda::CudaMhdSolver::try_new_checked() {
            if let Ok(acc) = solver.try_magnetic_forces(local, 1.0, cfg.simulation.box_size) {
                for (p, a) in local.iter_mut().zip(acc.iter()) {
                    if p.ptype == gadget_ng_core::ParticleType::Gas {
                        p.velocity.x += a.x * dt_mhd;
                        p.velocity.y += a.y * dt_mhd;
                        p.velocity.z += a.z * dt_mhd;
                    }
                }
                true
            } else {
                false
            }
        } else {
            false
        }
    } else {
        false
    };
    #[cfg(not(feature = "cuda"))]
    let cuda_mag_ok = false;

    if !cuda_mag_ok {
        gadget_ng_mhd::apply_magnetic_forces(local, dt_mhd);
    }

    #[cfg(feature = "cuda")]
    let dedner_cuda_ok = if cfg.performance.use_gpu_cuda && cfg.accelerators.cuda_mhd {
        let div_b = gadget_ng_mhd::compute_dedner_div_b(local);
        gadget_ng_cuda::CudaMhdSolver::try_new_checked()
            .ok()
            .and_then(|s| {
                s.try_dedner_cleaning(local, &div_b, dt_mhd, cfg.mhd.c_h, cfg.mhd.c_r)
                    .ok()
            })
            .is_some()
    } else {
        false
    };
    #[cfg(not(feature = "cuda"))]
    let dedner_cuda_ok = false;

    if !dedner_cuda_ok {
        gadget_ng_mhd::dedner_cleaning_step(local, cfg.mhd.c_h, cfg.mhd.c_r, dt_mhd);
    }

    if cfg.mhd.relativistic_mhd {
        gadget_ng_mhd::advance_srmhd(
            local,
            dt_mhd,
            gadget_ng_mhd::C_LIGHT,
            cfg.mhd.v_rel_threshold,
        );
    }

    {
        // Flux freeze: CUDA si `cuda_mhd` opt-in, else CPU.
        #[cfg(feature = "cuda")]
        let cuda_ok = if cfg.performance.use_gpu_cuda && cfg.accelerators.cuda_mhd {
            if let Ok(solver) = gadget_ng_cuda::CudaMhdSolver::try_new_checked() {
                solver
                    .try_mean_gas_density(local)
                    .ok()
                    .and_then(|rho_ref| {
                        solver
                            .try_apply_flux_freeze(
                                local,
                                cfg.sph.gamma,
                                cfg.mhd.beta_freeze,
                                rho_ref,
                            )
                            .ok()
                    })
                    .is_some()
            } else {
                false
            }
        } else {
            false
        };
        #[cfg(not(feature = "cuda"))]
        let cuda_ok = false;

        if !cuda_ok {
            let rho_ref_mhd = gadget_ng_mhd::mean_gas_density(local);
            gadget_ng_mhd::apply_flux_freeze(
                local,
                cfg.sph.gamma,
                cfg.mhd.beta_freeze,
                rho_ref_mhd,
            );
        }
    }

    // Braginskii viscosity: CUDA si `cuda_mhd` opt-in, else CPU.
    if cfg.mhd.eta_braginskii > 0.0 {
        #[cfg(feature = "cuda")]
        let cuda_brag_ok = if cfg.performance.use_gpu_cuda && cfg.accelerators.cuda_mhd {
            if let Ok(solver) = gadget_ng_cuda::CudaMhdSolver::try_new_checked() {
                solver
                    .try_braginskii_viscosity(local, dt_mhd, cfg.mhd.eta_braginskii)
                    .is_ok()
            } else {
                false
            }
        } else {
            false
        };
        #[cfg(not(feature = "cuda"))]
        let cuda_brag_ok = false;

        if !cuda_brag_ok {
            gadget_ng_mhd::apply_braginskii_viscosity(local, cfg.mhd.eta_braginskii, dt_mhd);
        }
    }

    // Reconnection + CR streaming + dynamo: CUDA si `cuda_mhd` opt-in, else CPU.
    // El kernel CUDA combina reconnection y dynamo en una sola pasada.
    if cfg.mhd.reconnection_enabled {
        #[cfg(feature = "cuda")]
        let cuda_recon_ok = if cfg.performance.use_gpu_cuda && cfg.accelerators.cuda_mhd {
            if let Ok(solver) = gadget_ng_cuda::CudaMhdSolver::try_new_checked() {
                solver
                    .try_reconnection_streaming_dynamo(
                        local,
                        dt_mhd,
                        cfg.sph.cr.streaming_coefficient,
                        cfg.mhd.f_reconnection,
                        cfg.mhd.dynamo_decay_time,
                    )
                    .is_ok()
            } else {
                false
            }
        } else {
            false
        };
        #[cfg(not(feature = "cuda"))]
        let cuda_recon_ok = false;

        if !cuda_recon_ok {
            gadget_ng_mhd::apply_magnetic_reconnection(
                local,
                cfg.mhd.f_reconnection,
                cfg.sph.gamma,
                dt_mhd,
            );
        }
    }

    // Phase 172: dinamo turbulento α-effect
    if cfg.mhd.dynamo_enabled {
        let mut v2_sum = 0.0_f64;
        let mut n_gas = 0usize;
        for p in local.iter() {
            if p.ptype == gadget_ng_core::ParticleType::Gas {
                v2_sum += p.velocity.x * p.velocity.x
                    + p.velocity.y * p.velocity.y
                    + p.velocity.z * p.velocity.z;
                n_gas += 1;
            }
        }
        let v_rms = if n_gas > 0 {
            (v2_sum / n_gas as f64).sqrt().max(1e-30)
        } else {
            1e-10
        };
        gadget_ng_mhd::apply_turbulent_dynamo(local, v_rms, dt_mhd, cfg.mhd.dynamo_decay_time);
    }
}

pub(crate) fn step_rt(
    local: &mut [gadget_ng_core::Particle],
    rt_field: &mut Option<gadget_ng_rt::RadiationField>,
    cfg: &gadget_ng_core::RunConfig,
) {
    if !cfg.rt.enabled {
        return;
    }
    if let Some(rf) = rt_field {
        let m1p = gadget_ng_rt::M1Params {
            c_red_factor: cfg.rt.c_red_factor,
            kappa_abs: cfg.rt.kappa_abs,
            kappa_scat: 0.0,
            substeps: cfg.rt.substeps,
            sigma_dust: 0.1,
        };

        // M1 advección: CUDA path opt-in si cuda_rt = true.
        #[cfg(feature = "cuda")]
        let m1_cuda_ok = if cfg.performance.use_gpu_cuda && cfg.accelerators.cuda_rt {
            if let Ok(solver) = gadget_ng_cuda::CudaRtSolver::try_new_checked() {
                solver.try_m1_advection(rf, cfg.simulation.dt, &m1p).is_ok()
            } else {
                false
            }
        } else {
            false
        };
        #[cfg(not(feature = "cuda"))]
        let m1_cuda_ok = false;

        if !m1_cuda_ok {
            gadget_ng_rt::m1_update(rf, cfg.simulation.dt, &m1p);
        }

        // RT→gas coupling: CUDA path para photoheating si disponible.
        #[cfg(feature = "cuda")]
        let cuda_ok = if cfg.performance.use_gpu_cuda && cfg.accelerators.cuda_rt {
            if let Ok(solver) = gadget_ng_cuda::CudaRtSolver::try_new_checked() {
                let gamma = gadget_ng_rt::photoionization_rate(rf, &m1p);
                solver
                    .try_apply_photoheating(
                        local,
                        rf,
                        &gamma,
                        cfg.simulation.dt,
                        cfg.simulation.box_size,
                    )
                    .is_ok()
            } else {
                false
            }
        } else {
            false
        };
        #[cfg(not(feature = "cuda"))]
        let cuda_ok = false;

        if !cuda_ok {
            gadget_ng_rt::radiation_gas_coupling_step(
                local,
                rf,
                &m1p,
                cfg.simulation.dt,
                cfg.simulation.box_size,
            );
        }
    }
}

pub(crate) fn step_sidm(
    local: &mut [gadget_ng_core::Particle],
    cfg: &gadget_ng_core::RunConfig,
    step: u64,
) {
    if !cfg.sidm.enabled {
        return;
    }
    let sidm_params = gadget_ng_tree::SidmParams {
        sigma_m: cfg.sidm.sigma_m,
        v_max: cfg.sidm.v_max,
    };

    // CUDA path: opt-in con cuda_tree + use_gpu_cuda (smoke/parity kernel).
    #[cfg(feature = "cuda")]
    let cuda_sidm_ok = if cfg.performance.use_gpu_cuda && cfg.accelerators.cuda_tree {
        if let Ok(solver) = gadget_ng_cuda::CudaTreeSolver::try_new_checked() {
            let h = local
                .iter()
                .map(|p| p.smoothing_length)
                .fold(0.0_f64, f64::max)
                .max(1e-10);
            solver
                .try_sidm_scatter(local, cfg.simulation.dt, cfg.sidm.sigma_m, h)
                .is_ok()
        } else {
            false
        }
    } else {
        false
    };
    #[cfg(not(feature = "cuda"))]
    let cuda_sidm_ok = false;

    if !cuda_sidm_ok {
        gadget_ng_tree::apply_sidm_scattering(
            local,
            &sidm_params,
            cfg.simulation.dt,
            cfg.simulation.seed + step,
        );
    }
}

pub(crate) fn step_fr(
    local: &mut [gadget_ng_core::Particle],
    cfg: &gadget_ng_core::RunConfig,
    a_current: f64,
) {
    if !cfg.modified_gravity.enabled {
        return;
    }
    if cfg.gravity.solver == gadget_ng_core::SolverKind::Pm {
        // El solver PM CPU aplica f(R) homogéneo o screening de malla en k-space.
        // Evitamos escalar dos veces la aceleración en ese camino.
        return;
    }
    let fr_params = gadget_ng_core::FRParams {
        f_r0: cfg.modified_gravity.f_r0,
        n: cfg.modified_gravity.n,
    };
    let cosmo_params = gadget_ng_core::CosmologyParams::from_cosmology_toml(
        cfg.cosmology.omega_m,
        cfg.cosmology.omega_lambda,
        cfg.cosmology.h0,
        cfg.cosmology.w0,
        cfg.cosmology.wa,
        cfg.cosmology.m_nu_ev,
        cfg.cosmology.h0 * 10.0,
    );
    gadget_ng_core::apply_modified_gravity(local, &fr_params, &cosmo_params, a_current);
}

pub(crate) fn step_agn(
    local: &mut [gadget_ng_core::Particle],
    cfg: &gadget_ng_core::RunConfig,
    agn_bhs: &mut Vec<gadget_ng_sph::BlackHole>,
    halo_centers: &[gadget_ng_core::Vec3],
) {
    if !cfg.sph.agn.enabled {
        return;
    }
    let periodic_box_agn = cfg.cosmology.periodic.then_some(cfg.simulation.box_size);
    let agn_params = gadget_ng_sph::AgnParams {
        eps_feedback: cfg.sph.agn.eps_feedback,
        m_seed: cfg.sph.agn.m_seed,
        v_kick_agn: cfg.sph.agn.v_kick_agn,
        r_influence: cfg.sph.agn.r_influence,
    };
    if cfg.sph.agn.pbh_seeding_enabled && agn_bhs.is_empty() {
        let pbh_params = gadget_ng_sph::PbhSeedingParams {
            n_seeds: cfg.sph.agn.pbh_n_seeds,
            m_seed: cfg.sph.agn.pbh_m_seed,
            min_host_mass: cfg.sph.agn.pbh_min_host_mass,
            seed: cfg.sph.agn.pbh_seed,
            initial_spin: cfg.sph.agn.initial_spin,
            host_kind: cfg.sph.agn.pbh_host_kind,
        };
        gadget_ng_sph::seed_primordial_black_holes(agn_bhs, local, &pbh_params);
    }
    let n_bh = cfg.sph.agn.n_agn_bh.max(1);
    if !halo_centers.is_empty() && (!cfg.sph.agn.pbh_seeding_enabled || agn_bhs.is_empty()) {
        let n_new = halo_centers.len().min(n_bh);
        if agn_bhs.len() != n_new {
            agn_bhs.resize_with(n_new, || {
                gadget_ng_sph::BlackHole::with_spin(
                    gadget_ng_core::Vec3::zero(),
                    agn_params.m_seed,
                    cfg.sph.agn.initial_spin,
                )
            });
        }
        for (bh, &pos) in agn_bhs.iter_mut().zip(halo_centers.iter()) {
            bh.pos = pos;
        }
    } else if agn_bhs.is_empty() {
        let center = cfg.simulation.box_size * 0.5;
        agn_bhs.push(gadget_ng_sph::BlackHole::with_spin(
            gadget_ng_core::Vec3::new(center, center, center),
            agn_params.m_seed,
            cfg.sph.agn.initial_spin,
        ));
    }
    gadget_ng_sph::grow_black_holes_periodic(
        agn_bhs,
        local,
        &agn_params,
        cfg.simulation.dt,
        periodic_box_agn,
    );
    if cfg.sph.agn.mergers_enabled {
        gadget_ng_sph::merge_black_holes(
            agn_bhs,
            cfg.sph.agn.merger_radius,
            cfg.sph.agn.recoil_velocity_scale,
            periodic_box_agn,
        );
    }
    gadget_ng_sph::apply_agn_feedback_bimodal_periodic(
        local,
        agn_bhs,
        &agn_params,
        cfg.sph.agn.f_edd_threshold,
        cfg.sph.agn.r_bubble,
        cfg.sph.agn.eps_radio,
        cfg.simulation.dt,
        periodic_box_agn,
    );
}

pub(crate) fn step_insitu(
    local: &[gadget_ng_core::Particle],
    cfg: &gadget_ng_core::RunConfig,
    a_current: f64,
    step: u64,
    out_dir: &std::path::Path,
    sph_chem_states: &[gadget_ng_rt::ChemState],
) -> Vec<gadget_ng_core::Vec3> {
    let (insitu_ran, insitu_fx) = crate::insitu::maybe_run_insitu(
        local,
        &cfg.insitu_analysis,
        cfg.simulation.box_size,
        a_current,
        step,
        out_dir,
        if sph_chem_states.is_empty() {
            None
        } else {
            Some(sph_chem_states)
        },
        cfg.cosmology.h0,
        cfg.cosmology.omega_m,
        cfg.cosmology.omega_lambda,
    );
    if insitu_ran && !insitu_fx.halo_centers.is_empty() {
        insitu_fx.halo_centers
    } else {
        Vec::new()
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn step_checkpoint<R: ParallelRuntime + ?Sized>(
    rt: &R,
    step: u64,
    a_current: f64,
    local: &[gadget_ng_core::Particle],
    total: usize,
    h_state_opt: Option<&gadget_ng_integrators::HierarchicalState>,
    out_dir: &std::path::Path,
    cfg_hash: &str,
    agn_bhs: &[gadget_ng_sph::BlackHole],
    chem: &[gadget_ng_rt::ChemState],
    checkpoint_interval: u64,
) -> Result<(), crate::error::CliError> {
    if checkpoint_interval > 0 && step.is_multiple_of(checkpoint_interval) {
        crate::engine::checkpoint::save_checkpoint(
            rt,
            step,
            a_current,
            local,
            total,
            h_state_opt,
            out_dir,
            cfg_hash,
            agn_bhs,
            chem,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments, clippy::collapsible_if)]
pub(crate) fn step_snap_frame<R: ParallelRuntime + ?Sized>(
    rt: &R,
    step: u64,
    a_current: f64,
    local: &[gadget_ng_core::Particle],
    total: usize,
    cfg: &gadget_ng_core::RunConfig,
    out_dir: &std::path::Path,
    prov: &gadget_ng_io::Provenance,
    snapshot_interval: u64,
) -> Result<(), crate::error::CliError> {
    if snapshot_interval > 0 && step.is_multiple_of(snapshot_interval) {
        if let Some(all_parts) = rt.root_gather_particles(local, total) {
            let frame_dir = out_dir.join("frames").join(format!("snap_{:06}", step));
            std::fs::create_dir_all(&frame_dir)
                .map_err(|e| crate::error::CliError::io(&frame_dir, e))?;
            let t = step as f64 * cfg.simulation.dt;
            let z = if cfg.cosmology.enabled {
                1.0 / a_current - 1.0
            } else {
                0.0
            };
            let env = crate::engine::provenance::snapshot_env_for(cfg, t, z);
            gadget_ng_io::write_snapshot_formatted(
                cfg.output.snapshot_format,
                &frame_dir,
                &all_parts,
                prov,
                &env,
            )?;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn step_sph(
    local: &mut Vec<gadget_ng_core::Particle>,
    cfg: &gadget_ng_core::RunConfig,
    cosmo_state: &Option<(gadget_ng_core::cosmology::CosmologyParams, f64)>,
    a_current: f64,
    rt_rank: usize,
    sph_step: u64,
) {
    if !cfg.sph.enabled {
        return;
    }
    let cf_sph = match cosmo_state {
        Some((cp, _)) => {
            let (d, k, k2) = cp.drift_kick_factors(a_current, cfg.simulation.dt);
            gadget_ng_integrators::CosmoFactors {
                drift: d,
                kick_half: k,
                kick_half2: k2,
            }
        }
        None => gadget_ng_integrators::CosmoFactors::flat(cfg.simulation.dt),
    };
    let gamma = cfg.sph.gamma;
    let alpha = cfg.sph.alpha_visc;
    let n_neigh = cfg.sph.n_neigh as f64;
    let periodic_box = cfg.cosmology.periodic.then_some(cfg.simulation.box_size);

    // CUDA SPH path: densidad + Balsara + fuerzas en GPU, integración KDK en CPU.
    // Activo si `[accelerators] cuda_sph = true` y `use_gpu_cuda = true`.
    #[cfg(feature = "cuda")]
    let cuda_sph_ok = if cfg.performance.use_gpu_cuda && cfg.accelerators.cuda_sph {
        if let Ok(solver) = gadget_ng_cuda::CudaSphSolver::try_new_checked() {
            let pbox = periodic_box;
            let mut sph_ok = true;

            // Paso 1: fuerzas al inicio del paso (para Kick1)
            match solver.try_sph_density_and_forces_core(local, pbox) {
                Ok((_, acc1, du_dt1)) => {
                    // Kick 1
                    for (i, p) in local.iter_mut().enumerate() {
                        p.velocity += p.acceleration * cf_sph.kick_half;
                        if p.ptype == gadget_ng_core::ParticleType::Gas {
                            p.velocity += acc1[i] * cf_sph.kick_half;
                            p.internal_energy =
                                (p.internal_energy + du_dt1[i] * cf_sph.kick_half).max(0.0);
                        }
                    }
                    // Drift
                    for p in local.iter_mut() {
                        p.position += p.velocity * cf_sph.drift;
                    }
                    if let Some(l) = pbox
                        && l > 0.0
                    {
                        for p in local.iter_mut() {
                            p.position =
                                gadget_ng_core::cosmology::wrap_position(p.position, l);
                        }
                    }

                    // Paso 2: fuerzas al nuevo tiempo (para Kick2)
                    match solver.try_sph_density_and_forces_core(local, pbox) {
                        Ok((_, acc2, du_dt2)) => {
                            // Kick 2
                            for (i, p) in local.iter_mut().enumerate() {
                                p.velocity += p.acceleration * cf_sph.kick_half2;
                                if p.ptype == gadget_ng_core::ParticleType::Gas {
                                    p.velocity += acc2[i] * cf_sph.kick_half2;
                                    p.internal_energy =
                                        (p.internal_energy + du_dt2[i] * cf_sph.kick_half2)
                                            .max(0.0);
                                }
                            }
                        }
                        Err(_) => sph_ok = false,
                    }
                }
                Err(_) => sph_ok = false,
            }
            sph_ok
        } else {
            false
        }
    } else {
        false
    };

    #[cfg(not(feature = "cuda"))]
    let cuda_sph_ok = false;

    if !cuda_sph_ok {
        gadget_ng_sph::sph_cosmo_kdk_step(
            local,
            cf_sph,
            gamma,
            alpha,
            n_neigh,
            periodic_box,
            |_parts| {},
        );
    }

    if cfg.sph.cooling != gadget_ng_core::CoolingKind::None {
        let redshift = if a_current > 0.0 {
            1.0 / a_current - 1.0
        } else {
            0.0
        };
        // CUDA path para cooling (reemplaza apply_cooling_*).
        #[cfg(feature = "cuda")]
        let cuda_cooling = if cfg.performance.use_gpu_cuda && cfg.accelerators.cuda_cooling {
            if let Ok(solver) = gadget_ng_cuda::CudaCoolingSolver::try_new_checked() {
                let f_mag_method = if cfg.sph.mag_suppress_cooling > 0.0 && cfg.mhd.enabled {
                    cfg.sph.mag_suppress_cooling
                } else {
                    0.0
                };
                solver
                    .try_apply_cooling(local, &cfg.sph, cfg.simulation.dt, redshift, f_mag_method)
                    .is_ok()
            } else {
                false
            }
        } else {
            false
        };
        #[cfg(not(feature = "cuda"))]
        let cuda_cooling = false;

        if !cuda_cooling {
            if cfg.sph.mag_suppress_cooling > 0.0 && cfg.mhd.enabled {
                gadget_ng_sph::apply_cooling_mhd_with_redshift(
                    local,
                    &cfg.sph,
                    cfg.simulation.dt,
                    redshift,
                );
            } else {
                gadget_ng_sph::apply_cooling_with_redshift(
                    local,
                    &cfg.sph,
                    cfg.simulation.dt,
                    redshift,
                );
            }
        }
    }

    if cfg.sph.dust.enabled {
        #[cfg(feature = "cuda")]
        let cuda_dust = if cfg.performance.use_gpu_cuda && cfg.accelerators.cuda_dust {
            if let Ok(solver) = gadget_ng_cuda::CudaDustSolver::try_new_checked() {
                solver
                    .try_update_dust(local, &cfg.sph.dust, cfg.sph.gamma, cfg.simulation.dt)
                    .is_ok()
            } else {
                false
            }
        } else {
            false
        };
        #[cfg(not(feature = "cuda"))]
        let cuda_dust = false;

        if !cuda_dust {
            gadget_ng_sph::update_dust(local, &cfg.sph.dust, cfg.sph.gamma, cfg.simulation.dt);
        }
    }
    if cfg.sph.dust.enabled && cfg.sph.dust.radiation_pressure_enabled {
        let z_ref = 0.5 * cfg.simulation.box_size;
        gadget_ng_sph::apply_dust_radiation_pressure_kick(
            local,
            &cfg.sph.dust,
            z_ref,
            cfg.simulation.dt,
        );
    }

    if cfg.sph.conduction.enabled {
        if cfg.sph.conduction.anisotropic {
            // CUDA: try_anisotropic_conduction — kernel pairwise O(N²), Wendland-C6 (AP-17).
            #[cfg(feature = "cuda")]
            let cond_cuda_ok = if cfg.performance.use_gpu_cuda && cfg.accelerators.cuda_mhd {
                gadget_ng_cuda::CudaMhdSolver::try_new_checked()
                    .ok()
                    .and_then(|s| {
                        s.try_anisotropic_conduction(
                            local,
                            cfg.sph.conduction.kappa_par,
                            cfg.sph.conduction.kappa_perp,
                            cfg.sph.gamma,
                            cfg.simulation.dt,
                        )
                        .ok()
                    })
                    .is_some()
            } else {
                false
            };
            #[cfg(not(feature = "cuda"))]
            let cond_cuda_ok = false;

            if !cond_cuda_ok {
                gadget_ng_mhd::apply_anisotropic_conduction(
                    local,
                    cfg.sph.conduction.kappa_par,
                    cfg.sph.conduction.kappa_perp,
                    cfg.sph.gamma,
                    cfg.simulation.dt,
                );
            }
        } else {
            gadget_ng_sph::apply_thermal_conduction_periodic(
                local,
                &cfg.sph.conduction,
                cfg.sph.gamma,
                cfg.sph.t_floor_k,
                cfg.simulation.dt,
                periodic_box,
            );
        }
    }

    if cfg.sph.molecular.enabled {
        let dust_cfg = cfg.sph.dust.enabled.then_some(&cfg.sph.dust);
        #[cfg(feature = "cuda")]
        let cuda_h2 = if cfg.performance.use_gpu_cuda && cfg.accelerators.cuda_h2 {
            if let Ok(solver) = gadget_ng_cuda::CudaMolecularSolver::try_new_checked() {
                solver
                    .try_update_h2(local, &cfg.sph.molecular, dust_cfg, cfg.simulation.dt)
                    .is_ok()
            } else {
                false
            }
        } else {
            false
        };
        #[cfg(not(feature = "cuda"))]
        let cuda_h2 = false;

        if !cuda_h2 {
            gadget_ng_sph::update_h2_fraction_with_dust(
                local,
                &cfg.sph.molecular,
                dust_cfg,
                cfg.simulation.dt,
            );
        }
    }

    if cfg.turbulence.enabled {
        gadget_ng_mhd::apply_turbulent_forcing(local, &cfg.turbulence, cfg.simulation.dt, sph_step);
    }
    if cfg.two_fluid.enabled {
        #[cfg(feature = "cuda")]
        let two_fluid_cuda_ok = if cfg.performance.use_gpu_cuda && cfg.accelerators.cuda_mhd {
            if let Ok(solver) = gadget_ng_cuda::CudaMhdSolver::try_new_checked() {
                solver
                    .try_electron_ion_coupling(local, cfg.two_fluid.nu_ei_coeff, cfg.simulation.dt)
                    .is_ok()
            } else {
                false
            }
        } else {
            false
        };
        #[cfg(not(feature = "cuda"))]
        let two_fluid_cuda_ok = false;

        if !two_fluid_cuda_ok {
            gadget_ng_mhd::apply_electron_ion_coupling(local, &cfg.two_fluid, cfg.simulation.dt);
        }
    }

    if cfg.sph.ism.enabled {
        let sfr_ism = gadget_ng_sph::compute_sfr_model(local, &cfg.sph.feedback, cfg.sph.gamma);
        let rho_sf = cfg.sph.feedback.rho_sf;
        gadget_ng_sph::update_ism_phases(local, &sfr_ism, rho_sf, &cfg.sph.ism, cfg.simulation.dt);
        gadget_ng_sph::apply_phase_transitions(
            local,
            cfg.simulation.dt,
            cfg.sph.gamma,
            cfg.sph.ism.q_star,
        );
    }

    if cfg.sph.feedback.enabled {
        let sfr = gadget_ng_sph::compute_sfr_model(local, &cfg.sph.feedback, cfg.sph.gamma);
        let mut fb_seed = sph_step
            .wrapping_mul(2654435761)
            .wrapping_add(rt_rank as u64);
        match cfg.sph.feedback.feedback_mode {
            gadget_ng_core::StellarFeedbackMode::Kinetic => {
                gadget_ng_sph::apply_sn_feedback(
                    local,
                    &sfr,
                    &cfg.sph.feedback,
                    cfg.simulation.dt,
                    &mut fb_seed,
                );
            }
            gadget_ng_core::StellarFeedbackMode::ThermalStochastic => {
                gadget_ng_sph::apply_thermal_feedback_stochastic(
                    local,
                    &sfr,
                    &cfg.sph.feedback,
                    cfg.sph.gamma,
                    cfg.simulation.dt,
                    &mut fb_seed,
                    periodic_box,
                );
            }
        }

        if cfg.sph.feedback.stellar_wind_enabled {
            let mut wind_seed = fb_seed.wrapping_add(0xC0FFEE);
            gadget_ng_sph::apply_stellar_wind_feedback(
                local,
                &sfr,
                &cfg.sph.feedback,
                cfg.simulation.dt,
                &mut wind_seed,
            );
        }

        if cfg.sph.cr.enabled {
            gadget_ng_sph::inject_cr_from_sn(
                local,
                &sfr,
                cfg.sph.cr.cr_fraction,
                cfg.simulation.dt,
            );
            if cfg.sph.cr.anisotropic_diffusion && cfg.mhd.enabled {
                // CUDA: try_cr_diffusion_anisotropic — kernel pairwise O(N²), Wendland-C6 (AP-17).
                #[cfg(feature = "cuda")]
                let cr_diff_cuda_ok = if cfg.performance.use_gpu_cuda && cfg.accelerators.cuda_mhd {
                    gadget_ng_cuda::CudaMhdSolver::try_new_checked()
                        .ok()
                        .and_then(|s| {
                            s.try_cr_diffusion_anisotropic(
                                local,
                                cfg.sph.cr.kappa_cr,
                                cfg.simulation.dt,
                            )
                            .ok()
                        })
                        .is_some()
                } else {
                    false
                };
                #[cfg(not(feature = "cuda"))]
                let cr_diff_cuda_ok = false;

                if !cr_diff_cuda_ok {
                    gadget_ng_mhd::diffuse_cr_anisotropic(
                        local,
                        cfg.sph.cr.kappa_cr,
                        cfg.sph.cr.b_cr_suppress,
                        cfg.simulation.dt,
                    );
                }
            } else {
                gadget_ng_sph::diffuse_cr_periodic(
                    local,
                    cfg.sph.cr.kappa_cr,
                    cfg.sph.cr.b_cr_suppress,
                    cfg.simulation.dt,
                    periodic_box,
                );
            }
            if cfg.sph.cr.hadronic_loss_coeff > 0.0 {
                gadget_ng_sph::apply_cr_hadronic_losses(
                    local,
                    cfg.sph.cr.hadronic_loss_coeff,
                    cfg.simulation.dt,
                );
            }
            if cfg.sph.cr.streaming_coefficient > 0.0 && cfg.mhd.enabled {
                // CUDA path: CR streaming + backreaction bajo cuda_cr.
                // El fallback CPU aplica streaming_crk + cr_pressure_backreaction.
                #[cfg(feature = "cuda")]
                let cr_cuda_ok = if cfg.performance.use_gpu_cuda && cfg.accelerators.cuda_cr {
                    if let Ok(solver) = gadget_ng_cuda::CudaMhdSolver::try_new_checked() {
                        let stream_ok = solver
                            .try_cr_streaming(
                                local,
                                cfg.simulation.dt,
                                cfg.sph.cr.streaming_coefficient,
                                periodic_box,
                            )
                            .is_ok();
                        let back_ok = if stream_ok {
                            solver
                                .try_cr_backreaction(local, periodic_box)
                                .map(|accels| {
                                    for (p, a) in local.iter_mut().zip(accels.iter()) {
                                        p.acceleration.x += a.x;
                                        p.acceleration.y += a.y;
                                        p.acceleration.z += a.z;
                                    }
                                })
                                .is_ok()
                        } else {
                            false
                        };
                        stream_ok && back_ok
                    } else {
                        false
                    }
                } else {
                    false
                };
                #[cfg(not(feature = "cuda"))]
                let cr_cuda_ok = false;

                if !cr_cuda_ok {
                    gadget_ng_mhd::streaming_crk(
                        local,
                        cfg.simulation.dt,
                        cfg.sph.cr.streaming_coefficient,
                        periodic_box,
                    );
                    gadget_ng_mhd::cr_pressure_backreaction(local, periodic_box);
                }
            }
        }

        let mut spawn_seed = fb_seed.wrapping_add(0xDEAD_BEEF);
        let mut next_gid = local.iter().map(|p| p.global_id).max().unwrap_or(0) + 1;
        let (new_stars, to_remove) = gadget_ng_sph::spawn_star_particles(
            local,
            &sfr,
            cfg.simulation.dt,
            &mut spawn_seed,
            &cfg.sph.feedback,
            &mut next_gid,
        );
        let mut remove_sorted = to_remove;
        remove_sorted.sort_unstable_by(|a, b| b.cmp(a));
        remove_sorted.dedup();
        for idx in remove_sorted {
            local.swap_remove(idx);
        }
        local.extend(new_stars);

        let dt_gyr = cfg.simulation.dt * 1e-3;
        gadget_ng_sph::advance_stellar_ages(local, dt_gyr);
        let mut ia_seed = fb_seed.wrapping_add(0xF00D);
        gadget_ng_sph::apply_snia_feedback_periodic(
            local,
            dt_gyr,
            &mut ia_seed,
            &cfg.sph.feedback,
            periodic_box,
        );
    }
}

pub(crate) fn step_reionization(
    local: &mut [gadget_ng_core::Particle],
    rt_field_opt: &mut Option<gadget_ng_rt::RadiationField>,
    sph_chem_states: &mut Vec<gadget_ng_rt::ChemState>,
    cfg: &gadget_ng_core::RunConfig,
    a_current: f64,
) {
    if !cfg.reionization.enabled {
        return;
    }
    let z_eor = if a_current > 0.0 {
        1.0 / a_current - 1.0
    } else {
        f64::INFINITY
    };
    if !(z_eor >= cfg.reionization.z_end && z_eor <= cfg.reionization.z_start) {
        return;
    }
    if let Some(rf) = rt_field_opt {
        let m1p = gadget_ng_rt::M1Params {
            c_red_factor: cfg.rt.c_red_factor,
            kappa_abs: cfg.rt.kappa_abs,
            kappa_scat: 0.0,
            substeps: cfg.rt.substeps,
            sigma_dust: 0.1,
        };
        if sph_chem_states.len() < local.len() {
            let extra = local.len() - sph_chem_states.len();
            sph_chem_states.extend(std::iter::repeat_n(
                gadget_ng_rt::ChemState::neutral(),
                extra,
            ));
        } else if sph_chem_states.len() > local.len() {
            sph_chem_states.truncate(local.len());
        }
        let n_src = cfg.reionization.n_sources.max(1);
        let lum = cfg.reionization.uv_luminosity;
        let bsz = cfg.simulation.box_size;
        let sources: Vec<gadget_ng_rt::UvSource> = (0..n_src)
            .map(|i| {
                let frac = (i as f64 + 0.5) / n_src as f64;
                gadget_ng_rt::UvSource {
                    pos: gadget_ng_core::Vec3::new(frac * bsz, frac * bsz, frac * bsz),
                    luminosity: lum,
                }
            })
            .collect();

        // ── RT chemistry CUDA path ────────────────────────────────────────────
        // try_chemistry_rates + try_apply_chemistry bajo cuda_rt_chem.
        // El fallback CPU es apply_chemistry (tasas + stiff solver en un paso).
        #[cfg(feature = "cuda")]
        let chem_cuda_ok = if cfg.performance.use_gpu_cuda && cfg.accelerators.cuda_rt_chem {
            if let Ok(solver) = gadget_ng_cuda::CudaRtSolver::try_new_checked() {
                let gamma_hi = solver
                    .try_chemistry_rates(local, rf, &m1p, bsz)
                    .unwrap_or_default();
                let temperatures: Vec<f64> = local
                    .iter()
                    .map(|p| {
                        ((5.0 / 3.0 - 1.0) * p.internal_energy * 1.0e10 / 1.38065e-16_f64).max(1.0)
                    })
                    .collect();
                let ptypes: Vec<gadget_ng_core::ParticleType> =
                    local.iter().map(|p| p.ptype).collect();
                solver
                    .try_apply_chemistry(
                        sph_chem_states,
                        &gamma_hi,
                        &temperatures,
                        &ptypes,
                        cfg.simulation.dt,
                        gadget_ng_rt::chemistry::ChemParams::default().n_h_ref,
                    )
                    .is_ok()
            } else {
                false
            }
        } else {
            false
        };
        #[cfg(not(feature = "cuda"))]
        let chem_cuda_ok = false;

        if !chem_cuda_ok {
            gadget_ng_rt::apply_chemistry(
                local,
                sph_chem_states,
                rf,
                &gadget_ng_rt::chemistry::ChemParams::default(),
                cfg.simulation.dt,
            );
        }

        // ── RT reionization stats CUDA path ───────────────────────────────────
        // try_reionization_stats bajo cuda_rt_chem.
        // El resultado ReionizationState actualmente se descarta en ambos paths.
        #[cfg(feature = "cuda")]
        let reion_cuda_ok = if cfg.performance.use_gpu_cuda && cfg.accelerators.cuda_rt_chem {
            if let Ok(solver) = gadget_ng_cuda::CudaRtSolver::try_new_checked() {
                solver
                    .try_reionization_stats(sph_chem_states, z_eor, n_src)
                    .is_ok()
            } else {
                false
            }
        } else {
            false
        };
        #[cfg(not(feature = "cuda"))]
        let reion_cuda_ok = false;

        if !reion_cuda_ok {
            let _reion_state =
                gadget_ng_rt::compute_reionization_state(sph_chem_states, z_eor, n_src);
        }

        // ── RT 21cm CUDA path ─────────────────────────────────────────────────
        // try_cm21_field bajo cuda_rt_chem; resultado actualmente se descarta
        // (guardado en log; integración futura con InsituResult).
        #[cfg(feature = "cuda")]
        if cfg.performance.use_gpu_cuda
            && cfg.accelerators.cuda_rt_chem
            && let Ok(solver) = gadget_ng_cuda::CudaRtSolver::try_new_checked()
        {
            let overdensity: Vec<f64> = local
                .iter()
                .map(|p| {
                    if p.smoothing_length > 0.0 {
                        p.mass / (p.smoothing_length * p.smoothing_length * p.smoothing_length)
                    } else {
                        1.0
                    }
                })
                .collect();
            let _ = solver.try_cm21_field(sph_chem_states, &overdensity, z_eor);
        }

        // ── reionization_step: deposit UV sources + M1 update ────────────────
        // (no tiene path CUDA propio; ya se cubrió M1 advection en step_rt)
        let _ = gadget_ng_rt::reionization_step(
            rf,
            sph_chem_states,
            &sources,
            &m1p,
            cfg.simulation.dt,
            bsz,
            z_eor,
        );
    }
}
