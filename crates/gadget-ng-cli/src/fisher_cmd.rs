use gadget_ng_analysis::fisher::{
    FisherConfig, FisherParams, correlation_matrix, fisher_matrix, fisher_uncertainties,
    k_bins_default,
};
use std::fs;
use std::path::Path;

use crate::error::CliError;

#[allow(clippy::too_many_arguments)]
pub fn run_fisher(
    omega_m: f64,
    omega_b: f64,
    h: f64,
    n_s: f64,
    sigma8: f64,
    w0: f64,
    wa: f64,
    m_nu_ev: f64,
    step_frac: f64,
    survey_volume: f64,
    use_nonlinear: bool,
    out: &Path,
) -> Result<(), CliError> {
    let fiducial = FisherParams {
        omega_m,
        omega_b,
        h,
        n_s,
        sigma8,
        w0,
        wa,
        m_nu_ev,
    };

    let config = FisherConfig {
        step_frac,
        k_bins: k_bins_default(),
        redshifts: vec![0.0, 0.5, 1.0, 2.0],
        survey_volume,
        use_nonlinear,
    };

    eprintln!("[gadget-ng] Fisher matrix: computing...");
    eprintln!(
        "[gadget-ng] Fiducial: Om={omega_m}, Ob={omega_b}, h={h}, ns={n_s}, s8={sigma8}, w0={w0}, wa={wa}, mnu={m_nu_ev}"
    );
    eprintln!(
        "[gadget-ng] Config: {} k-bins, {} redshifts, V={:.2e} (Mpc/h)^3, step={step_frac}%",
        config.k_bins.len(),
        config.redshifts.len(),
        config.survey_volume,
    );

    let fisher = fisher_matrix(&config, &fiducial);
    let sigmas = fisher_uncertainties(&fisher);
    let corr = correlation_matrix(&fisher);

    eprintln!("[gadget-ng] Marginalized 1-sigma uncertainties:");
    for (i, name) in sigmas.param_names.iter().enumerate() {
        eprintln!("  sigma({name}) = {:.6e}", sigmas.sigmas[i]);
    }

    #[derive(serde::Serialize)]
    struct FisherOutput {
        fiducial: FiducialCosmo,
        config: FisherConfigJson,
        uncertainties: gadget_ng_analysis::FisherUncertainties,
        correlation_matrix: Vec<f64>,
        fisher_matrix: Vec<f64>,
    }

    #[derive(serde::Serialize)]
    struct FiducialCosmo {
        omega_m: f64,
        omega_b: f64,
        h: f64,
        n_s: f64,
        sigma8: f64,
        w0: f64,
        wa: f64,
        m_nu_ev: f64,
    }

    #[derive(serde::Serialize)]
    struct FisherConfigJson {
        step_frac: f64,
        k_bins: Vec<f64>,
        redshifts: Vec<f64>,
        survey_volume: f64,
        use_nonlinear: bool,
    }

    let output = FisherOutput {
        fiducial: FiducialCosmo {
            omega_m: fiducial.omega_m,
            omega_b: fiducial.omega_b,
            h: fiducial.h,
            n_s: fiducial.n_s,
            sigma8: fiducial.sigma8,
            w0: fiducial.w0,
            wa: fiducial.wa,
            m_nu_ev: fiducial.m_nu_ev,
        },
        config: FisherConfigJson {
            step_frac: config.step_frac,
            k_bins: config.k_bins,
            redshifts: config.redshifts,
            survey_volume: config.survey_volume,
            use_nonlinear: config.use_nonlinear,
        },
        uncertainties: sigmas,
        correlation_matrix: corr,
        fisher_matrix: fisher.matrix,
    };

    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent).map_err(|e| CliError::Io {
            path: parent.to_path_buf(),
            source: e,
        })?;
    }
    let json_str = serde_json::to_string_pretty(&output)?;
    fs::write(out, json_str).map_err(|e| CliError::Io {
        path: out.to_path_buf(),
        source: e,
    })?;
    eprintln!("[gadget-ng] Fisher output written to {:?}", out);

    Ok(())
}
