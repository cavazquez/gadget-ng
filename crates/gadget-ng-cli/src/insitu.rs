//! Análisis in-situ durante el loop `stepping` (Phase 63).
//!
//! Calcula P(k), FoF y opcionalmente ξ(r) en intervalos configurables,
//! escribiendo `insitu_NNNNNN.json` en `<output_dir>/`.

use gadget_ng_analysis::{
    analyse, compute_pk_multipoles, two_point_correlation_fft,
    AnalysisParams, LosAxis, PkRsdParams, XiBin,
};
use gadget_ng_analysis::{PkBin, PkMultipoleBin, PkRsdBin};
use gadget_ng_core::{InsituAnalysisSection, Particle};
use serde::Serialize;
use std::path::Path;

/// Salida de un paso de análisis in-situ.
#[derive(Debug, Serialize)]
pub struct InsituResult {
    pub step: u64,
    pub a: f64,
    pub z: f64,
    pub n_halos: usize,
    pub m_total_halos: f64,
    pub power_spectrum: Vec<PkBinOut>,
    pub xi_r: Vec<XiBinOut>,
    /// P(k,μ) en espacio de redshift. Vacío si `pk_rsd_bins == 0`.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub pk_rsd: Vec<PkRsdBinOut>,
    /// Multipoles P₀/P₂/P₄. Vacío si `pk_rsd_bins == 0`.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub pk_multipoles: Vec<PkMultipoleOut>,
}

#[derive(Debug, Serialize)]
pub struct PkBinOut {
    pub k: f64,
    pub pk: f64,
    pub n_modes: usize,
}

#[derive(Debug, Serialize)]
pub struct XiBinOut {
    pub r: f64,
    pub xi: f64,
}

impl From<&PkBin> for PkBinOut {
    fn from(b: &PkBin) -> Self {
        Self { k: b.k, pk: b.pk, n_modes: b.n_modes as usize }
    }
}

impl From<&XiBin> for XiBinOut {
    fn from(b: &XiBin) -> Self {
        Self { r: b.r, xi: b.xi }
    }
}

#[derive(Debug, Serialize)]
pub struct PkRsdBinOut {
    pub k: f64,
    pub mu: f64,
    pub pk: f64,
    pub n_modes: u64,
}

#[derive(Debug, Serialize)]
pub struct PkMultipoleOut {
    pub k: f64,
    pub p0: f64,
    pub p2: f64,
    pub p4: f64,
}

impl From<&PkRsdBin> for PkRsdBinOut {
    fn from(b: &PkRsdBin) -> Self {
        Self { k: b.k, mu: b.mu, pk: b.pk, n_modes: b.n_modes }
    }
}

impl From<&PkMultipoleBin> for PkMultipoleOut {
    fn from(b: &PkMultipoleBin) -> Self {
        Self { k: b.k, p0: b.p0, p2: b.p2, p4: b.p4 }
    }
}

/// Ejecuta el análisis in-situ si el paso actual cumple el intervalo configurado.
///
/// Escribe `<out_dir>/insitu_{step:06}.json` con P(k), recuento de halos y ξ(r).
///
/// # Retorna
/// `true` si se ejecutó el análisis, `false` si el paso no estaba en el intervalo.
pub fn maybe_run_insitu(
    particles: &[Particle],
    cfg: &InsituAnalysisSection,
    box_size: f64,
    a: f64,
    step: u64,
    default_out_dir: &Path,
) -> bool {
    if !cfg.enabled || cfg.interval == 0 || step % cfg.interval != 0 {
        return false;
    }

    let out_dir = cfg
        .output_dir
        .as_deref()
        .unwrap_or(default_out_dir);

    if let Err(e) = std::fs::create_dir_all(out_dir) {
        eprintln!("[insitu] Error creando directorio {:?}: {e}", out_dir);
        return false;
    }

    let params = AnalysisParams {
        box_size,
        b: cfg.fof_b,
        min_particles: cfg.fof_min_part,
        pk_mesh: cfg.pk_mesh,
        ..Default::default()
    };

    match analyse(particles, &params) {
        result => {
            let n_halos = result.halos.len();
            let m_total_halos: f64 = result.halos.iter().map(|h| h.mass).sum();

            let xi_r: Vec<XiBinOut> = if cfg.xi_bins > 0 {
                two_point_correlation_fft(&result.power_spectrum, box_size, cfg.xi_bins)
                    .iter()
                    .map(XiBinOut::from)
                    .collect()
            } else {
                Vec::new()
            };

            // Phase 75: P(k,μ) en espacio de redshift
            let (pk_rsd_out, pk_multipoles_out) = if cfg.pk_rsd_bins > 0 {
                let positions: Vec<_> = particles.iter().map(|p| p.position).collect();
                let velocities: Vec<_> = particles.iter().map(|p| p.velocity).collect();
                let masses: Vec<f64> = particles.iter().map(|p| p.mass).collect();
                let h_a = 100.0; // H(a) en km/s/Mpc — aproximación para in-situ
                let rsd_params = PkRsdParams {
                    n_k_bins: cfg.pk_mesh / 2,
                    n_mu_bins: cfg.pk_rsd_bins,
                    los: LosAxis::Z,
                    scale_factor: a,
                    hubble_a: h_a,
                };
                let rsd_bins = compute_pk_multipoles(
                    &positions, &velocities, &masses, box_size, cfg.pk_mesh, &rsd_params,
                );
                let pk_rsd_raw = gadget_ng_analysis::pk_redshift_space(
                    &positions, &velocities, &masses, box_size, cfg.pk_mesh, &rsd_params,
                );
                let rsd_out: Vec<PkRsdBinOut> = pk_rsd_raw.iter().map(PkRsdBinOut::from).collect();
                let mult_out: Vec<PkMultipoleOut> = rsd_bins.iter().map(PkMultipoleOut::from).collect();
                (rsd_out, mult_out)
            } else {
                (Vec::new(), Vec::new())
            };

            let z = if a > 0.0 { 1.0 / a - 1.0 } else { f64::INFINITY };

            let insitu = InsituResult {
                step,
                a,
                z,
                n_halos,
                m_total_halos,
                power_spectrum: result.power_spectrum.iter().map(PkBinOut::from).collect(),
                xi_r,
                pk_rsd: pk_rsd_out,
                pk_multipoles: pk_multipoles_out,
            };

            let path = out_dir.join(format!("insitu_{step:06}.json"));
            match serde_json::to_string_pretty(&insitu) {
                Ok(json) => {
                    if let Err(e) = std::fs::write(&path, json) {
                        eprintln!("[insitu] Error escribiendo {:?}: {e}", path);
                    }
                }
                Err(e) => eprintln!("[insitu] Error serializando: {e}"),
            }
        }
    }
    true
}
