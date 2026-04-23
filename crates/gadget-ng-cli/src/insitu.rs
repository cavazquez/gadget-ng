//! Análisis in-situ durante el loop `stepping` (Phase 63).
//!
//! Calcula P(k), FoF y opcionalmente ξ(r) en intervalos configurables,
//! escribiendo `insitu_NNNNNN.json` en `<output_dir>/`.

use gadget_ng_analysis::{
    analyse, two_point_correlation_fft, AnalysisParams, XiBin,
};
use gadget_ng_core::{InsituAnalysisSection, Particle};
use gadget_ng_analysis::PkBin;
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

            let z = if a > 0.0 { 1.0 / a - 1.0 } else { f64::INFINITY };

            let insitu = InsituResult {
                step,
                a,
                z,
                n_halos,
                m_total_halos,
                power_spectrum: result.power_spectrum.iter().map(PkBinOut::from).collect(),
                xi_r,
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
