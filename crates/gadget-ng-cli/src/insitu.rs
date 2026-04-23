//! Análisis in-situ durante el loop `stepping` (Phase 63).
//!
//! Calcula P(k), FoF y opcionalmente ξ(r) en intervalos configurables,
//! escribiendo `insitu_NNNNNN.json` en `<output_dir>/`.

use gadget_ng_analysis::{
    analyse, bispectrum_equilateral, compute_assembly_bias, compute_pk_multipoles,
    two_point_correlation_fft, AnalysisParams, AssemblyBiasParams, AssemblyBiasResult, BkBin,
    LosAxis, PkRsdParams, XiBin,
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
    /// Bispectrum equilateral B(k). Vacío si `bispectrum_bins == 0`.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub bk_equilateral: Vec<BkBinOut>,
    /// Assembly bias (Spearman spin/concentración vs entorno). None si no activado.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub assembly_bias: Option<AssemblyBiasOut>,
    /// Perfil de temperatura del IGM T(z). None si `igm_temp_enabled == false`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub igm_temp: Option<gadget_ng_rt::IgmTempBin>,
}

/// Salida de un bin del bispectrum.
#[derive(Debug, Serialize)]
pub struct BkBinOut {
    pub k: f64,
    pub bk: f64,
    pub n_triangles: u64,
}

impl From<&BkBin> for BkBinOut {
    fn from(b: &BkBin) -> Self {
        Self { k: b.k, bk: b.bk, n_triangles: b.n_triangles }
    }
}

/// Salida del assembly bias.
#[derive(Debug, Serialize)]
pub struct AssemblyBiasOut {
    pub smooth_radius: f64,
    pub spearman_lambda: f64,
    pub spearman_concentration: f64,
    pub n_halos: usize,
}

impl From<&AssemblyBiasResult> for AssemblyBiasOut {
    fn from(r: &AssemblyBiasResult) -> Self {
        Self {
            smooth_radius: r.smooth_radius,
            spearman_lambda: r.spearman_lambda,
            spearman_concentration: r.spearman_concentration,
            n_halos: r.n_halos,
        }
    }
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

            // Phase 82c: Bispectrum equilateral B(k)
            let bk_equilateral_out: Vec<BkBinOut> = if cfg.bispectrum_bins > 0 {
                let positions: Vec<_> = particles.iter().map(|p| p.position).collect();
                let masses: Vec<f64> = particles.iter().map(|p| p.mass).collect();
                bispectrum_equilateral(&positions, &masses, box_size, cfg.pk_mesh, cfg.bispectrum_bins)
                    .iter()
                    .map(BkBinOut::from)
                    .collect()
            } else {
                Vec::new()
            };

            // Phase 82c: Assembly bias
            let assembly_bias_out: Option<AssemblyBiasOut> = if cfg.assembly_bias_enabled
                && result.halos.len() >= 4
            {
                let all_pos: Vec<_> = particles.iter().map(|p| p.position).collect();
                let all_mass: Vec<f64> = particles.iter().map(|p| p.mass).collect();

                let halo_pos: Vec<_> = result
                    .halos
                    .iter()
                    .map(|h| gadget_ng_core::Vec3::new(h.x_com, h.y_com, h.z_com))
                    .collect();
                let halo_mass: Vec<f64> = result.halos.iter().map(|h| h.mass).collect();

                // In-situ: usamos dispersión de velocidades como proxy de spin (sin membership).
                // El spin real requiere partículas miembro; se usa 0 como placeholder.
                let spins: Vec<f64> = result
                    .halos
                    .iter()
                    .map(|h| h.velocity_dispersion)
                    .collect();
                let concentrations: Vec<f64> = vec![0.0; result.halos.len()];

                let ab_params = AssemblyBiasParams {
                    smooth_radius: cfg.assembly_bias_smooth_r,
                    mesh: cfg.pk_mesh,
                    n_quartiles: 4,
                };
                let ab = compute_assembly_bias(
                    &halo_pos, &halo_mass, &spins, &concentrations,
                    &all_pos, &all_mass, box_size, &ab_params,
                );
                Some(AssemblyBiasOut::from(&ab))
            } else {
                None
            };

            let z = if a > 0.0 { 1.0 / a - 1.0 } else { f64::INFINITY };

            // Perfil de temperatura IGM (Phase 90)
            let igm_temp_out = if cfg.igm_temp_enabled {
                let gas_particles: Vec<&Particle> = particles
                    .iter()
                    .filter(|p| p.internal_energy > 0.0)
                    .collect();
                if gas_particles.is_empty() {
                    None
                } else {
                    let chem_neutral = vec![
                        gadget_ng_rt::ChemState::neutral();
                        gas_particles.len()
                    ];
                    let gas_owned: Vec<Particle> = gas_particles.iter().map(|p| (*p).clone()).collect();
                    let params = gadget_ng_rt::IgmTempParams::default();
                    Some(gadget_ng_rt::compute_igm_temp_profile(&gas_owned, &chem_neutral, 0.0, z, &params))
                }
            } else {
                None
            };

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
                bk_equilateral: bk_equilateral_out,
                assembly_bias: assembly_bias_out,
                igm_temp: igm_temp_out,
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
