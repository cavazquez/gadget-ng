//! Subcomando `gadget-ng analyze` — análisis completo de snapshots.
//!
//! Calcula, para un directorio de snapshot:
//! - Friends-of-Friends (FoF): catálogo de halos con masa, posición, velocidad.
//! - Espectro de potencia P(k) via CIC + FFT 3D.
//! - Función de correlación de 2 puntos ξ(r) via transformada de Hankel de P(k).
//! - Concentración c(M) para halos con N_part ≥ 50 usando ajuste NFW.
//! - Escribe `results.json` con todos los resultados en formato estructurado.

use crate::error::CliError;
use gadget_ng_analysis::{
    analyse, concentration_duffy2008, concentration_ludlow2016, find_subhalos,
    fit_nfw_concentration, measure_density_profile, r200_from_m200, rho_crit_z,
    two_point_correlation_fft, AnalysisParams, SubfindParams, RHO_CRIT_H2,
};
use gadget_ng_core::{SnapshotFormat, Vec3};
use gadget_ng_io::{write_halo_catalog_jsonl, HaloCatalogEntry, HaloCatalogHeader};
#[cfg(feature = "hdf5")]
use gadget_ng_io::SubhaloCatalogEntry;
use std::fs;
use std::path::Path;

/// Parámetros del comando `analyze`.
pub struct AnalyzeParams<'a> {
    pub snapshot_dir: &'a Path,
    pub out_path: &'a Path,
    /// Parámetro de enlace FoF (fracción de separación media).
    pub fof_b: f64,
    /// Número mínimo de partículas para halo FoF.
    pub min_particles: usize,
    /// Tamaño del grid PM para P(k).
    pub pk_mesh: usize,
    /// Número de bins para ξ(r).
    pub xi_bins: usize,
    /// Número mínimo de partículas para ajuste NFW.
    pub nfw_min_part: usize,
    /// Parámetros cosmológicos para c(M): (Omega_m, Omega_L, redshift).
    pub cosmology: Option<(f64, f64, f64)>,
    /// Tamaño físico de la caja en Mpc/h (para unidades de c(M)).
    pub box_size_mpc_h: Option<f64>,
    /// Ejecutar SUBFIND sobre cada halo (Phase 68). Desactivado por defecto.
    pub subfind: bool,
    /// Número mínimo de partículas de halo para correr SUBFIND.
    pub subfind_min_particles: usize,
    /// Escribir catálogo de halos en formato HDF5 (Phase 82d). Si no hay feature hdf5, usa JSONL.
    pub hdf5_catalog: bool,
}

impl<'a> Default for AnalyzeParams<'a> {
    fn default() -> Self {
        Self {
            snapshot_dir: Path::new("."),
            out_path: Path::new("results.json"),
            fof_b: 0.2,
            min_particles: 8,
            pk_mesh: 64,
            xi_bins: 20,
            nfw_min_part: 50,
            cosmology: None,
            box_size_mpc_h: None,
            subfind: false,
            subfind_min_particles: 50,
            hdf5_catalog: false,
        }
    }
}

/// Ejecuta el análisis completo y escribe `results.json`.
pub fn run_analyze(params: &AnalyzeParams<'_>) -> Result<(), CliError> {
    let data = gadget_ng_io::read_snapshot_formatted(SnapshotFormat::Jsonl, params.snapshot_dir)
        .map_err(CliError::Snapshot)?;
    let box_size = data.box_size;
    let n = data.particles.len();

    if n == 0 {
        eprintln!(
            "[analyze] ADVERTENCIA: snapshot vacío en {:?}",
            params.snapshot_dir
        );
        return Ok(());
    }

    eprintln!(
        "[analyze] {} partículas leídas desde {:?}",
        n, params.snapshot_dir
    );

    // Longitud de enlace física.
    let rho_bg = n as f64 / (box_size * box_size * box_size);
    let l_mean = rho_bg.cbrt().recip();
    let b = params.fof_b * l_mean;

    let analysis_params = AnalysisParams {
        box_size,
        b,
        min_particles: params.min_particles,
        rho_crit: rho_bg,
        pk_mesh: params.pk_mesh,
    };
    let result = analyse(&data.particles, &analysis_params);

    eprintln!(
        "[analyze] FoF: {} halos  P(k): {} bins",
        result.halos.len(),
        result.power_spectrum.len()
    );

    // ── ξ(r) via FFT ─────────────────────────────────────────────────────────
    let box_mpc_h = params.box_size_mpc_h.unwrap_or(box_size);
    let pk_phys: Vec<gadget_ng_analysis::PkBin> = result
        .power_spectrum
        .iter()
        .map(|b| gadget_ng_analysis::PkBin {
            k: b.k / box_mpc_h,
            pk: b.pk * box_mpc_h.powi(3),
            n_modes: b.n_modes,
        })
        .collect();
    let xi = two_point_correlation_fft(&pk_phys, box_mpc_h, params.xi_bins);
    eprintln!("[analyze] ξ(r): {} bins", xi.len());

    // ── c(M) para halos con N ≥ nfw_min_part ─────────────────────────────────
    let (omega_m, omega_l, z) = params.cosmology.unwrap_or((0.315, 0.685, 0.0));
    let rho_c_phys = rho_crit_z(omega_m, omega_l, z);
    let rho_crit_h2 = RHO_CRIT_H2;
    let m_part_msun_h = omega_m * rho_crit_h2 * box_mpc_h.powi(3) / n as f64;

    let mut cm_table: Vec<serde_json::Value> = Vec::new();

    for halo in &result.halos {
        if halo.n_particles < params.nfw_min_part {
            continue;
        }
        let m200 = halo.n_particles as f64 * m_part_msun_h;
        let r200_mpc_h = r200_from_m200(m200, rho_c_phys);
        let r200_code = r200_mpc_h / box_mpc_h;

        let (cx, cy, cz) = (halo.x_com, halo.y_com, halo.z_com);
        let radii_mpc_h: Vec<f64> = data
            .particles
            .iter()
            .filter_map(|p| {
                let mut dx = p.position.x - cx;
                let mut dy = p.position.y - cy;
                let mut dz = p.position.z - cz;
                // imagen mínima periódica (box normalizado [0,1])
                if dx > 0.5 {
                    dx -= 1.0;
                } else if dx < -0.5 {
                    dx += 1.0;
                }
                if dy > 0.5 {
                    dy -= 1.0;
                } else if dy < -0.5 {
                    dy += 1.0;
                }
                if dz > 0.5 {
                    dz -= 1.0;
                } else if dz < -0.5 {
                    dz += 1.0;
                }
                let r_code = (dx * dx + dy * dy + dz * dz).sqrt();
                if r_code < 2.0 * r200_code {
                    Some(r_code * box_mpc_h)
                } else {
                    None
                }
            })
            .collect();

        if radii_mpc_h.len() < params.nfw_min_part {
            continue;
        }

        let profile = measure_density_profile(
            &radii_mpc_h,
            m_part_msun_h,
            r200_mpc_h * 0.05,
            r200_mpc_h,
            12,
            None,
        );

        let c_measured = fit_nfw_concentration(&profile, m200, rho_c_phys, 1.0, 30.0, 60)
            .map(|f| r200_mpc_h / f.profile.r_s);

        cm_table.push(serde_json::json!({
            "halo_id": halo.halo_id,
            "n_particles": halo.n_particles,
            "m200_msun_h": m200,
            "r200_mpc_h": r200_mpc_h,
            "c_measured": c_measured,
            "c_duffy2008": concentration_duffy2008(m200, z),
            "c_ludlow2016": concentration_ludlow2016(m200, z),
        }));
    }
    eprintln!("[analyze] c(M): {} halos con ajuste NFW", cm_table.len());

    // ── SUBFIND (Phase 68) ────────────────────────────────────────────────────
    let mut subfind_results: Vec<serde_json::Value> = Vec::new();
    if params.subfind {
        let sfparams = SubfindParams {
            min_subhalo_particles: params.subfind_min_particles.max(5),
            ..Default::default()
        };
        for halo in &result.halos {
            if halo.n_particles < params.subfind_min_particles {
                continue;
            }
            // Recolectar partículas miembro del halo (las más cercanas al CoM dentro de r_vir).
            let cx = Vec3::new(halo.x_com, halo.y_com, halo.z_com);
            let r_vir = halo.r_vir.max(1e-10);
            let member_indices: Vec<usize> = data
                .particles
                .iter()
                .enumerate()
                .filter_map(|(idx, p)| {
                    let mut dx = p.position.x - halo.x_com;
                    let mut dy = p.position.y - halo.y_com;
                    let mut dz = p.position.z - halo.z_com;
                    let _ = cx;
                    let b2 = box_size * 0.5;
                    if dx > b2 { dx -= box_size; } else if dx < -b2 { dx += box_size; }
                    if dy > b2 { dy -= box_size; } else if dy < -b2 { dy += box_size; }
                    if dz > b2 { dz -= box_size; } else if dz < -b2 { dz += box_size; }
                    let r = (dx * dx + dy * dy + dz * dz).sqrt();
                    if r < 2.0 * r_vir { Some(idx) } else { None }
                })
                .collect();

            if member_indices.len() < params.subfind_min_particles {
                continue;
            }

            let pos: Vec<Vec3> = member_indices.iter().map(|&i| data.particles[i].position).collect();
            let vel: Vec<Vec3> = member_indices.iter().map(|&i| data.particles[i].velocity).collect();
            let mass: Vec<f64> = member_indices.iter().map(|&i| data.particles[i].mass).collect();

            let subhalos = find_subhalos(halo, &pos, &vel, &mass, &sfparams);
            if !subhalos.is_empty() {
                subfind_results.push(serde_json::json!({
                    "halo_id": halo.halo_id,
                    "n_subhalos": subhalos.len(),
                    "subhalos": subhalos.iter().map(|s| serde_json::json!({
                        "subhalo_id": s.subhalo_id,
                        "n_particles": s.n_particles,
                        "mass": s.mass,
                        "x_com": s.x_com,
                        "v_com": s.v_com,
                        "v_disp": s.v_disp,
                        "e_total": s.e_total,
                    })).collect::<Vec<_>>(),
                }));
            }
        }
        eprintln!("[analyze] SUBFIND: {} halos con subhalos", subfind_results.len());
    }

    // ── Serialización JSON ────────────────────────────────────────────────────
    let output = serde_json::json!({
        "snapshot": params.snapshot_dir.to_string_lossy(),
        "n_particles": n,
        "box_size": box_size,
        "box_size_mpc_h": box_mpc_h,
        "cosmology": {
            "omega_m": omega_m,
            "omega_lambda": omega_l,
            "z": z,
        },
        "halos": result.halos.iter().map(|h| serde_json::json!({
            "halo_id": h.halo_id,
            "n_particles": h.n_particles,
            "mass": h.mass,
            "x": h.x_com,
            "y": h.y_com,
            "z": h.z_com,
            "r_vir": h.r_vir,
        })).collect::<Vec<_>>(),
        "power_spectrum": result.power_spectrum.iter().map(|b| serde_json::json!({
            "k": b.k,
            "pk": b.pk,
            "n_modes": b.n_modes,
        })).collect::<Vec<_>>(),
        "xi_r": xi.iter().map(|b| serde_json::json!({
            "r": b.r,
            "xi": b.xi,
        })).collect::<Vec<_>>(),
        "concentration_mass": cm_table,
        "subfind": subfind_results,
    });

    if let Some(parent) = params.out_path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).map_err(|e| CliError::io(parent, e))?;
        }
    }
    let json_str = serde_json::to_string_pretty(&output)?;
    fs::write(params.out_path, &json_str).map_err(|e| CliError::io(params.out_path, e))?;

    // ── Phase 82d: Catálogo de halos HDF5 / JSONL ────────────────────────────
    if params.hdf5_catalog {
        let (omega_m_cat, _omega_l_cat, z_cat) = params.cosmology.unwrap_or((0.315, 0.685, 0.0));
        let rho_crit_h2_cat = RHO_CRIT_H2;
        let box_mpc_h_cat = params.box_size_mpc_h.unwrap_or(box_size);
        let m_part_cat = omega_m_cat * rho_crit_h2_cat * box_mpc_h_cat.powi(3) / n as f64;

        let halo_entries: Vec<HaloCatalogEntry> = result
            .halos
            .iter()
            .map(|h| HaloCatalogEntry {
                mass: h.n_particles as f64 * m_part_cat,
                pos: [h.x_com * box_mpc_h_cat, h.y_com * box_mpc_h_cat, h.z_com * box_mpc_h_cat],
                vel: [0.0, 0.0, 0.0],
                r200: h.r_vir * box_mpc_h_cat,
                spin_peebles: 0.0,
                npart: h.n_particles as i64,
            })
            .collect();

        let header = HaloCatalogHeader::new(z_cat, box_mpc_h_cat, halo_entries.len(), 0);

        // Determinar path del catálogo junto al results.json
        let cat_dir = params
            .out_path
            .parent()
            .unwrap_or_else(|| std::path::Path::new("."));
        let jsonl_path = cat_dir.join("halos.jsonl");

        #[cfg(feature = "hdf5")]
        {
            let subhalo_entries: Vec<SubhaloCatalogEntry> = Vec::new();
            let hdf5_path = cat_dir.join("halos.hdf5");
            match gadget_ng_io::write_halo_catalog_hdf5(
                &hdf5_path,
                &header,
                &halo_entries,
                &subhalo_entries,
            ) {
                Ok(_) => eprintln!("[analyze] Catálogo HDF5 escrito en {:?}", hdf5_path),
                Err(e) => eprintln!("[analyze] Error escribiendo HDF5, usando JSONL: {e}"),
            }
        }
        #[cfg(not(feature = "hdf5"))]
        {
            match write_halo_catalog_jsonl(&jsonl_path, &header, &halo_entries) {
                Ok(_) => eprintln!("[analyze] Catálogo JSONL escrito en {:?}", jsonl_path),
                Err(e) => eprintln!("[analyze] Error escribiendo catálogo JSONL: {e}"),
            }
        }
    }

    println!(
        "[analyze] Resultados escritos en {:?} ({} halos, {} bins ξ(r), {} halos c(M))",
        params.out_path,
        result.halos.len(),
        xi.len(),
        cm_table.len()
    );
    Ok(())
}
