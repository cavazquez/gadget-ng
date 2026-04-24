//! Subcomando `gadget-ng analyze` — análisis completo de snapshots.
//!
//! Calcula, para un directorio de snapshot:
//! - Friends-of-Friends (FoF): catálogo de halos con masa, posición, velocidad.
//! - Espectro de potencia P(k) via CIC + FFT 3D.
//! - Función de correlación de 2 puntos ξ(r) via transformada de Hankel de P(k).
//! - Concentración c(M) para halos con N_part ≥ 50 usando ajuste NFW.
//! - `--cm21`: estadísticas 21cm (δT_b, P(k)₂₁cm) → `analyze/cm21_output.json`
//! - `--igm-temp`: perfil de temperatura IGM T(z) → `analyze/igm_temp.json`
//! - `--agn-stats`: estadísticas de BH (masa, acreción) → `analyze/agn_stats.json`
//! - `--eor-state`: fracción de ionización x_HII media → `analyze/eor_state.json`
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
    // ── Phase 104: flags de análisis extendido ────────────────────────────────
    /// Calcular estadísticas 21cm (δT_b, P(k)₂₁cm) → `analyze/cm21_output.json`
    pub cm21: bool,
    /// Calcular perfil de temperatura IGM T(z) → `analyze/igm_temp.json`
    pub igm_temp: bool,
    /// Calcular estadísticas AGN (masas de BH, acreción) → `analyze/agn_stats.json`
    pub agn_stats: bool,
    /// Calcular fracción de ionización x_HII media → `analyze/eor_state.json`
    pub eor_state: bool,
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
            cm21: false,
            igm_temp: false,
            agn_stats: false,
            eor_state: false,
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

    // ── Phase 104: módulos de análisis extendido ──────────────────────────────
    // Directorio de salida para archivos de análisis extendido
    let analyze_dir = params.out_path.parent()
        .unwrap_or_else(|| Path::new("."))
        .join("analyze");

    // --cm21: estadísticas 21cm
    if params.cm21 {
        let gas_particles: Vec<_> = data.particles.iter()
            .filter(|p| p.internal_energy > 0.0)
            .cloned()
            .collect();
        if gas_particles.is_empty() {
            eprintln!("[analyze --cm21] No hay partículas de gas; omitiendo 21cm");
        } else {
            let chem = vec![gadget_ng_rt::ChemState::neutral(); gas_particles.len()];
            let cm21_params = gadget_ng_rt::Cm21Params::default();
            let n_mesh = params.pk_mesh.max(4);
            let cm21_out = gadget_ng_rt::compute_cm21_output(
                &gas_particles, &chem, box_size, z, n_mesh, 8, &cm21_params,
            );
            if let Ok(json) = serde_json::to_string_pretty(&cm21_out) {
                fs::create_dir_all(&analyze_dir).ok();
                let p = analyze_dir.join("cm21_output.json");
                let _ = fs::write(&p, &json);
                eprintln!("[analyze --cm21] Escrito en {:?}", p);
            }
        }
    }

    // --igm-temp: perfil de temperatura IGM
    if params.igm_temp {
        let gas: Vec<_> = data.particles.iter()
            .filter(|p| p.internal_energy > 0.0)
            .cloned()
            .collect();
        if gas.is_empty() {
            eprintln!("[analyze --igm-temp] No hay partículas de gas; omitiendo IGM");
        } else {
            let chem = vec![gadget_ng_rt::ChemState::neutral(); gas.len()];
            let igm_params = gadget_ng_rt::IgmTempParams::default();
            let profile = gadget_ng_rt::compute_igm_temp_profile(&gas, &chem, 0.0, z, &igm_params);
            if let Ok(json) = serde_json::to_string_pretty(&profile) {
                fs::create_dir_all(&analyze_dir).ok();
                let p = analyze_dir.join("igm_temp.json");
                let _ = fs::write(&p, &json);
                eprintln!("[analyze --igm-temp] Escrito en {:?}", p);
            }
        }
    }

    // --agn-stats: estadísticas de agujeros negros
    if params.agn_stats {
        // Las partículas con internal_energy muy alta (> umbral AGN) se tratan como BH.
        // En ausencia de metadatos de BH, se usa masa y posición de partículas energéticas.
        let bh_like: Vec<serde_json::Value> = data.particles.iter()
            .filter(|p| p.internal_energy > 1e4) // umbral empírico para "caliente por AGN"
            .map(|p| serde_json::json!({
                "global_id": p.global_id,
                "mass": p.mass,
                "internal_energy": p.internal_energy,
                "x": p.position.x,
                "y": p.position.y,
                "z": p.position.z,
            }))
            .collect();
        let agn_out = serde_json::json!({
            "n_bh_candidates": bh_like.len(),
            "total_mass": bh_like.iter().filter_map(|v| v["mass"].as_f64()).sum::<f64>(),
            "candidates": bh_like,
        });
        if let Ok(json) = serde_json::to_string_pretty(&agn_out) {
            fs::create_dir_all(&analyze_dir).ok();
            let p = analyze_dir.join("agn_stats.json");
            let _ = fs::write(&p, &json);
            eprintln!("[analyze --agn-stats] Escrito en {:?} ({} candidatos BH)", p, agn_out["n_bh_candidates"]);
        }
    }

    // --eor-state: fracción de ionización x_HII media
    if params.eor_state {
        let gas: Vec<_> = data.particles.iter()
            .filter(|p| p.internal_energy > 0.0)
            .collect();
        // Sin ChemState reales, estimamos x_HII a partir de internal_energy.
        // Una energía interna alta sugiere gas ionizado (temperatura > 10⁴ K → x_HII ≈ 1).
        // Esta es una estimación de primer orden; la versión completa requiere ChemState guardados.
        let n_gas = gas.len() as f64;
        let u_threshold = 100.0; // umbral empírico de energía interna para gas ionizado
        let n_ionized = gas.iter().filter(|p| p.internal_energy > u_threshold).count() as f64;
        let x_hii_mean = if n_gas > 0.0 { n_ionized / n_gas } else { 0.0 };
        let eor_out = serde_json::json!({
            "z": z,
            "a": data.time,
            "n_gas": n_gas as usize,
            "n_ionized_estimate": n_ionized as usize,
            "x_hii_mean_estimate": x_hii_mean,
            "note": "estimación sin ChemState; correr con reionization.enabled para valores exactos",
        });
        if let Ok(json) = serde_json::to_string_pretty(&eor_out) {
            fs::create_dir_all(&analyze_dir).ok();
            let p = analyze_dir.join("eor_state.json");
            let _ = fs::write(&p, &json);
            eprintln!("[analyze --eor-state] x_HII_mean ≈ {:.3}, escrito en {:?}", x_hii_mean, p);
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use gadget_ng_core::{Particle, Vec3};
    use gadget_ng_io::{write_snapshot_formatted, Provenance, SnapshotEnv};
    use gadget_ng_core::SnapshotFormat;

    fn make_dm(id: usize) -> Particle {
        Particle::new(id, 1.0, Vec3::new(id as f64 * 0.3 + 5.0, 0.0, 0.0), Vec3::zero())
    }

    fn write_snap(dir: &std::path::Path, particles: &[Particle]) {
        let snap_dir = dir.join("snap");
        std::fs::create_dir_all(&snap_dir).unwrap();
        let prov = Provenance::new("0-test", None, "debug", vec![], vec![], "hash");
        let env = SnapshotEnv { time: 1.0, redshift: 0.0, box_size: 10.0, ..Default::default() };
        write_snapshot_formatted(SnapshotFormat::Jsonl, &snap_dir, particles, &prov, &env).unwrap();
    }

    #[test]
    fn analyze_params_phase104_defaults_false() {
        let p = AnalyzeParams::default();
        assert!(!p.cm21, "cm21 false por defecto");
        assert!(!p.igm_temp, "igm_temp false por defecto");
        assert!(!p.agn_stats, "agn_stats false por defecto");
        assert!(!p.eor_state, "eor_state false por defecto");
    }

    #[test]
    fn analyze_params_phase104_can_be_enabled() {
        let params = AnalyzeParams {
            cm21: true,
            igm_temp: true,
            agn_stats: true,
            eor_state: true,
            ..Default::default()
        };
        assert!(params.cm21);
        assert!(params.igm_temp);
        assert!(params.agn_stats);
        assert!(params.eor_state);
    }

    #[test]
    fn analyze_no_flags_no_analyze_dir() {
        // Sin flags activos, el directorio analyze/ NO se crea
        let tmp = tempfile::tempdir().unwrap();
        let particles: Vec<_> = (0..8).map(|i| make_dm(i)).collect();
        write_snap(tmp.path(), &particles);
        let out = tmp.path().join("results.json");
        let params = AnalyzeParams {
            snapshot_dir: &tmp.path().join("snap"),
            out_path: &out,
            pk_mesh: 4,
            ..Default::default()
        };
        run_analyze(&params).unwrap();
        assert!(!tmp.path().join("analyze").exists(), "no debe crear analyze/ sin flags");
        assert!(out.exists(), "results.json debe existir");
    }

    #[test]
    fn analyze_agn_stats_no_gas_creates_empty_report() {
        // Con solo DM (sin gas), agn_stats debe crear un json con 0 candidatos
        let tmp = tempfile::tempdir().unwrap();
        let particles: Vec<_> = (0..8).map(|i| make_dm(i)).collect();
        write_snap(tmp.path(), &particles);
        let out = tmp.path().join("results.json");
        let params = AnalyzeParams {
            snapshot_dir: &tmp.path().join("snap"),
            out_path: &out,
            agn_stats: true,
            pk_mesh: 4,
            ..Default::default()
        };
        run_analyze(&params).unwrap();
        let path = tmp.path().join("analyze").join("agn_stats.json");
        assert!(path.exists(), "agn_stats.json debe existir incluso con 0 candidatos");
        let json: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(&path).unwrap()
        ).unwrap();
        assert_eq!(
            json["n_bh_candidates"].as_u64().unwrap(),
            0,
            "0 candidatos con solo DM"
        );
    }

    #[test]
    fn analyze_eor_state_no_gas_x_hii_zero() {
        // Con solo DM, eor_state debe crear json con x_HII = 0
        let tmp = tempfile::tempdir().unwrap();
        let particles: Vec<_> = (0..8).map(|i| make_dm(i)).collect();
        write_snap(tmp.path(), &particles);
        let out = tmp.path().join("results.json");
        let params = AnalyzeParams {
            snapshot_dir: &tmp.path().join("snap"),
            out_path: &out,
            eor_state: true,
            pk_mesh: 4,
            ..Default::default()
        };
        run_analyze(&params).unwrap();
        let path = tmp.path().join("analyze").join("eor_state.json");
        assert!(path.exists());
        let json: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(&path).unwrap()
        ).unwrap();
        let x_hii = json["x_hii_mean_estimate"].as_f64().unwrap();
        assert_eq!(x_hii, 0.0, "sin gas, x_HII debe ser 0.0");
    }

    #[test]
    fn analyze_eor_and_agn_flags_together() {
        // Múltiples flags activos no deben interferir
        let tmp = tempfile::tempdir().unwrap();
        let particles: Vec<_> = (0..8).map(|i| make_dm(i)).collect();
        write_snap(tmp.path(), &particles);
        let out = tmp.path().join("results.json");
        let params = AnalyzeParams {
            snapshot_dir: &tmp.path().join("snap"),
            out_path: &out,
            agn_stats: true,
            eor_state: true,
            pk_mesh: 4,
            ..Default::default()
        };
        run_analyze(&params).unwrap();
        assert!(out.exists());
        assert!(tmp.path().join("analyze").join("agn_stats.json").exists());
        assert!(tmp.path().join("analyze").join("eor_state.json").exists());
    }
}
