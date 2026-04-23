//! Subcomando `gadget-ng mah` — Historia de Acreción de Masa (Phase 67).

use crate::error::CliError;
use gadget_ng_analysis::{mah_main_branch, mah_mcbride2009, MergerForest};
use serde::Serialize;
use std::fs;
use std::path::Path;

/// Salida del subcomando `mah` en formato JSON.
#[derive(Serialize)]
struct MahOutput {
    halo_id: u64,
    mah: Vec<MahPoint>,
    mcbride_fit: Vec<MahPoint>,
}

#[derive(Serialize)]
struct MahPoint {
    snapshot: usize,
    redshift: f64,
    mass: f64,
}

/// Ejecuta el subcomando `mah`:
/// 1. Carga el merger tree JSON.
/// 2. Extrae la MAH del halo `root_id` a lo largo de la rama principal.
/// 3. Calcula el ajuste analítico McBride+2009.
/// 4. Escribe `mah.json`.
pub fn run_mah(
    merger_tree_path: &Path,
    redshifts: &[f64],
    root_id: u64,
    alpha: f64,
    beta: f64,
    out_path: &Path,
) -> Result<(), CliError> {
    // Cargar el merger tree.
    let json_str = fs::read_to_string(merger_tree_path)
        .map_err(|e| CliError::io(merger_tree_path, e))?;
    let forest: MergerForest =
        serde_json::from_str(&json_str)?;

    // Extraer MAH.
    let mah = mah_main_branch(&forest, root_id, redshifts);

    // Calcular m0 = masa en z=0 (último punto de la MAH = más reciente).
    let m0 = mah.masses.first().copied().unwrap_or(0.0);

    // Construir puntos de la MAH.
    let mah_points: Vec<MahPoint> = mah
        .snapshots
        .iter()
        .zip(mah.redshifts.iter())
        .zip(mah.masses.iter())
        .map(|((&snap, &z), &mass)| MahPoint {
            snapshot: snap,
            redshift: z,
            mass,
        })
        .collect();

    // Calcular fit McBride+2009 en los mismos redshifts.
    let fit_points: Vec<MahPoint> = mah
        .snapshots
        .iter()
        .zip(mah.redshifts.iter())
        .map(|(&snap, &z)| MahPoint {
            snapshot: snap,
            redshift: z,
            mass: mah_mcbride2009(m0, z, alpha, beta),
        })
        .collect();

    let output = MahOutput {
        halo_id: root_id,
        mah: mah_points,
        mcbride_fit: fit_points,
    };

    // Escribir JSON.
    if let Some(parent) = out_path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).map_err(|e| CliError::io(out_path, e))?;
        }
    }
    let json = serde_json::to_string_pretty(&output)?;
    fs::write(out_path, json).map_err(|e| CliError::io(out_path, e))?;

    eprintln!(
        "MAH de halo {} escrita en {} ({} puntos)",
        root_id,
        out_path.display(),
        output.mah.len()
    );
    Ok(())
}
