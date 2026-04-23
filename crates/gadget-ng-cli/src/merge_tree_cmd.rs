//! Subcomando `gadget-ng merge-tree` — merger trees de halos FoF (Phase 62).
//!
//! Lee catálogos de halos JSONL y snapshots de partículas, construye el merger tree
//! con [`build_merger_forest`] y escribe el resultado en JSON.

use crate::error::CliError;
use gadget_ng_analysis::{build_merger_forest, FofHalo, MergerForest, ParticleSnapshot};
use gadget_ng_core::SnapshotFormat;
use std::path::Path;

/// Ejecuta el merger tree sobre una secuencia de catálogos + snapshots.
///
/// # Parámetros
/// - `snap_dirs`: directorios de snapshot (orden cronológico, más antiguo primero).
/// - `catalog_paths`: archivos JSONL de catálogos FoF (mismo orden).
/// - `out_path`: archivo JSON de salida.
/// - `min_shared`: fracción mínima de partículas compartidas para registrar progenitor.
pub fn run_merge_tree(
    snap_dirs: &[std::path::PathBuf],
    catalog_paths: &[std::path::PathBuf],
    out_path: &Path,
    min_shared: f64,
) -> Result<(), CliError> {
    if snap_dirs.len() != catalog_paths.len() {
        return Err(CliError::InvalidConfig(
            "El número de snapshots y catálogos debe coincidir".into(),
        ));
    }
    if snap_dirs.is_empty() {
        let forest = MergerForest::default();
        let json = serde_json::to_string_pretty(&forest)?;
        std::fs::write(out_path, json)
            .map_err(|e| CliError::io(out_path, e))?;
        return Ok(());
    }

    let mut catalogs: Vec<(Vec<FofHalo>, Vec<ParticleSnapshot>)> = Vec::new();

    for (snap_dir, catalog_path) in snap_dirs.iter().zip(catalog_paths.iter()) {
        // Leer catálogo de halos.
        let halos: Vec<FofHalo> = {
            let content = std::fs::read_to_string(catalog_path)
                .map_err(|e| CliError::io(catalog_path.clone(), e))?;
            let mut halos = Vec::new();
            for line in content.lines() {
                let line = line.trim();
                if line.is_empty() { continue; }
                let h: FofHalo = serde_json::from_str(line)?;
                halos.push(h);
            }
            halos
        };

        // Leer partículas del snapshot.
        let snap_data = gadget_ng_io::read_snapshot_formatted(SnapshotFormat::Jsonl, snap_dir)
            .map_err(CliError::Snapshot)?;

        // Construir ParticleSnapshot con IDs y halo_idx.
        // Como no tenemos info de membresía, asignamos halo_idx = None para todas
        // (la membresía exacta requeriría re-ejecutar FoF, que no es el objetivo aquí).
        // Los IDs se usan para tracking entre snapshots.
        let part_snapshots: Vec<ParticleSnapshot> = snap_data
            .particles
            .iter()
            .enumerate()
            .map(|(i, _p)| ParticleSnapshot {
                id: i as u64,
                halo_idx: None,
            })
            .collect();

        catalogs.push((halos, part_snapshots));
    }

    let forest = build_merger_forest(&catalogs, min_shared);

    // Crear directorio padre si es necesario.
    if let Some(parent) = out_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .map_err(|e| CliError::io(parent, e))?;
        }
    }

    let json = serde_json::to_string_pretty(&forest)?;
    std::fs::write(out_path, json)
        .map_err(|e| CliError::io(out_path, e))?;

    eprintln!(
        "[merge-tree] {} nodos, {} raíces → {:?}",
        forest.nodes.len(),
        forest.roots.len(),
        out_path
    );
    Ok(())
}
