//! Subcomando `gadget-ng merge-tree` — merger trees de halos FoF (Phase 62).
//!
//! Lee catálogos de halos JSONL y snapshots de partículas, construye el merger tree
//! con [`build_merger_forest`] y escribe el resultado en JSON.
//!
//! ## Phase 107: membresía FoF real
//!
//! Anteriormente, todos los `ParticleSnapshot` tenían `halo_idx = None`, lo que
//! hacía que el algoritmo de matching por partículas no funcionara. Ahora se usa
//! [`particle_snapshots_from_catalog`] para asignar `halo_idx` real a cada
//! partícula mediante proximidad al centro de masa del halo más cercano dentro
//! de su radio de virial `r_vir`.

use crate::error::CliError;
use gadget_ng_analysis::{
    FofHalo, MergerForest, ParticleSnapshot, build_merger_forest, particle_snapshots_from_catalog,
};
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
        std::fs::write(out_path, json).map_err(|e| CliError::io(out_path, e))?;
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
                if line.is_empty() {
                    continue;
                }
                let h: FofHalo = serde_json::from_str(line)?;
                halos.push(h);
            }
            halos
        };

        // Leer partículas del snapshot.
        let snap_data = gadget_ng_io::read_snapshot_formatted(SnapshotFormat::Jsonl, snap_dir)
            .map_err(CliError::Snapshot)?;

        // Phase 107: construir ParticleSnapshot con halo_idx real mediante
        // asignación por proximidad al COM del halo dentro de r_vir.
        // Usar el box_size del snapshot; si no está disponible, inferir.
        let box_size = snap_data.box_size;
        let positions: Vec<gadget_ng_core::Vec3> =
            snap_data.particles.iter().map(|p| p.position).collect();
        let global_ids: Vec<u64> = snap_data
            .particles
            .iter()
            .map(|p| p.global_id as u64)
            .collect();

        let part_snapshots = if halos.is_empty() {
            // Sin halos: todas las partículas son campo
            global_ids
                .iter()
                .map(|&id| ParticleSnapshot { id, halo_idx: None })
                .collect()
        } else {
            particle_snapshots_from_catalog(&positions, &global_ids, &halos, box_size)
        };

        catalogs.push((halos, part_snapshots));
    }

    let forest = build_merger_forest(&catalogs, min_shared);

    // Crear directorio padre si es necesario.
    if let Some(parent) = out_path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent).map_err(|e| CliError::io(parent, e))?;
    }

    let json = serde_json::to_string_pretty(&forest)?;
    std::fs::write(out_path, json).map_err(|e| CliError::io(out_path, e))?;

    eprintln!(
        "[merge-tree] {} nodos, {} raíces → {:?}",
        forest.nodes.len(),
        forest.roots.len(),
        out_path
    );
    Ok(())
}
