//! Catálogo de halos HDF5 por snapshot (Phase 77).
//!
//! Escribe (y lee) un catálogo de halos FoF + subhalos SUBFIND en formato HDF5
//! compatible con `yt`, `h5py`, `Caesar` y `rockstar-galaxies`.
//!
//! ## Estructura del archivo
//!
//! ```text
//! halos_NNNNNN.hdf5
//!   /Header
//!     z                 (f64) — corrimiento al rojo
//!     a                 (f64) — factor de escala
//!     N_halos           (i64) — número de halos FoF
//!     N_subhalos        (i64) — número de subhalos (0 si SUBFIND no activo)
//!     BoxSize           (f64) — tamaño de la caja
//!   /Halos/
//!     Mass              [N f64] — masa FoF
//!     Pos               [N×3 f64] — posición del centro de masa
//!     Vel               [N×3 f64] — velocidad del centro de masa
//!     R200              [N f64] — radio virial R₂₀₀
//!     Spin_Peebles      [N f64] — parámetro de spin λ (0 si no disponible)
//!     Npart             [N i64] — número de partículas
//!   /Subhalos/           (opcional, si hay subhalos)
//!     Mass              [M f64]
//!     Pos               [M×3 f64]
//!     ParentHalo        [M i64] — índice del halo FoF padre
//! ```
//!
//! ## Uso sin feature hdf5
//!
//! Si el crate no está compilado con `--features hdf5`, las funciones de
//! escritura/lectura retornan `Err(SnapshotError::UnsupportedFormat(...))`.
//! La struct `HaloCatalogEntry` y `HaloCatalogHeader` siempre están disponibles.

use crate::error::SnapshotError;

/// Entrada de un halo en el catálogo.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HaloCatalogEntry {
    /// Masa del halo (unidades internas).
    pub mass: f64,
    /// Posición del centro de masa [x, y, z].
    pub pos: [f64; 3],
    /// Velocidad del centro de masa [vx, vy, vz].
    pub vel: [f64; 3],
    /// Radio virial R₂₀₀.
    pub r200: f64,
    /// Parámetro de spin λ (Peebles). 0.0 si no calculado.
    pub spin_peebles: f64,
    /// Número de partículas del halo.
    pub npart: i64,
}

/// Entrada de un subhalo (SUBFIND).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SubhaloCatalogEntry {
    /// Masa del subhalo.
    pub mass: f64,
    /// Posición del centro del subhalo.
    pub pos: [f64; 3],
    /// Índice del halo FoF padre.
    pub parent_halo: i64,
}

/// Metadatos del catálogo de halos.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HaloCatalogHeader {
    /// Corrimiento al rojo z.
    pub redshift: f64,
    /// Factor de escala a.
    pub scale_factor: f64,
    /// Tamaño de la caja.
    pub box_size: f64,
    /// Número de halos FoF.
    pub n_halos: i64,
    /// Número de subhalos.
    pub n_subhalos: i64,
}

impl HaloCatalogHeader {
    pub fn new(redshift: f64, box_size: f64, n_halos: usize, n_subhalos: usize) -> Self {
        let scale_factor = if redshift > -1.0 { 1.0 / (1.0 + redshift) } else { 1.0 };
        Self {
            redshift,
            scale_factor,
            box_size,
            n_halos: n_halos as i64,
            n_subhalos: n_subhalos as i64,
        }
    }
}

// ── Serialización JSONL (siempre disponible) ──────────────────────────────

/// Escribe el catálogo de halos en formato JSONL (una línea por halo).
pub fn write_halo_catalog_jsonl(
    path: &std::path::Path,
    header: &HaloCatalogHeader,
    halos: &[HaloCatalogEntry],
) -> Result<(), SnapshotError> {
    use std::io::Write;
    let mut f = std::fs::File::create(path)?;
    writeln!(f, "{}", serde_json::to_string(header)?)?;
    for h in halos {
        writeln!(f, "{}", serde_json::to_string(h)?)?;
    }
    Ok(())
}

/// Lee el catálogo de halos desde JSONL.
#[allow(clippy::filter_map_bool_then)]
pub fn read_halo_catalog_jsonl(
    path: &std::path::Path,
) -> Result<(HaloCatalogHeader, Vec<HaloCatalogEntry>), SnapshotError> {
    use std::io::BufRead;
    let file = std::fs::File::open(path)?;
    let mut lines = std::io::BufReader::new(file).lines();
    let header_line = lines
        .next()
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "sin header"))??;
    let header: HaloCatalogHeader = serde_json::from_str(&header_line)?;
    let halos: Vec<HaloCatalogEntry> = lines
        .map_while(Result::ok)
        .filter_map(|l| serde_json::from_str(&l).ok())
        .collect();
    Ok((header, halos))
}

// ── HDF5 (feature gate) ───────────────────────────────────────────────────

#[cfg(feature = "hdf5")]
pub use hdf5_impl::{read_halo_catalog_hdf5, write_halo_catalog_hdf5};

#[cfg(not(feature = "hdf5"))]
pub fn write_halo_catalog_hdf5(
    _path: &std::path::Path,
    _header: &HaloCatalogHeader,
    _halos: &[HaloCatalogEntry],
    _subhalos: &[SubhaloCatalogEntry],
) -> Result<(), SnapshotError> {
    Err(SnapshotError::UnsupportedFormat(
        "hdf5 (recompilar con --features hdf5)".into(),
    ))
}

#[cfg(not(feature = "hdf5"))]
pub fn read_halo_catalog_hdf5(
    _path: &std::path::Path,
) -> Result<(HaloCatalogHeader, Vec<HaloCatalogEntry>, Vec<SubhaloCatalogEntry>), SnapshotError> {
    Err(SnapshotError::UnsupportedFormat(
        "hdf5 (recompilar con --features hdf5)".into(),
    ))
}

#[cfg(feature = "hdf5")]
mod hdf5_impl {
    use super::*;
    use ndarray::{Array1, Array2};

    /// Escribe el catálogo de halos en formato HDF5 estilo Caesar/yt.
    pub fn write_halo_catalog_hdf5(
        path: &std::path::Path,
        header: &HaloCatalogHeader,
        halos: &[HaloCatalogEntry],
        subhalos: &[SubhaloCatalogEntry],
    ) -> Result<(), SnapshotError> {
        let file = hdf5::File::create(path)?;

        // ── /Header ──────────────────────────────────────────────────────
        let hdr = file.create_group("Header")?;
        hdr.new_attr_builder().with_data(&ndarray::arr1(&[header.redshift])).create("z")?;
        hdr.new_attr_builder().with_data(&ndarray::arr1(&[header.scale_factor])).create("a")?;
        hdr.new_attr_builder().with_data(&ndarray::arr1(&[header.box_size])).create("BoxSize")?;
        hdr.new_attr_builder().with_data(&ndarray::arr1(&[header.n_halos])).create("N_halos")?;
        hdr.new_attr_builder().with_data(&ndarray::arr1(&[header.n_subhalos])).create("N_subhalos")?;

        // ── /Halos/ ───────────────────────────────────────────────────────
        let n = halos.len();
        if n > 0 {
            let hg = file.create_group("Halos")?;

            let mass = Array1::from_vec(halos.iter().map(|h| h.mass).collect::<Vec<_>>());
            let mut pos = Array2::<f64>::zeros((n, 3));
            let mut vel = Array2::<f64>::zeros((n, 3));
            let r200 = Array1::from_vec(halos.iter().map(|h| h.r200).collect::<Vec<_>>());
            let spin = Array1::from_vec(halos.iter().map(|h| h.spin_peebles).collect::<Vec<_>>());
            let npart = Array1::from_vec(halos.iter().map(|h| h.npart).collect::<Vec<_>>());

            for (i, h) in halos.iter().enumerate() {
                pos[[i, 0]] = h.pos[0];
                pos[[i, 1]] = h.pos[1];
                pos[[i, 2]] = h.pos[2];
                vel[[i, 0]] = h.vel[0];
                vel[[i, 1]] = h.vel[1];
                vel[[i, 2]] = h.vel[2];
            }

            hg.new_dataset_builder().with_data(&mass).create("Mass")?;
            hg.new_dataset_builder().with_data(&pos).create("Pos")?;
            hg.new_dataset_builder().with_data(&vel).create("Vel")?;
            hg.new_dataset_builder().with_data(&r200).create("R200")?;
            hg.new_dataset_builder().with_data(&spin).create("Spin_Peebles")?;
            hg.new_dataset_builder().with_data(&npart).create("Npart")?;
        }

        // ── /Subhalos/ ────────────────────────────────────────────────────
        let m = subhalos.len();
        if m > 0 {
            let sg = file.create_group("Subhalos")?;
            let smass = Array1::from_vec(subhalos.iter().map(|s| s.mass).collect::<Vec<_>>());
            let mut spos = Array2::<f64>::zeros((m, 3));
            let sparent = Array1::from_vec(subhalos.iter().map(|s| s.parent_halo).collect::<Vec<_>>());
            for (i, s) in subhalos.iter().enumerate() {
                spos[[i, 0]] = s.pos[0];
                spos[[i, 1]] = s.pos[1];
                spos[[i, 2]] = s.pos[2];
            }
            sg.new_dataset_builder().with_data(&smass).create("Mass")?;
            sg.new_dataset_builder().with_data(&spos).create("Pos")?;
            sg.new_dataset_builder().with_data(&sparent).create("ParentHalo")?;
        }

        Ok(())
    }

    /// Lee el catálogo de halos desde HDF5.
    pub fn read_halo_catalog_hdf5(
        path: &std::path::Path,
    ) -> Result<(HaloCatalogHeader, Vec<HaloCatalogEntry>, Vec<SubhaloCatalogEntry>), SnapshotError> {
        let file = hdf5::File::open(path)?;

        // Header
        let hdr = file.group("Header")?;
        let z: Vec<f64> = hdr.attr("z")?.read_raw()?;
        let a: Vec<f64> = hdr.attr("a")?.read_raw()?;
        let bs: Vec<f64> = hdr.attr("BoxSize")?.read_raw()?;
        let nh: Vec<i64> = hdr.attr("N_halos")?.read_raw()?;
        let ns: Vec<i64> = hdr.attr("N_subhalos")?.read_raw()?;
        let header = HaloCatalogHeader {
            redshift: z[0], scale_factor: a[0], box_size: bs[0],
            n_halos: nh[0], n_subhalos: ns[0],
        };

        // Halos
        let halos = if let Ok(hg) = file.group("Halos") {
            let mass: Vec<f64> = hg.dataset("Mass")?.read_raw()?;
            let pos_flat: Vec<f64> = hg.dataset("Pos")?.read_raw()?;
            let vel_flat: Vec<f64> = hg.dataset("Vel")?.read_raw()?;
            let r200: Vec<f64> = hg.dataset("R200")?.read_raw()?;
            let spin: Vec<f64> = hg.dataset("Spin_Peebles")?.read_raw()?;
            let npart: Vec<i64> = hg.dataset("Npart")?.read_raw()?;
            (0..mass.len()).map(|i| HaloCatalogEntry {
                mass: mass[i],
                pos: [pos_flat[i * 3], pos_flat[i * 3 + 1], pos_flat[i * 3 + 2]],
                vel: [vel_flat[i * 3], vel_flat[i * 3 + 1], vel_flat[i * 3 + 2]],
                r200: r200[i],
                spin_peebles: spin[i],
                npart: npart[i],
            }).collect()
        } else {
            Vec::new()
        };

        // Subhalos
        let subhalos = if let Ok(sg) = file.group("Subhalos") {
            let smass: Vec<f64> = sg.dataset("Mass")?.read_raw()?;
            let spos_flat: Vec<f64> = sg.dataset("Pos")?.read_raw()?;
            let sparent: Vec<i64> = sg.dataset("ParentHalo")?.read_raw()?;
            (0..smass.len()).map(|i| SubhaloCatalogEntry {
                mass: smass[i],
                pos: [spos_flat[i * 3], spos_flat[i * 3 + 1], spos_flat[i * 3 + 2]],
                parent_halo: sparent[i],
            }).collect()
        } else {
            Vec::new()
        };

        Ok((header, halos, subhalos))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn sample_halos(n: usize) -> Vec<HaloCatalogEntry> {
        (0..n).map(|i| HaloCatalogEntry {
            mass: 1e12 * (i + 1) as f64,
            pos: [i as f64 * 0.1, 0.0, 0.0],
            vel: [0.0, i as f64 * 10.0, 0.0],
            r200: 0.5 * (i + 1) as f64,
            spin_peebles: 0.03 + i as f64 * 0.001,
            npart: 100 * (i + 1) as i64,
        }).collect()
    }

    #[test]
    fn halo_catalog_header_creation() {
        let h = HaloCatalogHeader::new(1.0, 100.0, 50, 10);
        assert_eq!(h.n_halos, 50);
        assert_eq!(h.n_subhalos, 10);
        assert!((h.scale_factor - 0.5).abs() < 1e-10, "a = 1/(1+z) = 0.5");
    }

    #[test]
    fn jsonl_roundtrip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("halos.jsonl");
        let header = HaloCatalogHeader::new(0.5, 100.0, 3, 0);
        let halos = sample_halos(3);

        write_halo_catalog_jsonl(&path, &header, &halos).unwrap();
        let (h2, halos2) = read_halo_catalog_jsonl(&path).unwrap();

        assert_eq!(h2.n_halos, 3);
        assert_eq!(halos2.len(), 3);
        assert!((halos2[0].mass - halos[0].mass).abs() < 1e-6);
        assert!((halos2[2].spin_peebles - halos[2].spin_peebles).abs() < 1e-10);
    }

    #[test]
    fn jsonl_empty_halos() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("empty.jsonl");
        let header = HaloCatalogHeader::new(0.0, 50.0, 0, 0);
        write_halo_catalog_jsonl(&path, &header, &[]).unwrap();
        let (h2, halos2) = read_halo_catalog_jsonl(&path).unwrap();
        assert_eq!(h2.n_halos, 0);
        assert_eq!(halos2.len(), 0);
    }

    #[test]
    fn halo_entry_serializes() {
        let e = HaloCatalogEntry {
            mass: 1e12, pos: [1.0, 2.0, 3.0], vel: [10.0, 0.0, -5.0],
            r200: 0.5, spin_peebles: 0.04, npart: 200,
        };
        let s = serde_json::to_string(&e).unwrap();
        let e2: HaloCatalogEntry = serde_json::from_str(&s).unwrap();
        assert!((e2.mass - 1e12).abs() < 1.0);
        assert_eq!(e2.npart, 200);
    }

    #[cfg(all(test, feature = "hdf5"))]
    mod hdf5_tests {
        use super::*;
        use tempfile::tempdir;

        #[test]
        fn hdf5_roundtrip() {
            let dir = tempdir().unwrap();
            let path = dir.path().join("halos.hdf5");
            let header = HaloCatalogHeader::new(1.0, 100.0, 4, 2);
            let halos = sample_halos(4);
            let subhalos = vec![
                SubhaloCatalogEntry { mass: 1e11, pos: [0.1, 0.0, 0.0], parent_halo: 0 },
                SubhaloCatalogEntry { mass: 5e10, pos: [0.5, 0.0, 0.0], parent_halo: 1 },
            ];

            write_halo_catalog_hdf5(&path, &header, &halos, &subhalos).unwrap();
            let (h2, halos2, subs2) = read_halo_catalog_hdf5(&path).unwrap();

            assert_eq!(halos2.len(), 4);
            assert_eq!(subs2.len(), 2);
            assert!((halos2[0].mass - halos[0].mass).abs() < 1.0);
            assert_eq!(subs2[0].parent_halo, 0);
            assert!((h2.redshift - 1.0).abs() < 1e-10);
        }
    }
}
