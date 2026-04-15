//! HDF5 al estilo GADGET-4 (grupos `Header`, `PartType1`, etc.) para interoperar con `yt` / `h5py`.
use std::path::Path;

use gadget_ng_core::{Particle, Vec3};
use ndarray::{arr1, Array1, Array2};

use crate::error::SnapshotError;
use crate::provenance::Provenance;
use crate::reader::{SnapshotData, SnapshotReader};
use crate::snapshot::{build_meta, write_sidecar_json};
use crate::writer::{SnapshotEnv, SnapshotWriter};

#[derive(Debug, Default, Clone, Copy)]
pub struct Hdf5Writer;

impl SnapshotWriter for Hdf5Writer {
    fn write(
        &self,
        out_dir: &Path,
        particles: &[Particle],
        provenance: &Provenance,
        env: &SnapshotEnv,
    ) -> Result<(), SnapshotError> {
        let meta = build_meta(particles.len(), provenance, env);
        write_sidecar_json(out_dir, &meta, provenance)?;

        let n = particles.len();
        let path = out_dir.join("snapshot.hdf5");
        let file = hdf5::File::create(&path)?;

        // --- Header (convención GADGET HDF5) ---
        let header = file.create_group("Header")?;
        let num_part_total: [i64; 6] = [0, n as i64, 0, 0, 0, 0];
        header
            .new_attr_builder()
            .with_data(&num_part_total)
            .create("NumPart_Total")?;
        header
            .new_attr_builder()
            .with_data(&num_part_total)
            .create("NumPart_ThisFile")?;
        let mass_table: [f64; 6] = [0.0; 6];
        header
            .new_attr_builder()
            .with_data(&mass_table)
            .create("MassTable")?;
        header
            .new_attr_builder()
            .with_data(&arr1(&[meta.time]))
            .create("Time")?;
        header
            .new_attr_builder()
            .with_data(&arr1(&[meta.redshift]))
            .create("Redshift")?;
        header
            .new_attr_builder()
            .with_data(&arr1(&[meta.box_size]))
            .create("BoxSize")?;
        let h_param = 1.0_f64;
        header
            .new_attr_builder()
            .with_data(&arr1(&[h_param]))
            .create("HubbleParam")?;
        let omega0 = 0.0_f64;
        header
            .new_attr_builder()
            .with_data(&arr1(&[omega0]))
            .create("Omega0")?;
        let omega_lambda = 0.0_f64;
        header
            .new_attr_builder()
            .with_data(&arr1(&[omega_lambda]))
            .create("OmegaLambda")?;

        // --- PartType1 = DM (N-body colisionless típico) ---
        let mut coords = Array2::<f64>::zeros((n, 3));
        let mut vels = Array2::<f64>::zeros((n, 3));
        let mut masses = Array1::<f64>::zeros(n);
        let mut ids = Array1::<i64>::zeros(n);
        for (i, p) in particles.iter().enumerate() {
            coords[[i, 0]] = p.position.x;
            coords[[i, 1]] = p.position.y;
            coords[[i, 2]] = p.position.z;
            vels[[i, 0]] = p.velocity.x;
            vels[[i, 1]] = p.velocity.y;
            vels[[i, 2]] = p.velocity.z;
            masses[i] = p.mass;
            ids[i] = p.global_id as i64;
        }

        let pt1 = file.create_group("PartType1")?;
        pt1.new_dataset_builder()
            .with_data(&coords)
            .create("Coordinates")?;
        pt1.new_dataset_builder()
            .with_data(&vels)
            .create("Velocities")?;
        pt1.new_dataset_builder()
            .with_data(&masses)
            .create("Masses")?;
        pt1.new_dataset_builder()
            .with_data(&ids)
            .create("ParticleIDs")?;

        // --- Provenance embebido ---
        let prov_g = file.create_group("Provenance")?;
        let json = serde_json::to_string(provenance)?;
        let json_bytes = Array1::from_vec(json.into_bytes());
        prov_g
            .new_dataset_builder()
            .with_data(&json_bytes)
            .create("gadget_ng_json_utf8")?;

        Ok(())
    }
}

/// Lector HDF5: reconstruye partículas desde `snapshot.hdf5` (grupos `Header` y `PartType1`).
#[derive(Debug, Default, Clone, Copy)]
pub struct Hdf5Reader;

impl SnapshotReader for Hdf5Reader {
    fn read(&self, dir: &Path) -> Result<SnapshotData, SnapshotError> {
        let file = hdf5::File::open(dir.join("snapshot.hdf5"))?;

        let header = file.group("Header")?;
        let time: Vec<f64> = header.attr("Time")?.read_raw()?;
        let redshift: Vec<f64> = header.attr("Redshift")?.read_raw()?;
        let box_size: Vec<f64> = header.attr("BoxSize")?.read_raw()?;

        let pt1 = file.group("PartType1")?;
        let coords: Vec<f64> = pt1.dataset("Coordinates")?.read_raw()?;
        let vels: Vec<f64> = pt1.dataset("Velocities")?.read_raw()?;
        let masses: Vec<f64> = pt1.dataset("Masses")?.read_raw()?;
        let ids: Vec<i64> = pt1.dataset("ParticleIDs")?.read_raw()?;

        let n = ids.len();
        let particles = (0..n)
            .map(|i| {
                Particle::new(
                    ids[i] as usize,
                    masses[i],
                    Vec3::new(coords[i * 3], coords[i * 3 + 1], coords[i * 3 + 2]),
                    Vec3::new(vels[i * 3], vels[i * 3 + 1], vels[i * 3 + 2]),
                )
            })
            .collect();

        Ok(SnapshotData {
            particles,
            time: time[0],
            redshift: redshift[0],
            box_size: box_size[0],
        })
    }
}

#[cfg(all(test, feature = "hdf5"))]
mod tests {
    use super::*;
    use gadget_ng_core::Vec3;

    #[test]
    fn hdf5_roundtrip_header_and_particles() {
        let dir = tempfile::tempdir().unwrap();
        let particles = vec![
            Particle::new(0, 1.0, Vec3::new(0.1, 0.2, 0.3), Vec3::zero()),
            Particle::new(1, 2.0, Vec3::new(1., 0., 0.), Vec3::new(0., 1., 0.)),
        ];
        let prov =
            crate::provenance::Provenance::new("0-test", None, "debug", vec![], vec![], "hash");
        let env = crate::writer::SnapshotEnv {
            time: 0.1,
            redshift: 0.0,
            box_size: 2.0,
        };
        Hdf5Writer
            .write(dir.path(), &particles, &prov, &env)
            .unwrap();

        let path = dir.path().join("snapshot.hdf5");
        let file = hdf5::File::open(&path).unwrap();
        let header = file.group("Header").unwrap();
        let npt: Vec<i64> = header.attr("NumPart_Total").unwrap().read_raw().unwrap();
        assert_eq!(npt[1], 2);

        let pt1 = file.group("PartType1").unwrap();
        let coords: Vec<f64> = pt1.dataset("Coordinates").unwrap().read_raw().unwrap();
        assert_eq!(coords.len(), 6);
        assert!((coords[0] - 0.1).abs() < 1e-12);
        let masses: Vec<f64> = pt1.dataset("Masses").unwrap().read_raw().unwrap();
        assert!((masses[0] - 1.0).abs() < 1e-12);
        let ids: Vec<i64> = pt1.dataset("ParticleIDs").unwrap().read_raw().unwrap();
        assert_eq!(ids[0], 0);
        assert_eq!(ids[1], 1);
    }

    #[test]
    fn hdf5_reader_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let particles = vec![
            Particle::new(0, 1.5, Vec3::new(0.1, 0.2, 0.3), Vec3::new(0.4, 0.5, 0.6)),
            Particle::new(1, 2.5, Vec3::new(1.0, 2.0, 3.0), Vec3::new(-1., 0., 1.)),
        ];
        let prov =
            crate::provenance::Provenance::new("0-test", None, "debug", vec![], vec![], "hash");
        let env = crate::writer::SnapshotEnv {
            time: 0.33,
            redshift: 2.0,
            box_size: 8.0,
        };
        Hdf5Writer
            .write(dir.path(), &particles, &prov, &env)
            .unwrap();
        let data = Hdf5Reader.read(dir.path()).unwrap();
        assert_eq!(data.particles.len(), 2);
        assert_eq!(data.particles[0], particles[0]);
        assert_eq!(data.particles[1], particles[1]);
        assert!((data.time - 0.33).abs() < 1e-12);
        assert!((data.box_size - 8.0).abs() < 1e-12);
        assert!((data.redshift - 2.0).abs() < 1e-12);
    }
}
