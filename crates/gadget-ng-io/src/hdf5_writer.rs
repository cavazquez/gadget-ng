//! HDF5 al estilo GADGET-4 (grupos `Header`, `PartType1`, etc.) para interoperar con `yt` / `h5py`.
use std::path::Path;

use gadget_ng_core::{Particle, Vec3};
use ndarray::{Array1, Array2};

use crate::error::SnapshotError;
use crate::gadget4_attrs::{write_gadget4_header, Gadget4Header};
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

        let path = out_dir.join("snapshot.hdf5");
        let file = hdf5::File::create(&path)?;

        // Separar gas (internal_energy > 0) de DM (internal_energy == 0)
        // Phase 102: PartType0 = gas SPH, PartType1 = DM colisionless
        let gas: Vec<&Particle> = particles.iter().filter(|p| p.internal_energy > 0.0).collect();
        let dm: Vec<&Particle> = particles.iter().filter(|p| p.internal_energy == 0.0).collect();
        let n_gas = gas.len();
        let n_dm = dm.len();

        // --- Header completo GADGET-4 (Phase 74 + Phase 102) ---
        let g4h = if n_gas > 0 {
            Gadget4Header::for_sph(
                n_gas,
                n_dm,
                env.time,
                env.box_size,
                env.omega_m,
                env.omega_lambda,
                0.045, // omega_baryon estándar Planck 2018
                env.h_dimless,
            )
        } else {
            Gadget4Header::for_nbody(
                n_dm,
                env.time,
                env.box_size,
                env.omega_m,
                env.omega_lambda,
                env.h_dimless,
            )
        };
        let header = file.create_group("Header")?;
        write_gadget4_header(&header, &g4h)?;

        // --- PartType0 = gas SPH (solo si hay partículas de gas) ---
        if n_gas > 0 {
            let mut coords = Array2::<f64>::zeros((n_gas, 3));
            let mut vels = Array2::<f64>::zeros((n_gas, 3));
            let mut masses = Array1::<f64>::zeros(n_gas);
            let mut ids = Array1::<i64>::zeros(n_gas);
            let mut u_int = Array1::<f64>::zeros(n_gas);
            let mut h_sml = Array1::<f64>::zeros(n_gas);
            // Phase 131: campos extendidos
            let mut metallicity = Array1::<f64>::zeros(n_gas);
            let mut cr_energy = Array1::<f64>::zeros(n_gas);
            let mut h2_fraction = Array1::<f64>::zeros(n_gas);
            let mut dust_to_gas = Array1::<f64>::zeros(n_gas);
            let mut b_field = Array2::<f64>::zeros((n_gas, 3));
            let mut psi_div = Array1::<f64>::zeros(n_gas);

            for (i, p) in gas.iter().enumerate() {
                coords[[i, 0]] = p.position.x;
                coords[[i, 1]] = p.position.y;
                coords[[i, 2]] = p.position.z;
                vels[[i, 0]] = p.velocity.x;
                vels[[i, 1]] = p.velocity.y;
                vels[[i, 2]] = p.velocity.z;
                masses[i] = p.mass;
                ids[i] = p.global_id as i64;
                u_int[i] = p.internal_energy;
                h_sml[i] = p.smoothing_length;
                metallicity[i] = p.metallicity;
                cr_energy[i] = p.cr_energy;
                h2_fraction[i] = p.h2_fraction;
                dust_to_gas[i] = p.dust_to_gas;
                b_field[[i, 0]] = p.b_field.x;
                b_field[[i, 1]] = p.b_field.y;
                b_field[[i, 2]] = p.b_field.z;
                psi_div[i] = p.psi_div;
            }
            let pt0 = file.create_group("PartType0")?;
            pt0.new_dataset_builder().with_data(&coords).create("Coordinates")?;
            pt0.new_dataset_builder().with_data(&vels).create("Velocities")?;
            pt0.new_dataset_builder().with_data(&masses).create("Masses")?;
            pt0.new_dataset_builder().with_data(&ids).create("ParticleIDs")?;
            pt0.new_dataset_builder().with_data(&u_int).create("InternalEnergy")?;
            pt0.new_dataset_builder().with_data(&h_sml).create("SmoothingLength")?;
            // Phase 131: campos físicos extendidos (siempre escritos para compatibilidad total)
            pt0.new_dataset_builder().with_data(&metallicity).create("Metallicity")?;
            pt0.new_dataset_builder().with_data(&cr_energy).create("CosmicRayEnergy")?;
            pt0.new_dataset_builder().with_data(&h2_fraction).create("H2Fraction")?;
            pt0.new_dataset_builder().with_data(&dust_to_gas).create("DustToGas")?;
            pt0.new_dataset_builder().with_data(&b_field).create("MagneticField")?;
            pt0.new_dataset_builder().with_data(&psi_div).create("DednerPsi")?;
        }

        // --- PartType4 = estrellas (Phase 131) ---
        {
            use gadget_ng_core::ParticleType;
            let stars: Vec<&Particle> = particles.iter().filter(|p| p.ptype == ParticleType::Star).collect();
            let n_stars = stars.len();
            if n_stars > 0 {
                let mut coords = Array2::<f64>::zeros((n_stars, 3));
                let mut vels = Array2::<f64>::zeros((n_stars, 3));
                let mut masses = Array1::<f64>::zeros(n_stars);
                let mut ids = Array1::<i64>::zeros(n_stars);
                let mut stellar_age = Array1::<f64>::zeros(n_stars);
                let mut star_metallicity = Array1::<f64>::zeros(n_stars);
                for (i, p) in stars.iter().enumerate() {
                    coords[[i, 0]] = p.position.x;
                    coords[[i, 1]] = p.position.y;
                    coords[[i, 2]] = p.position.z;
                    vels[[i, 0]] = p.velocity.x;
                    vels[[i, 1]] = p.velocity.y;
                    vels[[i, 2]] = p.velocity.z;
                    masses[i] = p.mass;
                    ids[i] = p.global_id as i64;
                    stellar_age[i] = p.stellar_age;
                    star_metallicity[i] = p.metallicity;
                }
                let pt4 = file.create_group("PartType4")?;
                pt4.new_dataset_builder().with_data(&coords).create("Coordinates")?;
                pt4.new_dataset_builder().with_data(&vels).create("Velocities")?;
                pt4.new_dataset_builder().with_data(&masses).create("Masses")?;
                pt4.new_dataset_builder().with_data(&ids).create("ParticleIDs")?;
                pt4.new_dataset_builder().with_data(&stellar_age).create("StellarAge")?;
                pt4.new_dataset_builder().with_data(&star_metallicity).create("Metallicity")?;
            }
        }

        // --- PartType1 = DM (N-body colisionless) ---
        if n_dm > 0 {
            let mut coords = Array2::<f64>::zeros((n_dm, 3));
            let mut vels = Array2::<f64>::zeros((n_dm, 3));
            let mut masses = Array1::<f64>::zeros(n_dm);
            let mut ids = Array1::<i64>::zeros(n_dm);
            for (i, p) in dm.iter().enumerate() {
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
            pt1.new_dataset_builder().with_data(&coords).create("Coordinates")?;
            pt1.new_dataset_builder().with_data(&vels).create("Velocities")?;
            pt1.new_dataset_builder().with_data(&masses).create("Masses")?;
            pt1.new_dataset_builder().with_data(&ids).create("ParticleIDs")?;
        }

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

        let mut particles: Vec<Particle> = Vec::new();

        // Leer PartType0 = gas SPH (Phase 102)
        if let Ok(pt0) = file.group("PartType0") {
            let coords: Vec<f64> = pt0.dataset("Coordinates")?.read_raw()?;
            let vels: Vec<f64> = pt0.dataset("Velocities")?.read_raw()?;
            let masses: Vec<f64> = pt0.dataset("Masses")?.read_raw()?;
            let ids: Vec<i64> = pt0.dataset("ParticleIDs")?.read_raw()?;
            let u_int_opt: Option<Vec<f64>> = pt0.dataset("InternalEnergy")
                .ok()
                .and_then(|d| d.read_raw().ok());
            let h_sml_opt: Option<Vec<f64>> = pt0.dataset("SmoothingLength")
                .ok()
                .and_then(|d| d.read_raw().ok());
            let n = ids.len();
            for i in 0..n {
                let mut p = Particle::new(
                    ids[i] as usize,
                    masses[i],
                    Vec3::new(coords[i * 3], coords[i * 3 + 1], coords[i * 3 + 2]),
                    Vec3::new(vels[i * 3], vels[i * 3 + 1], vels[i * 3 + 2]),
                );
                if let Some(ref u) = u_int_opt { p.internal_energy = u[i]; }
                if let Some(ref h) = h_sml_opt { p.smoothing_length = h[i]; }
                particles.push(p);
            }
        }

        // Leer PartType1 = DM colisionless
        if let Ok(pt1) = file.group("PartType1") {
            let coords: Vec<f64> = pt1.dataset("Coordinates")?.read_raw()?;
            let vels: Vec<f64> = pt1.dataset("Velocities")?.read_raw()?;
            let masses: Vec<f64> = pt1.dataset("Masses")?.read_raw()?;
            let ids: Vec<i64> = pt1.dataset("ParticleIDs")?.read_raw()?;
            let n = ids.len();
            for i in 0..n {
                particles.push(Particle::new(
                    ids[i] as usize,
                    masses[i],
                    Vec3::new(coords[i * 3], coords[i * 3 + 1], coords[i * 3 + 2]),
                    Vec3::new(vels[i * 3], vels[i * 3 + 1], vels[i * 3 + 2]),
                ));
            }
        }

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
            ..Default::default()
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
        // time = 1/(z+1) para que Gadget4Header calcule redshift == 2.0 exactamente
        let env = crate::writer::SnapshotEnv {
            time: 1.0 / 3.0,
            redshift: 2.0,
            box_size: 8.0,
            ..Default::default()
        };
        Hdf5Writer
            .write(dir.path(), &particles, &prov, &env)
            .unwrap();
        let data = Hdf5Reader.read(dir.path()).unwrap();
        assert_eq!(data.particles.len(), 2);
        assert_eq!(data.particles[0], particles[0]);
        assert_eq!(data.particles[1], particles[1]);
        assert!((data.time - 1.0 / 3.0).abs() < 1e-12);
        assert!((data.box_size - 8.0).abs() < 1e-12);
        assert!((data.redshift - 2.0).abs() < 1e-12);
    }
}
