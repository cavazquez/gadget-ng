//! Snapshot en formato **NetCDF-4** usando el crate `netcdf`.
//!
//! Produce un único fichero `snapshot.nc` más los sidecar `meta.json` /
//! `provenance.json` habituales. El layout sigue la convención usada en
//! simulaciones astrofísicas modernas:
//!
//! ```text
//! Dimensiones:
//!   particle   — número de partículas
//!
//! Variables (f64, dim particle):
//!   x, y, z     — posición
//!   vx, vy, vz  — velocidad
//!   mass        — masa
//!   id          — global_id (i64)
//!
//! Atributos globales:
//!   time, redshift, box_size, particle_count, schema_version
//! ```
//!
//! ## Lectura desde Python
//!
//! ```python
//! import xarray as xr
//! ds = xr.open_dataset("snapshot_final/snapshot.nc")
//! # O con netCDF4:
//! import netCDF4 as nc
//! ds = nc.Dataset("snapshot_final/snapshot.nc")
//! x = ds.variables["x"][:]
//! ```
//!
//! ## Lectura desde Julia
//!
//! ```julia
//! using NCDatasets
//! ds = NCDataset("snapshot_final/snapshot.nc")
//! x = ds["x"][:]
//! ```
use std::path::Path;

use gadget_ng_core::{Particle, Vec3};

use crate::error::SnapshotError;
use crate::provenance::Provenance;
use crate::reader::{SnapshotData, SnapshotReader};
use crate::snapshot::{build_meta, write_sidecar_json};
use crate::writer::{SnapshotEnv, SnapshotWriter};

const NC_FILE: &str = "snapshot.nc";

fn nc_err(e: netcdf::Error) -> SnapshotError {
    SnapshotError::Netcdf(e.to_string())
}

/// Escritor NetCDF-4: un fichero por snapshot (`snapshot.nc`).
#[derive(Debug, Default, Clone, Copy)]
pub struct NetcdfWriter;

impl SnapshotWriter for NetcdfWriter {
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
        let path = out_dir.join(NC_FILE);
        let mut file = netcdf::create(&path).map_err(nc_err)?;

        // Dimensión principal
        file.add_dimension("particle", n).map_err(nc_err)?;

        // Atributos globales de metadatos
        file.add_attribute("schema_version", meta.schema_version as i32)
            .map_err(nc_err)?;
        file.add_attribute("time", meta.time).map_err(nc_err)?;
        file.add_attribute("redshift", meta.redshift)
            .map_err(nc_err)?;
        file.add_attribute("box_size", meta.box_size)
            .map_err(nc_err)?;
        file.add_attribute("particle_count", n as i64)
            .map_err(nc_err)?;

        // Extraer columnas SoA para escritura eficiente
        let mut x = vec![0.0_f64; n];
        let mut y = vec![0.0_f64; n];
        let mut z = vec![0.0_f64; n];
        let mut vx = vec![0.0_f64; n];
        let mut vy = vec![0.0_f64; n];
        let mut vz = vec![0.0_f64; n];
        let mut mass = vec![0.0_f64; n];
        let mut id = vec![0_i64; n];

        for (i, p) in particles.iter().enumerate() {
            x[i] = p.position.x;
            y[i] = p.position.y;
            z[i] = p.position.z;
            vx[i] = p.velocity.x;
            vy[i] = p.velocity.y;
            vz[i] = p.velocity.z;
            mass[i] = p.mass;
            id[i] = p.global_id as i64;
        }

        write_var_f64(&mut file, "x", "code_length", &x)?;
        write_var_f64(&mut file, "y", "code_length", &y)?;
        write_var_f64(&mut file, "z", "code_length", &z)?;
        write_var_f64(&mut file, "vx", "code_velocity", &vx)?;
        write_var_f64(&mut file, "vy", "code_velocity", &vy)?;
        write_var_f64(&mut file, "vz", "code_velocity", &vz)?;
        write_var_f64(&mut file, "mass", "code_mass", &mass)?;

        {
            let mut var = file
                .add_variable::<i64>("id", &["particle"])
                .map_err(nc_err)?;
            var.put_attribute("long_name", "global particle id")
                .map_err(nc_err)?;
            var.put_values(&id, ..).map_err(nc_err)?;
        }

        Ok(())
    }
}

fn write_var_f64(
    file: &mut netcdf::FileMut,
    name: &str,
    units: &str,
    data: &[f64],
) -> Result<(), SnapshotError> {
    let mut var = file
        .add_variable::<f64>(name, &["particle"])
        .map_err(nc_err)?;
    var.put_attribute("units", units).map_err(nc_err)?;
    var.put_values(data, ..).map_err(nc_err)?;
    Ok(())
}

/// Lector NetCDF-4: reconstruye partículas y metadatos desde `snapshot.nc`.
#[derive(Debug, Default, Clone, Copy)]
pub struct NetcdfReader;

impl SnapshotReader for NetcdfReader {
    fn read(&self, dir: &Path) -> Result<SnapshotData, SnapshotError> {
        let path = dir.join(NC_FILE);
        let file = netcdf::open(&path).map_err(nc_err)?;

        let time = read_attr_f64(&file, "time")?;
        let redshift = read_attr_f64(&file, "redshift")?;
        let box_size = read_attr_f64(&file, "box_size")?;

        let x: Vec<f64> = get_var_f64(&file, "x")?;
        let y: Vec<f64> = get_var_f64(&file, "y")?;
        let z: Vec<f64> = get_var_f64(&file, "z")?;
        let vx: Vec<f64> = get_var_f64(&file, "vx")?;
        let vy: Vec<f64> = get_var_f64(&file, "vy")?;
        let vz: Vec<f64> = get_var_f64(&file, "vz")?;
        let mass: Vec<f64> = get_var_f64(&file, "mass")?;
        let id: Vec<i64> = get_var_i64(&file, "id")?;

        let n = id.len();
        let particles = (0..n)
            .map(|i| {
                Particle::new(
                    id[i] as usize,
                    mass[i],
                    Vec3::new(x[i], y[i], z[i]),
                    Vec3::new(vx[i], vy[i], vz[i]),
                )
            })
            .collect();

        Ok(SnapshotData {
            particles,
            time,
            redshift,
            box_size,
        })
    }
}

fn read_attr_f64(file: &netcdf::File, name: &str) -> Result<f64, SnapshotError> {
    use netcdf::AttributeValue;
    let attr = file
        .attribute(name)
        .ok_or_else(|| SnapshotError::Netcdf(format!("atributo '{name}' no encontrado")))?;
    match attr.value().map_err(nc_err)? {
        AttributeValue::Double(v) => Ok(v),
        AttributeValue::Float(v) => Ok(v as f64),
        other => Err(SnapshotError::Netcdf(format!(
            "tipo inesperado en atributo '{name}': {other:?}"
        ))),
    }
}

fn get_var_f64(file: &netcdf::File, name: &str) -> Result<Vec<f64>, SnapshotError> {
    file.variable(name)
        .ok_or_else(|| SnapshotError::Netcdf(format!("variable '{name}' no encontrada")))?
        .get_values::<f64, _>(..)
        .map_err(nc_err)
}

fn get_var_i64(file: &netcdf::File, name: &str) -> Result<Vec<i64>, SnapshotError> {
    file.variable(name)
        .ok_or_else(|| SnapshotError::Netcdf(format!("variable '{name}' no encontrada")))?
        .get_values::<i64, _>(..)
        .map_err(nc_err)
}

#[cfg(all(test, feature = "netcdf"))]
mod tests {
    use super::*;
    use gadget_ng_core::Vec3;

    fn dummy_prov() -> Provenance {
        crate::provenance::Provenance::new("0-test", None, "debug", vec![], vec![], "hash")
    }

    #[test]
    fn netcdf_writer_creates_file() {
        let dir = tempfile::tempdir().unwrap();
        let particles = vec![
            Particle::new(0, 1.0, Vec3::new(0.1, 0.2, 0.3), Vec3::zero()),
            Particle::new(1, 2.0, Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0)),
        ];
        NetcdfWriter
            .write(
                dir.path(),
                &particles,
                &dummy_prov(),
                &SnapshotEnv::default(),
            )
            .unwrap();
        assert!(dir.path().join(NC_FILE).exists());
        assert!(dir.path().join("meta.json").exists());
    }

    #[test]
    fn netcdf_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let particles = vec![
            Particle::new(0, 1.5, Vec3::new(0.1, 0.2, 0.3), Vec3::new(0.4, 0.5, 0.6)),
            Particle::new(1, 2.5, Vec3::new(1.0, 2.0, 3.0), Vec3::new(-1.0, 0.0, 1.0)),
        ];
        let env = SnapshotEnv {
            time: 2.71,
            redshift: 1.5,
            box_size: 12.0,
        };
        NetcdfWriter
            .write(dir.path(), &particles, &dummy_prov(), &env)
            .unwrap();
        let data = NetcdfReader.read(dir.path()).unwrap();
        assert_eq!(data.particles.len(), 2);
        assert_eq!(data.particles[0], particles[0]);
        assert_eq!(data.particles[1], particles[1]);
        assert!((data.time - 2.71).abs() < 1e-12);
        assert!((data.box_size - 12.0).abs() < 1e-12);
        assert!((data.redshift - 1.5).abs() < 1e-12);
    }

    #[test]
    fn netcdf_attributes_match_meta() {
        let dir = tempfile::tempdir().unwrap();
        let particles = vec![Particle::new(
            7,
            3.0,
            Vec3::new(0.5, 0.5, 0.5),
            Vec3::zero(),
        )];
        let env = SnapshotEnv {
            time: 0.99,
            redshift: 0.1,
            box_size: 5.0,
        };
        NetcdfWriter
            .write(dir.path(), &particles, &dummy_prov(), &env)
            .unwrap();

        let file = netcdf::open(dir.path().join(NC_FILE)).unwrap();
        let time = read_attr_f64(&file, "time").unwrap();
        let box_size = read_attr_f64(&file, "box_size").unwrap();
        assert!((time - 0.99).abs() < 1e-12);
        assert!((box_size - 5.0).abs() < 1e-12);
    }
}
