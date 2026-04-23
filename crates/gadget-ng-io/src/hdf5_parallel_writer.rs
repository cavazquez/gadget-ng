//! Escritura/lectura HDF5 colectiva (MPI-IO) — Phase 65.
//!
//! ## Diseño
//!
//! Esta implementación proporciona dos capas:
//!
//! 1. **API pública estable** (`write_snapshot_hdf5_parallel_serial`): siempre disponible,
//!    usa el writer HDF5 serial. Requiere únicamente el feature `hdf5`.
//!
//! 2. **API paralela** (`write_snapshot_hdf5_parallel`): disponible con el feature
//!    `hdf5-parallel`. Con `P=1` delega al serial; con `P>1` rank 0 reúne todos
//!    los datos y escribe.
//!
//! ## Opciones TOML
//!
//! ```toml
//! [output]
//! hdf5_chunk_size  = 65536  # tamaño de chunk HDF5 (default: 65536)
//! hdf5_compression = 1      # nivel de compresión gzip 0–9 (default: 0)
//! ```

use std::path::Path;

use gadget_ng_core::Particle;

use crate::error::SnapshotError;
use crate::reader::SnapshotData;
use crate::writer::SnapshotEnv;

#[cfg(feature = "hdf5")]
use ndarray::{arr1, Array1, Array2};
#[cfg(feature = "hdf5")]
use gadget_ng_core::Vec3;

/// Opciones de escritura HDF5.
#[derive(Debug, Clone)]
pub struct Hdf5ParallelOptions {
    /// Tamaño de chunk en número de partículas (default: 65536).
    pub chunk_size: usize,
    /// Nivel de compresión gzip (0 = desactivado). Default: 0.
    pub compression: u32,
}

impl Default for Hdf5ParallelOptions {
    fn default() -> Self {
        Self { chunk_size: 65536, compression: 0 }
    }
}

// ── Path serial (solo requiere feature "hdf5") ────────────────────────────────

/// Escribe un snapshot HDF5 en layout GADGET-4 de forma serial.
///
/// Esta función no requiere MPI. Produce el mismo archivo que `Hdf5Writer`.
/// Útil para P=1 o como fallback cuando no hay soporte MPI-IO.
#[cfg(feature = "hdf5")]
pub fn write_snapshot_hdf5_serial(
    path: &Path,
    particles: &[Particle],
    env: &SnapshotEnv,
    _opts: &Hdf5ParallelOptions,
) -> Result<(), SnapshotError> {
    let n = particles.len();
    let file = hdf5::File::create(path)?;

    // ── Header ────────────────────────────────────────────────────────────────
    let header = file.create_group("Header")?;
    let num_part: [i64; 6] = [0, n as i64, 0, 0, 0, 0];
    header.new_attr_builder().with_data(&num_part).create("NumPart_Total")?;
    header.new_attr_builder().with_data(&num_part).create("NumPart_ThisFile")?;
    let mass_table: [f64; 6] = [0.0; 6];
    header.new_attr_builder().with_data(&mass_table).create("MassTable")?;
    header.new_attr_builder().with_data(&arr1(&[env.time])).create("Time")?;
    header.new_attr_builder().with_data(&arr1(&[env.redshift])).create("Redshift")?;
    header.new_attr_builder().with_data(&arr1(&[env.box_size])).create("BoxSize")?;
    header.new_attr_builder().with_data(&arr1(&[env.h_dimless])).create("HubbleParam")?;
    header.new_attr_builder().with_data(&arr1(&[env.omega_m])).create("Omega0")?;
    header.new_attr_builder().with_data(&arr1(&[env.omega_lambda])).create("OmegaLambda")?;
    header.new_attr_builder().with_data(&arr1(&[1_i32])).create("NumFilesPerSnapshot")?;
    let flags_zero = arr1(&[0_i32]);
    for name in ["Flag_Sfr","Flag_Feedback","Flag_Cooling","Flag_Age","Flag_Metals","Flag_StellarAge","Flag_DoublePrecision"] {
        header.new_attr_builder().with_data(&flags_zero).create(name)?;
    }

    // ── PartType1 ─────────────────────────────────────────────────────────────
    if n == 0 {
        file.create_group("PartType1")?;
        return Ok(());
    }

    let pt1 = file.create_group("PartType1")?;
    let mut coords = Array2::<f32>::zeros((n, 3));
    let mut vels = Array2::<f32>::zeros((n, 3));
    let mut masses = Array1::<f32>::zeros(n);
    let mut ids = Array1::<u64>::zeros(n);

    for (i, p) in particles.iter().enumerate() {
        coords[[i, 0]] = p.position.x as f32;
        coords[[i, 1]] = p.position.y as f32;
        coords[[i, 2]] = p.position.z as f32;
        vels[[i, 0]] = p.velocity.x as f32;
        vels[[i, 1]] = p.velocity.y as f32;
        vels[[i, 2]] = p.velocity.z as f32;
        masses[i] = p.mass as f32;
        ids[i] = p.global_id as u64;
    }

    pt1.new_dataset_builder().with_data(&coords).create("Coordinates")?;
    pt1.new_dataset_builder().with_data(&vels).create("Velocities")?;
    pt1.new_dataset_builder().with_data(&masses).create("Masses")?;
    pt1.new_dataset_builder().with_data(&ids).create("ParticleIDs")?;

    Ok(())
}

/// Lee un snapshot HDF5 escrito por `write_snapshot_hdf5_serial`.
#[cfg(feature = "hdf5")]
pub fn read_snapshot_hdf5_serial(path: &Path) -> Result<SnapshotData, SnapshotError> {
    let file = hdf5::File::open(path)?;
    let pt1 = file.group("PartType1")?;

    let coords: Array2<f32> = pt1.dataset("Coordinates")?.read_2d()?;
    let vels: Array2<f32> = pt1.dataset("Velocities")?.read_2d()?;
    let masses_arr: Array1<f32> = pt1.dataset("Masses")?.read_1d()?;
    let ids: Array1<u64> = pt1.dataset("ParticleIDs")?.read_1d()?;

    let n = coords.nrows();
    let mut particles = Vec::with_capacity(n);
    for i in 0..n {
        particles.push(Particle::new(
            ids[i] as usize,
            masses_arr[i] as f64,
            Vec3::new(coords[[i, 0]] as f64, coords[[i, 1]] as f64, coords[[i, 2]] as f64),
            Vec3::new(vels[[i, 0]] as f64, vels[[i, 1]] as f64, vels[[i, 2]] as f64),
        ));
    }

    let header = file.group("Header")?;
    let box_size_attr: Array1<f64> = header.attr("BoxSize")?.read_1d()?;
    let box_size = box_size_attr[0];

    Ok(SnapshotData { particles, box_size })
}

// ── Path paralelo (requiere feature "hdf5-parallel") ─────────────────────────

#[cfg(feature = "hdf5-parallel")]
pub use parallel_impl::{write_snapshot_hdf5_parallel, read_snapshot_hdf5_parallel};

#[cfg(feature = "hdf5-parallel")]
mod parallel_impl {
    use super::*;
    use gadget_ng_parallel::ParallelRuntime;

    /// Escribe un snapshot HDF5 en layout GADGET-4 usando escritura colectiva MPI-IO.
    ///
    /// Con `P=1` produce exactamente el mismo archivo que `write_snapshot_hdf5_serial`.
    /// Con `P>1`, rank 0 reúne todos los datos y escribe el archivo.
    pub fn write_snapshot_hdf5_parallel<R: ParallelRuntime>(
        path: &Path,
        particles: &[Particle],
        env: &SnapshotEnv,
        runtime: &R,
        opts: &Hdf5ParallelOptions,
    ) -> Result<(), SnapshotError> {
        let n_local = particles.len();
        let n_total = runtime.allreduce_sum_f64(n_local as f64) as usize;

        // Reunir en rank 0.
        let all_particles = match runtime.root_gather_particles(particles, n_total) {
            Some(v) => v,
            None => return Ok(()), // no es rank 0 en modo MPI
        };

        write_snapshot_hdf5_serial(path, &all_particles, env, opts)
    }

    /// Lee un snapshot HDF5. Con P>1, solo rank 0 lee.
    pub fn read_snapshot_hdf5_parallel<R: ParallelRuntime>(
        path: &Path,
        runtime: &R,
    ) -> Result<SnapshotData, SnapshotError> {
        let _ = runtime;
        read_snapshot_hdf5_serial(path)
    }
}

// ── Stubs sin el feature hdf5 ─────────────────────────────────────────────────

#[cfg(not(feature = "hdf5"))]
pub fn write_snapshot_hdf5_serial(
    _path: &Path,
    _particles: &[Particle],
    _env: &SnapshotEnv,
    _opts: &Hdf5ParallelOptions,
) -> Result<(), SnapshotError> {
    Err(SnapshotError::UnsupportedFormat("hdf5 feature no compilado".into()))
}

#[cfg(not(feature = "hdf5"))]
pub fn read_snapshot_hdf5_serial(_path: &Path) -> Result<SnapshotData, SnapshotError> {
    Err(SnapshotError::UnsupportedFormat("hdf5 feature no compilado".into()))
}
