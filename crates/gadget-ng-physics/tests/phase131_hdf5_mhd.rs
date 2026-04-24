/// Phase 131 — HDF5: campos MHD + SPH completos
///
/// Tests sin HDF5 (siempre): verificaciones de estructura de datos.
/// Tests con HDF5: escritura real de b_field, psi_div, cr_energy,
///                 metallicity, h2_fraction, dust_to_gas, stellar_age.

// Estos tests no requieren HDF5: verifican la estructura de Particle
#[test]
fn particle_has_all_extended_fields() {
    use gadget_ng_core::{Particle, Vec3};
    let mut p = Particle::new_gas(0, 1.0, Vec3::zero(), Vec3::zero(), 1.0, 0.5);
    p.b_field = Vec3::new(1.0, 0.5, 0.2);
    p.psi_div = 0.01;
    p.cr_energy = 0.5;
    p.metallicity = 0.02;
    p.h2_fraction = 0.3;
    p.dust_to_gas = 0.005;

    assert_eq!(p.b_field.x, 1.0);
    assert_eq!(p.psi_div, 0.01);
    assert_eq!(p.cr_energy, 0.5);
    assert_eq!(p.metallicity, 0.02);
    assert_eq!(p.h2_fraction, 0.3);
    assert_eq!(p.dust_to_gas, 0.005);
}

#[test]
fn star_particle_has_stellar_age_and_metallicity() {
    use gadget_ng_core::{Particle, Vec3};
    let mut p = Particle::new_star(0, 1.0, Vec3::zero(), Vec3::zero(), 0.02);
    p.stellar_age = 5.0;
    assert_eq!(p.stellar_age, 5.0);
    assert_eq!(p.metallicity, 0.02);
}

#[test]
fn all_extended_fields_default_zero() {
    use gadget_ng_core::{Particle, Vec3};
    let p = Particle::new(0, 1.0, Vec3::zero(), Vec3::zero());
    assert_eq!(p.b_field.x, 0.0);
    assert_eq!(p.psi_div, 0.0);
    assert_eq!(p.cr_energy, 0.0);
    assert_eq!(p.h2_fraction, 0.0);
    assert_eq!(p.dust_to_gas, 0.0);
}

// Tests HDF5 (requieren feature "hdf5")
mod hdf5_mhd_tests {
    use gadget_ng_core::{Particle, Vec3};
    use gadget_ng_io::{
        Hdf5Writer, Provenance,
        SnapshotEnv, SnapshotWriter,
    };

    fn make_prov() -> Provenance {
        Provenance::new("0-test", None, "debug", vec![], vec![], "hash")
    }
    fn make_env() -> SnapshotEnv {
        SnapshotEnv { time: 0.5, redshift: 1.0, box_size: 20.0, ..Default::default() }
    }

    #[test]
    fn hdf5_writes_magnetic_field() {
        let dir = tempfile::tempdir().unwrap();
        let mut p = Particle::new_gas(0, 1.0, Vec3::zero(), Vec3::zero(), 1.0, 0.5);
        p.b_field = Vec3::new(2.0, 1.0, 0.5);
        Hdf5Writer.write(dir.path(), &[p], &make_prov(), &make_env()).unwrap();

        let file = hdf5::File::open(dir.path().join("snapshot.hdf5")).unwrap();
        let pt0 = file.group("PartType0").unwrap();
        let bf: Vec<f64> = pt0.dataset("MagneticField").unwrap().read_raw().unwrap();
        assert_eq!(bf.len(), 3);
        assert!((bf[0] - 2.0).abs() < 1e-12, "B.x = {}", bf[0]);
        assert!((bf[1] - 1.0).abs() < 1e-12, "B.y = {}", bf[1]);
    }

    #[test]
    fn hdf5_writes_cr_energy_and_metallicity() {
        let dir = tempfile::tempdir().unwrap();
        let mut p = Particle::new_gas(0, 1.0, Vec3::zero(), Vec3::zero(), 1.0, 0.5);
        p.cr_energy = 3.14;
        p.metallicity = 0.025;
        Hdf5Writer.write(dir.path(), &[p], &make_prov(), &make_env()).unwrap();

        let file = hdf5::File::open(dir.path().join("snapshot.hdf5")).unwrap();
        let pt0 = file.group("PartType0").unwrap();
        let cr: Vec<f64> = pt0.dataset("CosmicRayEnergy").unwrap().read_raw().unwrap();
        assert!((cr[0] - 3.14).abs() < 1e-12);
        let met: Vec<f64> = pt0.dataset("Metallicity").unwrap().read_raw().unwrap();
        assert!((met[0] - 0.025).abs() < 1e-12);
    }

    #[test]
    fn hdf5_writes_parttype4_stars() {
        let dir = tempfile::tempdir().unwrap();
        let mut star = Particle::new_star(0, 1.0, Vec3::new(1.0, 2.0, 3.0), Vec3::zero(), 0.02);
        star.stellar_age = 4.5;
        let gas = Particle::new_gas(1, 1.0, Vec3::zero(), Vec3::zero(), 1.0, 0.5);
        Hdf5Writer.write(dir.path(), &[star, gas], &make_prov(), &make_env()).unwrap();

        let file = hdf5::File::open(dir.path().join("snapshot.hdf5")).unwrap();
        let pt4 = file.group("PartType4").unwrap();
        let ages: Vec<f64> = pt4.dataset("StellarAge").unwrap().read_raw().unwrap();
        assert_eq!(ages.len(), 1);
        assert!((ages[0] - 4.5).abs() < 1e-12, "StellarAge = {}", ages[0]);
    }
}
