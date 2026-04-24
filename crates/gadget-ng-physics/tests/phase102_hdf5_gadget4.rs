//! Phase 102 — HDF5 layout GADGET-4 completo.
//!
//! Verifica:
//! 1. Header tiene todos los atributos obligatorios GADGET-4.
//! 2. PartType0 (gas) se escribe con InternalEnergy y SmoothingLength.
//! 3. PartType1 (DM) se escribe correctamente.
//! 4. El reader reconstruye todas las partículas (gas + DM) en orden correcto.
//! 5. for_sph() produce NumPart_Total[0] = n_gas, NumPart_Total[1] = n_dm.

use gadget_ng_io::gadget4_attrs::Gadget4Header;

#[test]
fn header_for_sph_counts() {
    let h = Gadget4Header::for_sph(100, 200, 0.5, 50.0, 0.3, 0.7, 0.045, 0.7);
    assert_eq!(h.num_part_total[0], 100, "gas");
    assert_eq!(h.num_part_total[1], 200, "DM");
    assert_eq!(h.num_part_this_file[0], 100);
    assert_eq!(h.num_part_this_file[1], 200);
    assert_eq!(h.num_files_per_snapshot, 1);
    assert_eq!(h.flag_double_precision, 1);
    assert_eq!(h.flag_sfr, 1, "SPH activa SFR");
    assert_eq!(h.flag_cooling, 1, "SPH activa cooling");
}

#[test]
fn header_nbody_no_gas() {
    let h = Gadget4Header::for_nbody(512, 1.0, 100.0, 0.3, 0.7, 0.7);
    assert_eq!(h.num_part_total[0], 0, "sin gas");
    assert_eq!(h.num_part_total[1], 512);
    assert_eq!(h.flag_sfr, 0);
    assert_eq!(h.flag_cooling, 0);
}

#[test]
fn header_mandatory_flags_present() {
    let h = Gadget4Header::default();
    // Todos los flags deben tener valores definidos (no acceso a campo sin valor)
    assert!(h.flag_sfr == 0 || h.flag_sfr == 1);
    assert!(h.flag_cooling == 0 || h.flag_cooling == 1);
    assert!(h.flag_feedback == 0 || h.flag_feedback == 1);
    assert!(h.flag_double_precision == 1, "debe ser doble precisión por defecto");
    assert_eq!(h.num_files_per_snapshot, 1, "mono-archivo por defecto");
}

#[test]
fn header_cosmological_params() {
    let h = Gadget4Header::for_sph(10, 10, 0.25, 100.0, 0.31, 0.69, 0.048, 0.677);
    assert!((h.omega_m - 0.31).abs() < 1e-12, "Omega0");
    assert!((h.omega_lambda - 0.69).abs() < 1e-12, "OmegaLambda");
    assert!((h.omega_baryon - 0.048).abs() < 1e-12, "OmegaBaryon");
    assert!((h.hubble_param - 0.677).abs() < 1e-12, "HubbleParam");
    assert!((h.redshift - 3.0).abs() < 1e-10, "z = 1/0.25 - 1 = 3");
}

#[test]
fn header_unit_constants_set() {
    use gadget_ng_io::gadget4_attrs::{KMS_IN_CMS, KPC_IN_CM, MSUN_IN_G};
    let h = Gadget4Header::default();
    assert!((h.unit_length_in_cm - KPC_IN_CM).abs() < 1.0);
    assert!((h.unit_mass_in_g - MSUN_IN_G).abs() < 1e28);
    assert!((h.unit_velocity_in_cm_per_s - KMS_IN_CMS).abs() < 1.0);
}

// Tests HDF5 con feature "hdf5"
#[cfg(feature = "hdf5")]
mod hdf5_tests {
    use gadget_ng_core::{Particle, Vec3};
    use gadget_ng_io::{
        gadget4_attrs::Gadget4Header,
        hdf5_writer::{Hdf5Reader, Hdf5Writer},
        provenance::Provenance,
        writer::{SnapshotEnv, SnapshotWriter},
        reader::SnapshotReader,
    };

    fn make_prov() -> Provenance {
        Provenance::new("0-test", None, "debug", vec![], vec![], "hash")
    }

    fn make_env() -> SnapshotEnv {
        SnapshotEnv { time: 0.5, redshift: 1.0, box_size: 20.0, ..Default::default() }
    }

    #[test]
    fn hdf5_gas_and_dm_parttype0_and_1() {
        let dir = tempfile::tempdir().unwrap();
        let mut particles = Vec::new();

        // 3 partículas de gas (internal_energy > 0)
        for i in 0..3 {
            let mut p = Particle::new(i, 1.0,
                Vec3::new(i as f64, 0.0, 0.0), Vec3::zero());
            p.internal_energy = 500.0;
            p.smoothing_length = 0.3;
            particles.push(p);
        }
        // 2 partículas DM (internal_energy = 0)
        for i in 3..5 {
            particles.push(Particle::new(i, 2.0,
                Vec3::new(i as f64, 1.0, 0.0), Vec3::zero()));
        }

        Hdf5Writer.write(dir.path(), &particles, &make_prov(), &make_env()).unwrap();

        let file = hdf5::File::open(dir.path().join("snapshot.hdf5")).unwrap();

        // Header: NumPart[0]=3 (gas), NumPart[1]=2 (DM)
        let hdr = file.group("Header").unwrap();
        let npt: Vec<i64> = hdr.attr("NumPart_Total").unwrap().read_raw().unwrap();
        assert_eq!(npt[0], 3, "gas NumPart_Total[0]");
        assert_eq!(npt[1], 2, "DM NumPart_Total[1]");

        // PartType0 debe existir con InternalEnergy y SmoothingLength
        let pt0 = file.group("PartType0").unwrap();
        let u: Vec<f64> = pt0.dataset("InternalEnergy").unwrap().read_raw().unwrap();
        assert_eq!(u.len(), 3);
        assert!((u[0] - 500.0).abs() < 1e-12);

        let h: Vec<f64> = pt0.dataset("SmoothingLength").unwrap().read_raw().unwrap();
        assert!((h[0] - 0.3).abs() < 1e-12);

        // PartType1 debe existir con DM
        let pt1 = file.group("PartType1").unwrap();
        let masses: Vec<f64> = pt1.dataset("Masses").unwrap().read_raw().unwrap();
        assert_eq!(masses.len(), 2);
        assert!((masses[0] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn hdf5_all_gas_no_parttype1() {
        let dir = tempfile::tempdir().unwrap();
        let mut particles = Vec::new();
        for i in 0..4 {
            let mut p = Particle::new(i, 1.0, Vec3::new(i as f64, 0.0, 0.0), Vec3::zero());
            p.internal_energy = 100.0;
            particles.push(p);
        }

        Hdf5Writer.write(dir.path(), &particles, &make_prov(), &make_env()).unwrap();
        let file = hdf5::File::open(dir.path().join("snapshot.hdf5")).unwrap();

        let npt: Vec<i64> = file.group("Header").unwrap()
            .attr("NumPart_Total").unwrap().read_raw().unwrap();
        assert_eq!(npt[0], 4, "todo gas");
        assert_eq!(npt[1], 0, "sin DM");

        // PartType0 presente
        assert!(file.group("PartType0").is_ok());
    }

    #[test]
    fn hdf5_reader_reads_gas_and_dm() {
        let dir = tempfile::tempdir().unwrap();
        let mut particles = Vec::new();

        let mut g = Particle::new(0, 1.0, Vec3::new(1.0, 2.0, 3.0), Vec3::new(0.1, 0.0, 0.0));
        g.internal_energy = 250.0;
        g.smoothing_length = 0.5;
        particles.push(g);

        particles.push(Particle::new(1, 3.0, Vec3::new(5.0, 6.0, 7.0), Vec3::new(0.0, 0.2, 0.0)));

        Hdf5Writer.write(dir.path(), &particles, &make_prov(), &make_env()).unwrap();
        let data = Hdf5Reader.read(dir.path()).unwrap();

        assert_eq!(data.particles.len(), 2);
        // Gas primero (id=0)
        let gas = data.particles.iter().find(|p| p.internal_energy > 0.0).unwrap();
        assert_eq!(gas.global_id, 0);
        assert!((gas.internal_energy - 250.0).abs() < 1e-12);
        assert!((gas.smoothing_length - 0.5).abs() < 1e-12);

        // DM (id=1)
        let dm = data.particles.iter().find(|p| p.internal_energy == 0.0).unwrap();
        assert_eq!(dm.global_id, 1);
        assert!((dm.mass - 3.0).abs() < 1e-12);
    }

    #[test]
    fn hdf5_flag_sfr_set_for_sph() {
        let dir = tempfile::tempdir().unwrap();
        let mut p = Particle::new(0, 1.0, Vec3::zero(), Vec3::zero());
        p.internal_energy = 100.0;
        let particles = vec![p];

        Hdf5Writer.write(dir.path(), &particles, &make_prov(), &make_env()).unwrap();
        let file = hdf5::File::open(dir.path().join("snapshot.hdf5")).unwrap();
        let hdr = file.group("Header").unwrap();

        let flag_sfr: Vec<i32> = hdr.attr("Flag_Sfr").unwrap().read_raw().unwrap();
        assert_eq!(flag_sfr[0], 1, "Flag_Sfr debe ser 1 con gas");

        let flag_dp: Vec<i32> = hdr.attr("Flag_DoublePrecision").unwrap().read_raw().unwrap();
        assert_eq!(flag_dp[0], 1, "Flag_DoublePrecision siempre 1");
    }
}
