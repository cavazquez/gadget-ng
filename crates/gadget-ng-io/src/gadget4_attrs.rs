//! Atributos estándar de GADGET-4 para el grupo `Header` HDF5 (Phase 74).
//!
//! Proporciona una estructura `Gadget4Header` con **todos** los atributos
//! definidos en la especificación GADGET-4 HDF5, y funciones para escribirlos
//! y leerlos. Esto asegura interoperabilidad con `yt`, `pynbody`, `h5py` y
//! herramientas de análisis de terceros.
//!
//! ## Atributos incluidos
//!
//! | Nombre                         | Tipo      | Descripción                              |
//! |-------------------------------|-----------|------------------------------------------|
//! | `NumPart_ThisFile`             | `[i64;6]` | Partículas por tipo en este archivo      |
//! | `NumPart_Total`                | `[i64;6]` | Total en toda la simulación              |
//! | `NumPart_Total_HW`             | `[i64;6]` | Palabra alta para N > 2³² (GADGET-4)    |
//! | `MassTable`                    | `[f64;6]` | Masa por tipo (0 = individual)           |
//! | `Time`                         | `f64`     | Factor de escala (o tiempo físico)       |
//! | `Redshift`                     | `f64`     | Corrimiento al rojo z = 1/a − 1         |
//! | `BoxSize`                      | `f64`     | Lado del cubo periódico (en kpc/h)      |
//! | `NumFilesPerSnapshot`          | `i32`     | Número de archivos por snapshot          |
//! | `Omega0`                       | `f64`     | Ω_matter                                 |
//! | `OmegaLambda`                  | `f64`     | Ω_Λ                                      |
//! | `OmegaBaryon`                  | `f64`     | Ω_b                                      |
//! | `HubbleParam`                  | `f64`     | h = H₀/100                              |
//! | `Flag_Sfr`                     | `i32`     | 1 = formación estelar activa            |
//! | `Flag_Cooling`                 | `i32`     | 1 = enfriamiento radiativo activo        |
//! | `Flag_Feedback`                | `i32`     | 1 = feedback activo                      |
//! | `Flag_StellarAge`              | `i32`     | 1 = edades estelares almacenadas         |
//! | `Flag_Metals`                  | `i32`     | 1 = metalicidades almacenadas            |
//! | `Flag_Entropy_ICs`             | `i32`     | 1 = ICs en entropía (no energía interna) |
//! | `Flag_DoublePrecision`         | `i32`     | 1 = coordenadas en f64                  |
//! | `Flag_IC_Info`                 | `i32`     | Tipo de condiciones iniciales            |
//! | `UnitLength_in_cm`             | `f64`     | 1 kpc/h en cm                            |
//! | `UnitMass_in_g`                | `f64`     | 1 M_sun/h en g                          |
//! | `UnitVelocity_in_cm_per_s`     | `f64`     | 1 km/s en cm/s                           |
//!
//! ## Referencia
//!
//! Springel (2021), GADGET-4 Code Paper; formato HDF5 en `src/io/hdf5_legacy.cc`.

// Constantes de conversión de unidades físicas
/// 1 kpc en cm  (= 3.085678e21 cm).
pub const KPC_IN_CM: f64 = 3.085_678_e21;
/// 1 M_sun en gramos (= 1.989e33 g).
pub const MSUN_IN_G: f64 = 1.989_e33;
/// 1 km/s en cm/s.
pub const KMS_IN_CMS: f64 = 1.0e5;

/// Descripción completa del encabezado GADGET-4.
///
/// Campos opcionales tienen valores por defecto seguros (0 o false).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Gadget4Header {
    // ── Recuentos de partículas ────────────────────────────────────────────
    /// Partículas por tipo en **este** archivo (tipos 0−5).
    pub num_part_this_file: [i64; 6],
    /// Partículas totales en la simulación (word baja; tipos 0−5).
    pub num_part_total: [i64; 6],
    /// Palabra alta de `num_part_total` para N > 2³² (suele ser 0).
    pub num_part_total_hw: [i64; 6],
    /// Masa por tipo en unidades internas (0 = masa individual por partícula).
    pub mass_table: [f64; 6],

    // ── Cosmología ─────────────────────────────────────────────────────────
    /// Factor de escala `a` (o tiempo físico en simulaciones no cosmológicas).
    pub time: f64,
    /// Corrimiento al rojo `z = 1/a − 1`.
    pub redshift: f64,
    /// Tamaño de la caja en kpc/h (unidades internas).
    pub box_size: f64,
    /// Ω_matter.
    pub omega_m: f64,
    /// Ω_Λ.
    pub omega_lambda: f64,
    /// Ω_baryon.
    pub omega_baryon: f64,
    /// Parámetro de Hubble reducido h = H₀/100 km/s/Mpc.
    pub hubble_param: f64,

    // ── Archivos ───────────────────────────────────────────────────────────
    /// Número de archivos por snapshot (1 para mono-archivo).
    pub num_files_per_snapshot: i32,

    // ── Flags de física ────────────────────────────────────────────────────
    /// Star Formation Rate activa.
    pub flag_sfr: i32,
    /// Enfriamiento radiativo activo.
    pub flag_cooling: i32,
    /// Feedback estelar activo.
    pub flag_feedback: i32,
    /// Edades estelares presentes.
    pub flag_stellar_age: i32,
    /// Metalicidades presentes.
    pub flag_metals: i32,
    /// ICs expresadas en entropía (en lugar de energía interna).
    pub flag_entropy_ics: i32,
    /// 1 = coordenadas almacenadas en doble precisión.
    pub flag_double_precision: i32,
    /// Tipo de IC Info (0 = no info; 1 = ZA; 2 = 2LPT).
    pub flag_ic_info: i32,

    // ── Unidades físicas ───────────────────────────────────────────────────
    /// 1 unidad de longitud en cm (por defecto: 1 kpc/h = 3.086e21 cm).
    pub unit_length_in_cm: f64,
    /// 1 unidad de masa en gramos (por defecto: 1 M_sun/h = 1.989e33 g).
    pub unit_mass_in_g: f64,
    /// 1 unidad de velocidad en cm/s (por defecto: 1 km/s = 1e5 cm/s).
    pub unit_velocity_in_cm_per_s: f64,
}

impl Default for Gadget4Header {
    fn default() -> Self {
        Self {
            num_part_this_file: [0; 6],
            num_part_total: [0; 6],
            num_part_total_hw: [0; 6],
            mass_table: [0.0; 6],
            time: 1.0,
            redshift: 0.0,
            box_size: 0.0,
            omega_m: 0.3,
            omega_lambda: 0.7,
            omega_baryon: 0.045,
            hubble_param: 0.7,
            num_files_per_snapshot: 1,
            flag_sfr: 0,
            flag_cooling: 0,
            flag_feedback: 0,
            flag_stellar_age: 0,
            flag_metals: 0,
            flag_entropy_ics: 0,
            flag_double_precision: 1,
            flag_ic_info: 0,
            unit_length_in_cm: KPC_IN_CM,
            unit_mass_in_g: MSUN_IN_G,
            unit_velocity_in_cm_per_s: KMS_IN_CMS,
        }
    }
}

impl Gadget4Header {
    /// Crea un header para un snapshot de N-body puro (solo PartType1 = DM).
    pub fn for_nbody(
        n_dm: usize,
        time: f64,
        box_size: f64,
        omega_m: f64,
        omega_lambda: f64,
        hubble_param: f64,
    ) -> Self {
        let mut h = Self::default();
        h.num_part_this_file[1] = n_dm as i64;
        h.num_part_total[1] = n_dm as i64;
        h.time = time;
        h.redshift = if time > 0.0 { 1.0 / time - 1.0 } else { 0.0 };
        h.box_size = box_size;
        h.omega_m = omega_m;
        h.omega_lambda = omega_lambda;
        h.hubble_param = hubble_param;
        h
    }

    /// Crea un header para un snapshot con gas (PartType0) y DM (PartType1).
    #[allow(clippy::too_many_arguments)]
    pub fn for_sph(
        n_gas: usize,
        n_dm: usize,
        time: f64,
        box_size: f64,
        omega_m: f64,
        omega_lambda: f64,
        omega_baryon: f64,
        hubble_param: f64,
    ) -> Self {
        let mut h = Self::default();
        h.num_part_this_file[0] = n_gas as i64;
        h.num_part_total[0] = n_gas as i64;
        h.num_part_this_file[1] = n_dm as i64;
        h.num_part_total[1] = n_dm as i64;
        h.time = time;
        h.redshift = if time > 0.0 { 1.0 / time - 1.0 } else { 0.0 };
        h.box_size = box_size;
        h.omega_m = omega_m;
        h.omega_lambda = omega_lambda;
        h.omega_baryon = omega_baryon;
        h.hubble_param = hubble_param;
        h.flag_sfr = 1;
        h.flag_cooling = 1;
        h
    }

    /// Suma total de partículas de todos los tipos.
    pub fn total_particles(&self) -> i64 {
        self.num_part_total.iter().sum()
    }

    /// Velocidad Hubble H(a) en km/s/Mpc (aproximación ΛCdM plana).
    pub fn hubble_of_a(&self, a: f64) -> f64 {
        let h100 = 100.0 * self.hubble_param;
        let om = self.omega_m;
        let ol = self.omega_lambda;
        h100 * (om / (a * a * a) + ol).sqrt()
    }
}

// ── Escritura / lectura HDF5 ──────────────────────────────────────────────

#[cfg(feature = "hdf5")]
pub use hdf5_impl::{read_gadget4_header, write_gadget4_header};

#[cfg(feature = "hdf5")]
mod hdf5_impl {
    use super::Gadget4Header;
    use crate::error::SnapshotError;
    use hdf5::Group;
    use ndarray::arr1;

    /// Escribe todos los atributos GADGET-4 en el grupo `Header` de un archivo HDF5.
    pub fn write_gadget4_header(
        header_group: &Group,
        h: &Gadget4Header,
    ) -> Result<(), SnapshotError> {
        macro_rules! attr_i64_arr {
            ($name:expr, $val:expr) => {
                header_group
                    .new_attr_builder()
                    .with_data(&$val)
                    .create($name)?;
            };
        }
        macro_rules! attr_f64 {
            ($name:expr, $val:expr) => {
                header_group
                    .new_attr_builder()
                    .with_data(&arr1(&[$val]))
                    .create($name)?;
            };
        }
        macro_rules! attr_i32 {
            ($name:expr, $val:expr) => {
                header_group
                    .new_attr_builder()
                    .with_data(&arr1(&[$val as i32]))
                    .create($name)?;
            };
        }
        macro_rules! attr_f64_arr {
            ($name:expr, $val:expr) => {
                header_group
                    .new_attr_builder()
                    .with_data(&$val)
                    .create($name)?;
            };
        }

        attr_i64_arr!("NumPart_ThisFile", h.num_part_this_file);
        attr_i64_arr!("NumPart_Total", h.num_part_total);
        attr_i64_arr!("NumPart_Total_HW", h.num_part_total_hw);
        attr_f64_arr!("MassTable", h.mass_table);

        attr_f64!("Time", h.time);
        attr_f64!("Redshift", h.redshift);
        attr_f64!("BoxSize", h.box_size);
        attr_f64!("Omega0", h.omega_m);
        attr_f64!("OmegaLambda", h.omega_lambda);
        attr_f64!("OmegaBaryon", h.omega_baryon);
        attr_f64!("HubbleParam", h.hubble_param);

        attr_i32!("NumFilesPerSnapshot", h.num_files_per_snapshot);

        attr_i32!("Flag_Sfr", h.flag_sfr);
        attr_i32!("Flag_Cooling", h.flag_cooling);
        attr_i32!("Flag_Feedback", h.flag_feedback);
        attr_i32!("Flag_StellarAge", h.flag_stellar_age);
        attr_i32!("Flag_Metals", h.flag_metals);
        attr_i32!("Flag_Entropy_ICs", h.flag_entropy_ics);
        attr_i32!("Flag_DoublePrecision", h.flag_double_precision);
        attr_i32!("Flag_IC_Info", h.flag_ic_info);

        attr_f64!("UnitLength_in_cm", h.unit_length_in_cm);
        attr_f64!("UnitMass_in_g", h.unit_mass_in_g);
        attr_f64!("UnitVelocity_in_cm_per_s", h.unit_velocity_in_cm_per_s);

        Ok(())
    }

    /// Lee los atributos GADGET-4 desde el grupo `Header` de un archivo HDF5.
    pub fn read_gadget4_header(header_group: &Group) -> Result<Gadget4Header, SnapshotError> {
        fn read_f64(g: &Group, name: &str) -> Result<f64, crate::error::SnapshotError> {
            let v: Vec<f64> = g.attr(name)?.read_raw()?;
            Ok(v[0])
        }
        fn read_i32(g: &Group, name: &str) -> Result<i32, crate::error::SnapshotError> {
            let v: Vec<i32> = g.attr(name)?.read_raw()?;
            Ok(v[0])
        }
        fn read_i64_arr6(g: &Group, name: &str) -> Result<[i64; 6], crate::error::SnapshotError> {
            let v: Vec<i64> = g.attr(name)?.read_raw()?;
            let mut arr = [0i64; 6];
            for (i, &x) in v.iter().take(6).enumerate() {
                arr[i] = x;
            }
            Ok(arr)
        }
        fn read_f64_arr6(g: &Group, name: &str) -> Result<[f64; 6], crate::error::SnapshotError> {
            let v: Vec<f64> = g.attr(name)?.read_raw()?;
            let mut arr = [0.0f64; 6];
            for (i, &x) in v.iter().take(6).enumerate() {
                arr[i] = x;
            }
            Ok(arr)
        }

        Ok(Gadget4Header {
            num_part_this_file: read_i64_arr6(header_group, "NumPart_ThisFile")?,
            num_part_total: read_i64_arr6(header_group, "NumPart_Total")?,
            num_part_total_hw: read_i64_arr6(header_group, "NumPart_Total_HW")
                .unwrap_or([0; 6]),
            mass_table: read_f64_arr6(header_group, "MassTable")?,
            time: read_f64(header_group, "Time")?,
            redshift: read_f64(header_group, "Redshift")?,
            box_size: read_f64(header_group, "BoxSize")?,
            omega_m: read_f64(header_group, "Omega0")?,
            omega_lambda: read_f64(header_group, "OmegaLambda")?,
            omega_baryon: read_f64(header_group, "OmegaBaryon").unwrap_or(0.045),
            hubble_param: read_f64(header_group, "HubbleParam")?,
            num_files_per_snapshot: read_i32(header_group, "NumFilesPerSnapshot")?,
            flag_sfr: read_i32(header_group, "Flag_Sfr")?,
            flag_cooling: read_i32(header_group, "Flag_Cooling")?,
            flag_feedback: read_i32(header_group, "Flag_Feedback")?,
            flag_stellar_age: read_i32(header_group, "Flag_StellarAge")?,
            flag_metals: read_i32(header_group, "Flag_Metals")?,
            flag_entropy_ics: read_i32(header_group, "Flag_Entropy_ICs")?,
            flag_double_precision: read_i32(header_group, "Flag_DoublePrecision")?,
            flag_ic_info: read_i32(header_group, "Flag_IC_Info").unwrap_or(0),
            unit_length_in_cm: read_f64(header_group, "UnitLength_in_cm")?,
            unit_mass_in_g: read_f64(header_group, "UnitMass_in_g")?,
            unit_velocity_in_cm_per_s: read_f64(header_group, "UnitVelocity_in_cm_per_s")?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_default_values() {
        let h = Gadget4Header::default();
        assert_eq!(h.num_part_total, [0; 6]);
        assert_eq!(h.flag_double_precision, 1);
        assert!((h.unit_length_in_cm - KPC_IN_CM).abs() < 1.0);
        assert!((h.unit_velocity_in_cm_per_s - KMS_IN_CMS).abs() < 1.0);
    }

    #[test]
    fn header_for_nbody() {
        let h = Gadget4Header::for_nbody(1024, 0.5, 100.0, 0.3, 0.7, 0.7);
        assert_eq!(h.num_part_total[1], 1024);
        assert_eq!(h.num_part_total[0], 0, "sin gas");
        assert!((h.redshift - 1.0).abs() < 1e-10, "z = 1/0.5 - 1 = 1");
        assert!((h.time - 0.5).abs() < 1e-10);
    }

    #[test]
    fn header_for_sph() {
        let h = Gadget4Header::for_sph(512, 2048, 0.25, 100.0, 0.3, 0.7, 0.045, 0.7);
        assert_eq!(h.num_part_total[0], 512, "gas");
        assert_eq!(h.num_part_total[1], 2048, "dm");
        assert_eq!(h.flag_sfr, 1);
        assert_eq!(h.flag_cooling, 1);
        assert!((h.redshift - 3.0).abs() < 1e-10, "z = 1/0.25 - 1 = 3");
    }

    #[test]
    fn total_particles_sum() {
        let mut h = Gadget4Header::default();
        h.num_part_total = [100, 200, 0, 0, 0, 0];
        assert_eq!(h.total_particles(), 300);
    }

    #[test]
    fn hubble_of_a_present_day() {
        let h = Gadget4Header::for_nbody(0, 1.0, 100.0, 0.3, 0.7, 0.7);
        let h0 = h.hubble_of_a(1.0);
        // H(a=1) = 100 × 0.7 × sqrt(0.3 + 0.7) = 70 km/s/Mpc
        assert!((h0 - 70.0).abs() < 0.01, "H(a=1) ≈ 70 km/s/Mpc: {h0}");
    }

    #[test]
    fn header_serializes_to_json() {
        let h = Gadget4Header::for_nbody(64, 0.5, 50.0, 0.3, 0.7, 0.7);
        let s = serde_json::to_string(&h).unwrap();
        let h2: Gadget4Header = serde_json::from_str(&s).unwrap();
        assert_eq!(h2.num_part_total[1], 64);
        assert!((h2.time - 0.5).abs() < 1e-12);
    }

    #[test]
    fn unit_constants_reasonable() {
        // 1 kpc ≈ 3.086e21 cm
        assert!((KPC_IN_CM / 3.086e21 - 1.0).abs() < 0.01);
        // 1 M_sun ≈ 1.989e33 g
        assert!((MSUN_IN_G / 1.989e33 - 1.0).abs() < 0.01);
        // 1 km/s = 1e5 cm/s
        assert!((KMS_IN_CMS - 1e5).abs() < 1.0);
    }

    #[cfg(all(test, feature = "hdf5"))]
    mod hdf5_tests {
        use super::super::{read_gadget4_header, write_gadget4_header};
        use super::*;

        #[test]
        fn write_read_roundtrip() {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("test.hdf5");
            let file = hdf5::File::create(&path).unwrap();
            let hdr_group = file.create_group("Header").unwrap();

            let h = Gadget4Header::for_nbody(512, 0.5, 100.0, 0.31, 0.69, 0.677);
            write_gadget4_header(&hdr_group, &h).unwrap();
            drop(hdr_group);
            drop(file);

            let file2 = hdf5::File::open(&path).unwrap();
            let hdr2 = file2.group("Header").unwrap();
            let h2 = read_gadget4_header(&hdr2).unwrap();

            assert_eq!(h2.num_part_total[1], 512);
            assert!((h2.time - 0.5).abs() < 1e-12);
            assert!((h2.omega_m - 0.31).abs() < 1e-12);
            assert_eq!(h2.flag_double_precision, 1);
            assert!((h2.unit_length_in_cm - KPC_IN_CM).abs() < 1.0);
        }
    }
}
