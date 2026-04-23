# Phase 74 — Output HDF5 con atributos estándar GADGET-4

**Fecha**: Abril 2026  
**Crate**: `gadget-ng-io`  
**Módulo**: `src/gadget4_attrs.rs`

---

## Motivación

El formato HDF5 de GADGET-4 es el estándar de facto para simulaciones cosmológicas de N-body.
Las herramientas `yt`, `pynbody`, `h5py`, `swiftsimio` y `Caesar` asumen una estructura
específica de atributos en el grupo `/Header`. Sin estos atributos, los archivos no pueden
cargarse directamente.

Esta phase completa el writer HDF5 existente (Phase 55) con **todos** los atributos del
estándar GADGET-4.

---

## Atributos Implementados

### Recuentos de partículas

| Atributo              | Tipo       | Descripción                              |
|-----------------------|------------|------------------------------------------|
| `NumPart_ThisFile`    | `[i64; 6]` | Partículas por tipo en este archivo      |
| `NumPart_Total`       | `[i64; 6]` | Total en toda la simulación              |
| `NumPart_Total_HW`    | `[i64; 6]` | Palabra alta para N > 2³²               |
| `MassTable`           | `[f64; 6]` | Masa fija por tipo (0 = individual)      |

### Cosmología

| Atributo          | Tipo    | Descripción                 |
|-------------------|---------|-----------------------------|
| `Time`            | `f64`   | Factor de escala a          |
| `Redshift`        | `f64`   | z = 1/a − 1                |
| `BoxSize`         | `f64`   | Lado del cubo (kpc/h)      |
| `Omega0`          | `f64`   | Ω_matter                   |
| `OmegaLambda`     | `f64`   | Ω_Λ                        |
| `OmegaBaryon`     | `f64`   | Ω_b *(nuevo en Phase 74)*  |
| `HubbleParam`     | `f64`   | h = H₀/100                 |

### Flags de física

| Atributo               | Tipo   | Descripción                                  |
|------------------------|--------|----------------------------------------------|
| `Flag_Sfr`             | `i32`  | Star formation                               |
| `Flag_Cooling`         | `i32`  | Enfriamiento radiativo                       |
| `Flag_Feedback`        | `i32`  | Feedback estelar                             |
| `Flag_StellarAge`      | `i32`  | Edades estelares                             |
| `Flag_Metals`          | `i32`  | Metalicidades                                |
| `Flag_Entropy_ICs`     | `i32`  | *(nuevo)* ICs en entropía vs energía interna |
| `Flag_DoublePrecision` | `i32`  | *(nuevo)* 1 = coordenadas en f64             |
| `Flag_IC_Info`         | `i32`  | *(nuevo)* tipo de IC (0=ninguno, 2=2LPT)    |

### Unidades físicas *(nuevas)*

| Atributo                        | Valor por defecto       |
|---------------------------------|-------------------------|
| `UnitLength_in_cm`              | 3.086e21 (1 kpc)        |
| `UnitMass_in_g`                 | 1.989e33 (1 M_sun)      |
| `UnitVelocity_in_cm_per_s`      | 1.0e5 (1 km/s)          |

---

## API Pública

```rust
pub struct Gadget4Header {
    pub num_part_this_file: [i64; 6],
    pub num_part_total: [i64; 6],
    pub num_part_total_hw: [i64; 6],
    pub mass_table: [f64; 6],
    pub time: f64, pub redshift: f64, pub box_size: f64,
    pub omega_m: f64, pub omega_lambda: f64, pub omega_baryon: f64,
    pub hubble_param: f64,
    pub num_files_per_snapshot: i32,
    pub flag_sfr: i32, pub flag_cooling: i32, pub flag_feedback: i32,
    pub flag_stellar_age: i32, pub flag_metals: i32,
    pub flag_entropy_ics: i32, pub flag_double_precision: i32,
    pub flag_ic_info: i32,
    pub unit_length_in_cm: f64,
    pub unit_mass_in_g: f64,
    pub unit_velocity_in_cm_per_s: f64,
}

// Constructores
impl Gadget4Header {
    pub fn for_nbody(n_dm, time, box_size, omega_m, omega_lambda, hubble_param) -> Self
    pub fn for_sph(n_gas, n_dm, ..., omega_baryon) -> Self
    pub fn total_particles(&self) -> i64
    pub fn hubble_of_a(&self, a: f64) -> f64   // H(a) en km/s/Mpc
}

// HDF5 I/O (feature = "hdf5")
pub fn write_gadget4_header(group: &hdf5::Group, h: &Gadget4Header) -> Result<()>
pub fn read_gadget4_header(group: &hdf5::Group) -> Result<Gadget4Header>
```

---

## Integración con el Writer Existente

El `Hdf5Writer` fue actualizado para delegar en `write_gadget4_header`, reemplazando
el código anterior que solo escribía un subconjunto de atributos.

---

## Tests

| Test                          | Verificación                                       |
|-------------------------------|----------------------------------------------------|
| `header_default_values`       | Flags correctos por defecto                        |
| `header_for_nbody`            | N_dm, z, a coherentes                             |
| `header_for_sph`              | N_gas, N_dm, flags de física activos              |
| `total_particles_sum`         | Suma de los 6 tipos                                |
| `hubble_of_a_present_day`     | H(a=1) = 70 km/s/Mpc para h=0.7                  |
| `header_serializes_to_json`   | Round-trip JSON completo                           |
| `unit_constants_reasonable`   | kpc en cm, M_sun en g, km/s en cm/s correctos     |
| `write_read_roundtrip` (HDF5) | Todos los atributos sobreviven write→read          |

---

## Compatibilidad

Con esta implementación, los snapshots HDF5 de gadget-ng son directamente legibles por:

- **yt** (`yt.load("snapshot.hdf5")`)
- **pynbody** (`pynbody.load("snapshot.hdf5")`)
- **h5py** (lectura directa de atributos)
- **GADGET-4** propio (para restart desde snapshots externos)
