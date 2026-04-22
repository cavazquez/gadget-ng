# Guía de usuario — gadget-ng

gadget-ng es una simulación N-body modular escrita en Rust.
Soporta múltiples solvers de gravedad, integración cosmológica, paralelismo
MPI y GPU, y diferentes formatos de salida.

---

## Tabla de contenidos

1. [Compilar e instalar](#compilar-e-instalar)
2. [Uso básico](#uso-básico)
3. [Estructura del archivo TOML](#estructura-del-archivo-toml)
4. [Solvers de gravedad](#solvers-de-gravedad)
5. [Sistema de unidades físicas](#sistema-de-unidades-físicas)
6. [Cosmología](#cosmología)
7. [Pasos temporales jerárquicos](#pasos-temporales-jerárquicos)
8. [Checkpointing y reanudación](#checkpointing-y-reanudación)
9. [GPU y paralelismo](#gpu-y-paralelismo)
10. [Árbol distribuido (MPI)](#árbol-distribuido-mpi)
11. [Formatos de salida](#formatos-de-salida)
12. [Benchmarks](#benchmarks)

---

## Compilar e instalar

```bash
# Construcción mínima (serial, sin GPU ni MPI):
cargo build --release

# Con GPU (wgpu, Vulkan/Metal/DX12):
cargo build --release --features gpu

# Con MPI (requiere libmpi-dev):
cargo build --release --features mpi

# Todas las características:
cargo build --release --features gpu,mpi,simd,bincode,hdf5
```

El binario queda en `target/release/gadget-ng`.

---

## Uso básico

```bash
# Verificar la configuración
gadget-ng config --config experimento.toml

# Ejecutar integración
gadget-ng stepping --config experimento.toml --out resultados/

# Ejecutar y guardar snapshot final
gadget-ng stepping --config experimento.toml --out resultados/ --snapshot

# Escribir solo el snapshot de condiciones iniciales
gadget-ng snapshot --config experimento.toml --out ic/

# Reanudar desde un checkpoint (ver sección checkpointing)
gadget-ng stepping --config experimento.toml --out resultados/ --resume resultados/
```

Con MPI:
```bash
mpirun -n 4 gadget-ng stepping --config experimento.toml --out resultados/
```

---

## Estructura del archivo TOML

Un archivo TOML mínimo:

```toml
[simulation]
particle_count = 512
box_size       = 1.0
dt             = 0.001
num_steps      = 100
softening      = 0.02
seed           = 42

[initial_conditions]
kind = "lattice"   # o "two_body"
```

Todas las demás secciones son opcionales; los valores omitidos toman
los defaults documentados a continuación.

---

## Solvers de gravedad

Configura el solver con la sección `[gravity]`.

```toml
[gravity]
solver       = "direct"       # "direct" | "barnes_hut" | "pm" | "tree_pm"
theta        = 0.5            # solo barnes_hut / tree_pm (aperture criterion)
pm_grid_size = 64             # solo pm / tree_pm (potencia de 2 recomendada)
r_split      = 0.0            # solo tree_pm (0.0 = automático: 2.5 × cell_size)
```

| Solver       | Complejidad  | Cuándo usar                         |
|--------------|-------------|--------------------------------------|
| `direct`     | O(N²)        | N < 1000, máxima precisión           |
| `barnes_hut` | O(N log N)   | N < 10⁶, buena precisión con θ≤0.5  |
| `pm`         | O(N log N)   | N grande, periodicidad importante    |
| `tree_pm`    | O(N log N)   | Combina precisión local (árbol) y largo alcance (PM) |

### Ejemplo Barnes-Hut

```toml
[gravity]
solver = "barnes_hut"
theta  = 0.4     # más estricto → más preciso, más lento

[performance]
deterministic = false  # activar Rayon
num_threads   = 8
```

### Ejemplo PM

```toml
[gravity]
solver       = "pm"
pm_grid_size = 128

[simulation]
particle_count = 10000
box_size       = 100.0
```

---

## Sistema de unidades físicas

Sin la sección `[units]`, gadget-ng usa unidades internas arbitrarias con
`G = simulation.gravitational_constant` (default 1.0).

Para simulaciones astrofísicas, añade `[units]` para que G se calcule
automáticamente a partir de `G = 4.3009×10⁻⁶ kpc Msun⁻¹ (km/s)²`.

```toml
[units]
enabled          = true
length_in_kpc    = 1.0       # 1 u.l. = 1 kpc
mass_in_msun     = 1.0e10    # 1 u.m. = 10¹⁰ M☉  (unidades GADGET clásicas)
velocity_in_km_s = 1.0       # 1 u.v. = 1 km/s
# G_int calculado = 4.3009e-6 × 1e10 / 1 / 1 = 4.3009e4 kpc (km/s)² / (10¹⁰ M☉)
```

Cuando `enabled = true`, el campo `meta.json` del snapshot incluye:
```json
"units": {
  "length_in_kpc": 1.0,
  "mass_in_msun": 1e10,
  "velocity_in_km_s": 1.0,
  "time_in_gyr": 0.97779,
  "g_internal": 43009.0
}
```

El campo `simulation.gravitational_constant` se ignora cuando `units.enabled = true`.

---

## Cosmología

La sección `[cosmology]` activa la integración del factor de escala `a(t)`
junto a las partículas (formulación de momentum canónico, estilo GADGET-4).

```toml
[cosmology]
enabled      = true
omega_m      = 0.3    # fracción de materia
omega_lambda = 0.7    # fracción de energía oscura
h0           = 0.1    # H₀ en unidades internas (≈ H₀ en km/s/kpc cuando v = km/s)
a_init       = 0.02   # factor de escala inicial (z=49 → a=0.02)
```

Con cosmología:
- Las posiciones son coordenadas comóviles.
- `velocity` almacena momentum canónico `p = a²·ẋ_c`.
- El snapshot reporta `redshift = 1/a − 1`.

---

## Pasos temporales jerárquicos

Inspirado en GADGET-4: cada partícula elige su propio paso temporal como
potencia de 2 basado en el criterio de Aarseth `dt_i = η √(ε / |aᵢ|)`.

```toml
[timestep]
hierarchical = true
eta          = 0.025   # parámetro de Aarseth (0.01–0.05)
max_level    = 6       # 2⁶ = 64 sub-pasos por paso base dt
```

Compatible con cosmología (activa factores drift/kick variables por nivel).

---

## Checkpointing y reanudación

Guarda el estado de la simulación periódicamente para poder reanudar
en caso de interrupción (ideal para particiones de tiempo en HPC).

```toml
[output]
checkpoint_interval = 100   # guardar checkpoint cada 100 pasos
```

El checkpoint se escribe en `<out>/checkpoint/` y contiene:
- `checkpoint.json` — paso completado, factor de escala `a`, hash del config.
- `particles.jsonl` — estado completo de las partículas.
- `hierarchical_state.json` — si se usa integrador jerárquico.

Para reanudar:
```bash
# Misma config, mismo --out; añade --resume apuntando al mismo out
gadget-ng stepping \
  --config experimento.toml \
  --out resultados/ \
  --resume resultados/
```

Si el archivo TOML cambió respecto al checkpoint, se emite una advertencia
pero la simulación continúa.

---

## GPU y paralelismo

### GPU (wgpu)

Requiere compilar con `--features gpu`:

```toml
[performance]
use_gpu = true   # fallback automático a CPU si no hay GPU disponible
```

El kernel WGSL calcula gravedad directa O(N²) en f32 (error ≈ 10⁻⁷ relativo).
Portátil: Vulkan, Metal, DX12, WebGPU.

### Rayon (multi-hilo)

Requiere `--features simd`:

```toml
[performance]
deterministic = false  # false → Rayon activo
num_threads   = 8      # None → detecta CPUs automáticamente
```

---

## Árbol distribuido (MPI)

El árbol distribuido reemplaza el `Allgather` global O(N) por halos
punto-a-punto O(N_halo), lo que permite simular N > memoria de un nodo.

```toml
[gravity]
solver = "barnes_hut"

[performance]
use_distributed_tree = true
halo_factor          = 0.5   # halo_width = 0.5 × slab_width
```

Limitaciones actuales:
- Solo slab 1D en el eje x (SFC pendiente en Hito 13).
- Solo funciona con `solver = "barnes_hut"` sin integrador jerárquico.

Ejecutar con MPI:
```bash
mpirun -n 8 gadget-ng stepping \
  --config experimento.toml \
  --out resultados/
```

---

## Formatos de salida

```toml
[output]
snapshot_format     = "jsonl"    # "jsonl" | "bincode" | "hdf5" | "msgpack" | "netcdf"
checkpoint_interval = 50         # 0 = desactivado
```

| Formato    | Feature necesaria | Interoperabilidad                     |
|------------|-------------------|----------------------------------------|
| `jsonl`    | (ninguna)         | Python/R/Julia/cualquier JSON reader  |
| `bincode`  | `bincode`         | Rust nativo, más rápido que JSONL     |
| `hdf5`     | `hdf5`            | h5py, Julia HDF5.jl, IDL              |
| `msgpack`  | `msgpack`         | Python msgpack, R RcppMsgPack          |
| `netcdf`   | `netcdf`          | xarray, netCDF4-python, NCDatasets.jl |

Cada snapshot es un directorio con:
- `meta.json` — metadatos (tiempo, redshift, box_size, unidades, provenance).
- `provenance.json` — versión, features, argumentos CLI, hash del config.
- Archivo de partículas (`particles.jsonl`, `particles.bin`, `snapshot.hdf5`, etc.).

---

## Análisis in-situ: P(k) corregido

El crate `gadget-ng-analysis` calcula el espectro de potencias `P(k)` directamente
sobre las posiciones de partículas con CIC + FFT 3D (deconvolución sinc).

### Medición básica

```rust
use gadget_ng_analysis::power_spectrum::power_spectrum;

let pk = power_spectrum(&positions, &masses, box_size, mesh);
// pk: Vec<PkBin { k, pk, n_modes }> ordenado por k
```

### Corrección absoluta A_grid·R(N)

El estimador interno `P_measured(k)` difiere del espectro físico por dos factores:

```text
P_measured(k) = A_grid(N) · R(N) · P_phys(k)

  A_grid(N) = 2·V² / N⁹     (Phase 34 — analítico)
  R(N)      = C · N^{-α}     (Phase 35/47 — calibrado numéricamente)
```

Para obtener `P_phys(k)` en (Mpc/h)³:

```rust
use gadget_ng_analysis::pk_correction::{correct_pk, RnModel};

// R(N) calibrado para N ∈ {8, 16, 32, 64, 128}
let model = RnModel::phase47_default();

// IMPORTANTE: box_mpc_h = None porque R(N) ya absorbe el factor de volumen.
let pk_phys = correct_pk(&pk_measured, box_internal, n_mesh, None, &model);
```

**Nota sobre unidades:** `R(N)` fue calibrado comparando `P_measured` (box interno = 1)
contra `P_cont` en (Mpc/h)³. La conversión ya está absorbida en `R`. Pasar
`box_mpc_h = Some(L)` aplicaría un factor `L³` adicional incorrecto.

### Con sustracción de shot noise

A alto k, el ruido de Poisson `P_shot = V/N_part` puede dominar la señal.
Sustráelo antes de aplicar la corrección:

```rust
use gadget_ng_analysis::pk_correction::correct_pk_with_shot_noise;

let pk_clean = correct_pk_with_shot_noise(
    &pk_measured,
    box_internal,
    n_mesh,
    None,           // box_mpc_h = None (misma razón que arriba)
    n_particles,
    &model,
);
// bins donde P_measured ≤ P_shot quedan en pk = 0.0
```

### Calibrar R(N) para un N personalizado

```rust
use gadget_ng_analysis::pk_correction::measure_rn;
use gadget_ng_core::EisensteinHuParams;

let eh = EisensteinHuParams::default();
let (r256, cv) = measure_rn(256, &[42, 137, 271, 314], 1.0, 100.0, 0.8, 0.965, &eh);
println!("R(256) = {r256:.6}  CV = {:.1}%", cv * 100.0);
```

La función genera ICs ZA in-process y mide P(k) sin necesitar una simulación completa.

### Valores de referencia Phase 47

| N | R(N) medido | CV | Fuente |
|---|-------------|-----|--------|
| 8 | 0.415382 | ~23% | Phase 35 |
| 16 | 0.139629 | ~14% | Phase 35 |
| 32 | 0.033752 | 3.6% | Phase 35/47 |
| 64 | 0.008834 | 1.4% | Phase 35/47 |
| 128 | **0.002252** | **1.0%** | **Phase 47** |

Fit log-log sobre {32,64,128}: `C=29.77, α=1.953`.

---

## Espectro no-lineal: Halofit (Takahashi+2012)

El módulo `gadget_ng_analysis::halofit` calcula el espectro de potencias
**no-lineal** a partir del lineal, usando el modelo de ajuste de
Takahashi et al. (2012) que mejora el Halofit original de Smith et al. (2003).

### Uso básico

```rust
use gadget_ng_analysis::halofit::{halofit_pk, p_linear_eh, HalofitCosmo};
use gadget_ng_core::{amplitude_for_sigma8, EisensteinHuParams};
use gadget_ng_core::cosmology::{growth_factor_d_ratio, CosmologyParams};

let cosmo = HalofitCosmo { omega_m0: 0.315, omega_de0: 0.685 };
let e = EisensteinHuParams { omega_m: 0.315, omega_b: 0.049, h: 0.674, t_cmb: 2.7255 };
let cp = CosmologyParams::new(0.315, 0.685, 0.1);
let amp = amplitude_for_sigma8(0.8, 0.965, &e);

// Espectro lineal a z=1 (a=0.5).
let a = 0.5;
let d = growth_factor_d_ratio(cp, a, 1.0);  // D(z=1)/D(z=0)
let p_lin_z1 = |k: f64| p_linear_eh(k, amp, 0.965, d, &e);

// Evaluar Halofit en varios k.
let k_vals = vec![0.05, 0.1, 0.3, 1.0, 3.0, 10.0];
let z = 1.0 / a - 1.0;  // z=1
let p_nl = halofit_pk(&k_vals, &p_lin_z1, &cosmo, z);

for (k, p) in &p_nl {
    let ratio = p / p_lin_z1(*k);
    println!("k={k:.2}: P_nl/P_lin = {ratio:.3}");
}
```

### Salida esperada (Planck18, z=0)

| k [h/Mpc] | P_nl/P_lin |
|-----------|-----------|
| 0.05      | 1.000     |
| 0.10      | 1.000     |
| 0.30      | 1.067     |
| 1.00      | 4.021     |
| 3.00      | 19.38     |
| 10.0      | 127.9     |

### Comparación con P_sim corregido

Para comparar el espectro simulado con Halofit:

```rust
use gadget_ng_analysis::pk_correction::{correct_pk, RnModel};
use gadget_ng_analysis::power_spectrum::power_spectrum;

// 1. Medir P(k) en la simulación.
let pk_raw = power_spectrum(&positions, &masses, box_internal, n_mesh);

// 2. Corregir con R(N).
let model = RnModel::phase47_default();
let pk_corr = correct_pk(&pk_raw, box_internal, n_mesh, None, &model);

// 3. Convertir k de código a h/Mpc.
let h = 0.674;
let box_mpc_h = 100.0;
let k_hmpc: Vec<f64> = pk_corr.iter().map(|b| b.k * h / box_mpc_h).collect();

// 4. Halofit a la misma z de la simulación.
let p_nl_pred = halofit_pk(&k_hmpc, &p_lin_fn, &cosmo, z_sim);

// 5. Comparar.
for (bin, (k, p_hf)) in pk_corr.iter().zip(p_nl_pred.iter()) {
    let ratio = bin.pk / p_hf;
    println!("k={k:.3} h/Mpc: P_sim/P_halofit = {ratio:.3}");
}
```

### Limitaciones

- Solo ΛCDM plano (w = −1, Ω_k = 0).
- Espectro de transferencia EH: el boost a k~0.3 h/Mpc puede estar
  subestimado ~40 % respecto a CAMB. Para comparaciones con observaciones
  usar CAMB/CLASS.
- Válido para z ≤ 10 y k ∈ [0.01, 30] h/Mpc (rango de calibración).

---

## Benchmarks

```bash
# Ejecutar todos los benchmarks
cargo bench --workspace

# Solo el solver Barnes-Hut
cargo bench -p gadget-ng-tree

# Solo el solver PM
cargo bench -p gadget-ng-pm

# Solo el solver TreePM
cargo bench -p gadget-ng-treepm

# Compilar sin ejecutar (verificar que compilan):
cargo bench --workspace --no-run
```

Los resultados HTML se generan en `target/criterion/`.
