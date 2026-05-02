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

Con **`solver = "barnes_hut"`** y **varios rangos MPI**, el motor usa por defecto una
descomposición **SFC (curva de Hilbert 3D)** y nodos **LET** remotos: evita reunir
todas las partículas en cada proceso salvo que actives el fallback explícito.

```toml
[gravity]
solver = "barnes_hut"

[performance]
# Opcional: fuerza el patrón antiguo tipo Allgather (depuración / comparación)
force_allgather_fallback = false
```

Cosmología **periódica** (`cosmology.periodic = true`) exige **`pm`** o **`tree_pm`**:
el árbol Barnes–Hut no implementa imágenes periódicas en el corto alcance; véase el mensaje
de error del CLI si combinas BH + caja periódica.

**Integrador jerárquico + MPI:** hay rutas SFC+LET para BH Newtoniano y para BH cosmológico
**aperiódico** (`periodic = false`); la combinación exacta depende de `[timestep]` y
`[cosmology]` — revisa [architecture.md](architecture.md) y los runbooks MPI si necesitas un modo concreto.

**Rutas legacy** (`use_distributed_tree`, slabs 1D en **x**, intercambio impar/par de halos)
siguen disponibles para compatibilidad cuando se desactiva el path SFC+LET por defecto;
suelen requerir `use_distributed_tree = true` y, según el modo, `use_sfc` / slab.

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

## Integrador cosmológico: timestep adaptativo (Phase 49)

El integrador cosmológico KDK usa la convención de momentum canónico QKSL:
`p = a²ẋ_c`, drift = `∫dt/a²`, kick = `∫dt/a`. El coupling correcto al
solver gravitacional es `g_cosmo = G·a³` (Phase 45). Para evoluciones largas
(a > 0.1) con un timestep fijo se produce inestabilidad numérica. La solución
es usar `adaptive_dt_cosmo`.

### API

```rust
use gadget_ng_core::adaptive_dt_cosmo;
use gadget_ng_core::cosmology::{gravity_coupling_qksl, CosmologyParams};
use gadget_ng_integrators::{leapfrog_cosmo_kdk_step, CosmoFactors};

let c = CosmologyParams::new(omega_m, omega_lambda, h0);
let softening = 0.005;
let eta_grav   = 0.025;  // fracción gravitacional (Quinn+1997)
let alpha_h    = 0.025;  // fracción del tiempo de Hubble
let dt_max     = 0.005;  // límite superior explícito

let mut a = a_init;
// Calcular aceleración inicial (primer paso):
let mut acc_scratch = vec![Vec3::zero(); n_particles];
compute_accelerations(&particles, gravity_coupling_qksl(G, a), &mut acc_scratch);

loop {
    if a >= a_target { break; }
    let acc_max = acc_scratch.iter().map(|v| v.norm()).fold(0.0_f64, f64::max);
    let dt = adaptive_dt_cosmo(c, a, acc_max, softening, eta_grav, alpha_h, dt_max);
    let g_cosmo = gravity_coupling_qksl(G, a);
    let (drift, kh, kh2) = c.drift_kick_factors(a, dt);
    let cf = CosmoFactors { drift, kick_half: kh, kick_half2: kh2 };
    a = c.advance_a(a, dt);
    leapfrog_cosmo_kdk_step(&mut particles, cf, &mut acc_scratch, |ps, out| {
        compute_accelerations(ps, g_cosmo, out);
    });
}
```

### Criterios de timestep

| Criterio | Fórmula | Parámetro típico |
|----------|---------|------------------|
| Gravitacional | `η·√(ε/|a_max|)` | `eta_grav = 0.025` |
| Hubble | `α_H/H(a)` | `alpha_h = 0.025` |
| Explícito | `dt_max` | depende del problema |

El timestep efectivo es `min(dt_grav, dt_hub, dt_max)`.

### Nota sobre consistencia de unidades

En los tests de regresión (G=1, H₀=0.1, box=1, N=32), la condición de
consistencia cosmológica `H₀ = √(8πGρ̄Ω_m/3)` NO se satisface. Esto implica
que el P(k) de la simulación no sigue exactamente D²(a) para evoluciones
largas — el streaming inicial (codificado en vel_factor = a²·f·H·Ψ) domina
sobre la respuesta gravitacional. Para simulaciones físicas, usar unidades
calibradas (`UnitsSection.enabled = true`) donde `G_KPC_MSUN_KMPS` asegura
consistencia con los parámetros cosmológicos.

---

## Unidades físicamente consistentes (Phase 50)

### El problema de consistencia

Los tests históricos usan `G = 1.0` y `H₀ = 0.1`, pero la ecuación de
Friedmann requiere que para una caja unitaria (ρ̄_m = 1):

```text
H₀² = 8π·G·ρ̄_m·Ω_m / 3
⟹  G_consistente = 3·Ω_m·H₀² / (8π)
```

Con `Ω_m = 0.315` y `H₀ = 0.1`:

```text
G_consistente ≈ 3.76×10⁻⁴     (no 1.0)
```

El ratio efectivo `(4πGρ̄)/H₀²` con `G=1` es **2660× mayor** que el valor
correcto `(3/2)Ω_m = 0.4725`. Esto hace que para evoluciones largas la
simulación no pueda reproducir el factor de crecimiento D(a) analítico.

### API: `g_code_consistent` y `cosmo_consistency_error`

```rust
use gadget_ng_core::{g_code_consistent, cosmo_consistency_error};

// G correcto para H₀=0.1, Ω_m=0.315, caja unitaria (ρ̄_m=1):
let g = g_code_consistent(0.315, 0.1);   // ≈ 3.76e-4

// Verificar consistencia de parámetros existentes:
let err = cosmo_consistency_error(g, 0.315, 0.1, 1.0);
assert!(err < 1e-10);   // consistencia exacta

// Diagnóstico de parámetros legacy:
let err_legacy = cosmo_consistency_error(1.0, 0.315, 0.1, 1.0);
// err_legacy ≈ 2659.5 → G_legacy 2660× inconsistente
```

### Uso en la simulación

```rust
use gadget_ng_core::{g_code_consistent, cosmology::gravity_coupling_qksl};

let omega_m = 0.315;
let h0_code = 0.1;
let g_phys = g_code_consistent(omega_m, h0_code);

// En el loop de integración:
let g_cosmo = gravity_coupling_qksl(g_phys, a);   // g_phys · a³
// kick: Δp = g_cosmo × F_PM × kick_factor
```

### Relación con UnitsSection

Cuando `UnitsSection.enabled = true`, `effective_g()` calcula el G interno
a partir de `G_KPC_MSUN_KMPS × mass_in_msun / length_in_kpc / velocity_in_km_s²`.
La función `g_code_consistent` es el análogo directo para simulaciones en
unidades de código puras (sin conversión a kpc/Msun/km·s⁻¹).

---

## G auto-consistente en el motor de producción (Phase 51)

### Activar con `auto_g = true`

```toml
[cosmology]
enabled      = true
auto_g       = true      # ← nuevo en Phase 51
omega_m      = 0.315
omega_lambda = 0.685
h0           = 0.1
a_init       = 0.02
```

Con `auto_g = true`, el motor calcula automáticamente:

```
G = 3 × Ω_m × H₀² / (8π) ≈ 3.76×10⁻⁴
```

y lo usa en toda la integración. El campo `simulation.gravitational_constant`
se ignora para el modo cosmológico.

### Advertencia automática de inconsistencia

Sin `auto_g`, si `simulation.gravitational_constant` difiere más de 1 % del
valor Friedmann-consistente, el motor emite en stderr:

```
[gadget-ng] ADVERTENCIA: G (1.0000e0) inconsistente con cosmología
(265854.9% fuera de G_consistente=3.7600e-4).
Usa [cosmology] auto_g = true para corregir automáticamente.
```

### Jerarquía de prioridad de G

| Prioridad | Condición | G efectiva |
|-----------|-----------|-----------|
| 1 (máxima) | `units.enabled = true` | `G_KPC_MSUN_KMPS × mass/length/v²` |
| 2 | `cosmology.auto_g = true` | `3·Ω_m·H₀²/(8π)` |
| 3 (fallback) | — | `simulation.gravitational_constant` |

### Diagnóstico en código

```rust
if let Some((g_consistent, rel_err)) = cfg.cosmo_g_diagnostic() {
    if rel_err > 0.01 {
        eprintln!("G inconsistente: {:.1}%", rel_err * 100.0);
    }
}
```

### Limitaciones de resolución

Para una verificación cuantitativa de `P(k)/P₀(k) ≈ D²(a)` se necesita
resolución suficiente (`N ≥ 64`) para reducir la varianza estadística del
espectro de potencias. En debug con `N = 8`, el ratio mediano puede diferir
hasta un 50 % de D²(a) solo por ruido de varianza cósmica con pocos modos.

En modo `release` con `N ≥ 64` y múltiples semillas, el error en D²(a) cae
por debajo del 10 %.

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

---

## Función de masa de halos: Press-Schechter y Sheth-Tormen (Phase 52)

### Motivación

La función de masa de halos (HMF) `dn/d ln M` es una de las predicciones más
importantes de la cosmología: da el número comoving de halos por unidad de
volumen y por intervalo logarítmico de masa. Es la base para comparar con
catálogos de halos observacionales (cúmulos de galaxias, lentes gravitacionales,
encuestas SZ) y para validar la formación de estructura de gadget-ng.

### Física implementada

**σ(M, z)** — varianza de fluctuaciones de densidad:

```
σ²(M, z) = [D(z)/D(0)]² × (1/2π²) ∫₀^∞ k³ P(k, z=0) W²(kR) dk
```

donde `R = (3M / 4πρ̄_m)^{1/3}` es el radio de Lagrange y `W(x)` el filtro
top-hat esférico. La normalización usa `P(k) = amp² × k^n_s × T²_EH(k)` con
`amp = sigma8 / sqrt(sigma_sq_unit)` (convenio del IC Zel'dovich).

**Press-Schechter (1974)**:

```
dn/d ln M = (ρ̄_m/M) × |d ln σ⁻¹/d ln M| × f_PS(σ)
f_PS(σ) = √(2/π) × (δ_c/σ) × exp(−δ_c²/(2σ²))        δ_c = 1.686
```

**Sheth-Tormen (1999)** (colapso elipsoidal):

```
f_ST(σ) = A × √(2a/π) × (δ_c/σ) × [1 + (σ²/(a δ_c²))^p] × exp(−a δ_c²/(2σ²))
          a = 0.707,  p = 0.3,  A = 0.3222
```

ST predice más halos masivos que PS (~20–50% más a M > 10¹⁴ M_sun/h) y
es un mejor ajuste a simulaciones N-body.

### Unidades

| Cantidad  | Unidad                        |
|-----------|-------------------------------|
| Masa      | M_sun/h                       |
| Longitud  | Mpc/h                         |
| ρ̄_m      | (M_sun/h)/(Mpc/h)³ = Ω_m × 2.775×10¹¹ |
| dn/d ln M | h³/Mpc³                       |

### API

```rust
use gadget_ng_analysis::halo_mass_function::{
    HmfParams, mass_function_table, sigma_m, total_halo_density,
};

// Parámetros de Planck 2018 (preset)
let params = HmfParams::planck2018();

// σ(M = 10¹³ M_sun/h) a z = 1
let sigma = sigma_m(1e13, &params, 1.0);
println!("σ(10¹³ M_sun/h, z=1) = {sigma:.4}");

// Tabla HMF (30 bins log-espaciados entre 10¹⁰ y 10¹⁵ M_sun/h a z=0)
let table = mass_function_table(&params, 1e10, 1e15, 30, 0.0);
for bin in &table {
    println!(
        "log10(M) = {:.2}  σ = {:.4}  dn/dlnM_PS = {:.3e}  dn/dlnM_ST = {:.3e}",
        bin.log10_m, bin.sigma, bin.n_ps, bin.n_st
    );
}

// Densidad total integrada
let (n_ps, n_st) = total_halo_density(&table);
println!("n_total: PS={n_ps:.3e}  ST={n_st:.3e}  [h³/Mpc³]");
```

### Cosmología personalizada

```rust
use gadget_ng_analysis::halo_mass_function::HmfParams;

let params = HmfParams {
    omega_m: 0.30,
    omega_lambda: 0.70,
    h: 0.70,
    sigma8: 0.80,
    n_s: 0.96,
    omega_b: 0.046,
    t_cmb: 2.7255,
};
```

### Valores de referencia (Planck 2018, z=0)

| log₁₀(M) | σ(M)   | dn/d ln M — PS | dn/d ln M — ST |
|-----------|--------|----------------|----------------|
| 10.0      | 3.74   | 3.0×10⁻¹       | 2.3×10⁻¹       |
| 12.0      | 2.05   | 5.0×10⁻³       | 3.4×10⁻³       |
| 14.0      | 0.86   | 3.0×10⁻⁵       | 2.5×10⁻⁵       |

n(>10¹⁴ M_sun/h) ≈ 3×10⁻⁵ h³/Mpc³ — coherente con catálogos ACT/SPT/eROSITA.

### Comparación con catálogos FoF

El módulo `gadget_ng_analysis::fof` provee halos FoF con masa `M = N_part × m_part`.
Para comparar cuantitativamente con la HMF analítica se recomienda:

- Usar cajas de ≥100 Mpc/h con N ≥ 128³ partículas.
- Resolver P(k) de la simulación con `gadget_ng_analysis::power_spectrum`.
- Normalizar la HMF al mismo σ₈ y cosmología que la simulación.

La comparación cualitativa (Test 7 de Phase 52) verifica que las masas FoF
caen en el rango donde σ(M) < 10 (halos colapsados).

---

## Perfiles de halos NFW y relación c(M) (Phase 53)

### Perfil NFW

El perfil de densidad de Navarro, Frenk & White (1996/1997):

```
ρ(r) = ρ_s / [(r/r_s)(1 + r/r_s)²]
```

Propiedades analíticas:
- `r ≪ r_s`: ρ ∝ r⁻¹ (interior plano)
- `r = r_s`: ρ ∝ r⁻² (punto de inflexión)
- `r ≫ r_s`: ρ ∝ r⁻³ (caída exterior)

### API

```rust
use gadget_ng_analysis::nfw::{
    NfwProfile, concentration_duffy2008, rho_crit_z, r200_from_m200,
    measure_density_profile,
};

let omega_m = 0.315;
let z = 0.0;
let m200 = 1e14_f64; // M_sun/h

// ρ_crit(z=0) en (M_sun/h)/(Mpc/h)³
let rho_c = rho_crit_z(omega_m, 0.685, z);

// Concentración Duffy+2008
let c = concentration_duffy2008(m200, z);

// Construir perfil NFW
let profile = NfwProfile::from_m200_c(m200, c, rho_c);
println!("r_s = {:.4} Mpc/h   ρ_s = {:.3e}", profile.r_s, profile.rho_s);

// Propiedades
let r200 = r200_from_m200(m200, rho_c);
println!("M(<r_200) = {:.4e}", profile.mass_enclosed(r200));
println!("ρ(r_s)   = {:.4e}", profile.density(profile.r_s));

// Medir perfil desde radios de partículas
let radii: Vec<f64> = vec![0.1, 0.3, 0.5, 0.8]; // Mpc/h
let m_part = 1e10_f64; // M_sun/h
let bins = measure_density_profile(&radii, m_part, 0.05*r200, r200, 10, Some(&profile));
for bin in &bins {
    println!("r={:.3}  n={}  ρ_med/ρ_NFW={:.2}", bin.r, bin.n_part, bin.rho/bin.rho_nfw.max(1e-99));
}
```

### Relación c(M, z) — Duffy et al. (2008)

```
c_200(M, z) = 5.71 × (M / 2×10¹² M_sun/h)^{-0.084} × (1+z)^{-0.47}
```

| log₁₀(M [M_sun/h]) | c (z=0) | r₂₀₀ [Mpc/h] | r_s [Mpc/h] |
|---------------------|---------|--------------|-------------|
| 12.0 (Vía Láctea)   | 6.05    | 0.163        | 0.027       |
| 14.0 (cúmulo)       | 4.11    | 0.755        | 0.184       |
| 15.0 (super-cúmulo) | 3.39    | 1.626        | 0.480       |

### Ajuste desde datos de partículas

```rust
use gadget_ng_analysis::nfw::fit_nfw_concentration;

// bins: medidos con measure_density_profile
// c_min, c_max: rango de búsqueda de concentración
let result = fit_nfw_concentration(&bins, m200, rho_c, 1.0, 20.0, 50);
if let Some(fit) = result {
    let c_fit = r200 / fit.profile.r_s;
    println!("c_fit = {c_fit:.2}  χ²_red = {:.4}", fit.chi2_red);
}
```


---

## Validación cuantitativa D²(a) — Phase 54

Phase 54 verifica que el crecimiento lineal del espectro de potencias satisface
cuantitativamente `P(k,a)/P(k,a₀) = [D(a)/D(a₀)]²` cuando se usa G consistente.

### Configuración

```toml
[cosmology]
enabled       = true
omega_m       = 0.315
omega_lambda  = 0.685
h0            = 0.1
a_init        = 0.02
auto_g        = true   # G = 3·Ω_m·H₀² / 8π ≈ 3.76e-4
```

### Ejecutar

```bash
# Correr todos los resolutions (N=64, 128, 256) en modo release
./experiments/nbody/phase54_growth_factor_validation/run_phase54.sh

# Omitir N=256 para prueba rápida (~30 s)
PHASE54_SKIP_N256=1 ./experiments/nbody/phase54_growth_factor_validation/run_phase54.sh
```

### Tolerancias verificadas

| N    | Error mediano D²(a) | Bins utilizados |
|------|--------------------:|-----------------|
| 64³  | < 30%               | k < k_nyq/2     |
| 128³ | < 15%               | k < k_nyq/2     |
| 256³ | < 10%               | k < k_nyq/2     |

El error decrece monótonamente con N (test de convergencia).

---

## Comparación FoF vs HMF — Phase 55

Phase 55 evoluciona a z=0 y compara el espectro de masas FoF con la HMF
analítica Press-Schechter / Sheth-Tormen.

### Parámetros físicos

| Parámetro      | Valor                  |
|----------------|------------------------|
| BOX            | 300 Mpc/h              |
| Cosmología     | Planck 2018 (σ₈=0.811) |
| FoF b          | 0.2                    |
| min_particles  | 20                     |
| G              | G_consistent ≈ 3.76e-4 |

### Masa de partícula por resolución

```rust
// m_part = Ω_m × ρ_crit_H2 × BOX_MPC_H³ / N_total
// con RHO_CRIT_H2 = 2.775e11 (M_sun/h)/(Mpc/h)³

use gadget_ng_analysis::{mass_function_table, HmfParams, RHO_CRIT_H2};
let m_part_n64  = 0.315 * RHO_CRIT_H2 * 300.0_f64.powi(3) / 64.0_f64.powi(3); // 9.0e12 M_sun/h
let m_part_n128 = 0.315 * RHO_CRIT_H2 * 300.0_f64.powi(3) / 128.0_f64.powi(3); // 1.1e12
let m_part_n256 = 0.315 * RHO_CRIT_H2 * 300.0_f64.powi(3) / 256.0_f64.powi(3); // 1.4e11
```

### Ejecutar

```bash
./experiments/nbody/phase55_fof_vs_hmf/run_phase55.sh

# Omitir N=256
PHASE55_SKIP_N256=1 ./experiments/nbody/phase55_fof_vs_hmf/run_phase55.sh
```

### Comparación analítica

```rust
use gadget_ng_analysis::{mass_function_table, HmfParams};

let params = HmfParams {
    omega_m: 0.315, omega_lambda: 0.685, omega_b: 0.049,
    h: 0.674, sigma8: 0.811, n_s: 0.965, t_cmb: 2.7255,
};
// dn/dlnM en h³/Mpc³ para 15 bins entre m_min y m_max
let table = mass_function_table(&params, 1e12, 1e15, 15, 0.0);
for bin in &table {
    println!("M={:.2e}: ST={:.3e}  PS={:.3e}", bin.m_msun_h, bin.n_st, bin.n_ps);
}
```

