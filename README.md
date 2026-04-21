# 🌌 gadget-ng

> Simulador **N-body cosmológico** en Rust, inspirado conceptualmente en la
> arquitectura y prácticas de [GADGET-4](https://wwwmpa.mpa-garching.mpg.de/gadget4/),
> sin compartir código ni historial git con el proyecto original.
>
> Cubre el pipeline completo de una corrida cosmológica contemporánea:
> **ICs 1LPT/2LPT con transferencia Eisenstein–Hu y normalización σ₈
> (`Legacy`/`Z0Sigma8`) → integrador leapfrog/Yoshida4 con factores de
> drift/kick cosmológicos → PM periódico / TreePM / Barnes–Hut / Direct, con
> versiones distribuidas (allreduce, slab alltoall, scatter-gather) → análisis
> in-situ (FoF, P(k)) → corrección absoluta de `P(k)` vía `pk_correction`
> (Phase 34–36) → validación externa contra CLASS (Phase 38) → barrido de
> resolución y transición shot-noise ↔ señal (Phase 41)**.

## 🧰 Herramientas y tecnologías

[![🦀 Rust](https://img.shields.io/badge/🦀_Rust-1.74%2B-orange?logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![🐍 Python](https://img.shields.io/badge/🐍_Python-3.10%2B-yellow?logo=python&logoColor=white)](https://www.python.org/)
[![🔌 MPI](https://img.shields.io/badge/🔌_MPI-OpenMPI%2FMPICH-red)](https://www.open-mpi.org/)
[![⚡ wgpu](https://img.shields.io/badge/⚡_wgpu-Vulkan%2FMetal%2FDX12-purple)](https://wgpu.rs/)
[![🌐 HDF5](https://img.shields.io/badge/🌐_HDF5-1.10%2B-blue)](https://www.hdfgroup.org/solutions/hdf5/)
[![📊 NumPy](https://img.shields.io/badge/📊_NumPy-SciPy%2FMatplotlib-blue?logo=numpy&logoColor=white)](https://numpy.org/)
[![🔭 CLASS](https://img.shields.io/badge/🔭_CLASS-classy_3.3%2B-green)](https://lesgourg.github.io/class_public/class.html)
[![🧪 GitHub_Actions](https://img.shields.io/badge/🧪_GitHub_Actions-CI-lightgrey?logo=githubactions&logoColor=white)](.github/workflows/ci.yml)
[![📜 License](https://img.shields.io/badge/📜_License-GPL--3.0-blue)](LICENSE)

| Ámbito | Herramientas |
|---|---|
| 🦀 **Core** | Rust 1.74+, workspace multi-crate, `serde`, `toml`, `clap`, `rayon` |
| 🔌 **Cómputo paralelo** | MPI (`OpenMPI` / `MPICH` vía `mpi` crate), SFC Hilbert 3D, LET, alltoall/allreduce/scatter-gather |
| ⚡ **GPU** | `wgpu` con shaders WGSL (Vulkan / Metal / DX12 / WebGPU) |
| 🗜️ **I/O** | JSONL, `bincode`, `hdf5` estilo GADGET, `msgpack`, `netcdf` |
| 🔭 **Cosmología de referencia** | `classy` 3.3+ (CLASS) para validación externa (Phase 38), Eisenstein–Hu interno |
| 📊 **Postproceso** | Python 3.10+, NumPy, SciPy, Matplotlib — mirrors de `pk_correction` y figuras por fase |
| 🧪 **CI / calidad** | GitHub Actions, `cargo fmt`, `cargo clippy -D warnings`, `cargo test --release` |
| 📁 **Experimentos** | TOML + orquestadores Bash + volcados JSON cacheables en `target/phaseNN/` |

---

## Tabla de contenidos

1. [🧰 Herramientas y tecnologías](#-herramientas-y-tecnologías)
2. [Características](#características)
3. [Inicio rápido](#inicio-rápido)
4. [Configuración TOML](#configuración-toml)
5. [Crates del workspace](#crates-del-workspace)
6. [Hitos de desarrollo](#hitos-de-desarrollo)
7. [Arquitectura de comunicación PM](#arquitectura-de-comunicación-pm)
8. [Condiciones iniciales y validación cosmológica](#condiciones-iniciales-y-validación-cosmológica)
9. [Corrección absoluta de `P(k)` (`pk_correction`)](#corrección-absoluta-de-pk-pk_correction)
10. [Convención de ICs y validación dinámica (Phase 37–41)](#convención-de-ics-y-validación-dinámica-phase-3741)
11. [Tests automáticos](#tests-automáticos)
12. [Reportes técnicos](#reportes-técnicos)
13. [Features opcionales](#features-opcionales)
14. [Calidad y CI](#calidad-y-ci)
15. [Estructura de experimentos](#estructura-de-experimentos)
16. [Licencia](#licencia)

---

## Características

| Componente | Descripción |
|---|---|
| **Integradores** | Leapfrog KDK + **Yoshida4** KDK, newtonianos **y cosmológicos** (drift/kick `∫dt'/a²`, `∫dt'/a`); timestep adaptativo estilo Aarseth |
| **Gravedad directa** | Pares Plummer-suavizados O(N²) — `DirectGravity` con SoA + SIMD |
| **Barnes–Hut + FMM** | Octree en arena, MAC `s/d < θ`, monopolo + cuadrupolo + **octupolo STF**, suavizado multipolar |
| **PM periódico** | CIC + FFT Poisson 3D periódica; solver `pm` (Fase 18) |
| **PM distribuido (allreduce)** | `allreduce_sum_f64_slice` O(nm³) — elimina `allgather` O(N·P) (Fase 19) |
| **PM slab (alltoall)** | Slab decomposition Z: FFT 3D distribuida con `alltoall_transpose`, grid **no replicado** (Fase 20) |
| **TreePM** | Barnes–Hut short-range + PM long-range; versión distribuida (Fase 21–25) con scatter-gather (Fase 24) |
| **Cosmología ΛCDM** | Friedmann ΛCDM, factor de escala `a(t)` por RK4, momentum canónico, diagnósticos `a/z/v_rms/δ_rms`; fallback EdS |
| **ICs cosmológicas** | Retícula cúbica + ZA (**1LPT**) y **2LPT**; transfer **Eisenstein–Hu no-wiggle** + normalización σ₈ con `NormalizationMode { Legacy, Z0Sigma8 }` (Phase 40) |
| **P(k) + corrección** | Estimador CIC + deconvolución; módulo [`pk_correction`](crates/gadget-ng-analysis/src/pk_correction.rs) con `A_grid = 2·V²/N⁹` (Phase 34) + `R(N)` (Phase 35) para amplitud absoluta, validado vs 🔭 CLASS (Phase 38) y en alta resolución hasta `N=128³` (Phase 41) |
| **MPI** | `ParallelRuntime` con SFC (**Hilbert 3D**), Locally Essential Trees (LET), overlap compute/comm |
| **SPH** | Kernel Wendland C2, densidad adaptativa, viscosidad artificial Monaghan |
| **GPU** | Compute shader WGSL vía `wgpu` (Vulkan/Metal/DX12/WebGPU); fallback CPU automático |
| **Análisis in-situ** | FoF (halos), espectro de potencia P(k), catálogos JSONL |
| **Checkpointing** | Guarda/reanuda desde snapshots comprimidos (`--resume`) |
| **Visualización** | Render CPU a PNG, proyecciones XY/XZ/YZ, colormap Viridis |
| **Configuración** | TOML + variables de entorno `GADGET_NG_*` |
| **Snapshots** | JSONL (default), **bincode** o **HDF5** estilo GADGET + `provenance.json`; formato auto-seleccionado por feature flag |
| **Unidades físicas** | Sección `[units]` opcional: kpc/M☉/km·s⁻¹ y `G` coherente |

---

## Inicio rápido

### Compilar

```bash
# Mínimo (CPU serial, sin GPU ni MPI):
cargo build --release -p gadget-ng-cli

# Con MPI (requiere libmpi-dev):
cargo build --release -p gadget-ng-cli --features mpi

# Con GPU (wgpu — Vulkan/Metal/DX12):
cargo build --release -p gadget-ng-cli --features gpu

# Con snapshots HDF5 (requiere libhdf5-dev):
cargo build --release -p gadget-ng-cli --features hdf5

# Todo activado:
cargo build --release -p gadget-ng-cli --features full
```

El binario queda en `target/release/gadget-ng`.

### Subcomandos disponibles

```bash
./target/release/gadget-ng --help
```

| Subcomando | Uso |
|---|---|
| `config` | Valida y muestra la configuración efectiva (TOML + env `GADGET_NG_*`) |
| `snapshot` | Escribe un snapshot del estado IC resuelto (sin evolucionar) |
| `stepping` | Integra `num_steps` pasos leapfrog KDK; opcional `--snapshot` final |
| `analyse` | FoF (halos) + espectro de potencia P(k) a partir de un snapshot |
| `visualize` | Renderiza un snapshot a PNG (proyección XY/XZ/YZ) |

### Ejecutar una simulación

```bash
# Esfera de Plummer (512 partículas, Barnes-Hut)
./target/release/gadget-ng stepping \
  --config examples/plummer_sphere.toml \
  --out runs/plummer --snapshot

# Cosmológica EdS con PM periódico (serial)
./target/release/gadget-ng stepping \
  --config experiments/nbody/phase18_periodic_pm/configs/eds_N512_pm.toml \
  --out runs/cosmo_pm

# Cosmológica ΛCDM con 2LPT + PM, snapshot inicial
./target/release/gadget-ng snapshot \
  --config experiments/nbody/phase30_linear_reference/configs/lcdm_N32_a002_2lpt_pm.toml \
  --out runs/lcdm_ic_2lpt

# Análisis de un snapshot (FoF + P(k))
./target/release/gadget-ng analyse \
  --snapshot runs/lcdm_ic_2lpt \
  --out runs/lcdm_ic_2lpt/analysis \
  --pk-mesh 32

# Cosmológica ΛCDM con PM slab distribuido (MPI, Fase 20)
mpirun -n 4 ./target/release/gadget-ng stepping \
  --config experiments/nbody/phase20_slab_pm/configs/lcdm_N2000_slab.toml \
  --out runs/cosmo_slab
```

### Con MPI

```bash
mpirun -n 4 ./target/release/gadget-ng stepping \
  --config examples/nbody_bh_1k.toml \
  --out runs/mpi --snapshot
```

### Reanudar desde un checkpoint

```bash
./target/release/gadget-ng stepping \
  --config examples/plummer_sphere.toml \
  --out runs/plummer --resume runs/plummer
```

Los snapshots incluyen `checkpoint.json` (paso, factor de escala `a`,
hash de config), `particles.jsonl|bin|h5` y, si aplica,
`hierarchical_state.json` para el integrador jerárquico.

---

## Configuración TOML

### Simulación básica

```toml
[simulation]
particle_count         = 512
box_size               = 1.0
dt                     = 0.005
num_steps              = 100
softening              = 0.0
gravitational_constant = 1.0
seed                   = 42

[initial_conditions]
# Opciones:
#   kind = "lattice"
#   kind = { two_body = { … } }
#   kind = { plummer   = { a = 1.0 } }
#   kind = { perturbed_lattice = { amplitude=0.05, velocity_amplitude=0.01 } }
#   kind = { zeldovich = { seed=42, grid_size=32, spectral_index=0.965,
#                          amplitude=1e-4, transfer="eisenstein_hu",
#                          sigma8=0.8, omega_b=0.049, h=0.674, t_cmb=2.7255,
#                          box_size_mpc_h=100.0, use_2lpt=true } }
kind = { perturbed_lattice = { amplitude = 0.05, velocity_amplitude = 0.01 } }

[gravity]
# Opciones: "direct" | "barnes_hut" | "pm" | "tree_pm"
solver       = "pm"
pm_grid_size = 32

[cosmology]
enabled       = true
omega_m       = 0.3
omega_lambda  = 0.7
h0            = 0.1   # unidades internas (ver `[units]`)
a_init        = 0.05
periodic      = true
```

### ICs cosmológicas 2LPT + EH + σ₈ (Fases 26–29)

```toml
[initial_conditions.kind.zeldovich]
seed            = 42
grid_size       = 32
spectral_index  = 0.965
amplitude       = 1.0e-4
transfer        = "eisenstein_hu"   # "power_law" | "eisenstein_hu"
sigma8          = 0.8               # normaliza la amplitud a z=0
omega_b         = 0.049
h               = 0.674
t_cmb           = 2.7255
box_size_mpc_h  = 100.0
use_2lpt        = true              # 1LPT=false / 2LPT=true
```

### PM distribuido (Fase 19 — allreduce grid)

```toml
[gravity]
solver          = "pm"
pm_grid_size    = 32
pm_distributed  = true   # allreduce O(nm³) elimina allgather O(N·P) de partículas
```

### PM slab con FFT distribuida (Fase 20 — alltoall)

```toml
[gravity]
solver       = "pm"
pm_grid_size = 32
pm_slab      = true   # FFT distribuida: O(nm³/P) por alltoall transpose
                      # requiere pm_grid_size % n_ranks == 0
```

### Unidades físicas (opcional)

```toml
[units]
enabled              = true
length_in_kpc        = 1.0e3   # box_size · length_in_kpc = kpc
mass_in_msun         = 1.0e10
velocity_in_km_s     = 1.0
```

Cuando `enabled = true`, `RunConfig::effective_g()` calcula `G` coherente
con `G = 4.3009e-6 kpc·M☉⁻¹·(km/s)²` y `meta.json` incluye el bloque
completo.

---

## Crates del workspace

```
gadget-ng/
├── crates/
│   ├── gadget-ng-core          # Vec3, Particle, RunConfig, CosmologyParams,
│   │                           # wrap_position, ic_zeldovich (1LPT+2LPT),
│   │                           # transfer EH no-wiggle + amplitude_for_sigma8
│   ├── gadget-ng-tree          # Octree + Barnes–Hut + FMM (cuadrupolo + octupolo STF),
│   │                           # SoA + SIMD, Locally Essential Trees (LET)
│   ├── gadget-ng-integrators   # leapfrog_kdk / yoshida4_kdk (newton + cosmológico),
│   │                           # integrador jerárquico, Aarseth timestep
│   ├── gadget-ng-parallel      # SerialRuntime / MpiRuntime, SFC Hilbert 3D,
│   │                           # alltoallv, allreduce, exchange_domain_{x,z}, halos
│   ├── gadget-ng-io            # Snapshots JSONL / Bincode / HDF5 + Provenance
│   ├── gadget-ng-pm            # PM: CIC, FFT Poisson periódica, slab_fft, slab_pm
│   ├── gadget-ng-treepm        # TreePM: BH short-range + PM long-range (serial + dist)
│   ├── gadget-ng-gpu           # Compute shaders WGSL vía wgpu
│   ├── gadget-ng-analysis      # FoF halos + P(k) + pk_correction (Phase 34–36)
│   ├── gadget-ng-sph           # SPH: Wendland C2, densidad adaptativa, visc. Monaghan
│   ├── gadget-ng-vis           # Visualización CPU: proyecciones, Viridis, PNG
│   ├── gadget-ng-physics       # Tests de validación física/cosmológica (Kepler,
│   │                           # Plummer, PM, TreePM, ICs, ensembles, pk_correction)
│   └── gadget-ng-cli           # Binario gadget-ng (clap), subcomandos config/snapshot/
│                               # stepping/analyse/visualize
├── examples/                   # Configuraciones TOML comentadas
├── experiments/nbody/          # Benchmarks y resultados por fase (41+ experimentos)
└── docs/reports/               # Reportes técnicos de cada fase (42 reportes)
```

---

## Hitos de desarrollo

| Fase | Descripción | Estado |
|------|-------------|--------|
| **1–2** | N-body directo O(N²), integrador leapfrog | ✅ |
| **3** | Benchmark vs GADGET-4 (fuerza, energía) | ✅ |
| **4** | Suavizado de multipolos, MAC mejorado | ✅ |
| **5** | Consistencia energía + MAC en distribuciones reales | ✅ |
| **6** | Integrador Yoshida4, convergencia de orden 4 | ✅ |
| **7** | Timestep adaptativo estilo Aarseth | ✅ |
| **8–9** | HPC: SFC Z-order, LET distribuido, halos p2p | ✅ |
| **10–11** | LetTree: árbol remoto compacto, validación paralela | ✅ |
| **12** | Reducción comunicación LET (`let_theta_export_factor`) | ✅ |
| **13** | Hilbert 3D SFC: balance de dominio mejorado vs Morton | ✅ |
| **14** | SoA + SIMD: kernels calientes en layout columnar | ✅ |
| **15–16** | SIMD explícito: tiling 4×N_i, leaf-max sweep | ✅ |
| **17a** | Cosmología serial: Friedmann ΛCDM, momentum canónico, `G/a` | ✅ |
| **17b** | Cosmología distribuida MPI con SFC+LET | ✅ |
| **18** | PM periódico: CIC + FFT Poisson, `wrap_position`, `minimum_image` | ✅ |
| **19** | PM distribuido sin allgather: `allreduce_sum_f64_slice` O(nm³) | ✅ |
| **20** | PM slab: FFT distribuida alltoall O(nm³/P), grid no replicado | ✅ |
| **21** | TreePM distribuido: BH local + PM slab (first light) | ✅ |
| **22** | TreePM: halos 3D (x,y,z) para short-range periódico | ✅ |
| **23** | TreePM: dominio 3D con SFC, métricas de balance | ✅ |
| **24** | TreePM PM scatter-gather (reemplaza allgather de densidad) | ✅ |
| **25** | Validación MPI de TreePM scatter-gather end-to-end | ✅ |
| **26** | ICs Zel'dovich 1LPT con power-law y validación numérica | ✅ |
| **27** | Transfer Eisenstein–Hu no-wiggle + normalización σ₈ | ✅ |
| **28** | ICs 2LPT (Buchert–Ehlers, corrección `-3/7` sobre 1LPT) | ✅ |
| **29** | Validación cruzada 1LPT vs 2LPT en crecimiento lineal | ✅ |
| **30** | Referencia lineal ΛCDM: P(k) medido vs EH + D₁(a)² | ✅ |
| **31** | Ensemble a resolución media: CV(R(k)) y shape | ✅ |
| **32** | Ensemble alta-res N=32³ · 6 seeds: crecimiento + PM/TreePM | ✅ |
| **33** | Análisis de normalización absoluta de `P(k)` (offset 17×) | ✅ |
| **34** | Cierre analítico del factor de grilla `A_grid = 2·V²/N⁹` | ✅ |
| **35** | Modelado del factor de muestreo `R(N) = C·N^(-α)` | ✅ |
| **36** | Validación práctica de `pk_correction` en corridas reales | ✅ |
| **37** | Reescalado físico opcional de ICs por `D(a_init)/D(1)` (exp.) | ✅ |
| **38** | 🔭 Validación externa mínima vs CLASS (IC snapshot) | ✅ |
| **39** | Convergencia temporal del integrador (barrido `dt`) | ✅ |
| **40** | Formalización de la convención física (`NormalizationMode`) | ✅ |
| **41** | Validación de alta resolución y transición shot-noise↔señal | ✅ |

---

## Arquitectura de comunicación PM

| Path | Activar | Comm/rank/paso | Solve |
|------|---------|----------------|-------|
| PM clásico (Fase 18) | `solver="pm"` | O(N·P) — allgather | Serial replicado |
| PM distribuido (Fase 19) | `pm_distributed=true` | O(nm³) — allreduce | Serial replicado |
| **PM slab (Fase 20)** | **`pm_slab=true`** | **O(nm³/P) — alltoall** | **Distribuido** |
| TreePM scatter-gather (Fase 24) | `solver="tree_pm"` + `pm_slab=true` | O(nm³/P) + halos BH | Distribuido |

Ejemplos concretos (bytes/rank/paso con nm=32):

| Ranks (P) | Fase 19 | Fase 20 |
|-----------|---------|---------|
| 1 | 262 KB | 262 KB (serial) |
| 2 | 262 KB | 131 KB |
| 4 | 262 KB |  66 KB |
| 8 | 262 KB |  33 KB |

La Fase 20 introduce `alltoall_transpose` entre slabs `z → y → x`,
permitiendo escalar la FFT 3D sin replicar el grid en cada rank.
La Fase 24 usa un scatter-gather equivalente para el cálculo de la
densidad desde partículas → grid sin `allgather` intermedio.

---

## Condiciones iniciales y validación cosmológica

Las Fases 26–32 cubren todo el pipeline de ICs cosmológicas y su
validación física:

- **Fase 26** — Zel'dovich 1LPT: `δ(k) → Ψ(k) = i·k/k²·δ(k)`,
  normalización por `σ(|n|)` en modo `power_law`.
- **Fase 27** — Transferencia EH no-wiggle + `amplitude_for_sigma8`
  (convierte σ₈ en amplitud absoluta resolviendo `σ²(R=8)=σ₈²`).
- **Fase 28** — 2LPT (Buchert–Ehlers): segundo orden en el
  desplazamiento, corrige transitorios de 1LPT a `a_init` alto.
- **Fase 29** — Validación 1LPT vs 2LPT: comparación de crecimiento
  `D₁(a)` y residuos al snapshot inicial.
- **Fase 30** — Referencia lineal ΛCDM: `P(k)` medido vs EH + CPT92
  `D(z)/D(0)` a `z ≈ 49`.
- **Fase 31** — Ensemble N=16³ · 4 seeds: CV de `R(k) = P_m/P_EH`.
- **Fase 32** — Ensemble alta-resolución N=32³ · 6 seeds: 10 tests
  sobre shape espectral, crecimiento, PM vs TreePM y reproducibilidad
  bit-idéntica.

Para configurar un IC cosmológico completo:

```toml
[initial_conditions.kind.zeldovich]
seed            = 42
grid_size       = 32
spectral_index  = 0.965
transfer        = "eisenstein_hu"
sigma8          = 0.8
omega_b         = 0.049
h               = 0.674
t_cmb           = 2.7255
box_size_mpc_h  = 100.0
use_2lpt        = true
```

---

## Corrección absoluta de `P(k)` (`pk_correction`)

Las Fases 33–36 cierran la normalización absoluta del estimador de
`P(k)`. El problema identificado en Phase 33 — un offset de amplitud
`~10¹⁴` entre `P(k)` medido y la referencia continua — se decompone
en dos factores:

```text
P_measured(k) = A_grid(N) · R(N) · P_cont(k)

  A_grid(N) = 2·V²/N⁹       (Phase 34 — cerrado analíticamente)
  R(N)      = C · N^(-α)    (Phase 35 — fit log-log, R² = 0.997)
              con C = 22.108, α = 1.8714, tabla exacta
              para N ∈ {8, 16, 32, 64}
```

La corrección se aplica en postproceso vía la API pública de
[`gadget_ng_analysis::pk_correction`](crates/gadget-ng-analysis/src/pk_correction.rs):

```rust
use gadget_ng_analysis::pk_correction::{correct_pk, RnModel};
use gadget_ng_analysis::power_spectrum::power_spectrum;

let mesh  = 32;
let pk    = power_spectrum(&positions, &masses, box_size, mesh);
let model = RnModel::phase35_default();
let pk_phys = correct_pk(&pk, box_size, mesh, /* box_mpc_h */ None, &model);
// pk_phys[i].pk está ahora en (Mpc/h)³ y compara directamente con EH·D(a)²
```

Phase 36 validó la corrección end-to-end sobre 27 snapshots reales
(N=32³/64³, 3 seeds, 1LPT y 2LPT, PM): la mediana
`|log₁₀(P_m/P_ref)|` baja de **14.7–18.0** a **0.03** en el snapshot
IC — una mejora de factor `~10¹⁴`, con `mean(P_c/P_ref) = 1.00 ± 0.05`
y `CV ≤ 0.15`. El mismo resultado se reproduce vía CLI real
(`gadget-ng snapshot → analyse → apply_phase36_correction.py`).

🔭 **Phase 38** cierra la validación externa contra CLASS (`classy 3.3.4.0`,
referencia reproducible en
[`experiments/nbody/phase38_class_validation/reference/`](experiments/nbody/phase38_class_validation/reference/)):
`median|log10(P_c/P_CLASS)| ∈ [0.022, 0.046]` y `mean(P_c/P_CLASS) ∈ [0.95,
1.04]` sobre 12 mediciones, factor de mejora **161× a N=32³** y **761× a
N=64³** frente a `P_measured/P_CLASS`.

📈 **Phase 41** extiende la validación a alta resolución (N ∈ {32, 64, 128})
y demuestra empíricamente la transición **shot-noise ↔ señal**
(`P_shot = V/N_p`):

| `N`   | `P_shot` [(Mpc/h)³] | `S/N(k_min)` en IC (Z0Sigma8) |
|-------|---------------------|-------------------------------|
|  32   | 30.52               | 0.374 → shot-noise domina     |
|  64   | 3.815               | 2.21  → transición            |
| 128   | 0.4768              | 16.06 → señal limpia          |

`pk_correction` cierra en IC a `median|log10(P_c/P_ref)| ≤ 0.049` para los
tres N, confirmando que el resultado de Phase 38 se extiende a `N = 128³`.

Para más detalles, ver
[`docs/reports/2026-04-phase36-pk-correction-validation.md`](docs/reports/2026-04-phase36-pk-correction-validation.md),
[`docs/reports/2026-04-phase38-class-camb-minimal-validation.md`](docs/reports/2026-04-phase38-class-camb-minimal-validation.md)
y
[`docs/reports/2026-04-phase41-high-resolution-validation.md`](docs/reports/2026-04-phase41-high-resolution-validation.md).

---

## Convención de ICs y validación dinámica (Phase 37–41)

Las Phases 37–41 formalizan la convención de normalización de ICs
cosmológicas y exploran sus límites dinámicos:

- **Phase 37** introduce el flag experimental `rescale_to_a_init` (luego
  renombrado a `NormalizationMode` en Phase 40): aplica `Ψ¹ ← s·Ψ¹` y
  `Ψ² ← s²·Ψ²` con `s = D(a_init)/D(1)` (CPT92). Decisión: **B** — la
  implementación es exacta (residuo `< 2·10⁻⁶`) pero no mejora snapshots
  evolucionados a `N=32³`, queda experimental.
- **Phase 38** valida `pk_correction` contra 🔭 CLASS en ambas convenciones
  (`legacy` vs `z=0`, `rescaled` vs `z=49`) — confirma que el cierre es
  intrínseco a la corrección y no depende de la normalización.
- **Phase 39** barre `dt ∈ {4·10⁻⁴, 2·10⁻⁴, 1·10⁻⁴, 5·10⁻⁵}` y demuestra
  que reducir `dt` **no** reduce el error espectral (pendiente log-log
  observada ≈ `−0.06` vs predicción `+2` para KDK). Decisión: mantener
  `dt = 4·10⁻⁴` como default.
- **Phase 40** reemplaza el booleano `rescale_to_a_init` por una enum
  tipada `NormalizationMode { Legacy, Z0Sigma8 }` (⚠️ breaking change en
  TOML), audita la implementación LPT (sin bugs) y verifica
  empíricamente `σ₈(Z0Sigma8)/σ₈(Legacy) = s` a precisión de máquina.
  Decisión: **B** — `Z0Sigma8` queda experimental por dominancia de
  shot-noise a `N=32³`.
- **Phase 41** resuelve Phase 40-B: a `N ≥ 64³` la señal supera el
  shot-noise floor (`S/N(k_min) = 2.21` en IC) y a `N = 128³` el margen
  es ×16. Decisión: **cierre parcial** — el eje señal/ruido queda
  cerrado; el eje evolución lineal/no-lineal requiere softening físico y/o
  integrador adaptativo (trabajo futuro).

```toml
# Convención Phase 40+ — reemplaza `rescale_to_a_init = true/false`
[initial_conditions.kind.zeldovich]
# ...
normalization_mode = "legacy"     # default, bit-compatible con Phases 26–39
# normalization_mode = "z0_sigma8" # σ₈ referido a a=1 (CAMB/CLASS), Phase 40+
```

Reportes: [Phase 37](docs/reports/2026-04-phase37-growth-rescaled-ics.md) ·
[Phase 38](docs/reports/2026-04-phase38-class-camb-minimal-validation.md) ·
[Phase 39](docs/reports/2026-04-phase39-dt-convergence.md) ·
[Phase 40](docs/reports/2026-04-phase40-physical-ics-normalization.md) ·
[Phase 41](docs/reports/2026-04-phase41-high-resolution-validation.md).

---

## Tests automáticos

```bash
# Tests unitarios de todos los crates
cargo test

# Tests de validación física (N-body + cosmología)
cargo test -p gadget-ng-physics --release

# Tests específicos por fase
cargo test -p gadget-ng-physics --test cosmo_serial                       # Fase 17a
cargo test -p gadget-ng-physics --test cosmo_pm                           # Fase 18
cargo test -p gadget-ng-physics --test cosmo_pm_dist                      # Fase 19
cargo test -p gadget-ng-physics --test cosmo_pm_slab                      # Fase 20
cargo test -p gadget-ng-physics --test phase30_linear_reference --release # Fase 30
cargo test -p gadget-ng-physics --test phase31_ensemble          --release # Fase 31
cargo test -p gadget-ng-physics --test phase32_high_res_ensemble --release # Fase 32
cargo test -p gadget-ng-physics --test phase33_pk_normalization  --release # Fase 33
cargo test -p gadget-ng-physics --test phase34_discrete_normalization --release # Fase 34
cargo test -p gadget-ng-physics --test phase35_rn_modeling       --release # Fase 35
cargo test -p gadget-ng-physics --test phase36_pk_correction_validation --release # Fase 36
cargo test -p gadget-ng-physics --test phase37_growth_rescaled_ics --release # Fase 37
cargo test -p gadget-ng-physics --test phase38_class_validation  --release # Fase 38 (CLASS)
cargo test -p gadget-ng-physics --test phase39_dt_convergence    --release # Fase 39
cargo test -p gadget-ng-physics --test phase40_physical_ics_normalization --release # Fase 40
cargo test -p gadget-ng-physics --test phase41_high_resolution_validation --release # Fase 41
# Rerun rápido Phase 41 desde cache de disco (~0.7 s):
PHASE41_USE_CACHE=1 PHASE41_SKIP_N256=1 cargo test -p gadget-ng-physics \
    --test phase41_high_resolution_validation --release
```

Tests de validación cubiertos:

- **Kepler**: conservación de energía y momento angular (1–2 cuerpos).
- **Plummer**: ratio virial Q ≈ 0.5 en equilibrio.
- **Cosmología serial**: EdS y ΛCDM, `a(t)` por RK4, sin NaN.
- **PM periódico** (Fase 18): CIC masa, Poisson sinusoidal, `G/a`, wrap.
- **PM distribuido** (Fase 19): equivalencia serial/MPI, allreduce.
- **PM slab** (Fase 20): `SlabLayout`, ghost CIC, transpose roundtrip,
  Poisson sanity a P = {1, 2, 4} ranks.
- **TreePM scatter-gather** (Fase 24–25): paridad con TreePM replicado.
- **ICs ZA + EH + σ₈** (Fase 26–27): transferencia, normalización,
  reproducibilidad bit-idéntica.
- **2LPT** (Fase 28–29): crecimiento `D₁(a)`, residuos 1LPT vs 2LPT.
- **Ensembles cosmológicos** (Fase 30–32): shape espectral, `R(k)`,
  crecimiento lineal, PM vs TreePM, reproducibilidad.
- **`pk_correction`** (Fase 34–36): roundtrip DFT, `A_grid` cerrado,
  `R(N)` estable, corrección absoluta sobre snapshots reales.
- **Reescalado físico de ICs** (Fase 37): `s = D(a_init)/D(1)` exacto a
  `< 2·10⁻⁶`, crecimiento `D(a)` sensible a la convención.
- **Validación externa vs CLASS** (Fase 38): factor de mejora ≥161× y
  `median|log10(P_c/P_CLASS)| ≤ 0.046` en ambas convenciones.
- **Convergencia temporal** (Fase 39): barrido `dt` + diagnósticos
  `delta_rms(a)`, sin NaN/Inf.
- **`NormalizationMode`** (Fase 40): enum tipado, equivalencia bit-idéntica
  del modo `Legacy` y ratio `s` exacto del modo `Z0Sigma8`.
- **Alta resolución** (Fase 41): transición shot-noise ↔ señal a `N ≥ 64³`,
  cierre de `pk_correction` en IC hasta `N = 128³`, con caché JSON para
  rerun sub-segundo (`PHASE41_USE_CACHE=1`).

---

## Reportes técnicos

Los reportes en [`docs/reports/`](docs/reports/) documentan cada fase
con contexto, metodología, resultados y limitaciones.

### N-body + HPC

| Reporte | Tema |
|---------|------|
| [`phase3-gadget4-benchmark`](docs/reports/2026-04-phase3-gadget4-benchmark.md) | Benchmark vs GADGET-4 |
| [`phase4-multipole-softening`](docs/reports/2026-04-phase4-multipole-softening.md) | Suavizado de multipolos |
| [`phase5-energy-mac-consistency`](docs/reports/2026-04-phase5-energy-mac-consistency.md) | Consistencia energía + MAC |
| [`phase6-higher-order-integrator`](docs/reports/2026-04-phase6-higher-order-integrator.md) | Yoshida4 |
| [`phase7-aarseth-timestep`](docs/reports/2026-04-phase7-aarseth-timestep.md) | Timestep adaptativo |
| [`phase8-hpc-scaling`](docs/reports/2026-04-phase8-hpc-scaling.md) | Escalado HPC |
| [`phase9-hpc-local`](docs/reports/2026-04-phase9-hpc-local.md) | SFC + halos locales |
| [`phase10-let-tree`](docs/reports/2026-04-phase10-let-tree.md) | LetTree compacto |
| [`phase11-let-tree-parallel-validation`](docs/reports/2026-04-phase11-let-tree-parallel-validation.md) | Validación LET paralelo |
| [`phase12-let-communication-reduction`](docs/reports/2026-04-phase12-let-communication-reduction.md) | Reducción comm LET |
| [`phase13-hilbert-decomposition`](docs/reports/2026-04-phase13-hilbert-decomposition.md) | Hilbert 3D SFC |
| [`phase14-soa-simd`](docs/reports/2026-04-phase14-soa-simd.md) | SoA + SIMD |
| [`phase15-explicit-simd`](docs/reports/2026-04-phase15-explicit-simd.md) | SIMD explícito |
| [`phase15b-leaf-max-sweep`](docs/reports/2026-04-phase15b-leaf-max-sweep.md) | Leaf-max sweep |
| [`phase16-tiled-simd-multi-i`](docs/reports/2026-04-phase16-tiled-simd-multi-i.md) | Tiling 4×N_i |

### Cosmología + PM + TreePM

| Reporte | Tema |
|---------|------|
| [`phase17a-cosmology-serial`](docs/reports/2026-04-phase17a-cosmology-serial.md) | Cosmología ΛCDM serial |
| [`phase17b-cosmology-distributed`](docs/reports/2026-04-phase17b-cosmology-distributed.md) | Cosmología MPI + SFC+LET |
| [`phase18-periodic-pm`](docs/reports/2026-04-phase18-periodic-pm.md) | PM periódico con CIC + FFT |
| [`phase19-distributed-pm`](docs/reports/2026-04-phase19-distributed-pm.md) | PM sin allgather |
| [`phase20-slab-distributed-pm`](docs/reports/2026-04-phase20-slab-distributed-pm.md) | PM slab FFT distribuida |
| [`phase21-distributed-treepm`](docs/reports/2026-04-phase21-distributed-treepm.md) | TreePM distribuido (first light) |
| [`phase22-treepm-3d-halo`](docs/reports/2026-04-phase22-treepm-3d-halo.md) | Halos 3D para SR periódico |
| [`phase23-treepm-sr-3d-domain`](docs/reports/2026-04-phase23-treepm-sr-3d-domain.md) | Dominio 3D con SFC |
| [`phase24-treepm-pm-scatter-gather`](docs/reports/2026-04-phase24-treepm-pm-scatter-gather.md) | Scatter-gather PM |
| [`phase25-treepm-scatter-gather-mpi-validation`](docs/reports/2026-04-phase25-treepm-scatter-gather-mpi-validation.md) | Validación MPI scatter-gather |

### ICs, ensembles y normalización de `P(k)`

| Reporte | Tema |
|---------|------|
| [`phase26-zeldovich-ics-validation`](docs/reports/2026-04-phase26-zeldovich-ics-validation.md) | ICs Zel'dovich (1LPT) |
| [`phase27-transfer-sigma8-ics`](docs/reports/2026-04-phase27-transfer-sigma8-ics.md) | Transfer EH + σ₈ |
| [`phase28-2lpt-ics`](docs/reports/2026-04-phase28-2lpt-ics.md) | ICs 2LPT |
| [`phase29-1lpt-vs-2lpt-validation`](docs/reports/2026-04-phase29-1lpt-vs-2lpt-validation.md) | 1LPT vs 2LPT |
| [`phase30-linear-reference-validation`](docs/reports/2026-04-phase30-linear-reference-validation.md) | Referencia lineal ΛCDM |
| [`phase31-ensemble-higher-resolution-validation`](docs/reports/2026-04-phase31-ensemble-higher-resolution-validation.md) | Ensemble N=16³ |
| [`phase32-high-resolution-ensemble-validation`](docs/reports/2026-04-phase32-high-resolution-ensemble-validation.md) | Ensemble N=32³ · 6 seeds |
| [`phase33-pk-normalization-analysis`](docs/reports/2026-04-phase33-pk-normalization-analysis.md) | Análisis del offset 17× |
| [`phase34-discrete-normalization-closure`](docs/reports/2026-04-phase34-discrete-normalization-closure.md) | Cierre `A_grid = 2·V²/N⁹` |
| [`phase35-rn-modeling`](docs/reports/2026-04-phase35-rn-modeling.md) | Modelado `R(N)` |
| [`phase36-pk-correction-validation`](docs/reports/2026-04-phase36-pk-correction-validation.md) | Validación `pk_correction` |
| [`phase37-growth-rescaled-ics`](docs/reports/2026-04-phase37-growth-rescaled-ics.md) | Reescalado físico `D(a)` experimental |
| [`phase38-class-camb-minimal-validation`](docs/reports/2026-04-phase38-class-camb-minimal-validation.md) | 🔭 Validación externa vs CLASS |
| [`phase39-dt-convergence`](docs/reports/2026-04-phase39-dt-convergence.md) | Barrido `dt` y diagnósticos dinámicos |
| [`phase40-physical-ics-normalization`](docs/reports/2026-04-phase40-physical-ics-normalization.md) | `NormalizationMode { Legacy, Z0Sigma8 }` |
| [`phase41-high-resolution-validation`](docs/reports/2026-04-phase41-high-resolution-validation.md) | 📈 Alta resolución y shot-noise↔señal |

### Meta

| Reporte | Tema |
|---------|------|
| [`gadget-ng-treepm-evolution-paper`](docs/reports/2026-04-gadget-ng-treepm-evolution-paper.md) | Paper-style sobre evolución TreePM |
| [`validation-phase`](docs/reports/2026-04-validation-phase.md) | Protocolo general de validación |

---

## Features opcionales

| Feature | Descripción |
|---------|-------------|
| `mpi` | Enlaza a MPI para `MpiRuntime` con descomposición SFC Hilbert |
| `gpu` | Aceleración GPU vía `wgpu` (Vulkan/Metal/DX12/WebGPU) |
| `simd` | Vectorización explícita con `rayon` + SIMD |
| `bincode` | Snapshots binarios `particles.bin` |
| `hdf5` | Snapshots `snapshot.hdf5` estilo GADGET (requiere `libhdf5-dev`) |
| `msgpack` | Snapshots compactos MessagePack |
| `netcdf` | Snapshots NetCDF4 (requiere `libnetcdf-dev`) |
| `full` | Todas las anteriores activadas |

---

## Calidad y CI

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --release
cargo build --release --features mpi
```

GitHub Actions: [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

Convención de estilo:

- Código y comentarios en **español** salvo nombres de items públicos.
- Tests de fase nuevos bajo `crates/gadget-ng-physics/tests/phaseNN_*.rs`
  con orquestador `experiments/nbody/phaseNN_*/run_phaseNN.sh`.
- Un reporte por fase en `docs/reports/2026-MM-phaseNN-*.md`.
- `CHANGELOG.md` actualizado bajo `[Unreleased]` antes de cerrar PR.

---

## Estructura de experimentos

Cada experimento vive en `experiments/nbody/phaseNN_*/` con una
estructura uniforme:

```
experiments/nbody/phaseNN_<slug>/
├── configs/           # Uno o varios .toml
├── scripts/           # Python de postproceso + generación de figuras
├── output/            # Artefactos de corrida (ignorado por git en general)
├── figures/           # PNGs generados
└── run_phaseNN.sh     # Orquestador: ejecuta, mide, plotea
```

Ejemplos actuales:

```
experiments/nbody/
├── phase18_periodic_pm/      # PM periódico: N=512..2000, grid 16³..32³
├── phase19_distributed_pm/   # PM allreduce: comparativa vs clásico
├── phase20_slab_pm/          # PM slab alltoall: escalado P=1,2,4
├── phase26_zeldovich_ics/    # Validación ZA + reproducibilidad
├── phase27_transfer_sigma8/  # EH + σ₈ + roundtrip σ
├── phase28_2lpt/             # 2LPT: corrección al campo de velocidad
├── phase29_1lpt_vs_2lpt/     # Crecimiento D₁(a) comparado
├── phase30_linear_reference/ # Referencia lineal vs EH + D(a)²
├── phase31_ensemble_higher_res/
├── phase33_pk_normalization/ # Caracterización del offset 17×
├── phase34_discrete_normalization/ # A_grid = 2·V²/N⁹
├── phase35_rn_modeling/      # Fit R(N), figuras, demo
├── phase36_pk_correction_validation/ # Validación práctica end-to-end
│   ├── configs/lcdm_N32_2lpt_pm_phase36.toml
│   ├── scripts/apply_phase36_correction.py
│   ├── scripts/plot_phase36.py
│   └── run_phase36.sh
├── phase37_growth_rescaled_ics/ # Reescalado físico opcional de ICs
├── phase38_class_validation/    # 🔭 Referencia CLASS + comparación por N
│   └── reference/               # JSON generado por classy 3.3.4.0
├── phase39_dt_convergence/      # Barrido dt ∈ {4e-4..5e-5}
├── phase40_physical_ics_normalization/ # Enum NormalizationMode
└── phase41_high_resolution_validation/ # N ∈ {32, 64, 128}, shot-noise
    ├── configs/lcdm_N{128,256}_2lpt_pm_{legacy,z0_sigma8}.toml
    ├── scripts/apply_phase41_correction.py
    ├── scripts/plot_phase41_resolution.py
    └── run_phase41.sh           # PHASE41_SKIP_N256=1, PHASE41_USE_CACHE=1
```

---

## Licencia

Este repositorio se distribuye bajo la
[GNU General Public License v3.0](LICENSE) (GPL-3.0).
