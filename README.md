# gadget-ng

> Simulador **N-body cosmológico** en Rust, inspirado conceptualmente en la
> arquitectura y prácticas de [GADGET-4](https://wwwmpa.mpa-garching.mpg.de/gadget4/),
> sin compartir código ni historial git con el proyecto original.
>
> Cubre el pipeline completo de una corrida cosmológica contemporánea:
> **ICs 1LPT/2LPT con transferencia Eisenstein–Hu y normalización σ₈ →
> integrador leapfrog/Yoshida4 con factores de drift/kick cosmológicos →
> PM periódico / TreePM / Barnes–Hut / Direct, con versiones
> distribuidas (allreduce, slab alltoall, scatter-gather) → análisis
> in-situ (FoF, P(k)) → corrección absoluta de `P(k)` vía `pk_correction`
> (Phase 34–36)**.

![CI](https://github.com/cristian/gadget-ng/actions/workflows/ci.yml/badge.svg)
![Rust](https://img.shields.io/badge/rust-1.74%2B-orange?logo=rust)
![License](https://img.shields.io/badge/license-GPL--3.0-blue)

---

## Tabla de contenidos

1. [Características](#características)
2. [Inicio rápido](#inicio-rápido)
3. [Configuración TOML](#configuración-toml)
4. [Crates del workspace](#crates-del-workspace)
5. [Hitos de desarrollo](#hitos-de-desarrollo)
6. [Arquitectura de comunicación PM](#arquitectura-de-comunicación-pm)
7. [Condiciones iniciales y validación cosmológica](#condiciones-iniciales-y-validación-cosmológica)
8. [Corrección absoluta de `P(k)` (`pk_correction`)](#corrección-absoluta-de-pk-pk_correction)
9. [Tests automáticos](#tests-automáticos)
10. [Reportes técnicos](#reportes-técnicos)
11. [Features opcionales](#features-opcionales)
12. [Calidad y CI](#calidad-y-ci)
13. [Estructura de experimentos](#estructura-de-experimentos)
14. [Licencia](#licencia)

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
| **ICs cosmológicas** | Retícula cúbica + ZA (**1LPT**) y **2LPT**; transfer **Eisenstein–Hu no-wiggle** + normalización σ₈ |
| **P(k) + corrección** | Estimador CIC + deconvolución; módulo [`pk_correction`](crates/gadget-ng-analysis/src/pk_correction.rs) con `A_grid = 2·V²/N⁹` (Phase 34) + `R(N)` (Phase 35) para amplitud absoluta |
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
├── experiments/nbody/          # Benchmarks y resultados por fase (34+ experimentos)
└── docs/reports/               # Reportes técnicos de cada fase (37 reportes)
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

Para más detalles, ver
[`docs/reports/2026-04-phase36-pk-correction-validation.md`](docs/reports/2026-04-phase36-pk-correction-validation.md).

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
└── phase36_pk_correction_validation/ # Validación práctica end-to-end
    ├── configs/lcdm_N32_2lpt_pm_phase36.toml
    ├── scripts/apply_phase36_correction.py
    ├── scripts/plot_phase36.py
    └── run_phase36.sh
```

---

## Licencia

Este repositorio se distribuye bajo la
[GNU General Public License v3.0](LICENSE) (GPL-3.0).
