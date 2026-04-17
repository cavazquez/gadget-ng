# gadget-ng

> Simulador **N-body cosmológico** en Rust, inspirado conceptualmente en la arquitectura y prácticas de [GADGET-4](https://wwwmpa.mpa-garching.mpg.de/gadget4/), sin compartir código ni historial git con el proyecto original.

![CI](https://github.com/cristian/gadget-ng/actions/workflows/ci.yml/badge.svg)
![Rust](https://img.shields.io/badge/rust-1.74%2B-orange?logo=rust)
![License](https://img.shields.io/badge/license-GPL--3.0-blue)

---

## Características

| Componente | Descripción |
|---|---|
| **Integradores** | Leapfrog KDK + **Yoshida4** KDK, cosmológicos y newtonianos |
| **Gravedad directa** | Pares Plummer-suavizados O(N²) — `DirectGravity` |
| **Barnes–Hut + FMM** | Octree en arena, MAC `s/d < θ`, monopolo + cuadrupolo + **octupolo STF** |
| **PM periódico** | CIC + FFT Poisson 3D periódica; solver `pm` y `tree_pm` |
| **PM distribuido** | Slab decomposition Z: FFT 3D distribuida con alltoall transposes (Fase 20) |
| **Cosmología ΛCDM** | Friedmann ΛCDM, factor de escala `a(t)`, momentum canónico, diagnósticos `a/z/v_rms/δ_rms` |
| **MPI** | `ParallelRuntime` con SFC (**Hilbert 3D**), Locally Essential Trees (LET), overlap compute/comm |
| **SPH** | Kernel Wendland C2, densidad adaptativa, viscosidad artificial Monaghan |
| **GPU** | Compute shader WGSL vía `wgpu` (Vulkan/Metal/DX12); fallback CPU automático |
| **Checkpointing** | Guarda/reanuda desde snapshots comprimidos (`--resume`) |
| **Análisis in-situ** | FoF (halos), espectro de potencia P(k), catálogos JSONL |
| **Visualización** | Render CPU a PNG, proyecciones XY/XZ/YZ, colormap Viridis |
| **Configuración** | TOML + variables de entorno `GADGET_NG_*` |
| **Snapshots** | JSONL (default), **bincode** o **HDF5** estilo GADGET + `provenance.json` |
| **ICs** | Retícula cúbica, dos cuerpos, Plummer, **PerturbedLattice** cosmológica |

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

# Todo activado:
cargo build --release -p gadget-ng-cli --features full
```

El binario queda en `target/release/gadget-ng`.

### Ver ayuda

```bash
./target/release/gadget-ng --help
./target/release/gadget-ng stepping --help
```

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

---

## Configuración TOML

### Simulación básica

```toml
[simulation]
particle_count     = 512
box_size           = 1.0
dt                 = 0.005
num_steps          = 100
softening          = 0.0
gravitational_constant = 1.0
seed               = 42

[initial_conditions]
# Opciones: "lattice" | { two_body = {...} } | { plummer = { a=1.0 } }
# | { perturbed_lattice = { amplitude=0.05, velocity_amplitude=0.01 } }
kind = { perturbed_lattice = { amplitude = 0.05, velocity_amplitude = 0.01 } }

[gravity]
# Opciones: "direct" | "barnes_hut" | "pm" | "tree_pm"
solver       = "pm"
pm_grid_size = 32

[cosmology]
enabled       = true
omega_m       = 0.3
omega_lambda  = 0.7
h0            = 70.0   # km/s/Mpc
a_init        = 0.05
periodic      = true
box_size      = 1.0
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

---

## Crates del workspace

```
gadget-ng/
├── crates/
│   ├── gadget-ng-core          # Vec3, Particle, RunConfig, CosmologyParams, wrap_position
│   ├── gadget-ng-tree          # Octree + Barnes-Hut + FMM (cuadrupolo + octupolo STF)
│   │                           # SoA + SIMD, Locally Essential Trees (LET)
│   ├── gadget-ng-integrators   # leapfrog_kdk / yoshida4_kdk (newton + cosmológico)
│   ├── gadget-ng-parallel      # SerialRuntime / MpiRuntime, SFC Hilbert 3D,
│   │                           # alltoallv, allreduce, exchange_domain_{x,z}, halos
│   ├── gadget-ng-io            # Snapshots JSONL / Bincode / HDF5 + Provenance
│   ├── gadget-ng-pm            # PM: CIC, FFT Poisson periódica, slab_fft, slab_pm
│   ├── gadget-ng-treepm        # TreePM: Barnes-Hut short-range + PM long-range
│   ├── gadget-ng-gpu           # Compute shaders WGSL vía wgpu
│   ├── gadget-ng-analysis      # FoF halos + espectro de potencia P(k)
│   ├── gadget-ng-sph           # SPH: Wendland C2, densidad adaptativa, viscosidad Monaghan
│   ├── gadget-ng-vis           # Visualización CPU: proyecciones, colormap Viridis, PNG
│   ├── gadget-ng-physics       # Tests de validación física (Kepler, Plummer, cosmología)
│   └── gadget-ng-cli           # Binario gadget-ng (clap)
├── examples/                   # Configuraciones TOML comentadas
├── experiments/nbody/          # Benchmarks y resultados por fase
└── docs/reports/               # Reportes técnicos de cada fase
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
| **20** | **PM slab: FFT distribuida alltoall O(nm³/P), grid no replicado** | ✅ |

---

## Arquitectura de comunicación PM

| Path | Activar | Comm/rank/paso | Solve |
|------|---------|----------------|-------|
| PM clásico (Fase 18) | `solver="pm"` | O(N·P) — allgather | Serial replicado |
| PM distribuido (Fase 19) | `pm_distributed=true` | O(nm³) — allreduce | Serial replicado |
| **PM slab (Fase 20)** | **`pm_slab=true`** | **O(nm³/P) — alltoall** | **Distribuido** |

Ejemplos concretos (bytes/rank/paso con nm=32):

| Ranks (P) | Fase 19 | Fase 20 |
|-----------|---------|---------|
| 1 | 262 KB | 262 KB (serial) |
| 2 | 262 KB | 131 KB |
| 4 | 262 KB | 66 KB |
| 8 | 262 KB | 33 KB |

---

## Tests automáticos

```bash
# Tests unitarios de todos los crates
cargo test

# Tests de validación física (Fases 17-20)
cargo test -p gadget-ng-physics

# Tests específicos por fase
cargo test -p gadget-ng-physics --test cosmo_serial    # Fase 17a: cosmología serial
cargo test -p gadget-ng-physics --test cosmo_pm        # Fase 18: PM periódico (8 tests)
cargo test -p gadget-ng-physics --test cosmo_pm_dist   # Fase 19: PM distribuido (7 tests)
cargo test -p gadget-ng-physics --test cosmo_pm_slab   # Fase 20: PM slab (8 tests)
```

Tests de validación física cubiertos:
- **Kepler**: conservación de energía y momento angular
- **Plummer**: ratio virial Q ≈ 0.5 en equilibrio
- **Cosmología serial**: EdS y ΛCDM, `a(t)`, sin NaN
- **PM periódico**: CIC masa, Poisson sinusoidal, `G/a`, wrap
- **PM distribuido**: equivalencia serial/MPI, allreduce
- **PM slab**: SlabLayout, ghost CIC, transpose roundtrip, Poisson sanity

---

## Reportes técnicos

Los reportes en [`docs/reports/`](docs/reports/) documentan cada fase:

| Reporte | Fase |
|---------|------|
| [`2026-04-phase17a-cosmology-serial.md`](docs/reports/2026-04-phase17a-cosmology-serial.md) | Cosmología ΛCDM serial |
| [`2026-04-phase17b-cosmology-distributed.md`](docs/reports/2026-04-phase17b-cosmology-distributed.md) | Cosmología MPI + SFC+LET |
| [`2026-04-phase18-periodic-pm.md`](docs/reports/2026-04-phase18-periodic-pm.md) | PM periódico con CIC + FFT |
| [`2026-04-phase19-distributed-pm.md`](docs/reports/2026-04-phase19-distributed-pm.md) | PM sin allgather de partículas |
| [`2026-04-phase20-slab-distributed-pm.md`](docs/reports/2026-04-phase20-slab-distributed-pm.md) | PM slab con FFT distribuida |

---

## Features opcionales

| Feature | Descripción |
|---------|-------------|
| `mpi` | Enlaza a MPI para `MpiRuntime` con descomposición SFC Hilbert |
| `gpu` | Aceleración GPU vía `wgpu` (Vulkan/Metal/DX12/WebGPU) |
| `simd` | Vectorización con `rayon` + SIMD explícito |
| `bincode` | Snapshots binarios `particles.bin` |
| `hdf5` | Snapshots `snapshot.hdf5` (GADGET-like; requiere `libhdf5-dev`) |
| `full` | Todas las anteriores activadas |

---

## Calidad y CI

```bash
cargo fmt --check
cargo clippy -- -D warnings
cargo test
cargo build --features mpi
```

GitHub Actions: [`.github/workflows/ci.yml`](.github/workflows/ci.yml)

---

## Estructura de experimentos

```
experiments/nbody/
├── phase17a_cosmo_serial/    # ICs cosmo + run serial EdS/ΛCDM
├── phase17b_cosmo_distributed/ # Paridad serial vs MPI
├── phase18_periodic_pm/      # PM periódico: N=512..2000, grid 16³..32³
├── phase19_distributed_pm/   # PM allreduce: comparativa vs clásico
└── phase20_slab_pm/          # PM slab alltoall: escalado P=1,2,4
    ├── configs/
    │   ├── eds_N512_slab.toml
    │   ├── lcdm_N2000_slab.toml
    │   └── eds_N4000_slab.toml
    ├── run_phase20.sh
    └── analyze_phase20.py
```

---

## Licencia

Este repositorio se distribuye bajo la [GNU General Public License v3.0](LICENSE) (GPL-3.0).
