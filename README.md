# gadget-ng

> Simulador **N-body** en Rust, inspirado conceptualmente en la arquitectura y prácticas de [GADGET-4](https://wwwmpa.mpa-garching.mpg.de/gadget4/), sin compartir código ni historial git con el proyecto original.

![CI](https://github.com/cristian/gadget-ng/actions/workflows/ci.yml/badge.svg)
![Rust](https://img.shields.io/badge/rust-1.74%2B-orange?logo=rust)
![License](https://img.shields.io/badge/license-GPL--3.0-blue)

---

## Características

| Componente | Descripción |
|---|---|
| **Integración** | Leapfrog **KDK** (kick–drift–kick) con paso global |
| **Gravedad directa** | Pares Plummer-suavizados, O(N²) — `DirectGravity` |
| **Barnes–Hut + FMM** | Octree en arena, MAC `s/d < θ`, monopolo + cuadrupolo + **octupolo STF**, error < 0.1 % vs directo |
| **GPU** | Compute shader WGSL vía `wgpu` (Vulkan/Metal/DX12); fallback CPU automático |
| **MPI** | `ParallelRuntime` con descomposición **SFC (Z-order)** y balanceo dinámico |
| **Cosmología** | Integración ΛCDM con momento canónico, factores Drift/Kick, pasos jerárquicos |
| **SPH** | Kernel Wendland C2, densidad adaptativa, viscosidad artificial Monaghan |
| **Checkpointing** | Guarda/reanuda desde snapshots comprimidos (`--resume`) |
| **Análisis in-situ** | FoF (halos), espectro de potencia P(k), catálogos JSONL |
| **Visualización** | Render CPU a PNG, proyecciones XY/XZ/YZ/Perspectiva, colormap Viridis |
| **Configuración** | TOML + variables de entorno `GADGET_NG_*` |
| **Snapshots** | JSONL (default), **bincode** o **HDF5** estilo GADGET + `provenance.json` |
| **ICs** | Retícula cúbica, dos cuerpos, **esfera de Plummer** con equilibrio virial |

---

## Inicio rápido

### Compilar

```bash
# Mínimo (CPU serial, sin GPU ni MPI):
cargo build --release -p gadget-ng-cli

# Con GPU (wgpu — Vulkan/Metal/DX12):
cargo build --release -p gadget-ng-cli --features gpu

# Con MPI (requiere libmpi-dev):
cargo build --release -p gadget-ng-cli --features mpi

# Todo activado:
cargo build --release -p gadget-ng-cli --features full
```

El binario queda en `target/release/gadget-ng`.

### Ver ayuda

```bash
./target/release/gadget-ng --help
./target/release/gadget-ng stepping --help
./target/release/gadget-ng visualize --help
```

### Ejecutar una simulación

```bash
# Esfera de Plummer (512 partículas, Barnes-Hut)
./target/release/gadget-ng stepping \
  --config examples/plummer_sphere.toml \
  --out runs/plummer --snapshot

# Órbita Kepleriana (2 cuerpos)
./target/release/gadget-ng stepping \
  --config examples/kepler_orbit.toml \
  --out runs/kepler --snapshot

# Cosmológica ΛCDM (z=49 → z=0)
./target/release/gadget-ng stepping \
  --config examples/cosmological.toml \
  --out runs/cosmo --snapshot
```

### Visualizar un snapshot

```bash
./target/release/gadget-ng visualize \
  --snapshot runs/plummer/snapshot_final \
  --output frame.png \
  --width 1024 --height 1024 \
  --projection xy --color velocity
```

Opciones de proyección: `xy` (default), `xz`, `yz`.
Opciones de color: `velocity` (default, mapa Viridis), `white`.

### Analizar: halos FoF + P(k)

```bash
./target/release/gadget-ng analyse \
  --snapshot runs/nbody/snapshot_final \
  --out runs/nbody/analysis \
  --linking-length 0.2 \
  --pk-mesh 64
```

Genera `analysis/halos.jsonl` y `analysis/power_spectrum.jsonl`.

### Reanudar desde un checkpoint

```bash
# Primera corrida con checkpointing activado (checkpoint_interval > 0 en el TOML):
./target/release/gadget-ng stepping \
  --config examples/plummer_sphere.toml \
  --out runs/plummer

# Reanudar:
./target/release/gadget-ng stepping \
  --config examples/plummer_sphere.toml \
  --out runs/plummer --resume runs/plummer
```

### Con MPI

```bash
mpiexec -n 4 ./target/release/gadget-ng stepping \
  --config examples/nbody_bh_1k.toml \
  --out runs/mpi --snapshot
```

---

## Ejemplos

Ver [`examples/`](examples/) para configuraciones completas y comentadas:

| Archivo | Descripción |
|---------|-------------|
| [`plummer_sphere.toml`](examples/plummer_sphere.toml) | Esfera de Plummer, equilibrio virial, checkpointing |
| [`kepler_orbit.toml`](examples/kepler_orbit.toml) | Órbita circular Sol-Tierra, 1 período |
| [`nbody_bh_1k.toml`](examples/nbody_bh_1k.toml) | 1000 partículas Barnes-Hut, θ=0.4, análisis |
| [`cosmological.toml`](examples/cosmological.toml) | ΛCDM z=49→0, integración con momento canónico |

---

## Estructura del TOML

```toml
[simulation]
particle_count = 512
box_size       = 20.0
dt             = 0.01
num_steps      = 200
softening      = 0.1
seed           = 42

[initial_conditions]
# Opciones: "lattice" | { two_body = {...} } | { plummer = { a = 1.0 } }
kind = { plummer = { a = 1.0 } }

[gravity]
# Opciones: "direct" | "barnes_hut" | "pm" | "tree_pm"
solver = "barnes_hut"
theta  = 0.5

[cosmology]
enabled      = true
omega_m      = 0.3
omega_lambda = 0.7
hubble0      = 67.4   # km/s/Mpc
a_begin      = 0.02
a_end        = 1.0

[units]
enabled           = true
length_in_kpc     = 1.0
mass_in_1e10_msol = 1.0
velocity_in_km_s  = 1.0

[output]
snapshot_format    = "jsonl"   # "jsonl" | "bincode" | "hdf5"
checkpoint_interval = 100      # 0 = desactivado
```

---

## Crates del workspace

```
gadget-ng/
├── crates/
│   ├── gadget-ng-core          # Vec3, Particle, RunConfig, DirectGravity / GravitySolver
│   ├── gadget-ng-tree          # Octree + Barnes-Hut + FMM (cuadrupolo + octupolo STF)
│   ├── gadget-ng-integrators   # leapfrog_kdk_step (KDK) + cosmológico
│   ├── gadget-ng-parallel      # SerialRuntime / MpiRuntime + SFC decomposition
│   ├── gadget-ng-io            # Snapshots JSONL / Bincode / HDF5 + Provenance
│   ├── gadget-ng-pm            # Particle Mesh (FFT, CIC)
│   ├── gadget-ng-treepm        # TreePM (Barnes-Hut short-range + PM long-range)
│   ├── gadget-ng-gpu           # Compute shaders WGSL via wgpu
│   ├── gadget-ng-analysis      # FoF halos + espectro de potencia P(k)
│   ├── gadget-ng-sph           # SPH: Wendland C2, densidad adaptativa, viscosidad Monaghan
│   ├── gadget-ng-vis           # Visualización CPU: proyecciones, colormap Viridis, PNG
│   ├── gadget-ng-physics       # Tests de validación física (Kepler, Sod, Plummer virial)
│   └── gadget-ng-cli           # Binario gadget-ng (clap)
├── examples/                   # Configuraciones TOML comentadas
├── experiments/
│   └── nbody/mvp_smoke/        # Configs y validaciones del experimento base
├── docs/
│   ├── architecture.md
│   ├── roadmap.md
│   └── user-guide.md
└── scripts/
    ├── check.sh
    └── validation/
```

---

## Calidad y CI

```bash
./scripts/check.sh            # fmt + clippy -D warnings + test + build --features mpi
./scripts/validation/compare_serial_mpi.sh      # paridad serial vs MPI (DirectGravity)
./scripts/validation/compare_serial_mpi_bh.sh   # paridad serial vs MPI (Barnes-Hut)
```

GitHub Actions: [`.github/workflows/ci.yml`](.github/workflows/ci.yml)

Tests de validación física (`cargo test -p gadget-ng-physics`):
- **Kepler**: conservación de energía y momento angular en órbita circular y elíptica
- **Sod**: condiciones iniciales del tubo de choque 1D SPH
- **Plummer**: ratio virial Q ≈ 0.5 en equilibrio

---

## Features opcionales

| Feature | Descripción |
|---------|-------------|
| `mpi` | Enlaza a MPI para `MpiRuntime` con descomposición SFC |
| `gpu` | Aceleración GPU vía `wgpu` (Vulkan/Metal/DX12/WebGPU) |
| `simd` | Vectorización con `rayon` en core |
| `bincode` | Snapshots binarios `particles.bin` |
| `hdf5` | Snapshots `snapshot.hdf5` (GADGET-like; requiere `libhdf5-dev`) |
| `full` | Todas las anteriores activadas |

---

## Documentación

- [`docs/user-guide.md`](docs/user-guide.md) — referencia completa de opciones
- [`docs/architecture.md`](docs/architecture.md) — decisiones de diseño y comparativa con GADGET-4
- [`docs/roadmap.md`](docs/roadmap.md) — hitos completados (Fase 1 y 2)
- [`examples/README.md`](examples/README.md) — ejemplos listos para usar

---

## Licencia

Este repositorio se distribuye bajo la [GNU General Public License v3.0](LICENSE) (GPL-3.0).
