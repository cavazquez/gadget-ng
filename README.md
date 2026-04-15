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
| **Gravedad directa** | Pares Plummer-suavizados, \(O(N^2)\) — `DirectGravity` |
| **Barnes–Hut** | Octree en arena, MAC `s/d < θ`, monopolo, \(O(N\log N)\) — `BarnesHutGravity` |
| **MPI** | `ParallelRuntime` con `SerialRuntime` / `MpiRuntime` (`--features mpi`) |
| **Configuración** | TOML + variables de entorno `GADGET_NG_*` |
| **Snapshots** | JSONL (default), **bincode** o **HDF5** estilo GADGET (`[output] snapshot_format`) + `provenance.json` |

---

## Inicio rápido

### Compilar y ver ayuda

```bash
cargo build
cargo run -p gadget-ng-cli -- --help
```

### Ejecutar una simulación (modo serial)

```bash
cargo run -p gadget-ng-cli -- stepping \
  --config experiments/nbody/mvp_smoke/config/default.toml \
  --out experiments/nbody/mvp_smoke/runs/demo \
  --snapshot
```

### Ejecutar con Barnes–Hut

```bash
cargo run -p gadget-ng-cli -- stepping \
  --config experiments/nbody/mvp_smoke/config/barnes_hut.toml \
  --out experiments/nbody/mvp_smoke/runs/demo_bh \
  --snapshot
```

### Ejecutar con MPI (4 rangos)

```bash
cargo build --features mpi
mpiexec -n 4 target/debug/gadget-ng stepping \
  --config experiments/nbody/mvp_smoke/config/default.toml \
  --out experiments/nbody/mvp_smoke/runs/mpi \
  --snapshot
```

### Configuración `[gravity]` en TOML

```toml
[gravity]
solver = "barnes_hut"   # "direct" (default) | "barnes_hut"
theta  = 0.5            # criterio MAC s/d < theta (solo Barnes–Hut)
```

### Formato de snapshot `[output]`

```toml
[output]
snapshot_format = "jsonl"   # "jsonl" | "bincode" | "hdf5"
```

- **HDF5**: compilar el binario con `cargo build -p gadget-ng-cli --features hdf5` (requiere `libhdf5-dev` en Linux).

---

## Crates del workspace

```
gadget-ng/
├── crates/
│   ├── gadget-ng-core         # Vec3, Particle, RunConfig, DirectGravity / trait GravitySolver
│   ├── gadget-ng-tree         # Octree en arena + BarnesHutGravity
│   ├── gadget-ng-integrators  # leapfrog_kdk_step (KDK)
│   ├── gadget-ng-parallel     # SerialRuntime / MpiRuntime
│   ├── gadget-ng-io           # Snapshots JSONL + Provenance
│   └── gadget-ng-cli          # Binario gadget-ng
├── experiments/
│   └── nbody/mvp_smoke/       # Configuraciones, scripts y validaciones del experimento base
├── docs/
│   ├── architecture.md
│   └── roadmap.md
└── scripts/
    ├── check.sh
    └── validation/
        ├── compare_serial_mpi.sh
        └── compare_serial_mpi_bh.sh
```

---

## Calidad y CI

```bash
./scripts/check.sh            # fmt + clippy -D warnings + test + build --features mpi
./scripts/validation/compare_serial_mpi.sh      # paridad serial vs MPI (DirectGravity)
./scripts/validation/compare_serial_mpi_bh.sh   # paridad serial vs MPI (Barnes–Hut)
```

GitHub Actions: [`.github/workflows/ci.yml`](.github/workflows/ci.yml)

---

## Features opcionales

| Feature | Estado | Descripción |
|---------|--------|-------------|
| `mpi` | activo | Enlaza a MPI para `MpiRuntime` |
| `simd` | reservado | Vectorización con `rayon` en core |
| `bincode` | opcional | Snapshots binarios `particles.bin` en `gadget-ng-io` |
| `hdf5` | opcional | Snapshots `snapshot.hdf5` (GADGET-like) en `gadget-ng-io` |
| `gpu` | placeholder | Aceleración GPU (SoA, fase futura) |

---

## Documentación

- [`docs/architecture.md`](docs/architecture.md) — decisiones de diseño y comparativa con GADGET-4
- [`docs/roadmap.md`](docs/roadmap.md) — hitos completados y planificados
- [`docs/runbooks/local-dev.md`](docs/runbooks/local-dev.md) — entorno local
- [`docs/runbooks/mpi-cluster.md`](docs/runbooks/mpi-cluster.md) — ejecución en clúster
- [`experiments/nbody/mvp_smoke/docs/validation.md`](experiments/nbody/mvp_smoke/docs/validation.md) — tolerancias y criterios de validación

---

## Licencia

Este repositorio se distribuye bajo la [GNU General Public License v3.0](LICENSE) (GPL-3.0). El texto completo está en el fichero `LICENSE` en la raíz del proyecto.
