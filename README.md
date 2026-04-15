# gadget-ng

Simulador **N-body** en Rust (MVP), inspirado conceptualmente en la arquitectura y prácticas de **GADGET-4** ([documentación y paper](https://wwwmpa.mpa-garching.mpg.de/gadget4/)), sin compartir código ni historial git con el proyecto original.

## Estado del MVP

- **Integración**: leapfrog **KDK** (kick–drift–kick) con paso global.
- **Gravedad**: parejas **Plummer-suavizadas** en modo **directo \(O(N^2)\)** (`DirectGravity`).
- **MPI**: capa `ParallelRuntime` con `SerialRuntime` y `MpiRuntime` (`--features mpi`), sin duplicar el bucle de integración.
- **Configuración**: TOML + entorno `GADGET_NG_` (ver [docs/runbooks/local-dev.md](docs/runbooks/local-dev.md)).
- **Estructura de experimentos**: `experiments/<dominio>/<campaña>/` con `config/`, `scripts/`, `runs/`, `reports/`, `docs/`.

## Binario `gadget-ng`

```text
cargo run -p gadget-ng-cli -- --help
cargo run -p gadget-ng-cli -- config --config experiments/nbody/mvp_smoke/config/default.toml
cargo run -p gadget-ng-cli -- stepping --config experiments/nbody/mvp_smoke/config/default.toml --out experiments/nbody/mvp_smoke/runs/demo --snapshot
cargo run -p gadget-ng-cli -- snapshot --config experiments/nbody/mvp_smoke/config/default.toml --out experiments/nbody/mvp_smoke/runs/demo_ic
```

### MPI

```bash
cargo build --features mpi
mpiexec -n 4 target/debug/gadget-ng stepping --config experiments/nbody/mvp_smoke/config/default.toml --out experiments/nbody/mvp_smoke/runs/mpi --snapshot
```

## Calidad y CI

```bash
./scripts/check.sh   # fmt + clippy -D warnings + test + build mpi
```

GitHub Actions: [.github/workflows/ci.yml](.github/workflows/ci.yml).

## Documentación

- [docs/architecture.md](docs/architecture.md) — decisiones respecto a GADGET-4.
- [docs/roadmap.md](docs/roadmap.md)
- [docs/runbooks/local-dev.md](docs/runbooks/local-dev.md)
- [docs/runbooks/mpi-cluster.md](docs/runbooks/mpi-cluster.md)
- [experiments/nbody/mvp_smoke/docs/validation.md](experiments/nbody/mvp_smoke/docs/validation.md) — tolerancias.

## Features opcionales

| Feature | Descripción |
|---------|-------------|
| `mpi` | Binario enlazado a MPI (`mpi` crate). |
| `simd` | Reserva para vectorización (depende de `rayon` en core). |
| `netcdf` | *Stub* documentado en `gadget-ng-io`. |
| `gpu` | *Placeholder* para aceleración GPU. |

## Licencia

Ver [LICENSE](LICENSE).
