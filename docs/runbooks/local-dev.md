# Runbook: desarrollo local

## Requisitos

- Rust estable (`rustup`) con `rustfmt` y `clippy`.
- Para MPI: OpenMPI o MPICH (`libopenmpi-dev`, `openmpi-bin` en Debian/Ubuntu).
- Para `./scripts/check.sh` (usa `clippy`/`test` con **todas** las features): `libhdf5-dev` en Debian/Ubuntu (enlazado por el crate `hdf5`).

## Comandos

```bash
cargo build
cargo test
cargo run -p gadget-ng-cli -- config --config experiments/nbody/mvp_smoke/config/default.toml
cargo run -p gadget-ng-cli -- stepping --config experiments/nbody/mvp_smoke/config/default.toml --out experiments/nbody/mvp_smoke/runs/local --snapshot
```

## Calidad

```bash
./scripts/check.sh
```

## MPI local

```bash
cargo build --features mpi
./scripts/mpi/run_smoke.sh
./scripts/validation/compare_serial_mpi.sh
```

Variable opcional: `MPIRUN` (por defecto `mpiexec`).
